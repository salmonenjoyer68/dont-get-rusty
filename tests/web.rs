//! Web Demo tests: the real HTTP routes over the real Agent Loop, with only
//! the model faked.
//!
//! `scripts/fetch-assets.sh` must have run: the Sandbox compiles the real
//! CPython Guest even though this Task calls only host file tools.

#![cfg(feature = "web")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::fs::Workspace;
use dont_get_rusty::llm::{
    Chat, ChatResponse, Choice, LlmError, Message, ResponseMessage, ToolCall,
};
use dont_get_rusty::policy::Policy;
use dont_get_rusty::sandbox::Sandbox;
use dont_get_rusty::web::{AppState, Hub, serve_on};
use reqwest::header::{HOST, ORIGIN};
use serde_json::{Value, json};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

struct FakeChat {
    replies: Mutex<VecDeque<ChatResponse>>,
}

impl FakeChat {
    fn new(replies: Vec<ChatResponse>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
        })
    }
}

#[async_trait]
impl Chat for FakeChat {
    async fn chat(
        &self,
        _messages: &[Message],
        _tools: &[Value],
    ) -> Result<ChatResponse, LlmError> {
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .ok_or_else(|| LlmError::Request("fake chat ran out of scripted replies".to_owned()))
    }
}

fn reply(content: Option<&str>, tool_calls: Vec<ToolCall>) -> ChatResponse {
    ChatResponse {
        choices: vec![Choice {
            message: ResponseMessage {
                content: content.map(ToOwned::to_owned),
                tool_calls,
            },
            finish_reason: None,
        }],
    }
}

/// Builds a server state whose Task can only touch host file tools.
fn test_state(
    workspace: &std::path::Path,
    audit_cwd: &std::path::Path,
    audit_path: &std::path::Path,
) -> Arc<AppState> {
    test_state_with(
        workspace,
        audit_cwd,
        audit_path,
        vec![
            reply(
                Some("Let me look."),
                vec![ToolCall::new("c1", "list_files", "{}".to_owned())],
            ),
            reply(Some("Found one file."), Vec::new()),
        ],
    )
}

/// [`test_state`] with an explicit script for the fake model.
fn test_state_with(
    workspace: &std::path::Path,
    audit_cwd: &std::path::Path,
    audit_path: &std::path::Path,
    replies: Vec<ChatResponse>,
) -> Arc<AppState> {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    let sandbox = Arc::new(Sandbox::new(image.root()).expect("failed to create sandbox"));
    let chat = FakeChat::new(replies);
    let policy = Policy::default();
    let workspace =
        Workspace::open_with_quota(workspace, policy.workspace_quota_bytes()).expect("workspace");
    let workspace_root = workspace.root_path().to_path_buf();
    Arc::new(AppState {
        hub: Arc::new(Hub::new()),
        sandbox,
        chat,
        workspace: Arc::new(workspace),
        workspace_root,
        policy,
        audit_cwd: audit_cwd.to_path_buf(),
        audit_path: audit_path.to_path_buf(),
        model: "test-model".to_owned(),
    })
}

/// Polls `/api/state` until the current Task reports `task_finished`.
async fn wait_for_task_finished(client: &reqwest::Client, base: &str) -> Value {
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while tokio::time::Instant::now() < deadline {
        let snapshot: Value = client
            .get(format!("{base}/api/state"))
            .send()
            .await
            .expect("state")
            .json()
            .await
            .expect("state json");
        let events = snapshot["events"].as_array().cloned().unwrap_or_default();
        if events
            .iter()
            .any(|event| event["type"] == json!("task_finished"))
        {
            return snapshot;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("the Task should finish");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_task_streams_events_and_the_page_can_read_workspace_and_audit() {
    let workspace = tempfile::tempdir().expect("workspace dir");
    std::fs::write(
        workspace.path().join("README.md"),
        "hello from the workspace\n",
    )
    .expect("seed file");
    let audit_dir = tempfile::tempdir().expect("audit dir");
    let audit_path = audit_dir.path().join(".agent").join("audit.jsonl");
    std::fs::create_dir_all(audit_path.parent().expect("parent")).expect("audit parent");

    let state = test_state(workspace.path(), audit_dir.path(), &audit_path);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(serve_on(listener, Arc::clone(&state)));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    let response = client
        .post(format!("{base}/api/task"))
        .json(&json!({ "task": "list the workspace" }))
        .send()
        .await
        .expect("submit");
    assert_eq!(response.status(), 202);

    // Wait for the Agent Loop to finish, then inspect the retained transcript.
    let snapshot = wait_for_task_finished(&client, &base).await;
    assert_eq!(snapshot["running"], json!(false));

    let events = snapshot["events"].as_array().expect("events");
    let kinds: Vec<&str> = events
        .iter()
        .filter_map(|event| event["type"].as_str())
        .collect();
    for expected in [
        "task_started",
        "step_started",
        "assistant_text",
        "call_started",
        "observation",
        "task_finished",
    ] {
        assert!(
            kinds.contains(&expected),
            "missing `{expected}` in {kinds:?}"
        );
    }
    assert_eq!(kinds.last(), Some(&"task_finished"));
    let last = events.last().expect("finished event");
    assert_eq!(last["answer"], json!("Found one file."));
    assert_eq!(last["steps"], json!(2));

    let listing: Vec<Value> = client
        .get(format!("{base}/api/workspace"))
        .send()
        .await
        .expect("listing")
        .json()
        .await
        .expect("listing json");
    assert!(
        listing
            .iter()
            .any(|entry| entry["name"] == json!("README.md") && entry["kind"] == json!("file")),
        "workspace listing should contain the seeded file: {listing:?}"
    );

    // The page always sends `?path=`, which must mean the Workspace root.
    let root: Vec<Value> = client
        .get(format!("{base}/api/workspace?path="))
        .send()
        .await
        .expect("root listing")
        .json()
        .await
        .expect("root listing json");
    assert_eq!(
        root, listing,
        "an empty path should list the Workspace root"
    );

    let file = client
        .get(format!("{base}/api/workspace/file?path=README.md"))
        .send()
        .await
        .expect("file")
        .text()
        .await
        .expect("file text");
    assert_eq!(file, "hello from the workspace\n");

    let audit: Vec<Value> = client
        .get(format!("{base}/api/audit"))
        .send()
        .await
        .expect("audit")
        .json()
        .await
        .expect("audit json");
    assert!(
        audit.iter().any(
            |event| event["tool"] == json!("list_files") && event["decision"] == json!("allow")
        ),
        "audit should record the allowed call: {audit:?}"
    );

    // The SSE stream replays the retained transcript, so a page that connects
    // after the Task still sees how it ran.
    let mut stream = client
        .get(format!("{base}/api/events?after=0"))
        .send()
        .await
        .expect("events");
    assert_eq!(stream.status(), 200);
    let mut body = String::new();
    let replayed = tokio::time::timeout(Duration::from_secs(5), async {
        while !body.contains("task_finished") {
            let chunk = stream.chunk().await.expect("chunk").expect("body");
            body.push_str(&String::from_utf8_lossy(&chunk));
        }
    })
    .await;
    assert!(replayed.is_ok(), "SSE should replay the finished Task");
    assert!(body.contains("call_started"), "SSE body: {body}");
    assert!(body.contains("observation"), "SSE body: {body}");

    server.abort();
}

/// Every Web Demo Task opens its own Audit Log, so it must get its own session
/// id while its Step counter restarts at 0. Sharing a process-wide session made
/// two consecutive Tasks look like a single session that restarted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn consecutive_tasks_get_fresh_audit_sessions_with_steps_starting_at_zero() {
    let workspace = tempfile::tempdir().expect("workspace dir");
    let audit_dir = tempfile::tempdir().expect("audit dir");
    let audit_path = audit_dir.path().join(".agent").join("audit.jsonl");
    std::fs::create_dir_all(audit_path.parent().expect("parent")).expect("audit parent");

    let replies = (0..2)
        .flat_map(|_| {
            [
                reply(
                    Some("Let me look."),
                    vec![ToolCall::new("c1", "list_files", "{}".to_owned())],
                ),
                reply(Some("Found one file."), Vec::new()),
            ]
        })
        .collect();
    let state = test_state_with(workspace.path(), audit_dir.path(), &audit_path, replies);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(serve_on(listener, Arc::clone(&state)));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    for task in ["first task", "second task"] {
        let response = client
            .post(format!("{base}/api/task"))
            .json(&json!({ "task": task }))
            .send()
            .await
            .expect("submit");
        assert_eq!(response.status(), 202);
        let snapshot = wait_for_task_finished(&client, &base).await;
        assert_eq!(snapshot["running"], json!(false));
    }
    server.abort();

    let text = std::fs::read_to_string(&audit_path).expect("the audit log should be readable");
    let events: Vec<Value> = text
        .lines()
        .map(|line| serde_json::from_str(line).expect("every line is one JSON event"))
        .collect();
    assert_eq!(events.len(), 2, "each Task wrote one Tool Call event");
    let sessions: Vec<&str> = events
        .iter()
        .map(|event| {
            event["session"]
                .as_str()
                .expect("every event carries a session id")
        })
        .collect();
    assert_ne!(
        sessions[0], sessions[1],
        "two consecutive Tasks must not share an Audit Log session: {sessions:?}"
    );
    for event in &events {
        assert_eq!(
            event["step"],
            json!(0),
            "each Task's first Tool Call is Step 0"
        );
    }
}

/// Two browser tabs both see an event pushed after they connected.
///
/// The Hub must wake every SSE subscriber, not just one: a single-waiter
/// wakeup leaves the other tab stale until it reconnects.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn both_sse_readers_receive_an_event_pushed_after_they_connect() {
    let workspace = tempfile::tempdir().expect("workspace dir");
    let audit_dir = tempfile::tempdir().expect("audit dir");
    let audit_path = audit_dir.path().join(".agent").join("audit.jsonl");
    std::fs::create_dir_all(audit_path.parent().expect("parent")).expect("audit parent");

    let state = test_state(workspace.path(), audit_dir.path(), &audit_path);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(serve_on(listener, Arc::clone(&state)));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // A finished warm-up Task gives both readers a replay to drain, so their
    // "ready" signals below mean the stream is connected and caught up.
    let warm_up = state.hub.start("warm-up").expect("warm-up Task");
    drop(warm_up);

    let (ready_tx, mut ready_rx) = tokio::sync::mpsc::channel::<()>(2);
    let mut readers = Vec::new();
    for _ in 0..2 {
        let client = client.clone();
        let url = format!("{base}/api/events?after=0");
        let ready = ready_tx.clone();
        readers.push(tokio::spawn(async move {
            let mut response = client.get(url).send().await.expect("events");
            assert_eq!(response.status(), 200);
            let mut body = String::new();
            let mut announced = false;
            loop {
                let chunk = response.chunk().await.expect("chunk").expect("body");
                body.push_str(&String::from_utf8_lossy(&chunk));
                if !announced && body.contains(r#""type":"task_finished""#) {
                    announced = true;
                    ready.send(()).await.expect("ready");
                }
                if body.contains(r#""type":"task_started","task":"second task""#) {
                    return body;
                }
            }
        }));
    }
    drop(ready_tx);
    for _ in 0..2 {
        tokio::time::timeout(Duration::from_secs(10), ready_rx.recv())
            .await
            .expect("both readers should replay the warm-up")
            .expect("ready signal");
    }
    // Let each reader leave the replay loop and wait on the Hub's change
    // signal, so the measured event can only arrive through that wakeup.
    tokio::time::sleep(Duration::from_millis(500)).await;

    let second_task = state.hub.start("second task").expect("second Task");
    let joined = tokio::time::timeout(Duration::from_secs(5), async {
        let first = (&mut readers[0]).await.expect("first reader");
        let second = (&mut readers[1]).await.expect("second reader");
        (first, second)
    })
    .await;
    if joined.is_err() {
        for reader in &readers {
            reader.abort();
        }
    }
    let (first_body, second_body) =
        joined.expect("both SSE readers should receive the event within 5s");
    for body in [&first_body, &second_body] {
        assert!(
            body.contains(r#""type":"task_started","task":"second task""#),
            "both readers should see the second Task start: {body}"
        );
    }

    drop(second_task);
    server.abort();
}

/// A DNS-rebinding page keeps its own Host and Origin while its hostname
/// resolves to `127.0.0.1`, so binding loopback alone does not protect the
/// Workspace, the Audit Log, or the Task route. Every request must name a
/// loopback Host; a state-changing request must not carry a foreign Origin.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dns_rebinding_requests_are_rejected_while_same_origin_requests_work() {
    let workspace = tempfile::tempdir().expect("workspace dir");
    std::fs::write(
        workspace.path().join("README.md"),
        "hello from the workspace\n",
    )
    .expect("seed file");
    let audit_dir = tempfile::tempdir().expect("audit dir");
    let audit_path = audit_dir.path().join(".agent").join("audit.jsonl");
    std::fs::create_dir_all(audit_path.parent().expect("parent")).expect("audit parent");

    let state = test_state(workspace.path(), audit_dir.path(), &audit_path);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let server = tokio::spawn(serve_on(listener, Arc::clone(&state)));
    let client = reqwest::Client::new();
    let base = format!("http://{addr}");
    let loopback_with_port = format!("localhost:{}", addr.port());

    // A missing or foreign Host is a malformed request (400); a foreign Origin
    // on a state-changing request is a refused authority (403). The tests below
    // pin the exact status so the two cannot drift back into "either".
    // A foreign Host reads like the attacker's page, not the Web Demo.
    for host in [
        "attacker.example",
        "127.0.0.1.attacker.example",
        // A port is not an identity: it must be numeric, and an IPv6 loopback
        // is not one of the Web Demo's loopback names.
        "127.0.0.1:notaport",
        "[::1]:7878",
    ] {
        let response = client
            .get(format!("{base}/api/workspace"))
            .header(HOST, host)
            .send()
            .await
            .expect("foreign host");
        assert_eq!(
            response.status(),
            reqwest::StatusCode::BAD_REQUEST,
            "Host `{host}` should be rejected as a bad request"
        );
    }

    // Loopback Hosts keep working, with or without an explicit port and in
    // any case.
    for host in [
        "127.0.0.1".to_owned(),
        format!("127.0.0.1:{}", addr.port()),
        "localhost".to_owned(),
        loopback_with_port.clone(),
        format!("LOCALHOST:{}", addr.port()),
    ] {
        let response = client
            .get(format!("{base}/api/workspace"))
            .header(HOST, host.clone())
            .send()
            .await
            .expect("loopback host");
        assert_eq!(
            response.status(),
            reqwest::StatusCode::OK,
            "Host `{host}` should be served"
        );
    }

    // HTTP/1.0 leaves Host optional; a request without one cannot be trusted.
    let mut raw = tokio::net::TcpStream::connect(addr)
        .await
        .expect("raw connect");
    raw.write_all(b"GET /api/workspace HTTP/1.0\r\n\r\n")
        .await
        .expect("raw write");
    let mut raw_response = String::new();
    raw.read_to_string(&mut raw_response)
        .await
        .expect("raw read");
    assert!(
        raw_response.contains(" 400 "),
        "a request without a Host should be rejected: {raw_response:?}"
    );

    // The page and its SSE stream are same-origin and still served.
    let page = client.get(format!("{base}/")).send().await.expect("page");
    assert_eq!(page.status(), reqwest::StatusCode::OK);
    let events = client
        .get(format!("{base}/api/events?after=0"))
        .header(HOST, loopback_with_port.clone())
        .send()
        .await
        .expect("events");
    assert_eq!(events.status(), reqwest::StatusCode::OK);

    // A state-changing request with a present, foreign Origin is refused even
    // when its Host is loopback, and it must not start a Task.
    for origin in [
        "http://attacker.example".to_owned(),
        "http://127.0.0.1.attacker.example".to_owned(),
        // Userinfo and paths are not part of an Origin a browser serializes.
        "http://localhost@attacker.example".to_owned(),
        "http://localhost/".to_owned(),
        "null".to_owned(),
    ] {
        let response = client
            .post(format!("{base}/api/task"))
            .header(HOST, loopback_with_port.clone())
            .header(ORIGIN, origin.clone())
            .json(&json!({ "task": "exfiltrate the workspace" }))
            .send()
            .await
            .expect("foreign origin");
        assert_eq!(
            response.status(),
            reqwest::StatusCode::FORBIDDEN,
            "Origin `{origin}` should be refused"
        );
    }
    let (_, events) = state.hub.snapshot();
    assert!(
        events.is_empty(),
        "a refused request must not start a Task: {events:?}"
    );

    // A missing Origin is allowed: a same-origin fetch may omit it, and the
    // request must reach the handler rather than be refused by the guard.
    let response = client
        .post(format!("{base}/api/decision/unknown"))
        .header(HOST, loopback_with_port.clone())
        .json(&json!({ "allow": true }))
        .send()
        .await
        .expect("missing origin");
    assert_eq!(
        response.status(),
        reqwest::StatusCode::NOT_FOUND,
        "a POST without an Origin must reach the handler"
    );

    // Loopback Origins are same-origin and keep working.
    for origin in [
        format!("http://localhost:{}", addr.port()),
        format!("http://LOCALHOST:{}", addr.port()),
        format!("https://127.0.0.1:{}", addr.port()),
    ] {
        let response = client
            .post(format!("{base}/api/decision/unknown"))
            .header(HOST, loopback_with_port.clone())
            .header(ORIGIN, origin.clone())
            .json(&json!({ "allow": true }))
            .send()
            .await
            .expect("loopback origin");
        assert_eq!(
            response.status(),
            reqwest::StatusCode::NOT_FOUND,
            "Origin `{origin}` should be served"
        );
    }

    server.abort();
}
