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
use serde_json::{Value, json};
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
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    let sandbox = Arc::new(Sandbox::new(image.root()).expect("failed to create sandbox"));
    let chat = FakeChat::new(vec![
        reply(
            Some("Let me look."),
            vec![ToolCall::new("c1", "list_files", "{}".to_owned())],
        ),
        reply(Some("Found one file."), Vec::new()),
    ]);
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
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut finished = None;
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
        let done = events
            .iter()
            .any(|event| event["type"] == json!("task_finished"));
        if done {
            finished = Some(snapshot);
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let snapshot = finished.expect("the Task should finish");
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
