//! End-to-end CLI contract: the real binary, the real CPython Guest, a real
//! Workspace, and a local stub OpenAI server.
//!
//! The stub binds an ephemeral loopback port and serves one scripted reply per
//! connection, so the run never needs Ollama or any network beyond localhost.
//! `scripts/fetch-assets.sh` must have run first.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::mpsc::{Receiver, channel};
use std::thread;
use std::time::Duration;

use dont_get_rusty::assets::RuntimeImage;
use serde_json::{Value, json};

/// One scripted reply, served to one connection in order.
struct Reply {
    body: String,
}

impl Reply {
    fn json(body: impl Into<String>) -> Self {
        Self { body: body.into() }
    }
}

struct StubServer {
    base_url: String,
    requests: Receiver<String>,
}

impl StubServer {
    fn spawn(replies: Vec<Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, rx) = channel();

        thread::spawn(move || {
            for reply in replies {
                let (mut stream, _) = listener.accept().unwrap();
                let body = read_http_request(&mut stream);
                let _ = tx.send(body);
                let head = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n",
                    reply.body.len()
                );
                stream.write_all(head.as_bytes()).unwrap();
                stream.write_all(reply.body.as_bytes()).unwrap();
                stream.flush().unwrap();
            }
        });

        Self {
            base_url: format!("http://{address}"),
            requests: rx,
        }
    }

    fn next_request(&self) -> Value {
        let body = self
            .requests
            .recv_timeout(Duration::from_secs(60))
            .expect("the CLI should send exactly the scripted requests");
        serde_json::from_str(&body).expect("the CLI sends a JSON chat-completions request")
    }
}

fn read_http_request(stream: &mut TcpStream) -> String {
    let mut head = Vec::new();
    let mut byte = [0_u8; 1];
    loop {
        let read = stream.read(&mut byte).unwrap();
        assert_ne!(read, 0, "client closed before sending complete headers");
        head.push(byte[0]);
        if head.ends_with(b"\r\n\r\n") {
            break;
        }
    }
    let head = String::from_utf8(head).unwrap();
    let content_length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);

    let mut body = vec![0_u8; content_length];
    stream.read_exact(&mut body).unwrap();
    String::from_utf8(body).unwrap()
}

const TASK: &str = "write notes/out.txt and run the code";
const CONTENT: &str = "written by the host tool\n";
const GUEST_CODE: &str = "print('guest-ran', 2 + 3)";
const FINAL_ANSWER: &str = "Wrote notes/out.txt and ran the code: guest-ran 5.";

fn replies() -> Vec<Reply> {
    let tool_calls = json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": json!({ "path": "notes/out.txt", "content": CONTENT }).to_string(),
                        }
                    },
                    {
                        "id": "call_run",
                        "type": "function",
                        "function": {
                            "name": "run_code",
                            "arguments": json!({ "code": GUEST_CODE }).to_string(),
                        }
                    }
                ]
            },
            "finish_reason": "tool_calls"
        }]
    });
    let final_turn = json!({
        "choices": [{
            "message": { "role": "assistant", "content": FINAL_ANSWER },
            "finish_reason": "stop"
        }]
    });
    vec![
        Reply::json(tool_calls.to_string()),
        Reply::json(final_turn.to_string()),
    ]
}

/// The rendered content of every `tool` message in one request, in order.
fn tool_contents(request: &Value) -> Vec<String> {
    request["messages"]
        .as_array()
        .expect("messages is an array")
        .iter()
        .filter(|message| message["role"] == "tool")
        .map(|message| message["content"].as_str().unwrap().to_owned())
        .collect()
}

#[test]
fn one_run_writes_the_file_executes_the_guest_and_audits_each_call() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().join("cwd");
    let workspace = dir.path().join("workspace");
    std::fs::create_dir_all(&cwd).unwrap();
    std::fs::create_dir_all(&workspace).unwrap();
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    let assets = image
        .root()
        .canonicalize()
        .expect("the Runtime Image root should resolve");

    let server = StubServer::spawn(replies());
    let policy = cwd.join("policy.toml");
    std::fs::write(
        &policy,
        format!(
            "[llm]\nbase_url = \"{}\"\nmodel = \"stub-model\"\nrequest_timeout_s = 30\n",
            server.base_url
        ),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_dont-get-rusty"))
        .arg(TASK)
        .arg("--workspace")
        .arg(&workspace)
        .arg("--policy")
        .arg(&policy)
        .arg("--yes")
        .current_dir(&cwd)
        .env("DGR_ASSETS", &assets)
        // Keep the stub reachable directly; nothing else is contacted.
        .env("NO_PROXY", "127.0.0.1,localhost")
        .env("no_proxy", "127.0.0.1,localhost")
        .output()
        .unwrap();

    assert!(
        output.status.success(),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        format!("{FINAL_ANSWER}\n"),
        "stdout is the final answer"
    );
    assert_eq!(
        String::from_utf8_lossy(&output.stderr),
        "",
        "a successful run writes no host diagnostics"
    );

    // The host tool wrote the file the model asked for.
    assert_eq!(
        std::fs::read_to_string(workspace.join("notes/out.txt")).unwrap(),
        CONTENT
    );

    // Exactly the two scripted requests were sent, and the Guest really ran:
    // the second request carries the run_code Observation with its output.
    let first = server.next_request();
    let second = server.next_request();
    assert!(
        server.requests.try_recv().is_err(),
        "no extra model request was made"
    );
    assert_eq!(first["model"], "stub-model");
    assert_eq!(
        tool_contents(&first).len(),
        0,
        "the first turn carries no Observations"
    );

    let observations = tool_contents(&second);
    assert_eq!(observations.len(), 2, "both Tool Calls were executed");
    assert!(
        observations[0].contains("tool: write_file")
            && observations[0].contains(&format!("wrote {} bytes to notes/out.txt", CONTENT.len())),
        "write_file observation: {}",
        observations[0]
    );
    assert!(
        observations[1].contains("tool: run_code") && observations[1].contains("exit_code: 0"),
        "run_code observation: {}",
        observations[1]
    );
    assert!(
        observations[1].contains("guest-ran 5"),
        "the Guest executed the code: {}",
        observations[1]
    );

    // One Audit Event per Tool Call, with the Step and tool that produced it.
    let audit_path = cwd.join(".agent").join("audit.jsonl");
    let audit = std::fs::read_to_string(&audit_path).expect("the Audit Log should exist");
    let events: Vec<Value> = audit
        .lines()
        .map(|line| serde_json::from_str(line).expect("every line is one JSON event"))
        .collect();
    assert_eq!(events.len(), 2, "exactly one event per Tool Call: {audit}");
    let steps: Vec<u64> = events
        .iter()
        .map(|event| event["step"].as_u64().unwrap())
        .collect();
    assert_eq!(steps, [0, 0], "both calls came from the first Step");
    let tools: Vec<&str> = events
        .iter()
        .map(|event| event["tool"].as_str().unwrap())
        .collect();
    assert_eq!(tools, ["write_file", "run_code"]);
    let decisions: Vec<&str> = events
        .iter()
        .map(|event| event["decision"].as_str().unwrap())
        .collect();
    assert_eq!(decisions, ["allow", "allow"]);
    for event in &events {
        assert!(
            event.get("error").is_none(),
            "neither call failed on the host: {audit}"
        );
    }
    assert_eq!(events[0]["exit_code"], Value::Null);
    assert_eq!(events[1]["exit_code"], json!(0));
    assert_eq!(events[1]["stderr_bytes"], json!(0));
    assert!(
        !audit.contains("guest-ran") && !audit.contains(CONTENT.trim()),
        "neither code nor file bodies are stored: {audit}"
    );

    // Production wiring keeps the compiled module under `.agent/cache`.
    assert!(
        cwd.join(".agent").join("cache").is_dir(),
        "the CLI uses `<cwd>/.agent/cache`"
    );

    // The Audit Log and the cache stay outside the Workspace.
    assert!(!workspace.join(".agent").exists());
}
