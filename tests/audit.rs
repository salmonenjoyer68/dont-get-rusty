//! Audit Log: one append-only JSON line per Tool Call.
//!
//! The pure tests drive [`AuditLog`] directly; the Agent Loop test scripts a
//! fake `Chat` and asserts what the loop actually wrote for a batch of calls.
//! Every test uses temporary directories: the working directory holding
//! `.agent/audit.jsonl` and the Workspace are always separate.

use std::collections::VecDeque;
use std::fs;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex};

use async_trait::async_trait;
use dont_get_rusty::agent::{
    Agent, AgentConfig, Observation, non_interactive_overwrite, non_interactive_pause,
};
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::audit::{AuditError, AuditLog, MAX_ARGS_PREFIX};
use dont_get_rusty::fs::Workspace;
use dont_get_rusty::llm::{
    Chat, ChatResponse, Choice, LlmError, Message, ResponseMessage, ToolCall,
};
use dont_get_rusty::policy::{Decision, Policy};
use dont_get_rusty::sandbox::{RunResult, Sandbox};
use serde_json::{Value, json};
use tempfile::TempDir;
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

static SANDBOX: LazyLock<Arc<Sandbox>> = LazyLock::new(|| {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    Arc::new(Sandbox::new(image.root()).expect("failed to create sandbox"))
});

/// Every event in `path`, oldest first.
fn events(path: &Path) -> Vec<Value> {
    let text = fs::read_to_string(path).expect("audit log should be readable");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("every line is one JSON event"))
        .collect()
}

/// A completed Run with no output, for building an allowed Observation.
fn run_result(exit_code: i32) -> RunResult {
    RunResult {
        exit_code,
        stdout: "1\n".to_owned(),
        stderr: String::new(),
        timed_out: false,
        truncated: false,
        duration_ms: 3,
        stdout_bytes: 2,
        stderr_bytes: 0,
    }
}

#[test]
fn an_allowed_and_a_denied_call_record_the_expected_fields() {
    let cwd = TempDir::new().unwrap();
    let workspace = TempDir::new().unwrap();
    let log = AuditLog::open(cwd.path(), workspace.path()).expect("audit log should open");
    assert_eq!(
        log.path(),
        cwd.path()
            .canonicalize()
            .unwrap()
            .join(".agent")
            .join("audit.jsonl"),
        "the log lives under the working directory"
    );

    let allowed = ToolCall::new("a", "run_code", json!({ "code": "print(1)" }).to_string());
    let allowed_observation = Observation::from_run("run_code", run_result(0));
    log.record(2, &allowed, &Decision::Allow, &allowed_observation, 5)
        .expect("an allowed event should be appended");

    let reason = "`..` components are not permitted";
    let denied = ToolCall::new(
        "b",
        "read_file",
        json!({ "path": "../etc/passwd" }).to_string(),
    );
    let denied_observation = Observation::error("read_file", reason);
    log.record(2, &denied, &Decision::deny(reason), &denied_observation, 0)
        .expect("a denied event should be appended");

    let events = events(log.path());
    assert_eq!(events.len(), 2, "one event per Tool Call");

    let allowed = &events[0];
    assert_eq!(allowed["session"], json!(log.session()));
    assert_eq!(allowed["step"], json!(2));
    assert_eq!(allowed["tool"], json!("run_code"));
    assert_eq!(allowed["decision"], json!("allow"));
    assert!(
        allowed.get("denial_reason").is_none(),
        "an allowed call carries no denial reason"
    );
    assert!(
        allowed.get("error").is_none(),
        "a successful Run carries no execution error"
    );
    assert_eq!(
        allowed["args_sha256"],
        json!("5042e249312efe2f73490958fe0ff77c27bc08a69a2c190b612876a721784b27"),
        "the hash covers the full raw arguments"
    );
    let prefix = allowed["args_prefix"].as_str().unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(prefix).unwrap(),
        json!({ "code": "<omitted: 8 bytes>" }),
        "code is replaced by a byte-count marker"
    );
    assert_eq!(allowed["exit_code"], json!(0));
    assert_eq!(allowed["stdout_bytes"], json!(2));
    assert_eq!(allowed["stderr_bytes"], json!(0));
    assert_eq!(allowed["truncated"], json!(false));
    assert_eq!(allowed["duration_ms"], json!(5));
    let ts = allowed["ts"].as_str().unwrap();
    OffsetDateTime::parse(ts, &Rfc3339).expect("the timestamp is RFC 3339");
    assert!(ts.ends_with('Z'), "the timestamp is UTC: {ts}");

    let denied = &events[1];
    assert_eq!(denied["session"], json!(log.session()));
    assert_eq!(denied["step"], json!(2));
    assert_eq!(denied["tool"], json!("read_file"));
    assert_eq!(denied["decision"], json!("deny"));
    assert_eq!(denied["denial_reason"], json!(reason));
    assert!(
        denied.get("error").is_none(),
        "a Policy denial carries no execution error"
    );
    assert_eq!(
        denied["args_sha256"],
        json!("b4ccd46d7a551685a132c7d57ad76df154a3c4ca455a952fa664ff9db46b62db")
    );
    assert_eq!(denied["exit_code"], Value::Null);
    assert_eq!(denied["stdout_bytes"], json!(0));
    assert_eq!(denied["stderr_bytes"], json!(0));
    assert_eq!(denied["truncated"], json!(false));
}

#[test]
fn code_file_bodies_and_piped_input_never_reach_the_log() {
    let cwd = TempDir::new().unwrap();
    let workspace = TempDir::new().unwrap();
    let log = AuditLog::open(cwd.path(), workspace.path()).expect("audit log should open");

    let code = "SECRET_CODE_MARKER = 'delete everything'";
    let content = "SECRET_FILE_BODY_MARKER\nsecond line\n";
    let stdin = "SECRET_PIPED_INPUT_MARKER\n";
    log.record(
        0,
        &ToolCall::new("a", "run_code", json!({ "code": code }).to_string()),
        &Decision::Allow,
        &Observation::error("run_code", "invalid arguments"),
        1,
    )
    .unwrap();
    log.record(
        0,
        &ToolCall::new(
            "b",
            "write_file",
            json!({ "path": "notes.txt", "content": content }).to_string(),
        ),
        &Decision::Allow,
        &Observation::from_text("write_file", "wrote 39 bytes to notes.txt"),
        1,
    )
    .unwrap();
    log.record(
        0,
        &ToolCall::new(
            "c",
            "run_code",
            json!({ "code": "print(input())", "stdin": stdin }).to_string(),
        ),
        &Decision::Allow,
        &Observation::from_run("run_code", run_result(0)),
        1,
    )
    .unwrap();

    let text = fs::read_to_string(log.path()).unwrap();
    assert!(
        !text.contains("SECRET_CODE_MARKER"),
        "the code body must never be stored: {text}"
    );
    assert!(
        !text.contains("SECRET_FILE_BODY_MARKER"),
        "the file body must never be stored: {text}"
    );
    assert!(
        !text.contains("SECRET_PIPED_INPUT_MARKER"),
        "piped stdin must never be stored: {text}"
    );
    assert!(text.contains("<omitted:"), "redaction marks the fields");

    let events = events(log.path());
    assert_eq!(events.len(), 3);
    for event in &events {
        let prefix = event["args_prefix"].as_str().unwrap();
        let prefix: Value = serde_json::from_str(prefix).expect("the prefix is valid JSON");
        assert!(
            prefix.as_object().unwrap().values().all(|value| {
                value.as_str() != Some(code)
                    && value.as_str() != Some(content)
                    && value.as_str() != Some(stdin)
            }),
            "no argument value is a stored body: {prefix}"
        );
    }
    assert_eq!(
        events[0]["args_prefix"].as_str().unwrap(),
        format!("{{\"code\":\"<omitted: {} bytes>\"}}", code.len()),
        "the marker counts the full body"
    );
    assert_eq!(
        events[2]["args_prefix"].as_str().unwrap(),
        format!(
            "{{\"code\":\"<omitted: {} bytes>\",\"stdin\":\"<omitted: {} bytes>\"}}",
            "print(input())".len(),
            stdin.len()
        ),
        "piped stdin is redacted alongside the code"
    );
}

#[test]
fn a_long_payload_is_hashed_in_full_and_the_prefix_is_cut_on_a_char_boundary() {
    let cwd = TempDir::new().unwrap();
    let workspace = TempDir::new().unwrap();
    let log = AuditLog::open(cwd.path(), workspace.path()).expect("audit log should open");

    // `path` is metadata, not a body, so it is the one argument that survives
    // redaction long enough to exercise the prefix cut.
    let path = "é".repeat(1500);
    let arguments = json!({ "path": path }).to_string();
    assert!(arguments.len() > MAX_ARGS_PREFIX);
    log.record(
        7,
        &ToolCall::new("a", "read_file", arguments.clone()),
        &Decision::Allow,
        &Observation::from_text("read_file", "ok"),
        0,
    )
    .unwrap();

    let events = events(log.path());
    let event = &events[0];
    assert_eq!(
        event["args_sha256"],
        json!("0c94b4e486194e5d3522978d5a8d633a93c902556342f1dbf324df3c21e18ef0"),
        "the hash covers the full payload, not the prefix"
    );
    let prefix = event["args_prefix"].as_str().unwrap();
    assert!(
        prefix.len() <= MAX_ARGS_PREFIX && prefix.len() + 1 >= MAX_ARGS_PREFIX,
        "the prefix fills the budget up to a character boundary, got {} bytes",
        prefix.len()
    );
    assert!(
        arguments.starts_with(prefix),
        "truncation only removes the tail and keeps whole characters"
    );
    assert!(arguments.is_char_boundary(prefix.len()));
}

#[test]
fn the_log_lives_outside_the_workspace() {
    let cwd = TempDir::new().unwrap();
    let workspace_root = cwd.path().join("workspace");
    fs::create_dir_all(&workspace_root).unwrap();
    let log = AuditLog::open(cwd.path(), &workspace_root).expect("audit log should open");

    let cwd = cwd.path().canonicalize().unwrap();
    assert_eq!(log.path(), cwd.join(".agent").join("audit.jsonl"));
    assert!(
        !log.path()
            .starts_with(workspace_root.canonicalize().unwrap()),
        "the log must not be reachable through the Workspace"
    );
    assert!(cwd.join(".agent").is_dir(), "the directory is created");
}

#[test]
fn a_log_inside_the_workspace_is_refused() {
    let dir = TempDir::new().unwrap();
    let error = AuditLog::open(dir.path(), dir.path()).unwrap_err();

    assert!(matches!(error, AuditError::InsideWorkspace { .. }));
    assert!(
        error.to_string().contains("inside the Workspace"),
        "the error names the problem: {error}"
    );
    assert!(
        !dir.path().join(".agent").exists(),
        "a refused start must not create `.agent` in the Workspace"
    );
}

// ---- Agent Loop: every Tool Call is recorded exactly once ----

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

#[tokio::test]
async fn the_loop_records_one_event_per_tool_call_in_request_order() {
    let cwd = TempDir::new().unwrap();
    let workspace_dir = TempDir::new().unwrap();
    let audit = AuditLog::open(cwd.path(), workspace_dir.path()).expect("audit log should open");
    let audit_path = audit.path().to_path_buf();

    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![
                ToolCall::new(
                    "write",
                    "write_file",
                    json!({ "path": "notes.txt", "content": "hello" }).to_string(),
                ),
                ToolCall::new(
                    "denied",
                    "read_file",
                    json!({ "path": "../outside" }).to_string(),
                ),
                ToolCall::new("unknown", "teleport", "{}".to_owned()),
                ToolCall::new("malformed", "run_code", "{".to_owned()),
            ],
        ),
        reply(Some("done"), Vec::new()),
    ]);
    let workspace = Workspace::open(workspace_dir.path()).expect("workspace should open");
    let agent = Agent::new(
        chat,
        SANDBOX.clone(),
        workspace,
        AgentConfig::default(),
        Policy::default(),
        audit,
        non_interactive_pause(),
        non_interactive_overwrite(),
    );

    assert_eq!(agent.run("audit every call").await.unwrap().answer, "done");

    let events = events(&audit_path);
    assert_eq!(events.len(), 4, "one event per Tool Call, no more, no less");
    let tools: Vec<&str> = events
        .iter()
        .map(|event| event["tool"].as_str().unwrap())
        .collect();
    assert_eq!(
        tools,
        ["write_file", "read_file", "teleport", "run_code"],
        "events follow the order of the calls"
    );
    let decisions: Vec<&str> = events
        .iter()
        .map(|event| event["decision"].as_str().unwrap())
        .collect();
    assert_eq!(
        decisions,
        ["allow", "deny", "deny", "allow"],
        "decisions are the Policy verdict; a malformed run_code was allowed"
    );
    for event in &events {
        assert_eq!(event["step"], json!(0), "the Step that produced the call");
        assert_eq!(event["session"], events[0]["session"]);
        assert_eq!(event["exit_code"], Value::Null);
    }

    assert!(events[0].get("denial_reason").is_none());
    assert!(events[0].get("error").is_none());
    assert_eq!(
        events[1]["denial_reason"],
        json!("invalid path: `..` components are not permitted")
    );
    assert!(
        events[1].get("error").is_none(),
        "a Policy denial carries no execution error"
    );
    assert_eq!(events[2]["denial_reason"], json!("unknown tool `teleport`"));
    assert!(
        events[2].get("error").is_none(),
        "a Policy denial carries no execution error"
    );
    assert!(
        events[3].get("denial_reason").is_none(),
        "an allowed call has no denial reason"
    );
    assert!(
        events[3]["error"]
            .as_str()
            .is_some_and(|error| error.starts_with("invalid arguments: ")),
        "the parse error is recorded as an execution error: {}",
        events[3]["error"]
    );
    assert_eq!(
        events[3]["args_prefix"],
        json!("<unparsed arguments: 1 bytes>"),
        "a payload that cannot be parsed is not stored"
    );
    assert!(
        events[0]["stdout_bytes"].as_u64().unwrap() > 0,
        "the write_file result summary is recorded"
    );
    assert!(
        !audit_path.starts_with(workspace_dir.path().canonicalize().unwrap()),
        "the log stays outside the Workspace"
    );
}

#[tokio::test]
async fn a_declined_overwrite_keeps_the_policy_allow_and_records_an_error() {
    let cwd = TempDir::new().unwrap();
    let workspace_dir = TempDir::new().unwrap();
    fs::write(workspace_dir.path().join("notes.txt"), "original").unwrap();
    let audit = AuditLog::open(cwd.path(), workspace_dir.path()).expect("audit log should open");
    let audit_path = audit.path().to_path_buf();

    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![
                ToolCall::new(
                    "declined",
                    "write_file",
                    json!({ "path": "notes.txt", "content": "clobbered" }).to_string(),
                ),
                ToolCall::new(
                    "failing",
                    "run_code",
                    json!({ "code": "import sys; sys.exit(3)" }).to_string(),
                ),
            ],
        ),
        reply(Some("done"), Vec::new()),
    ]);
    let workspace = Workspace::open(workspace_dir.path()).expect("workspace should open");
    let agent = Agent::new(
        chat,
        SANDBOX.clone(),
        workspace,
        AgentConfig::default(),
        Policy::default(),
        audit,
        non_interactive_pause(),
        non_interactive_overwrite(),
    );

    assert_eq!(agent.run("audit failures").await.unwrap().answer, "done");

    assert_eq!(
        fs::read_to_string(workspace_dir.path().join("notes.txt")).unwrap(),
        "original",
        "the declined write touched nothing"
    );

    let events = events(&audit_path);
    assert_eq!(events.len(), 2, "one event per Tool Call");

    let declined = &events[0];
    assert_eq!(declined["tool"], json!("write_file"));
    assert_eq!(
        declined["decision"],
        json!("allow"),
        "the Policy allowed the write; the host refused it at execution"
    );
    assert!(
        declined.get("denial_reason").is_none(),
        "a declined overwrite is not a Policy denial"
    );
    assert!(
        declined["error"]
            .as_str()
            .is_some_and(|error| error.contains("was not confirmed")),
        "the execution failure is the recorded error: {}",
        declined["error"]
    );
    assert_eq!(declined["exit_code"], Value::Null);

    let failing = &events[1];
    assert_eq!(failing["tool"], json!("run_code"));
    assert_eq!(
        failing["decision"],
        json!("allow"),
        "a Guest Run that exits non-zero still ran"
    );
    assert_eq!(failing["exit_code"], json!(3));
    assert!(
        failing.get("denial_reason").is_none(),
        "an allowed Guest Run has no denial reason"
    );
    assert!(
        failing.get("error").is_none(),
        "a completed Guest Run with a non-zero exit records no host error"
    );
}
