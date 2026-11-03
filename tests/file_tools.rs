//! File tools through the Agent Loop: `read_file`, `write_file`, and
//! `list_files` against a real Workspace, with scripted fake LLM replies.
//!
//! The Guest test loads the real Runtime Image, so `scripts/fetch-assets.sh`
//! must have run first. Every test uses a temporary Workspace; nothing outside
//! it is touched.

use std::collections::VecDeque;
use std::fs;
use std::sync::{Arc, LazyLock, Mutex};

use async_trait::async_trait;
use dont_get_rusty::agent::{Agent, AgentConfig};
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::fs::Workspace;
use dont_get_rusty::llm::{
    Chat, ChatResponse, Choice, LlmError, Message, ResponseMessage, ToolCall,
};
use dont_get_rusty::policy::Policy;
use dont_get_rusty::sandbox::Sandbox;
use serde_json::{Value, json};
use tempfile::TempDir;

static SANDBOX: LazyLock<Arc<Sandbox>> = LazyLock::new(|| {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    Arc::new(Sandbox::new(image.root()).expect("failed to create sandbox"))
});

struct FakeChat {
    replies: Mutex<VecDeque<ChatResponse>>,
    requests: Mutex<Vec<Vec<Message>>>,
}

impl FakeChat {
    fn new(replies: Vec<ChatResponse>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<Vec<Message>> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Chat for FakeChat {
    async fn chat(&self, messages: &[Message], _tools: &[Value]) -> Result<ChatResponse, LlmError> {
        self.requests.lock().unwrap().push(messages.to_vec());
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

fn write_file(id: &str, path: &str, content: &str) -> ToolCall {
    ToolCall::new(
        id,
        "write_file",
        json!({ "path": path, "content": content }).to_string(),
    )
}

fn read_file(id: &str, path: &str) -> ToolCall {
    ToolCall::new(id, "read_file", json!({ "path": path }).to_string())
}

fn list_files(id: &str, path: Option<&str>) -> ToolCall {
    let arguments = match path {
        Some(path) => json!({ "path": path }),
        None => json!({}),
    };
    ToolCall::new(id, "list_files", arguments.to_string())
}

/// The content of every `tool` message in one conversation, in order.
fn tool_contents(messages: &[Message]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

fn agent(chat: Arc<FakeChat>, workspace: Workspace) -> Agent {
    Agent::new(
        chat,
        SANDBOX.clone(),
        workspace,
        AgentConfig::default(),
        Policy::default(),
    )
}

#[tokio::test]
async fn writes_then_reads_through_the_workspace() {
    let dir = TempDir::new().unwrap();
    let chat = FakeChat::new(vec![
        reply(None, vec![write_file("w", "notes/a.txt", "hello")]),
        reply(None, vec![read_file("r", "notes/a.txt")]),
        reply(Some("done"), Vec::new()),
    ]);
    let workspace = Workspace::open(dir.path()).unwrap();
    let result = agent(chat.clone(), workspace)
        .run("store then read")
        .await
        .unwrap();

    assert_eq!(result.answer, "done");
    assert_eq!(
        fs::read_to_string(dir.path().join("notes/a.txt")).unwrap(),
        "hello",
        "the write landed in the Workspace"
    );

    let requests = chat.requests();
    assert_eq!(requests.len(), 3, "one LLM call per Step");

    let write = &tool_contents(&requests[1])[0];
    assert!(write.contains("tool: write_file"), "write: {write}");
    assert!(write.contains("exit_code: none"), "write: {write}");
    assert!(
        write.contains("wrote 5 bytes to notes/a.txt"),
        "write: {write}"
    );

    let read = tool_contents(&requests[2]).last().cloned().unwrap();
    assert!(read.contains("tool: read_file"), "read: {read}");
    assert!(read.contains("exit_code: none"), "read: {read}");
    assert!(read.contains("stdout_bytes: 5"), "read: {read}");
    assert!(
        read.contains("--- begin stdout (5 bytes) ---\nhello--- end stdout ---\n"),
        "read: {read}"
    );
}

#[tokio::test]
async fn list_output_uses_the_stable_line_format() {
    let dir = TempDir::new().unwrap();
    let workspace = Workspace::open(dir.path()).unwrap();
    workspace.write_file("a.txt", b"hi").unwrap();
    workspace.write_file("b/c.txt", b"x").unwrap();

    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![list_files("root", None), list_files("sub", Some("b"))],
        ),
        reply(Some("listed"), Vec::new()),
    ]);
    let result = agent(chat.clone(), workspace).run("list").await.unwrap();
    assert_eq!(result.answer, "listed");

    let contents = tool_contents(&chat.requests()[1]);
    assert_eq!(contents.len(), 2);
    assert!(
        contents[0].contains(
            "--- begin stdout (21 bytes) ---\n\
             file\t2\ta.txt\n\
             dir\t-\tb\n\
             --- end stdout ---\n"
        ),
        "root listing: {}",
        contents[0]
    );
    assert!(
        contents[1].contains(
            "--- begin stdout (13 bytes) ---\n\
             file\t1\tc.txt\n\
             --- end stdout ---\n"
        ),
        "sub listing: {}",
        contents[1]
    );
    assert!(contents[0].contains("exit_code: none"), "{}", contents[0]);
}

#[tokio::test]
async fn a_quota_denied_write_is_an_error_observation_and_the_loop_continues() {
    let dir = TempDir::new().unwrap();
    let chat = FakeChat::new(vec![
        reply(None, vec![write_file("w", "big.txt", "too large")]),
        reply(Some("gave up"), Vec::new()),
    ]);
    let workspace = Workspace::open_with_quota(dir.path(), 1).unwrap();
    let result = agent(chat.clone(), workspace).run("write").await.unwrap();

    assert_eq!(
        result.answer, "gave up",
        "the loop continued past the denial"
    );
    assert_eq!(chat.requests().len(), 2);
    assert!(
        !dir.path().join("big.txt").exists(),
        "a denied write creates nothing"
    );

    let denied = &tool_contents(&chat.requests()[1])[0];
    assert!(denied.contains("tool: write_file"), "denied: {denied}");
    assert!(denied.contains("exit_code: none"), "denied: {denied}");
    assert!(
        denied.contains("error: workspace quota exceeded"),
        "denied: {denied}"
    );
    assert!(denied.contains("limit is 1"), "denied: {denied}");
}

#[tokio::test]
async fn traversal_and_absolute_paths_are_denied_and_touch_nothing_outside() {
    let dir = TempDir::new().unwrap();
    let workspace_dir = dir.path().join("ws");
    fs::create_dir(&workspace_dir).unwrap();
    let sentinel = dir.path().join("sentinel.txt");
    fs::write(&sentinel, b"sentinel").unwrap();
    let absolute = dir.path().join("absolute.txt");

    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![
                write_file("parent", "../sentinel.txt", "pwned"),
                write_file("absolute", absolute.to_str().unwrap(), "pwned"),
            ],
        ),
        reply(Some("denied"), Vec::new()),
    ]);
    let workspace = Workspace::open(&workspace_dir).unwrap();
    let result = agent(chat.clone(), workspace).run("escape").await.unwrap();

    assert_eq!(result.answer, "denied");
    assert_eq!(chat.requests().len(), 2);
    assert_eq!(
        fs::read(&sentinel).unwrap(),
        b"sentinel",
        "the outside file is unchanged"
    );
    assert!(!absolute.exists(), "no outside file was created");

    let contents = tool_contents(&chat.requests()[1]);
    assert_eq!(contents.len(), 2);
    assert!(
        contents[0].contains("error: invalid path: `..` components are not permitted"),
        "parent traversal: {}",
        contents[0]
    );
    assert!(
        contents[1].contains("error: invalid path: absolute paths are not permitted"),
        "absolute path: {}",
        contents[1]
    );
}

#[tokio::test]
async fn the_guest_reads_host_written_files_but_cannot_write() {
    let dir = TempDir::new().unwrap();
    let code = "\
print(open('/workspace/data.txt').read())
try:
    open('/workspace/guest.txt', 'w').write('pwned')
    print('write allowed')
except OSError as error:
    print('write denied:', type(error).__name__)";
    let chat = FakeChat::new(vec![
        reply(None, vec![write_file("w", "data.txt", "42")]),
        reply(
            None,
            vec![ToolCall::new(
                "run",
                "run_code",
                json!({ "code": code }).to_string(),
            )],
        ),
        reply(Some("verified"), Vec::new()),
    ]);
    let workspace = Workspace::open(dir.path()).unwrap();
    let result = agent(chat.clone(), workspace).run("verify").await.unwrap();

    assert_eq!(result.answer, "verified");
    assert_eq!(
        fs::read_to_string(dir.path().join("data.txt")).unwrap(),
        "42"
    );
    assert!(
        !dir.path().join("guest.txt").exists(),
        "the Guest cannot create files in the Workspace"
    );

    let run = tool_contents(&chat.requests()[2]).last().cloned().unwrap();
    assert!(run.contains("tool: run_code"), "run: {run}");
    assert!(run.contains("exit_code: 0"), "run: {run}");
    assert!(
        run.contains("42"),
        "the Guest read the host-written file: {run}"
    );
    assert!(
        run.contains("write denied:"),
        "the Guest write failed: {run}"
    );
    assert!(
        !run.contains("write allowed"),
        "the Guest gained a write capability: {run}"
    );
}
