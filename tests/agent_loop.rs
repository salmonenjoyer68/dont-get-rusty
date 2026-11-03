//! Agent Loop tests: a scripted fake `Chat` drives the real Sandbox.
//!
//! The Guest is the real CPython Runtime Image, so `scripts/fetch-assets.sh`
//! must have run first. Only the LLM is fake: replies are scripted, requests
//! are captured, and nothing touches the network.

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock, Mutex};

use async_trait::async_trait;
use dont_get_rusty::agent::{
    Agent, AgentConfig, AgentError, AgentResult, PausePrompt, SYSTEM_PROMPT, non_interactive_pause,
};
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::llm::{
    Chat, ChatResponse, Choice, LlmError, Message, ResponseMessage, ToolCall,
};
use dont_get_rusty::sandbox::Sandbox;
use serde_json::{Value, json};

static SANDBOX: LazyLock<Arc<Sandbox>> = LazyLock::new(|| {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    Arc::new(Sandbox::new(image.root()).expect("failed to create sandbox"))
});

#[derive(Debug, Clone)]
struct Request {
    messages: Vec<Message>,
    tools: Vec<Value>,
}

struct FakeChat {
    replies: Mutex<VecDeque<ChatResponse>>,
    requests: Mutex<Vec<Request>>,
}

impl FakeChat {
    fn new(replies: Vec<ChatResponse>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<Request> {
        self.requests.lock().unwrap().clone()
    }
}

#[async_trait]
impl Chat for FakeChat {
    async fn chat(&self, messages: &[Message], tools: &[Value]) -> Result<ChatResponse, LlmError> {
        self.requests.lock().unwrap().push(Request {
            messages: messages.to_vec(),
            tools: tools.to_vec(),
        });
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

fn run_code(id: &str, code: &str) -> ToolCall {
    ToolCall::new(id, "run_code", json!({ "code": code }).to_string())
}

fn tool_messages(messages: &[Message]) -> Vec<(&str, &str)> {
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool {
                tool_call_id,
                content,
            } => Some((tool_call_id.as_str(), content.as_str())),
            _ => None,
        })
        .collect()
}

fn assistant_tool_calls(messages: &[Message]) -> Option<Vec<ToolCall>> {
    messages.iter().find_map(|message| match message {
        Message::Assistant { tool_calls, .. } => Some(tool_calls.clone()),
        _ => None,
    })
}

fn agent(chat: Arc<FakeChat>, workspace: &tempfile::TempDir) -> Agent {
    agent_with(
        chat,
        workspace,
        AgentConfig::default(),
        non_interactive_pause(),
    )
}

fn agent_with(
    chat: Arc<FakeChat>,
    workspace: &tempfile::TempDir,
    config: AgentConfig,
    pause: PausePrompt,
) -> Agent {
    Agent::new(chat, SANDBOX.clone(), workspace.path(), config, pause)
}

#[tokio::test]
async fn batched_calls_execute_in_request_order_before_the_next_llm_call() {
    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![
                run_code("first", "print('alpha')"),
                run_code("second", "print('beta')"),
            ],
        ),
        reply(Some("finished"), Vec::new()),
    ]);
    let workspace = tempfile::tempdir().unwrap();
    let result = agent(chat.clone(), &workspace)
        .run("run both")
        .await
        .unwrap();

    assert_eq!(
        result,
        AgentResult {
            answer: "finished".to_owned(),
            steps: 2,
        }
    );

    let requests = chat.requests();
    assert_eq!(requests.len(), 2, "one LLM call per Step");
    assert_eq!(requests[0].tools.len(), 1, "only run_code is advertised");
    assert_eq!(requests[0].tools[0]["function"]["name"], "run_code");
    assert!(matches!(
        &requests[0].messages[0],
        Message::System { content } if content == SYSTEM_PROMPT
    ));

    let second = &requests[1].messages;
    let calls = assistant_tool_calls(second).expect("assistant message with tool calls");
    assert_eq!(calls.len(), 2, "the assistant message keeps its tool calls");
    assert_eq!(calls[0].id, "first");
    assert_eq!(calls[1].id, "second");

    let observations = tool_messages(second);
    assert_eq!(
        observations.len(),
        2,
        "all Observations precede the next call"
    );
    assert_eq!(observations[0].0, "first");
    assert_eq!(observations[1].0, "second");
    assert!(
        observations[0].1.contains("alpha"),
        "first observation: {}",
        observations[0].1
    );
    assert!(
        !observations[0].1.contains("beta"),
        "first observation ran another call: {}",
        observations[0].1
    );
    assert!(
        observations[1].1.contains("beta"),
        "second observation: {}",
        observations[1].1
    );
    assert!(observations[0].1.contains("tool: run_code"));
    assert!(observations[0].1.contains("exit_code: 0"));
    assert!(observations[0].1.contains("first_stderr_line: none"));
}

#[tokio::test]
async fn a_reply_without_tool_calls_is_the_final_answer() {
    let chat = FakeChat::new(vec![reply(Some("the answer is 42"), Vec::new())]);
    let workspace = tempfile::tempdir().unwrap();
    let result = agent(chat.clone(), &workspace).run("answer").await.unwrap();

    assert_eq!(
        result,
        AgentResult {
            answer: "the answer is 42".to_owned(),
            steps: 1,
        }
    );
    assert_eq!(chat.requests().len(), 1);
    assert_eq!(chat.requests()[0].messages.len(), 2);
    assert!(tool_messages(&chat.requests()[0].messages).is_empty());
}

#[tokio::test]
async fn the_step_limit_stops_at_25_without_an_extra_llm_call() {
    let replies = (0..30)
        .map(|step| reply(None, vec![run_code(&format!("call-{step}"), "pass")]))
        .collect();
    let chat = FakeChat::new(replies);
    let workspace = tempfile::tempdir().unwrap();
    let error = agent(chat.clone(), &workspace)
        .run("never finish")
        .await
        .unwrap_err();

    assert!(
        matches!(error, AgentError::StepLimit { max_steps: 25 }),
        "unexpected error: {error}"
    );
    assert_eq!(chat.requests().len(), 25, "no 26th LLM call");
}

#[tokio::test]
async fn invalid_arguments_and_unknown_tools_become_observations_and_the_loop_continues() {
    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![
                ToolCall::new("bad", "run_code", "{not json"),
                ToolCall::new("", "read_file", "{}"),
            ],
        ),
        reply(Some("recovered"), Vec::new()),
    ]);
    let workspace = tempfile::tempdir().unwrap();
    let result = agent(chat.clone(), &workspace)
        .run("break tools")
        .await
        .unwrap();

    assert_eq!(result.answer, "recovered");
    assert_eq!(chat.requests().len(), 2, "the loop continued after errors");

    let second = &chat.requests()[1].messages;
    let observations = tool_messages(second);
    assert_eq!(observations.len(), 2);
    assert!(
        observations[0].1.contains("invalid arguments"),
        "invalid JSON observation: {}",
        observations[0].1
    );
    assert!(
        observations[1].1.contains("unknown tool `read_file`"),
        "unknown tool observation: {}",
        observations[1].1
    );

    let calls = assistant_tool_calls(second).expect("assistant message with tool calls");
    assert_eq!(calls[0].id, "bad");
    assert!(!calls[1].id.is_empty(), "a missing call id is synthesized");
    assert_eq!(observations[1].0, calls[1].id);
}

#[tokio::test]
async fn run_code_observations_reflect_real_guest_execution() {
    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![ToolCall::new(
                "calc",
                "run_code",
                json!({ "code": "print(2 + 3)" }).to_string(),
            )],
        ),
        reply(Some("5"), Vec::new()),
    ]);
    let workspace = tempfile::tempdir().unwrap();
    let result = agent(chat.clone(), &workspace)
        .run("compute")
        .await
        .unwrap();

    assert_eq!(result.answer, "5");
    let second = &chat.requests()[1].messages;
    let content = tool_messages(second)[0].1;
    assert!(content.contains("tool: run_code"), "content: {content}");
    assert!(content.contains("exit_code: 0"), "content: {content}");
    assert!(content.contains("timed_out: false"), "content: {content}");
    assert!(content.contains("stdout_bytes: 2"), "content: {content}");
    assert!(
        content.contains("--- begin stdout (2 bytes) ---\n5\n"),
        "content: {content}"
    );
}

/// A `run_code` that writes `boom` to stderr and exits non-zero.
fn failing_run(id: &str) -> ToolCall {
    run_code(id, "import sys; sys.stderr.write('boom\\n'); sys.exit(3)")
}

/// Records every pause prompt and answers with `answer`.
fn recording_pause(answer: bool) -> (PausePrompt, Arc<Mutex<Vec<String>>>) {
    let prompts = Arc::new(Mutex::new(Vec::new()));
    let recorder = Arc::clone(&prompts);
    let pause: PausePrompt = Arc::new(move |last_stderr: &str| {
        recorder.lock().unwrap().push(last_stderr.to_owned());
        answer
    });
    (pause, prompts)
}

#[tokio::test]
async fn three_consecutive_failed_runs_pause_once_and_yes_resets_the_counter() {
    let chat = FakeChat::new(vec![
        reply(None, vec![failing_run("f1")]),
        reply(None, vec![failing_run("f2")]),
        reply(None, vec![failing_run("f3")]),
        reply(None, vec![failing_run("f4")]),
        reply(Some("finished"), Vec::new()),
    ]);
    let (pause, prompts) = recording_pause(true);
    let workspace = tempfile::tempdir().unwrap();
    let result = agent_with(chat.clone(), &workspace, AgentConfig::default(), pause)
        .run("fail repeatedly")
        .await
        .unwrap();

    assert_eq!(
        result,
        AgentResult {
            answer: "finished".to_owned(),
            steps: 5,
        }
    );
    let prompts = prompts.lock().unwrap();
    assert_eq!(
        prompts.len(),
        1,
        "the counter reset after yes, so the fourth failure does not pause"
    );
    assert!(prompts[0].contains("boom"), "last stderr: {}", prompts[0]);
    assert_eq!(chat.requests().len(), 5);
}

#[tokio::test]
async fn declining_the_pause_ends_the_loop_with_the_last_stderr() {
    let chat = FakeChat::new(vec![
        reply(None, vec![failing_run("f1")]),
        reply(None, vec![failing_run("f2")]),
        reply(None, vec![failing_run("f3")]),
        reply(Some("never reached"), Vec::new()),
    ]);
    let (pause, prompts) = recording_pause(false);
    let workspace = tempfile::tempdir().unwrap();
    let error = agent_with(chat.clone(), &workspace, AgentConfig::default(), pause)
        .run("give up")
        .await
        .unwrap_err();

    match error {
        AgentError::FailedRuns {
            failed_runs,
            stderr,
        } => {
            assert_eq!(failed_runs, 3);
            assert!(stderr.contains("boom"), "stderr: {stderr}");
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(prompts.lock().unwrap().len(), 1);
    assert_eq!(
        chat.requests().len(),
        3,
        "no LLM call after the pause ended the loop"
    );
}

#[tokio::test]
async fn a_non_interactive_session_never_continues_after_repeated_failures() {
    let chat = FakeChat::new(vec![
        reply(None, vec![failing_run("f1")]),
        reply(None, vec![failing_run("f2")]),
        reply(None, vec![failing_run("f3")]),
        reply(Some("never reached"), Vec::new()),
    ]);
    let workspace = tempfile::tempdir().unwrap();
    let error = agent(chat.clone(), &workspace)
        .run("no terminal")
        .await
        .unwrap_err();

    match error {
        AgentError::FailedRuns {
            failed_runs,
            stderr,
        } => {
            assert_eq!(failed_runs, 3);
            assert!(stderr.contains("boom"), "stderr: {stderr}");
        }
        other => panic!("unexpected error: {other}"),
    }
    assert_eq!(chat.requests().len(), 3);
}

#[tokio::test]
async fn a_run_with_exit_zero_resets_the_failure_counter() {
    let chat = FakeChat::new(vec![
        reply(None, vec![failing_run("f1")]),
        reply(None, vec![failing_run("f2")]),
        reply(None, vec![run_code("ok", "print('fine')")]),
        reply(None, vec![failing_run("f3")]),
        reply(None, vec![failing_run("f4")]),
        reply(Some("finished"), Vec::new()),
    ]);
    let (pause, prompts) = recording_pause(true);
    let workspace = tempfile::tempdir().unwrap();
    let result = agent_with(chat.clone(), &workspace, AgentConfig::default(), pause)
        .run("succeed in between")
        .await
        .unwrap();

    assert_eq!(result.answer, "finished");
    assert_eq!(result.steps, 6);
    assert!(
        prompts.lock().unwrap().is_empty(),
        "two failures, success, two failures never reaches the threshold"
    );
}

#[tokio::test]
async fn error_observations_do_not_count_as_failed_runs() {
    let chat = FakeChat::new(vec![
        reply(None, vec![ToolCall::new("bad", "run_code", "{not json")]),
        reply(None, vec![ToolCall::new("unknown", "read_file", "{}")]),
        reply(None, vec![ToolCall::new("missing", "run_code", "{}")]),
        reply(Some("finished"), Vec::new()),
    ]);
    let (pause, prompts) = recording_pause(false);
    let workspace = tempfile::tempdir().unwrap();
    let result = agent_with(chat.clone(), &workspace, AgentConfig::default(), pause)
        .run("break the tools")
        .await
        .unwrap();

    assert_eq!(result.answer, "finished");
    assert!(
        prompts.lock().unwrap().is_empty(),
        "invalid arguments and unknown tools are not failed Runs"
    );
}
