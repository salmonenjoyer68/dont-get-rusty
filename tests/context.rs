//! Context compaction: pure budget tests plus one Agent Loop run with a fake
//! `Chat` and the real Sandbox.
//!
//! The pure tests build [`Entry`] values directly, so their Observations are
//! deterministic. The integration test floods stdout through the real Guest,
//! then asserts what the host actually sends to the model.

use std::collections::VecDeque;
use std::sync::{Arc, LazyLock, Mutex};

use async_trait::async_trait;
use dont_get_rusty::agent::{
    Agent, AgentConfig, Observation, SYSTEM_PROMPT, non_interactive_overwrite,
    non_interactive_pause,
};
use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::audit::AuditLog;
use dont_get_rusty::context::{HistoryEntry, compact};
use dont_get_rusty::fs::Workspace;
use dont_get_rusty::llm::{
    Chat, ChatResponse, Choice, LlmError, Message, ResponseMessage, ToolCall,
};
use dont_get_rusty::policy::Policy;
use dont_get_rusty::sandbox::{RunResult, Sandbox};
use serde_json::{Value, json};

static SANDBOX: LazyLock<Arc<Sandbox>> = LazyLock::new(|| {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    Arc::new(Sandbox::new(image.root()).expect("failed to create sandbox"))
});

/// Holds the Audit Log directory for the whole test binary, outside every
/// temporary Workspace.
static AUDIT_DIR: LazyLock<tempfile::TempDir> =
    LazyLock::new(|| tempfile::tempdir().expect("audit directory should exist"));

#[derive(Debug, Clone)]
struct Request {
    messages: Vec<Message>,
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
    async fn chat(&self, messages: &[Message], _tools: &[Value]) -> Result<ChatResponse, LlmError> {
        self.requests.lock().unwrap().push(Request {
            messages: messages.to_vec(),
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

fn call(id: &str) -> ToolCall {
    run_code(id, "pass")
}

fn observation(exit_code: i32, stdout: &str, stderr: &str) -> Observation {
    Observation::from_run(
        "run_code",
        RunResult {
            exit_code,
            stdout: stdout.to_owned(),
            stderr: stderr.to_owned(),
            timed_out: false,
            truncated: false,
            duration_ms: 7,
            stdout_bytes: stdout.len() as u64,
            stderr_bytes: stderr.len() as u64,
        },
    )
}

fn tool_content(messages: &[Message], tool_call_id: &str) -> String {
    messages
        .iter()
        .find_map(|message| match message {
            Message::Tool {
                tool_call_id: id,
                content,
            } if id == tool_call_id => Some(content.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no tool message for `{tool_call_id}`"))
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

fn history(old: Observation, new: Observation) -> Vec<HistoryEntry> {
    vec![
        HistoryEntry::Message(Message::system("system")),
        HistoryEntry::Message(Message::user("task")),
        HistoryEntry::Message(Message::assistant(None, vec![call("old"), call("new")])),
        HistoryEntry::Observation {
            tool_call_id: "old".to_owned(),
            observation: old,
        },
        HistoryEntry::Observation {
            tool_call_id: "new".to_owned(),
            observation: new,
        },
    ]
}

#[test]
fn the_newest_observation_stays_verbatim_and_older_ones_are_stubbed() {
    let old = observation(1, "old stdout", "old stderr");
    let new = observation(0, "new stdout", "");
    let new_text = new.to_content();
    let entries = history(old, new);
    let fixed = "system".chars().count() + "task".chars().count();
    let budget = fixed + new_text.chars().count() + 200;

    let messages = compact(&entries, budget);

    assert_eq!(messages.len(), 5);
    assert_eq!(messages[0], Message::system("system"));
    assert_eq!(messages[1], Message::user("task"));
    assert_eq!(
        messages[2],
        Message::assistant(None, vec![call("old"), call("new")]),
        "assistant tool_calls are never rewritten"
    );

    let old_text = tool_content(&messages, "old");
    assert!(old_text.contains("tool: run_code"), "{old_text}");
    assert!(old_text.contains("exit_code: 1"), "{old_text}");
    assert!(old_text.contains("stdout_bytes: 10"), "{old_text}");
    assert!(old_text.contains("stderr_bytes: 10"), "{old_text}");
    assert!(
        old_text.contains("first_stderr_line: old stderr"),
        "{old_text}"
    );
    assert!(old_text.contains("compacted Observation"), "{old_text}");
    assert!(!old_text.contains("old stdout"), "{old_text}");

    assert_eq!(tool_content(&messages, "new"), new_text);

    let total: usize = messages
        .iter()
        .map(|message| match message {
            Message::System { content } | Message::User { content } => content.chars().count(),
            Message::Assistant { content, .. } => {
                content.as_deref().map_or(0, |text| text.chars().count())
            }
            Message::Tool { content, .. } => content.chars().count(),
        })
        .sum();
    assert!(total <= budget, "{total} characters exceed {budget}");
}

#[test]
fn compaction_is_monotonic_once_a_newer_observation_is_stubbed() {
    let newest = observation(0, "new stdout", "");
    let middle = observation(0, &"m".repeat(2_000), "");
    let oldest = observation(0, "oldest stdout", "");
    let newest_text = newest.to_content();
    let oldest_text = oldest.to_content();
    let entries = vec![
        HistoryEntry::Message(Message::system("system")),
        HistoryEntry::Message(Message::user("task")),
        HistoryEntry::Message(Message::assistant(
            None,
            vec![call("oldest"), call("middle"), call("newest")],
        )),
        HistoryEntry::Observation {
            tool_call_id: "oldest".to_owned(),
            observation: oldest,
        },
        HistoryEntry::Observation {
            tool_call_id: "middle".to_owned(),
            observation: middle,
        },
        HistoryEntry::Observation {
            tool_call_id: "newest".to_owned(),
            observation: newest,
        },
    ];
    let fixed = "system".chars().count() + "task".chars().count();
    // The newest fits verbatim, the middle does not, and the small oldest would
    // still fit in the budget left after the middle's stub. Monotonic
    // compaction must stub it anyway because a newer Observation was stubbed.
    let budget = fixed + newest_text.chars().count() + 600;

    let messages = compact(&entries, budget);

    assert_eq!(
        tool_content(&messages, "newest"),
        newest_text,
        "the newest Observation stays verbatim"
    );
    assert!(
        tool_content(&messages, "middle").contains("compacted Observation"),
        "the oversized middle Observation is stubbed"
    );
    let rendered_oldest = tool_content(&messages, "oldest");
    assert!(
        rendered_oldest.contains("compacted Observation"),
        "an older Observation must not outlive a stubbed newer one: {rendered_oldest}"
    );
    assert_ne!(
        rendered_oldest, oldest_text,
        "the oldest Observation is not sent verbatim"
    );
}

#[test]
fn an_oversized_newest_observation_is_sent_head_and_tail_with_a_marker() {
    let huge = observation(0, &"x".repeat(5_000), "");
    let entries = vec![
        HistoryEntry::Message(Message::system("system")),
        HistoryEntry::Message(Message::user("task")),
        HistoryEntry::Message(Message::assistant(None, vec![call("huge")])),
        HistoryEntry::Observation {
            tool_call_id: "huge".to_owned(),
            observation: huge,
        },
    ];
    let budget = "system".chars().count() + "task".chars().count() + 200;

    let messages = compact(&entries, budget);
    let text = tool_content(&messages, "huge");

    assert!(text.contains("characters omitted"), "{text}");
    assert!(text.chars().count() <= 200, "{}", text.chars().count());
    assert!(text.starts_with("tool: run_code"), "{text}");
    assert!(text.ends_with("--- end stderr ---\n"), "{text}");
    assert!(
        !text.contains(&"x".repeat(200)),
        "the middle of the output must be dropped"
    );
}

#[test]
fn a_budget_with_no_room_still_answers_every_tool_call() {
    let entries = vec![
        HistoryEntry::Message(Message::system("system")),
        HistoryEntry::Message(Message::user("task")),
        HistoryEntry::Message(Message::assistant(None, vec![call("bad"), call("ok")])),
        HistoryEntry::Observation {
            tool_call_id: "bad".to_owned(),
            observation: Observation::error("run_code", "invalid arguments: oops"),
        },
        HistoryEntry::Observation {
            tool_call_id: "ok".to_owned(),
            observation: observation(0, "fine", ""),
        },
    ];

    let messages = compact(&entries, 0);

    assert_eq!(messages[0], Message::system("system"));
    assert_eq!(messages[1], Message::user("task"));
    assert_eq!(
        messages[2],
        Message::assistant(None, vec![call("bad"), call("ok")])
    );

    let observations = tool_messages(&messages);
    assert_eq!(observations.len(), 2);
    assert_eq!(observations[0].0, "bad");
    assert_eq!(observations[1].0, "ok");

    let error_text = observations[0].1;
    assert!(
        error_text.contains("error: invalid arguments: oops"),
        "{error_text}"
    );
    assert!(error_text.contains("exit_code: none"), "{error_text}");
    assert!(
        observations[1].1.contains("characters omitted"),
        "{}",
        observations[1].1
    );
}

#[test]
fn a_history_within_budget_is_returned_unchanged() {
    let only = observation(0, "fine", "");
    let text = only.to_content();
    let entries = vec![
        HistoryEntry::Message(Message::system("system")),
        HistoryEntry::Message(Message::user("task")),
        HistoryEntry::Message(Message::assistant(None, vec![call("only")])),
        HistoryEntry::Observation {
            tool_call_id: "only".to_owned(),
            observation: only,
        },
    ];
    let budget = "system".chars().count() + "task".chars().count() + text.chars().count();

    let messages = compact(&entries, budget);

    assert_eq!(
        messages,
        vec![
            Message::system("system"),
            Message::user("task"),
            Message::assistant(None, vec![call("only")]),
            Message::tool("only", text),
        ]
    );
}

#[tokio::test]
async fn compaction_sends_exactly_one_llm_request_per_step() {
    let chat = FakeChat::new(vec![
        reply(
            None,
            vec![
                run_code("a", "print('a' * 4000)"),
                run_code("b", "print('b' * 4000)"),
                run_code("c", "print('c' * 4000)"),
            ],
        ),
        reply(Some("done"), Vec::new()),
    ]);
    let task = "flood the context";
    let fixed = SYSTEM_PROMPT.chars().count() + task.chars().count();
    let config = AgentConfig {
        context_budget_chars: fixed + 300,
        ..AgentConfig::default()
    };
    let workspace = tempfile::tempdir().unwrap();
    let workspace = Workspace::open(workspace.path()).expect("workspace should open");
    let audit =
        AuditLog::open(AUDIT_DIR.path(), workspace.root_path()).expect("audit log should open");
    let result = Agent::new(
        chat.clone(),
        SANDBOX.clone(),
        workspace,
        config,
        Policy::default(),
        audit,
        non_interactive_pause(),
        non_interactive_overwrite(),
    )
    .run(task)
    .await
    .unwrap();

    assert_eq!(result.answer, "done");
    assert_eq!(chat.requests().len(), 2, "compaction never calls the model");

    let request = &chat.requests()[1].messages;
    let calls = assistant_tool_calls(request).expect("assistant message with tool calls");
    let observations = tool_messages(request);
    assert_eq!(observations.len(), calls.len(), "one answer per tool call");
    for ((id, content), call) in observations.iter().zip(&calls) {
        assert_eq!(*id, call.id);
        assert!(!content.is_empty());
    }

    assert!(
        observations[0].1.contains("compacted Observation"),
        "{}",
        observations[0].1
    );
    assert!(
        observations[1].1.contains("compacted Observation"),
        "{}",
        observations[1].1
    );
    assert!(
        observations[2].1.contains("characters omitted"),
        "the newest observation is head+tail: {}",
        observations[2].1
    );
}
