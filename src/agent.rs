//! The Agent Loop: alternating LLM turns with Tool Call execution.
//!
//! A Step is one LLM turn. Every Tool Call in a reply
//! executes in the order the model requested it, and all of their
//! [`Observation`]s are appended to the conversation before the next LLM
//! request. The loop ends when a reply carries no `tool_calls` (the final
//! answer) or when the configured Step limit is reached.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;

use crate::context::{self, Entry};
use crate::llm::{Chat, LlmError, Message, ToolCall};
use crate::sandbox::{RunResult, Sandbox};
use crate::tools::{self, RunCodeArgs};

/// The v1 system prompt, verbatim from `docs/design.md` § System Prompt.
pub const SYSTEM_PROMPT: &str = "\
You are a coding agent running locally. You can write and run code only inside
a WebAssembly sandbox with no network and a restricted workspace folder.

- Use tools to inspect files and run code. Do not guess outputs.
- Python is available. Standard library only.
- Keep code small and testable. Run it, read the result, fix errors.
- If a task needs network or files outside the workspace, say so and ask the user.
- Use the provided tools. Reply without a tool call only when you have the final answer.
- Final answer: what you did, which files changed, how to run it.";

/// Host-side limits for one [`Agent::run`].
///
/// `base_url` and `model` are carried here for the Policy wiring and the LLM
/// client; `max_steps`, `max_failed_runs`, and `context_budget_chars` are
/// enforced by the loop itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    /// Maximum number of Steps before the loop stops.
    pub max_steps: usize,
    /// Consecutive failed `run_code` Runs that pause the loop; zero disables
    /// the pause.
    pub max_failed_runs: usize,
    /// Character budget for the conversation before compaction; see
    /// [`crate::context::compact`] for exactly what counts.
    pub context_budget_chars: usize,
    /// OpenAI-compatible API root, including the version segment.
    pub base_url: String,
    /// Model name sent in every request.
    pub model: String,
    /// Deadline for one LLM request.
    pub request_timeout: Duration,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_steps: 25,
            max_failed_runs: 3,
            context_budget_chars: 16_000,
            base_url: "http://localhost:11434/v1".to_owned(),
            model: "qwen3.5:4b".to_owned(),
            request_timeout: Duration::from_secs(120),
        }
    }
}

/// The host-produced result of one Tool Call, as fed back to the model.
///
/// Fields other than [`Observation::tool`] are `None`/zero for an error
/// Observation (invalid arguments, unknown tool, Sandbox setup failure).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation {
    /// Name of the tool that was called.
    pub tool: String,
    /// Guest exit code for a completed Run.
    pub exit_code: Option<i32>,
    /// Captured stdout, verbatim.
    pub stdout: String,
    /// Captured stderr, verbatim.
    pub stderr: String,
    /// Whether the Run hit the Sandbox deadline.
    pub timed_out: bool,
    /// Wall-clock duration of the Run in milliseconds.
    pub duration_ms: u64,
    /// Number of stdout bytes captured for the model.
    pub stdout_bytes: u64,
    /// Number of stderr bytes captured for the model.
    pub stderr_bytes: u64,
    /// First line of stderr, if the Run produced any.
    pub first_stderr_line: Option<String>,
    /// Host-side error, when the Tool Call could not be executed at all.
    pub error: Option<String>,
}

impl Observation {
    /// Builds an Observation from a completed Guest Run.
    pub fn from_run(tool: impl Into<String>, run: RunResult) -> Self {
        let stdout_bytes = run.stdout.len() as u64;
        let stderr_bytes = run.stderr.len() as u64;
        let first_stderr_line = run.stderr.lines().next().map(ToOwned::to_owned);
        Self {
            tool: tool.into(),
            exit_code: Some(run.exit_code),
            stdout: run.stdout,
            stderr: run.stderr,
            timed_out: run.timed_out,
            duration_ms: run.duration_ms,
            stdout_bytes,
            stderr_bytes,
            first_stderr_line,
            error: None,
        }
    }

    /// Builds an error Observation; the loop continues after these.
    pub fn error(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
            duration_ms: 0,
            stdout_bytes: 0,
            stderr_bytes: 0,
            first_stderr_line: None,
            error: Some(message.into()),
        }
    }

    /// Renders the Observation as the deterministic text of a
    /// [`Message::tool`] content. stdout and stderr are preserved verbatim
    /// between explicit begin/end markers.
    pub fn to_content(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("tool: {}\n", self.tool));
        if let Some(error) = &self.error {
            out.push_str(&format!("error: {error}\n"));
        }
        match self.exit_code {
            Some(code) => out.push_str(&format!("exit_code: {code}\n")),
            None => out.push_str("exit_code: none\n"),
        }
        out.push_str(&format!("timed_out: {}\n", self.timed_out));
        out.push_str(&format!("duration_ms: {}\n", self.duration_ms));
        out.push_str(&format!("stdout_bytes: {}\n", self.stdout_bytes));
        out.push_str(&format!("stderr_bytes: {}\n", self.stderr_bytes));
        match &self.first_stderr_line {
            Some(line) => out.push_str(&format!("first_stderr_line: {line}\n")),
            None => out.push_str("first_stderr_line: none\n"),
        }
        out.push_str(&format!(
            "--- begin stdout ({} bytes) ---\n",
            self.stdout_bytes
        ));
        out.push_str(&self.stdout);
        out.push_str("--- end stdout ---\n");
        out.push_str(&format!(
            "--- begin stderr ({} bytes) ---\n",
            self.stderr_bytes
        ));
        out.push_str(&self.stderr);
        out.push_str("--- end stderr ---\n");
        out
    }
}

/// Why an [`Agent::run`] stopped without a final answer.
#[derive(Debug, Error)]
pub enum AgentError {
    /// The LLM request failed after the client's own retry.
    #[error("LLM request failed: {0}")]
    Llm(#[from] LlmError),
    /// A successful response carried no `choices`.
    #[error("invalid LLM response: no choices")]
    EmptyReply,
    /// The loop used every allowed Step without a reply lacking tool calls.
    #[error("step limit reached: {max_steps} Steps without a final answer")]
    StepLimit { max_steps: usize },
    /// Consecutive `run_code` Runs failed and the user declined to continue.
    #[error("stopped after {failed_runs} consecutive failed Runs; last stderr:\n{stderr}")]
    FailedRuns { failed_runs: usize, stderr: String },
}

/// A completed Agent Loop run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentResult {
    /// The final assistant message (empty if the model omitted content).
    pub answer: String,
    /// Number of LLM turns consumed.
    pub steps: usize,
}

/// Decides whether the Loop continues after
/// [`AgentConfig::max_failed_runs`] consecutive failed Runs.
///
/// The argument is the last failed Run's stderr. Returning `true` resets the
/// failure counter and lets the loop continue; `false` ends it with
/// [`AgentError::FailedRuns`].
pub type PausePrompt = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// The safe default for a session without a terminal: never continue.
pub fn non_interactive_pause() -> PausePrompt {
    Arc::new(|_last_stderr| false)
}

/// Drives one task through the LLM and the Sandbox.
pub struct Agent {
    chat: Arc<dyn Chat>,
    sandbox: Arc<Sandbox>,
    workspace: PathBuf,
    config: AgentConfig,
    pause: PausePrompt,
}

impl Agent {
    /// Uses `chat` for LLM turns, `sandbox` for `run_code`, and `workspace`
    /// as the read-only `/workspace` preopen.
    ///
    /// `pause` is asked before every further Step once
    /// [`AgentConfig::max_failed_runs`] consecutive Runs have failed; pass
    /// [`non_interactive_pause`] when no user can answer.
    pub fn new(
        chat: Arc<dyn Chat>,
        sandbox: Arc<Sandbox>,
        workspace: impl Into<PathBuf>,
        config: AgentConfig,
        pause: PausePrompt,
    ) -> Self {
        Self {
            chat,
            sandbox,
            workspace: workspace.into(),
            config,
            pause,
        }
    }

    /// Runs `task` until the model answers or the Step limit is reached.
    pub async fn run(&self, task: &str) -> Result<AgentResult, AgentError> {
        let mut conversation = vec![
            Entry::Message(Message::system(SYSTEM_PROMPT)),
            Entry::Message(Message::user(task)),
        ];
        let tools = vec![tools::run_code_tool()];
        let mut failed_runs = 0usize;

        for step in 0..self.config.max_steps {
            // The full conversation is kept; only the request view is compacted.
            let request = context::compact(&conversation, self.config.context_budget_chars);
            let response = self.chat.chat(&request, &tools).await?;
            let choice = response
                .choices
                .into_iter()
                .next()
                .ok_or(AgentError::EmptyReply)?;
            let content = choice.message.content;
            let mut tool_calls = choice.message.tool_calls;

            if tool_calls.is_empty() {
                return Ok(AgentResult {
                    answer: content.unwrap_or_default(),
                    steps: step + 1,
                });
            }

            // The id is echoed back in the matching `tool` message; some
            // servers omit it, so synthesize a stable one.
            for (index, call) in tool_calls.iter_mut().enumerate() {
                if call.id.trim().is_empty() {
                    call.id = format!("call_{step}_{index}");
                }
            }

            conversation.push(Entry::Message(Message::assistant(
                content,
                tool_calls.clone(),
            )));

            // Every Observation must be in the conversation before the next
            // LLM call.
            for call in &tool_calls {
                let observation = self.execute(call).await;

                // A failed Run is a `run_code` with a non-zero exit code,
                // including traps and timeouts. Error Observations have no
                // exit code: they neither count nor reset the counter.
                if let Some(exit_code) = observation.exit_code {
                    if exit_code == 0 {
                        failed_runs = 0;
                    } else {
                        failed_runs += 1;
                        if self.config.max_failed_runs > 0
                            && failed_runs >= self.config.max_failed_runs
                        {
                            if !(self.pause)(&observation.stderr) {
                                return Err(AgentError::FailedRuns {
                                    failed_runs,
                                    stderr: observation.stderr,
                                });
                            }
                            failed_runs = 0;
                        }
                    }
                }

                conversation.push(Entry::Observation {
                    tool_call_id: call.id.clone(),
                    observation,
                });
            }
        }

        Err(AgentError::StepLimit {
            max_steps: self.config.max_steps,
        })
    }

    /// Executes one Tool Call. Only `run_code` is wired in v1; every failure
    /// becomes an Observation and never aborts the loop.
    ///
    /// The Sandbox drives its own Tokio runtime internally, so a Run must not
    /// happen on an async worker thread: it runs on the blocking pool.
    async fn execute(&self, call: &ToolCall) -> Observation {
        if call.function.name != "run_code" {
            return Observation::error(
                call.function.name.clone(),
                format!("unknown tool `{}`", call.function.name),
            );
        }

        let args: RunCodeArgs = match serde_json::from_str(&call.function.arguments) {
            Ok(args) => args,
            Err(error) => {
                return Observation::error("run_code", format!("invalid arguments: {error}"));
            }
        };

        let sandbox = Arc::clone(&self.sandbox);
        let workspace = self.workspace.clone();
        let result = tokio::task::spawn_blocking(move || {
            sandbox.run(&args.code, args.stdin.as_deref(), Some(&workspace))
        })
        .await;

        match result {
            Ok(Ok(run)) => Observation::from_run("run_code", run),
            Ok(Err(error)) => {
                Observation::error("run_code", format!("sandbox setup failed: {error}"))
            }
            Err(error) => Observation::error(
                "run_code",
                format!("sandbox task failed before reporting a result: {error}"),
            ),
        }
    }
}
