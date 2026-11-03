//! The Agent Loop: alternating LLM turns with Tool Call execution.
//!
//! A Step is one LLM turn. Every Tool Call in a reply
//! executes in the order the model requested it, and all of their
//! [`Observation`]s are appended to the conversation before the next LLM
//! request. The loop ends when a reply carries no `tool_calls` (the final
//! answer) or when the configured Step limit is reached.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::audit::{AuditError, AuditLog};
use crate::context::{self, Entry};
use crate::fs::{Workspace, format_listing, resolve_rel};
use crate::llm::{Chat, LlmError, Message, ToolCall};
use crate::policy::{Decision, Policy};
use crate::sandbox::{RunResult, Sandbox};
use crate::tools::{self, ListFilesArgs, ReadFileArgs, RunCodeArgs, WriteFileArgs};

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
/// Built from the Policy's `[agent]` and `[llm]` sections. The loop enforces
/// `max_steps`, `max_failed_runs`, and `context_budget_chars`; `base_url`,
/// `model`, and `request_timeout` configure the LLM client.
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
/// `exit_code` is `Some` only for a completed Guest Run; host file tools and
/// error Observations carry `None`. An error Observation (invalid arguments,
/// unknown tool, Policy denial, Workspace quota or escape failure, Sandbox
/// setup failure) has a message in [`Observation::error`] and the loop
/// continues.
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
    /// Whether the Sandbox discarded output bytes for the model; the affected
    /// stream carries a truncation marker. Host file tools never truncate.
    pub truncated: bool,
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
            truncated: run.truncated,
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
            truncated: false,
            duration_ms: 0,
            stdout_bytes: 0,
            stderr_bytes: 0,
            first_stderr_line: None,
            error: Some(message.into()),
        }
    }

    /// Builds an Observation for a host file tool's text output
    /// (`read_file`, `write_file`, `list_files`). There is no Guest Run, so the
    /// exit code is `None` and the byte count is the captured stdout length.
    pub fn from_text(tool: impl Into<String>, output: impl Into<String>) -> Self {
        let stdout = output.into();
        Self {
            tool: tool.into(),
            exit_code: None,
            stdout_bytes: stdout.len() as u64,
            stdout,
            stderr: String::new(),
            timed_out: false,
            truncated: false,
            duration_ms: 0,
            stderr_bytes: 0,
            first_stderr_line: None,
            error: None,
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
        out.push_str(&format!("truncated: {}\n", self.truncated));
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
    /// The Audit Log could not be opened or appended to.
    #[error(transparent)]
    Audit(#[from] AuditError),
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

/// Decides whether `write_file` may overwrite a file that existed before this
/// session's first write to it.
///
/// The argument is the model-supplied Workspace-relative path, for the prompt
/// text. Returning `true` allows the write; `false` leaves the file untouched
/// and becomes a denial Observation, so the loop continues. The Agent only
/// consults this when the target existed before the session and the path is
/// not one the agent has already written.
pub type OverwritePrompt = Arc<dyn Fn(&str) -> bool + Send + Sync>;

/// The safe default for a session without a terminal: never overwrite.
pub fn non_interactive_overwrite() -> OverwritePrompt {
    Arc::new(|_path| false)
}

/// Answers yes to every overwrite without asking (the `--yes` behavior).
pub fn always_overwrite() -> OverwritePrompt {
    Arc::new(|_path| true)
}

/// Drives one task through the LLM, the Workspace, and the Sandbox.
pub struct Agent {
    chat: Arc<dyn Chat>,
    sandbox: Arc<Sandbox>,
    workspace: Workspace,
    config: AgentConfig,
    policy: Policy,
    audit: AuditLog,
    pause: PausePrompt,
    overwrite: OverwritePrompt,
    /// Workspace-relative paths this session has successfully written. They
    /// are the "created earlier in the session" files that overwrite without a
    /// prompt; a pre-existing path enters the set only after its overwrite was
    /// confirmed.
    session_writes: Mutex<HashSet<PathBuf>>,
}

impl Agent {
    /// Uses `chat` for LLM turns, `sandbox` for `run_code`, `workspace` for
    /// the file tools and the Guest's read-only `/workspace` preopen,
    /// `policy` for every Tool Call decision, and `audit` to record every Tool
    /// Call exactly once.
    ///
    /// `pause` is asked before every further Step once
    /// [`AgentConfig::max_failed_runs`] consecutive Runs have failed; pass
    /// [`non_interactive_pause`] when no user can answer.
    ///
    /// `overwrite` is asked before overwriting a file that existed before the
    /// session; pass [`non_interactive_overwrite`] when no user can answer.
    /// Files the agent created or already overwrote this session never reach
    /// it.
    // One argument per collaborator the session owns; grouping them would
    // only add an anonymous bag struct.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chat: Arc<dyn Chat>,
        sandbox: Arc<Sandbox>,
        workspace: Workspace,
        config: AgentConfig,
        policy: Policy,
        audit: AuditLog,
        pause: PausePrompt,
        overwrite: OverwritePrompt,
    ) -> Self {
        Self {
            chat,
            sandbox,
            workspace,
            config,
            policy,
            audit,
            pause,
            overwrite,
            session_writes: Mutex::new(HashSet::new()),
        }
    }

    /// Runs `task` until the model answers or the Step limit is reached.
    pub async fn run(&self, task: &str) -> Result<AgentResult, AgentError> {
        let mut conversation = vec![
            Entry::Message(Message::system(SYSTEM_PROMPT)),
            Entry::Message(Message::user(task)),
        ];
        let tools = tools::tool_definitions();
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
                let observation = self.execute(call, step).await?;

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

    /// Executes one Tool Call, consulting the Policy first and appending one
    /// Audit Event for it.
    ///
    /// The decision is made once, here, from the raw arguments, so every Tool
    /// Call has a recorded verdict even when its arguments are malformed or its
    /// name is unknown. A Policy allow whose execution fails (quota, escape,
    /// overwrite decline, malformed arguments, Sandbox setup) is recorded as a
    /// denial with the error as the reason, so the log never says `allow` for
    /// something that did not happen. A completed Guest Run stays `allow` even
    /// with a non-zero exit code: the Guest ran, and the exit code records the
    /// outcome. A denial becomes an error Observation; an Audit write failure
    /// ends the loop instead of dropping the event.
    async fn execute(&self, call: &ToolCall, step: usize) -> Result<Observation, AgentError> {
        let started = Instant::now();
        let decision = self.decide(call);
        let observation = match &decision {
            Decision::Allow => self.dispatch(call).await,
            Decision::Deny { reason } => {
                Observation::error(call.function.name.as_str(), reason.clone())
            }
        };
        // A Tool Call that was allowed to run but failed is audited as a
        // denial: nothing crossed the boundary, and the error says why.
        let recorded = match (&decision, observation.error.as_deref()) {
            (Decision::Allow, Some(reason)) => Decision::deny(reason),
            _ => decision.clone(),
        };
        let duration_ms = started.elapsed().as_millis() as u64;
        self.audit
            .record(step, call, &recorded, &observation, duration_ms)?;
        Ok(observation)
    }

    /// The Policy verdict for `call`.
    ///
    /// The path is read straight out of the raw arguments, so a payload that
    /// fails to deserialize still reaches the Policy; arguments without a
    /// readable `path` are checked as a call with no path, which the Policy
    /// denies for `read_file`/`write_file`.
    fn decide(&self, call: &ToolCall) -> Decision {
        let path = argument_path(&call.function.arguments);
        self.policy.check(&call.function.name, path.as_deref())
    }

    /// Routes an allowed Tool Call to its tool. Every failure is an error
    /// Observation, so the loop never aborts on a malformed call.
    async fn dispatch(&self, call: &ToolCall) -> Observation {
        match call.function.name.as_str() {
            "run_code" => self.run_code(call).await,
            "read_file" => self.read_file(call),
            "write_file" => self.write_file(call),
            "list_files" => self.list_files(call),
            other => Observation::error(other, format!("unknown tool `{other}`")),
        }
    }

    /// Runs the Guest on the blocking pool; the Sandbox drives its own Tokio
    /// runtime internally, so a Run must not happen on an async worker thread.
    async fn run_code(&self, call: &ToolCall) -> Observation {
        let args: RunCodeArgs = match parse_args("run_code", call) {
            Ok(args) => args,
            Err(error) => return *error,
        };

        let sandbox = Arc::clone(&self.sandbox);
        let workspace = self.workspace.root_path().to_path_buf();
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

    /// Reads one Workspace file. The bytes are decoded lossily for the model;
    /// the reported byte count is the captured text.
    fn read_file(&self, call: &ToolCall) -> Observation {
        let args: ReadFileArgs = match parse_args("read_file", call) {
            Ok(args) => args,
            Err(error) => return *error,
        };
        match self.workspace.read_file(&args.path) {
            Ok(bytes) => Observation::from_text("read_file", String::from_utf8_lossy(&bytes)),
            Err(error) => Observation::error("read_file", error.to_string()),
        }
    }

    /// Writes one Workspace file through the `cap-std` handle, so the quota
    /// and containment checks run before anything is created or truncated.
    ///
    /// The overwrite prompt is required exactly when the target is a file that
    /// existed before this session and has not already been written by the
    /// agent. A pre-existing path is prompted once; after a confirmed write,
    /// later writes to it are silent because it is now a session file. A
    /// denial leaves the file untouched and becomes an error Observation, so
    /// the loop continues. `[fs].confirm_overwrite = false` disables the
    /// prompt. Paths the agent successfully creates or writes are tracked in
    /// memory for the life of the Agent and are never prompted again.
    fn write_file(&self, call: &ToolCall) -> Observation {
        let args: WriteFileArgs = match parse_args("write_file", call) {
            Ok(args) => args,
            Err(error) => return *error,
        };
        let rel = match resolve_rel(Path::new(&args.path)) {
            Ok(rel) => rel,
            Err(error) => return Observation::error("write_file", error.to_string()),
        };
        let already_written = self
            .session_writes
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .contains(&rel);
        let confirm = |_rel: &Path| {
            !self.policy.fs.confirm_overwrite || already_written || (self.overwrite)(&args.path)
        };
        match self
            .workspace
            .write_file_with_confirm(&args.path, args.content.as_bytes(), confirm)
        {
            Ok(info) => {
                self.session_writes
                    .lock()
                    .unwrap_or_else(|poison| poison.into_inner())
                    .insert(rel);
                Observation::from_text(
                    "write_file",
                    format!("wrote {} bytes to {}", info.bytes_written, args.path),
                )
            }
            Err(error) => Observation::error("write_file", error.to_string()),
        }
    }

    /// Lists a Workspace directory in the stable `kind<TAB>size<TAB>name`
    /// format; `None` lists the Workspace root.
    fn list_files(&self, call: &ToolCall) -> Observation {
        let args: ListFilesArgs = match parse_args("list_files", call) {
            Ok(args) => args,
            Err(error) => return *error,
        };
        match self
            .workspace
            .list_files(args.path.as_deref().map(std::path::Path::new))
        {
            Ok(entries) => Observation::from_text("list_files", format_listing(&entries)),
            Err(error) => Observation::error("list_files", error.to_string()),
        }
    }
}

/// The `path` argument of a Tool Call, read from the raw JSON.
///
/// The Policy runs before deserialization so that a malformed or surprising
/// payload still gets a real verdict; a missing or non-string `path` is `None`.
fn argument_path(arguments: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(arguments).ok()?;
    value
        .get("path")?
        .as_str()
        .map(std::borrow::ToOwned::to_owned)
}

/// Parses a Tool Call's arguments, turning malformed ones into a boxed error
/// Observation named after `tool`.
fn parse_args<T: DeserializeOwned>(tool: &str, call: &ToolCall) -> Result<T, Box<Observation>> {
    serde_json::from_str(&call.function.arguments).map_err(|error| {
        Box::new(Observation::error(
            tool,
            format!("invalid arguments: {error}"),
        ))
    })
}
