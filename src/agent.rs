//! The Agent Loop: alternating LLM turns with Tool Call execution.
//!
//! A Step is one LLM turn. Every Tool Call in a reply
//! executes in the order the model requested it, and all of their
//! [`Observation`]s are appended to the conversation before the next LLM
//! request. The loop ends when a reply carries no `tool_calls` (the final
//! answer) or when the configured Step limit is reached.

use std::sync::Arc;
use std::time::Duration;

use serde::de::DeserializeOwned;
use thiserror::Error;

use crate::fs::{Workspace, format_listing};
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
/// Built from the Policy's `[agent]` and `[llm]` sections. The v1 loop
/// enforces `max_steps`; `max_failed_runs` and `context_budget_chars` are
/// carried for the tickets that add the pause prompt and compaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentConfig {
    /// Maximum number of Steps before the loop stops.
    pub max_steps: usize,
    /// Consecutive failed `run_code` Runs that pause the loop (ticket #6).
    pub max_failed_runs: usize,
    /// Character budget for the conversation before compaction (ticket #6).
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
}

/// A completed Agent Loop run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AgentResult {
    /// The final assistant message (empty if the model omitted content).
    pub answer: String,
    /// Number of LLM turns consumed.
    pub steps: usize,
}

/// Drives one task through the LLM, the Workspace, and the Sandbox.
pub struct Agent {
    chat: Arc<dyn Chat>,
    sandbox: Arc<Sandbox>,
    workspace: Workspace,
    config: AgentConfig,
    policy: Policy,
}

impl Agent {
    /// Uses `chat` for LLM turns, `sandbox` for `run_code`, `workspace` for
    /// the file tools and the Guest's read-only `/workspace` preopen, and
    /// `policy` for the `[agent]`/`[llm]` settings already resolved into
    /// `config` plus every Tool Call decision.
    pub fn new(
        chat: Arc<dyn Chat>,
        sandbox: Arc<Sandbox>,
        workspace: Workspace,
        config: AgentConfig,
        policy: Policy,
    ) -> Self {
        Self {
            chat,
            sandbox,
            workspace,
            config,
            policy,
        }
    }

    /// Runs `task` until the model answers or the Step limit is reached.
    pub async fn run(&self, task: &str) -> Result<AgentResult, AgentError> {
        let mut messages = vec![Message::system(SYSTEM_PROMPT), Message::user(task)];
        let tools = tools::tool_definitions();

        for step in 0..self.config.max_steps {
            let response = self.chat.chat(&messages, &tools).await?;
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

            messages.push(Message::assistant(content, tool_calls.clone()));

            // Every Observation must be in the conversation before the next
            // LLM call.
            for call in &tool_calls {
                let observation = self.execute(call).await;
                messages.push(Message::tool(call.id.clone(), observation.to_content()));
            }
        }

        Err(AgentError::StepLimit {
            max_steps: self.config.max_steps,
        })
    }

    /// Executes one Tool Call. Every Tool Call passes through
    /// [`Policy::check`] and every failure becomes an Observation, so the loop
    /// never aborts on a denial or a Tool error.
    async fn execute(&self, call: &ToolCall) -> Observation {
        match call.function.name.as_str() {
            "run_code" => self.run_code(call).await,
            "read_file" => self.read_file(call),
            "write_file" => self.write_file(call),
            "list_files" => self.list_files(call),
            other => self
                .denied(other, None)
                .unwrap_or_else(|| Observation::error(other, format!("unknown tool `{other}`"))),
        }
    }

    /// Runs the Guest on the blocking pool; the Sandbox drives its own Tokio
    /// runtime internally, so a Run must not happen on an async worker thread.
    async fn run_code(&self, call: &ToolCall) -> Observation {
        let args: RunCodeArgs = match parse_args("run_code", call) {
            Ok(args) => args,
            Err(error) => return *error,
        };
        if let Some(denied) = self.denied("run_code", None) {
            return denied;
        }

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
        if let Some(denied) = self.denied("read_file", Some(&args.path)) {
            return denied;
        }
        match self.workspace.read_file(&args.path) {
            Ok(bytes) => Observation::from_text("read_file", String::from_utf8_lossy(&bytes)),
            Err(error) => Observation::error("read_file", error.to_string()),
        }
    }

    /// Writes one Workspace file through the `cap-std` handle, so the quota
    /// and containment checks run before anything is created or truncated.
    /// Overwrite confirmation is ticket #8; `confirm_overwrite` is parsed but
    /// not consulted yet.
    fn write_file(&self, call: &ToolCall) -> Observation {
        let args: WriteFileArgs = match parse_args("write_file", call) {
            Ok(args) => args,
            Err(error) => return *error,
        };
        if let Some(denied) = self.denied("write_file", Some(&args.path)) {
            return denied;
        }
        match self
            .workspace
            .write_file(&args.path, args.content.as_bytes())
        {
            Ok(info) => Observation::from_text(
                "write_file",
                format!("wrote {} bytes to {}", info.bytes_written, args.path),
            ),
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
        let path = args.path.as_deref();
        if let Some(denied) = self.denied("list_files", path) {
            return denied;
        }
        match self.workspace.list_files(path.map(std::path::Path::new)) {
            Ok(entries) => Observation::from_text("list_files", format_listing(&entries)),
            Err(error) => Observation::error("list_files", error.to_string()),
        }
    }

    /// Consults the Policy and renders a denial as an error Observation.
    fn denied(&self, tool: &str, path: Option<&str>) -> Option<Observation> {
        match self.policy.check(tool, path) {
            Decision::Allow => None,
            Decision::Deny { reason } => Some(Observation::error(tool, reason)),
        }
    }
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
