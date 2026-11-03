//! The append-only Audit Log: one JSON line per Tool Call.
//!
//! Every Tool Call the Agent Loop handles — allowed, denied, malformed, or
//! failed — appends exactly one [`AuditEvent`] to `<cwd>/.agent/audit.jsonl`.
//! The file lives outside the Workspace (opening it fails otherwise), so the
//! Guest's read-only preopen cannot expose it and the model's file tools
//! cannot reach it.
//!
//! The log proves what crossed the boundary without duplicating it. The
//! `code`, `content`, and `stdin` argument fields are replaced by an
//! `<omitted: N bytes>` marker *before* the arguments are shortened to
//! [`MAX_ARGS_PREFIX`] bytes, and `args_sha256` hashes the full raw argument
//! string, so an event can be matched against the payload it describes. A
//! payload that is not a JSON object is not stored at all, for the same
//! reason: nothing in it can be identified and redacted. Piped `stdin` is
//! redacted too: it is model-supplied data crossing the boundary, not
//! metadata.
//!
//! `decision` is `deny` both for a Policy refusal and for a Tool Call that
//! was allowed to run but failed on the host (quota, escape, overwrite
//! decline, malformed arguments, Sandbox setup); `denial_reason` carries the
//! error. A Guest Run always records `allow`, even when its exit code is
//! non-zero: the Guest ran, and its exit code is the recorded result.
//!
//! Each event is one synchronous append followed by a flush, so a crash keeps
//! every completed event and at worst loses the one in flight. A write failure
//! is an error, never a silently dropped event.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};

use serde::Serialize;
use sha2::{Digest, Sha256};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

use crate::agent::Observation;
use crate::llm::ToolCall;
use crate::policy::Decision;

/// Directory created under the working directory for runtime artifacts.
pub const DIR: &str = ".agent";

/// File name inside [`DIR`].
pub const FILE: &str = "audit.jsonl";

/// Maximum number of argument bytes kept in [`AuditEvent::args_prefix`].
pub const MAX_ARGS_PREFIX: usize = 2 * 1024;

/// Argument fields that carry code, file bodies, or piped input and are never
/// stored.
const REDACTED_FIELDS: [&str; 3] = ["code", "content", "stdin"];

/// One Audit Event: what was called, what the Policy decided, what it did.
///
/// Field order is the serialized line order. `decision` is `deny` for a
/// Policy refusal and for a host-side execution failure; `denial_reason` is
/// absent only when the call was allowed and produced a non-error result
/// (including a Guest Run with a non-zero exit code). `exit_code` is `null`
/// for the host file tools and for error Observations, matching
/// [`Observation::exit_code`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuditEvent {
    /// RFC 3339 UTC timestamp of the host write.
    pub ts: String,
    /// Id shared by every event of one process; generated once per process.
    pub session: String,
    /// Step (LLM turn) that produced the Tool Call, zero-based.
    pub step: usize,
    /// Tool name exactly as the model sent it, including unknown names.
    pub tool: String,
    /// Lowercase hex SHA-256 of the full raw `function.arguments` string.
    pub args_sha256: String,
    /// Redacted argument prefix, at most [`MAX_ARGS_PREFIX`] bytes.
    pub args_prefix: String,
    /// `allow` or `deny`.
    pub decision: &'static str,
    /// Why the Policy denied the call; absent for allowed calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denial_reason: Option<String>,
    /// Guest exit code, if the Tool Call completed a Run.
    pub exit_code: Option<i32>,
    /// stdout bytes captured for the model.
    pub stdout_bytes: u64,
    /// stderr bytes captured for the model.
    pub stderr_bytes: u64,
    /// Whether the Sandbox discarded output bytes for the model.
    pub truncated: bool,
    /// Host-observed wall-clock duration of the Tool Call.
    pub duration_ms: u64,
}

impl AuditEvent {
    /// Builds the event for one Tool Call from its Policy decision and the
    /// Observation the loop is about to feed back to the model.
    pub fn new(
        session: impl Into<String>,
        step: usize,
        call: &ToolCall,
        decision: &Decision,
        observation: &Observation,
        duration_ms: u64,
    ) -> Self {
        Self {
            ts: utc_now(),
            session: session.into(),
            step,
            tool: call.function.name.clone(),
            args_sha256: sha256_hex(&call.function.arguments),
            args_prefix: args_prefix(&call.function.arguments),
            decision: if decision.is_allowed() {
                "allow"
            } else {
                "deny"
            },
            denial_reason: decision.reason().map(ToOwned::to_owned),
            exit_code: observation.exit_code,
            stdout_bytes: observation.stdout_bytes,
            stderr_bytes: observation.stderr_bytes,
            truncated: observation.truncated,
            duration_ms,
        }
    }
}

/// The append-only JSONL file every Tool Call is written to.
#[derive(Debug)]
pub struct AuditLog {
    path: PathBuf,
    session: String,
    file: Mutex<File>,
}

impl AuditLog {
    /// Opens `<cwd>/.agent/audit.jsonl`, creating `.agent/` when needed.
    ///
    /// Refuses — rather than writing somewhere the model can read — when that
    /// path would land inside `workspace_root`. Both sides are canonicalized
    /// first, so a symlinked or relative Workspace is compared by its real
    /// location.
    pub fn open(cwd: &Path, workspace_root: &Path) -> Result<Self, AuditError> {
        let dir = cwd.join(DIR);
        std::fs::create_dir_all(&dir).map_err(|source| AuditError::Directory {
            path: dir.clone(),
            source,
        })?;
        let dir = dir.canonicalize().map_err(|source| AuditError::Resolve {
            path: dir.clone(),
            source,
        })?;
        let workspace = workspace_root
            .canonicalize()
            .map_err(|source| AuditError::Resolve {
                path: workspace_root.to_path_buf(),
                source,
            })?;
        let path = dir.join(FILE);
        if path.starts_with(&workspace) {
            return Err(AuditError::InsideWorkspace { path, workspace });
        }
        Self::at(path)
    }

    /// Opens the log at `path`, appending to any existing file.
    ///
    /// The parent directory must exist; [`AuditLog::open`] creates `.agent/`
    /// and enforces the outside-the-Workspace rule.
    pub fn at(path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        let path = path.into();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|source| AuditError::Open {
                path: path.clone(),
                source,
            })?;
        Ok(Self {
            path,
            session: SESSION.clone(),
            file: Mutex::new(file),
        })
    }

    /// The file events are appended to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The id shared by every event of this process.
    pub fn session(&self) -> &str {
        &self.session
    }

    /// Appends one event for `call`, returning a write failure to the caller so
    /// the loop can stop instead of losing the record.
    pub fn record(
        &self,
        step: usize,
        call: &ToolCall,
        decision: &Decision,
        observation: &Observation,
        duration_ms: u64,
    ) -> Result<(), AuditError> {
        self.record_event(&AuditEvent::new(
            &self.session,
            step,
            call,
            decision,
            observation,
            duration_ms,
        ))
    }

    /// Appends `event` as one JSON line and flushes it.
    pub fn record_event(&self, event: &AuditEvent) -> Result<(), AuditError> {
        let mut line =
            serde_json::to_string(event).map_err(|source| AuditError::Encode { source })?;
        line.push('\n');
        let mut file = self
            .file
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        file.write_all(line.as_bytes())
            .and_then(|()| file.flush())
            .map_err(|source| AuditError::Write {
                path: self.path.clone(),
                source,
            })
    }
}

/// The id shared by every event of one process.
///
/// A process handles one session, so the id is generated once and reused for
/// every event.
static SESSION: LazyLock<String> = LazyLock::new(|| format!("{:016x}", rand::random::<u64>()));

/// The current time as RFC 3339 in UTC, e.g. `2026-11-03T12:00:00.123Z`.
fn utc_now() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        // Rfc3339 is a well-known format and `OffsetDateTime` always supports
        // it; the fallback only keeps a broken clock from aborting a session.
        .unwrap_or_else(|_| "1970-01-01T00:00:00Z".to_owned())
}

/// Lowercase hex SHA-256 of `text`.
fn sha256_hex(text: &str) -> String {
    let digest = Sha256::digest(text.as_bytes());
    let mut hex = String::with_capacity(64);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// The argument prefix stored in an event.
///
/// Code, file bodies, and piped stdin are replaced by a byte-count marker
/// before the prefix is cut, so no amount of truncation can expose them;
/// `args_sha256` still covers the full payload. A payload that is not a JSON
/// object is reduced to its length, because its fields cannot be identified
/// and redacted.
fn args_prefix(arguments: &str) -> String {
    let redacted = match serde_json::from_str::<serde_json::Value>(arguments) {
        Ok(serde_json::Value::Object(mut object)) => {
            for field in REDACTED_FIELDS {
                if let Some(value) = object.get_mut(field) {
                    let bytes = match &*value {
                        serde_json::Value::String(text) => text.len(),
                        other => other.to_string().len(),
                    };
                    *value = serde_json::Value::String(format!("<omitted: {bytes} bytes>"));
                }
            }
            serde_json::to_string(&serde_json::Value::Object(object)).ok()
        }
        _ => None,
    };
    let redacted =
        redacted.unwrap_or_else(|| format!("<unparsed arguments: {} bytes>", arguments.len()));
    truncate_utf8(&redacted, MAX_ARGS_PREFIX).to_owned()
}

/// The longest prefix of `text` that is at most `max` bytes and ends on a
/// character boundary.
fn truncate_utf8(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// An Audit Log that could not be opened, or an event that could not be
/// appended.
#[derive(Debug, thiserror::Error)]
pub enum AuditError {
    #[error("failed to create the audit directory `{path}`")]
    Directory {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to resolve `{path}`")]
    Resolve {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error(
        "audit log `{path}` is inside the Workspace `{workspace}`; keep the Audit Log outside \
         the Workspace so the Guest and the model cannot read it"
    )]
    InsideWorkspace { path: PathBuf, workspace: PathBuf },
    #[error("failed to open audit log `{path}`")]
    Open {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("failed to encode an audit event")]
    Encode {
        #[source]
        source: serde_json::Error,
    },
    #[error("failed to append to audit log `{path}`")]
    Write {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}
