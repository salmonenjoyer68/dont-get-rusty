//! The Audit Log: one JSON line per Tool Call.
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
//! Each opened log is one session: it gets a fresh id, and every event it
//! appends carries that id. The CLI opens one log per process; the Web Demo
//! opens one per Task.
//!
//! `decision`/`denial_reason` are the Policy verdict alone: `deny` only when
//! the Policy refused the call, with the reason. A Tool Call the Policy allowed
//! but that failed on the host (quota, escape, overwrite decline, malformed
//! arguments, Sandbox setup) stays `allow` and records the host-side error in
//! `error`; a Policy denial carries `denial_reason` and no `error`. A Guest Run
//! always records `allow`, even when its exit code is non-zero: the Guest ran,
//! and its exit code is the recorded result.
//!
//! Each event is one synchronous append followed by a flush, so a crash keeps
//! every completed event and at worst loses the one in flight. A write failure
//! is an error, never a silently dropped event.
//!
//! The log and its quarantine file must be regular files: a symlinked final
//! component is refused rather than followed, even when the link currently
//! resolves outside the Workspace, because following it would let a link
//! redirect host-written audit records into the Workspace. A directory, FIFO,
//! or device at either path is refused too: a FIFO would accept appends with
//! nothing persisted, and the file tools must never block on it.
//!
//! Complete records are never rewritten or deleted, so the log is append-only
//! except at open time. A write that fails halfway leaves part of a line at the
//! end of the file; opening the log repairs that tail before anything is
//! appended: a fragment that is a complete serialized event is terminated, and
//! an incomplete one is moved to `<audit.jsonl>.quarantine`, so the next event
//! always starts on its own line.

use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
/// Field order is the serialized line order. `decision` is the Policy verdict:
/// `deny` for a Policy refusal, `allow` otherwise. `denial_reason` is present
/// only for a Policy refusal. `error` is present only when the Policy allowed
/// the call but it failed on the host. `exit_code` is `null` for the host file
/// tools and for error Observations, matching [`Observation::exit_code`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuditEvent {
    /// RFC 3339 UTC timestamp of the host write.
    pub ts: String,
    /// Id shared by every event written through one Audit Log; generated when
    /// the log is opened.
    pub session: String,
    /// Step (LLM turn) that produced the Tool Call, zero-based.
    pub step: usize,
    /// Tool name exactly as the model sent it, including unknown names.
    pub tool: String,
    /// Lowercase hex SHA-256 of the full raw `function.arguments` string.
    pub args_sha256: String,
    /// Redacted argument prefix, at most [`MAX_ARGS_PREFIX`] bytes.
    pub args_prefix: String,
    /// `allow` or `deny`, from the Policy alone.
    pub decision: &'static str,
    /// Why the Policy denied the call; absent for allowed calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub denial_reason: Option<String>,
    /// Host-side failure of a Policy-allowed call; absent otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Guest exit code, if the Tool Call completed a Run.
    pub exit_code: Option<i32>,
    /// Total stdout bytes the Tool Call produced, before truncation.
    pub stdout_bytes: u64,
    /// Total stderr bytes the Tool Call produced, before truncation.
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
        let allowed = decision.is_allowed();
        Self {
            ts: utc_now(),
            session: session.into(),
            step,
            tool: call.function.name.clone(),
            args_sha256: sha256_hex(&call.function.arguments),
            args_prefix: args_prefix(&call.function.arguments),
            decision: if allowed { "allow" } else { "deny" },
            denial_reason: decision.reason().map(ToOwned::to_owned),
            error: if allowed {
                observation.error.clone()
            } else {
                None
            },
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
    quarantine: PathBuf,
    session: String,
    file: Mutex<File>,
}

impl AuditLog {
    /// Opens `<cwd>/.agent/audit.jsonl`, creating `.agent/` when needed.
    ///
    /// Refuses — rather than writing somewhere the model can read — when that
    /// path would land inside `workspace_root`. The Workspace is canonicalized
    /// first and the requested directory is made absolute lexically, so an
    /// inside-the-Workspace request is refused *before* `create_dir_all` touches
    /// the disk. After creation the directory is canonicalized again, so a
    /// symlink that resolves into the Workspace is caught too. The
    /// `audit.jsonl` component itself must be a regular file: a symlinked log
    /// is refused rather than followed, so it cannot resolve into the
    /// Workspace after the directory check. A refused start leaves the
    /// Workspace exactly as it was.
    pub fn open(cwd: &Path, workspace_root: &Path) -> Result<Self, AuditError> {
        let requested = cwd.join(DIR);
        let dir = std::path::absolute(&requested).map_err(|source| AuditError::Resolve {
            path: requested,
            source,
        })?;
        let workspace = workspace_root
            .canonicalize()
            .map_err(|source| AuditError::Resolve {
                path: workspace_root.to_path_buf(),
                source,
            })?;

        let lexical_path = dir.join(FILE);
        if lexical_path.starts_with(&workspace) {
            return Err(AuditError::InsideWorkspace {
                path: lexical_path,
                workspace,
            });
        }

        let dir_existed = dir.is_dir();
        std::fs::create_dir_all(&dir).map_err(|source| AuditError::Directory {
            path: dir.clone(),
            source,
        })?;
        let created = !dir_existed;
        let dir = dir.canonicalize().map_err(|source| AuditError::Resolve {
            path: dir.clone(),
            source,
        })?;
        let path = dir.join(FILE);
        if path.starts_with(&workspace) {
            // A symlinked ancestor can hide an inside-the-Workspace path from
            // the lexical check. Refuse, and undo the directory this call
            // created so the Workspace is left exactly as it was.
            if created {
                let _ = std::fs::remove_dir(&dir);
            }
            return Err(AuditError::InsideWorkspace { path, workspace });
        }
        Self::at(path)
    }

    /// Opens the log at `path`, appending to any existing file.
    ///
    /// The parent directory must exist; [`AuditLog::open`] creates `.agent/`
    /// and enforces the outside-the-Workspace rule. The final log component and
    /// a pre-existing quarantine file must both be regular files: a symlink is
    /// refused rather than followed, even when it currently resolves outside
    /// the Workspace, and a directory, FIFO, or device is refused too. A
    /// symlinked quarantine is refused even when the log tail is clean, so a
    /// later repair cannot be redirected. An unterminated final record left by
    /// a failed append is repaired before the log is returned, so the next
    /// event cannot merge into the fragment.
    ///
    /// Each handle gets a fresh session id: the events it writes are one
    /// identifiable session even when another Audit Log opens in the same
    /// process. The CLI opens one log per process, so its run stays one
    /// session; the Web Demo opens one log per Task, so each Task gets its own.
    pub fn at(path: impl Into<PathBuf>) -> Result<Self, AuditError> {
        let path = path.into();
        let quarantine = quarantine_path(&path);
        // Validate the quarantine even on a clean log: it only gets opened
        // during a later repair, and by then it is too late to refuse startup.
        refuse_special(&quarantine)?;
        let mut file = open_append_nofollow(&path)?;
        repair_unterminated_tail(&path, &quarantine, &mut file)?;
        Ok(Self {
            quarantine,
            path,
            session: new_session(),
            file: Mutex::new(file),
        })
    }

    /// The file events are appended to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The file where an incomplete final record is preserved when the log is
    /// opened. Complete records stay in the log.
    pub fn quarantine_path(&self) -> &Path {
        &self.quarantine
    }

    /// The id shared by every event this Audit Log writes.
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

/// A fresh id for one Audit Log, shared by every event that log writes.
///
/// Generated per handle, not per process: the CLI opens one log per process,
/// so its events share one id, while the Web Demo opens one log per Task, so
/// each Task gets a new id instead of reusing a session whose Step counter
/// restarted at 0.
fn new_session() -> String {
    format!("{:016x}", rand::random::<u64>())
}

/// Opens `path` for appending after checking it is a regular file.
///
/// A symlink is refused even when it currently resolves outside the Workspace:
/// the Audit Log is a regular file the process creates and owns, and following
/// a link would reopen the path for redirection after the Workspace check. On
/// Unix, `O_NOFOLLOW` closes the window between the metadata check and the
/// open, and `O_NONBLOCK` keeps a FIFO swapped into that window from blocking
/// the open. The opened handle's metadata is checked again, so a non-regular
/// file that replaced the checked path is refused too.
fn open_append_nofollow(path: &Path) -> Result<File, AuditError> {
    refuse_special(path)?;
    let file = append_options()
        .read(true)
        .open(path)
        .map_err(|source| AuditError::Open {
            path: path.to_path_buf(),
            source,
        })?;
    let metadata = file.metadata().map_err(|source| AuditError::Open {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(AuditError::NotRegularFile {
            path: path.to_path_buf(),
        });
    }
    Ok(file)
}

/// Open options shared by the log and its quarantine file: append, create, and
/// never follow a symlinked final component.
fn append_options() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.create(true).append(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    options
}

/// Refuses a path that is a symlink or not a regular file.
///
/// Resolving a link would let a host actor redirect host-written audit records
/// somewhere they are not meant to go, including into the Workspace where the
/// model can read them. Refusing every symlink keeps the rule independent of
/// the link's current target. Directories, FIFOs, and devices are refused for
/// the same reason plus one more: appending to a FIFO persists nothing, and a
/// read would block. A missing path is fine: the log is created on open.
fn refuse_special(path: &Path) -> Result<(), AuditError> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => Err(AuditError::Symlink {
            path: path.to_path_buf(),
        }),
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) => Err(AuditError::NotRegularFile {
            path: path.to_path_buf(),
        }),
        Err(source) if source.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(source) => Err(AuditError::Open {
            path: path.to_path_buf(),
            source,
        }),
    }
}

/// The path where a fragment moved out of `path` is preserved.
fn quarantine_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".quarantine");
    PathBuf::from(name)
}

/// Makes sure a log whose last append failed halfway cannot merge the next
/// event into the fragment.
///
/// A final record that is complete JSON but missing only its newline is
/// terminated and kept. An incomplete fragment is appended to the quarantine
/// file and then cut from the log; complete records are left byte for byte.
fn repair_unterminated_tail(
    path: &Path,
    quarantine: &Path,
    file: &mut File,
) -> Result<(), AuditError> {
    let Some((start, fragment)) = unterminated_tail(file).map_err(|source| AuditError::Repair {
        path: path.to_path_buf(),
        source,
    })?
    else {
        return Ok(());
    };

    // A prefix of a serialized event that parses as a JSON object can only be
    // the whole event missing its newline: any earlier cut would leave an
    // unclosed string, array, or object.
    let complete =
        serde_json::from_slice::<serde_json::Value>(&fragment).is_ok_and(|value| value.is_object());
    if complete {
        return file
            .write_all(b"\n")
            .and_then(|()| file.flush())
            .map_err(|source| AuditError::Repair {
                path: path.to_path_buf(),
                source,
            });
    }

    // The quarantine file holds audit bytes too, so it is subject to the same
    // regular-file rule as the log itself. [`AuditLog::at`] already checked it;
    // this closes the window before the fragment is written.
    refuse_special(quarantine)?;
    let mut quarantined = open_append_nofollow(quarantine).map_err(|error| match error {
        AuditError::Open { path, source } => AuditError::Repair { path, source },
        other => other,
    })?;
    quarantined
        .write_all(&fragment)
        .and_then(|()| quarantined.write_all(b"\n"))
        .and_then(|()| quarantined.flush())
        .map_err(|source| AuditError::Repair {
            path: quarantine.to_path_buf(),
            source,
        })?;
    // Truncate only after the fragment is safely preserved.
    file.set_len(start)
        .and_then(|()| file.flush())
        .map_err(|source| AuditError::Repair {
            path: path.to_path_buf(),
            source,
        })
}

/// The byte offset and bytes of a final record that lacks its newline.
///
/// Returns `None` when the file is empty or already ends with `\n`. The read
/// starts at the last byte and walks backwards only until the previous newline,
/// so a healthy log is not scanned.
fn unterminated_tail(file: &mut File) -> io::Result<Option<(u64, Vec<u8>)>> {
    const CHUNK: u64 = 8 * 1024;

    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(len - 1))?;
    let mut last = [0u8; 1];
    file.read_exact(&mut last)?;
    if last[0] == b'\n' {
        return Ok(None);
    }

    let mut tail = Vec::new();
    let mut end = len;
    loop {
        let start = end.saturating_sub(CHUNK);
        let mut chunk = vec![0u8; (end - start) as usize];
        file.seek(SeekFrom::Start(start))?;
        file.read_exact(&mut chunk)?;
        if let Some(newline) = chunk.iter().rposition(|byte| *byte == b'\n') {
            let mut fragment = chunk[newline + 1..].to_vec();
            fragment.extend_from_slice(&tail);
            return Ok(Some((start + newline as u64 + 1, fragment)));
        }
        chunk.extend_from_slice(&tail);
        tail = chunk;
        if start == 0 {
            return Ok(Some((0, tail)));
        }
        end = start;
    }
}

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
    #[error(
        "audit path `{path}` is a symlink; the Audit Log and its quarantine file must be regular \
         files so they cannot redirect records into the Workspace"
    )]
    Symlink { path: PathBuf },
    #[error(
        "audit path `{path}` is not a regular file; the Audit Log and its quarantine file must be \
         regular files so records are persisted and nothing can block or redirect them"
    )]
    NotRegularFile { path: PathBuf },
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
    #[error("failed to repair an unterminated audit record in `{path}`")]
    Repair {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
}
