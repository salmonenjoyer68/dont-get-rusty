//! Host-side Policy loaded from `policy.toml`: the limits every Run and every
//! file operation obeys.
//!
//! The model can neither read nor modify a Policy. The file lives outside the
//! Workspace (the CLI refuses a Policy file inside it) and is parsed only by
//! the host; a Tool Call reaches disk through [`crate::fs::Workspace`], which
//! roots every path at the Workspace and enforces the quota.
//!
//! Every field has a documented default, so a partial file — or no file at all
//! — is valid; a missing file named with `--policy` is an error. Unknown keys
//! are rejected so a typo cannot silently fall back to a weaker default.
//!
//! `[sandbox] network = true` is rejected: v1 has no Guest network at all (the
//! WASI socket stubs are nonfunctional and no preopen carries a socket), so
//! accepting the key would promise a capability that does not exist.

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::Deserialize;

use crate::agent::AgentConfig;
use crate::fs::{FsError, PathError, resolve_rel};
use crate::sandbox::SandboxCfg;

/// The Policy file consulted when `--policy` is not given.
pub const DEFAULT_FILE: &str = "policy.toml";

/// The complete Policy.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct Policy {
    pub sandbox: SandboxPolicy,
    pub fs: FsPolicy,
    pub agent: AgentPolicy,
    pub llm: LlmPolicy,
}

impl Policy {
    /// Loads the Policy selected by `explicit`, returning it together with the
    /// file it came from (`None` when the built-in defaults were used).
    ///
    /// `Some(path)` must exist. Without `explicit`, `./policy.toml` is used
    /// when present, so running from the repository root needs no flags.
    pub fn load(explicit: Option<&Path>) -> Result<(Self, Option<PathBuf>), PolicyError> {
        match explicit {
            Some(path) => Ok((Self::from_file(path)?, Some(path.to_path_buf()))),
            None => {
                let default = Path::new(DEFAULT_FILE);
                if default.is_file() {
                    Ok((Self::from_file(default)?, Some(default.to_path_buf())))
                } else {
                    Ok((Self::default(), None))
                }
            }
        }
    }

    /// Loads `path` when it exists, otherwise the built-in defaults.
    ///
    /// A file that exists but cannot be read or parsed is still an error; only
    /// absence falls back.
    pub fn load_or_default(path: &Path) -> Result<Self, PolicyError> {
        if path.is_file() {
            Self::from_file(path)
        } else {
            Ok(Self::default())
        }
    }

    /// Parses `path`; a missing file is reported as [`PolicyError::Missing`].
    pub fn from_file(path: &Path) -> Result<Self, PolicyError> {
        let text = std::fs::read_to_string(path).map_err(|source| match source.kind() {
            io::ErrorKind::NotFound => PolicyError::Missing {
                path: path.to_path_buf(),
            },
            _ => PolicyError::Read {
                path: path.to_path_buf(),
                source,
            },
        })?;
        let policy: Self = toml::from_str(&text).map_err(|source| PolicyError::Parse {
            path: path.to_path_buf(),
            source,
        })?;
        if policy.sandbox.network {
            // v1 grants no Guest network path at all; refuse rather than
            // pretend the request took effect.
            return Err(PolicyError::NetworkUnsupported {
                path: path.to_path_buf(),
            });
        }
        Ok(policy)
    }

    /// Decides whether a Tool Call may execute.
    ///
    /// `path` is the model-supplied path for the file tools and `None` for
    /// `run_code` or a Workspace-root `list_files`. A denial never aborts the
    /// Agent Loop: the reason becomes an error Observation, and ticket #7's
    /// Audit Log records the [`Decision`] alongside the Tool Call.
    ///
    /// The path rules mirror [`resolve_rel`]; `cap-std` remains the security
    /// boundary, so this denies early and explainably rather than substituting
    /// for the containment check at the actual open.
    pub fn check(&self, tool: &str, path: Option<&str>) -> Decision {
        match (tool, path) {
            ("run_code", _) => Decision::Allow,
            ("list_files", None) => Decision::Allow,
            ("read_file" | "write_file" | "list_files", Some(path)) => {
                match resolve_rel(Path::new(path)) {
                    Ok(rel)
                        if rel.as_os_str().is_empty()
                            && (tool == "read_file" || tool == "write_file") =>
                    {
                        Decision::deny(FsError::InvalidPath(PathError::Root).to_string())
                    }
                    Ok(_) => Decision::Allow,
                    Err(error) => Decision::deny(error.to_string()),
                }
            }
            ("read_file" | "write_file", None) => {
                Decision::deny(format!("tool `{tool}` requires a path"))
            }
            (other, _) => Decision::deny(format!("unknown tool `{other}`")),
        }
    }

    /// The Workspace root named by `[fs].root`.
    pub fn workspace_root(&self) -> &Path {
        &self.fs.root
    }

    /// The Workspace quota in bytes, from `[fs].max_total_mb`.
    pub fn workspace_quota_bytes(&self) -> u64 {
        self.fs.max_total_mb.saturating_mul(1024 * 1024)
    }

    /// Per-Run limits for the Sandbox, from `[sandbox]`.
    pub fn sandbox_cfg(&self) -> SandboxCfg {
        SandboxCfg {
            memory_mb: self.sandbox.memory_mb,
            timeout: Duration::from_secs(self.sandbox.timeout_s),
            max_output_kb: self.sandbox.max_output_kb,
        }
    }

    /// Agent Loop and LLM client settings, from `[agent]` and `[llm]`.
    pub fn agent_config(&self) -> AgentConfig {
        AgentConfig {
            max_steps: self.agent.max_steps,
            max_failed_runs: self.agent.max_failed_runs,
            context_budget_chars: self.agent.context_budget_chars,
            base_url: self.llm.base_url.clone(),
            model: self.llm.model.clone(),
            request_timeout: Duration::from_secs(self.llm.request_timeout_s),
        }
    }
}

/// `[sandbox]`: per-Run limits for the Guest.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SandboxPolicy {
    /// Per-linear-memory cap in mebibytes. Default: 256.
    pub memory_mb: usize,
    /// Wall-clock deadline for one Run. Default: 10.
    pub timeout_s: u64,
    /// Retained bytes per output stream, in kibibytes. Default: 64.
    pub max_output_kb: usize,
    /// Reserved for a future proxied-network mode. v1 has no Guest network and
    /// rejects `true` at load time. Default: false.
    pub network: bool,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            memory_mb: 256,
            timeout_s: 10,
            max_output_kb: 64,
            network: false,
        }
    }
}

/// `[fs]`: the Workspace the model may read and write.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FsPolicy {
    /// The Workspace root, relative to the working directory unless absolute.
    /// `--workspace` overrides it. Default: `./workspace`.
    pub root: PathBuf,
    /// Total size of the Workspace's regular files, in mebibytes. Default: 50.
    pub max_total_mb: u64,
    /// Whether `write_file` asks before overwriting a file that existed
    /// before the session. Files the agent created earlier in the session are
    /// always silent; `--yes` disables the prompt. Default: true.
    pub confirm_overwrite: bool,
}

impl Default for FsPolicy {
    fn default() -> Self {
        Self {
            root: PathBuf::from("./workspace"),
            max_total_mb: 50,
            confirm_overwrite: true,
        }
    }
}

/// `[agent]`: Agent Loop limits.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentPolicy {
    /// LLM turns before the loop stops. Default: 25.
    pub max_steps: usize,
    /// Consecutive failed Runs that pause the loop. Default: 3.
    pub max_failed_runs: usize,
    /// Character budget for the conversation before compaction. Default: 16000.
    pub context_budget_chars: usize,
}

impl Default for AgentPolicy {
    fn default() -> Self {
        Self {
            max_steps: 25,
            max_failed_runs: 3,
            context_budget_chars: 16_000,
        }
    }
}

/// `[llm]`: the OpenAI-compatible endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmPolicy {
    /// API root, including the version segment.
    /// Default: `http://localhost:11434/v1`.
    pub base_url: String,
    /// Model name sent in every request. Default: `qwen3.5:4b`.
    pub model: String,
    /// Deadline for one HTTP request. Default: 120.
    pub request_timeout_s: u64,
}

impl Default for LlmPolicy {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434/v1".to_owned(),
            model: "qwen3.5:4b".to_owned(),
            request_timeout_s: 120,
        }
    }
}

/// A Policy's verdict on one Tool Call, recorded by the Audit Log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// The Tool Call may execute.
    Allow,
    /// The Tool Call is refused before execution; `reason` reaches the model as
    /// an error Observation.
    Deny { reason: String },
}

impl Decision {
    /// A refusal carrying `reason`.
    pub fn deny(reason: impl Into<String>) -> Self {
        Self::Deny {
            reason: reason.into(),
        }
    }

    /// Whether the Tool Call may execute.
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allow)
    }

    /// The denial reason, or `None` when allowed.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Self::Allow => None,
            Self::Deny { reason } => Some(reason),
        }
    }
}

/// A Policy file that could not be used.
#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("policy file `{path}` does not exist")]
    Missing { path: PathBuf },
    #[error("failed to read policy file `{path}`")]
    Read {
        path: PathBuf,
        #[source]
        source: io::Error,
    },
    #[error("invalid policy file `{path}`: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
    #[error(
        "policy file `{path}`: [sandbox] network = true is not supported in v1, which has no \
         Guest network; set network = false"
    )]
    NetworkUnsupported { path: PathBuf },
}
