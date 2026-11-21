//! Policy loading: defaults, partial files, explicit paths, and the decision
//! surface the Agent Loop consults. Every test uses a temporary directory.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use dont_get_rusty::agent::AgentConfig;
use dont_get_rusty::policy::{Policy, PolicyError};
use dont_get_rusty::sandbox::SandboxCfg;
use tempfile::TempDir;

fn temp_policy(body: &str) -> (TempDir, PathBuf) {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("policy.toml");
    std::fs::write(&path, body).unwrap();
    (dir, path)
}

/// The working directory is process-global, so every test that resolves
/// `./policy.toml` holds this lock while it runs.
static CWD_LOCK: Mutex<()> = Mutex::new(());

fn lock_cwd() -> MutexGuard<'static, ()> {
    CWD_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Points the working directory at a temporary directory for the rest of the
/// test and restores the previous one on drop.
struct CwdGuard {
    previous: PathBuf,
    _lock: MutexGuard<'static, ()>,
}

impl CwdGuard {
    fn enter(dir: &Path) -> Self {
        let lock = lock_cwd();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(dir).unwrap();
        Self {
            previous,
            _lock: lock,
        }
    }
}

impl Drop for CwdGuard {
    fn drop(&mut self) {
        std::env::set_current_dir(&self.previous).expect("failed to restore the working directory");
    }
}

#[test]
fn defaults_match_the_documented_values() {
    let policy = Policy::default();

    assert_eq!(policy.sandbox.memory_mb, 256);
    assert_eq!(policy.sandbox.timeout_s, 10);
    assert_eq!(policy.sandbox.max_output_kb, 64);
    assert!(!policy.sandbox.network);

    assert_eq!(policy.fs.root, Path::new("./workspace"));
    assert_eq!(policy.fs.max_total_mb, 50);
    assert!(policy.fs.confirm_overwrite);

    assert_eq!(policy.agent.max_steps, 25);
    assert_eq!(policy.agent.max_failed_runs, 3);
    assert_eq!(policy.agent.context_budget_chars, 16_000);

    assert_eq!(policy.llm.base_url, "http://localhost:11434/v1");
    assert_eq!(policy.llm.model, "qwen3.5:4b");
    assert_eq!(policy.llm.request_timeout_s, 120);
}

#[test]
fn the_committed_policy_file_matches_the_defaults() {
    let _lock = lock_cwd();
    let from_file = Policy::from_file(Path::new("policy.toml")).expect("policy.toml should parse");
    assert_eq!(from_file, Policy::default());
}

#[test]
fn a_partial_file_keeps_every_other_default() {
    let (_dir, path) = temp_policy(
        "[agent]\n\
         max_steps = 4\n\
         \n\
         [fs]\n\
         max_total_mb = 2\n",
    );
    let policy = Policy::from_file(&path).unwrap();

    assert_eq!(policy.agent.max_steps, 4);
    assert_eq!(policy.agent.max_failed_runs, 3, "unspecified default");
    assert_eq!(policy.fs.max_total_mb, 2);
    assert_eq!(policy.fs.root, Path::new("./workspace"));
    assert_eq!(policy.sandbox, Policy::default().sandbox);
    assert_eq!(policy.llm, Policy::default().llm);
}

#[test]
fn an_empty_file_uses_every_default() {
    let (_dir, path) = temp_policy("");
    assert_eq!(Policy::from_file(&path).unwrap(), Policy::default());
}

#[test]
fn a_missing_explicit_policy_file_is_an_error_naming_it() {
    let dir = TempDir::new().unwrap();
    let missing = dir.path().join("absent.toml");

    let error = Policy::load(Some(&missing)).unwrap_err();
    assert!(
        matches!(&error, PolicyError::Missing { path } if path == &missing),
        "unexpected error: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains(missing.to_str().unwrap()),
        "message names the file: {message}"
    );
    assert!(
        message.contains("does not exist"),
        "message explains it: {message}"
    );
}

#[test]
fn without_the_flag_the_default_file_is_used_when_present() {
    let _lock = lock_cwd();
    let (policy, source) = Policy::load(None).unwrap();
    assert_eq!(source, Some(PathBuf::from("policy.toml")));
    assert_eq!(policy, Policy::default());
}

#[test]
fn without_the_flag_an_absent_default_file_uses_the_defaults() {
    let dir = TempDir::new().unwrap();
    let _cwd = CwdGuard::enter(dir.path());

    let (policy, source) = Policy::load(None).unwrap();
    assert_eq!(policy, Policy::default());
    assert_eq!(source, None);
}

#[test]
fn without_the_flag_a_directory_named_policy_toml_is_an_error() {
    let dir = TempDir::new().unwrap();
    std::fs::create_dir(dir.path().join("policy.toml")).unwrap();
    let _cwd = CwdGuard::enter(dir.path());

    let error = Policy::load(None).unwrap_err();
    assert!(
        matches!(&error, PolicyError::Read { path, .. } if path == Path::new("policy.toml")),
        "a directory must not select the defaults: {error:?}"
    );
    let message = error.to_string();
    assert!(
        message.contains("policy.toml"),
        "message names the file: {message}"
    );
    assert!(
        message.contains("failed to read"),
        "message explains the failure: {message}"
    );
}

#[test]
fn without_the_flag_an_absent_default_file_falls_back_to_defaults() {
    let dir = TempDir::new().unwrap();
    let absent = dir.path().join("policy.toml");

    let policy = Policy::load_or_default(&absent).unwrap();
    assert_eq!(policy, Policy::default());
}

#[test]
fn load_or_default_reads_a_file_that_exists() {
    let (_dir, path) = temp_policy("[sandbox]\nmemory_mb = 128\n");
    let policy = Policy::load_or_default(&path).unwrap();
    assert_eq!(policy.sandbox.memory_mb, 128);
    assert_eq!(policy.agent.max_steps, 25);
}

#[test]
fn load_or_default_propagates_a_directory_named_policy_toml() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("policy.toml");
    std::fs::create_dir(&path).unwrap();

    let error = Policy::load_or_default(&path).unwrap_err();
    assert!(
        matches!(&error, PolicyError::Read { path: reported, .. } if reported == &path),
        "a directory must not select the defaults: {error:?}"
    );
}

#[test]
fn an_explicit_path_to_a_directory_is_an_error() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("policy.toml");
    std::fs::create_dir(&path).unwrap();

    let error = Policy::load(Some(&path)).unwrap_err();
    assert!(
        matches!(&error, PolicyError::Read { path: reported, .. } if reported == &path),
        "unexpected error: {error:?}"
    );
}

#[cfg(unix)]
#[test]
fn load_or_default_propagates_an_unreadable_file() {
    use std::os::unix::fs::PermissionsExt;

    let (_dir, path) = temp_policy("[agent]\nmax_steps = 4\n");
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o000)).unwrap();

    // Root can read a mode-000 file, so the case under test cannot occur here.
    if std::fs::read_to_string(&path).is_ok() {
        eprintln!("skipping: this process can read mode-000 files");
        return;
    }

    let error = Policy::load_or_default(&path).unwrap_err();
    assert!(
        matches!(&error, PolicyError::Read { path: reported, .. } if reported == &path),
        "an unreadable file must not select the defaults: {error:?}"
    );
    assert!(
        error.to_string().contains(path.to_str().unwrap()),
        "message names the file: {error}"
    );
}

#[cfg(unix)]
#[test]
fn load_or_default_propagates_a_symlink_loop() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("policy.toml");
    std::os::unix::fs::symlink("policy.toml", &path).unwrap();

    let error = Policy::load_or_default(&path).unwrap_err();
    assert!(
        matches!(&error, PolicyError::Read { path: reported, .. } if reported == &path),
        "a symlink that cannot be read must not select the defaults: {error:?}"
    );
}

#[test]
fn network_true_is_rejected_rather_than_ignored() {
    let (_dir, path) = temp_policy("[sandbox]\nnetwork = true\n");
    let error = Policy::from_file(&path).unwrap_err();
    assert!(
        matches!(&error, PolicyError::NetworkUnsupported { path: reported } if reported == &path),
        "unexpected error: {error:?}"
    );
    let message = error.to_string();
    assert!(message.contains("network"), "message: {message}");
    assert!(message.contains("v1"), "message: {message}");
}

#[test]
fn unknown_keys_are_rejected_instead_of_silently_defaulting() {
    let (_dir, path) = temp_policy("[agent]\nmax_step = 4\n");
    let error = Policy::from_file(&path).unwrap_err();
    assert!(
        matches!(&error, PolicyError::Parse { path: reported, .. } if reported == &path),
        "unexpected error: {error:?}"
    );
    assert!(error.to_string().contains("max_step"), "message: {error}");
}

#[test]
fn sections_supply_the_sandbox_workspace_and_agent_wiring() {
    let (_dir, path) = temp_policy(
        "[sandbox]\n\
         memory_mb = 128\n\
         timeout_s = 3\n\
         max_output_kb = 8\n\
         \n\
         [fs]\n\
         root = \"/tmp/example-workspace\"\n\
         max_total_mb = 2\n\
         \n\
         [agent]\n\
         max_steps = 4\n\
         max_failed_runs = 1\n\
         context_budget_chars = 100\n\
         \n\
         [llm]\n\
         base_url = \"http://example.test/v1\"\n\
         model = \"test-model\"\n\
         request_timeout_s = 7\n",
    );
    let policy = Policy::from_file(&path).unwrap();

    assert_eq!(
        policy.sandbox_cfg(),
        SandboxCfg {
            memory_mb: 128,
            timeout: Duration::from_secs(3),
            max_output_kb: 8,
            cache_dir: None,
        }
    );
    assert_eq!(policy.workspace_quota_bytes(), 2 * 1024 * 1024);
    assert_eq!(policy.workspace_root(), Path::new("/tmp/example-workspace"));
    assert_eq!(
        policy.agent_config(),
        AgentConfig {
            max_steps: 4,
            max_failed_runs: 1,
            context_budget_chars: 100,
            base_url: "http://example.test/v1".to_owned(),
            model: "test-model".to_owned(),
            request_timeout: Duration::from_secs(7),
        }
    );
}

#[test]
fn check_allows_workspace_paths_and_denies_traversal() {
    let policy = Policy::default();

    assert!(policy.check("run_code", None).is_allowed());
    assert!(policy.check("read_file", Some("notes/a.txt")).is_allowed());
    assert!(policy.check("write_file", Some("notes/a.txt")).is_allowed());
    assert!(policy.check("list_files", None).is_allowed());

    let deny = |tool: &str, path: Option<&str>| {
        policy
            .check(tool, path)
            .reason()
            .unwrap_or_else(|| panic!("{tool} {path:?} should be denied"))
            .to_owned()
    };
    assert_eq!(
        deny("write_file", Some("../escape.txt")),
        "invalid path: `..` components are not permitted"
    );
    assert_eq!(
        deny("read_file", Some("/etc/passwd")),
        "invalid path: absolute paths are not permitted"
    );
    assert_eq!(deny("read_file", Some("")), "invalid path: path is empty");
    assert_eq!(
        deny("write_file", Some(".")),
        "invalid path: path names the workspace root"
    );
    assert_eq!(deny("delete_file", None), "unknown tool `delete_file`");
}
