//! Policy loading: defaults, partial files, explicit paths, and the decision
//! surface the Agent Loop consults. Every test uses a temporary directory.

use std::path::{Path, PathBuf};
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
    let (policy, source) = Policy::load(None).unwrap();
    assert_eq!(source, Some(PathBuf::from("policy.toml")));
    assert_eq!(policy, Policy::default());
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
