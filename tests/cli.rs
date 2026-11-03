//! CLI contract: flags, and refusal to start without a verified Runtime Image.

use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use dont_get_rusty::assets::FETCH_COMMAND;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_dont-get-rusty"))
}

#[test]
fn help_lists_task_and_flags() {
    let output = binary().arg("--help").output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("--workspace"), "help: {stdout}");
    assert!(stdout.contains("--policy"), "help: {stdout}");
    assert!(stdout.contains("--yes"), "help: {stdout}");
    assert!(stdout.contains("task"), "help: {stdout}");
}

#[test]
fn refuses_to_run_without_a_runtime_image() {
    let dir = tempfile::tempdir().unwrap();
    let output = binary()
        .arg("say hello")
        .env("DGR_ASSETS", dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success());
    assert!(stderr.contains(FETCH_COMMAND), "stderr: {stderr}");
}

#[test]
fn a_tampered_runtime_image_is_refused_at_startup() {
    let dir = tempfile::tempdir().unwrap();
    let assets = dir.path().join("assets");
    std::fs::create_dir_all(assets.join("lib/python3.15")).unwrap();
    std::fs::write(assets.join("python.wasm"), b"not the pinned guest").unwrap();
    std::fs::write(
        assets.join("checksums.txt"),
        format!("{}  python.wasm\n", "0".repeat(64)),
    )
    .unwrap();

    let output = binary()
        .arg("say hello")
        .env("DGR_ASSETS", &assets)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        !output.status.success(),
        "a tampered Guest must be refused before it runs: {stderr}"
    );
    assert!(
        stderr.to_lowercase().contains("sha-256") || stderr.to_lowercase().contains("corrupt"),
        "stderr names the hash failure: {stderr}"
    );
    assert!(stderr.contains(FETCH_COMMAND), "stderr: {stderr}");
    assert!(
        !dir.path().join(".agent").exists(),
        "nothing is written before asset verification"
    );
}

#[test]
fn creates_a_missing_workspace_directory() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("nested/workspace");

    // The binary resolves the Workspace before compiling the Sandbox or
    // contacting the LLM; kill it once the directory exists so the test never
    // depends on a running model.
    let mut child = binary()
        .arg("say hello")
        .arg("--workspace")
        .arg(&workspace)
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while !workspace.is_dir() && Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();

    assert!(workspace.is_dir(), "workspace was not created");
}

/// Waits until `workspace` exists, then kills the child so the test never
/// depends on a running model or a compiled Sandbox.
fn create_workspace_then_stop(mut child: std::process::Child, workspace: &std::path::Path) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !workspace.is_dir() && Instant::now() < deadline {
        if child.try_wait().unwrap().is_some() {
            break;
        }
        thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    let _ = child.wait();
}

#[test]
fn a_missing_policy_file_fails_with_a_clear_message() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("absent-policy.toml");
    let output = binary()
        .arg("say hello")
        .arg("--policy")
        .arg(&missing)
        .env("DGR_ASSETS", dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success());
    assert!(
        stderr.contains(missing.to_str().unwrap()),
        "stderr names the file: {stderr}"
    );
    assert!(
        stderr.contains("does not exist"),
        "stderr explains it: {stderr}"
    );
}

#[test]
fn the_workspace_flag_overrides_the_policy_root() {
    let dir = tempfile::tempdir().unwrap();
    let policy = dir.path().join("policy.toml");
    let policy_root = dir.path().join("policy-workspace");
    let flag_root = dir.path().join("flag-workspace");
    std::fs::write(
        &policy,
        format!("[fs]\nroot = \"{}\"\n", policy_root.display()),
    )
    .unwrap();

    let child = binary()
        .arg("say hello")
        .arg("--policy")
        .arg(&policy)
        .arg("--workspace")
        .arg(&flag_root)
        .spawn()
        .unwrap();
    create_workspace_then_stop(child, &flag_root);

    assert!(flag_root.is_dir(), "the flag directory was created");
    assert!(
        !policy_root.exists(),
        "the policy root is not created when --workspace is given"
    );
}

#[test]
fn a_policy_file_inside_the_workspace_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("workspace");
    std::fs::create_dir(&workspace).unwrap();
    let policy = workspace.join("policy.toml");
    std::fs::write(&policy, "").unwrap();

    let output = binary()
        .arg("say hello")
        .arg("--policy")
        .arg(&policy)
        .arg("--workspace")
        .arg(&workspace)
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success());
    assert!(stderr.contains("inside the Workspace"), "stderr: {stderr}");
}

/// The Audit Log is written under the working directory, so using the working
/// directory as the Workspace leaves it nowhere safe to go.
#[test]
fn an_audit_log_inside_the_workspace_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let output = binary()
        .arg("say hello")
        .arg("--workspace")
        .arg(".")
        .current_dir(dir.path())
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(!output.status.success());
    assert!(stderr.contains("Audit Log"), "stderr: {stderr}");
    assert!(stderr.contains("inside the Workspace"), "stderr: {stderr}");
    assert!(
        !dir.path().join(".agent").join("audit.jsonl").exists(),
        "no event is written inside the Workspace"
    );
}
