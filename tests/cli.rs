//! CLI contract: flags, and refusal to start without a verified Runtime Image.

use std::process::Command;

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
