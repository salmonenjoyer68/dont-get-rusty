//! Real-Guest integration tests for the Sandbox.
//!
//! These run CPython in Wasmtime against the fetched Runtime Image, so
//! `scripts/fetch-assets.sh` must have run first.

use std::sync::LazyLock;

use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::sandbox::{RunResult, Sandbox};

static SANDBOX: LazyLock<Sandbox> = LazyLock::new(|| {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    Sandbox::new(image.root()).expect("failed to create sandbox")
});

fn shared_sandbox() -> &'static Sandbox {
    &SANDBOX
}

fn run(code: &str) -> RunResult {
    shared_sandbox()
        .run(code, None, None)
        .expect("guest run should not fail at the host level")
}

#[test]
fn prints_hello_to_stdout_and_exits_zero() {
    let result = run("print('hello')");

    assert_eq!(result.exit_code, 0);
    assert_eq!(result.stdout, "hello\n");
    assert_eq!(result.stderr, "");
    assert!(!result.timed_out);
}

#[test]
fn propagates_guest_exit_code() {
    let result = run("import sys\nsys.exit(3)");

    assert_eq!(result.exit_code, 3);
    assert_eq!(result.stdout, "");
    assert_eq!(result.stderr, "");
    assert!(!result.timed_out);
}

#[test]
fn maps_uncaught_exception_to_exit_one_with_traceback() {
    let result = run("raise ValueError('boom')");

    assert_eq!(result.exit_code, 1);
    assert!(
        result.stderr.contains("Traceback"),
        "stderr: {}",
        result.stderr
    );
    assert!(
        result.stderr.contains("ValueError"),
        "stderr: {}",
        result.stderr
    );
    assert!(!result.timed_out);
}

#[test]
fn guest_environment_is_empty() {
    let result = run("import os\nprint(len(os.environ))");

    assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
    assert_eq!(result.stdout, "0\n");
    assert_eq!(result.stderr, "");
}

#[test]
fn host_files_are_not_reachable() {
    let result = run("open('/etc/passwd').read()");

    assert_eq!(result.exit_code, 1, "stderr: {}", result.stderr);
    assert!(
        result.stderr.contains("No such file"),
        "stderr: {}",
        result.stderr
    );
}

#[test]
fn guest_receives_piped_stdin() {
    let result = shared_sandbox()
        .run(
            "import sys\nprint(sys.stdin.read(), end='')",
            Some("piped\n"),
            None,
        )
        .expect("guest run should not fail at the host level");

    assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
    assert_eq!(result.stdout, "piped\n");
    assert_eq!(result.stderr, "");
}

#[test]
fn guest_cannot_write_to_the_runtime_image() {
    let result = run("open('/escape.txt', 'w')");

    assert_eq!(result.exit_code, 1, "stderr: {}", result.stderr);
    assert!(
        result.stderr.contains("PermissionError")
            || result.stderr.contains("Operation not permitted"),
        "stderr: {}",
        result.stderr
    );
}
