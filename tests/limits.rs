//! Run-limit and termination-mapping tests against the real CPython Guest.
//!
//! `scripts/fetch-assets.sh` must have run first. Most tests share one Sandbox
//! with a 1 s deadline so the infinite-loop case stays fast; the production
//! default of 10 s is asserted without waiting for it.

use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::sandbox::{RunResult, Sandbox, SandboxCfg};

/// A short deadline keeps the timeout test quick without changing any other
/// limit.
const FAST_TIMEOUT: Duration = Duration::from_secs(1);

/// Guest Runs share the host's CPU and memory; running one at a time keeps the
/// suite fast and the short deadline meaningful even under `cargo test`'s
/// parallel harness.
static RUN_LOCK: Mutex<()> = Mutex::new(());

static FAST_SANDBOX: LazyLock<Sandbox> = LazyLock::new(|| {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    Sandbox::with_cfg(
        image.root(),
        SandboxCfg {
            timeout: FAST_TIMEOUT,
            ..SandboxCfg::default()
        },
    )
    .expect("failed to create sandbox")
});

fn run(code: &str) -> RunResult {
    let _guard = RUN_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    FAST_SANDBOX
        .run(code, None, None)
        .expect("guest run should not fail at the host level")
}

fn last_stderr_line(result: &RunResult) -> Option<&str> {
    result.stderr.lines().last()
}

#[test]
fn default_limits_are_256_mb_10s_and_64_kib_per_stream() {
    let cfg = SandboxCfg::default();

    assert_eq!(cfg.memory_mb, 256);
    assert_eq!(cfg.timeout, Duration::from_secs(10));
    assert_eq!(cfg.max_output_kb, 64);
}

#[test]
fn infinite_loop_traps_at_the_epoch_deadline() {
    let result = run("print('partial')\nwhile True:\n    pass");

    assert!(result.timed_out, "stderr: {}", result.stderr);
    assert_eq!(result.exit_code, 124);
    assert_eq!(result.stdout, "partial\n");
    assert_eq!(
        last_stderr_line(&result),
        Some("sandbox: timed out after 1s")
    );
    assert!(
        !result.stderr.contains("fuel"),
        "epoch deadline, not fuel: {}",
        result.stderr
    );
    assert!(
        result.duration_ms >= 800,
        "deadline fired too early: {} ms",
        result.duration_ms
    );
}

#[test]
fn memory_bomb_traps_at_the_memory_limit() {
    let result = run("print('allocating', end='')\nbytearray(300 * 1024 * 1024)");

    assert_eq!(result.exit_code, 137, "stderr: {}", result.stderr);
    assert!(!result.timed_out);
    assert_eq!(result.stdout, "allocating");
    assert_eq!(
        last_stderr_line(&result),
        Some("sandbox: memory limit exceeded (256 MB)")
    );
    assert!(result.duration_ms > 0);
}

#[test]
fn output_flood_is_truncated_per_stream() {
    let result = run(concat!(
        "import sys\n",
        "chunk = 'x' * 100_000\n",
        "for _ in range(200):\n",
        "    sys.stdout.write(chunk)\n",
        "    sys.stderr.write(chunk)",
    ));

    assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
    assert!(!result.timed_out);

    // 200 * 100_000 bytes per stream; the first 64 KiB is retained.
    let discarded = 20_000_000 - 64 * 1024;
    let marker = format!("[sandbox: output truncated; {discarded} bytes discarded]");
    assert_eq!(
        result.stdout,
        format!("{}\n{marker}", "x".repeat(64 * 1024))
    );
    assert_eq!(
        result.stderr,
        format!("{}\n{marker}", "x".repeat(64 * 1024))
    );
    assert!(result.stdout.len() < 70 * 1024);
    assert!(result.stderr.len() < 70 * 1024);
}

#[test]
fn guest_proc_exit_code_is_propagated() {
    let result = run("import sys\nprint('before', end='')\nsys.exit(3)");

    assert_eq!(result.exit_code, 3);
    assert!(!result.timed_out);
    assert_eq!(result.stdout, "before");
    assert_eq!(result.stderr, "");
    assert!(result.duration_ms > 0);
}

#[test]
fn each_run_starts_from_a_fresh_store() {
    let first = run("import builtins\nbuiltins.leaked = 'yes'\nprint('set')");
    let second = run("import builtins\nprint(hasattr(builtins, 'leaked'))");

    assert_eq!(first.exit_code, 0, "stderr: {}", first.stderr);
    assert_eq!(first.stdout, "set\n");
    assert_eq!(second.exit_code, 0, "stderr: {}", second.stderr);
    assert_eq!(second.stdout, "False\n");
}

#[test]
fn duration_is_populated_and_partial_stdout_survives_traps() {
    let normal = run("print('ok')");
    assert_eq!(normal.exit_code, 0);
    assert_eq!(normal.stdout, "ok\n");
    assert!(normal.duration_ms > 0);

    let uncaught = run("print('before', end='')\nraise ValueError('boom')");
    assert_eq!(
        uncaught.exit_code, 1,
        "duration={}ms stderr={}",
        uncaught.duration_ms, uncaught.stderr
    );
    assert_eq!(uncaught.stdout, "before");
    assert!(!uncaught.timed_out);
    assert!(uncaught.duration_ms > 0);
}

#[test]
fn arbitrary_wasm_traps_stay_exit_one() {
    // `os.abort()` raises a wasm trap rather than a WASI proc_exit.
    let result = run("import os\nprint('before', end='')\nos.abort()");

    assert_eq!(result.exit_code, 1, "stderr: {}", result.stderr);
    assert!(!result.timed_out);
    assert_eq!(result.stdout, "before");
    assert!(
        result
            .stderr
            .lines()
            .last()
            .is_some_and(|line| line.starts_with("sandbox: trap: ")),
        "stderr: {}",
        result.stderr
    );
    assert!(result.duration_ms > 0);
}

#[test]
fn compute_heavy_guest_runs_to_completion_without_fuel_metering() {
    let result = run("total = 0\nfor i in range(2_000_000):\n    total += i\nprint(total)");

    assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
    assert!(!result.timed_out);
    assert_eq!(result.stdout, "1999999000000\n");
}
