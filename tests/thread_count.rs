//! Regression tests for the Sandbox's epoch-ticker thread.
//!
//! Dropping a [`Sandbox`] must stop and join its ticker thread, including when
//! construction fails after the ticker has started. The test observes the
//! process thread count through `/proc/self/task`, so it lives in its own test
//! binary with exactly one test: the count is process-wide, and libtest runs
//! tests in threads within one process, so a concurrent test here would move
//! the baseline. Tests in other binaries are separate processes and cannot
//! interfere.
//!
//! Construction uses a minimal empty wasm module rather than the real Guest:
//! the ticker's lifetime does not depend on the compiled bytes, and compiling
//! the 32 MB CPython module three times would dominate the suite's runtime.

#![cfg(target_os = "linux")]

use std::time::{Duration, Instant};

use dont_get_rusty::sandbox::{Sandbox, SandboxCfg, SandboxError};

/// The smallest valid WebAssembly module: magic and version only.
const EMPTY_WASM: &[u8] = b"\0asm\x01\0\0\0";

/// The number of threads in this process.
fn thread_count() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("/proc/self/task should be readable on Linux")
        .count()
}

/// Waits a bounded time for this process to have exactly `expected` threads.
///
/// `join` returns once the thread has exited, but Linux can still list the
/// exiting task under `/proc/self/task` for a moment: the kernel clears the
/// join futex before removing the task entry. A short poll keeps the assertion
/// deterministic, and a real leak still fails because a thread that was never
/// stopped never goes away.
fn wait_for_threads(expected: usize) -> usize {
    let deadline = Instant::now() + Duration::from_secs(1);
    loop {
        let count = thread_count();
        if count == expected || Instant::now() >= deadline {
            return count;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// A Runtime Image directory holding an empty, valid `python.wasm`.
fn empty_guest_image() -> tempfile::TempDir {
    let image = tempfile::tempdir().unwrap();
    std::fs::write(image.path().join("python.wasm"), EMPTY_WASM).unwrap();
    image
}

#[test]
fn dropping_a_sandbox_stops_its_epoch_ticker() {
    let image = empty_guest_image();
    let baseline = thread_count();

    for _ in 0..3 {
        // No module cache: Wasmtime's cache owns a process-wide worker thread
        // of its own, which would move the count and is unrelated to us.
        let sandbox = Sandbox::new(image.path()).expect("sandbox setup should succeed");
        assert_eq!(
            thread_count(),
            baseline + 1,
            "a live Sandbox should own exactly one epoch-ticker thread"
        );
        drop(sandbox);
        assert_eq!(
            wait_for_threads(baseline),
            baseline,
            "dropping the Sandbox must stop and join its epoch-ticker thread"
        );
    }

    // A construction failure after the ticker starts must not leak it either:
    // an empty directory has no python.wasm, so compiling the Guest module
    // fails after `new_engine` has already spawned the ticker.
    let without_guest = tempfile::tempdir().unwrap();
    match Sandbox::with_cfg(without_guest.path(), SandboxCfg::default()) {
        Err(SandboxError::Wasmtime(_)) => {}
        Err(other) => panic!("expected a module-compilation failure, got {other}"),
        Ok(_) => panic!("an empty Runtime Image must not produce a Sandbox"),
    }
    assert_eq!(
        wait_for_threads(baseline),
        baseline,
        "a failed Sandbox construction must not leak its epoch-ticker thread"
    );
}
