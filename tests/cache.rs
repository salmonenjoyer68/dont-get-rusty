//! Compiled-module cache tests.
//!
//! The spec caches the compiled Guest module under `.agent/cache/`, keyed by
//! the wasm bytes and the Wasmtime version, so a second process start does not
//! pay the full compile cost. These tests drive the real Runtime Image, so
//! `scripts/fetch-assets.sh` must have run first.

use std::fs;
use std::path::{Path, PathBuf};

use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::sandbox::{Lang, RunResult, Sandbox, SandboxCfg, SandboxError};

fn runtime_image() -> RuntimeImage {
    RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`")
}

fn cached_cfg(cache_dir: &Path) -> SandboxCfg {
    SandboxCfg {
        cache_dir: Some(cache_dir.to_path_buf()),
        ..SandboxCfg::default()
    }
}

/// Every regular file under `root`, recursively.
fn files(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("cache directory should be readable") {
            let entry = entry.unwrap();
            let file_type = entry.file_type().unwrap();
            if file_type.is_dir() {
                stack.push(entry.path());
            } else if file_type.is_file() {
                found.push(entry.path());
            }
        }
    }
    found
}

#[test]
fn the_compiled_module_is_cached_and_a_second_sandbox_reuses_it() {
    let dir = tempfile::tempdir().unwrap();
    // The parent does not exist: the Sandbox creates the cache directory.
    let cache = dir.path().join(".agent/cache");
    let image = runtime_image();

    let first = Sandbox::with_cfg(image.root(), cached_cfg(&cache))
        .expect("a missing cache directory is created, not an error");
    let result: RunResult = first
        .run(Lang::Python, "print('cached')", None, None)
        .expect("guest run should not fail at the host level");
    assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
    assert_eq!(result.stdout, "cached\n");
    assert!(cache.is_dir(), "the cache directory was created");

    // Wasmtime lays the cache out as modules/<compiler>-<version>/<hash>;
    // the compiled artifact is the large file there.
    let cached = files(&cache);
    let artifact = cached
        .iter()
        .find(|path| {
            path.ancestors().any(|ancestor| {
                ancestor
                    .file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("wasmtime-"))
            }) && fs::metadata(path).is_ok_and(|metadata| metadata.len() > 1024 * 1024)
        })
        .unwrap_or_else(|| {
            panic!(
                "no compiled artifact under `{}`: {cached:?}",
                cache.display()
            )
        });
    let before = fs::metadata(artifact).unwrap();

    // A second Sandbox on the same cache directory starts from the cached
    // artifact: the Guest runs and the artifact is not rewritten.
    let second = Sandbox::with_cfg(image.root(), cached_cfg(&cache))
        .expect("the cached artifact should load");
    let result = second
        .run(Lang::Python, "print('reused')", None, None)
        .expect("guest run should not fail at the host level");
    assert_eq!(result.exit_code, 0, "stderr: {}", result.stderr);
    assert_eq!(result.stdout, "reused\n");

    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;

        let after = fs::metadata(artifact).unwrap();
        assert_eq!(
            before.ino(),
            after.ino(),
            "a cache hit must not rewrite the compiled artifact"
        );
        assert_eq!(
            before.len(),
            after.len(),
            "a cache hit must not rewrite the compiled artifact"
        );
        assert_eq!(
            before.mtime_nsec(),
            after.mtime_nsec(),
            "a cache hit must not rewrite the compiled artifact"
        );
    }
}

#[test]
fn an_unusable_cache_directory_is_a_clear_setup_error() {
    let dir = tempfile::tempdir().unwrap();
    // A regular file where the cache directory should be: creating the
    // directory must fail, and the failure must be reported, not panic.
    let blocker = dir.path().join("not-a-directory");
    fs::write(&blocker, "not a directory").unwrap();
    let cache = blocker.join("cache");
    let image = runtime_image();

    let result = Sandbox::with_cfg(image.root(), cached_cfg(&cache));
    let Err(error) = result else {
        panic!("an unusable cache directory must not be silently accepted");
    };
    match &error {
        SandboxError::Cache { path, .. } => {
            assert_eq!(path, &cache, "the error names the cache path");
            assert!(
                error.to_string().contains(&cache.display().to_string()),
                "the error message names the path: {error}"
            );
        }
        other => panic!("expected SandboxError::Cache, got {other}"),
    }
}
