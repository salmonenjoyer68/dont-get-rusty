//! Consolidated escape suite: the Guest and the host file tools stay inside
//! their Capabilities.
//!
//! The Guest cases run the real CPython Runtime Image in Wasmtime, so
//! `scripts/fetch-assets.sh` must have run first. Every test uses a fresh
//! temporary directory and checks the outside sentinel, so a passing test
//! means nothing outside the allowed roots was read, written, or created.

use std::fs;
use std::path::Path;
use std::sync::LazyLock;

use dont_get_rusty::assets::RuntimeImage;
use dont_get_rusty::fs::{FsError, PathError, Workspace};
use dont_get_rusty::sandbox::{RunResult, Sandbox};

static SANDBOX: LazyLock<Sandbox> = LazyLock::new(|| {
    let image =
        RuntimeImage::discover().expect("runtime image missing: run `scripts/fetch-assets.sh`");
    Sandbox::new(image.root()).expect("failed to create sandbox")
});

fn run(code: &str, workspace: &Path) -> RunResult {
    SANDBOX
        .run(code, None, Some(workspace))
        .expect("guest run should not fail at the host level")
}

fn assert_denied_by_the_guest(result: &RunResult) {
    assert_eq!(result.exit_code, 1, "stderr: {}", result.stderr);
    assert!(
        result.stderr.contains("PermissionError")
            || result.stderr.contains("Operation not permitted")
            || result.stderr.contains("No such file")
            || result.stderr.contains("Read-only file system"),
        "the Guest must be denied, stderr: {}",
        result.stderr
    );
}

// ---- Guest: the preopen set is the sandbox ----

#[test]
fn guest_cannot_read_host_system_files() {
    let dir = tempfile::tempdir().unwrap();

    let result = run("print(open('/etc/passwd').read())", dir.path());

    assert_denied_by_the_guest(&result);
    assert!(
        !result.stdout.contains("root:"),
        "the host passwd file reached the Guest: {}",
        result.stdout
    );
}

#[test]
fn guest_cannot_write_through_the_workspace_preopen() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("ws");
    fs::create_dir(&workspace).unwrap();
    let sentinel = workspace.join("keep.txt");
    fs::write(&sentinel, "keep me\n").unwrap();

    let result = run("open('/workspace/../x', 'w').write('pwned')", &workspace);

    assert_denied_by_the_guest(&result);
    assert!(
        !dir.path().join("x").exists(),
        "`/workspace/../x` escaped the read-only preopen"
    );
    assert_eq!(
        fs::read_to_string(&sentinel).unwrap(),
        "keep me\n",
        "the existing Workspace file is untouched"
    );
}

#[test]
fn guest_cannot_create_files_in_the_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let workspace = dir.path().join("ws");
    fs::create_dir(&workspace).unwrap();
    let inside = workspace.join("keep.txt");
    fs::write(&inside, "keep me\n").unwrap();
    let sentinel = dir.path().join("outside.txt");
    fs::write(&sentinel, "outside sentinel\n").unwrap();

    let result = run("open('/workspace/x', 'w').write('pwned')", &workspace);

    assert_denied_by_the_guest(&result);
    assert!(
        !workspace.join("x").exists(),
        "the Guest created a file in the Workspace"
    );
    assert_eq!(
        fs::read_to_string(&inside).unwrap(),
        "keep me\n",
        "the host-written Workspace file is untouched"
    );
    assert_eq!(
        fs::read_to_string(&sentinel).unwrap(),
        "outside sentinel\n",
        "the outside sentinel is unchanged"
    );
}

// ---- Host file tools: cap-std containment ----

fn workspace_with_sentinel(dir: &Path) -> (Workspace, std::path::PathBuf) {
    let root = dir.join("ws");
    fs::create_dir(&root).unwrap();
    let sentinel = dir.join("sentinel.txt");
    fs::write(&sentinel, "outside sentinel\n").unwrap();
    (Workspace::open(&root).unwrap(), sentinel)
}

#[test]
fn host_write_file_rejects_traversal_before_touching_the_disk() {
    let dir = tempfile::tempdir().unwrap();
    let (workspace, sentinel) = workspace_with_sentinel(dir.path());

    let error = workspace
        .write_file("../sentinel.txt", b"pwned")
        .unwrap_err();
    assert!(matches!(error, FsError::InvalidPath(PathError::ParentDir)));
    assert_eq!(fs::read_to_string(&sentinel).unwrap(), "outside sentinel\n");
}

#[test]
fn host_write_file_rejects_absolute_paths_before_touching_the_disk() {
    let dir = tempfile::tempdir().unwrap();
    let (workspace, sentinel) = workspace_with_sentinel(dir.path());
    let absolute = dir.path().join("absolute.txt");

    let error = workspace.write_file(&absolute, b"pwned").unwrap_err();
    assert!(matches!(error, FsError::InvalidPath(PathError::Absolute)));
    assert!(!absolute.exists(), "an absolute host path was created");
    assert_eq!(fs::read_to_string(&sentinel).unwrap(), "outside sentinel\n");
}

#[cfg(unix)]
#[test]
fn host_write_file_rejects_symlinks_that_escape_the_workspace() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().unwrap();
    let (workspace, sentinel) = workspace_with_sentinel(dir.path());
    let link = dir.path().join("ws").join("escape.txt");
    symlink(&sentinel, &link).unwrap();
    let outside_dir = dir.path().join("outside");
    fs::create_dir(&outside_dir).unwrap();
    symlink(&outside_dir, dir.path().join("ws").join("escape_dir")).unwrap();
    let secret = outside_dir.join("secret.txt");
    fs::write(&secret, "outside secret\n").unwrap();

    for target in ["escape.txt", "escape_dir/secret.txt"] {
        match workspace.write_file(target, b"pwned") {
            Err(FsError::Io(error)) => {
                assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied)
            }
            other => panic!("expected a permission denial for {target}, got {other:?}"),
        }
    }

    assert_eq!(
        fs::read_to_string(&sentinel).unwrap(),
        "outside sentinel\n",
        "the symlink target was rewritten"
    );
    assert_eq!(fs::read_to_string(&secret).unwrap(), "outside secret\n");
    assert!(
        !outside_dir.join("new.txt").exists(),
        "a write through the escaping directory symlink created a file"
    );
}
