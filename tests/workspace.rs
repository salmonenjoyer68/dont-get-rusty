//! Host-side Workspace FS: read/write/list, path rejection, quota, and escapes.
//!
//! Every test uses a fresh temporary directory; nothing outside it is touched.

use std::fs;
use std::path::Path;

use dont_get_rusty::fs::{
    DEFAULT_QUOTA_BYTES, Entry, EntryKind, FsError, PathError, Workspace, format_listing,
    resolve_rel,
};
use proptest::prelude::*;
use tempfile::TempDir;

fn open(root: &Path, quota: u64) -> Workspace {
    Workspace::open_with_quota(root, quota).expect("workspace should open")
}

fn expect_invalid<T: std::fmt::Debug>(result: Result<T, FsError>, expected: PathError) {
    match result {
        Err(FsError::InvalidPath(actual)) => assert_eq!(actual, expected),
        other => panic!("expected InvalidPath({expected:?}), got {other:?}"),
    }
}

fn expect_denied<T: std::fmt::Debug>(result: Result<T, FsError>) {
    match result {
        Err(FsError::Io(error)) => {
            assert_eq!(error.kind(), std::io::ErrorKind::PermissionDenied)
        }
        other => panic!("expected PermissionDenied, got {other:?}"),
    }
}

#[test]
fn default_quota_is_fifty_megabytes() {
    let dir = TempDir::new().unwrap();
    let ws = Workspace::open(dir.path()).unwrap();
    assert_eq!(ws.quota(), DEFAULT_QUOTA_BYTES);
    assert_eq!(DEFAULT_QUOTA_BYTES, 50 * 1024 * 1024);
}

#[test]
fn reads_writes_and_lists_through_the_workspace_handle() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), DEFAULT_QUOTA_BYTES);

    let written = ws.write_file("notes/a.txt", b"hello").unwrap();
    assert_eq!(
        written,
        dont_get_rusty::fs::WriteInfo {
            existed: false,
            bytes_written: 5
        }
    );
    assert_eq!(ws.read_file("notes/a.txt").unwrap(), b"hello");
    assert!(dir.path().join("notes/a.txt").is_file());

    let overwritten = ws.write_file("notes/a.txt", b"hello, world").unwrap();
    assert_eq!(
        overwritten,
        dont_get_rusty::fs::WriteInfo {
            existed: true,
            bytes_written: 12
        }
    );
    assert_eq!(ws.read_file("notes/a.txt").unwrap(), b"hello, world");

    let entries = ws.list_files(Some(Path::new("notes"))).unwrap();
    assert_eq!(
        entries,
        vec![Entry {
            name: "a.txt".into(),
            kind: EntryKind::File,
            size: 12,
        }]
    );
    let root_entries = ws.list_files(None).unwrap();
    assert_eq!(
        root_entries,
        vec![Entry {
            name: "notes".into(),
            kind: EntryKind::Dir,
            size: 0,
        }]
    );
    assert_eq!(ws.list_files(Some(Path::new("."))).unwrap(), root_entries);
}

#[test]
fn listing_format_is_stable_and_model_readable() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), DEFAULT_QUOTA_BYTES);
    ws.write_file("b.txt", b"12345").unwrap();
    ws.write_file("a.txt", b"").unwrap();
    fs::create_dir(dir.path().join("sub")).unwrap();

    let listing = format_listing(&ws.list_files(None).unwrap());
    assert_eq!(listing, "file\t0\ta.txt\nfile\t5\tb.txt\ndir\t-\tsub\n");

    let empty = TempDir::new().unwrap();
    let empty_ws = open(empty.path(), DEFAULT_QUOTA_BYTES);
    assert_eq!(
        format_listing(&empty_ws.list_files(None).unwrap()),
        "(empty directory)\n"
    );
}

#[test]
fn empty_and_root_paths_are_rejected_but_root_is_listable() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), DEFAULT_QUOTA_BYTES);

    expect_invalid(ws.read_file(""), PathError::Empty);
    expect_invalid(ws.write_file("", b"x"), PathError::Empty);
    expect_invalid(ws.read_file("."), PathError::Root);
    expect_invalid(ws.write_file(".", b"x"), PathError::Root);

    assert!(ws.list_files(None).unwrap().is_empty());
    assert!(ws.list_files(Some(Path::new("."))).unwrap().is_empty());
    assert!(!dir.path().join("x").exists());
}

#[test]
fn every_parent_component_and_absolute_path_is_rejected() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), DEFAULT_QUOTA_BYTES);

    for path in ["..", "../x", "a/../b", "a/b/../../../x", "a//.."] {
        expect_invalid(ws.read_file(path), PathError::ParentDir);
        expect_invalid(ws.write_file(path, b"nope"), PathError::ParentDir);
        expect_invalid(ws.list_files(Some(Path::new(path))), PathError::ParentDir);
    }
    for path in ["/etc/passwd", "/workspace/x"] {
        expect_invalid(ws.read_file(path), PathError::Absolute);
        expect_invalid(ws.write_file(path, b"nope"), PathError::Absolute);
    }

    // Nothing was created anywhere in or beside the workspace.
    assert!(fs::read_dir(dir.path()).unwrap().next().is_none());
}

#[test]
fn quota_is_checked_before_any_directory_or_file_is_created() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), 10);

    ws.write_file("a.txt", b"123456").unwrap();
    assert_eq!(ws.total_size().unwrap(), 6);

    match ws.write_file("nested/deep/b.txt", b"123456") {
        Err(FsError::QuotaExceeded { projected, limit }) => {
            assert_eq!(projected, 12);
            assert_eq!(limit, 10);
        }
        other => panic!("expected QuotaExceeded, got {other:?}"),
    }
    assert!(!dir.path().join("nested").exists());
    assert!(!dir.path().join("b.txt").exists());
    assert_eq!(ws.total_size().unwrap(), 6);
}

#[test]
fn denied_overwrite_leaves_the_old_file_intact() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), 10);
    ws.write_file("keep.txt", b"original").unwrap();

    match ws.write_file("keep.txt", b"this is too large") {
        Err(FsError::QuotaExceeded { limit, .. }) => assert_eq!(limit, 10),
        other => panic!("expected QuotaExceeded, got {other:?}"),
    }
    assert_eq!(ws.read_file("keep.txt").unwrap(), b"original");
    assert_eq!(ws.total_size().unwrap(), 8);

    // Shrinking overwrite is allowed and lowers the total.
    let info = ws.write_file("keep.txt", b"ok").unwrap();
    assert!(info.existed);
    assert_eq!(ws.read_file("keep.txt").unwrap(), b"ok");
    assert_eq!(ws.total_size().unwrap(), 2);
}

#[test]
fn a_write_that_fits_after_replacing_the_target_is_allowed() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), 10);
    ws.write_file("a.txt", b"1234567").unwrap();
    ws.write_file("b.txt", b"123").unwrap();

    let info = ws.write_file("a.txt", b"12345").unwrap();
    assert!(info.existed);
    assert_eq!(info.bytes_written, 5);
    assert_eq!(ws.total_size().unwrap(), 8);
}

#[test]
fn non_file_targets_are_rejected_without_touching_them() {
    let dir = TempDir::new().unwrap();
    let ws = open(dir.path(), DEFAULT_QUOTA_BYTES);
    fs::create_dir(dir.path().join("sub")).unwrap();

    match ws.write_file("sub", b"x") {
        Err(FsError::NotAFile) => {}
        other => panic!("expected NotAFile, got {other:?}"),
    }
    assert!(dir.path().join("sub").is_dir());
    assert!(!dir.path().join("sub/x").exists());
}

#[test]
fn resolve_rel_drops_curdir_and_returns_a_relative_path() {
    assert_eq!(resolve_rel(Path::new("./a/./b")).unwrap(), Path::new("a/b"));
    assert_eq!(resolve_rel(Path::new(".")).unwrap(), Path::new(""));
    assert_eq!(
        resolve_rel(Path::new("a/..")).unwrap_err().to_string(),
        "invalid path: `..` components are not permitted"
    );
}

#[cfg(unix)]
mod symlinks {
    use super::*;
    use std::os::unix::fs::symlink;

    #[test]
    fn symlinks_to_outside_are_denied_and_touch_nothing() {
        let dir = TempDir::new().unwrap();
        let ws_root = dir.path().join("ws");
        let outside = dir.path().join("outside");
        fs::create_dir(&ws_root).unwrap();
        fs::create_dir(&outside).unwrap();
        fs::write(outside.join("secret.txt"), b"secret").unwrap();
        fs::write(dir.path().join("sentinel.txt"), b"sentinel").unwrap();

        let ws = open(&ws_root, DEFAULT_QUOTA_BYTES);
        symlink(dir.path().join("sentinel.txt"), ws_root.join("file_escape")).unwrap();
        symlink(&outside, ws_root.join("dir_escape")).unwrap();
        symlink("/etc/passwd", ws_root.join("abs_escape")).unwrap();
        symlink("loop", ws_root.join("loop")).unwrap();

        expect_denied(ws.read_file("file_escape"));
        expect_denied(ws.read_file("abs_escape"));
        expect_denied(ws.read_file("dir_escape/secret.txt"));
        expect_denied(ws.write_file("file_escape", b"pwned"));
        expect_denied(ws.write_file("abs_escape", b"pwned"));
        expect_denied(ws.write_file("dir_escape/new.txt", b"pwned"));
        expect_denied(ws.list_files(Some(Path::new("dir_escape"))));
        match ws.read_file("loop") {
            Err(FsError::Io(error)) => {
                let message = error.to_string().to_lowercase();
                assert!(
                    message.contains("too many levels") || message.contains("symbolic link"),
                    "unexpected symlink loop error: {error}"
                );
            }
            other => panic!("expected a symlink loop error, got {other:?}"),
        }

        assert_eq!(
            fs::read(dir.path().join("sentinel.txt")).unwrap(),
            b"sentinel"
        );
        assert_eq!(fs::read(outside.join("secret.txt")).unwrap(), b"secret");
        let outside_entries: Vec<_> = fs::read_dir(&outside)
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(
            outside_entries,
            vec![std::ffi::OsString::from("secret.txt")]
        );
    }

    #[test]
    fn in_workspace_symlinks_and_symlink_entries_are_reported_without_following() {
        let dir = TempDir::new().unwrap();
        let ws = open(dir.path(), DEFAULT_QUOTA_BYTES);
        ws.write_file("target.txt", b"hello").unwrap();
        symlink("target.txt", dir.path().join("link.txt")).unwrap();

        assert_eq!(ws.read_file("link.txt").unwrap(), b"hello");

        let entries = ws.list_files(None).unwrap();
        assert_eq!(
            entries,
            vec![
                Entry {
                    name: "link.txt".into(),
                    kind: EntryKind::Symlink,
                    size: 0,
                },
                Entry {
                    name: "target.txt".into(),
                    kind: EntryKind::File,
                    size: 5,
                },
            ]
        );
        // Symlinks do not add to the total.
        assert_eq!(ws.total_size().unwrap(), 5);
    }
}

#[cfg(unix)]
fn assert_temp_root_has_only_workspace(root: &Path) {
    let mut names: Vec<_> = fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    assert_eq!(names, vec!["sentinel.txt".to_owned(), "ws".to_owned()]);
}

fn hostile_component() -> impl Strategy<Value = String> {
    prop_oneof![
        3 => Just("..".to_string()),
        2 => Just(".".to_string()),
        4 => prop::string::string_regex("[A-Za-z0-9_.-]{1,16}").unwrap(),
        1 => Just(String::new()),
        1 => Just("..\\..".to_string()),
        1 => Just("C:".to_string()),
        1 => Just("%2e%2e".to_string()),
        1 => prop::string::string_regex("[\\x01-\\x1f]{1,4}").unwrap(),
    ]
}

fn hostile_path() -> impl Strategy<Value = String> {
    prop_oneof![
        6 => prop::collection::vec(hostile_component(), 0..5)
            .prop_map(|parts| parts.join("/")),
        1 => prop::string::string_regex("/[A-Za-z0-9_./-]{0,24}").unwrap(),
        1 => prop::string::string_regex("(\\.\\./){1,3}[A-Za-z0-9_.-]{0,16}").unwrap(),
        1 => Just("a/../../x".to_string()),
        1 => Just("/etc/passwd".to_string()),
        1 => Just("..".to_string()),
    ]
}

const SENTINEL: &[u8] = b"outside sentinel";

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

    #[test]
    fn hostile_paths_never_escape_the_workspace(path in hostile_path()) {
        let temp = TempDir::new().unwrap();
        let ws_root = temp.path().join("ws");
        fs::create_dir(&ws_root).unwrap();
        let sentinel = temp.path().join("sentinel.txt");
        fs::write(&sentinel, SENTINEL).unwrap();

        let ws = open(&ws_root, DEFAULT_QUOTA_BYTES);
        let resolved = Path::new(&path);

        if let Ok(rel) = resolve_rel(resolved) {
            for component in rel.components() {
                prop_assert!(matches!(
                    component,
                    std::path::Component::Normal(_) | std::path::Component::CurDir
                ));
            }
        }

        if let Ok(bytes) = ws.read_file(resolved) {
            prop_assert_ne!(bytes.as_slice(), SENTINEL);
        }
        let _ = ws.write_file(resolved, SENTINEL);

        prop_assert_eq!(fs::read(&sentinel).unwrap(), SENTINEL);
        prop_assert!(ws.total_size().unwrap() <= ws.quota());
        #[cfg(unix)]
        assert_temp_root_has_only_workspace(temp.path());
        temp.close().unwrap();
    }

    #[test]
    fn hostile_paths_with_random_symlinks_never_escape(path in hostile_path(), link in "[A-Za-z0-9_.-]{1,8}") {
        let temp = TempDir::new().unwrap();
        let ws_root = temp.path().join("ws");
        fs::create_dir(&ws_root).unwrap();
        let sentinel = temp.path().join("sentinel.txt");
        fs::write(&sentinel, SENTINEL).unwrap();

        let link_path = ws_root.join(&link);
        let created = std::os::unix::fs::symlink("../sentinel.txt", &link_path)
            .or_else(|_| std::os::unix::fs::symlink("/etc/passwd", &link_path));
        prop_assume!(created.is_ok());
        let ws = open(&ws_root, DEFAULT_QUOTA_BYTES);

        // Both targets are outside the Workspace, so every access must be denied.
        prop_assert!(ws.read_file(Path::new(&link)).is_err());
        prop_assert!(ws.write_file(Path::new(&link), SENTINEL).is_err());
        let _ = ws.write_file(Path::new(&path), SENTINEL);

        prop_assert_eq!(fs::read(&sentinel).unwrap(), SENTINEL);
        assert_temp_root_has_only_workspace(temp.path());
        temp.close().unwrap();
    }
}
