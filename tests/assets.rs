//! Startup asset verification: the Runtime Image must be present and hash-pinned.

use std::fs;

use dont_get_rusty::assets::{FETCH_COMMAND, RuntimeImage};

#[test]
fn missing_runtime_image_names_the_fetch_script() {
    let dir = tempfile::tempdir().unwrap();

    let err = RuntimeImage::locate(dir.path()).unwrap_err();
    let message = err.to_string();

    assert!(message.contains(FETCH_COMMAND), "message: {message}");
}

#[test]
fn mismatched_wasm_hash_names_the_fetch_script() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("lib/python3.15")).unwrap();
    fs::write(dir.path().join("python.wasm"), b"not the pinned guest").unwrap();

    let err = RuntimeImage::locate(dir.path()).unwrap_err();
    let message = err.to_string();

    assert!(message.contains(FETCH_COMMAND), "message: {message}");
    assert!(
        message.to_lowercase().contains("sha-256") || message.to_lowercase().contains("hash"),
        "message: {message}"
    );
}

#[test]
fn locate_accepts_the_fetched_runtime_image() {
    let image = RuntimeImage::discover().expect("run `scripts/fetch-assets.sh` first");

    assert!(image.python_wasm().ends_with("python.wasm"));
    assert!(image.root().join("lib/python3.15").is_dir());
}
