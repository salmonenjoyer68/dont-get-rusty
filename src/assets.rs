//! Runtime Image discovery and verification.
//!
//! The Runtime Image is the read-only host directory holding the CPython Guest:
//! `python.wasm` plus `lib/python3.15/`. It is fetched by
//! `scripts/fetch-assets.sh` into the gitignored `assets/` directory and pinned
//! by SHA-256. A missing or mismatched image must stop the binary before it runs
//! anything.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

/// Pinned SHA-256 of the extracted `python.wasm` (CPython 3.15.2, wasi_sdk 33).
pub const PYTHON_WASM_SHA256: &str =
    "dd076791fb6d1286a5dfe105852dfb3367e10562de68ca0a618e615625cdf636";

/// Re-run this to repair an unusable Runtime Image.
pub const FETCH_COMMAND: &str = "scripts/fetch-assets.sh";

/// Overrides the Runtime Image root. Used by the CLI tests; the default is the
/// repository's `assets/` directory.
const ASSETS_ENV: &str = "DGR_ASSETS";

const PYTHON_WASM: &str = "python.wasm";
const STDLIB_DIR: &str = "lib/python3.15";

/// A verified Runtime Image, ready to mount read-only at `/`.
#[derive(Debug, Clone)]
pub struct RuntimeImage {
    root: PathBuf,
    python_wasm: PathBuf,
}

impl RuntimeImage {
    /// Finds the Runtime Image the way the CLI does: `DGR_ASSETS` if set,
    /// otherwise `./assets/`, otherwise the assets directory of the crate.
    pub fn discover() -> Result<Self, AssetError> {
        Self::locate(default_root())
    }

    /// Verifies the Runtime Image at `root`: `python.wasm` must exist, its
    /// SHA-256 must match the pin, and the standard library must be present.
    pub fn locate(root: impl Into<PathBuf>) -> Result<Self, AssetError> {
        let root = root.into();
        let python_wasm = root.join(PYTHON_WASM);

        if !python_wasm.is_file() {
            return Err(AssetError::Missing {
                root,
                missing: PYTHON_WASM.to_owned(),
            });
        }
        if !root.join(STDLIB_DIR).is_dir() {
            return Err(AssetError::Missing {
                root,
                missing: STDLIB_DIR.to_owned(),
            });
        }

        let actual = sha256_file(&python_wasm)?;
        if actual != PYTHON_WASM_SHA256 {
            return Err(AssetError::HashMismatch {
                path: python_wasm,
                expected: PYTHON_WASM_SHA256,
                actual,
            });
        }

        Ok(Self { root, python_wasm })
    }

    /// The directory to mount read-only at `/` in the Guest.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// The verified `python.wasm`.
    pub fn python_wasm(&self) -> &Path {
        &self.python_wasm
    }
}

fn default_root() -> PathBuf {
    if let Some(dir) = std::env::var_os(ASSETS_ENV) {
        return PathBuf::from(dir);
    }
    let cwd_assets = PathBuf::from("assets");
    if cwd_assets.is_dir() {
        return cwd_assets;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("assets")
}

fn sha256_file(path: &Path) -> Result<String, AssetError> {
    let file = File::open(path).map_err(|source| AssetError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = reader.read(&mut buffer).map_err(|source| AssetError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    use fmt::Write as _;

    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

/// Why the Runtime Image cannot be used. Every message names the fix command.
#[derive(Debug)]
pub enum AssetError {
    Missing {
        root: PathBuf,
        missing: String,
    },
    HashMismatch {
        path: PathBuf,
        expected: &'static str,
        actual: String,
    },
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

impl fmt::Display for AssetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { root, missing } => write!(
                f,
                "runtime image not found at `{}`: missing {missing}; run `{FETCH_COMMAND}` to fetch it",
                root.display()
            ),
            Self::HashMismatch {
                path,
                expected,
                actual,
            } => write!(
                f,
                "runtime image is corrupt: `{}` has SHA-256 {actual}, expected {expected}; run `{FETCH_COMMAND}` to refetch it",
                path.display()
            ),
            Self::Io { path, source } => write!(
                f,
                "failed to read `{}`: {source}; run `{FETCH_COMMAND}` to refetch the runtime image",
                path.display()
            ),
        }
    }
}

impl std::error::Error for AssetError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            _ => None,
        }
    }
}
