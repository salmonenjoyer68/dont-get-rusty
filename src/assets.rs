//! Runtime Image discovery and verification.
//!
//! The Runtime Image is the read-only host directory holding the CPython Guest:
//! `python.wasm` plus `lib/python3.15/`. It is fetched by
//! `scripts/fetch-assets.sh` into the gitignored `assets/` directory and pinned
//! by `checksums.txt`, which lists a SHA-256 for every file in the image.
//! `python.wasm` additionally has an embedded pin, so a tampered checksum file
//! cannot bless a tampered Guest. A missing image, a missing or corrupt
//! checksums file, or any mismatched file must stop the binary before it runs
//! anything.

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Component, Path, PathBuf};

use sha2::{Digest, Sha256};

/// Pinned SHA-256 of the extracted `python.wasm` (CPython 3.15.2, wasi_sdk 33).
pub const PYTHON_WASM_SHA256: &str =
    "dd076791fb6d1286a5dfe105852dfb3367e10562de68ca0a618e615625cdf636";

/// Re-run this to repair an unusable Runtime Image.
pub const FETCH_COMMAND: &str = "scripts/fetch-assets.sh";

/// Overrides the Runtime Image root. Used by the CLI tests and by deployments
/// that keep the image outside the repository; the default is the
/// repository's `assets/` directory.
const ASSETS_ENV: &str = "DGR_ASSETS";

const PYTHON_WASM: &str = "python.wasm";
const STDLIB_DIR: &str = "lib/python3.15";
const CHECKSUMS: &str = "checksums.txt";

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

    /// Verifies the Runtime Image at `root`: `python.wasm` and the standard
    /// library must be present, `checksums.txt` must list every image file with
    /// a SHA-256 that still matches, and `python.wasm` must match its embedded
    /// pin.
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

        let checksums_path = root.join(CHECKSUMS);
        let text =
            std::fs::read_to_string(&checksums_path).map_err(|source| match source.kind() {
                std::io::ErrorKind::NotFound => AssetError::Missing {
                    root: root.clone(),
                    missing: CHECKSUMS.to_owned(),
                },
                _ => AssetError::Io {
                    path: checksums_path.clone(),
                    source,
                },
            })?;
        let entries = parse_checksums(&text).map_err(|reason| AssetError::Checksums {
            path: checksums_path.clone(),
            reason,
        })?;

        for (relative, expected) in &entries {
            let path = root.join(relative);
            if !path.is_file() {
                return Err(AssetError::Missing {
                    root: root.clone(),
                    missing: relative.display().to_string(),
                });
            }
            let actual = sha256_file(&path)?;
            if &actual != expected {
                return Err(AssetError::HashMismatch {
                    path,
                    expected: expected.clone(),
                    actual,
                });
            }
        }

        // The checksum file is data, not a trust anchor: `python.wasm` must
        // also match the pin compiled into the binary.
        let listed = entries
            .iter()
            .find(|(relative, _)| relative == Path::new(PYTHON_WASM))
            .map(|(_, hash)| hash)
            .ok_or_else(|| AssetError::Checksums {
                path: checksums_path,
                reason: format!("no entry for `{PYTHON_WASM}`"),
            })?;
        if listed.as_str() != PYTHON_WASM_SHA256 {
            return Err(AssetError::HashMismatch {
                path: python_wasm,
                expected: PYTHON_WASM_SHA256.to_owned(),
                actual: listed.clone(),
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

/// Parses a `checksums.txt`: `<sha256>  <relative path>` per line, blank lines
/// and `#` comments ignored.
///
/// Paths must stay inside the image (no absolute paths, no `..`), so a corrupt
/// or hostile checksums file cannot point startup at an arbitrary host file.
fn parse_checksums(text: &str) -> Result<Vec<(PathBuf, String)>, String> {
    let mut entries = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let (Some(hash), Some(relative)) = (fields.next(), fields.next()) else {
            return Err(format!("line {} is not `<sha256> <path>`", number + 1));
        };
        if fields.next().is_some() {
            return Err(format!("line {} has extra fields", number + 1));
        }
        if hash.len() != 64 || !hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("line {} has a malformed SHA-256", number + 1));
        }
        let relative = Path::new(relative);
        if relative.as_os_str().is_empty()
            || relative
                .components()
                .any(|component| !matches!(component, Component::Normal(_)))
        {
            return Err(format!("line {} has an unsafe path", number + 1));
        }
        entries.push((relative.to_path_buf(), hash.to_ascii_lowercase()));
    }
    if entries.is_empty() {
        return Err("the file lists no image files".to_owned());
    }
    Ok(entries)
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
    Checksums {
        path: PathBuf,
        reason: String,
    },
    HashMismatch {
        path: PathBuf,
        expected: String,
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
            Self::Checksums { path, reason } => write!(
                f,
                "runtime image checksums `{}` are unusable: {reason}; run `{FETCH_COMMAND}` to refetch the runtime image",
                path.display()
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
