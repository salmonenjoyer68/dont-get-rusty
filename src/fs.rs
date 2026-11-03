//! The host-side Workspace: every durable file operation goes through here.
//!
//! A [`Workspace`] holds one [`cap_std::fs::Dir`] capability for the Workspace
//! root. All paths are model-supplied and workspace-relative; absolute paths,
//! Windows prefixes, and every `..` component are rejected before any disk
//! access. `cap-std` resolves the remaining path component at a time and
//! re-roots symlink expansions at the Workspace handle, so a link that points
//! outside the Workspace fails with [`std::io::ErrorKind::PermissionDenied`]
//! and touches nothing.
//!
//! Writes enforce a total-size quota. The whole tree is summed and the target's
//! current size subtracted *before* any directory is created or file truncated,
//! so a denied write leaves the Workspace exactly as it was. The total is a
//! snapshot: an external host process could change the tree between the scan
//! and the write. That is acceptable for a single-user local CLI; it is not a
//! defense against a concurrently hostile host process. The Guest's preopen is
//! read-only (see [`crate::sandbox`]), so the Guest cannot race the scan either.

use std::fmt;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};

/// Default Workspace quota: 50 MB of regular-file bytes.
pub const DEFAULT_QUOTA_BYTES: u64 = 50 * 1024 * 1024;

/// Guards the recursive size scan against pathological nesting.
const MAX_DEPTH: u32 = 64;

/// Why a model-supplied path was rejected before touching the filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathError {
    /// The path was empty.
    Empty,
    /// The path names the Workspace root, which is not a file or directory
    /// target for these operations.
    Root,
    /// The path contains a `..` component.
    ParentDir,
    /// The path is absolute (`/`, a drive, or a UNC prefix).
    Absolute,
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "path is empty",
            Self::Root => "path names the workspace root",
            Self::ParentDir => "`..` components are not permitted",
            Self::Absolute => "absolute paths are not permitted",
        })
    }
}

impl std::error::Error for PathError {}

/// A Workspace operation failure.
#[derive(Debug, thiserror::Error)]
pub enum FsError {
    #[error("invalid path: {0}")]
    InvalidPath(#[from] PathError),
    #[error("path does not name a regular file")]
    NotAFile,
    #[error("workspace quota exceeded: write would use {projected} bytes, limit is {limit}")]
    QuotaExceeded { projected: u64, limit: u64 },
    #[error("workspace tree is nested too deeply")]
    TooDeep,
    /// An overwrite confirmation callback answered no; nothing was written.
    #[error("overwrite of `{}` was not confirmed; the file was left unchanged", .0.display())]
    OverwriteDeclined(PathBuf),
    /// `cap-std` failures, including escape denials (`PermissionDenied`).
    #[error(transparent)]
    Io(#[from] io::Error),
}

/// Result of a successful [`Workspace::write_file`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteInfo {
    /// Whether the target existed as a regular file before this write.
    pub existed: bool,
    /// Number of bytes written (the full content length).
    pub bytes_written: u64,
}

/// What a directory entry is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    File,
    Dir,
    Symlink,
    Other,
}

impl fmt::Display for EntryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::File => "file",
            Self::Dir => "dir",
            Self::Symlink => "symlink",
            Self::Other => "other",
        })
    }
}

/// One entry of a directory listing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry {
    /// Entry name, lossily decoded to UTF-8; control characters are escaped
    /// when rendered.
    pub name: String,
    pub kind: EntryKind,
    /// Size in bytes for regular files, `0` for everything else.
    pub size: u64,
}

impl fmt::Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size = if self.kind == EntryKind::File {
            self.size.to_string()
        } else {
            "-".to_owned()
        };
        write!(f, "{}\t{}\t{}", self.kind, size, escape_name(&self.name))
    }
}

/// Renders a listing in the stable, line-oriented format fed to the model:
/// one `kind<TAB>size<TAB>name` line per entry (`-` as size for non-files), or
/// `(empty directory)` for an empty directory.
pub fn format_listing(entries: &[Entry]) -> String {
    if entries.is_empty() {
        return "(empty directory)\n".to_owned();
    }
    let mut out = String::new();
    for entry in entries {
        out.push_str(&entry.to_string());
        out.push('\n');
    }
    out
}

fn escape_name(name: &str) -> String {
    name.chars()
        .flat_map(|c| {
            if c.is_control() {
                c.escape_default().collect::<Vec<_>>()
            } else {
                vec![c]
            }
        })
        .collect()
}

/// Rejects absolute paths and every `..` component, and drops `.` components.
///
/// Returns an empty path for input like `"."` (the Workspace root). This is the
/// model-facing rule; it is not the security boundary — `cap-std` enforces
/// containment at the actual open, which also covers symlink targets.
pub fn resolve_rel(path: &Path) -> Result<PathBuf, FsError> {
    if path.as_os_str().is_empty() {
        return Err(PathError::Empty.into());
    }
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => out.push(name),
            Component::CurDir => {}
            Component::ParentDir => return Err(PathError::ParentDir.into()),
            Component::RootDir | Component::Prefix(_) => return Err(PathError::Absolute.into()),
        }
    }
    Ok(out)
}

/// Validates a path and rejects the Workspace root itself.
fn resolve_target(path: &Path) -> Result<PathBuf, FsError> {
    let rel = resolve_rel(path)?;
    if rel.as_os_str().is_empty() {
        return Err(PathError::Root.into());
    }
    Ok(rel)
}

/// The host-side Workspace capability.
pub struct Workspace {
    root: Dir,
    path: PathBuf,
    quota: u64,
}

impl Workspace {
    /// Opens `root` as a Workspace with the default 50 MB quota.
    pub fn open(root: impl AsRef<Path>) -> Result<Self, FsError> {
        Self::open_with_quota(root, DEFAULT_QUOTA_BYTES)
    }

    /// Opens `root` as a Workspace with an explicit total-size quota.
    pub fn open_with_quota(root: impl AsRef<Path>, quota: u64) -> Result<Self, FsError> {
        let path = root.as_ref().to_path_buf();
        Ok(Self {
            root: Dir::open_ambient_dir(&path, ambient_authority())?,
            path,
            quota,
        })
    }

    /// The host path this Workspace was opened from, used for the Guest's
    /// read-only `/workspace` preopen. Model-supplied paths are never resolved
    /// against it; they stay relative to the capability.
    pub fn root_path(&self) -> &Path {
        &self.path
    }

    /// The configured quota in bytes.
    pub fn quota(&self) -> u64 {
        self.quota
    }

    /// Reads a regular file (or an in-workspace symlink's target) entirely.
    pub fn read_file(&self, path: impl AsRef<Path>) -> Result<Vec<u8>, FsError> {
        let rel = resolve_target(path.as_ref())?;
        Ok(self.root.read(rel)?)
    }

    /// Writes `content`, enforcing the quota before touching the disk.
    ///
    /// A denied write creates no directories and leaves an existing file
    /// intact. Overwriting counts as `total - old + new`.
    pub fn write_file(&self, path: impl AsRef<Path>, content: &[u8]) -> Result<WriteInfo, FsError> {
        self.write_file_with_confirm(path, content, |_| true)
    }

    /// Writes `content`, enforcing the quota before touching the disk, and
    /// asks `confirm` before overwriting an existing file.
    ///
    /// Every traversal, target-type, and quota check runs first. Only then is
    /// `confirm` called with the Workspace-relative target; returning `false`
    /// leaves the Workspace exactly as it was and yields
    /// [`FsError::OverwriteDeclined`]. `confirm` is never called for a target
    /// that did not exist, because creating a file destroys nothing.
    pub fn write_file_with_confirm(
        &self,
        path: impl AsRef<Path>,
        content: &[u8],
        confirm: impl FnOnce(&Path) -> bool,
    ) -> Result<WriteInfo, FsError> {
        let rel = resolve_target(path.as_ref())?;

        let total = self.total_size()?;
        let existing = match self.root.metadata(&rel) {
            Ok(metadata) if metadata.is_file() => Some(metadata.len()),
            Ok(_) => return Err(FsError::NotAFile),
            Err(error) if error.kind() == io::ErrorKind::NotFound => None,
            Err(error) => return Err(error.into()),
        };
        let old = existing.unwrap_or(0);
        let projected = total
            .saturating_sub(old)
            .saturating_add(content.len() as u64);
        if projected > self.quota {
            return Err(FsError::QuotaExceeded {
                projected,
                limit: self.quota,
            });
        }

        // Every check has passed; the confirmation runs before anything on
        // disk is created or truncated.
        if existing.is_some() && !confirm(&rel) {
            return Err(FsError::OverwriteDeclined(rel));
        }

        // Only now is anything on disk created or truncated.
        if let Some(parent) = rel.parent()
            && !parent.as_os_str().is_empty()
        {
            self.root.create_dir_all(parent)?;
        }
        let mut file = self.root.open_with(
            &rel,
            OpenOptions::new().write(true).create(true).truncate(true),
        )?;
        file.write_all(content)?;

        Ok(WriteInfo {
            existed: existing.is_some(),
            bytes_written: content.len() as u64,
        })
    }

    /// Lists a directory (the Workspace root when `path` is `None`), sorted by
    /// name. Symlinks are reported as links; they are not followed.
    pub fn list_files(&self, path: Option<&Path>) -> Result<Vec<Entry>, FsError> {
        let rel = match path {
            None => PathBuf::new(),
            Some(path) => resolve_rel(path)?,
        };
        let dir = if rel.as_os_str().is_empty() {
            self.root.try_clone()?
        } else {
            self.root.open_dir(&rel)?
        };

        let mut entries = Vec::new();
        for entry in dir.entries()? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            let kind = if file_type.is_dir() {
                EntryKind::Dir
            } else if file_type.is_file() {
                EntryKind::File
            } else if file_type.is_symlink() {
                EntryKind::Symlink
            } else {
                EntryKind::Other
            };
            let size = match kind {
                EntryKind::File => entry.metadata()?.len(),
                _ => 0,
            };
            entries.push(Entry {
                name: entry.file_name().to_string_lossy().into_owned(),
                kind,
                size,
            });
        }
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(entries)
    }

    /// Total size of the Workspace's regular files in bytes.
    ///
    /// Symlinks are neither followed nor counted, so outside targets cannot be
    /// sized through the Workspace and loops cannot hang the scan. Hard links
    /// are counted once per directory entry.
    pub fn total_size(&self) -> Result<u64, FsError> {
        fn walk(dir: &Dir, depth: u32) -> Result<u64, FsError> {
            if depth > MAX_DEPTH {
                return Err(FsError::TooDeep);
            }
            let mut total = 0u64;
            for entry in dir.entries()? {
                let entry = entry?;
                let file_type = entry.file_type()?;
                if file_type.is_dir() {
                    total = total.saturating_add(walk(&entry.open_dir()?, depth + 1)?);
                } else if file_type.is_file() {
                    total = total.saturating_add(entry.metadata()?.len());
                }
            }
            Ok(total)
        }
        walk(&self.root, 0)
    }
}
