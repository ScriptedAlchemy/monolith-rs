//! Misc utilities mirroring Python `monolith/utils.py`.
//!
//! Python's file mixes TensorFlow-specific monkey-patching and `tf.io.gfile`
//! filesystem operations. In Rust we:
//! - keep TF runtime optional by *not* linking TF;
//! - provide pure filesystem equivalents for local paths;
//! - provide a minimal "monkey patch" marker for parity tests.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::error::{MonolithError, Result};

/// Mirrors `utils.enable_monkey_patch()` as a no-op marker.
///
/// In Python this sets `tensorflow.python.training.monitored_session._PREEMPTION_ERRORS`
/// to include `AbortedError`. Rust has no TF runtime, so we expose an observable
/// flag for parity tests and for any future TF bindings.
static MONKEY_PATCH_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn enable_monkey_patch() {
    MONKEY_PATCH_ENABLED.store(true, Ordering::SeqCst);
}

pub fn monkey_patch_enabled() -> bool {
    MONKEY_PATCH_ENABLED.load(Ordering::SeqCst)
}

/// Copy a single file with retry + skip-nonexist semantics.
pub fn copy_file(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    overwrite: bool,
    skip_nonexist: bool,
    max_retries: usize,
) -> Result<()> {
    let src = src.as_ref();
    let dst = dst.as_ref();

    for _ in 0..max_retries.max(1) {
        match fs::metadata(src) {
            Ok(meta) => {
                if !meta.is_file() {
                    return Err(MonolithError::PyValueError {
                        message: format!("{} is not a file!", src.display()),
                    });
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                if skip_nonexist {
                    continue;
                }
                return Err(MonolithError::InternalError {
                    message: format!("{} not found", src.display()),
                });
            }
            Err(e) => {
                return Err(MonolithError::InternalError {
                    message: format!("Failed to stat {}: {}", src.display(), e),
                });
            }
        }

        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent).map_err(|e| MonolithError::InternalError {
                message: format!("Failed to create dir {}: {}", parent.display(), e),
            })?;
        }

        if dst.exists() && !overwrite {
            return Ok(());
        }

        match fs::copy(src, dst) {
            Ok(_) => return Ok(()),
            Err(e) if e.kind() == io::ErrorKind::NotFound && skip_nonexist => continue,
            Err(e) => {
                return Err(MonolithError::InternalError {
                    message: format!(
                        "Failed to copy {} -> {}: {}",
                        src.display(),
                        dst.display(),
                        e
                    ),
                })
            }
        }
    }
    Ok(())
}

/// Recursively copies a directory tree.
///
/// Mirrors Python `CopyRecursively(src, dst, max_workers, skip_nonexist, max_retries)`:
/// - If `src` doesn't exist: return Ok when `skip_nonexist`, else error.
/// - If `src` is a file: copy it to `dst`.
/// - If `dst` exists and `src` is a directory: remove `dst` then recreate it.
/// - When `max_workers > 1`, files are copied in parallel.
pub fn copy_recursively(
    src: impl AsRef<Path>,
    dst: impl AsRef<Path>,
    max_workers: usize,
    skip_nonexist: bool,
    max_retries: usize,
) -> Result<()> {
    let src = src.as_ref().to_path_buf();
    let dst = dst.as_ref().to_path_buf();

    let meta = match fs::metadata(&src) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            if skip_nonexist {
                return Ok(());
            }
            return Err(MonolithError::PyValueError {
                message: format!("{} doesn't exist!", src.display()),
            });
        }
        Err(e) => {
            return Err(MonolithError::InternalError {
                message: format!("Failed to stat {}: {}", src.display(), e),
            })
        }
    };

    if meta.is_file() {
        return copy_file(&src, &dst, true, skip_nonexist, max_retries);
    }

    if dst.exists() {
        fs::remove_dir_all(&dst).map_err(|e| MonolithError::InternalError {
            message: format!("Failed to remove {}: {}", dst.display(), e),
        })?;
    }
    fs::create_dir_all(&dst).map_err(|e| MonolithError::InternalError {
        message: format!("Failed to create {}: {}", dst.display(), e),
    })?;

    let mut file_pairs: Vec<(PathBuf, PathBuf)> = Vec::new();

    fn walk(
        src: &Path,
        dst: &Path,
        file_pairs: &mut Vec<(PathBuf, PathBuf)>,
        max_workers: usize,
        skip_nonexist: bool,
        max_retries: usize,
    ) -> Result<()> {
        for entry in fs::read_dir(src).map_err(|e| MonolithError::InternalError {
            message: format!("Failed to read_dir {}: {}", src.display(), e),
        })? {
            let entry = entry.map_err(|e| MonolithError::InternalError {
                message: format!("Failed to read_dir entry under {}: {}", src.display(), e),
            })?;
            let src_path = entry.path();
            let dst_path = dst.join(entry.file_name());
            let meta = entry.metadata().map_err(|e| MonolithError::InternalError {
                message: format!("Failed to stat {}: {}", src_path.display(), e),
            })?;

            if meta.is_dir() {
                if dst_path.exists() {
                    fs::remove_dir_all(&dst_path).map_err(|e| MonolithError::InternalError {
                        message: format!("Failed to remove {}: {}", dst_path.display(), e),
                    })?;
                }
                fs::create_dir_all(&dst_path).map_err(|e| MonolithError::InternalError {
                    message: format!("Failed to create {}: {}", dst_path.display(), e),
                })?;
                walk(
                    &src_path,
                    &dst_path,
                    file_pairs,
                    max_workers,
                    skip_nonexist,
                    max_retries,
                )?;
            } else if meta.is_file() {
                if max_workers > 1 {
                    file_pairs.push((src_path, dst_path));
                } else {
                    copy_file(src_path, dst_path, true, skip_nonexist, max_retries)?;
                }
            }
        }
        Ok(())
    }

    walk(
        &src,
        &dst,
        &mut file_pairs,
        max_workers,
        skip_nonexist,
        max_retries,
    )?;

    if max_workers > 1 && !file_pairs.is_empty() {
        // Use rayon for simplicity; rayon is already in workspace deps.
        // (Core crate doesn't currently depend on rayon, so keep this sequential
        // unless the caller opts in by enabling the "parallel" feature in their crate.)
        // For now, implement a small thread pool using std::thread.
        let workers = max_workers.max(1);
        let chunks: Vec<Vec<(PathBuf, PathBuf)>> = (0..workers)
            .map(|i| {
                file_pairs
                    .iter()
                    .skip(i)
                    .step_by(workers)
                    .cloned()
                    .collect()
            })
            .collect();

        let mut handles = Vec::new();
        for chunk in chunks {
            let handle = std::thread::spawn(move || -> Result<()> {
                for (s, d) in chunk {
                    copy_file(s, d, true, skip_nonexist, max_retries)?;
                }
                Ok(())
            });
            handles.push(handle);
        }

        for h in handles {
            match h.join() {
                Ok(r) => r?,
                Err(_) => {
                    return Err(MonolithError::InternalError {
                        message: "copy_recursively worker thread panicked".to_string(),
                    })
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_monkey_patch_marker() {
        assert!(!monkey_patch_enabled());
        enable_monkey_patch();
        assert!(monkey_patch_enabled());
    }

    #[test]
    fn test_copy_recursively_multithreaded() {
        let tmp = tempdir().unwrap();
        let src = tmp.path().join("src");
        let dst = tmp.path().join("dst");

        fs::create_dir_all(src.join("subdir")).unwrap();
        fs::write(src.join("file.txt"), "root").unwrap();
        fs::write(src.join("subdir").join("innerfile.txt"), "inner").unwrap();

        copy_recursively(&src, &dst, 2, true, 5).unwrap();
        let got = fs::read_to_string(dst.join("subdir").join("innerfile.txt")).unwrap();
        assert_eq!(got, "inner");
    }
}
