//! Path helpers mirroring Python `monolith/path_utils.py`.
//!
//! Python's `find_main()` is intentionally very lightweight and avoids depending on
//! TensorFlow/absl. It derives the "monolith base directory" from the on-disk
//! location of `path_utils.py` under a Bazel-like layout.
//!
//! In Rust we use the same substring-based heuristic, but allow an override for
//! tests and non-Bazel usage:
//! - `MONOLITH_MAIN_DIR`: if set, returned directly (after sanity check).
//!
//! This keeps filesystem semantics deterministic and testable without requiring
//! Bazel to place our binary under `__main__`.

use std::path::{Path, PathBuf};

use crate::error::{MonolithError, Result};

/// Find base directory of our codebase.
///
/// Mirrors Python `monolith.path_utils.find_main()`:
/// - Try to locate one of `"/__main__/"`, `"/site-packages/"`, or `"/monolith/"` in
///   the path of the currently-linked crate file.
/// - If the split is `"/monolith/"`, return the directory before that.
/// - Otherwise return `prefix + split.strip('/')`.
/// - Validate the returned dir contains a `monolith/` subdirectory.
pub fn find_main() -> Result<PathBuf> {
    if let Ok(v) = std::env::var("MONOLITH_MAIN_DIR") {
        let p = PathBuf::from(v);
        if p.join("monolith").exists() {
            return Ok(p);
        }
        return Err(MonolithError::PyValueError {
            message: format!(
                "MONOLITH_MAIN_DIR does not contain monolith/ subdir: {}",
                p.display()
            ),
        });
    }

    // Use this file's on-disk location as the Python version uses __file__.
    let path = Path::new(file!());
    let abs = std::fs::canonicalize(path).map_err(|e| MonolithError::InternalError {
        message: format!("Failed to canonicalize {}: {}", path.display(), e),
    })?;
    let abs_s = abs.to_string_lossy();

    let splits = ["/__main__/", "/site-packages/", "/monolith/"];
    let mut main_dir: Option<PathBuf> = None;

    for split in splits {
        if let Some(end) = abs_s.rfind(split) {
            if split == "/monolith/" {
                main_dir = Some(PathBuf::from(&abs_s[..end]));
            } else {
                main_dir = Some(PathBuf::from(format!(
                    "{}/{}",
                    &abs_s[..end],
                    split.trim_matches('/')
                )));
            }
            break;
        }
    }

    let main_dir = main_dir.ok_or_else(|| MonolithError::PyValueError {
        message: format!(
            "Unable to find the monolith base directory. This file directory is {}. Are you running under bazel structure?",
            abs.display()
        ),
    })?;

    if main_dir.join("monolith").exists() {
        Ok(main_dir)
    } else {
        Err(MonolithError::PyValueError {
            message: format!(
                "Unable to find the monolith base directory. This file directory is {}. Are you running under bazel structure?",
                abs.display()
            ),
        })
    }
}

/// Returns the resolved path for a "libops" style reference.
///
/// Mirrors Python `monolith.path_utils.get_libops_path(lib_name)` which returns:
/// `os.path.join(find_main(), lib_name)`.
pub fn get_libops_path(lib_name: impl AsRef<Path>) -> Result<PathBuf> {
    Ok(find_main()?.join(lib_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_main_override() {
        // This workspace root contains `monolith/`, so we can safely override.
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        assert!(root.join("monolith").exists());

        std::env::set_var("MONOLITH_MAIN_DIR", root.to_string_lossy().to_string());
        let got = find_main().unwrap();
        assert_eq!(got, root);
        std::env::remove_var("MONOLITH_MAIN_DIR");
    }

    #[test]
    fn test_get_libops_path_points_to_file() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        std::env::set_var("MONOLITH_MAIN_DIR", root.to_string_lossy().to_string());
        let p = get_libops_path("monolith/utils_test.py").unwrap();
        assert!(p.exists());
        std::env::remove_var("MONOLITH_MAIN_DIR");
    }
}
