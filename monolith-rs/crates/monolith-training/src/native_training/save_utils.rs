//! Python `monolith.native_training.save_utils` parity helpers.
//!
//! The real Python module wraps TensorFlow's Saver/checkpoint logic. In Rust we
//! only need the filesystem- and pbtxt-oriented pieces that are referenced by
//! other parity code/tests:
//! - Reading/writing `monolith_checkpoint` (MonolithCheckpointState pbtxt).

use monolith_proto::monolith::native_training::MonolithCheckpointState;
use monolith_proto::text_format;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub const MONOLITH_CKPT_STATE_FILE_NAME: &str = "monolith_checkpoint";

#[derive(Debug, Error)]
pub enum SaveUtilsError {
    #[error("I/O error at {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{0}")]
    Pbtxt(#[from] text_format::PbtxtError),
}

pub type Result<T> = std::result::Result<T, SaveUtilsError>;

fn read_to_string(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).map_err(|e| SaveUtilsError::Io {
        path: path.to_path_buf(),
        source: e,
    })
}

fn atomic_write_string(path: &Path, content: &str, overwrite: bool) -> Result<()> {
    if path.exists() && !overwrite {
        return Ok(());
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent).map_err(|e| SaveUtilsError::Io {
        path: parent.to_path_buf(),
        source: e,
    })?;

    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, content).map_err(|e| SaveUtilsError::Io {
        path: tmp.clone(),
        source: e,
    })?;
    std::fs::rename(&tmp, path).map_err(|e| SaveUtilsError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok(())
}

/// Parse `MonolithCheckpointState` from the `monolith_checkpoint` file.
///
/// Parity notes:
/// - Python returns `None` on file read errors or parse errors; Rust mirrors
///   this by returning `Ok(None)` for any parse error and for missing file.
/// - When `remove_invalid_path=true`, Python:
///   - makes relative `exempt_model_checkpoint_paths` absolute by prefixing `checkpoint_dir`
///   - removes paths that don't exist (using TF's `checkpoint_exists`)
pub fn get_monolith_checkpoint_state(
    checkpoint_dir: &Path,
    filename: Option<&str>,
    remove_invalid_path: bool,
) -> Result<Option<MonolithCheckpointState>> {
    let name = filename.unwrap_or(MONOLITH_CKPT_STATE_FILE_NAME);
    let path = checkpoint_dir.join(name);
    if !path.exists() {
        return Ok(None);
    }

    let text = match read_to_string(&path) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };

    let mut ckpt: MonolithCheckpointState =
        match text_format::parse_pbtxt("monolith.native_training.MonolithCheckpointState", &text) {
            Ok(m) => m,
            Err(_) => return Ok(None),
        };

    if remove_invalid_path {
        // Normalize relative paths and drop missing ones.
        let mut keep: Vec<String> = Vec::new();
        for p in ckpt.exempt_model_checkpoint_paths.iter() {
            let p_abs = if Path::new(p).is_absolute() {
                PathBuf::from(p)
            } else {
                checkpoint_dir.join(p)
            };
            if p_abs.exists() {
                keep.push(p_abs.to_string_lossy().to_string());
            }
        }
        ckpt.exempt_model_checkpoint_paths = keep;
    }

    Ok(Some(ckpt))
}

/// Write `MonolithCheckpointState` to `monolith_checkpoint` pbtxt.
pub fn write_monolith_checkpoint_state(
    checkpoint_dir: &Path,
    state: &MonolithCheckpointState,
    overwrite: bool,
) -> Result<PathBuf> {
    let path = checkpoint_dir.join(MONOLITH_CKPT_STATE_FILE_NAME);
    let text = text_format::to_pbtxt("monolith.native_training.MonolithCheckpointState", state)?;
    atomic_write_string(&path, &text, overwrite)?;
    Ok(path)
}
