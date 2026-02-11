//! Python `monolith.native_training.runner_utils` parity helpers.
//!
//! The Python module mostly deals with:
//! - choosing discovery backend (TF_CONFIG/MLP/Consul/ZK)
//! - copying a `checkpoint` file from `restore_dir` into `model_dir` (chief-only)
//!
//! In Rust we implement the behavior used by tests:
//! - `get_discovery_*` builders (already exist in `py_discovery`)
//! - `copy_checkpoint_from_restore_dir` which mirrors the file behavior.

use std::fs;
use std::path::{Path, PathBuf};

use monolith_proto::monolith::native_training::monolith_checkpoint_state::HashTableType;
use monolith_proto::monolith::native_training::MonolithCheckpointState;
use monolith_proto::text_format;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunnerUtilsError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("restore_dir does not contain checkpoint file: {0}")]
    MissingCheckpoint(PathBuf),
    #[error("restore_ckpt {restore_ckpt} is not in checkpoint.all_model_checkpoint_paths")]
    RestoreCkptNotFound { restore_ckpt: String },
}

/// Minimal subset of TensorFlow's `CheckpointState` used by the Python tests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointState {
    pub model_checkpoint_path: String,
    pub all_model_checkpoint_paths: Vec<String>,
}

fn parse_checkpoint_pbtxt(input: &str) -> Option<CheckpointState> {
    // Extremely small parser matching the pbtxt written by Python tests:
    // model_checkpoint_path: 'model.ckpt-61'
    // all_model_checkpoint_paths: 'model.ckpt-61'
    // ...
    let mut model_checkpoint_path: Option<String> = None;
    let mut all: Vec<String> = Vec::new();

    for line in input.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let (k, v) = match line.split_once(':') {
            Some(kv) => kv,
            None => continue,
        };
        let k = k.trim();
        let mut v = v.trim().to_string();
        // Drop surrounding quotes.
        if (v.starts_with('\'') && v.ends_with('\'')) || (v.starts_with('"') && v.ends_with('"')) {
            v = v[1..v.len() - 1].to_string();
        }
        match k {
            "model_checkpoint_path" => model_checkpoint_path = Some(v),
            "all_model_checkpoint_paths" => all.push(v),
            _ => {}
        }
    }

    model_checkpoint_path.map(|m| CheckpointState {
        model_checkpoint_path: m,
        all_model_checkpoint_paths: all,
    })
}

fn write_checkpoint_pbtxt(state: &CheckpointState) -> String {
    // Keep format close to TF's text_format output.
    let mut out = String::new();
    out.push_str(&format!(
        "model_checkpoint_path: \"{}\"\n",
        state.model_checkpoint_path
    ));
    for p in &state.all_model_checkpoint_paths {
        out.push_str(&format!("all_model_checkpoint_paths: \"{}\"\n", p));
    }
    out
}

/// Copy a `checkpoint` file from `restore_dir` into `model_dir` and write a `restore_ckpt` marker.
///
/// Mirrors the behavior in Python's `RunnerConfig._copy_ckpt_file()` as used by
/// `monolith/native_training/runner_utils_test.py`.
pub fn copy_checkpoint_from_restore_dir(
    restore_dir: &Path,
    model_dir: &Path,
    restore_ckpt: Option<&str>,
) -> Result<CheckpointState, RunnerUtilsError> {
    let src = restore_dir.join("checkpoint");
    if !src.exists() {
        return Err(RunnerUtilsError::MissingCheckpoint(src));
    }

    fs::create_dir_all(model_dir)?;

    let src_txt = fs::read_to_string(&src)?;
    let restore_state = parse_checkpoint_pbtxt(&src_txt).unwrap_or(CheckpointState {
        model_checkpoint_path: "model.ckpt-0".to_string(),
        all_model_checkpoint_paths: Vec::new(),
    });

    let picked = if let Some(ckpt) = restore_ckpt {
        if restore_state
            .all_model_checkpoint_paths
            .iter()
            .any(|p| p == ckpt)
        {
            ckpt.to_string()
        } else if restore_state
            .all_model_checkpoint_paths
            .iter()
            .any(|p| Path::new(p).file_name().and_then(|s| s.to_str()) == Some(ckpt))
        {
            // Accept basename-only input.
            let base = ckpt;
            let found = restore_state
                .all_model_checkpoint_paths
                .iter()
                .find(|p| Path::new(p).file_name().and_then(|s| s.to_str()) == Some(base))
                .cloned()
                .unwrap();
            found
        } else {
            return Err(RunnerUtilsError::RestoreCkptNotFound {
                restore_ckpt: ckpt.to_string(),
            });
        }
    } else {
        restore_state.model_checkpoint_path.clone()
    };

    let dst_state = CheckpointState {
        model_checkpoint_path: picked.clone(),
        all_model_checkpoint_paths: vec![picked.clone()],
    };

    let dst_checkpoint_file = model_dir.join("checkpoint");
    if !dst_checkpoint_file.exists() {
        fs::write(&dst_checkpoint_file, write_checkpoint_pbtxt(&dst_state))?;
    }

    // Write restore_ckpt marker (Python uses it as a flag that restore_dir ckpt already applied).
    let restore_marker = model_dir.join("restore_ckpt");
    if !restore_marker.exists() {
        fs::write(&restore_marker, picked)?;
    }

    // Create/update monolith_checkpoint for parity.
    //
    // Python writes MonolithCheckpointState pbtxt and carries forward any existing state
    // already in model_dir, then appends the chosen restore checkpoint path into
    // `exempt_model_checkpoint_paths`.
    let monolith_checkpoint = model_dir.join("monolith_checkpoint");
    if !monolith_checkpoint.exists() {
        // Use defaults consistent with Python's SaverHook path: type = CUCKOO_HASH_MAP.
        let mut st = MonolithCheckpointState::default();
        st.builtin_hash_table_type = Some(HashTableType::CuckooHashMap as i32);
        st.exempt_model_checkpoint_paths = dst_state.all_model_checkpoint_paths.clone();
        let pbtxt = text_format::to_pbtxt("monolith.native_training.MonolithCheckpointState", &st)
            .map_err(|e| RunnerUtilsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        fs::write(&monolith_checkpoint, pbtxt)?;
    } else {
        // Best-effort merge: parse, update exempt list, then overwrite.
        let existing = fs::read_to_string(&monolith_checkpoint)?;
        let mut st: MonolithCheckpointState = text_format::parse_pbtxt(
            "monolith.native_training.MonolithCheckpointState",
            &existing,
        )
        .unwrap_or_default();
        for p in &dst_state.all_model_checkpoint_paths {
            if !st.exempt_model_checkpoint_paths.iter().any(|x| x == p) {
                st.exempt_model_checkpoint_paths.push(p.clone());
            }
        }
        if st.builtin_hash_table_type.is_none() {
            st.builtin_hash_table_type = Some(HashTableType::CuckooHashMap as i32);
        }
        let pbtxt = text_format::to_pbtxt("monolith.native_training.MonolithCheckpointState", &st)
            .map_err(|e| RunnerUtilsError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;
        fs::write(&monolith_checkpoint, pbtxt)?;
    }

    Ok(dst_state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_copy_checkpoint_from_restore_dir() {
        let tmp = tempdir().unwrap();
        let restore_dir = tmp.path().join("restore_dir");
        let model_dir = tmp.path().join("model_dir");
        fs::create_dir_all(&restore_dir).unwrap();

        let pbtxt = r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
all_model_checkpoint_paths: "model.ckpt-0"
"#;
        fs::write(restore_dir.join("checkpoint"), pbtxt).unwrap();

        let state =
            copy_checkpoint_from_restore_dir(&restore_dir, &model_dir, Some("model.ckpt-30"))
                .unwrap();
        assert_eq!(
            Path::new(&state.model_checkpoint_path).file_name().unwrap(),
            "model.ckpt-30"
        );
        assert!(model_dir.join("monolith_checkpoint").exists());
        assert!(model_dir.join("restore_ckpt").exists());
        assert!(model_dir.join("checkpoint").exists());
    }
}
