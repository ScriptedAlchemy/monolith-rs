//! Checkpointer trait for save/restore operations.
//!
//! This module defines the core `Checkpointer` trait that all checkpoint
//! implementations must satisfy.

use crate::state::ModelState;
use crate::CheckpointError;
use std::path::{Path, PathBuf};

/// Result type for checkpointer operations.
pub type Result<T> = std::result::Result<T, CheckpointError>;

/// Trait for checkpoint serialization and deserialization.
///
/// Implementors of this trait provide the logic for saving and restoring
/// model state to/from persistent storage.
///
/// # Examples
///
/// ```no_run
/// use monolith_checkpoint::{Checkpointer, ModelState, JsonCheckpointer};
/// use std::path::Path;
///
/// fn main() -> monolith_checkpoint::Result<()> {
///     let checkpointer = JsonCheckpointer::new();
///     let state = ModelState::new(1000);
///
///     // Save checkpoint
///     checkpointer.save(Path::new("/tmp/checkpoint"), &state)?;
///
///     // Restore checkpoint
///     let restored = checkpointer.restore(Path::new("/tmp/checkpoint"))?;
///     Ok(())
/// }
/// ```
pub trait Checkpointer: Send + Sync {
    /// Save model state to the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path where the checkpoint should be saved
    /// * `state` - Model state to serialize
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or I/O fails.
    fn save(&self, path: &Path, state: &ModelState) -> Result<()>;

    /// Restore model state from the specified path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint to restore
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint doesn't exist, is corrupted,
    /// or deserialization fails.
    fn restore(&self, path: &Path) -> Result<ModelState>;

    /// Find the latest checkpoint in a directory.
    ///
    /// Scans the directory for valid checkpoints and returns the path
    /// to the most recent one based on the global step or timestamp.
    ///
    /// # Arguments
    ///
    /// * `dir` - Directory to search for checkpoints
    ///
    /// # Returns
    ///
    /// Returns `Some(path)` if a checkpoint is found, `None` otherwise.
    fn latest(&self, dir: &Path) -> Option<PathBuf>;
}

/// JSON-based checkpoint implementation.
///
/// Serializes model state to JSON format. This is useful for debugging
/// and compatibility, but may not be the most efficient for large models.
#[derive(Debug, Clone, Default)]
pub struct JsonCheckpointer {
    /// Whether to pretty-print JSON output.
    pub pretty: bool,
}

impl JsonCheckpointer {
    /// Create a new JSON checkpointer.
    pub fn new() -> Self {
        Self { pretty: false }
    }

    /// Create a new JSON checkpointer with pretty printing.
    pub fn pretty() -> Self {
        Self { pretty: true }
    }

    /// Get the checkpoint filename for a given step.
    #[allow(dead_code)]
    fn checkpoint_filename(step: u64) -> String {
        format!("checkpoint-{}.json", step)
    }

    /// Parse step number from checkpoint filename.
    fn parse_step(filename: &str) -> Option<u64> {
        if filename.starts_with("checkpoint-") && filename.ends_with(".json") {
            let step_str = filename
                .strip_prefix("checkpoint-")?
                .strip_suffix(".json")?;
            step_str.parse().ok()
        } else {
            None
        }
    }
}

impl Checkpointer for JsonCheckpointer {
    fn save(&self, path: &Path, state: &ModelState) -> Result<()> {
        tracing::info!(path = %path.display(), step = state.global_step, "Saving checkpoint");

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CheckpointError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        // Serialize to JSON
        let json = if self.pretty {
            serde_json::to_string_pretty(state)
        } else {
            serde_json::to_string(state)
        }
        .map_err(CheckpointError::Serialization)?;

        // Write to file
        std::fs::write(path, json).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        tracing::debug!(
            path = %path.display(),
            size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0),
            "Checkpoint saved"
        );

        Ok(())
    }

    fn restore(&self, path: &Path) -> Result<ModelState> {
        tracing::info!(path = %path.display(), "Restoring checkpoint");

        if !path.exists() {
            return Err(CheckpointError::NotFound(path.to_path_buf()));
        }

        let json = std::fs::read_to_string(path).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let state: ModelState =
            serde_json::from_str(&json).map_err(CheckpointError::Deserialization)?;

        tracing::info!(
            path = %path.display(),
            step = state.global_step,
            tables = state.hash_tables.len(),
            "Checkpoint restored"
        );

        Ok(state)
    }

    fn latest(&self, dir: &Path) -> Option<PathBuf> {
        if !dir.is_dir() {
            return None;
        }

        let mut latest_step = 0u64;
        let mut latest_path = None;

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                    if let Some(step) = Self::parse_step(filename) {
                        if step > latest_step {
                            latest_step = step;
                            latest_path = Some(path);
                        }
                    }
                }
            }
        }

        latest_path
    }
}

/// Binary checkpoint implementation using MessagePack or similar format.
///
/// More efficient than JSON for large models, but less human-readable.
#[derive(Debug, Clone, Default)]
pub struct BinaryCheckpointer;

impl BinaryCheckpointer {
    /// Create a new binary checkpointer.
    pub fn new() -> Self {
        Self
    }

    /// Get the checkpoint filename for a given step.
    #[allow(dead_code)]
    fn checkpoint_filename(step: u64) -> String {
        format!("checkpoint-{}.bin", step)
    }

    /// Parse step number from checkpoint filename.
    fn parse_step(filename: &str) -> Option<u64> {
        if filename.starts_with("checkpoint-") && filename.ends_with(".bin") {
            let step_str = filename.strip_prefix("checkpoint-")?.strip_suffix(".bin")?;
            step_str.parse().ok()
        } else {
            None
        }
    }
}

impl Checkpointer for BinaryCheckpointer {
    fn save(&self, path: &Path, state: &ModelState) -> Result<()> {
        tracing::info!(path = %path.display(), step = state.global_step, "Saving binary checkpoint");

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CheckpointError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        // For now, use JSON as the binary format placeholder
        // In production, this would use MessagePack, bincode, or similar
        let data = serde_json::to_vec(state).map_err(CheckpointError::Serialization)?;

        std::fs::write(path, data).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        Ok(())
    }

    fn restore(&self, path: &Path) -> Result<ModelState> {
        tracing::info!(path = %path.display(), "Restoring binary checkpoint");

        if !path.exists() {
            return Err(CheckpointError::NotFound(path.to_path_buf()));
        }

        let data = std::fs::read(path).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        let state: ModelState =
            serde_json::from_slice(&data).map_err(CheckpointError::Deserialization)?;

        Ok(state)
    }

    fn latest(&self, dir: &Path) -> Option<PathBuf> {
        if !dir.is_dir() {
            return None;
        }

        let mut latest_step = 0u64;
        let mut latest_path = None;

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                    if let Some(step) = Self::parse_step(filename) {
                        if step > latest_step {
                            latest_step = step;
                            latest_path = Some(path);
                        }
                    }
                }
            }
        }

        latest_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_json_checkpointer_save_restore() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");

        let checkpointer = JsonCheckpointer::new();
        let mut state = ModelState::new(1000);
        state.set_metadata("test_key", "test_value");

        // Save
        checkpointer.save(&path, &state).unwrap();
        assert!(path.exists());

        // Restore
        let restored = checkpointer.restore(&path).unwrap();
        assert_eq!(restored.global_step, 1000);
        assert_eq!(
            restored.metadata.get("test_key"),
            Some(&"test_value".to_string())
        );
    }

    #[test]
    fn test_json_checkpointer_pretty() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.json");

        let checkpointer = JsonCheckpointer::pretty();
        let state = ModelState::new(500);

        checkpointer.save(&path, &state).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains('\n')); // Pretty printed has newlines
    }

    #[test]
    fn test_json_checkpointer_restore_not_found() {
        let checkpointer = JsonCheckpointer::new();
        let result = checkpointer.restore(Path::new("/nonexistent/path/checkpoint.json"));
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[test]
    fn test_json_checkpointer_latest() {
        let dir = tempdir().unwrap();
        let checkpointer = JsonCheckpointer::new();

        // Create multiple checkpoints
        for step in [100, 500, 300, 700, 200] {
            let path = dir.path().join(format!("checkpoint-{}.json", step));
            let state = ModelState::new(step);
            checkpointer.save(&path, &state).unwrap();
        }

        // Find latest
        let latest = checkpointer.latest(dir.path());
        assert!(latest.is_some());
        let latest_path = latest.unwrap();
        assert!(latest_path.to_str().unwrap().contains("checkpoint-700"));
    }

    #[test]
    fn test_json_checkpointer_latest_empty_dir() {
        let dir = tempdir().unwrap();
        let checkpointer = JsonCheckpointer::new();

        let latest = checkpointer.latest(dir.path());
        assert!(latest.is_none());
    }

    #[test]
    fn test_binary_checkpointer_save_restore() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.bin");

        let checkpointer = BinaryCheckpointer::new();
        let state = ModelState::new(2000);

        checkpointer.save(&path, &state).unwrap();
        assert!(path.exists());

        let restored = checkpointer.restore(&path).unwrap();
        assert_eq!(restored.global_step, 2000);
    }

    #[test]
    fn test_parse_step() {
        assert_eq!(
            JsonCheckpointer::parse_step("checkpoint-100.json"),
            Some(100)
        );
        assert_eq!(JsonCheckpointer::parse_step("checkpoint-0.json"), Some(0));
        assert_eq!(
            JsonCheckpointer::parse_step("checkpoint-999999.json"),
            Some(999999)
        );
        assert_eq!(JsonCheckpointer::parse_step("invalid.json"), None);
        assert_eq!(JsonCheckpointer::parse_step("checkpoint-abc.json"), None);
    }
}
