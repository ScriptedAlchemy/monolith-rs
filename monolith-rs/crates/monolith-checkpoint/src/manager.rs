//! Checkpoint manager for tracking and managing checkpoint lifecycle.
//!
//! This module provides `CheckpointManager`, which handles:
//! - Tracking checkpoint history
//! - Automatic cleanup of old checkpoints
//! - Finding the latest checkpoint for restore operations

use crate::checkpointer::{Checkpointer, Result};
use crate::state::ModelState;
use crate::CheckpointError;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Information about a saved checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// Path to the checkpoint file.
    pub path: PathBuf,

    /// Global step at which this checkpoint was saved.
    pub global_step: u64,

    /// Timestamp when checkpoint was created (Unix epoch seconds).
    pub timestamp: u64,
}

/// Configuration for the checkpoint manager.
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Directory where checkpoints are stored.
    pub checkpoint_dir: PathBuf,

    /// Maximum number of checkpoints to keep.
    /// Older checkpoints are automatically deleted.
    pub max_to_keep: usize,

    /// Whether to keep best checkpoints based on a metric.
    pub keep_best: bool,

    /// Interval in steps between automatic checkpoints.
    /// Set to 0 to disable automatic checkpointing.
    pub checkpoint_interval: u64,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            checkpoint_dir: PathBuf::from("checkpoints"),
            max_to_keep: 5,
            keep_best: false,
            checkpoint_interval: 1000,
        }
    }
}

impl CheckpointConfig {
    /// Create a new checkpoint configuration.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_dir` - Directory where checkpoints will be stored
    pub fn new(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.into(),
            ..Default::default()
        }
    }

    /// Set the maximum number of checkpoints to keep.
    pub fn with_max_to_keep(mut self, max_to_keep: usize) -> Self {
        self.max_to_keep = max_to_keep;
        self
    }

    /// Enable keeping best checkpoints.
    pub fn with_keep_best(mut self, keep_best: bool) -> Self {
        self.keep_best = keep_best;
        self
    }

    /// Set the checkpoint interval.
    pub fn with_checkpoint_interval(mut self, interval: u64) -> Self {
        self.checkpoint_interval = interval;
        self
    }
}

/// Manages checkpoint lifecycle including saving, restoring, and cleanup.
///
/// # Examples
///
/// ```no_run
/// use monolith_checkpoint::{CheckpointManager, CheckpointConfig, JsonCheckpointer, ModelState};
///
/// fn main() -> monolith_checkpoint::Result<()> {
///     let config = CheckpointConfig::new("/tmp/checkpoints")
///         .with_max_to_keep(3);
///
///     let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());
///
///     // Save checkpoint
///     let state = ModelState::new(1000);
///     manager.save(&state)?;
///
///     // Restore latest
///     let restored = manager.restore_latest()?;
///     Ok(())
/// }
/// ```
pub struct CheckpointManager<C: Checkpointer> {
    /// Configuration for checkpoint management.
    config: CheckpointConfig,

    /// The checkpointer implementation to use.
    checkpointer: Arc<C>,

    /// History of saved checkpoints (oldest first).
    checkpoint_history: VecDeque<CheckpointInfo>,
}

impl<C: Checkpointer> CheckpointManager<C> {
    /// Create a new checkpoint manager.
    ///
    /// # Arguments
    ///
    /// * `config` - Checkpoint configuration
    /// * `checkpointer` - Checkpointer implementation to use
    pub fn new(config: CheckpointConfig, checkpointer: C) -> Self {
        Self {
            config,
            checkpointer: Arc::new(checkpointer),
            checkpoint_history: VecDeque::new(),
        }
    }

    /// Get the checkpoint directory.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.config.checkpoint_dir
    }

    /// Get the number of tracked checkpoints.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoint_history.len()
    }

    /// Get the configuration.
    pub fn config(&self) -> &CheckpointConfig {
        &self.config
    }

    /// Save a checkpoint and manage history.
    ///
    /// This saves the model state and automatically cleans up old checkpoints
    /// if the number exceeds `max_to_keep`.
    ///
    /// # Arguments
    ///
    /// * `state` - Model state to save
    ///
    /// # Returns
    ///
    /// Returns the checkpoint info for the saved checkpoint.
    pub fn save(&mut self, state: &ModelState) -> Result<CheckpointInfo> {
        let filename = format!("checkpoint-{}.json", state.global_step);
        let path = self.config.checkpoint_dir.join(filename);

        tracing::info!(
            step = state.global_step,
            path = %path.display(),
            "Saving checkpoint via manager"
        );

        // Save the checkpoint
        self.checkpointer.save(&path, state)?;

        // Create checkpoint info
        let info = CheckpointInfo {
            path: path.clone(),
            global_step: state.global_step,
            timestamp: state.timestamp,
        };

        // Add to history
        self.checkpoint_history.push_back(info.clone());

        // Cleanup old checkpoints
        self.cleanup_old()?;

        Ok(info)
    }

    /// Restore the latest checkpoint.
    ///
    /// # Returns
    ///
    /// Returns the restored model state, or an error if no checkpoint exists.
    pub fn restore_latest(&self) -> Result<ModelState> {
        let latest_path = self
            .checkpointer
            .latest(&self.config.checkpoint_dir)
            .ok_or_else(|| CheckpointError::NotFound(self.config.checkpoint_dir.clone()))?;

        tracing::info!(path = %latest_path.display(), "Restoring latest checkpoint");

        self.checkpointer.restore(&latest_path)
    }

    /// Restore a specific checkpoint by path.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the checkpoint to restore
    pub fn restore(&self, path: &Path) -> Result<ModelState> {
        self.checkpointer.restore(path)
    }

    /// Restore a checkpoint by step number.
    ///
    /// # Arguments
    ///
    /// * `step` - Global step number to restore
    pub fn restore_step(&self, step: u64) -> Result<ModelState> {
        let filename = format!("checkpoint-{}.json", step);
        let path = self.config.checkpoint_dir.join(filename);
        self.checkpointer.restore(&path)
    }

    /// Clean up old checkpoints, keeping only `max_to_keep` most recent.
    ///
    /// This method removes the oldest checkpoints when the total count
    /// exceeds the configured maximum.
    pub fn cleanup_old(&mut self) -> Result<()> {
        while self.checkpoint_history.len() > self.config.max_to_keep {
            if let Some(old_checkpoint) = self.checkpoint_history.pop_front() {
                tracing::info!(
                    path = %old_checkpoint.path.display(),
                    step = old_checkpoint.global_step,
                    "Removing old checkpoint"
                );

                if old_checkpoint.path.exists() {
                    std::fs::remove_file(&old_checkpoint.path).map_err(|e| {
                        CheckpointError::Io {
                            path: old_checkpoint.path.clone(),
                            source: e,
                        }
                    })?;
                }
            }
        }

        Ok(())
    }

    /// List all available checkpoints in the checkpoint directory.
    ///
    /// Returns checkpoints sorted by step number (ascending).
    pub fn list_checkpoints(&self) -> Vec<PathBuf> {
        let mut checkpoints = Vec::new();

        if let Some(latest) = self.checkpointer.latest(&self.config.checkpoint_dir) {
            // Start from latest and work backwards
            checkpoints.push(latest);
        }

        // Also scan directory for any checkpoints not in our history
        if let Ok(entries) = std::fs::read_dir(&self.config.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|f| f.to_str())
                    .map(|f| f.starts_with("checkpoint-") && f.ends_with(".json"))
                    .unwrap_or(false)
                    && !checkpoints.contains(&path)
                {
                    checkpoints.push(path);
                }
            }
        }

        // Sort by extracting step number
        checkpoints.sort_by_key(|p| {
            p.file_name()
                .and_then(|f| f.to_str())
                .and_then(|f| {
                    f.strip_prefix("checkpoint-")
                        .and_then(|s| s.strip_suffix(".json"))
                        .and_then(|s| s.parse::<u64>().ok())
                })
                .unwrap_or(0)
        });

        checkpoints
    }

    /// Check if a checkpoint should be saved at the given step.
    ///
    /// # Arguments
    ///
    /// * `step` - Current training step
    ///
    /// # Returns
    ///
    /// Returns `true` if a checkpoint should be saved based on the interval.
    pub fn should_checkpoint(&self, step: u64) -> bool {
        if self.config.checkpoint_interval == 0 {
            return false;
        }
        step > 0 && step.is_multiple_of(self.config.checkpoint_interval)
    }

    /// Initialize the manager by scanning the checkpoint directory.
    ///
    /// This populates the checkpoint history from existing files.
    pub fn initialize(&mut self) -> Result<()> {
        self.checkpoint_history.clear();

        if !self.config.checkpoint_dir.exists() {
            return Ok(());
        }

        let mut checkpoints: Vec<(u64, PathBuf)> = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&self.config.checkpoint_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(filename) = path.file_name().and_then(|f| f.to_str()) {
                    if filename.starts_with("checkpoint-") && filename.ends_with(".json") {
                        if let Some(step) = filename
                            .strip_prefix("checkpoint-")
                            .and_then(|s| s.strip_suffix(".json"))
                            .and_then(|s| s.parse::<u64>().ok())
                        {
                            checkpoints.push((step, path));
                        }
                    }
                }
            }
        }

        // Sort by step
        checkpoints.sort_by_key(|(step, _)| *step);

        // Populate history
        for (step, path) in checkpoints {
            self.checkpoint_history.push_back(CheckpointInfo {
                path,
                global_step: step,
                timestamp: 0, // Unknown for existing checkpoints
            });
        }

        tracing::info!(
            count = self.checkpoint_history.len(),
            "Initialized checkpoint manager"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpointer::JsonCheckpointer;
    use tempfile::tempdir;

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.max_to_keep, 5);
        assert_eq!(config.checkpoint_interval, 1000);
        assert!(!config.keep_best);
    }

    #[test]
    fn test_checkpoint_config_builder() {
        let config = CheckpointConfig::new("/tmp/ckpts")
            .with_max_to_keep(10)
            .with_checkpoint_interval(500)
            .with_keep_best(true);

        assert_eq!(config.checkpoint_dir, PathBuf::from("/tmp/ckpts"));
        assert_eq!(config.max_to_keep, 10);
        assert_eq!(config.checkpoint_interval, 500);
        assert!(config.keep_best);
    }

    #[test]
    fn test_checkpoint_manager_save() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path()).with_max_to_keep(3);
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        let state = ModelState::new(1000);
        let info = manager.save(&state).unwrap();

        assert_eq!(info.global_step, 1000);
        assert!(info.path.exists());
        assert_eq!(manager.checkpoint_count(), 1);
    }

    #[test]
    fn test_checkpoint_manager_cleanup() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path()).with_max_to_keep(2);
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        // Save 4 checkpoints
        for step in [100, 200, 300, 400] {
            let state = ModelState::new(step);
            manager.save(&state).unwrap();
        }

        // Should only keep 2
        assert_eq!(manager.checkpoint_count(), 2);

        // Old ones should be deleted
        assert!(!dir.path().join("checkpoint-100.json").exists());
        assert!(!dir.path().join("checkpoint-200.json").exists());

        // Recent ones should exist
        assert!(dir.path().join("checkpoint-300.json").exists());
        assert!(dir.path().join("checkpoint-400.json").exists());
    }

    #[test]
    fn test_checkpoint_manager_restore_latest() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path());
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        // Save multiple checkpoints
        for step in [100, 500, 300] {
            let mut state = ModelState::new(step);
            state.set_metadata("step", &step.to_string());
            manager.save(&state).unwrap();
        }

        // Restore latest
        let restored = manager.restore_latest().unwrap();
        assert_eq!(restored.global_step, 500);
    }

    #[test]
    fn test_checkpoint_manager_restore_step() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path());
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        // Save checkpoints
        for step in [100, 200, 300] {
            let state = ModelState::new(step);
            manager.save(&state).unwrap();
        }

        // Restore specific step
        let restored = manager.restore_step(200).unwrap();
        assert_eq!(restored.global_step, 200);
    }

    #[test]
    fn test_checkpoint_manager_should_checkpoint() {
        let config = CheckpointConfig::default().with_checkpoint_interval(100);
        let manager = CheckpointManager::new(config, JsonCheckpointer::new());

        assert!(!manager.should_checkpoint(0));
        assert!(!manager.should_checkpoint(50));
        assert!(manager.should_checkpoint(100));
        assert!(!manager.should_checkpoint(150));
        assert!(manager.should_checkpoint(200));
    }

    #[test]
    fn test_checkpoint_manager_should_checkpoint_disabled() {
        let config = CheckpointConfig::default().with_checkpoint_interval(0);
        let manager = CheckpointManager::new(config, JsonCheckpointer::new());

        assert!(!manager.should_checkpoint(0));
        assert!(!manager.should_checkpoint(100));
        assert!(!manager.should_checkpoint(1000));
    }

    #[test]
    fn test_checkpoint_manager_list_checkpoints() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path());
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        // Save checkpoints
        for step in [300, 100, 200] {
            let state = ModelState::new(step);
            manager.save(&state).unwrap();
        }

        let checkpoints = manager.list_checkpoints();
        assert_eq!(checkpoints.len(), 3);

        // Should be sorted by step
        assert!(checkpoints[0].to_str().unwrap().contains("100"));
        assert!(checkpoints[1].to_str().unwrap().contains("200"));
        assert!(checkpoints[2].to_str().unwrap().contains("300"));
    }

    #[test]
    fn test_checkpoint_manager_initialize() {
        let dir = tempdir().unwrap();

        // First, create some checkpoints
        {
            let config = CheckpointConfig::new(dir.path());
            let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());
            for step in [100, 200, 300] {
                let state = ModelState::new(step);
                manager.save(&state).unwrap();
            }
        }

        // Now create a new manager and initialize from existing files
        let config = CheckpointConfig::new(dir.path());
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());
        manager.initialize().unwrap();

        assert_eq!(manager.checkpoint_count(), 3);
    }
}
