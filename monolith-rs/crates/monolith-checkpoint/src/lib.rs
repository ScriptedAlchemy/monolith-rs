//! Checkpoint serialization and model export for Monolith.
//!
//! This crate provides functionality for:
//!
//! - **Save/Restore**: Persist model state to disk and restore from checkpoints
//! - **Model Export**: Export trained models for serving inference
//!
//! # Core Components
//!
//! - [`Checkpointer`]: Trait for checkpoint serialization implementations
//! - [`CheckpointManager`]: Manages checkpoint lifecycle (save, restore, cleanup)
//! - [`ModelState`]: Complete model state representation
//! - [`ModelExporter`]: Export models for serving
//!
//! # Examples
//!
//! ## Basic Checkpointing
//!
//! ```no_run
//! use monolith_checkpoint::{
//!     CheckpointManager, CheckpointConfig, JsonCheckpointer, ModelState,
//! };
//!
//! fn main() -> monolith_checkpoint::Result<()> {
//!     // Create checkpoint manager
//!     let config = CheckpointConfig::new("/tmp/checkpoints")
//!         .with_max_to_keep(5)
//!         .with_checkpoint_interval(1000);
//!
//!     let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());
//!
//!     // Save checkpoint
//!     let state = ModelState::new(1000);
//!     manager.save(&state)?;
//!
//!     // Restore latest checkpoint
//!     let restored = manager.restore_latest()?;
//!     Ok(())
//! }
//! ```
//!
//! ## Model Export
//!
//! ```no_run
//! use monolith_checkpoint::{
//!     ModelExporter, ExportConfig, ExportFormat, ModelState,
//! };
//!
//! fn main() -> monolith_checkpoint::Result<()> {
//!     // Create exporter
//!     let config = ExportConfig::new("/tmp/exported_model")
//!         .with_format(ExportFormat::SavedModel)
//!         .with_version("1.0.0");
//!
//!     let exporter = ModelExporter::new(config);
//!
//!     // Export model
//!     let state = ModelState::new(10000);
//!     exporter.export(&state)?;
//!     Ok(())
//! }
//! ```
//!
//! # Checkpoint Formats
//!
//! The crate supports multiple checkpoint formats:
//!
//! - **JSON**: Human-readable, good for debugging
//! - **Binary**: More compact, faster to read/write
//! - **SavedModel**: Directory structure suitable for serving
//!
//! # State Management
//!
//! Model state includes:
//!
//! - **Hash Tables**: Embedding tables with feature ID to vector mappings
//! - **Dense Parameters**: Non-embedding model weights
//! - **Optimizer State**: Learning rate, moments, accumulators
//! - **Metadata**: Training step, timestamps, custom metadata

pub mod checkpointer;
pub mod export;
pub mod manager;
pub mod serialization;
pub mod state;

// Re-export main types
pub use checkpointer::{BinaryCheckpointer, Checkpointer, JsonCheckpointer};
pub use export::{ExportConfig, ExportFormat, ExportManifest, ModelExporter};
pub use manager::{CheckpointConfig, CheckpointInfo, CheckpointManager};
pub use serialization::{
    BincodeSerializer, Checkpoint, CheckpointDelta, CheckpointMetadata, CheckpointReader,
    CheckpointSerializer, CheckpointWriter, CompressionType, JsonSerializer, MessagePackSerializer,
};
pub use state::{HashTableState, ModelState, OptimizerState};

use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during checkpoint operations.
#[derive(Error, Debug)]
pub enum CheckpointError {
    /// I/O error during checkpoint operations.
    #[error("I/O error at {path}: {source}")]
    Io {
        /// Path where the error occurred.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// Checkpoint file not found.
    #[error("Checkpoint not found: {0}")]
    NotFound(PathBuf),

    /// Error during serialization.
    #[error("Serialization error: {0}")]
    Serialization(#[source] serde_json::Error),

    /// Error during deserialization.
    #[error("Deserialization error: {0}")]
    Deserialization(#[source] serde_json::Error),

    /// Checkpoint version mismatch.
    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch {
        /// Expected version.
        expected: u32,
        /// Found version.
        found: u32,
    },

    /// Corrupted checkpoint data.
    #[error("Corrupted checkpoint: {0}")]
    Corrupted(String),

    /// Invalid checkpoint configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for checkpoint operations.
pub type Result<T> = std::result::Result<T, CheckpointError>;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_end_to_end_checkpoint_workflow() {
        let dir = tempdir().unwrap();

        // Create manager
        let config = CheckpointConfig::new(dir.path()).with_max_to_keep(3);
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        // Create model state with data
        let mut state = ModelState::new(1000);

        // Add embedding table
        let mut table = HashTableState::new("item_embeddings", 64);
        table.insert(100, vec![0.1; 64]);
        table.insert(200, vec![0.2; 64]);
        table.insert(300, vec![0.3; 64]);
        state.add_hash_table(table);

        // Add optimizer state
        let mut opt = OptimizerState::new("adam", "embeddings", 0.001, 1000);
        opt.first_moments
            .insert("item_embeddings".to_string(), vec![0.0; 64]);
        state.add_optimizer(opt);

        // Add dense params
        state.add_dense_param("output.weight", vec![0.5; 256]);

        // Save checkpoint
        let info = manager.save(&state).unwrap();
        assert_eq!(info.global_step, 1000);

        // Restore and verify
        let restored = manager.restore_latest().unwrap();
        assert_eq!(restored.global_step, 1000);
        assert_eq!(restored.hash_tables.len(), 1);
        assert_eq!(restored.hash_tables[0].name, "item_embeddings");
        assert_eq!(restored.hash_tables[0].len(), 3);
        assert_eq!(restored.optimizers.len(), 1);
        assert_eq!(restored.dense_params.len(), 1);
    }

    #[test]
    fn test_end_to_end_export_workflow() {
        let checkpoint_dir = tempdir().unwrap();
        let export_dir = tempdir().unwrap();

        // Create and save checkpoint
        let config = CheckpointConfig::new(checkpoint_dir.path());
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        let mut state = ModelState::new(5000);
        let mut table = HashTableState::new("user_features", 32);
        table.insert(1, vec![1.0; 32]);
        state.add_hash_table(table);
        state.add_dense_param("classifier.weight", vec![0.1; 128]);

        manager.save(&state).unwrap();

        // Export for serving
        let export_config = ExportConfig::new(export_dir.path())
            .with_format(ExportFormat::SavedModel)
            .with_version("1.0.0");

        let exporter = ModelExporter::new(export_config);
        exporter.export(&state).unwrap();

        // Verify export
        let manifest = ModelExporter::load_manifest(export_dir.path()).unwrap();
        assert_eq!(manifest.version, "1.0.0");
        assert_eq!(manifest.global_step, 5000);

        // Load embedding table from export
        let loaded_table =
            ModelExporter::load_embedding_table(export_dir.path(), "user_features").unwrap();
        assert_eq!(loaded_table.len(), 1);
        assert_eq!(loaded_table.entries.get(&1).unwrap().len(), 32);
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let dir = tempdir().unwrap();
        let config = CheckpointConfig::new(dir.path()).with_max_to_keep(2);
        let mut manager = CheckpointManager::new(config, JsonCheckpointer::new());

        // Save multiple checkpoints
        for step in [1000, 2000, 3000, 4000] {
            let state = ModelState::new(step);
            manager.save(&state).unwrap();
        }

        // Verify only 2 are kept
        let checkpoints = manager.list_checkpoints();
        assert_eq!(checkpoints.len(), 2);

        // Verify they are the latest ones
        let restored = manager.restore_latest().unwrap();
        assert_eq!(restored.global_step, 4000);
    }

    #[test]
    fn test_error_handling() {
        let checkpointer = JsonCheckpointer::new();

        // Try to restore non-existent checkpoint
        let err = checkpointer
            .restore(std::path::Path::new("/nonexistent/path.json"))
            .expect_err("restoring a missing checkpoint path should fail");
        assert!(
            matches!(
                &err,
                CheckpointError::NotFound(path)
                    if path.to_str().is_some_and(|p| p.contains("nonexistent"))
            ),
            "expected NotFound checkpoint error containing missing path context, got: {err:?}"
        );
    }
}
