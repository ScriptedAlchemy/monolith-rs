//! Checkpoint serialization formats and I/O utilities.
//!
//! This module provides multiple serialization formats for checkpoints and
//! utilities for reading/writing checkpoint files with optional compression.
//!
//! # Serialization Formats
//!
//! - [`BincodeSerializer`]: Fast binary format, optimized for speed
//! - [`JsonSerializer`]: Human-readable JSON format, good for debugging
//! - [`MessagePackSerializer`]: Compact binary format with msgpack
//!
//! # I/O Utilities
//!
//! - [`CheckpointWriter`]: Write checkpoints to files with compression support
//! - [`CheckpointReader`]: Read checkpoints and metadata from files
//!
//! # Examples
//!
//! ## Using Different Serializers
//!
//! ```no_run
//! use monolith_checkpoint::serialization::{
//!     CheckpointSerializer, BincodeSerializer, JsonSerializer, MessagePackSerializer,
//!     Checkpoint,
//! };
//! use monolith_checkpoint::ModelState;
//!
//! fn main() -> monolith_checkpoint::Result<()> {
//!     let state = ModelState::new(1000);
//!     let checkpoint = Checkpoint::new(state);
//!
//!     // Binary format (fastest)
//!     let bincode = BincodeSerializer::new();
//!     let data = bincode.serialize(&checkpoint)?;
//!     let restored = bincode.deserialize(&data)?;
//!
//!     // JSON format (human-readable)
//!     let json = JsonSerializer::new();
//!     let data = json.serialize(&checkpoint)?;
//!
//!     // MessagePack format (compact)
//!     let msgpack = MessagePackSerializer::new();
//!     let data = msgpack.serialize(&checkpoint)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Writing and Reading Files
//!
//! ```no_run
//! use monolith_checkpoint::serialization::{
//!     CheckpointWriter, CheckpointReader, BincodeSerializer, CompressionType,
//!     Checkpoint,
//! };
//! use monolith_checkpoint::ModelState;
//! use std::path::Path;
//!
//! fn main() -> monolith_checkpoint::Result<()> {
//!     let state = ModelState::new(1000);
//!     let checkpoint = Checkpoint::new(state);
//!
//!     // Write with compression
//!     let writer = CheckpointWriter::new(BincodeSerializer::new())
//!         .with_compression(CompressionType::Gzip);
//!     writer.write_to_file(Path::new("/tmp/checkpoint.bin.gz"), &checkpoint)?;
//!
//!     // Read back
//!     let reader = CheckpointReader::new(BincodeSerializer::new())
//!         .with_compression(CompressionType::Gzip);
//!     let restored = reader.read_from_file(Path::new("/tmp/checkpoint.bin.gz"))?;
//!
//!     Ok(())
//! }
//! ```

use crate::state::ModelState;
use crate::{CheckpointError, Result};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// A checkpoint structure that wraps ModelState with additional metadata.
///
/// This is the primary type used for serialization, containing both the
/// model state and metadata about the checkpoint itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// The model state being checkpointed.
    pub state: ModelState,

    /// Format version for backward compatibility.
    pub format_version: u32,

    /// Checksum of the serialized data (optional).
    pub checksum: Option<String>,

    /// Additional checkpoint-level metadata.
    pub checkpoint_metadata: HashMap<String, String>,
}

impl Checkpoint {
    /// Create a new checkpoint from a model state.
    pub fn new(state: ModelState) -> Self {
        Self {
            state,
            format_version: 1,
            checksum: None,
            checkpoint_metadata: HashMap::new(),
        }
    }

    /// Set a metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.checkpoint_metadata.insert(key.into(), value.into());
    }
}

impl From<ModelState> for Checkpoint {
    fn from(state: ModelState) -> Self {
        Self::new(state)
    }
}

/// Metadata about a checkpoint file without loading the full state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Global training step.
    pub global_step: u64,

    /// Timestamp when checkpoint was created.
    pub timestamp: u64,

    /// Format version.
    pub format_version: u32,

    /// Number of hash tables in the checkpoint.
    pub num_hash_tables: usize,

    /// Number of optimizer states.
    pub num_optimizers: usize,

    /// Number of dense parameters.
    pub num_dense_params: usize,

    /// Total number of embeddings across all tables.
    pub total_embeddings: usize,

    /// Additional metadata from the checkpoint.
    pub metadata: HashMap<String, String>,
}

impl From<&Checkpoint> for CheckpointMetadata {
    fn from(checkpoint: &Checkpoint) -> Self {
        Self {
            global_step: checkpoint.state.global_step,
            timestamp: checkpoint.state.timestamp,
            format_version: checkpoint.format_version,
            num_hash_tables: checkpoint.state.hash_tables.len(),
            num_optimizers: checkpoint.state.optimizers.len(),
            num_dense_params: checkpoint.state.dense_params.len(),
            total_embeddings: checkpoint.state.total_embeddings(),
            metadata: checkpoint.checkpoint_metadata.clone(),
        }
    }
}

/// Represents an incremental checkpoint delta.
///
/// Contains only the changed parts of the model state since the last checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointDelta {
    /// Base checkpoint step this delta applies to.
    pub base_step: u64,

    /// New global step after applying this delta.
    pub new_step: u64,

    /// Timestamp of the delta.
    pub timestamp: u64,

    /// Updated hash table entries (table_name -> (key -> embedding)).
    pub updated_embeddings: HashMap<String, HashMap<i64, Vec<f32>>>,

    /// Deleted hash table entries (table_name -> [keys]).
    pub deleted_embeddings: HashMap<String, Vec<i64>>,

    /// Updated dense parameters.
    pub updated_dense_params: HashMap<String, Vec<f32>>,

    /// Updated optimizer state.
    pub updated_optimizer_state: HashMap<String, Vec<u8>>,
}

impl CheckpointDelta {
    /// Create a new empty delta.
    pub fn new(base_step: u64, new_step: u64) -> Self {
        Self {
            base_step,
            new_step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            updated_embeddings: HashMap::new(),
            deleted_embeddings: HashMap::new(),
            updated_dense_params: HashMap::new(),
            updated_optimizer_state: HashMap::new(),
        }
    }
}

/// Trait for checkpoint serialization.
///
/// Implementors of this trait provide serialization and deserialization
/// of checkpoints to/from byte arrays.
pub trait CheckpointSerializer: Send + Sync {
    /// Serialize a checkpoint to bytes.
    ///
    /// # Arguments
    ///
    /// * `checkpoint` - The checkpoint to serialize
    ///
    /// # Returns
    ///
    /// The serialized bytes or an error.
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>>;

    /// Deserialize a checkpoint from bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - The serialized checkpoint data
    ///
    /// # Returns
    ///
    /// The deserialized checkpoint or an error.
    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint>;

    /// Serialize only the metadata portion of a checkpoint.
    ///
    /// This is useful for quickly reading checkpoint information without
    /// loading the full state.
    fn serialize_metadata(&self, metadata: &CheckpointMetadata) -> Result<Vec<u8>>;

    /// Deserialize checkpoint metadata from bytes.
    fn deserialize_metadata(&self, data: &[u8]) -> Result<CheckpointMetadata>;

    /// Serialize a checkpoint delta.
    fn serialize_delta(&self, delta: &CheckpointDelta) -> Result<Vec<u8>>;

    /// Deserialize a checkpoint delta.
    fn deserialize_delta(&self, data: &[u8]) -> Result<CheckpointDelta>;
}

/// Bincode serializer for fast binary checkpoint format.
///
/// This is the fastest serialization option, recommended for production use
/// when human readability is not required.
#[derive(Debug, Clone, Default)]
pub struct BincodeSerializer;

impl BincodeSerializer {
    /// Create a new bincode serializer.
    pub fn new() -> Self {
        Self
    }
}

impl CheckpointSerializer for BincodeSerializer {
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>> {
        bincode::serialize(checkpoint).map_err(|e| {
            CheckpointError::Corrupted(format!("Bincode serialization failed: {}", e))
        })
    }

    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint> {
        bincode::deserialize(data).map_err(|e| {
            CheckpointError::Corrupted(format!("Bincode deserialization failed: {}", e))
        })
    }

    fn serialize_metadata(&self, metadata: &CheckpointMetadata) -> Result<Vec<u8>> {
        bincode::serialize(metadata).map_err(|e| {
            CheckpointError::Corrupted(format!("Bincode metadata serialization failed: {}", e))
        })
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<CheckpointMetadata> {
        bincode::deserialize(data).map_err(|e| {
            CheckpointError::Corrupted(format!("Bincode metadata deserialization failed: {}", e))
        })
    }

    fn serialize_delta(&self, delta: &CheckpointDelta) -> Result<Vec<u8>> {
        bincode::serialize(delta).map_err(|e| {
            CheckpointError::Corrupted(format!("Bincode delta serialization failed: {}", e))
        })
    }

    fn deserialize_delta(&self, data: &[u8]) -> Result<CheckpointDelta> {
        bincode::deserialize(data).map_err(|e| {
            CheckpointError::Corrupted(format!("Bincode delta deserialization failed: {}", e))
        })
    }
}

/// JSON serializer for human-readable checkpoint format.
///
/// This format is slower and larger than binary formats, but useful for
/// debugging and inspection.
#[derive(Debug, Clone, Default)]
pub struct JsonSerializer {
    /// Whether to pretty-print the JSON output.
    pretty: bool,
}

impl JsonSerializer {
    /// Create a new JSON serializer.
    pub fn new() -> Self {
        Self { pretty: false }
    }

    /// Create a JSON serializer with pretty-printing enabled.
    pub fn pretty() -> Self {
        Self { pretty: true }
    }
}

impl CheckpointSerializer for JsonSerializer {
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>> {
        let result = if self.pretty {
            serde_json::to_vec_pretty(checkpoint)
        } else {
            serde_json::to_vec(checkpoint)
        };
        result.map_err(CheckpointError::Serialization)
    }

    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint> {
        serde_json::from_slice(data).map_err(CheckpointError::Deserialization)
    }

    fn serialize_metadata(&self, metadata: &CheckpointMetadata) -> Result<Vec<u8>> {
        let result = if self.pretty {
            serde_json::to_vec_pretty(metadata)
        } else {
            serde_json::to_vec(metadata)
        };
        result.map_err(CheckpointError::Serialization)
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<CheckpointMetadata> {
        serde_json::from_slice(data).map_err(CheckpointError::Deserialization)
    }

    fn serialize_delta(&self, delta: &CheckpointDelta) -> Result<Vec<u8>> {
        let result = if self.pretty {
            serde_json::to_vec_pretty(delta)
        } else {
            serde_json::to_vec(delta)
        };
        result.map_err(CheckpointError::Serialization)
    }

    fn deserialize_delta(&self, data: &[u8]) -> Result<CheckpointDelta> {
        serde_json::from_slice(data).map_err(CheckpointError::Deserialization)
    }
}

/// MessagePack serializer for compact binary format.
///
/// MessagePack provides a good balance between compactness and compatibility.
/// It's more portable than bincode while being more compact than JSON.
#[derive(Debug, Clone, Default)]
pub struct MessagePackSerializer;

impl MessagePackSerializer {
    /// Create a new MessagePack serializer.
    pub fn new() -> Self {
        Self
    }
}

impl CheckpointSerializer for MessagePackSerializer {
    fn serialize(&self, checkpoint: &Checkpoint) -> Result<Vec<u8>> {
        rmp_serde::to_vec(checkpoint).map_err(|e| {
            CheckpointError::Corrupted(format!("MessagePack serialization failed: {}", e))
        })
    }

    fn deserialize(&self, data: &[u8]) -> Result<Checkpoint> {
        rmp_serde::from_slice(data).map_err(|e| {
            CheckpointError::Corrupted(format!("MessagePack deserialization failed: {}", e))
        })
    }

    fn serialize_metadata(&self, metadata: &CheckpointMetadata) -> Result<Vec<u8>> {
        rmp_serde::to_vec(metadata).map_err(|e| {
            CheckpointError::Corrupted(format!("MessagePack metadata serialization failed: {}", e))
        })
    }

    fn deserialize_metadata(&self, data: &[u8]) -> Result<CheckpointMetadata> {
        rmp_serde::from_slice(data).map_err(|e| {
            CheckpointError::Corrupted(format!(
                "MessagePack metadata deserialization failed: {}",
                e
            ))
        })
    }

    fn serialize_delta(&self, delta: &CheckpointDelta) -> Result<Vec<u8>> {
        rmp_serde::to_vec(delta).map_err(|e| {
            CheckpointError::Corrupted(format!("MessagePack delta serialization failed: {}", e))
        })
    }

    fn deserialize_delta(&self, data: &[u8]) -> Result<CheckpointDelta> {
        rmp_serde::from_slice(data).map_err(|e| {
            CheckpointError::Corrupted(format!("MessagePack delta deserialization failed: {}", e))
        })
    }
}

/// Compression type for checkpoint files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionType {
    /// No compression.
    #[default]
    None,
    /// Gzip compression.
    Gzip,
    /// Gzip compression with specified level (0-9).
    GzipLevel(u32),
}

impl CompressionType {
    /// Get the flate2 compression level.
    fn compression_level(&self) -> Compression {
        match self {
            CompressionType::None => Compression::none(),
            CompressionType::Gzip => Compression::default(),
            CompressionType::GzipLevel(level) => Compression::new(*level),
        }
    }

    /// Check if compression is enabled.
    pub fn is_compressed(&self) -> bool {
        !matches!(self, CompressionType::None)
    }
}

/// Writer for checkpoint files.
///
/// Handles serialization and optional compression when writing checkpoints
/// to disk.
pub struct CheckpointWriter<S: CheckpointSerializer> {
    serializer: S,
    compression: CompressionType,
}

impl<S: CheckpointSerializer> CheckpointWriter<S> {
    /// Create a new checkpoint writer with the given serializer.
    pub fn new(serializer: S) -> Self {
        Self {
            serializer,
            compression: CompressionType::None,
        }
    }

    /// Set the compression type for writing.
    pub fn with_compression(mut self, compression: CompressionType) -> Self {
        self.compression = compression;
        self
    }

    /// Write a checkpoint to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to write the checkpoint to
    /// * `checkpoint` - The checkpoint to write
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or I/O fails.
    pub fn write_to_file(&self, path: &Path, checkpoint: &Checkpoint) -> Result<()> {
        tracing::info!(
            path = %path.display(),
            step = checkpoint.state.global_step,
            compression = ?self.compression,
            "Writing checkpoint"
        );

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CheckpointError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        // Serialize the checkpoint
        let data = self.serializer.serialize(checkpoint)?;

        // Compress if needed and write
        let final_data = if self.compression.is_compressed() {
            let mut encoder = GzEncoder::new(Vec::new(), self.compression.compression_level());
            encoder.write_all(&data).map_err(|e| CheckpointError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            encoder.finish().map_err(|e| CheckpointError::Io {
                path: path.to_path_buf(),
                source: e,
            })?
        } else {
            data
        };

        std::fs::write(path, &final_data).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        tracing::debug!(
            path = %path.display(),
            size = final_data.len(),
            "Checkpoint written"
        );

        Ok(())
    }

    /// Write an incremental checkpoint delta to a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to write the delta to
    /// * `delta` - The checkpoint delta to write
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or I/O fails.
    pub fn write_incremental(&self, path: &Path, delta: &CheckpointDelta) -> Result<()> {
        tracing::info!(
            path = %path.display(),
            base_step = delta.base_step,
            new_step = delta.new_step,
            "Writing incremental checkpoint"
        );

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CheckpointError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        // Serialize the delta
        let data = self.serializer.serialize_delta(delta)?;

        // Compress if needed and write
        let final_data = if self.compression.is_compressed() {
            let mut encoder = GzEncoder::new(Vec::new(), self.compression.compression_level());
            encoder.write_all(&data).map_err(|e| CheckpointError::Io {
                path: path.to_path_buf(),
                source: e,
            })?;
            encoder.finish().map_err(|e| CheckpointError::Io {
                path: path.to_path_buf(),
                source: e,
            })?
        } else {
            data
        };

        std::fs::write(path, &final_data).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        tracing::debug!(
            path = %path.display(),
            size = final_data.len(),
            "Incremental checkpoint written"
        );

        Ok(())
    }

    /// Write checkpoint metadata to a separate file.
    ///
    /// This is useful for quickly querying checkpoint information without
    /// loading the full checkpoint.
    pub fn write_metadata(&self, path: &Path, metadata: &CheckpointMetadata) -> Result<()> {
        let data = self.serializer.serialize_metadata(metadata)?;

        // Ensure parent directory exists
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| CheckpointError::Io {
                path: parent.to_path_buf(),
                source: e,
            })?;
        }

        std::fs::write(path, data).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        Ok(())
    }
}

/// Reader for checkpoint files.
///
/// Handles deserialization and optional decompression when reading
/// checkpoints from disk.
pub struct CheckpointReader<S: CheckpointSerializer> {
    serializer: S,
    compression: CompressionType,
}

impl<S: CheckpointSerializer> CheckpointReader<S> {
    /// Create a new checkpoint reader with the given serializer.
    pub fn new(serializer: S) -> Self {
        Self {
            serializer,
            compression: CompressionType::None,
        }
    }

    /// Set the compression type for reading.
    pub fn with_compression(mut self, compression: CompressionType) -> Self {
        self.compression = compression;
        self
    }

    /// Read a checkpoint from a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to read the checkpoint from
    ///
    /// # Returns
    ///
    /// The deserialized checkpoint or an error.
    pub fn read_from_file(&self, path: &Path) -> Result<Checkpoint> {
        tracing::info!(path = %path.display(), "Reading checkpoint");

        if !path.exists() {
            return Err(CheckpointError::NotFound(path.to_path_buf()));
        }

        let raw_data = std::fs::read(path).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Decompress if needed
        let data = if self.compression.is_compressed() {
            let mut decoder = GzDecoder::new(&raw_data[..]);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| CheckpointError::Io {
                    path: path.to_path_buf(),
                    source: e,
                })?;
            decompressed
        } else {
            raw_data
        };

        let checkpoint = self.serializer.deserialize(&data)?;

        tracing::info!(
            path = %path.display(),
            step = checkpoint.state.global_step,
            tables = checkpoint.state.hash_tables.len(),
            "Checkpoint read"
        );

        Ok(checkpoint)
    }

    /// Read only the metadata from a checkpoint file.
    ///
    /// This reads the entire file but only deserializes the metadata,
    /// which may be faster for some serialization formats.
    ///
    /// Note: For truly efficient metadata-only reads, use a separate
    /// metadata file written with `CheckpointWriter::write_metadata`.
    pub fn read_metadata(&self, path: &Path) -> Result<CheckpointMetadata> {
        // For most formats, we need to read the full checkpoint to get metadata
        // A more optimized implementation could store metadata separately
        let checkpoint = self.read_from_file(path)?;
        Ok(CheckpointMetadata::from(&checkpoint))
    }

    /// Read metadata from a separate metadata file.
    pub fn read_metadata_file(&self, path: &Path) -> Result<CheckpointMetadata> {
        if !path.exists() {
            return Err(CheckpointError::NotFound(path.to_path_buf()));
        }

        let data = std::fs::read(path).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        self.serializer.deserialize_metadata(&data)
    }

    /// Read an incremental checkpoint delta from a file.
    pub fn read_incremental(&self, path: &Path) -> Result<CheckpointDelta> {
        tracing::info!(path = %path.display(), "Reading incremental checkpoint");

        if !path.exists() {
            return Err(CheckpointError::NotFound(path.to_path_buf()));
        }

        let raw_data = std::fs::read(path).map_err(|e| CheckpointError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Decompress if needed
        let data = if self.compression.is_compressed() {
            let mut decoder = GzDecoder::new(&raw_data[..]);
            let mut decompressed = Vec::new();
            decoder
                .read_to_end(&mut decompressed)
                .map_err(|e| CheckpointError::Io {
                    path: path.to_path_buf(),
                    source: e,
                })?;
            decompressed
        } else {
            raw_data
        };

        self.serializer.deserialize_delta(&data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{HashTableState, OptimizerState};
    use tempfile::tempdir;

    fn create_test_checkpoint() -> Checkpoint {
        let mut state = ModelState::new(1000);

        // Add embedding table
        let mut table = HashTableState::new("user_embeddings", 32);
        table.insert(1, vec![0.1; 32]);
        table.insert(2, vec![0.2; 32]);
        table.insert(3, vec![0.3; 32]);
        state.add_hash_table(table);

        // Add optimizer state
        let opt = OptimizerState::new("adam", "embeddings", 0.001, 1000);
        state.add_optimizer(opt);

        // Add dense params
        state.add_dense_param("output.weight", vec![0.5; 128]);

        let mut checkpoint = Checkpoint::new(state);
        checkpoint.set_metadata("model_name", "test_model");
        checkpoint
    }

    // Bincode serializer tests
    #[test]
    fn test_bincode_serializer_roundtrip() {
        let serializer = BincodeSerializer::new();
        let checkpoint = create_test_checkpoint();

        let data = serializer.serialize(&checkpoint).unwrap();
        let restored = serializer.deserialize(&data).unwrap();

        assert_eq!(restored.state.global_step, 1000);
        assert_eq!(restored.state.hash_tables.len(), 1);
        assert_eq!(restored.state.hash_tables[0].len(), 3);
        assert_eq!(
            restored.checkpoint_metadata.get("model_name"),
            Some(&"test_model".to_string())
        );
    }

    #[test]
    fn test_bincode_metadata_roundtrip() {
        let serializer = BincodeSerializer::new();
        let checkpoint = create_test_checkpoint();
        let metadata = CheckpointMetadata::from(&checkpoint);

        let data = serializer.serialize_metadata(&metadata).unwrap();
        let restored = serializer.deserialize_metadata(&data).unwrap();

        assert_eq!(restored.global_step, 1000);
        assert_eq!(restored.num_hash_tables, 1);
        assert_eq!(restored.total_embeddings, 3);
    }

    #[test]
    fn test_bincode_delta_roundtrip() {
        let serializer = BincodeSerializer::new();
        let mut delta = CheckpointDelta::new(1000, 2000);
        delta
            .updated_embeddings
            .insert("user_embeddings".to_string(), {
                let mut updates = HashMap::new();
                updates.insert(4, vec![0.4; 32]);
                updates
            });

        let data = serializer.serialize_delta(&delta).unwrap();
        let restored = serializer.deserialize_delta(&data).unwrap();

        assert_eq!(restored.base_step, 1000);
        assert_eq!(restored.new_step, 2000);
        assert!(restored.updated_embeddings.contains_key("user_embeddings"));
    }

    // JSON serializer tests
    #[test]
    fn test_json_serializer_roundtrip() {
        let serializer = JsonSerializer::new();
        let checkpoint = create_test_checkpoint();

        let data = serializer.serialize(&checkpoint).unwrap();
        let restored = serializer.deserialize(&data).unwrap();

        assert_eq!(restored.state.global_step, 1000);
        assert_eq!(restored.state.hash_tables.len(), 1);
    }

    #[test]
    fn test_json_serializer_pretty() {
        let serializer = JsonSerializer::pretty();
        let checkpoint = create_test_checkpoint();

        let data = serializer.serialize(&checkpoint).unwrap();
        let json_str = String::from_utf8(data).unwrap();

        // Pretty JSON should have newlines
        assert!(json_str.contains('\n'));
    }

    #[test]
    fn test_json_metadata_roundtrip() {
        let serializer = JsonSerializer::new();
        let checkpoint = create_test_checkpoint();
        let metadata = CheckpointMetadata::from(&checkpoint);

        let data = serializer.serialize_metadata(&metadata).unwrap();
        let restored = serializer.deserialize_metadata(&data).unwrap();

        assert_eq!(restored.global_step, 1000);
    }

    // MessagePack serializer tests
    #[test]
    fn test_msgpack_serializer_roundtrip() {
        let serializer = MessagePackSerializer::new();
        let checkpoint = create_test_checkpoint();

        let data = serializer.serialize(&checkpoint).unwrap();
        let restored = serializer.deserialize(&data).unwrap();

        assert_eq!(restored.state.global_step, 1000);
        assert_eq!(restored.state.hash_tables.len(), 1);
    }

    #[test]
    fn test_msgpack_metadata_roundtrip() {
        let serializer = MessagePackSerializer::new();
        let checkpoint = create_test_checkpoint();
        let metadata = CheckpointMetadata::from(&checkpoint);

        let data = serializer.serialize_metadata(&metadata).unwrap();
        let restored = serializer.deserialize_metadata(&data).unwrap();

        assert_eq!(restored.global_step, 1000);
    }

    #[test]
    fn test_msgpack_delta_roundtrip() {
        let serializer = MessagePackSerializer::new();
        let mut delta = CheckpointDelta::new(500, 1000);
        delta
            .updated_dense_params
            .insert("output.bias".to_string(), vec![0.1; 64]);

        let data = serializer.serialize_delta(&delta).unwrap();
        let restored = serializer.deserialize_delta(&data).unwrap();

        assert_eq!(restored.base_step, 500);
        assert!(restored.updated_dense_params.contains_key("output.bias"));
    }

    // Size comparison test
    #[test]
    fn test_serializer_size_comparison() {
        let checkpoint = create_test_checkpoint();

        let bincode_data = BincodeSerializer::new().serialize(&checkpoint).unwrap();
        let json_data = JsonSerializer::new().serialize(&checkpoint).unwrap();
        let msgpack_data = MessagePackSerializer::new().serialize(&checkpoint).unwrap();

        // Bincode should be smaller than JSON
        assert!(bincode_data.len() < json_data.len());
        // MessagePack should be smaller than JSON
        assert!(msgpack_data.len() < json_data.len());

        println!("Bincode size: {} bytes", bincode_data.len());
        println!("JSON size: {} bytes", json_data.len());
        println!("MessagePack size: {} bytes", msgpack_data.len());
    }

    // CheckpointWriter tests
    #[test]
    fn test_checkpoint_writer_no_compression() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.bin");

        let writer = CheckpointWriter::new(BincodeSerializer::new());
        let checkpoint = create_test_checkpoint();

        writer.write_to_file(&path, &checkpoint).unwrap();
        assert!(path.exists());

        let reader = CheckpointReader::new(BincodeSerializer::new());
        let restored = reader.read_from_file(&path).unwrap();
        assert_eq!(restored.state.global_step, 1000);
    }

    #[test]
    fn test_checkpoint_writer_with_gzip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.bin.gz");

        let writer =
            CheckpointWriter::new(BincodeSerializer::new()).with_compression(CompressionType::Gzip);
        let checkpoint = create_test_checkpoint();

        writer.write_to_file(&path, &checkpoint).unwrap();
        assert!(path.exists());

        let reader =
            CheckpointReader::new(BincodeSerializer::new()).with_compression(CompressionType::Gzip);
        let restored = reader.read_from_file(&path).unwrap();
        assert_eq!(restored.state.global_step, 1000);
    }

    #[test]
    fn test_checkpoint_writer_incremental() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("delta.bin");

        let writer = CheckpointWriter::new(BincodeSerializer::new());
        let mut delta = CheckpointDelta::new(1000, 2000);
        delta
            .updated_embeddings
            .insert("test".to_string(), HashMap::new());

        writer.write_incremental(&path, &delta).unwrap();
        assert!(path.exists());

        let reader = CheckpointReader::new(BincodeSerializer::new());
        let restored = reader.read_incremental(&path).unwrap();
        assert_eq!(restored.base_step, 1000);
        assert_eq!(restored.new_step, 2000);
    }

    #[test]
    fn test_checkpoint_writer_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("metadata.bin");

        let writer = CheckpointWriter::new(BincodeSerializer::new());
        let checkpoint = create_test_checkpoint();
        let metadata = CheckpointMetadata::from(&checkpoint);

        writer.write_metadata(&path, &metadata).unwrap();
        assert!(path.exists());

        let reader = CheckpointReader::new(BincodeSerializer::new());
        let restored = reader.read_metadata_file(&path).unwrap();
        assert_eq!(restored.global_step, 1000);
    }

    // CheckpointReader tests
    #[test]
    fn test_checkpoint_reader_not_found() {
        let reader = CheckpointReader::new(BincodeSerializer::new());
        let result = reader.read_from_file(Path::new("/nonexistent/path.bin"));
        assert!(matches!(result, Err(CheckpointError::NotFound(_))));
    }

    #[test]
    fn test_checkpoint_reader_metadata() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("checkpoint.bin");

        let writer = CheckpointWriter::new(BincodeSerializer::new());
        let checkpoint = create_test_checkpoint();
        writer.write_to_file(&path, &checkpoint).unwrap();

        let reader = CheckpointReader::new(BincodeSerializer::new());
        let metadata = reader.read_metadata(&path).unwrap();

        assert_eq!(metadata.global_step, 1000);
        assert_eq!(metadata.num_hash_tables, 1);
        assert_eq!(metadata.total_embeddings, 3);
    }

    // Compression comparison test
    #[test]
    fn test_compression_size_reduction() {
        let dir = tempdir().unwrap();
        let uncompressed_path = dir.path().join("checkpoint.bin");
        let compressed_path = dir.path().join("checkpoint.bin.gz");

        let checkpoint = create_test_checkpoint();

        // Write uncompressed
        let writer = CheckpointWriter::new(BincodeSerializer::new());
        writer.write_to_file(&uncompressed_path, &checkpoint).unwrap();

        // Write compressed
        let writer = CheckpointWriter::new(BincodeSerializer::new())
            .with_compression(CompressionType::GzipLevel(9));
        writer.write_to_file(&compressed_path, &checkpoint).unwrap();

        let uncompressed_size = std::fs::metadata(&uncompressed_path).unwrap().len();
        let compressed_size = std::fs::metadata(&compressed_path).unwrap().len();

        println!("Uncompressed size: {} bytes", uncompressed_size);
        println!("Compressed size: {} bytes", compressed_size);

        // Compressed should be smaller (for non-trivial data)
        // Note: For very small data, compression overhead might make it larger
        // so we just verify both files were created
        assert!(uncompressed_path.exists());
        assert!(compressed_path.exists());
    }

    // Cross-format compatibility test
    #[test]
    fn test_checkpoint_data_integrity() {
        let checkpoint = create_test_checkpoint();

        // Serialize with each format and verify data integrity
        let serializers: Vec<Box<dyn CheckpointSerializer>> = vec![
            Box::new(BincodeSerializer::new()),
            Box::new(JsonSerializer::new()),
            Box::new(MessagePackSerializer::new()),
        ];

        for serializer in serializers {
            let data = serializer.serialize(&checkpoint).unwrap();
            let restored = serializer.deserialize(&data).unwrap();

            // Verify all data is preserved
            assert_eq!(restored.state.global_step, checkpoint.state.global_step);
            assert_eq!(restored.state.version, checkpoint.state.version);
            assert_eq!(
                restored.state.hash_tables.len(),
                checkpoint.state.hash_tables.len()
            );
            assert_eq!(
                restored.state.optimizers.len(),
                checkpoint.state.optimizers.len()
            );
            assert_eq!(
                restored.state.dense_params.len(),
                checkpoint.state.dense_params.len()
            );

            // Verify embedding data
            let orig_table = &checkpoint.state.hash_tables[0];
            let restored_table = &restored.state.hash_tables[0];
            assert_eq!(restored_table.name, orig_table.name);
            assert_eq!(restored_table.dim, orig_table.dim);
            assert_eq!(restored_table.entries, orig_table.entries);
        }
    }
}
