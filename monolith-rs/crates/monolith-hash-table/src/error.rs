//! Error types for hash table operations.

use thiserror::Error;

/// Errors that can occur during hash table operations.
#[derive(Error, Debug)]
pub enum HashTableError {
    /// The embedding dimension doesn't match the table configuration.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected embedding dimension.
        expected: usize,
        /// Actual embedding dimension provided.
        actual: usize,
    },

    /// The number of IDs doesn't match the number of embeddings.
    #[error("id count ({id_count}) doesn't match embedding count ({embedding_count})")]
    CountMismatch {
        /// Number of IDs provided.
        id_count: usize,
        /// Number of embeddings provided.
        embedding_count: usize,
    },

    /// The hash table is full and cannot accept more entries.
    #[error("hash table is full (capacity: {capacity})")]
    TableFull {
        /// Maximum capacity of the table.
        capacity: usize,
    },

    /// The requested ID was not found in the table.
    #[error("id {id} not found")]
    IdNotFound {
        /// The ID that was not found.
        id: i64,
    },

    /// An invalid shard index was specified.
    #[error("invalid shard index: {index} (num_shards: {num_shards})")]
    InvalidShardIndex {
        /// The invalid shard index.
        index: usize,
        /// Total number of shards.
        num_shards: usize,
    },

    /// An internal error occurred.
    #[error("internal error: {message}")]
    Internal {
        /// Description of the internal error.
        message: String,
    },
}

/// A specialized Result type for hash table operations.
pub type Result<T> = std::result::Result<T, HashTableError>;
