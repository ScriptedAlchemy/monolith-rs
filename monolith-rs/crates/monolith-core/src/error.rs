//! Error types for the Monolith core library.
//!
//! This module defines the error types used throughout the monolith-core crate,
//! providing structured error handling with detailed context.

use thiserror::Error;

use crate::fid::{Fid, SlotId};

/// The main error type for monolith-core operations.
#[derive(Debug, Error)]
pub enum MonolithError {
    /// Error when a slot ID is invalid or out of range.
    #[error("Invalid slot ID: {slot_id}")]
    InvalidSlotId {
        /// The invalid slot ID that was provided.
        slot_id: SlotId,
    },

    /// Error when a feature ID is invalid.
    #[error("Invalid feature ID: {fid}")]
    InvalidFid {
        /// The invalid feature ID that was provided.
        fid: Fid,
    },

    /// Error when an embedding dimension is invalid.
    #[error("Invalid embedding dimension: expected {expected}, got {actual}")]
    InvalidDimension {
        /// The expected dimension.
        expected: usize,
        /// The actual dimension that was provided.
        actual: usize,
    },

    /// Error when a feature slot is not found.
    #[error("Feature slot not found: {slot_id}")]
    SlotNotFound {
        /// The slot ID that was not found.
        slot_id: SlotId,
    },

    /// Error when a feature slice is out of bounds.
    #[error("Feature slice out of bounds: offset {offset}, length {length}, total {total}")]
    SliceOutOfBounds {
        /// The starting offset of the slice.
        offset: usize,
        /// The requested length of the slice.
        length: usize,
        /// The total available length.
        total: usize,
    },

    /// Error during configuration parsing or validation.
    #[error("Configuration error: {message}")]
    ConfigError {
        /// A description of the configuration error.
        message: String,
    },

    /// Error when an initializer configuration is invalid.
    #[error("Invalid initializer configuration: {message}")]
    InvalidInitializer {
        /// A description of why the initializer is invalid.
        message: String,
    },

    /// Error when serialization or deserialization fails.
    #[error("Serialization error: {message}")]
    SerializationError {
        /// A description of the serialization error.
        message: String,
    },

    /// Generic internal error with a message.
    #[error("Internal error: {message}")]
    InternalError {
        /// A description of the internal error.
        message: String,
    },
}

/// A specialized Result type for monolith-core operations.
pub type Result<T> = std::result::Result<T, MonolithError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = MonolithError::InvalidSlotId { slot_id: 42 };
        assert_eq!(err.to_string(), "Invalid slot ID: 42");

        let err = MonolithError::InvalidFid { fid: 12345 };
        assert_eq!(err.to_string(), "Invalid feature ID: 12345");

        let err = MonolithError::InvalidDimension {
            expected: 64,
            actual: 32,
        };
        assert_eq!(
            err.to_string(),
            "Invalid embedding dimension: expected 64, got 32"
        );

        let err = MonolithError::SlotNotFound { slot_id: 10 };
        assert_eq!(err.to_string(), "Feature slot not found: 10");

        let err = MonolithError::SliceOutOfBounds {
            offset: 5,
            length: 10,
            total: 8,
        };
        assert_eq!(
            err.to_string(),
            "Feature slice out of bounds: offset 5, length 10, total 8"
        );

        let err = MonolithError::ConfigError {
            message: "missing field".to_string(),
        };
        assert_eq!(err.to_string(), "Configuration error: missing field");
    }

    #[test]
    fn test_result_type() {
        fn success_fn() -> Result<i32> {
            Ok(42)
        }

        fn error_fn() -> Result<i32> {
            Err(MonolithError::InternalError {
                message: "test error".to_string(),
            })
        }

        assert!(success_fn().is_ok());
        assert!(error_fn().is_err());
    }
}
