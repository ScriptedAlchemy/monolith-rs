//! Error types for the monolith-layers crate.
//!
//! This module defines error types for neural network layer operations,
//! including shape mismatches, initialization errors, and computation failures.

use thiserror::Error;

/// Error type for layer operations.
#[derive(Debug, Error)]
pub enum LayerError {
    /// Shape mismatch between expected and actual tensor shapes.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// The expected shape
        expected: Vec<usize>,
        /// The actual shape that was provided
        actual: Vec<usize>,
    },

    /// Invalid input dimension for the layer.
    #[error("Invalid input dimension: expected {expected}, got {actual}")]
    InvalidInputDimension {
        /// The expected input dimension
        expected: usize,
        /// The actual input dimension
        actual: usize,
    },

    /// Invalid output dimension for the layer.
    #[error("Invalid output dimension: expected {expected}, got {actual}")]
    InvalidOutputDimension {
        /// The expected output dimension
        expected: usize,
        /// The actual output dimension
        actual: usize,
    },

    /// Error during weight initialization.
    #[error("Initialization error: {message}")]
    InitializationError {
        /// Description of the initialization error
        message: String,
    },

    /// Error during forward pass computation.
    #[error("Forward pass error: {message}")]
    ForwardError {
        /// Description of the forward pass error
        message: String,
    },

    /// Error during backward pass computation.
    #[error("Backward pass error: {message}")]
    BackwardError {
        /// Description of the backward pass error
        message: String,
    },

    /// Layer has not been initialized with an input.
    #[error("Layer not initialized: forward pass must be called before backward pass")]
    NotInitialized,

    /// Configuration error for the layer.
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Description of the configuration error
        message: String,
    },

    /// Embedding lookup error.
    #[error("Embedding lookup error: {message}")]
    EmbeddingError {
        /// Description of the embedding error
        message: String,
    },
}

/// Result type alias for layer operations.
pub type LayerResult<T> = Result<T, LayerError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LayerError::ShapeMismatch {
            expected: vec![32, 64],
            actual: vec![32, 128],
        };
        assert!(err.to_string().contains("Shape mismatch"));

        let err = LayerError::InvalidInputDimension {
            expected: 64,
            actual: 128,
        };
        assert!(err.to_string().contains("Invalid input dimension"));

        let err = LayerError::NotInitialized;
        assert!(err.to_string().contains("not initialized"));
    }
}
