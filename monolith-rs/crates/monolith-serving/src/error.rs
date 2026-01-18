//! Error types for the monolith-serving crate.
//!
//! This module defines all error types used throughout the serving infrastructure.

use thiserror::Error;

/// Result type alias for serving operations.
pub type ServingResult<T> = Result<T, ServingError>;

/// Errors that can occur in the serving infrastructure.
#[derive(Debug, Error)]
pub enum ServingError {
    /// Model loading failed.
    #[error("Failed to load model: {0}")]
    ModelLoadError(String),

    /// No model is currently loaded.
    #[error("No model is currently loaded")]
    ModelNotLoaded,

    /// Server error.
    #[error("Server error: {0}")]
    ServerError(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Not connected to parameter servers.
    #[error("Not connected to parameter servers")]
    NotConnected,

    /// Parameter sync failed.
    #[error("Parameter sync failed: {0}")]
    SyncError(String),

    /// Prediction failed.
    #[error("Prediction failed: {0}")]
    PredictionError(String),

    /// Embedding lookup failed.
    #[error("Embedding lookup failed: {0}")]
    EmbeddingError(String),

    /// Request timeout.
    #[error("Request timed out after {0} ms")]
    Timeout(u64),

    /// Invalid request.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// gRPC error.
    #[error("gRPC error: {0}")]
    GrpcError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Internal error.
    #[error("Internal error: {0}")]
    Internal(String),
}

impl ServingError {
    /// Create a model load error.
    pub fn model_load(msg: impl Into<String>) -> Self {
        Self::ModelLoadError(msg.into())
    }

    /// Create a server error.
    pub fn server(msg: impl Into<String>) -> Self {
        Self::ServerError(msg.into())
    }

    /// Create a config error.
    pub fn config(msg: impl Into<String>) -> Self {
        Self::ConfigError(msg.into())
    }

    /// Create a sync error.
    pub fn sync(msg: impl Into<String>) -> Self {
        Self::SyncError(msg.into())
    }

    /// Create a prediction error.
    pub fn prediction(msg: impl Into<String>) -> Self {
        Self::PredictionError(msg.into())
    }

    /// Create an embedding error.
    pub fn embedding(msg: impl Into<String>) -> Self {
        Self::EmbeddingError(msg.into())
    }

    /// Create an invalid request error.
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self::InvalidRequest(msg.into())
    }

    /// Create a timeout error.
    pub fn timeout(ms: u64) -> Self {
        Self::Timeout(ms)
    }

    /// Create an internal error.
    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }

    /// Check if this is a retriable error.
    pub fn is_retriable(&self) -> bool {
        matches!(
            self,
            Self::NotConnected | Self::SyncError(_) | Self::Timeout(_) | Self::GrpcError(_)
        )
    }

    /// Check if this is a client error (bad request).
    pub fn is_client_error(&self) -> bool {
        matches!(self, Self::InvalidRequest(_))
    }

    /// Check if this is a server error.
    pub fn is_server_error(&self) -> bool {
        matches!(
            self,
            Self::ServerError(_)
                | Self::ModelLoadError(_)
                | Self::ModelNotLoaded
                | Self::Internal(_)
        )
    }
}

#[cfg(feature = "grpc")]
impl From<tonic::Status> for ServingError {
    fn from(status: tonic::Status) -> Self {
        Self::GrpcError(status.message().to_string())
    }
}

impl From<candle_core::Error> for ServingError {
    fn from(err: candle_core::Error) -> Self {
        ServingError::PredictionError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ServingError::ModelLoadError("test error".to_string());
        assert_eq!(err.to_string(), "Failed to load model: test error");

        let err = ServingError::ModelNotLoaded;
        assert_eq!(err.to_string(), "No model is currently loaded");

        let err = ServingError::Timeout(5000);
        assert_eq!(err.to_string(), "Request timed out after 5000 ms");
    }

    #[test]
    fn test_error_constructors() {
        let err = ServingError::model_load("load failed");
        assert!(matches!(err, ServingError::ModelLoadError(_)));

        let err = ServingError::server("server failed");
        assert!(matches!(err, ServingError::ServerError(_)));

        let err = ServingError::config("config invalid");
        assert!(matches!(err, ServingError::ConfigError(_)));

        let err = ServingError::timeout(1000);
        assert!(matches!(err, ServingError::Timeout(1000)));
    }

    #[test]
    fn test_is_retriable() {
        assert!(ServingError::NotConnected.is_retriable());
        assert!(ServingError::SyncError("test".to_string()).is_retriable());
        assert!(ServingError::Timeout(1000).is_retriable());
        assert!(ServingError::GrpcError("test".to_string()).is_retriable());

        assert!(!ServingError::ModelNotLoaded.is_retriable());
        assert!(!ServingError::InvalidRequest("test".to_string()).is_retriable());
    }

    #[test]
    fn test_is_client_error() {
        assert!(ServingError::InvalidRequest("bad request".to_string()).is_client_error());
        assert!(!ServingError::ServerError("internal".to_string()).is_client_error());
    }

    #[test]
    fn test_is_server_error() {
        assert!(ServingError::ServerError("error".to_string()).is_server_error());
        assert!(ServingError::ModelNotLoaded.is_server_error());
        assert!(ServingError::ModelLoadError("error".to_string()).is_server_error());
        assert!(ServingError::Internal("error".to_string()).is_server_error());

        assert!(!ServingError::InvalidRequest("bad".to_string()).is_server_error());
        assert!(!ServingError::NotConnected.is_server_error());
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let serving_err: ServingError = io_err.into();
        assert!(matches!(serving_err, ServingError::IoError(_)));
    }
}
