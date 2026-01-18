//! Server configuration for the Monolith serving infrastructure.
//!
//! This module provides configuration structures for setting up gRPC servers,
//! model loading, and parameter synchronization.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;

/// Configuration for the gRPC serving server.
///
/// This struct contains all settings needed to configure a Monolith serving instance,
/// including network settings, model paths, and resource limits.
///
/// # Example
///
/// ```
/// use monolith_serving::config::ServerConfig;
///
/// let config = ServerConfig::builder()
///     .host("0.0.0.0")
///     .port(8080)
///     .model_path("/models/recommendation")
///     .build();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Host address to bind to (default: "0.0.0.0")
    pub host: String,

    /// Port to listen on (default: 8080)
    pub port: u16,

    /// Number of worker threads for handling requests
    pub num_workers: usize,

    /// Path to the exported model directory
    pub model_path: PathBuf,

    /// Maximum concurrent requests per connection
    pub max_concurrent_requests: usize,

    /// Request timeout duration
    pub request_timeout: Duration,

    /// Enable health check endpoint
    pub health_check_enabled: bool,

    /// Health check interval
    pub health_check_interval: Duration,

    /// Maximum message size in bytes (default: 4MB)
    pub max_message_size: usize,

    /// Parameter server configuration for syncing embeddings
    pub parameter_server: Option<ParameterServerConfig>,

    /// Model loader configuration
    pub model_loader: ModelLoaderConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            num_workers: num_cpus(),
            model_path: PathBuf::from("./model"),
            max_concurrent_requests: 100,
            request_timeout: Duration::from_secs(30),
            health_check_enabled: true,
            health_check_interval: Duration::from_secs(10),
            max_message_size: 4 * 1024 * 1024, // 4MB
            parameter_server: None,
            model_loader: ModelLoaderConfig::default(),
        }
    }
}

impl ServerConfig {
    /// Create a new configuration builder.
    pub fn builder() -> ServerConfigBuilder {
        ServerConfigBuilder::default()
    }

    /// Get the socket address string for binding.
    pub fn socket_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.port == 0 {
            return Err(ConfigError::InvalidPort);
        }
        if self.num_workers == 0 {
            return Err(ConfigError::InvalidWorkerCount);
        }
        if self.max_message_size == 0 {
            return Err(ConfigError::InvalidMessageSize);
        }
        Ok(())
    }
}

/// Builder for [`ServerConfig`].
#[derive(Debug, Default)]
pub struct ServerConfigBuilder {
    host: Option<String>,
    port: Option<u16>,
    num_workers: Option<usize>,
    model_path: Option<PathBuf>,
    max_concurrent_requests: Option<usize>,
    request_timeout: Option<Duration>,
    health_check_enabled: Option<bool>,
    health_check_interval: Option<Duration>,
    max_message_size: Option<usize>,
    parameter_server: Option<ParameterServerConfig>,
    model_loader: Option<ModelLoaderConfig>,
}

impl ServerConfigBuilder {
    /// Set the host address.
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.host = Some(host.into());
        self
    }

    /// Set the port number.
    pub fn port(mut self, port: u16) -> Self {
        self.port = Some(port);
        self
    }

    /// Set the number of worker threads.
    pub fn num_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = Some(num_workers);
        self
    }

    /// Set the model path.
    pub fn model_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Set the maximum concurrent requests.
    pub fn max_concurrent_requests(mut self, max: usize) -> Self {
        self.max_concurrent_requests = Some(max);
        self
    }

    /// Set the request timeout.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Enable or disable health check.
    pub fn health_check_enabled(mut self, enabled: bool) -> Self {
        self.health_check_enabled = Some(enabled);
        self
    }

    /// Set the health check interval.
    pub fn health_check_interval(mut self, interval: Duration) -> Self {
        self.health_check_interval = Some(interval);
        self
    }

    /// Set the maximum message size.
    pub fn max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = Some(size);
        self
    }

    /// Set the parameter server configuration.
    pub fn parameter_server(mut self, config: ParameterServerConfig) -> Self {
        self.parameter_server = Some(config);
        self
    }

    /// Set the model loader configuration.
    pub fn model_loader(mut self, config: ModelLoaderConfig) -> Self {
        self.model_loader = Some(config);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> ServerConfig {
        let default = ServerConfig::default();
        ServerConfig {
            host: self.host.unwrap_or(default.host),
            port: self.port.unwrap_or(default.port),
            num_workers: self.num_workers.unwrap_or(default.num_workers),
            model_path: self.model_path.unwrap_or(default.model_path),
            max_concurrent_requests: self
                .max_concurrent_requests
                .unwrap_or(default.max_concurrent_requests),
            request_timeout: self.request_timeout.unwrap_or(default.request_timeout),
            health_check_enabled: self
                .health_check_enabled
                .unwrap_or(default.health_check_enabled),
            health_check_interval: self
                .health_check_interval
                .unwrap_or(default.health_check_interval),
            max_message_size: self.max_message_size.unwrap_or(default.max_message_size),
            parameter_server: self.parameter_server.or(default.parameter_server),
            model_loader: self.model_loader.unwrap_or(default.model_loader),
        }
    }
}

/// Configuration for connecting to parameter servers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterServerConfig {
    /// List of parameter server addresses
    pub addresses: Vec<String>,

    /// Connection timeout
    pub connect_timeout: Duration,

    /// Request timeout for pull/push operations
    pub request_timeout: Duration,

    /// Retry count for failed operations
    pub max_retries: u32,

    /// Interval between sync operations
    pub sync_interval: Duration,

    /// Enable automatic background sync
    pub auto_sync_enabled: bool,
}

impl Default for ParameterServerConfig {
    fn default() -> Self {
        Self {
            addresses: vec!["localhost:9000".to_string()],
            connect_timeout: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            sync_interval: Duration::from_secs(60),
            auto_sync_enabled: false,
        }
    }
}

/// Configuration for model loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelLoaderConfig {
    /// Whether to load the model lazily on first request
    pub lazy_loading: bool,

    /// Whether to watch for model updates
    pub watch_for_updates: bool,

    /// Interval for checking model updates
    pub update_check_interval: Duration,

    /// Whether to preload embeddings into memory
    pub preload_embeddings: bool,

    /// Maximum memory usage for cached embeddings (in bytes)
    pub max_embedding_cache_size: usize,
}

impl Default for ModelLoaderConfig {
    fn default() -> Self {
        Self {
            lazy_loading: false,
            watch_for_updates: false,
            update_check_interval: Duration::from_secs(300),
            preload_embeddings: true,
            max_embedding_cache_size: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Configuration errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ConfigError {
    /// Invalid port number
    #[error("Invalid port number: port cannot be 0")]
    InvalidPort,

    /// Invalid worker count
    #[error("Invalid worker count: must be at least 1")]
    InvalidWorkerCount,

    /// Invalid message size
    #[error("Invalid message size: must be greater than 0")]
    InvalidMessageSize,

    /// Model path not found
    #[error("Model path not found: {0}")]
    ModelPathNotFound(PathBuf),

    /// Invalid configuration file
    #[error("Invalid configuration file: {0}")]
    InvalidConfigFile(String),
}

/// Get the number of CPUs available.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.port, 8080);
        assert!(config.num_workers > 0);
        assert!(config.health_check_enabled);
    }

    #[test]
    fn test_config_builder() {
        let config = ServerConfig::builder()
            .host("127.0.0.1")
            .port(9090)
            .num_workers(4)
            .max_concurrent_requests(50)
            .build();

        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 9090);
        assert_eq!(config.num_workers, 4);
        assert_eq!(config.max_concurrent_requests, 50);
    }

    #[test]
    fn test_socket_addr() {
        let config = ServerConfig::builder()
            .host("192.168.1.1")
            .port(8888)
            .build();

        assert_eq!(config.socket_addr(), "192.168.1.1:8888");
    }

    #[test]
    fn test_config_validation() {
        let mut config = ServerConfig::default();
        assert!(config.validate().is_ok());

        config.port = 0;
        assert!(matches!(config.validate(), Err(ConfigError::InvalidPort)));

        config.port = 8080;
        config.num_workers = 0;
        assert!(matches!(
            config.validate(),
            Err(ConfigError::InvalidWorkerCount)
        ));
    }

    #[test]
    fn test_parameter_server_config_default() {
        let config = ParameterServerConfig::default();
        assert!(!config.addresses.is_empty());
        assert_eq!(config.max_retries, 3);
        assert!(!config.auto_sync_enabled);
    }

    #[test]
    fn test_model_loader_config_default() {
        let config = ModelLoaderConfig::default();
        assert!(!config.lazy_loading);
        assert!(!config.watch_for_updates);
        assert!(config.preload_embeddings);
    }
}
