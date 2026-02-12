//! gRPC server implementation for Monolith serving.
//!
//! This module provides the main server implementation that hosts the
//! AgentService and handles health checks.

use crate::agent_service::AgentServiceImpl;
use crate::config::ServerConfig;
use crate::error::{ServingError, ServingResult};
use crate::model_loader::ModelLoader;
use crate::parameter_sync::ParameterSyncClient;
#[cfg(feature = "grpc")]
use crate::tfserving_server::{TfServingModelServer, TfServingPredictionServer};
use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
#[cfg(feature = "grpc")]
use tonic::transport::Server as TonicServer;
use tracing::{error, info, warn};

#[cfg(feature = "grpc")]
use monolith_proto::tensorflow_serving::apis::{
    model_service_server::ModelServiceServer, prediction_service_server::PredictionServiceServer,
};

/// Server state enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerState {
    /// Server is not started
    Stopped,
    /// Server is starting up
    Starting,
    /// Server is running and ready
    Running,
    /// Server is shutting down
    ShuttingDown,
    /// Server encountered an error
    Error,
}

/// Health status of the server.
#[derive(Debug, Clone)]
pub struct HealthStatus {
    /// Overall health status
    pub healthy: bool,

    /// Server state
    pub state: ServerState,

    /// Whether the model is loaded
    pub model_loaded: bool,

    /// Whether connected to parameter servers
    pub ps_connected: bool,

    /// Server uptime in seconds
    pub uptime_secs: u64,

    /// Number of active connections
    pub active_connections: u64,

    /// Additional health details
    pub details: std::collections::HashMap<String, String>,
}

/// gRPC server for Monolith serving.
///
/// The `Server` manages the gRPC server lifecycle, including starting,
/// stopping, and health monitoring.
///
/// # Example
///
/// ```no_run
/// use monolith_serving::server::Server;
/// use monolith_serving::config::ServerConfig;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ServerConfig::builder()
///     .host("0.0.0.0")
///     .port(8080)
///     .model_path("/models/recommendation")
///     .build();
///
/// let server = Server::new(config);
/// server.start().await?;
///
/// // Server is now running...
/// // To stop:
/// server.stop().await?;
/// # Ok(())
/// # }
/// ```
pub struct Server {
    /// Server configuration
    config: ServerConfig,

    /// Current server state
    state: Arc<RwLock<ServerState>>,

    /// Model loader
    model_loader: Arc<ModelLoader>,

    /// Parameter sync client
    param_sync: Option<Arc<ParameterSyncClient>>,

    /// Agent service implementation
    agent_service: Arc<RwLock<Option<Arc<AgentServiceImpl>>>>,

    /// Server start time
    start_time: Arc<RwLock<Option<Instant>>>,

    /// Shutdown signal sender
    shutdown_tx: Arc<RwLock<Option<mpsc::Sender<()>>>>,

    /// Whether the server is running
    running: Arc<AtomicBool>,

    /// Active connection counter
    active_connections: Arc<std::sync::atomic::AtomicU64>,
}

impl Server {
    /// Create a new server with the given configuration.
    pub fn new(config: ServerConfig) -> Self {
        let model_loader = Arc::new(ModelLoader::new(config.model_loader.clone()));

        let param_sync = config
            .parameter_server
            .as_ref()
            .map(|ps_config| Arc::new(ParameterSyncClient::new(ps_config.clone())));

        Self {
            config,
            state: Arc::new(RwLock::new(ServerState::Stopped)),
            model_loader,
            param_sync,
            agent_service: Arc::new(RwLock::new(None)),
            start_time: Arc::new(RwLock::new(None)),
            shutdown_tx: Arc::new(RwLock::new(None)),
            running: Arc::new(AtomicBool::new(false)),
            active_connections: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Start the gRPC server.
    ///
    /// This method:
    /// 1. Validates the configuration
    /// 2. Loads the model
    /// 3. Connects to parameter servers (if configured)
    /// 4. Starts the gRPC server
    ///
    /// # Errors
    ///
    /// Returns an error if the server cannot be started.
    pub async fn start(&self) -> ServingResult<()> {
        // Check current state
        {
            let current_state = *self.state.read();
            if current_state == ServerState::Running {
                warn!("Server is already running");
                return Ok(());
            }
            if current_state == ServerState::Starting {
                return Err(ServingError::ServerError(
                    "Server is already starting".to_string(),
                ));
            }
        }

        // Update state to starting
        *self.state.write() = ServerState::Starting;
        info!("Starting Monolith serving server...");

        // Validate configuration
        self.config.validate().map_err(|e| {
            *self.state.write() = ServerState::Error;
            ServingError::ConfigError(e.to_string())
        })?;

        // Load the model
        info!("Loading model from: {:?}", self.config.model_path);
        if let Err(e) = self.model_loader.load(&self.config.model_path).await {
            *self.state.write() = ServerState::Error;
            error!("Failed to load model: {}", e);
            return Err(e);
        }

        // Connect to parameter servers if configured
        if let Some(ref param_sync) = self.param_sync {
            info!("Connecting to parameter servers...");
            if let Err(e) = param_sync.connect().await {
                warn!("Failed to connect to parameter servers: {}", e);
                // Don't fail startup, just warn
            }
        }

        // Create agent service
        let agent_service = Arc::new(AgentServiceImpl::new(
            Arc::clone(&self.model_loader),
            self.param_sync.clone(),
        ));
        *self.agent_service.write() = Some(agent_service);

        // Create shutdown channel
        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        *self.shutdown_tx.write() = Some(shutdown_tx);

        #[cfg(feature = "grpc")]
        {
            let predict_service = TfServingPredictionServer::new(Arc::clone(
                self.agent_service.read().as_ref().expect("agent service"),
            ));
            let model_service = TfServingModelServer::new(Arc::clone(&self.model_loader));
            let socket_addr = self.config.socket_addr().parse().map_err(|e| {
                *self.state.write() = ServerState::Error;
                ServingError::ServerError(format!("Invalid bind address: {e}"))
            })?;

            tokio::spawn(async move {
                let server = TonicServer::builder()
                    .add_service(PredictionServiceServer::new(predict_service))
                    .add_service(ModelServiceServer::new(model_service))
                    .serve_with_shutdown(socket_addr, async move {
                        let _ = shutdown_rx.recv().await;
                    });

                if let Err(e) = server.await {
                    error!("TF Serving gRPC server error: {}", e);
                }
            });
        }

        // Record start time
        *self.start_time.write() = Some(Instant::now());
        self.running.store(true, Ordering::SeqCst);
        *self.state.write() = ServerState::Running;

        let addr = self.config.socket_addr();
        info!("Server started on {}", addr);

        // In a real implementation, we would start the tonic gRPC server here
        // For now, we just set up the state and wait for shutdown

        // Spawn health check task if enabled
        if self.config.health_check_enabled {
            let state = Arc::clone(&self.state);
            let model_loader = Arc::clone(&self.model_loader);
            let interval = self.config.health_check_interval;

            tokio::spawn(async move {
                let mut interval_timer = tokio::time::interval(interval);
                loop {
                    interval_timer.tick().await;
                    if *state.read() != ServerState::Running {
                        break;
                    }
                    // Perform health check
                    if !model_loader.is_ready() {
                        warn!("Health check: model not ready");
                    }
                }
            });
        }

        Ok(())
    }

    /// Stop the gRPC server gracefully.
    ///
    /// This method initiates a graceful shutdown, allowing in-flight
    /// requests to complete before fully stopping.
    pub async fn stop(&self) -> ServingResult<()> {
        let current_state = *self.state.read();
        if current_state == ServerState::Stopped {
            return Ok(());
        }

        info!("Stopping server...");
        *self.state.write() = ServerState::ShuttingDown;

        // Send shutdown signal
        let tx = self.shutdown_tx.write().take();
        if let Some(tx) = tx {
            let _ = tx.send(()).await;
        }

        // Wait for active connections to drain (with timeout)
        let drain_timeout = std::time::Duration::from_secs(30);
        let drain_start = Instant::now();

        while self.active_connections.load(Ordering::SeqCst) > 0 {
            if drain_start.elapsed() > drain_timeout {
                warn!("Connection drain timeout, forcing shutdown");
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Disconnect from parameter servers
        if let Some(ref param_sync) = self.param_sync {
            param_sync.disconnect().await;
        }

        // Unload model
        self.model_loader.unload();

        // Clear agent service
        *self.agent_service.write() = None;

        self.running.store(false, Ordering::SeqCst);
        *self.state.write() = ServerState::Stopped;
        *self.start_time.write() = None;

        info!("Server stopped");
        Ok(())
    }

    /// Get the current health status of the server.
    pub fn health(&self) -> HealthStatus {
        let state = *self.state.read();
        let uptime_secs = self
            .start_time
            .read()
            .map(|t| t.elapsed().as_secs())
            .unwrap_or(0);

        let model_loaded = self.model_loader.is_ready();
        let ps_connected = self
            .param_sync
            .as_ref()
            .map(|ps| ps.is_connected())
            .unwrap_or(true); // True if PS is not configured

        let healthy = state == ServerState::Running && model_loaded;

        let mut details = std::collections::HashMap::new();
        details.insert("host".to_string(), self.config.host.clone());
        details.insert("port".to_string(), self.config.port.to_string());
        details.insert("workers".to_string(), self.config.num_workers.to_string());

        if let Some(model) = self.model_loader.current_model() {
            details.insert("model_version".to_string(), model.version.clone());
        }

        HealthStatus {
            healthy,
            state,
            model_loaded,
            ps_connected,
            uptime_secs,
            active_connections: self.active_connections.load(Ordering::SeqCst),
            details,
        }
    }

    /// Check if the server is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the current server state.
    pub fn state(&self) -> ServerState {
        *self.state.read()
    }

    /// Get the server configuration.
    pub fn config(&self) -> &ServerConfig {
        &self.config
    }

    /// Get the model loader.
    pub fn model_loader(&self) -> &Arc<ModelLoader> {
        &self.model_loader
    }

    /// Get the agent service (if running).
    pub fn agent_service(&self) -> Option<Arc<AgentServiceImpl>> {
        self.agent_service.read().clone()
    }

    /// Get the parameter sync client (if configured).
    pub fn param_sync(&self) -> Option<&Arc<ParameterSyncClient>> {
        self.param_sync.as_ref()
    }

    /// Reload the model.
    ///
    /// This reloads the model from disk without restarting the server.
    pub async fn reload_model(&self) -> ServingResult<()> {
        if *self.state.read() != ServerState::Running {
            return Err(ServingError::ServerError(
                "Server is not running".to_string(),
            ));
        }

        info!("Reloading model...");
        self.model_loader.reload().await?;

        // Recreate agent service with new model
        let agent_service = Arc::new(AgentServiceImpl::new(
            Arc::clone(&self.model_loader),
            self.param_sync.clone(),
        ));
        *self.agent_service.write() = Some(agent_service);

        info!("Model reloaded successfully");
        Ok(())
    }

    /// Increment the active connection counter.
    #[allow(dead_code)]
    pub(crate) fn increment_connections(&self) {
        self.active_connections.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement the active connection counter.
    #[allow(dead_code)]
    pub(crate) fn decrement_connections(&self) {
        self.active_connections.fetch_sub(1, Ordering::SeqCst);
    }
}

impl std::fmt::Debug for Server {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Server")
            .field("config", &self.config)
            .field("state", &*self.state.read())
            .field("running", &self.running.load(Ordering::SeqCst))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_config(model_path: std::path::PathBuf) -> ServerConfig {
        ServerConfig::builder()
            .host("127.0.0.1")
            .port(0) // Use port 0 for testing to avoid conflicts
            .model_path(model_path)
            .build()
    }

    #[tokio::test]
    async fn test_server_creation() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let config = test_config(model_path);
        let server = Server::new(config);

        assert_eq!(server.state(), ServerState::Stopped);
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn test_server_start_stop() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let mut config = test_config(model_path);
        config.port = 18080; // Use a specific port for this test
        config.health_check_enabled = false;

        let server = Server::new(config);

        // Start server
        server
            .start()
            .await
            .expect("server start should succeed for valid test configuration");
        assert_eq!(server.state(), ServerState::Running);
        assert!(server.is_running());

        // Stop server
        server
            .stop()
            .await
            .expect("server stop should succeed after start");
        assert_eq!(server.state(), ServerState::Stopped);
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn test_server_health() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let mut config = test_config(model_path);
        config.port = 18081;
        config.health_check_enabled = false;

        let server = Server::new(config);

        // Before starting
        let health = server.health();
        assert!(!health.healthy);
        assert_eq!(health.state, ServerState::Stopped);
        assert!(!health.model_loaded);

        // Start server
        server.start().await.unwrap();

        // After starting
        let health = server.health();
        assert!(health.healthy);
        assert_eq!(health.state, ServerState::Running);
        assert!(health.model_loaded);
        assert!(health.uptime_secs < 5);

        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_server_start_twice() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let mut config = test_config(model_path);
        config.port = 18082;
        config.health_check_enabled = false;

        let server = Server::new(config);
        server.start().await.unwrap();

        // Starting again should be ok (idempotent)
        server
            .start()
            .await
            .expect("starting a running server should be idempotent");

        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_server_invalid_config() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let mut config = test_config(model_path);
        config.port = 0; // Invalid - will fail validation

        let server = Server::new(config);
        let result = server.start().await;

        assert!(result.is_err());
        assert_eq!(server.state(), ServerState::Error);
    }

    #[tokio::test]
    async fn test_server_connection_counter() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let mut config = test_config(model_path);
        config.port = 18083;
        config.health_check_enabled = false;

        let server = Server::new(config);
        server.start().await.unwrap();

        // Simulate connections
        server.increment_connections();
        server.increment_connections();
        assert_eq!(server.health().active_connections, 2);

        server.decrement_connections();
        assert_eq!(server.health().active_connections, 1);

        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_server_reload_model() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let mut config = test_config(model_path);
        config.port = 18084;
        config.health_check_enabled = false;

        let server = Server::new(config);
        server.start().await.unwrap();

        // Reload model
        server
            .reload_model()
            .await
            .expect("reloading model should succeed while server is running");
        assert!(server.model_loader().is_ready());

        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn test_server_reload_not_running() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model");
        std::fs::create_dir_all(&model_path).unwrap();

        let config = test_config(model_path);
        let server = Server::new(config);

        // Try to reload without starting
        let result = server.reload_model().await;
        assert!(result.is_err());
    }

    #[test]
    fn test_server_state_enum() {
        assert_eq!(ServerState::Stopped, ServerState::Stopped);
        assert_ne!(ServerState::Running, ServerState::Stopped);
    }

    #[test]
    fn test_health_status() {
        let health = HealthStatus {
            healthy: true,
            state: ServerState::Running,
            model_loaded: true,
            ps_connected: true,
            uptime_secs: 100,
            active_connections: 5,
            details: std::collections::HashMap::new(),
        };

        assert!(health.healthy);
        assert_eq!(health.state, ServerState::Running);
        assert_eq!(health.uptime_secs, 100);
        assert_eq!(health.active_connections, 5);
    }
}
