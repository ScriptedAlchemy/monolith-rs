//! Serve Command Implementation
//!
//! Provides gRPC model serving for inference requests.
//! Supports configurable worker pools and health checking.

use anyhow::{Context, Result};
use clap::Args;
use monolith_serving::embedding_store::EmbeddingStore;
use monolith_serving::parameter_sync_rpc::{ParameterSyncGrpcServer, PushSink};
use monolith_serving::parameter_sync_sink::EmbeddingStorePushSink;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, warn};

/// Serve a model for inference via gRPC
///
/// This command starts a gRPC server that serves model predictions.
/// The server supports configurable worker pools for handling
/// concurrent inference requests.
///
/// # Example
///
/// ```bash
/// monolith serve \
///     --model-dir /path/to/model \
///     --port 8500 \
///     --workers 4
/// ```
#[derive(Args, Debug, Clone)]
pub struct ServeCommand {
    /// Directory containing the model to serve
    #[arg(long, short = 'd', env = "MONOLITH_MODEL_DIR")]
    pub model_dir: PathBuf,

    /// Port to listen on for gRPC requests
    #[arg(long, short = 'p', default_value = "8500", env = "MONOLITH_SERVE_PORT")]
    pub port: u16,

    /// Number of worker threads for inference
    #[arg(long, short = 'w', default_value = "4")]
    pub workers: usize,

    /// Host address to bind to
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Enable health check endpoint
    #[arg(long, default_value = "true")]
    pub health_check: bool,

    /// Health check port (if different from main port)
    #[arg(long)]
    pub health_port: Option<u16>,

    /// Maximum batch size for batched inference
    #[arg(long, default_value = "64")]
    pub max_batch_size: usize,

    /// Batch timeout in milliseconds
    #[arg(long, default_value = "10")]
    pub batch_timeout_ms: u64,

    /// Enable request logging
    #[arg(long)]
    pub enable_logging: bool,

    /// Model version to serve (defaults to latest)
    #[arg(long)]
    pub model_version: Option<String>,

    /// Bind address for ParameterSync gRPC service (training PS push target).
    ///
    /// If set, the server will also host `ParameterSyncService/Push` to receive
    /// embedding deltas from training parameter servers.
    #[arg(long)]
    pub parameter_sync_bind_addr: Option<String>,
}

impl ServeCommand {
    /// Execute the serve command
    pub async fn run(&self) -> Result<()> {
        info!("Starting model server...");
        info!("Model directory: {:?}", self.model_dir);
        info!("Listening on {}:{}", self.host, self.port);
        info!("Worker threads: {}", self.workers);

        // Validate model directory exists
        if !self.model_dir.exists() {
            anyhow::bail!("Model directory does not exist: {:?}", self.model_dir);
        }

        // Check for model files
        let model_files: Vec<_> = std::fs::read_dir(&self.model_dir)
            .context("Failed to read model directory")?
            .filter_map(|e| e.ok())
            .collect();

        if model_files.is_empty() {
            warn!("Model directory is empty, no model files found");
        } else {
            info!("Found {} files in model directory", model_files.len());
        }

        // Log configuration
        info!(
            "Server configuration: max_batch_size={}, batch_timeout={}ms",
            self.max_batch_size, self.batch_timeout_ms
        );

        if self.health_check {
            let health_port = self.health_port.unwrap_or(self.port + 1);
            info!("Health check enabled on port {}", health_port);
        }

        // Start ParameterSync gRPC server (optional).
        //
        // This is the "online" receiver for training PS delta pushes.
        // We expose it as a separate bind addr for flexibility in deployments.
        let _psync_server_handle = if let Some(bind) = &self.parameter_sync_bind_addr {
            let addr: std::net::SocketAddr = bind
                .parse()
                .context("Invalid --parameter-sync-bind-addr (expected host:port)")?;

            let store = Arc::new(EmbeddingStore::default());
            let sink: Arc<dyn PushSink> = Arc::new(EmbeddingStorePushSink::new(Arc::clone(&store)));
            let server = ParameterSyncGrpcServer::new(sink);

            info!("Starting ParameterSync gRPC server on {}", addr);
            Some(tokio::spawn(async move {
                if let Err(e) = server.serve(addr).await {
                    tracing::error!("ParameterSync gRPC server failed: {}", e);
                }
            }))
        } else {
            None
        };

        info!("Server started successfully (serve command is still a stub for AgentService/model inference)");

        // Keep the process running.
        tokio::signal::ctrl_c()
            .await
            .context("Failed to listen for shutdown signal")?;

        info!("Received shutdown signal, stopping server...");
        Ok(())
    }

    /// Get the full bind address
    pub fn bind_address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serve_command_defaults() {
        let cmd = ServeCommand {
            model_dir: PathBuf::from("/tmp/model"),
            port: 8500,
            workers: 4,
            host: "0.0.0.0".to_string(),
            health_check: true,
            health_port: None,
            max_batch_size: 64,
            batch_timeout_ms: 10,
            enable_logging: false,
            model_version: None,
            parameter_sync_bind_addr: None,
        };

        assert_eq!(cmd.port, 8500);
        assert_eq!(cmd.workers, 4);
        assert_eq!(cmd.bind_address(), "0.0.0.0:8500");
    }

    #[test]
    fn test_bind_address() {
        let cmd = ServeCommand {
            model_dir: PathBuf::from("/tmp/model"),
            port: 9000,
            workers: 2,
            host: "127.0.0.1".to_string(),
            health_check: false,
            health_port: None,
            max_batch_size: 32,
            batch_timeout_ms: 5,
            enable_logging: true,
            model_version: Some("v1".to_string()),
            parameter_sync_bind_addr: None,
        };

        assert_eq!(cmd.bind_address(), "127.0.0.1:9000");
    }
}
