//! gRPC Serving Server Example for Monolith-rs
//!
//! This example demonstrates how to set up and run a Monolith gRPC serving server,
//! including:
//! - Loading model checkpoints
//! - Configuring the server
//! - Handling predictions via gRPC
//! - Graceful shutdown
//!
//! # Usage
//!
//! Start the server:
//! ```bash
//! cargo run --example grpc_server -- --port 50051 --model-path /path/to/model
//! ```
//!
//! With demo model (creates temporary model for testing):
//! ```bash
//! cargo run --example grpc_server -- --port 50051 --demo
//! ```

use clap::Parser;
use monolith_serving::{
    config::{ModelLoaderConfig, ParameterServerConfig, ServerConfig},
    grpc::{GrpcServerConfig, ServerType, ServingServer},
    AgentServiceImpl, ModelLoader, ParameterSyncClient, Server, ServingError, ServingResult,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::signal;
use tracing::{error, info, warn};

// ============================================================================
// Command Line Arguments
// ============================================================================

/// gRPC server for Monolith serving infrastructure
#[derive(Parser, Debug)]
#[command(name = "grpc_server")]
#[command(about = "Monolith gRPC serving server example", long_about = None)]
struct Args {
    /// Host address to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(short, long, default_value_t = 50051)]
    port: u16,

    /// Path to the model checkpoint directory
    #[arg(short, long)]
    model_path: Option<PathBuf>,

    /// Number of worker threads
    #[arg(short, long, default_value_t = 4)]
    workers: usize,

    /// Maximum concurrent requests
    #[arg(long, default_value_t = 100)]
    max_concurrent: usize,

    /// Request timeout in seconds
    #[arg(long, default_value_t = 30)]
    timeout: u64,

    /// Parameter server addresses (comma-separated)
    #[arg(long)]
    ps_addresses: Option<String>,

    /// Enable health check endpoint
    #[arg(long, default_value_t = true)]
    health_check: bool,

    /// Health check interval in seconds
    #[arg(long, default_value_t = 10)]
    health_interval: u64,

    /// Run with demo model (creates temporary model)
    #[arg(long)]
    demo: bool,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Shard ID for this instance
    #[arg(long, default_value_t = 0)]
    shard_id: i32,

    /// Replica ID for this instance
    #[arg(long, default_value_t = 0)]
    replica_id: i32,

    /// Server type (entry, ps, dense)
    #[arg(long, default_value = "entry")]
    server_type: String,
}

// ============================================================================
// Demo Model Creation
// ============================================================================

/// Create a demo model directory for testing.
fn create_demo_model() -> ServingResult<PathBuf> {
    let temp_dir = std::env::temp_dir().join("monolith_demo_model");

    // Create the directory
    std::fs::create_dir_all(&temp_dir).map_err(|e| {
        ServingError::model_load(format!("Failed to create demo model directory: {}", e))
    })?;

    // Create metadata.json
    let metadata = r#"{
        "name": "demo_recommendation_model",
        "description": "Demo model for testing the gRPC serving infrastructure",
        "version": "1.0.0",
        "export_timestamp": 1704067200,
        "global_step": 10000,
        "input_format": "example",
        "output_format": "scores"
    }"#;

    std::fs::write(temp_dir.join("metadata.json"), metadata)
        .map_err(|e| ServingError::model_load(format!("Failed to write metadata.json: {}", e)))?;

    // Create slot_config.json
    let slot_config = r#"[
        {"slot_id": 0, "dim": 64, "feature_name": "user_id", "pooled": false},
        {"slot_id": 1, "dim": 64, "feature_name": "item_id", "pooled": true},
        {"slot_id": 2, "dim": 32, "feature_name": "category", "pooled": true},
        {"slot_id": 3, "dim": 32, "feature_name": "brand", "pooled": true},
        {"slot_id": 4, "dim": 16, "feature_name": "price_bucket", "pooled": false},
        {"slot_id": 5, "dim": 128, "feature_name": "user_history", "pooled": true},
        {"slot_id": 6, "dim": 32, "feature_name": "context", "pooled": false},
        {"slot_id": 7, "dim": 16, "feature_name": "device_type", "pooled": false}
    ]"#;

    std::fs::write(temp_dir.join("slot_config.json"), slot_config).map_err(|e| {
        ServingError::model_load(format!("Failed to write slot_config.json: {}", e))
    })?;

    // Create a placeholder embedding file
    let embeddings_dir = temp_dir.join("embeddings");
    std::fs::create_dir_all(&embeddings_dir).map_err(|e| {
        ServingError::model_load(format!("Failed to create embeddings directory: {}", e))
    })?;

    info!("Created demo model at: {:?}", temp_dir);
    Ok(temp_dir)
}

// ============================================================================
// Server Configuration
// ============================================================================

/// Build server configuration from command line arguments.
fn build_server_config(args: &Args, model_path: PathBuf) -> ServerConfig {
    let mut builder = ServerConfig::builder()
        .host(&args.host)
        .port(args.port)
        .num_workers(args.workers)
        .model_path(model_path)
        .max_concurrent_requests(args.max_concurrent)
        .request_timeout(Duration::from_secs(args.timeout))
        .health_check_enabled(args.health_check)
        .health_check_interval(Duration::from_secs(args.health_interval))
        .model_loader(ModelLoaderConfig {
            lazy_loading: false,
            watch_for_updates: false,
            update_check_interval: Duration::from_secs(300),
            preload_embeddings: true,
            max_embedding_cache_size: 1024 * 1024 * 1024, // 1GB
        });

    // Configure parameter servers if provided
    if let Some(ref ps_addrs) = args.ps_addresses {
        let addresses: Vec<String> = ps_addrs
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if !addresses.is_empty() {
            builder = builder.parameter_server(ParameterServerConfig {
                addresses,
                connect_timeout: Duration::from_secs(5),
                request_timeout: Duration::from_secs(30),
                max_retries: 3,
                sync_interval: Duration::from_secs(60),
                auto_sync_enabled: true,
            });
        }
    }

    builder.build()
}

/// Build gRPC server configuration from command line arguments.
fn build_grpc_config(args: &Args) -> GrpcServerConfig {
    let server_type = match args.server_type.to_lowercase().as_str() {
        "ps" => ServerType::Ps,
        "entry" => ServerType::Entry,
        "dense" => ServerType::Dense,
        _ => {
            warn!(
                "Unknown server type '{}', defaulting to Entry",
                args.server_type
            );
            ServerType::Entry
        }
    };

    GrpcServerConfig::builder()
        .bind_address(format!("{}:{}", args.host, args.port))
        .max_connections(args.max_concurrent * 10)
        .connection_timeout(Duration::from_secs(10))
        .request_timeout(Duration::from_secs(args.timeout))
        .keepalive_interval(Duration::from_secs(30))
        .keepalive_timeout(Duration::from_secs(10))
        .max_message_size(4 * 1024 * 1024) // 4MB
        .tcp_nodelay(true)
        .server_type(server_type)
        .shard_id(args.shard_id)
        .replica_id(args.replica_id)
        .build()
}

// ============================================================================
// Server Runner
// ============================================================================

/// Run the high-level Server (wraps model loading and agent service).
async fn run_high_level_server(args: Args, model_path: PathBuf) -> ServingResult<()> {
    info!("Starting high-level Monolith serving server...");

    // Build configuration
    let config = build_server_config(&args, model_path);

    info!("Server Configuration:");
    info!("  Address: {}:{}", config.host, config.port);
    info!("  Workers: {}", config.num_workers);
    info!("  Model Path: {:?}", config.model_path);
    info!(
        "  Max Concurrent Requests: {}",
        config.max_concurrent_requests
    );
    info!("  Request Timeout: {:?}", config.request_timeout);
    info!("  Health Check: {}", config.health_check_enabled);

    if let Some(ref ps_config) = config.parameter_server {
        info!("  Parameter Servers: {:?}", ps_config.addresses);
    }

    // Create server
    let server = Arc::new(Server::new(config));

    // Start server
    server.start().await?;
    info!("Server started successfully");

    // Print health status
    let health = server.health();
    info!("Health Status:");
    info!("  Healthy: {}", health.healthy);
    info!("  Model Loaded: {}", health.model_loaded);
    info!("  PS Connected: {}", health.ps_connected);

    // Wait for CTRL+C
    info!("Server is running. Press CTRL+C to stop.");
    wait_for_shutdown_signal().await;

    // Initiate graceful shutdown
    info!("Initiating graceful shutdown...");
    if let Err(e) = server.stop().await {
        error!("Error during shutdown: {}", e);
    }

    // Wait a bit for graceful shutdown
    tokio::time::sleep(Duration::from_millis(500)).await;

    info!("Server stopped");
    Ok(())
}

/// Run the low-level gRPC ServingServer.
async fn run_grpc_server(args: Args, model_path: PathBuf) -> ServingResult<()> {
    info!("Starting gRPC ServingServer...");

    // Build configurations
    let grpc_config = build_grpc_config(&args);
    let model_loader_config = ModelLoaderConfig {
        lazy_loading: false,
        watch_for_updates: false,
        update_check_interval: Duration::from_secs(300),
        preload_embeddings: true,
        max_embedding_cache_size: 1024 * 1024 * 1024,
    };

    info!("gRPC Configuration:");
    info!("  Bind Address: {}", grpc_config.bind_address);
    info!("  Max Connections: {}", grpc_config.max_connections);
    info!("  Server Type: {:?}", grpc_config.server_type);
    info!("  Shard ID: {}", grpc_config.shard_id);
    info!("  Replica ID: {}", grpc_config.replica_id);

    // Load model
    let model_loader = Arc::new(ModelLoader::new(model_loader_config));
    info!("Loading model from: {:?}", model_path);
    model_loader.load(&model_path).await?;
    info!("Model loaded successfully");

    // Create parameter sync client if configured
    let param_sync = if let Some(ref ps_addrs) = args.ps_addresses {
        let addresses: Vec<String> = ps_addrs
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        if !addresses.is_empty() {
            let ps_config = ParameterServerConfig {
                addresses,
                connect_timeout: Duration::from_secs(5),
                request_timeout: Duration::from_secs(30),
                max_retries: 3,
                sync_interval: Duration::from_secs(60),
                auto_sync_enabled: false,
            };

            let client = Arc::new(ParameterSyncClient::new(ps_config));
            info!("Connecting to parameter servers...");
            if let Err(e) = client.connect().await {
                warn!("Failed to connect to parameter servers: {}", e);
            } else {
                info!("Connected to parameter servers");
            }
            Some(client)
        } else {
            None
        }
    } else {
        None
    };

    // Create agent service
    let agent_service = Arc::new(AgentServiceImpl::new(
        Arc::clone(&model_loader),
        param_sync.clone(),
    ));
    info!("Agent service created");

    // Create serving server
    let server = Arc::new(ServingServer::with_services(
        grpc_config.clone(),
        Some(agent_service),
        Some(model_loader),
        param_sync,
    ));

    // Register self as replica (for service discovery)
    let bind_addr = grpc_config.bind_address.clone();
    server.register_replica(grpc_config.server_type, bind_addr.clone());

    info!("ServingServer created");
    info!(
        "Registered as {:?} replica at {}",
        grpc_config.server_type, bind_addr
    );

    // Spawn server task
    let server_clone = Arc::clone(&server);
    let bind_address = grpc_config.bind_address.clone();
    let server_handle = tokio::spawn(async move {
        if let Err(e) = server_clone.serve(&bind_address).await {
            error!("Server error: {}", e);
        }
    });

    info!(
        "Server is running on {}. Press CTRL+C to stop.",
        grpc_config.bind_address
    );

    // Wait for shutdown signal
    wait_for_shutdown_signal().await;

    // Initiate shutdown
    info!("Shutting down server...");
    server.shutdown();

    // Wait for server task to complete
    let _ = tokio::time::timeout(Duration::from_secs(5), server_handle).await;

    info!("Server stopped");
    Ok(())
}

/// Wait for CTRL+C or termination signal.
async fn wait_for_shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install CTRL+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received CTRL+C signal");
        },
        _ = terminate => {
            info!("Received SIGTERM signal");
        },
    }
}

// ============================================================================
// Entry Point
// ============================================================================

#[tokio::main]
async fn main() {
    let args = Args::parse();

    // Initialize logging
    if args.verbose {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .init();
    } else {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();
    }

    info!("=== Monolith gRPC Server ===");

    // Determine model path
    let model_path = if args.demo {
        match create_demo_model() {
            Ok(path) => path,
            Err(e) => {
                error!("Failed to create demo model: {}", e);
                std::process::exit(1);
            }
        }
    } else if let Some(ref path) = args.model_path {
        path.clone()
    } else {
        error!("No model path specified. Use --model-path or --demo flag.");
        std::process::exit(1);
    };

    // Validate model path exists
    if !model_path.exists() {
        error!("Model path does not exist: {:?}", model_path);
        std::process::exit(1);
    }

    info!("Model Path: {:?}", model_path);

    // Run the appropriate server type
    // The high-level server provides integrated model loading and health checks
    // The low-level gRPC server provides more control over the service implementation

    let result = if args.server_type == "entry" {
        // Use high-level server for entry servers (handles full request flow)
        run_high_level_server(args, model_path).await
    } else {
        // Use low-level gRPC server for PS and Dense servers
        run_grpc_server(args, model_path).await
    };

    match result {
        Ok(()) => {
            info!("Server exited normally");
            std::process::exit(0);
        }
        Err(e) => {
            error!("Server error: {}", e);

            if e.is_server_error() {
                error!("This is a server-side error. Check logs for details.");
            }

            std::process::exit(1);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_demo_model() {
        let path = create_demo_model().expect("demo model creation should succeed");
        assert!(path.exists());
        assert!(path.join("metadata.json").exists());
        assert!(path.join("slot_config.json").exists());
        assert!(path.join("embeddings").exists());

        // Cleanup
        let _ = std::fs::remove_dir_all(&path);
    }

    #[test]
    fn test_build_server_config() {
        let args = Args {
            host: "127.0.0.1".to_string(),
            port: 8080,
            model_path: None,
            workers: 8,
            max_concurrent: 200,
            timeout: 60,
            ps_addresses: Some("ps1:9000,ps2:9000".to_string()),
            health_check: true,
            health_interval: 15,
            demo: false,
            verbose: false,
            shard_id: 1,
            replica_id: 2,
            server_type: "entry".to_string(),
        };

        let model_path = PathBuf::from("/test/model");
        let config = build_server_config(&args, model_path);

        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.num_workers, 8);
        assert_eq!(config.max_concurrent_requests, 200);
        assert!(config.health_check_enabled);
        assert!(config.parameter_server.is_some());

        let ps_config = config.parameter_server.unwrap();
        assert_eq!(ps_config.addresses.len(), 2);
    }

    #[test]
    fn test_build_grpc_config() {
        let args = Args {
            host: "0.0.0.0".to_string(),
            port: 50051,
            model_path: None,
            workers: 4,
            max_concurrent: 100,
            timeout: 30,
            ps_addresses: None,
            health_check: true,
            health_interval: 10,
            demo: false,
            verbose: false,
            shard_id: 0,
            replica_id: 1,
            server_type: "ps".to_string(),
        };

        let config = build_grpc_config(&args);

        assert_eq!(config.bind_address, "0.0.0.0:50051");
        assert_eq!(config.max_connections, 1000);
        assert_eq!(config.server_type, ServerType::Ps);
        assert_eq!(config.shard_id, 0);
        assert_eq!(config.replica_id, 1);
    }

    #[test]
    fn test_server_type_parsing() {
        let mut args = Args {
            host: "0.0.0.0".to_string(),
            port: 50051,
            model_path: None,
            workers: 4,
            max_concurrent: 100,
            timeout: 30,
            ps_addresses: None,
            health_check: true,
            health_interval: 10,
            demo: false,
            verbose: false,
            shard_id: 0,
            replica_id: 0,
            server_type: "entry".to_string(),
        };

        let config = build_grpc_config(&args);
        assert_eq!(config.server_type, ServerType::Entry);

        args.server_type = "ps".to_string();
        let config = build_grpc_config(&args);
        assert_eq!(config.server_type, ServerType::Ps);

        args.server_type = "dense".to_string();
        let config = build_grpc_config(&args);
        assert_eq!(config.server_type, ServerType::Dense);

        // Unknown type defaults to Entry
        args.server_type = "unknown".to_string();
        let config = build_grpc_config(&args);
        assert_eq!(config.server_type, ServerType::Entry);
    }

    #[tokio::test]
    async fn test_model_loader() {
        // Create demo model
        let model_path = create_demo_model().unwrap();

        // Load model
        let config = ModelLoaderConfig::default();
        let loader = ModelLoader::new(config);
        loader
            .load(&model_path)
            .await
            .expect("model loader should load generated demo model");
        assert!(loader.is_ready());

        // Cleanup
        let _ = std::fs::remove_dir_all(&model_path);
    }
}
