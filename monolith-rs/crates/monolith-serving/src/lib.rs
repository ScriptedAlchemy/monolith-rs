//! gRPC serving infrastructure for Monolith.
//!
//! This crate provides the serving layer for the Monolith recommendation system,
//! including gRPC service implementations, model loading, and parameter synchronization.
//!
//! # Overview
//!
//! The monolith-serving crate enables deploying trained Monolith models for online inference.
//! It provides:
//!
//! - **AgentService**: gRPC service for handling prediction requests
//! - **Server**: High-performance gRPC server with health checks
//! - **ServingServer**: gRPC server implementing the AgentService proto
//! - **ModelLoader**: Load and manage exported models for serving
//! - **ParameterSyncClient**: Sync embeddings with parameter servers
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                      gRPC Clients                           │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         Server                               │
//! │  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
//! │  │  Health Check   │  │  AgentService   │  │   Metrics    │ │
//! │  └─────────────────┘  └─────────────────┘  └──────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!              ┌───────────────┼───────────────┐
//!              ▼               ▼               ▼
//! ┌────────────────┐ ┌─────────────────┐ ┌───────────────────┐
//! │  ModelLoader   │ │ EmbeddingCache  │ │ ParameterSyncClient│
//! └────────────────┘ └─────────────────┘ └───────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    Parameter Servers                         │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```no_run
//! use monolith_serving::{Server, ServerConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Configure the server
//! let config = ServerConfig::builder()
//!     .host("0.0.0.0")
//!     .port(8080)
//!     .model_path("/models/recommendation")
//!     .build();
//!
//! // Create and start the server
//! let server = Server::new(config);
//! server.start().await?;
//!
//! // Server is now running and accepting requests...
//!
//! // To stop the server gracefully:
//! server.stop().await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Configuration
//!
//! The serving infrastructure can be configured through [`ServerConfig`]:
//!
//! ```
//! use monolith_serving::config::{ServerConfig, ParameterServerConfig, ModelLoaderConfig};
//! use std::time::Duration;
//!
//! let config = ServerConfig::builder()
//!     .host("0.0.0.0")
//!     .port(8080)
//!     .num_workers(4)
//!     .model_path("/path/to/model")
//!     .max_concurrent_requests(100)
//!     .request_timeout(Duration::from_secs(30))
//!     .health_check_enabled(true)
//!     .parameter_server(ParameterServerConfig {
//!         addresses: vec!["ps1:9000".to_string(), "ps2:9000".to_string()],
//!         ..Default::default()
//!     })
//!     .model_loader(ModelLoaderConfig {
//!         preload_embeddings: true,
//!         ..Default::default()
//!     })
//!     .build();
//! ```
//!
//! # Making Predictions
//!
//! Once the server is running, clients can make prediction requests through gRPC.
//! The [`AgentServiceImpl`] handles these requests:
//!
//! ```no_run
//! use monolith_serving::agent_service::{AgentServiceImpl, PredictRequest, FeatureInput};
//! use monolith_serving::model_loader::ModelLoader;
//! use monolith_serving::config::ModelLoaderConfig;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let model_loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
//! model_loader.load("/path/to/model").await?;
//!
//! let service = AgentServiceImpl::new(model_loader, None);
//!
//! let request = PredictRequest {
//!     request_id: "req-001".to_string(),
//!     features: vec![
//!         FeatureInput {
//!             name: "user_id".to_string(),
//!             slot_id: 0,
//!             fids: vec![12345],
//!             values: None,
//!         },
//!         FeatureInput {
//!             name: "item_ids".to_string(),
//!             slot_id: 1,
//!             fids: vec![100, 200, 300],
//!             values: Some(vec![1.0, 1.0, 1.0]),
//!         },
//!     ],
//!     return_embeddings: false,
//!     context: None,
//! };
//!
//! let response = service.predict(request).await?;
//! println!("Prediction scores: {:?}", response.scores);
//! # Ok(())
//! # }
//! ```
//!
//! # gRPC Server
//!
//! The [`ServingServer`] provides a high-level gRPC server implementing the AgentService:
//!
//! ```no_run
//! use monolith_serving::grpc::{ServingServer, GrpcServerConfig, ServerType};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = GrpcServerConfig::builder()
//!     .bind_address("0.0.0.0:50051")
//!     .max_connections(1000)
//!     .server_type(ServerType::Entry)
//!     .build();
//!
//! let server = ServingServer::new(config);
//!
//! // Register replicas for service discovery
//! server.register_replica(ServerType::Ps, "ps1:9000".to_string());
//!
//! // Start the server
//! server.serve("0.0.0.0:50051").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Parameter Synchronization
//!
//! For distributed deployments, embeddings can be synchronized with parameter servers:
//!
//! ```no_run
//! use monolith_serving::parameter_sync::ParameterSyncClient;
//! use monolith_serving::config::ParameterServerConfig;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = ParameterServerConfig {
//!     addresses: vec!["ps1:9000".to_string()],
//!     auto_sync_enabled: true,
//!     ..Default::default()
//! };
//!
//! let client = ParameterSyncClient::new(config);
//! client.connect().await?;
//!
//! // Pull embeddings for specific feature IDs
//! let embeddings = client.pull(0, &[1, 2, 3]).await?;
//!
//! // Start background sync
//! client.start_background_sync()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Features
//!
//! - `grpc` (default): Enable gRPC server and client functionality
//!
//! # Error Handling
//!
//! All operations return [`ServingResult<T>`] which wraps [`ServingError`]:
//!
//! ```
//! use monolith_serving::error::{ServingError, ServingResult};
//!
//! fn handle_error(result: ServingResult<()>) {
//!     match result {
//!         Ok(_) => println!("Success"),
//!         Err(ServingError::ModelNotLoaded) => println!("Model not loaded"),
//!         Err(ServingError::NotConnected) => println!("Not connected to PS"),
//!         Err(e) if e.is_retriable() => println!("Retriable error: {}", e),
//!         Err(e) => println!("Error: {}", e),
//!     }
//! }
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod agent_service;
pub mod config;
pub mod embedding_store;
pub mod error;
pub mod grpc;
pub mod grpc_agent;
pub mod model_loader;
pub mod parameter_sync;
pub mod parameter_sync_rpc;
pub mod parameter_sync_sink;
pub mod server;

// Re-export main types at crate root for convenience
pub use agent_service::{AgentServiceImpl, FeatureInput, PredictRequest, PredictResponse};
pub use config::{ModelLoaderConfig, ParameterServerConfig, ServerConfig};
pub use embedding_store::EmbeddingStore;
pub use error::{ServingError, ServingResult};
pub use grpc::{
    AgentService, AgentServiceGrpcImpl, GrpcServerConfig, GrpcServerConfigBuilder, GrpcServerState,
    GrpcServiceStats, ServerType, ServingServer,
};
pub use model_loader::{LoadedModel, ModelLoader};
pub use parameter_sync::{EmbeddingData, ParameterSyncClient, SyncRequest, SyncResponse};
pub use parameter_sync_rpc::{ParameterSyncGrpcServer, ParameterSyncRpcClient};
pub use parameter_sync_sink::EmbeddingStorePushSink;
pub use server::{HealthStatus, Server, ServerState};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Crate name.
pub const NAME: &str = env!("CARGO_PKG_NAME");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
        assert_eq!(NAME, "monolith-serving");
    }

    #[test]
    fn test_re_exports() {
        // Verify that all re-exported types are accessible
        let _ = ServerConfig::default();
        let _ = ParameterServerConfig::default();
        let _ = ModelLoaderConfig::default();
    }

    #[tokio::test]
    async fn test_integration_flow() {
        use tempfile::tempdir;

        // Create a test model directory
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        // Create server config
        let config = ServerConfig::builder()
            .host("127.0.0.1")
            .port(18090)
            .model_path(&model_path)
            .health_check_enabled(false)
            .build();

        // Create and start server
        let server = Server::new(config);
        server.start().await.unwrap();

        // Check health
        let health = server.health();
        assert!(health.healthy);
        assert_eq!(health.state, ServerState::Running);
        assert!(health.model_loaded);

        // Get agent service and make a prediction
        let agent_service = server.agent_service().unwrap();
        let request = PredictRequest {
            request_id: "integration-test".to_string(),
            features: vec![FeatureInput {
                name: "test".to_string(),
                slot_id: 0,
                fids: vec![1],
                values: None,
            }],
            return_embeddings: true,
            context: None,
        };

        let response = agent_service.predict(request).await.unwrap();
        assert!(response.success);
        assert!(!response.scores.is_empty());

        // Stop server
        server.stop().await.unwrap();
        assert_eq!(server.state(), ServerState::Stopped);
    }
}
