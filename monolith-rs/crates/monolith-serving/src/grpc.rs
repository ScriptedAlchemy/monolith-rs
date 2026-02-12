//! gRPC server implementation for Monolith AgentService.
//!
//! This module provides the gRPC service implementation for the AgentService
//! as defined in `agent_service.proto`, including heartbeat, replica discovery,
//! and resource monitoring.
//!
//! # Architecture
//!
//! The gRPC server is built on top of `tonic` and provides:
//! - `AgentServiceImpl`: The gRPC service implementation
//! - `ServingServer`: High-level server management
//! - `GrpcServerConfig`: Configuration for the gRPC server
//!
//! # Example
//!
//! ```no_run
//! use monolith_serving::grpc::{ServingServer, GrpcServerConfig};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let config = GrpcServerConfig::default();
//! let server = ServingServer::new(config);
//!
//! // Start the server
//! server.serve("0.0.0.0:50051").await?;
//! # Ok(())
//! # }
//! ```

use crate::agent_service::AgentServiceImpl as EmbeddingService;
use crate::error::{ServingError, ServingResult};
use crate::model_loader::ModelLoader;
use crate::parameter_sync::ParameterSyncClient;
use monolith_proto::monolith::serving::agent_service as agent_proto;
use monolith_proto::monolith::serving::agent_service::agent_service_server::{
    AgentService as AgentServiceProto, AgentServiceServer,
};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::oneshot;
use tonic::transport::Server as TonicServer;
use tonic::{Request, Response, Status};
use tracing::{debug, error, info, warn};

// ============================================================================
// Proto-generated types (normally generated from agent_service.proto)
// ============================================================================

/// Server type enumeration matching proto definition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum ServerType {
    /// Parameter Server
    Ps = 0,
    /// Entry server
    Entry = 1,
    /// Dense server
    Dense = 2,
}

impl From<i32> for ServerType {
    fn from(val: i32) -> Self {
        match val {
            0 => ServerType::Ps,
            1 => ServerType::Entry,
            2 => ServerType::Dense,
            _ => ServerType::Ps, // Default to PS
        }
    }
}

impl From<ServerType> for i32 {
    fn from(val: ServerType) -> Self {
        val as i32
    }
}

/// Address list message.
#[derive(Debug, Clone, Default)]
pub struct AddressList {
    /// List of addresses
    pub address: Vec<String>,
}

/// Request for getting replicas.
#[derive(Debug, Clone)]
pub struct GetReplicasRequest {
    /// Server type to query
    pub server_type: ServerType,
    /// Task ID
    pub task: i32,
    /// Model name
    pub model_name: String,
}

/// Response with replica addresses.
#[derive(Debug, Clone)]
pub struct GetReplicasResponse {
    /// List of replica addresses
    pub address_list: Option<AddressList>,
}

/// Request for resource information.
#[derive(Debug, Clone, Default)]
pub struct GetResourceRequest {}

/// Response with resource information.
#[derive(Debug, Clone)]
pub struct GetResourceResponse {
    /// Server address
    pub address: String,
    /// Shard ID
    pub shard_id: i32,
    /// Replica ID
    pub replica_id: i32,
    /// Memory usage in bytes
    pub memory: i64,
    /// CPU utilization (0.0 - 1.0)
    pub cpu: f32,
    /// Network utilization (0.0 - 1.0)
    pub network: f32,
    /// Work load (0.0 - 1.0)
    pub work_load: f32,
}

impl Default for GetResourceResponse {
    fn default() -> Self {
        Self {
            address: String::new(),
            shard_id: 0,
            replica_id: 0,
            memory: 0,
            cpu: 0.0,
            network: 0.0,
            work_load: 0.0,
        }
    }
}

/// Heartbeat request message.
#[derive(Debug, Clone)]
pub struct HeartBeatRequest {
    /// Server type sending the heartbeat
    pub server_type: ServerType,
}

/// Heartbeat response message.
#[derive(Debug, Clone, Default)]
pub struct HeartBeatResponse {
    /// Map of server type name to address list
    pub addresses: HashMap<String, AddressList>,
}

// ============================================================================
// Request for parameter synchronization
// ============================================================================

/// Request for syncing parameters.
#[derive(Debug, Clone)]
pub struct SyncParametersRequest {
    /// Slot ID to sync
    pub slot_id: i32,
    /// Feature IDs to sync
    pub fids: Vec<i64>,
    /// Whether to force full sync
    pub full_sync: bool,
    /// Request timestamp
    pub timestamp: u64,
}

/// Response for parameter sync.
#[derive(Debug, Clone)]
pub struct SyncParametersResponse {
    /// Whether sync was successful
    pub success: bool,
    /// Number of embeddings synced
    pub num_synced: i64,
    /// Duration of sync in milliseconds
    pub duration_ms: i64,
    /// Error message if failed
    pub error_message: Option<String>,
}

/// Module information for GetModules.
#[derive(Debug, Clone)]
pub struct ModuleInfo {
    /// Module name
    pub name: String,
    /// Module version
    pub version: String,
    /// Whether the module is loaded
    pub loaded: bool,
    /// Module-specific metadata
    pub metadata: HashMap<String, String>,
}

/// Request for getting modules.
#[derive(Debug, Clone, Default)]
pub struct GetModulesRequest {
    /// Filter by module name prefix (optional)
    pub name_prefix: Option<String>,
}

/// Response with module information.
#[derive(Debug, Clone, Default)]
pub struct GetModulesResponse {
    /// List of modules
    pub modules: Vec<ModuleInfo>,
}

// ============================================================================
// gRPC Server Configuration
// ============================================================================

/// Configuration for the gRPC server.
///
/// This struct contains all settings needed to configure the gRPC server,
/// including network settings, resource limits, and timeout configurations.
#[derive(Debug, Clone)]
pub struct GrpcServerConfig {
    /// Address to bind the server to (e.g., "0.0.0.0:50051")
    pub bind_address: String,

    /// Maximum number of concurrent connections
    pub max_connections: usize,

    /// Connection timeout duration
    pub connection_timeout: Duration,

    /// Request timeout duration
    pub request_timeout: Duration,

    /// Keep-alive interval for connections
    pub keepalive_interval: Duration,

    /// Keep-alive timeout
    pub keepalive_timeout: Duration,

    /// Maximum message size in bytes
    pub max_message_size: usize,

    /// Enable TCP nodelay
    pub tcp_nodelay: bool,

    /// Server type for this instance
    pub server_type: ServerType,

    /// Shard ID for this instance
    pub shard_id: i32,

    /// Replica ID for this instance
    pub replica_id: i32,
}

impl Default for GrpcServerConfig {
    fn default() -> Self {
        Self {
            bind_address: "0.0.0.0:50051".to_string(),
            max_connections: 1000,
            connection_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(30),
            keepalive_interval: Duration::from_secs(30),
            keepalive_timeout: Duration::from_secs(10),
            max_message_size: 4 * 1024 * 1024, // 4MB
            tcp_nodelay: true,
            server_type: ServerType::Entry,
            shard_id: 0,
            replica_id: 0,
        }
    }
}

impl GrpcServerConfig {
    /// Create a new builder for GrpcServerConfig.
    pub fn builder() -> GrpcServerConfigBuilder {
        GrpcServerConfigBuilder::default()
    }

    /// Get the socket address for binding.
    pub fn socket_addr(&self) -> Result<SocketAddr, std::net::AddrParseError> {
        self.bind_address.parse()
    }
}

/// Builder for GrpcServerConfig.
#[derive(Debug, Default)]
pub struct GrpcServerConfigBuilder {
    bind_address: Option<String>,
    max_connections: Option<usize>,
    connection_timeout: Option<Duration>,
    request_timeout: Option<Duration>,
    keepalive_interval: Option<Duration>,
    keepalive_timeout: Option<Duration>,
    max_message_size: Option<usize>,
    tcp_nodelay: Option<bool>,
    server_type: Option<ServerType>,
    shard_id: Option<i32>,
    replica_id: Option<i32>,
}

impl GrpcServerConfigBuilder {
    /// Set the bind address.
    pub fn bind_address(mut self, addr: impl Into<String>) -> Self {
        self.bind_address = Some(addr.into());
        self
    }

    /// Set maximum connections.
    pub fn max_connections(mut self, max: usize) -> Self {
        self.max_connections = Some(max);
        self
    }

    /// Set connection timeout.
    pub fn connection_timeout(mut self, timeout: Duration) -> Self {
        self.connection_timeout = Some(timeout);
        self
    }

    /// Set request timeout.
    pub fn request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = Some(timeout);
        self
    }

    /// Set keepalive interval.
    pub fn keepalive_interval(mut self, interval: Duration) -> Self {
        self.keepalive_interval = Some(interval);
        self
    }

    /// Set keepalive timeout.
    pub fn keepalive_timeout(mut self, timeout: Duration) -> Self {
        self.keepalive_timeout = Some(timeout);
        self
    }

    /// Set maximum message size.
    pub fn max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = Some(size);
        self
    }

    /// Set TCP nodelay.
    pub fn tcp_nodelay(mut self, enabled: bool) -> Self {
        self.tcp_nodelay = Some(enabled);
        self
    }

    /// Set server type.
    pub fn server_type(mut self, server_type: ServerType) -> Self {
        self.server_type = Some(server_type);
        self
    }

    /// Set shard ID.
    pub fn shard_id(mut self, id: i32) -> Self {
        self.shard_id = Some(id);
        self
    }

    /// Set replica ID.
    pub fn replica_id(mut self, id: i32) -> Self {
        self.replica_id = Some(id);
        self
    }

    /// Build the configuration.
    pub fn build(self) -> GrpcServerConfig {
        let default = GrpcServerConfig::default();
        GrpcServerConfig {
            bind_address: self.bind_address.unwrap_or(default.bind_address),
            max_connections: self.max_connections.unwrap_or(default.max_connections),
            connection_timeout: self
                .connection_timeout
                .unwrap_or(default.connection_timeout),
            request_timeout: self.request_timeout.unwrap_or(default.request_timeout),
            keepalive_interval: self
                .keepalive_interval
                .unwrap_or(default.keepalive_interval),
            keepalive_timeout: self.keepalive_timeout.unwrap_or(default.keepalive_timeout),
            max_message_size: self.max_message_size.unwrap_or(default.max_message_size),
            tcp_nodelay: self.tcp_nodelay.unwrap_or(default.tcp_nodelay),
            server_type: self.server_type.unwrap_or(default.server_type),
            shard_id: self.shard_id.unwrap_or(default.shard_id),
            replica_id: self.replica_id.unwrap_or(default.replica_id),
        }
    }
}

// ============================================================================
// AgentService gRPC Implementation
// ============================================================================

/// Statistics for the gRPC service.
#[derive(Debug, Clone, Default)]
pub struct GrpcServiceStats {
    /// Total heartbeat requests
    pub heartbeat_requests: u64,
    /// Total get replicas requests
    pub get_replicas_requests: u64,
    /// Total get resource requests
    pub get_resource_requests: u64,
    /// Total get modules requests
    pub get_modules_requests: u64,
    /// Total sync requests
    pub sync_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
}

/// Implementation of the AgentService gRPC service.
///
/// This struct implements the gRPC trait for AgentService as defined in
/// `agent_service.proto`, handling heartbeat, replica discovery, and
/// resource monitoring requests.
pub struct AgentServiceGrpcImpl {
    /// Server configuration
    config: GrpcServerConfig,

    /// Known replicas organized by server type
    replicas: Arc<RwLock<HashMap<ServerType, Vec<String>>>>,

    /// Embedding service for model-related operations
    embedding_service: Option<Arc<EmbeddingService>>,

    /// Model loader reference
    model_loader: Option<Arc<ModelLoader>>,

    /// Parameter sync client
    param_sync: Option<Arc<ParameterSyncClient>>,

    /// Service statistics
    stats: Arc<RwLock<GrpcServiceStats>>,

    /// Server start time
    start_time: Instant,

    /// Active connections counter
    active_connections: Arc<std::sync::atomic::AtomicU64>,
}

impl AgentServiceGrpcImpl {
    /// Create a new AgentService implementation.
    pub fn new(config: GrpcServerConfig) -> Self {
        Self {
            config,
            replicas: Arc::new(RwLock::new(HashMap::new())),
            embedding_service: None,
            model_loader: None,
            param_sync: None,
            stats: Arc::new(RwLock::new(GrpcServiceStats::default())),
            start_time: Instant::now(),
            active_connections: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Set the embedding service.
    pub fn with_embedding_service(mut self, service: Arc<EmbeddingService>) -> Self {
        self.embedding_service = Some(service);
        self
    }

    /// Set the model loader.
    pub fn with_model_loader(mut self, loader: Arc<ModelLoader>) -> Self {
        self.model_loader = Some(loader);
        self
    }

    /// Set the parameter sync client.
    pub fn with_param_sync(mut self, client: Arc<ParameterSyncClient>) -> Self {
        self.param_sync = Some(client);
        self
    }

    /// Register a replica for a server type.
    pub fn register_replica(&self, server_type: ServerType, address: String) {
        let mut replicas = self.replicas.write();
        let addresses = replicas.entry(server_type).or_default();
        if !addresses.contains(&address) {
            addresses.push(address.clone());
            info!("Registered replica {:?}: {}", server_type, address);
        }
    }

    /// Unregister a replica.
    pub fn unregister_replica(&self, server_type: ServerType, address: &str) {
        let mut replicas = self.replicas.write();
        if let Some(addresses) = replicas.get_mut(&server_type) {
            addresses.retain(|a| a != address);
            info!("Unregistered replica {:?}: {}", server_type, address);
        }
    }

    /// Handle a heartbeat request.
    ///
    /// Heartbeats are used for service discovery and health monitoring.
    /// The response includes all known replica addresses organized by server type.
    pub async fn heartbeat(
        &self,
        request: HeartBeatRequest,
    ) -> Result<HeartBeatResponse, ServingError> {
        let start = Instant::now();
        debug!("Heartbeat received from {:?}", request.server_type);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.heartbeat_requests += 1;
        }

        // Build response with all known addresses
        let replicas = self.replicas.read();
        let mut addresses = HashMap::new();

        for (server_type, addrs) in replicas.iter() {
            let key = match server_type {
                ServerType::Ps => "PS",
                ServerType::Entry => "ENTRY",
                ServerType::Dense => "DENSE",
            };
            addresses.insert(
                key.to_string(),
                AddressList {
                    address: addrs.clone(),
                },
            );
        }

        // Update latency stats
        let latency = start.elapsed().as_secs_f64() * 1000.0;
        self.update_latency(latency);

        debug!("Heartbeat processed in {:.2}ms", latency);

        Ok(HeartBeatResponse { addresses })
    }

    /// Handle a get replicas request.
    ///
    /// Returns the addresses of replicas for the requested server type and task.
    pub async fn get_replicas(
        &self,
        request: GetReplicasRequest,
    ) -> Result<GetReplicasResponse, ServingError> {
        let start = Instant::now();
        debug!(
            "GetReplicas request for {:?}, task={}, model={}",
            request.server_type, request.task, request.model_name
        );

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.get_replicas_requests += 1;
        }

        let replicas = self.replicas.read();
        let address_list = replicas.get(&request.server_type).map(|addrs| AddressList {
            address: addrs.clone(),
        });

        let latency = start.elapsed().as_secs_f64() * 1000.0;
        self.update_latency(latency);

        debug!("GetReplicas completed in {:.2}ms", latency);

        Ok(GetReplicasResponse { address_list })
    }

    /// Handle a get resource request.
    ///
    /// Returns resource usage information for this server instance.
    pub async fn get_resource(
        &self,
        _request: GetResourceRequest,
    ) -> Result<GetResourceResponse, ServingError> {
        let start = Instant::now();
        debug!("GetResource request");

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.get_resource_requests += 1;
        }

        // Collect resource information
        let memory = self.get_memory_usage();
        let cpu = self.get_cpu_usage();
        let work_load = self.calculate_work_load();

        let response = GetResourceResponse {
            address: self.config.bind_address.clone(),
            shard_id: self.config.shard_id,
            replica_id: self.config.replica_id,
            memory,
            cpu,
            network: 0.0, // Network monitoring not implemented
            work_load,
        };

        let latency = start.elapsed().as_secs_f64() * 1000.0;
        self.update_latency(latency);

        debug!("GetResource completed in {:.2}ms", latency);

        Ok(response)
    }

    /// Handle a get modules request.
    ///
    /// Returns information about loaded modules and their status.
    pub async fn get_modules(
        &self,
        request: GetModulesRequest,
    ) -> Result<GetModulesResponse, ServingError> {
        let start = Instant::now();
        debug!("GetModules request, prefix={:?}", request.name_prefix);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.get_modules_requests += 1;
        }

        let mut modules = Vec::new();

        // Add model module info
        if let Some(ref loader) = self.model_loader {
            if let Some(model) = loader.current_model() {
                let mut metadata = HashMap::new();
                metadata.insert("path".to_string(), model.path.display().to_string());
                metadata.insert(
                    "loaded_ago_secs".to_string(),
                    model.loaded_at.elapsed().as_secs().to_string(),
                );

                let module = ModuleInfo {
                    name: "model".to_string(),
                    version: model.version.clone(),
                    loaded: true,
                    metadata,
                };

                if Self::matches_prefix(&module.name, &request.name_prefix) {
                    modules.push(module);
                }
            }
        }

        // Add embedding service module info
        if let Some(ref service) = self.embedding_service {
            let stats = service.stats();
            let mut metadata = HashMap::new();
            metadata.insert(
                "total_requests".to_string(),
                stats.total_requests.to_string(),
            );
            metadata.insert("cache_size".to_string(), service.cache_size().to_string());

            let module = ModuleInfo {
                name: "embedding_service".to_string(),
                version: "1.0".to_string(),
                loaded: service.is_ready(),
                metadata,
            };

            if Self::matches_prefix(&module.name, &request.name_prefix) {
                modules.push(module);
            }
        }

        // Add parameter sync module info
        if let Some(ref sync) = self.param_sync {
            let stats = sync.stats();
            let mut metadata = HashMap::new();
            metadata.insert("total_pulls".to_string(), stats.total_pulls.to_string());
            metadata.insert("total_pushes".to_string(), stats.total_pushes.to_string());

            let module = ModuleInfo {
                name: "parameter_sync".to_string(),
                version: "1.0".to_string(),
                loaded: sync.is_connected(),
                metadata,
            };

            if Self::matches_prefix(&module.name, &request.name_prefix) {
                modules.push(module);
            }
        }

        let latency = start.elapsed().as_secs_f64() * 1000.0;
        self.update_latency(latency);

        debug!(
            "GetModules completed in {:.2}ms, {} modules",
            latency,
            modules.len()
        );

        Ok(GetModulesResponse { modules })
    }

    /// Handle a sync parameters request.
    ///
    /// Synchronizes embeddings for the specified slot and feature IDs.
    pub async fn sync_parameters(
        &self,
        request: SyncParametersRequest,
    ) -> Result<SyncParametersResponse, ServingError> {
        let start = Instant::now();
        debug!(
            "SyncParameters request for slot={}, fids={}, full_sync={}",
            request.slot_id,
            request.fids.len(),
            request.full_sync
        );

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.sync_requests += 1;
        }

        // Check if param sync is available
        let param_sync = self
            .param_sync
            .as_ref()
            .ok_or_else(|| ServingError::sync("Parameter sync client not configured"))?;

        // Perform sync
        let result = if request.full_sync {
            param_sync.full_sync(request.slot_id).await
        } else {
            param_sync
                .pull(request.slot_id, &request.fids)
                .await
                .map(|embeddings| crate::parameter_sync::SyncResponse {
                    slot_id: request.slot_id,
                    num_embeddings: embeddings.len(),
                    duration: start.elapsed(),
                    success: true,
                    error_message: None,
                })
        };

        let duration_ms = start.elapsed().as_millis() as i64;
        self.update_latency(duration_ms as f64);

        match result {
            Ok(sync_response) => {
                debug!(
                    "SyncParameters completed in {}ms, synced {} embeddings",
                    duration_ms, sync_response.num_embeddings
                );

                Ok(SyncParametersResponse {
                    success: true,
                    num_synced: sync_response.num_embeddings as i64,
                    duration_ms,
                    error_message: None,
                })
            }
            Err(e) => {
                warn!("SyncParameters failed: {}", e);
                self.stats.write().failed_requests += 1;

                Ok(SyncParametersResponse {
                    success: false,
                    num_synced: 0,
                    duration_ms,
                    error_message: Some(e.to_string()),
                })
            }
        }
    }

    /// Get service statistics.
    pub fn stats(&self) -> GrpcServiceStats {
        self.stats.read().clone()
    }

    /// Get server uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Get the number of active connections.
    pub fn active_connections(&self) -> u64 {
        self.active_connections.load(Ordering::SeqCst)
    }

    // Private helper methods

    fn update_latency(&self, latency_ms: f64) {
        let mut stats = self.stats.write();
        let total = (stats.heartbeat_requests
            + stats.get_replicas_requests
            + stats.get_resource_requests
            + stats.get_modules_requests
            + stats.sync_requests) as f64;

        if total > 0.0 {
            stats.avg_latency_ms = (stats.avg_latency_ms * (total - 1.0) + latency_ms) / total;
        }
    }

    fn get_memory_usage(&self) -> i64 {
        // In a real implementation, we would use system APIs
        // For now, return a placeholder
        0
    }

    fn get_cpu_usage(&self) -> f32 {
        // In a real implementation, we would use system APIs
        0.0
    }

    fn calculate_work_load(&self) -> f32 {
        let connections = self.active_connections.load(Ordering::SeqCst) as f32;
        let max_connections = self.config.max_connections as f32;
        (connections / max_connections).min(1.0)
    }

    fn matches_prefix(name: &str, prefix: &Option<String>) -> bool {
        match prefix {
            Some(p) => name.starts_with(p),
            None => true,
        }
    }
}

impl std::fmt::Debug for AgentServiceGrpcImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentServiceGrpcImpl")
            .field("config", &self.config)
            .field("uptime_secs", &self.uptime_secs())
            .field("active_connections", &self.active_connections())
            .finish()
    }
}

// ============================================================================
// gRPC Service Trait Implementation (tonic)
// ============================================================================

/// Tonic gRPC service trait for AgentService.
///
/// This trait defines the gRPC interface that tonic will use.
/// In a production setup, this would be generated from the proto file.
#[tonic::async_trait]
pub trait AgentService: Send + Sync + 'static {
    /// Get replicas for a server type.
    async fn get_replicas(
        &self,
        request: Request<GetReplicasRequest>,
    ) -> Result<Response<GetReplicasResponse>, Status>;

    /// Get resource information.
    async fn get_resource(
        &self,
        request: Request<GetResourceRequest>,
    ) -> Result<Response<GetResourceResponse>, Status>;

    /// Send a heartbeat.
    async fn heart_beat(
        &self,
        request: Request<HeartBeatRequest>,
    ) -> Result<Response<HeartBeatResponse>, Status>;
}

#[tonic::async_trait]
impl AgentService for AgentServiceGrpcImpl {
    async fn get_replicas(
        &self,
        request: Request<GetReplicasRequest>,
    ) -> Result<Response<GetReplicasResponse>, Status> {
        match self.get_replicas(request.into_inner()).await {
            Ok(response) => Ok(Response::new(response)),
            Err(e) => {
                error!("GetReplicas error: {}", e);
                Err(Status::internal(e.to_string()))
            }
        }
    }

    async fn get_resource(
        &self,
        request: Request<GetResourceRequest>,
    ) -> Result<Response<GetResourceResponse>, Status> {
        match self.get_resource(request.into_inner()).await {
            Ok(response) => Ok(Response::new(response)),
            Err(e) => {
                error!("GetResource error: {}", e);
                Err(Status::internal(e.to_string()))
            }
        }
    }

    async fn heart_beat(
        &self,
        request: Request<HeartBeatRequest>,
    ) -> Result<Response<HeartBeatResponse>, Status> {
        match self.heartbeat(request.into_inner()).await {
            Ok(response) => Ok(Response::new(response)),
            Err(e) => {
                error!("HeartBeat error: {}", e);
                Err(Status::internal(e.to_string()))
            }
        }
    }
}

// ============================================================================
// Real tonic gRPC boundary (compatible with Python agent_service.proto)
// ============================================================================

/// Adapter that exposes [`AgentServiceGrpcImpl`] over the real `AgentService` proto.
///
/// The bulk of this module is a “hand-written proto surface” used by examples/tests.
/// For real network compatibility with the Python monolith, we serve the generated
/// `monolith.serving.agent_service.AgentService` here and map messages in/out.
#[derive(Clone)]
struct AgentServiceTonicAdapter {
    inner: Arc<AgentServiceGrpcImpl>,
}

impl AgentServiceTonicAdapter {
    fn new(inner: Arc<AgentServiceGrpcImpl>) -> Self {
        Self { inner }
    }

    fn to_local_server_type(st: i32) -> ServerType {
        // `ServerType` here is the local enum defined at the top of this module.
        ServerType::from(st)
    }

    #[allow(dead_code)]
    fn to_proto_server_type(st: ServerType) -> i32 {
        // Generated proto enum is encoded as i32 in messages.
        i32::from(st)
    }
}

#[tonic::async_trait]
impl AgentServiceProto for AgentServiceTonicAdapter {
    async fn get_replicas(
        &self,
        request: Request<agent_proto::GetReplicasRequest>,
    ) -> Result<Response<agent_proto::GetReplicasResponse>, Status> {
        let req = request.into_inner();
        let local_req = GetReplicasRequest {
            server_type: Self::to_local_server_type(req.server_type),
            task: req.task,
            model_name: req.model_name,
        };

        let local_resp = self
            .inner
            .get_replicas(local_req)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let address = local_resp
            .address_list
            .map(|l| agent_proto::AddressList { address: l.address });

        Ok(Response::new(agent_proto::GetReplicasResponse {
            address_list: address,
        }))
    }

    async fn get_resource(
        &self,
        _request: Request<agent_proto::GetResourceRequest>,
    ) -> Result<Response<agent_proto::GetResourceResponse>, Status> {
        let local_resp = self
            .inner
            .get_resource(GetResourceRequest {})
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        Ok(Response::new(agent_proto::GetResourceResponse {
            address: local_resp.address,
            shard_id: local_resp.shard_id,
            replica_id: local_resp.replica_id,
            memory: local_resp.memory,
            cpu: local_resp.cpu,
            network: local_resp.network,
            work_load: local_resp.work_load,
        }))
    }

    async fn heart_beat(
        &self,
        request: Request<agent_proto::HeartBeatRequest>,
    ) -> Result<Response<agent_proto::HeartBeatResponse>, Status> {
        let req = request.into_inner();
        let local_req = HeartBeatRequest {
            server_type: Self::to_local_server_type(req.server_type),
        };

        let local_resp = self
            .inner
            .heartbeat(local_req)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;

        let addresses = local_resp
            .addresses
            .into_iter()
            .map(|(k, v)| (k, agent_proto::AddressList { address: v.address }))
            .collect();

        Ok(Response::new(agent_proto::HeartBeatResponse { addresses }))
    }
}

// ============================================================================
// ServingServer - High-level Server Management
// ============================================================================

/// State of the gRPC server.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpcServerState {
    /// Server is stopped
    Stopped,
    /// Server is starting
    Starting,
    /// Server is running
    Running,
    /// Server is shutting down
    ShuttingDown,
    /// Server encountered an error
    Error,
}

/// High-level gRPC server manager.
///
/// The `ServingServer` provides a high-level interface for managing the
/// gRPC server lifecycle, including starting, stopping, and monitoring.
///
/// # Example
///
/// ```no_run
/// use monolith_serving::grpc::{ServingServer, GrpcServerConfig};
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = GrpcServerConfig::builder()
///     .bind_address("0.0.0.0:50051")
///     .max_connections(1000)
///     .build();
///
/// let server = ServingServer::new(config);
/// server.serve("0.0.0.0:50051").await?;
/// # Ok(())
/// # }
/// ```
pub struct ServingServer {
    /// Server configuration
    config: GrpcServerConfig,

    /// Current server state
    state: Arc<RwLock<GrpcServerState>>,

    /// Agent service implementation
    service: Arc<AgentServiceGrpcImpl>,

    /// Shutdown signal sender
    shutdown_tx: Arc<RwLock<Option<oneshot::Sender<()>>>>,

    /// Whether the server is running
    running: Arc<AtomicBool>,
}

impl ServingServer {
    /// Create a new serving server with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The server configuration
    pub fn new(config: GrpcServerConfig) -> Self {
        let service = Arc::new(AgentServiceGrpcImpl::new(config.clone()));

        Self {
            config,
            state: Arc::new(RwLock::new(GrpcServerState::Stopped)),
            service,
            shutdown_tx: Arc::new(RwLock::new(None)),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Create a new serving server with custom services.
    pub fn with_services(
        config: GrpcServerConfig,
        embedding_service: Option<Arc<EmbeddingService>>,
        model_loader: Option<Arc<ModelLoader>>,
        param_sync: Option<Arc<ParameterSyncClient>>,
    ) -> Self {
        let mut service = AgentServiceGrpcImpl::new(config.clone());

        if let Some(es) = embedding_service {
            service = service.with_embedding_service(es);
        }
        if let Some(ml) = model_loader {
            service = service.with_model_loader(ml);
        }
        if let Some(ps) = param_sync {
            service = service.with_param_sync(ps);
        }

        Self {
            config,
            state: Arc::new(RwLock::new(GrpcServerState::Stopped)),
            service: Arc::new(service),
            shutdown_tx: Arc::new(RwLock::new(None)),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Start the gRPC server and listen for requests.
    ///
    /// This method binds to the specified address and starts accepting
    /// incoming gRPC connections. It runs until `shutdown()` is called.
    ///
    /// # Arguments
    ///
    /// * `addr` - The address to bind to (e.g., "0.0.0.0:50051")
    ///
    /// # Errors
    ///
    /// Returns an error if the server cannot bind to the address or
    /// if an error occurs during startup.
    pub async fn serve(&self, addr: impl AsRef<str>) -> ServingResult<()> {
        let addr_str = addr.as_ref();

        // Check current state
        {
            let current_state = *self.state.read();
            if current_state == GrpcServerState::Running {
                warn!("Server is already running");
                return Ok(());
            }
            if current_state == GrpcServerState::Starting {
                return Err(ServingError::server("Server is already starting"));
            }
        }

        // Update state to starting
        *self.state.write() = GrpcServerState::Starting;
        info!("Starting gRPC server on {}", addr_str);

        // Parse the address
        let socket_addr: SocketAddr = addr_str.parse().map_err(|e| {
            *self.state.write() = GrpcServerState::Error;
            ServingError::config(format!("Invalid address '{}': {}", addr_str, e))
        })?;

        // Create shutdown channel
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        *self.shutdown_tx.write() = Some(shutdown_tx);

        self.running.store(true, Ordering::SeqCst);
        *self.state.write() = GrpcServerState::Running;

        info!("gRPC server started on {}", socket_addr);

        let adapter = AgentServiceTonicAdapter::new(Arc::clone(&self.service));

        // Serve the real proto-compatible AgentService.
        //
        // This is the network boundary used by the Python monolith. The local
        // `AgentServiceGrpcImpl` still backs the behavior, but messages are
        // mapped to the generated protobuf types.
        TonicServer::builder()
            .add_service(AgentServiceServer::new(adapter))
            .serve_with_shutdown(socket_addr, async move {
                let _ = shutdown_rx.await;
                info!("Received shutdown signal");
            })
            .await
            .map_err(|e| {
                *self.state.write() = GrpcServerState::Error;
                self.running.store(false, Ordering::SeqCst);
                ServingError::server(format!("gRPC server error: {}", e))
            })?;

        *self.state.write() = GrpcServerState::Stopped;
        self.running.store(false, Ordering::SeqCst);

        info!("gRPC server stopped");
        Ok(())
    }

    /// Shutdown the server gracefully.
    ///
    /// This method signals the server to stop accepting new connections
    /// and waits for existing connections to complete.
    pub fn shutdown(&self) {
        info!("Initiating server shutdown");
        *self.state.write() = GrpcServerState::ShuttingDown;

        // Send shutdown signal
        if let Some(tx) = self.shutdown_tx.write().take() {
            let _ = tx.send(());
        }

        self.running.store(false, Ordering::SeqCst);
        info!("Server shutdown complete");
    }

    /// Check if the server is running.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Get the current server state.
    pub fn state(&self) -> GrpcServerState {
        *self.state.read()
    }

    /// Get the server configuration.
    pub fn config(&self) -> &GrpcServerConfig {
        &self.config
    }

    /// Get the agent service implementation.
    pub fn service(&self) -> &Arc<AgentServiceGrpcImpl> {
        &self.service
    }

    /// Register a replica with the server.
    pub fn register_replica(&self, server_type: ServerType, address: String) {
        self.service.register_replica(server_type, address);
    }

    /// Unregister a replica from the server.
    pub fn unregister_replica(&self, server_type: ServerType, address: &str) {
        self.service.unregister_replica(server_type, address);
    }
}

impl std::fmt::Debug for ServingServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServingServer")
            .field("config", &self.config)
            .field("state", &*self.state.read())
            .field("running", &self.running.load(Ordering::SeqCst))
            .finish()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ModelLoaderConfig;
    use tempfile::tempdir;

    // ========================================================================
    // GrpcServerConfig Tests
    // ========================================================================

    #[test]
    fn test_default_config() {
        let config = GrpcServerConfig::default();
        assert_eq!(config.bind_address, "0.0.0.0:50051");
        assert_eq!(config.max_connections, 1000);
        assert_eq!(config.server_type, ServerType::Entry);
        assert_eq!(config.shard_id, 0);
        assert_eq!(config.replica_id, 0);
    }

    #[test]
    fn test_config_builder() {
        let config = GrpcServerConfig::builder()
            .bind_address("127.0.0.1:9090")
            .max_connections(500)
            .connection_timeout(Duration::from_secs(5))
            .request_timeout(Duration::from_secs(60))
            .server_type(ServerType::Ps)
            .shard_id(1)
            .replica_id(2)
            .build();

        assert_eq!(config.bind_address, "127.0.0.1:9090");
        assert_eq!(config.max_connections, 500);
        assert_eq!(config.connection_timeout, Duration::from_secs(5));
        assert_eq!(config.request_timeout, Duration::from_secs(60));
        assert_eq!(config.server_type, ServerType::Ps);
        assert_eq!(config.shard_id, 1);
        assert_eq!(config.replica_id, 2);
    }

    #[test]
    fn test_socket_addr_parsing() {
        let config = GrpcServerConfig::default();
        let addr = config
            .socket_addr()
            .expect("default gRPC bind address should parse into SocketAddr");
        assert_eq!(addr.port(), 50051);

        let invalid_config = GrpcServerConfig::builder()
            .bind_address("invalid:address:port")
            .build();
        invalid_config
            .socket_addr()
            .expect_err("invalid gRPC bind address should fail socket parsing");
    }

    // ========================================================================
    // ServerType Tests
    // ========================================================================

    #[test]
    fn test_server_type_conversion() {
        assert_eq!(ServerType::from(0), ServerType::Ps);
        assert_eq!(ServerType::from(1), ServerType::Entry);
        assert_eq!(ServerType::from(2), ServerType::Dense);
        assert_eq!(ServerType::from(99), ServerType::Ps); // Default

        assert_eq!(i32::from(ServerType::Ps), 0);
        assert_eq!(i32::from(ServerType::Entry), 1);
        assert_eq!(i32::from(ServerType::Dense), 2);
    }

    // ========================================================================
    // AgentServiceGrpcImpl Tests
    // ========================================================================

    #[tokio::test]
    async fn test_agent_service_creation() {
        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config);

        assert_eq!(service.active_connections(), 0);
        assert!(service.uptime_secs() < 5);

        let stats = service.stats();
        assert_eq!(stats.heartbeat_requests, 0);
        assert_eq!(stats.get_replicas_requests, 0);
    }

    #[tokio::test]
    async fn test_heartbeat() {
        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config);

        // Register some replicas
        service.register_replica(ServerType::Ps, "ps1:9000".to_string());
        service.register_replica(ServerType::Ps, "ps2:9000".to_string());
        service.register_replica(ServerType::Entry, "entry1:8080".to_string());

        let request = HeartBeatRequest {
            server_type: ServerType::Entry,
        };

        let response = service.heartbeat(request).await.unwrap();

        // Should have addresses for PS and ENTRY
        assert!(response.addresses.contains_key("PS"));
        assert!(response.addresses.contains_key("ENTRY"));
        assert_eq!(response.addresses.get("PS").unwrap().address.len(), 2);
        assert_eq!(response.addresses.get("ENTRY").unwrap().address.len(), 1);

        let stats = service.stats();
        assert_eq!(stats.heartbeat_requests, 1);
    }

    #[tokio::test]
    async fn test_get_replicas() {
        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config);

        // Register replicas
        service.register_replica(ServerType::Ps, "ps1:9000".to_string());
        service.register_replica(ServerType::Ps, "ps2:9000".to_string());

        let request = GetReplicasRequest {
            server_type: ServerType::Ps,
            task: 0,
            model_name: "test_model".to_string(),
        };

        let response = service.get_replicas(request).await.unwrap();

        assert!(response.address_list.is_some());
        let addresses = response.address_list.unwrap();
        assert_eq!(addresses.address.len(), 2);
        assert!(addresses.address.contains(&"ps1:9000".to_string()));
        assert!(addresses.address.contains(&"ps2:9000".to_string()));

        let stats = service.stats();
        assert_eq!(stats.get_replicas_requests, 1);
    }

    #[tokio::test]
    async fn test_get_replicas_empty() {
        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config);

        let request = GetReplicasRequest {
            server_type: ServerType::Dense,
            task: 0,
            model_name: "test_model".to_string(),
        };

        let response = service.get_replicas(request).await.unwrap();
        assert!(response.address_list.is_none());
    }

    #[tokio::test]
    async fn test_get_resource() {
        let config = GrpcServerConfig::builder()
            .bind_address("127.0.0.1:50051")
            .shard_id(1)
            .replica_id(2)
            .build();

        let service = AgentServiceGrpcImpl::new(config);

        let request = GetResourceRequest {};
        let response = service.get_resource(request).await.unwrap();

        assert_eq!(response.address, "127.0.0.1:50051");
        assert_eq!(response.shard_id, 1);
        assert_eq!(response.replica_id, 2);
        assert!(response.work_load >= 0.0 && response.work_load <= 1.0);

        let stats = service.stats();
        assert_eq!(stats.get_resource_requests, 1);
    }

    #[tokio::test]
    async fn test_get_modules_with_model() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        let model_loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
        model_loader.load(&model_path).await.unwrap();

        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config).with_model_loader(model_loader);

        let request = GetModulesRequest { name_prefix: None };
        let response = service.get_modules(request).await.unwrap();

        // Should have at least the model module
        assert!(!response.modules.is_empty());

        let model_module = response.modules.iter().find(|m| m.name == "model");
        assert!(model_module.is_some());
        assert!(model_module.unwrap().loaded);

        let stats = service.stats();
        assert_eq!(stats.get_modules_requests, 1);
    }

    #[tokio::test]
    async fn test_get_modules_with_prefix() {
        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config);

        let request = GetModulesRequest {
            name_prefix: Some("embedding".to_string()),
        };
        let response = service.get_modules(request).await.unwrap();

        // All returned modules should start with "embedding"
        for module in &response.modules {
            assert!(module.name.starts_with("embedding"));
        }
    }

    #[tokio::test]
    async fn test_register_unregister_replica() {
        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config);

        // Register
        service.register_replica(ServerType::Ps, "ps1:9000".to_string());
        service.register_replica(ServerType::Ps, "ps2:9000".to_string());

        let request = GetReplicasRequest {
            server_type: ServerType::Ps,
            task: 0,
            model_name: "test".to_string(),
        };

        let response = service.get_replicas(request.clone()).await.unwrap();
        assert_eq!(response.address_list.as_ref().unwrap().address.len(), 2);

        // Unregister
        service.unregister_replica(ServerType::Ps, "ps1:9000");

        let response = service.get_replicas(request).await.unwrap();
        assert_eq!(response.address_list.as_ref().unwrap().address.len(), 1);
        assert!(response
            .address_list
            .unwrap()
            .address
            .contains(&"ps2:9000".to_string()));
    }

    #[tokio::test]
    async fn test_duplicate_registration() {
        let config = GrpcServerConfig::default();
        let service = AgentServiceGrpcImpl::new(config);

        // Register same address twice
        service.register_replica(ServerType::Ps, "ps1:9000".to_string());
        service.register_replica(ServerType::Ps, "ps1:9000".to_string());

        let request = GetReplicasRequest {
            server_type: ServerType::Ps,
            task: 0,
            model_name: "test".to_string(),
        };

        let response = service.get_replicas(request).await.unwrap();
        // Should only have one entry (no duplicates)
        assert_eq!(response.address_list.unwrap().address.len(), 1);
    }

    // ========================================================================
    // ServingServer Tests
    // ========================================================================

    #[test]
    fn test_serving_server_creation() {
        let config = GrpcServerConfig::default();
        let server = ServingServer::new(config);

        assert_eq!(server.state(), GrpcServerState::Stopped);
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn test_serving_server_with_services() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("test_model");
        std::fs::create_dir_all(&model_path).unwrap();

        let model_loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
        model_loader.load(&model_path).await.unwrap();

        let embedding_service = Arc::new(EmbeddingService::new(model_loader.clone(), None));

        let config = GrpcServerConfig::default();
        let server =
            ServingServer::with_services(config, Some(embedding_service), Some(model_loader), None);

        assert_eq!(server.state(), GrpcServerState::Stopped);
    }

    #[tokio::test]
    async fn test_serving_server_shutdown() {
        let config = GrpcServerConfig::default();
        let server = Arc::new(ServingServer::new(config));

        // Start server in background
        let server_clone = Arc::clone(&server);
        let handle = tokio::spawn(async move { server_clone.serve("127.0.0.1:50099").await });

        // Give server time to start
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Should be running (or starting)
        assert!(
            server.state() == GrpcServerState::Running
                || server.state() == GrpcServerState::Starting
        );

        // Shutdown
        server.shutdown();

        // Wait for server to stop
        let _ = tokio::time::timeout(Duration::from_secs(2), handle).await;

        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn test_serving_server_register_replica() {
        let config = GrpcServerConfig::default();
        let server = ServingServer::new(config);

        server.register_replica(ServerType::Ps, "ps1:9000".to_string());
        server.register_replica(ServerType::Entry, "entry1:8080".to_string());

        // Verify through service
        let request = GetReplicasRequest {
            server_type: ServerType::Ps,
            task: 0,
            model_name: "test".to_string(),
        };

        let response = server.service().get_replicas(request).await.unwrap();
        assert!(response.address_list.is_some());
        assert_eq!(response.address_list.unwrap().address.len(), 1);
    }

    // ========================================================================
    // Mock Client for Integration Testing
    // ========================================================================

    /// Mock gRPC client for testing.
    pub struct MockAgentServiceClient {
        service: Arc<AgentServiceGrpcImpl>,
    }

    impl MockAgentServiceClient {
        /// Create a new mock client connected to the given service.
        pub fn new(service: Arc<AgentServiceGrpcImpl>) -> Self {
            Self { service }
        }

        /// Send a heartbeat request.
        pub async fn heartbeat(&self, server_type: ServerType) -> ServingResult<HeartBeatResponse> {
            let request = HeartBeatRequest { server_type };
            self.service.heartbeat(request).await
        }

        /// Get replicas for a server type.
        pub async fn get_replicas(
            &self,
            server_type: ServerType,
            task: i32,
            model_name: &str,
        ) -> ServingResult<GetReplicasResponse> {
            let request = GetReplicasRequest {
                server_type,
                task,
                model_name: model_name.to_string(),
            };
            self.service.get_replicas(request).await
        }

        /// Get resource information.
        pub async fn get_resource(&self) -> ServingResult<GetResourceResponse> {
            let request = GetResourceRequest {};
            self.service.get_resource(request).await
        }

        /// Get modules information.
        pub async fn get_modules(
            &self,
            name_prefix: Option<&str>,
        ) -> ServingResult<GetModulesResponse> {
            let request = GetModulesRequest {
                name_prefix: name_prefix.map(|s| s.to_string()),
            };
            self.service.get_modules(request).await
        }

        /// Sync parameters.
        pub async fn sync_parameters(
            &self,
            slot_id: i32,
            fids: Vec<i64>,
            full_sync: bool,
        ) -> ServingResult<SyncParametersResponse> {
            let request = SyncParametersRequest {
                slot_id,
                fids,
                full_sync,
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            };
            self.service.sync_parameters(request).await
        }
    }

    #[tokio::test]
    async fn test_mock_client_heartbeat() {
        let config = GrpcServerConfig::default();
        let service = Arc::new(AgentServiceGrpcImpl::new(config));

        // Register some replicas
        service.register_replica(ServerType::Ps, "ps1:9000".to_string());

        let client = MockAgentServiceClient::new(service);

        let response = client.heartbeat(ServerType::Entry).await.unwrap();
        assert!(response.addresses.contains_key("PS"));
    }

    #[tokio::test]
    async fn test_mock_client_full_workflow() {
        let config = GrpcServerConfig::builder()
            .shard_id(1)
            .replica_id(0)
            .build();

        let service = Arc::new(AgentServiceGrpcImpl::new(config));
        let client = MockAgentServiceClient::new(service.clone());

        // Register replicas
        service.register_replica(ServerType::Ps, "ps1:9000".to_string());
        service.register_replica(ServerType::Ps, "ps2:9000".to_string());
        service.register_replica(ServerType::Entry, "entry1:8080".to_string());

        // Test heartbeat
        let hb_response = client.heartbeat(ServerType::Entry).await.unwrap();
        assert_eq!(hb_response.addresses.get("PS").unwrap().address.len(), 2);

        // Test get replicas
        let replicas = client
            .get_replicas(ServerType::Ps, 0, "test_model")
            .await
            .unwrap();
        assert_eq!(replicas.address_list.unwrap().address.len(), 2);

        // Test get resource
        let resource = client.get_resource().await.unwrap();
        assert_eq!(resource.shard_id, 1);
        assert_eq!(resource.replica_id, 0);

        // Test get modules
        let modules = client.get_modules(None).await.unwrap();
        // May be empty without model loader, that's fine

        // Verify stats
        let stats = service.stats();
        assert_eq!(stats.heartbeat_requests, 1);
        assert_eq!(stats.get_replicas_requests, 1);
        assert_eq!(stats.get_resource_requests, 1);
        assert_eq!(stats.get_modules_requests, 1);
    }

    // ========================================================================
    // Message Type Tests
    // ========================================================================

    #[test]
    fn test_address_list() {
        let list = AddressList {
            address: vec!["a:1".to_string(), "b:2".to_string()],
        };
        assert_eq!(list.address.len(), 2);
    }

    #[test]
    fn test_get_resource_response_default() {
        let response = GetResourceResponse::default();
        assert!(response.address.is_empty());
        assert_eq!(response.shard_id, 0);
        assert_eq!(response.replica_id, 0);
        assert_eq!(response.memory, 0);
        assert_eq!(response.cpu, 0.0);
        assert_eq!(response.network, 0.0);
        assert_eq!(response.work_load, 0.0);
    }

    #[test]
    fn test_sync_parameters_request() {
        let request = SyncParametersRequest {
            slot_id: 1,
            fids: vec![100, 200, 300],
            full_sync: false,
            timestamp: 12345,
        };

        assert_eq!(request.slot_id, 1);
        assert_eq!(request.fids.len(), 3);
        assert!(!request.full_sync);
        assert_eq!(request.timestamp, 12345);
    }

    #[test]
    fn test_module_info() {
        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), "value".to_string());

        let module = ModuleInfo {
            name: "test_module".to_string(),
            version: "1.0.0".to_string(),
            loaded: true,
            metadata,
        };

        assert_eq!(module.name, "test_module");
        assert_eq!(module.version, "1.0.0");
        assert!(module.loaded);
        assert_eq!(module.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_grpc_service_stats_default() {
        let stats = GrpcServiceStats::default();
        assert_eq!(stats.heartbeat_requests, 0);
        assert_eq!(stats.get_replicas_requests, 0);
        assert_eq!(stats.get_resource_requests, 0);
        assert_eq!(stats.get_modules_requests, 0);
        assert_eq!(stats.sync_requests, 0);
        assert_eq!(stats.failed_requests, 0);
        assert_eq!(stats.avg_latency_ms, 0.0);
    }

    #[test]
    fn test_grpc_server_state() {
        assert_eq!(GrpcServerState::Stopped, GrpcServerState::Stopped);
        assert_ne!(GrpcServerState::Running, GrpcServerState::Stopped);
    }
}
