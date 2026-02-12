//! Parameter synchronization client for syncing with parameter servers.
//!
//! This module provides the [`ParameterSyncClient`] for pulling and pushing
//! embeddings to/from parameter servers in a distributed training environment.

use crate::config::ParameterServerConfig;
use crate::error::{ServingError, ServingResult};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

/// Request for syncing parameters.
#[derive(Debug, Clone)]
pub struct SyncRequest {
    /// Slot ID to sync
    pub slot_id: i32,

    /// Feature IDs to sync
    pub fids: Vec<i64>,

    /// Whether this is a full sync or incremental
    pub full_sync: bool,
}

/// Response from a parameter sync operation.
#[derive(Debug, Clone)]
pub struct SyncResponse {
    /// Slot ID that was synced
    pub slot_id: i32,

    /// Number of embeddings synced
    pub num_embeddings: usize,

    /// Duration of the sync operation
    pub duration: Duration,

    /// Whether the sync was successful
    pub success: bool,

    /// Error message if sync failed
    pub error_message: Option<String>,
}

/// Statistics for parameter sync operations.
#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    /// Total number of pull operations
    pub total_pulls: u64,

    /// Successful pull operations
    pub successful_pulls: u64,

    /// Failed pull operations
    pub failed_pulls: u64,

    /// Total number of push operations
    pub total_pushes: u64,

    /// Successful push operations
    pub successful_pushes: u64,

    /// Failed push operations
    pub failed_pushes: u64,

    /// Total bytes pulled
    pub bytes_pulled: u64,

    /// Total bytes pushed
    pub bytes_pushed: u64,

    /// Average pull latency in milliseconds
    pub avg_pull_latency_ms: f64,

    /// Average push latency in milliseconds
    pub avg_push_latency_ms: f64,

    /// Last sync timestamp
    pub last_sync_time: Option<Instant>,
}

/// Embedding data for sync operations.
#[derive(Debug, Clone)]
pub struct EmbeddingData {
    /// Feature ID
    pub fid: i64,

    /// Embedding vector
    pub embedding: Vec<f32>,

    /// Optional version/timestamp
    pub version: u64,
}

/// Client for synchronizing parameters with parameter servers.
///
/// The `ParameterSyncClient` manages connections to parameter servers and
/// provides methods for pulling and pushing embeddings.
///
/// # Example
///
/// ```no_run
/// use monolith_serving::parameter_sync::ParameterSyncClient;
/// use monolith_serving::config::ParameterServerConfig;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let config = ParameterServerConfig::default();
/// let client = ParameterSyncClient::new(config);
///
/// // Connect to parameter servers
/// client.connect().await?;
///
/// // Pull embeddings for specific feature IDs
/// let embeddings = client.pull(0, &[1, 2, 3]).await?;
///
/// // Push updated embeddings
/// client.push(0, embeddings).await?;
/// # Ok(())
/// # }
/// ```
pub struct ParameterSyncClient {
    /// Configuration for the parameter servers
    config: ParameterServerConfig,

    /// Connection state
    connected: Arc<RwLock<bool>>,

    /// Active parameter server connections (indexed by address)
    connections: Arc<RwLock<HashMap<String, ConnectionState>>>,

    /// Sync statistics
    stats: Arc<RwLock<SyncStats>>,

    /// Shutdown signal sender
    shutdown_tx: Arc<RwLock<Option<mpsc::Sender<()>>>>,
}

/// State of a connection to a parameter server.
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct ConnectionState {
    /// Server address
    address: String,

    /// Whether the connection is healthy
    healthy: bool,

    /// Last successful operation timestamp
    last_success: Option<Instant>,

    /// Number of consecutive failures
    consecutive_failures: u32,
}

impl ParameterSyncClient {
    /// Create a new parameter sync client with the given configuration.
    pub fn new(config: ParameterServerConfig) -> Self {
        Self {
            config,
            connected: Arc::new(RwLock::new(false)),
            connections: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SyncStats::default())),
            shutdown_tx: Arc::new(RwLock::new(None)),
        }
    }

    /// Connect to all configured parameter servers.
    ///
    /// # Errors
    ///
    /// Returns an error if unable to connect to any parameter server.
    pub async fn connect(&self) -> ServingResult<()> {
        info!(
            "Connecting to {} parameter servers",
            self.config.addresses.len()
        );

        let mut connections = self.connections.write();

        for address in &self.config.addresses {
            debug!("Connecting to parameter server: {}", address);

            // In a real implementation, we would establish gRPC connections here
            // For now, simulate connection
            let state = ConnectionState {
                address: address.clone(),
                healthy: true,
                last_success: Some(Instant::now()),
                consecutive_failures: 0,
            };

            connections.insert(address.clone(), state);
            info!("Connected to parameter server: {}", address);
        }

        *self.connected.write() = true;
        info!("Successfully connected to all parameter servers");

        Ok(())
    }

    /// Disconnect from all parameter servers.
    pub async fn disconnect(&self) {
        info!("Disconnecting from parameter servers");

        // Send shutdown signal if background sync is running
        let tx = self.shutdown_tx.write().take();
        if let Some(tx) = tx {
            let _ = tx.send(()).await;
        }

        self.connections.write().clear();
        *self.connected.write() = false;

        info!("Disconnected from all parameter servers");
    }

    /// Check if the client is connected.
    pub fn is_connected(&self) -> bool {
        *self.connected.read()
    }

    /// Pull embeddings from parameter servers.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID to pull embeddings for
    /// * `fids` - The feature IDs to pull
    ///
    /// # Returns
    ///
    /// A vector of embedding data for the requested feature IDs.
    ///
    /// # Errors
    ///
    /// Returns an error if the client is not connected or if the pull operation fails.
    pub async fn pull(&self, slot_id: i32, fids: &[i64]) -> ServingResult<Vec<EmbeddingData>> {
        if !self.is_connected() {
            return Err(ServingError::NotConnected);
        }

        let start = Instant::now();
        debug!("Pulling {} embeddings for slot {}", fids.len(), slot_id);

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_pulls += 1;
        }

        // In a real implementation, we would:
        // 1. Determine which PS shard owns each FID
        // 2. Send parallel requests to each shard
        // 3. Collect and merge responses

        // Simulate pull operation
        let embeddings: Vec<EmbeddingData> = fids
            .iter()
            .map(|&fid| EmbeddingData {
                fid,
                embedding: vec![0.0; 64], // Default embedding dimension
                version: 1,
            })
            .collect();

        let duration = start.elapsed();

        // Update stats on success
        {
            let mut stats = self.stats.write();
            stats.successful_pulls += 1;
            stats.bytes_pulled += (embeddings.len() * 64 * 4) as u64; // 64 dims * 4 bytes
            stats.last_sync_time = Some(Instant::now());

            // Update average latency
            let total = stats.successful_pulls as f64;
            stats.avg_pull_latency_ms =
                (stats.avg_pull_latency_ms * (total - 1.0) + duration.as_millis() as f64) / total;
        }

        debug!("Pulled {} embeddings in {:?}", embeddings.len(), duration);

        Ok(embeddings)
    }

    /// Push embeddings to parameter servers.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID to push embeddings for
    /// * `embeddings` - The embeddings to push
    ///
    /// # Errors
    ///
    /// Returns an error if the client is not connected or if the push operation fails.
    pub async fn push(&self, slot_id: i32, embeddings: Vec<EmbeddingData>) -> ServingResult<()> {
        if !self.is_connected() {
            return Err(ServingError::NotConnected);
        }

        let start = Instant::now();
        debug!(
            "Pushing {} embeddings for slot {}",
            embeddings.len(),
            slot_id
        );

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.total_pushes += 1;
        }

        // In a real implementation, we would:
        // 1. Determine which PS shard owns each FID
        // 2. Batch updates by shard
        // 3. Send parallel push requests

        // Simulate push operation
        let num_embeddings = embeddings.len();
        let bytes_pushed = embeddings
            .iter()
            .map(|e| e.embedding.len() * 4)
            .sum::<usize>() as u64;

        let duration = start.elapsed();

        // Update stats on success
        {
            let mut stats = self.stats.write();
            stats.successful_pushes += 1;
            stats.bytes_pushed += bytes_pushed;
            stats.last_sync_time = Some(Instant::now());

            // Update average latency
            let total = stats.successful_pushes as f64;
            stats.avg_push_latency_ms =
                (stats.avg_push_latency_ms * (total - 1.0) + duration.as_millis() as f64) / total;
        }

        debug!("Pushed {} embeddings in {:?}", num_embeddings, duration);

        Ok(())
    }

    /// Perform a full sync for a slot.
    ///
    /// This pulls all embeddings for the given slot from parameter servers.
    pub async fn full_sync(&self, slot_id: i32) -> ServingResult<SyncResponse> {
        let start = Instant::now();
        info!("Starting full sync for slot {}", slot_id);

        // In a real implementation, we would iterate through all embeddings
        // For now, simulate a full sync
        let embeddings = self.pull(slot_id, &[]).await?;

        let duration = start.elapsed();
        let response = SyncResponse {
            slot_id,
            num_embeddings: embeddings.len(),
            duration,
            success: true,
            error_message: None,
        };

        info!("Full sync completed for slot {} in {:?}", slot_id, duration);

        Ok(response)
    }

    /// Start background sync task.
    ///
    /// This spawns a background task that periodically syncs with parameter servers.
    pub fn start_background_sync(&self) -> ServingResult<()> {
        if !self.config.auto_sync_enabled {
            warn!("Auto sync is disabled in configuration");
            return Ok(());
        }

        let (tx, mut rx) = mpsc::channel::<()>(1);
        *self.shutdown_tx.write() = Some(tx);

        let interval = self.config.sync_interval;
        let connected = Arc::clone(&self.connected);
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);

            loop {
                tokio::select! {
                    _ = interval_timer.tick() => {
                        if *connected.read() {
                            debug!("Background sync tick");
                            // In a real implementation, we would sync here
                            stats.write().last_sync_time = Some(Instant::now());
                        }
                    }
                    _ = rx.recv() => {
                        info!("Background sync shutting down");
                        break;
                    }
                }
            }
        });

        info!("Started background sync with interval {:?}", interval);
        Ok(())
    }

    /// Get current sync statistics.
    pub fn stats(&self) -> SyncStats {
        self.stats.read().clone()
    }

    /// Get the health status of all connections.
    pub fn connection_health(&self) -> HashMap<String, bool> {
        self.connections
            .read()
            .iter()
            .map(|(addr, state)| (addr.clone(), state.healthy))
            .collect()
    }

    /// Get the configuration.
    pub fn config(&self) -> &ParameterServerConfig {
        &self.config
    }

    /// Mark a connection as unhealthy after a failure.
    #[allow(dead_code)]
    fn mark_unhealthy(&self, address: &str) {
        if let Some(state) = self.connections.write().get_mut(address) {
            state.healthy = false;
            state.consecutive_failures += 1;
            error!(
                "Connection to {} marked unhealthy (failures: {})",
                address, state.consecutive_failures
            );
        }
    }

    /// Mark a connection as healthy after a successful operation.
    #[allow(dead_code)]
    fn mark_healthy(&self, address: &str) {
        if let Some(state) = self.connections.write().get_mut(address) {
            state.healthy = true;
            state.consecutive_failures = 0;
            state.last_success = Some(Instant::now());
        }
    }
}

impl std::fmt::Debug for ParameterSyncClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParameterSyncClient")
            .field("config", &self.config)
            .field("connected", &*self.connected.read())
            .field("num_connections", &self.connections.read().len())
            .finish()
    }
}

impl Drop for ParameterSyncClient {
    fn drop(&mut self) {
        // Ensure shutdown signal is sent
        if let Some(tx) = self.shutdown_tx.write().take() {
            // Use blocking send since we're in drop
            let _ = tx.try_send(());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> ParameterServerConfig {
        ParameterServerConfig {
            addresses: vec!["localhost:9000".to_string(), "localhost:9001".to_string()],
            connect_timeout: Duration::from_secs(1),
            request_timeout: Duration::from_secs(5),
            max_retries: 3,
            sync_interval: Duration::from_millis(100),
            auto_sync_enabled: false,
        }
    }

    #[tokio::test]
    async fn test_client_creation() {
        let config = test_config();
        let client = ParameterSyncClient::new(config);

        assert!(!client.is_connected());
        assert!(client.connection_health().is_empty());
    }

    #[tokio::test]
    async fn test_connect_disconnect() {
        let config = test_config();
        let client = ParameterSyncClient::new(config);

        // Connect
        client
            .connect()
            .await
            .expect("parameter sync client should connect in test setup");
        assert!(client.is_connected());
        assert_eq!(client.connection_health().len(), 2);

        // Disconnect
        client.disconnect().await;
        assert!(!client.is_connected());
        assert!(client.connection_health().is_empty());
    }

    #[tokio::test]
    async fn test_pull_not_connected() {
        let config = test_config();
        let client = ParameterSyncClient::new(config);

        let err = client
            .pull(0, &[1, 2, 3])
            .await
            .expect_err("pull should fail when client is not connected");
        assert!(matches!(err, ServingError::NotConnected));
    }

    #[tokio::test]
    async fn test_pull_embeddings() {
        let config = test_config();
        let client = ParameterSyncClient::new(config);
        client.connect().await.unwrap();

        let fids = vec![1, 2, 3, 4, 5];
        let embeddings = client
            .pull(0, &fids)
            .await
            .expect("pull should succeed for connected client");
        assert_eq!(embeddings.len(), fids.len());

        for (i, emb) in embeddings.iter().enumerate() {
            assert_eq!(emb.fid, fids[i]);
            assert_eq!(emb.embedding.len(), 64);
        }
    }

    #[tokio::test]
    async fn test_push_embeddings() {
        let config = test_config();
        let client = ParameterSyncClient::new(config);
        client.connect().await.unwrap();

        let embeddings = vec![
            EmbeddingData {
                fid: 1,
                embedding: vec![0.1; 64],
                version: 1,
            },
            EmbeddingData {
                fid: 2,
                embedding: vec![0.2; 64],
                version: 1,
            },
        ];

        client
            .push(0, embeddings)
            .await
            .expect("push should succeed for connected client");
    }

    #[tokio::test]
    async fn test_stats_tracking() {
        let config = test_config();
        let client = ParameterSyncClient::new(config);
        client.connect().await.unwrap();

        // Perform some operations
        client.pull(0, &[1, 2, 3]).await.unwrap();
        client.pull(0, &[4, 5]).await.unwrap();
        client
            .push(
                0,
                vec![EmbeddingData {
                    fid: 1,
                    embedding: vec![0.1; 64],
                    version: 1,
                }],
            )
            .await
            .unwrap();

        let stats = client.stats();
        assert_eq!(stats.total_pulls, 2);
        assert_eq!(stats.successful_pulls, 2);
        assert_eq!(stats.total_pushes, 1);
        assert_eq!(stats.successful_pushes, 1);
        assert!(stats.bytes_pulled > 0);
        assert!(stats.bytes_pushed > 0);
        assert!(stats.last_sync_time.is_some());
    }

    #[tokio::test]
    async fn test_full_sync() {
        let config = test_config();
        let client = ParameterSyncClient::new(config);
        client.connect().await.unwrap();

        let response = client.full_sync(0).await.unwrap();
        assert_eq!(response.slot_id, 0);
        assert!(response.success);
        assert!(response.error_message.is_none());
    }

    #[test]
    fn test_embedding_data() {
        let data = EmbeddingData {
            fid: 42,
            embedding: vec![0.5; 32],
            version: 100,
        };

        assert_eq!(data.fid, 42);
        assert_eq!(data.embedding.len(), 32);
        assert_eq!(data.version, 100);
    }

    #[test]
    fn test_sync_request() {
        let request = SyncRequest {
            slot_id: 1,
            fids: vec![10, 20, 30],
            full_sync: false,
        };

        assert_eq!(request.slot_id, 1);
        assert_eq!(request.fids.len(), 3);
        assert!(!request.full_sync);
    }
}
