//! Distributed Parameter Server with gRPC.
//!
//! This module provides a production-ready PS-Worker implementation matching
//! Python's `DistributedHashTable` behavior in `distributed_ps.py`.
//!
//! # Key Features
//!
//! - **ID Deduplication**: Unique IDs before lookup/apply to minimize network traffic
//! - **Shard Routing**: `id % num_ps` routing to distribute load across PS instances
//! - **Parallel Fanout**: Concurrent requests to all relevant PS shards
//! - **Gradient Aggregation**: Sum gradients for duplicate IDs before sending
//! - **Remap to Original Order**: Return embeddings in original ID order
//!
//! # Example
//!
//! ```ignore
//! use monolith_training::distributed_ps::{PsServer, PsClient};
//!
//! // Start a PS server
//! let server = PsServer::new(0, 8192).await?;
//! server.start("0.0.0.0:50051").await?;
//!
//! // Create a client
//! let client = PsClient::connect(&["localhost:50051", "localhost:50052"]).await?;
//!
//! // Lookup embeddings (handles dedup and remap automatically)
//! let embeddings = client.lookup("user_embeddings", &[1, 2, 1, 3, 2], 32).await?;
//! assert_eq!(embeddings.len(), 5 * 32); // Original order preserved
//!
//! // Apply gradients (handles aggregation automatically)
//! client.apply_gradients("user_embeddings", &[1, 2, 1], &gradients, 32, 0.01).await?;
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::{join_all, try_join_all};
use parking_lot::RwLock;
use thiserror::Error;
use tokio::sync::{Mutex as TokioMutex, Notify};
use tonic::{Request, Response, Status};

use crate::parameter_sync_replicator::DirtyTracker;
use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable, ZerosInitializer};

// Import generated proto types
use monolith_proto::monolith::ps_training::{
    BatchApplyGradientsRequest, BatchApplyGradientsResponse, BatchLookupRequest, BatchLookupResponse,
    parameter_server_training_client::ParameterServerTrainingClient,
    parameter_server_training_server::{ParameterServerTraining, ParameterServerTrainingServer},
    ApplyGradientsRequest, ApplyGradientsResponse, BarrierRequest, BarrierResponse,
    GetStatsRequest, GetStatsResponse, HealthCheckRequest, HealthCheckResponse, LookupRequest,
    LookupResponse, TableStats,
};

/// Errors that can occur in distributed PS operations.
#[derive(Debug, Error)]
pub enum PsError {
    /// Failed to connect to a PS instance.
    #[error("Connection failed to {0}: {1}")]
    ConnectionFailed(String, String),

    /// RPC call failed.
    #[error("RPC error: {0}")]
    RpcError(#[from] tonic::Status),

    /// Timeout waiting for response.
    #[error("Timeout after {0:?}")]
    Timeout(Duration),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Table not found.
    #[error("Table not found: {0}")]
    TableNotFound(String),

    /// Dimension mismatch.
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

pub type PsResult<T> = Result<T, PsError>;

// ============================================================================
// Embedding Table (Server-side storage)
// ============================================================================

/// An embedding table stored on a single PS shard.
#[derive(Debug)]
pub struct EmbeddingTable {
    /// Table name.
    name: String,
    /// Embedding dimension.
    dim_size: usize,
    /// Backing storage. We wrap in a lock to allow concurrent RPCs.
    table: RwLock<CuckooEmbeddingHashTable>,
    /// Statistics.
    num_lookups: AtomicU64,
    num_applies: AtomicU64,
}

impl EmbeddingTable {
    /// Creates a new embedding table.
    pub fn new(name: impl Into<String>, dim_size: usize) -> Self {
        // NOTE: capacity is a knob we will want to surface via config.
        let capacity = 1_000_000;
        // Python tests expect zero-initialized embeddings on first lookup.
        let ht = CuckooEmbeddingHashTable::with_initializer(
            capacity,
            dim_size,
            Arc::new(ZerosInitializer),
        );
        Self {
            name: name.into(),
            dim_size,
            table: RwLock::new(ht),
            num_lookups: AtomicU64::new(0),
            num_applies: AtomicU64::new(0),
        }
    }

    /// Looks up embeddings for the given FIDs.
    /// Creates embeddings for missing FIDs if `create_if_missing` is true.
    pub fn lookup(&self, fids: &[i64], create_if_missing: bool) -> (Vec<f32>, Vec<bool>) {
        let mut out = vec![0.0f32; fids.len() * self.dim_size];
        let mut found = vec![false; fids.len()];

        if create_if_missing {
            let mut table = self.table.write();
            // Track which ids already existed so we can return `found` correctly.
            for (i, &fid) in fids.iter().enumerate() {
                found[i] = table.contains(fid);
            }

            // Ensure entries exist (initializer is used inside the hash table).
            if let Err(e) = table.lookup_or_initialize(fids, &mut out) {
                // For now we surface errors by returning zeroed outputs. The RPC layer
                // will encode failure via status_code/error_message in a future pass.
                tracing::error!("lookup_or_initialize failed: {e}");
            } else {
                // Any ids that were missing are now initialized; keep `found` marking
                // whether it existed *before* initialization (Python parity).
            }
        } else {
            let table = self.table.read();
            // lookup() errors if any id missing. We want per-id found flags for parity,
            // so we check contains() per id and only lookup() if all are present.
            let all_present = fids.iter().all(|&fid| table.contains(fid));
            if all_present {
                if let Err(e) = table.lookup(fids, &mut out) {
                    tracing::error!("lookup failed: {e}");
                } else {
                    found.fill(true);
                }
            } else {
                // Partial-found case: per-id lookup to keep behavior deterministic.
                // This is not fast but is correct and matches the proto contract.
                for (i, &fid) in fids.iter().enumerate() {
                    if table.contains(fid) {
                        let mut tmp = vec![0.0f32; self.dim_size];
                        if table.lookup(&[fid], &mut tmp).is_ok() {
                            let start = i * self.dim_size;
                            out[start..start + self.dim_size].copy_from_slice(&tmp);
                            found[i] = true;
                        }
                    }
                }
            }
        }

        self.num_lookups
            .fetch_add(fids.len() as u64, Ordering::Relaxed);
        (out, found)
    }

    /// Applies gradients to embeddings using SGD.
    pub fn apply_gradients(
        &self,
        fids: &[i64],
        gradients: &[f32],
        learning_rate: f32,
    ) -> (usize, usize) {
        let mut table = self.table.write();
        table.set_learning_rate(learning_rate);

        // monolith-hash-table currently errors if any id is missing, but the
        // proto response includes per-request counts. We compute counts by
        // pre-checking existence and applying updates for found ids only.
        let mut num_updated = 0usize;
        let mut num_not_found = 0usize;
        for (i, &fid) in fids.iter().enumerate() {
            if table.contains(fid) {
                let start = i * self.dim_size;
                let end = start + self.dim_size;
                let grad = &gradients[start..end];
                if table.apply_gradients(&[fid], grad).is_ok() {
                    num_updated += 1;
                }
            } else {
                num_not_found += 1;
            }
        }

        self.num_applies
            .fetch_add(fids.len() as u64, Ordering::Relaxed);
        (num_updated, num_not_found)
    }

    /// Export embeddings for a batch of FIDs (found-only).
    ///
    /// Returns `(found_fids, flat_embeddings)` where `flat_embeddings` is
    /// `found_fids.len() * dim` in row-major order.
    pub fn export_embeddings(&self, fids: &[i64]) -> (Vec<i64>, Vec<f32>) {
        let mut found_fids = Vec::new();
        let mut flat = Vec::new();

        let table = self.table.read();
        for &fid in fids {
            if table.contains(fid) {
                let mut tmp = vec![0.0f32; self.dim_size];
                if table.lookup(&[fid], &mut tmp).is_ok() {
                    found_fids.push(fid);
                    flat.extend_from_slice(&tmp);
                }
            }
        }

        (found_fids, flat)
    }

    /// Returns the number of embeddings.
    pub fn len(&self) -> usize {
        self.table.read().size()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim_size
    }

    /// Returns true if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.table.read().size() == 0
    }

    /// Returns table statistics.
    pub fn stats(&self) -> TableStats {
        let table = self.table.read();
        TableStats {
            name: self.name.clone(),
            num_embeddings: table.size() as i64,
            dim_size: self.dim_size as i32,
            memory_bytes: (table.size() * self.dim_size * 4) as i64,
        }
    }
}

// ============================================================================
// PS Server Implementation
// ============================================================================

/// Parameter Server that stores embedding tables and serves RPC requests.
pub struct PsServer {
    /// Shard ID of this PS.
    shard_id: i32,
    /// Embedding tables indexed by name.
    tables: RwLock<HashMap<String, Arc<EmbeddingTable>>>,
    /// Default embedding dimension.
    default_dim: usize,
    /// Start time for uptime calculation.
    start_time: Instant,
    /// Request counters.
    lookup_count: AtomicI64,
    apply_count: AtomicI64,
    /// Aggregate request latencies in microseconds.
    lookup_latency_us_total: AtomicI64,
    apply_latency_us_total: AtomicI64,
    /// Barrier state for synchronization.
    barriers: RwLock<HashMap<String, Arc<BarrierState>>>,
    /// Optional dirty tracker for ParameterSync replication.
    dirty_tracker: RwLock<Option<Arc<DirtyTracker>>>,
}

impl PsServer {
    /// Default embedding dimension for tables created without explicit dim_size in request.
    pub fn default_dim(&self) -> usize {
        self.default_dim
    }

    /// Attach a dirty tracker for ParameterSync replication.
    pub fn set_dirty_tracker(&self, tracker: Arc<DirtyTracker>) {
        *self.dirty_tracker.write() = Some(tracker);
    }
}

/// Spawn a tonic gRPC server serving the given PS implementation.
///
/// This is the missing piece that turns the in-crate PS implementation into a
/// real networked server (fixes the earlier "no actual RPC" parity gap).
pub async fn serve_ps(
    ps: Arc<PsServer>,
    bind_addr: std::net::SocketAddr,
) -> Result<(), tonic::transport::Error> {
    tonic::transport::Server::builder()
        .add_service(ps.into_service())
        .serve(bind_addr)
        .await
}

struct BarrierState {
    num_workers: i32,
    arrived: AtomicI64,
    state: TokioMutex<BarrierRoundState>,
    notify: Notify,
}

struct BarrierRoundState {
    generation: u64,
    arrived_workers: HashSet<i32>,
}

impl PsServer {
    /// Creates a new PS server.
    pub fn new(shard_id: i32, default_dim: usize) -> Arc<Self> {
        Arc::new(Self {
            shard_id,
            tables: RwLock::new(HashMap::new()),
            default_dim,
            start_time: Instant::now(),
            lookup_count: AtomicI64::new(0),
            apply_count: AtomicI64::new(0),
            lookup_latency_us_total: AtomicI64::new(0),
            apply_latency_us_total: AtomicI64::new(0),
            barriers: RwLock::new(HashMap::new()),
            dirty_tracker: RwLock::new(None),
        })
    }

    /// Gets or creates an embedding table.
    pub fn get_or_create_table(&self, name: &str, dim_size: usize) -> Arc<EmbeddingTable> {
        // Fast path: read lock
        {
            let tables = self.tables.read();
            if let Some(table) = tables.get(name) {
                return table.clone();
            }
        }

        // Slow path: write lock to create
        let mut tables = self.tables.write();
        tables
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(EmbeddingTable::new(name, dim_size)))
            .clone()
    }

    /// Creates a tonic service for this server.
    pub fn into_service(self: Arc<Self>) -> ParameterServerTrainingServer<PsServerHandle> {
        ParameterServerTrainingServer::new(PsServerHandle(self))
    }
}

/// Newtype wrapper for `Arc<PsServer>` to implement gRPC trait.
///
/// This is needed due to Rust's orphan rule - we cannot implement a foreign
/// trait (`ParameterServerTraining`) directly on a foreign type (`Arc<T>`).
#[derive(Clone)]
pub struct PsServerHandle(pub Arc<PsServer>);

impl std::ops::Deref for PsServerHandle {
    type Target = PsServer;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[tonic::async_trait]
impl ParameterServerTraining for PsServerHandle {
    async fn lookup(
        &self,
        request: Request<LookupRequest>,
    ) -> Result<Response<LookupResponse>, Status> {
        let started = Instant::now();
        let req = request.into_inner();
        let dim_size = if req.dim_size > 0 {
            req.dim_size as usize
        } else {
            self.default_dim
        };

        let table = self.get_or_create_table(&req.table_name, dim_size);

        let (embeddings, found) = table.lookup(&req.fids, req.create_if_missing);

        // If we created new embeddings, mark them dirty so they can be replicated online.
        if req.create_if_missing {
            if let Some(tracker) = self.dirty_tracker.read().clone() {
                let mut newly_created = Vec::new();
                for (&fid, &was_found) in req.fids.iter().zip(found.iter()) {
                    if !was_found {
                        newly_created.push(fid);
                    }
                }
                if !newly_created.is_empty() {
                    tracker.mark_dirty(&req.table_name, &newly_created);
                }
            }
        }

        let num_found = found.iter().filter(|&&f| f).count() as i32;
        let num_initialized = found.iter().filter(|&&f| !f).count() as i32;

        self.lookup_count.fetch_add(1, Ordering::Relaxed);
        let elapsed_us = started.elapsed().as_micros().max(1) as i64;
        self.lookup_latency_us_total
            .fetch_add(elapsed_us, Ordering::Relaxed);

        Ok(Response::new(LookupResponse {
            status_code: 0,
            error_message: String::new(),
            embeddings,
            found,
            num_found,
            num_initialized,
        }))
    }

    async fn apply_gradients(
        &self,
        request: Request<ApplyGradientsRequest>,
    ) -> Result<Response<ApplyGradientsResponse>, Status> {
        let started = Instant::now();
        let req = request.into_inner();
        let dim_size = if req.dim_size > 0 {
            req.dim_size as usize
        } else {
            self.default_dim
        };

        // Validate gradient size
        let expected_size = req.fids.len() * dim_size;
        if req.gradients.len() != expected_size {
            self.apply_count.fetch_add(1, Ordering::Relaxed);
            let elapsed_us = started.elapsed().as_micros().max(1) as i64;
            self.apply_latency_us_total
                .fetch_add(elapsed_us, Ordering::Relaxed);
            return Ok(Response::new(ApplyGradientsResponse {
                status_code: 1,
                error_message: format!(
                    "Gradient size mismatch: expected {}, got {}",
                    expected_size,
                    req.gradients.len()
                ),
                num_updated: 0,
                num_not_found: 0,
            }));
        }

        let table = self.get_or_create_table(&req.table_name, dim_size);
        let (num_updated, num_not_found) =
            table.apply_gradients(&req.fids, &req.gradients, req.learning_rate);

        // Mark updated fids dirty so replicator can push to online.
        if num_updated > 0 {
            if let Some(tracker) = self.dirty_tracker.read().clone() {
                // Conservative: mark all fids in request as dirty; not-found fids are
                // harmless because export filters by contains().
                tracker.mark_dirty(&req.table_name, &req.fids);
            }
        }

        self.apply_count.fetch_add(1, Ordering::Relaxed);
        let elapsed_us = started.elapsed().as_micros().max(1) as i64;
        self.apply_latency_us_total
            .fetch_add(elapsed_us, Ordering::Relaxed);

        Ok(Response::new(ApplyGradientsResponse {
            status_code: 0,
            error_message: String::new(),
            num_updated: num_updated as i32,
            num_not_found: num_not_found as i32,
        }))
    }

    async fn barrier(
        &self,
        request: Request<BarrierRequest>,
    ) -> Result<Response<BarrierResponse>, Status> {
        let req = request.into_inner();
        if req.num_workers <= 0 {
            return Ok(Response::new(BarrierResponse {
                status_code: 1,
                error_message: "num_workers must be > 0".to_string(),
                num_arrived: 0,
            }));
        }
        if req.worker_id < 0 || req.worker_id >= req.num_workers {
            return Ok(Response::new(BarrierResponse {
                status_code: 1,
                error_message: format!(
                    "worker_id {} out of range for num_workers={}",
                    req.worker_id, req.num_workers
                ),
                num_arrived: 0,
            }));
        }

        // Get or create barrier state
        let barrier_state = {
            let mut barriers = self.barriers.write();
            barriers
                .entry(req.barrier_id.clone())
                .or_insert_with(|| {
                    Arc::new(BarrierState {
                        num_workers: req.num_workers,
                        arrived: AtomicI64::new(0),
                        state: TokioMutex::new(BarrierRoundState {
                            generation: 0,
                            arrived_workers: HashSet::new(),
                        }),
                        notify: Notify::new(),
                    })
                })
                .clone()
        };

        if req.num_workers != barrier_state.num_workers {
            return Ok(Response::new(BarrierResponse {
                status_code: 1,
                error_message: format!(
                    "Barrier {} expects num_workers={}, got {}",
                    req.barrier_id, barrier_state.num_workers, req.num_workers
                ),
                num_arrived: barrier_state.arrived.load(Ordering::SeqCst) as i32,
            }));
        }

        let generation = {
            let mut state = barrier_state.state.lock().await;
            if !state.arrived_workers.insert(req.worker_id) {
                return Ok(Response::new(BarrierResponse {
                    status_code: 1,
                    error_message: format!(
                        "Worker {} already arrived for barrier {}",
                        req.worker_id, req.barrier_id
                    ),
                    num_arrived: state.arrived_workers.len() as i32,
                }));
            }
            barrier_state
                .arrived
                .store(state.arrived_workers.len() as i64, Ordering::SeqCst);

            if state.arrived_workers.len() == barrier_state.num_workers as usize {
                state.generation += 1;
                state.arrived_workers.clear();
                barrier_state.arrived.store(0, Ordering::SeqCst);
                drop(state);
                barrier_state.notify.notify_waiters();
                return Ok(Response::new(BarrierResponse {
                    status_code: 0,
                    error_message: String::new(),
                    num_arrived: barrier_state.num_workers,
                }));
            }
            state.generation
        };

        // Wait until this barrier generation releases or timeout.
        let timeout = Duration::from_millis(req.timeout_ms as u64);
        let wait_result = tokio::time::timeout(timeout, async {
            loop {
                {
                    let state = barrier_state.state.lock().await;
                    if state.generation > generation {
                        return;
                    }
                }
                barrier_state.notify.notified().await;
            }
        })
        .await;

        match wait_result {
            Ok(()) => Ok(Response::new(BarrierResponse {
                status_code: 0,
                error_message: String::new(),
                num_arrived: barrier_state.num_workers,
            })),
            Err(_) => {
                let mut state = barrier_state.state.lock().await;
                if state.generation > generation {
                    return Ok(Response::new(BarrierResponse {
                        status_code: 0,
                        error_message: String::new(),
                        num_arrived: barrier_state.num_workers,
                    }));
                }
                state.arrived_workers.remove(&req.worker_id);
                let remaining = state.arrived_workers.len() as i32;
                barrier_state.arrived.store(remaining as i64, Ordering::SeqCst);
                Ok(Response::new(BarrierResponse {
                    status_code: 1,
                    error_message: "Barrier timeout".to_string(),
                    num_arrived: remaining,
                }))
            }
        }
    }

    async fn health_check(
        &self,
        _request: Request<HealthCheckRequest>,
    ) -> Result<Response<HealthCheckResponse>, Status> {
        use monolith_proto::monolith::ps_training::health_check_response::Status as HealthStatus;

        Ok(Response::new(HealthCheckResponse {
            status: HealthStatus::Healthy as i32,
            message: "OK".to_string(),
            uptime_seconds: self.start_time.elapsed().as_secs() as i64,
        }))
    }

    async fn get_stats(
        &self,
        request: Request<GetStatsRequest>,
    ) -> Result<Response<GetStatsResponse>, Status> {
        let req = request.into_inner();

        let tables = self.tables.read();
        let mut total_embeddings = 0i64;
        let mut memory_bytes = 0i64;
        let mut table_stats = Vec::new();

        for table in tables.values() {
            let stats = table.stats();
            total_embeddings += stats.num_embeddings;
            memory_bytes += stats.memory_bytes;
            if req.include_table_stats {
                table_stats.push(stats);
            }
        }

        let lookup_count = self.lookup_count.load(Ordering::Relaxed);
        let apply_count = self.apply_count.load(Ordering::Relaxed);
        let lookup_latency_total = self.lookup_latency_us_total.load(Ordering::Relaxed);
        let apply_latency_total = self.apply_latency_us_total.load(Ordering::Relaxed);
        let avg_lookup_latency_us = if lookup_count > 0 {
            lookup_latency_total / lookup_count
        } else {
            0
        };
        let avg_apply_latency_us = if apply_count > 0 {
            apply_latency_total / apply_count
        } else {
            0
        };

        Ok(Response::new(GetStatsResponse {
            shard_id: self.shard_id,
            total_embeddings,
            memory_bytes,
            table_stats,
            lookup_count,
            apply_gradients_count: apply_count,
            avg_lookup_latency_us,
            avg_apply_latency_us,
        }))
    }
}

// ============================================================================
// PS Client Implementation (Worker-side)
// ============================================================================

/// Client for communicating with PS shards.
///
/// Handles:
/// - ID deduplication before sending
/// - Shard routing (id % num_ps)
/// - Parallel fanout to multiple shards
/// - Remapping results to original order
/// - Gradient aggregation for duplicate IDs
#[derive(Clone)]
pub struct PsClient {
    /// gRPC clients for each PS shard.
    clients: Vec<ParameterServerTrainingClient<tonic::transport::Channel>>,
    /// Number of PS shards.
    num_shards: usize,
}

impl PsClient {
    /// Connects to multiple PS instances.
    pub async fn connect(addrs: &[&str]) -> PsResult<Self> {
        if addrs.is_empty() {
            return Err(PsError::InvalidConfig(
                "at least one PS address is required".to_string(),
            ));
        }
        let mut clients = Vec::with_capacity(addrs.len());

        for addr in addrs {
            let endpoint = format!("http://{}", addr);
            let client = ParameterServerTrainingClient::connect(endpoint.clone())
                .await
                .map_err(|e| PsError::ConnectionFailed(addr.to_string(), e.to_string()))?;
            clients.push(client);
        }

        Ok(Self {
            num_shards: clients.len(),
            clients,
        })
    }

    /// Looks up embeddings for the given FIDs.
    ///
    /// Handles deduplication, shard routing, and remapping automatically.
    pub async fn lookup(
        &self,
        table_name: &str,
        fids: &[i64],
        dim_size: usize,
        create_if_missing: bool,
    ) -> PsResult<Vec<f32>> {
        let response = self
            .lookup_detailed(table_name, fids, dim_size, create_if_missing)
            .await?;
        Ok(response.embeddings)
    }

    /// Looks up embeddings and returns full response metadata.
    ///
    /// This variant exposes found/initialized counters and per-FID found flags
    /// alongside embedding vectors.
    pub async fn lookup_detailed(
        &self,
        table_name: &str,
        fids: &[i64],
        dim_size: usize,
        create_if_missing: bool,
    ) -> PsResult<LookupResponse> {
        self.lookup_response(table_name, fids, dim_size, create_if_missing)
            .await
    }

    /// Applies gradients for the given FIDs.
    ///
    /// Handles gradient aggregation for duplicate IDs automatically.
    pub async fn apply_gradients(
        &self,
        table_name: &str,
        fids: &[i64],
        gradients: &[f32],
        dim_size: usize,
        learning_rate: f32,
        global_step: i64,
    ) -> PsResult<(i32, i32)> {
        let response = self
            .apply_gradients_detailed(
                table_name,
                fids,
                gradients,
                dim_size,
                learning_rate,
                global_step,
            )
            .await?;
        Ok((response.num_updated, response.num_not_found))
    }

    /// Applies gradients and returns full response metadata.
    pub async fn apply_gradients_detailed(
        &self,
        table_name: &str,
        fids: &[i64],
        gradients: &[f32],
        dim_size: usize,
        learning_rate: f32,
        global_step: i64,
    ) -> PsResult<ApplyGradientsResponse> {
        self.apply_gradients_response(
            table_name,
            fids,
            gradients,
            dim_size,
            learning_rate,
            global_step,
        )
        .await
    }

    async fn lookup_response(
        &self,
        table_name: &str,
        fids: &[i64],
        dim_size: usize,
        create_if_missing: bool,
    ) -> PsResult<LookupResponse> {
        if dim_size == 0 {
            return Err(PsError::InvalidConfig(
                "dim_size must be greater than zero".to_string(),
            ));
        }
        if self.clients.is_empty() {
            return Err(PsError::InvalidConfig("no PS clients configured".to_string()));
        }
        if fids.is_empty() {
            return Ok(LookupResponse {
                status_code: 0,
                error_message: String::new(),
                embeddings: Vec::new(),
                found: Vec::new(),
                num_found: 0,
                num_initialized: 0,
            });
        }

        // Step 1: Deduplicate and record original positions
        let (unique_fids, original_to_unique) = dedup_ids(fids);

        // Step 2: Route unique IDs to shards
        let shard_batches = route_to_shards(&unique_fids, self.num_shards);

        // Step 3: Parallel lookup from all shards
        let mut shard_results: HashMap<usize, (Vec<f32>, Vec<bool>)> = HashMap::new();
        // Track exact (shard_id, local_idx) to avoid ad-hoc integer encoding collisions
        // for large per-shard batches.
        let mut shard_fid_to_idx: HashMap<i64, (usize, usize)> = HashMap::new();
        let mut lookup_futures = Vec::new();
        for (shard_id, shard_fids) in shard_batches.iter().enumerate() {
            if shard_fids.is_empty() {
                continue;
            }

            // Track position in this shard's results.
            for (local_idx, &fid) in shard_fids.iter().enumerate() {
                shard_fid_to_idx.insert(fid, (shard_id, local_idx));
            }

            let mut client = self.clients[shard_id].clone();
            let request = LookupRequest {
                table_name: table_name.to_string(),
                fids: shard_fids.clone(),
                dim_size: dim_size as i32,
                create_if_missing,
                timeout_ms: 5000,
            };
            lookup_futures.push(async move {
                let response = client.lookup(Request::new(request)).await?.into_inner();
                if response.status_code != 0 {
                    return Err(PsError::RpcError(Status::internal(response.error_message)));
                }
                Ok::<(usize, Vec<f32>, Vec<bool>), PsError>((shard_id, response.embeddings, response.found))
            });
        }
        let lookup_outputs = try_join_all(lookup_futures).await?;
        for (shard_id, embeddings, found) in lookup_outputs {
            shard_results.insert(shard_id, (embeddings, found));
        }

        // Step 4: Reconstruct unique embeddings/found in order
        let mut unique_embeddings = vec![0.0f32; unique_fids.len() * dim_size];
        let mut unique_found = vec![false; unique_fids.len()];
        for (unique_idx, &fid) in unique_fids.iter().enumerate() {
            if let Some(&(shard_id, local_idx)) = shard_fid_to_idx.get(&fid) {
                if let Some((shard_emb, shard_found)) = shard_results.get(&shard_id) {
                    let src_start = local_idx * dim_size;
                    let dst_start = unique_idx * dim_size;
                    unique_embeddings[dst_start..dst_start + dim_size]
                        .copy_from_slice(&shard_emb[src_start..src_start + dim_size]);
                    unique_found[unique_idx] = *shard_found.get(local_idx).unwrap_or(&false);
                }
            }
        }

        // Step 5: Remap to original order (with duplicates)
        let mut embeddings = vec![0.0f32; fids.len() * dim_size];
        let mut found = vec![false; fids.len()];
        for (orig_idx, &unique_idx) in original_to_unique.iter().enumerate() {
            let src_start = unique_idx * dim_size;
            let dst_start = orig_idx * dim_size;
            embeddings[dst_start..dst_start + dim_size]
                .copy_from_slice(&unique_embeddings[src_start..src_start + dim_size]);
            found[orig_idx] = unique_found[unique_idx];
        }

        let num_found = found.iter().filter(|&&v| v).count() as i32;
        let num_initialized = found.len() as i32 - num_found;
        Ok(LookupResponse {
            status_code: 0,
            error_message: String::new(),
            embeddings,
            found,
            num_found,
            num_initialized,
        })
    }

    async fn apply_gradients_response(
        &self,
        table_name: &str,
        fids: &[i64],
        gradients: &[f32],
        dim_size: usize,
        learning_rate: f32,
        global_step: i64,
    ) -> PsResult<ApplyGradientsResponse> {
        if dim_size == 0 {
            return Err(PsError::InvalidConfig(
                "dim_size must be greater than zero".to_string(),
            ));
        }
        if fids.is_empty() {
            return Ok(ApplyGradientsResponse {
                status_code: 0,
                error_message: String::new(),
                num_updated: 0,
                num_not_found: 0,
            });
        }
        let expected = fids.len() * dim_size;
        if gradients.len() != expected {
            return Err(PsError::DimensionMismatch {
                expected,
                actual: gradients.len(),
            });
        }
        if self.clients.is_empty() {
            return Err(PsError::InvalidConfig("no PS clients configured".to_string()));
        }

        // Step 1: Aggregate gradients for duplicate IDs
        let (unique_fids, aggregated_grads) = aggregate_gradients(fids, gradients, dim_size);

        // Step 2: Route unique IDs to shards
        let shard_batches = route_to_shards(&unique_fids, self.num_shards);

        // Build per-shard gradient batches
        let mut shard_grads: HashMap<usize, Vec<f32>> = HashMap::new();
        let fid_to_unique_idx: HashMap<i64, usize> = unique_fids
            .iter()
            .enumerate()
            .map(|(i, &fid)| (fid, i))
            .collect();

        for (shard_id, shard_fids) in shard_batches.iter().enumerate() {
            let mut grads = Vec::with_capacity(shard_fids.len() * dim_size);
            for &fid in shard_fids {
                if let Some(&unique_idx) = fid_to_unique_idx.get(&fid) {
                    let start = unique_idx * dim_size;
                    grads.extend_from_slice(&aggregated_grads[start..start + dim_size]);
                }
            }
            shard_grads.insert(shard_id, grads);
        }

        // Step 3: Parallel apply to all shards
        let mut apply_futures = Vec::new();
        for (shard_id, shard_fids) in shard_batches.into_iter().enumerate() {
            if shard_fids.is_empty() {
                continue;
            }

            let mut client = self.clients[shard_id].clone();
            let request = ApplyGradientsRequest {
                table_name: table_name.to_string(),
                fids: shard_fids,
                gradients: shard_grads.remove(&shard_id).unwrap_or_default(),
                dim_size: dim_size as i32,
                learning_rate,
                global_step,
                timeout_ms: 5000,
            };
            apply_futures.push(async move {
                let response = client.apply_gradients(Request::new(request)).await?.into_inner();
                if response.status_code != 0 {
                    return Err(PsError::RpcError(Status::internal(response.error_message)));
                }
                Ok::<(i32, i32), PsError>((response.num_updated, response.num_not_found))
            });
        }
        let apply_outputs = try_join_all(apply_futures).await?;
        let (mut total_updated, mut total_not_found) = (0i32, 0i32);
        for (updated, not_found) in apply_outputs {
            total_updated += updated;
            total_not_found += not_found;
        }

        Ok(ApplyGradientsResponse {
            status_code: 0,
            error_message: String::new(),
            num_updated: total_updated,
            num_not_found: total_not_found,
        })
    }

    /// Batched multi-table lookup helper.
    ///
    /// Each request entry is processed with the same semantics as [`Self::lookup`],
    /// and failures are encoded in per-entry response status codes.
    pub async fn batch_lookup(&self, request: BatchLookupRequest) -> PsResult<BatchLookupResponse> {
        let futures = request.requests.into_iter().map(|req| async move {
            if req.dim_size <= 0 {
                return LookupResponse {
                    status_code: 1,
                    error_message: PsError::InvalidConfig(
                        "dim_size must be greater than zero".to_string(),
                    )
                    .to_string(),
                    embeddings: Vec::new(),
                    found: Vec::new(),
                    num_found: 0,
                    num_initialized: 0,
                };
            }
            match self
                .lookup_response(
                    &req.table_name,
                    &req.fids,
                    req.dim_size as usize,
                    req.create_if_missing,
                )
                .await
            {
                Ok(resp) => resp,
                Err(e) => LookupResponse {
                    status_code: 1,
                    error_message: e.to_string(),
                    embeddings: Vec::new(),
                    found: Vec::new(),
                    num_found: 0,
                    num_initialized: 0,
                },
            }
        });
        let responses = join_all(futures).await;
        Ok(BatchLookupResponse { responses })
    }

    /// Batched multi-table apply helper.
    ///
    /// Each request entry is processed with the same semantics as [`Self::apply_gradients`],
    /// and failures are encoded in per-entry response status codes.
    pub async fn batch_apply_gradients(
        &self,
        request: BatchApplyGradientsRequest,
    ) -> PsResult<BatchApplyGradientsResponse> {
        let futures = request.requests.into_iter().map(|req| async move {
            if req.dim_size <= 0 {
                return ApplyGradientsResponse {
                    status_code: 1,
                    error_message: PsError::InvalidConfig(
                        "dim_size must be greater than zero".to_string(),
                    )
                    .to_string(),
                    num_updated: 0,
                    num_not_found: 0,
                };
            }
            match self
                .apply_gradients_response(
                    &req.table_name,
                    &req.fids,
                    &req.gradients,
                    req.dim_size as usize,
                    req.learning_rate,
                    req.global_step,
                )
                .await
            {
                Ok(resp) => resp,
                Err(e) => ApplyGradientsResponse {
                    status_code: 1,
                    error_message: e.to_string(),
                    num_updated: 0,
                    num_not_found: 0,
                },
            }
        });
        let responses = join_all(futures).await;
        Ok(BatchApplyGradientsResponse { responses })
    }

    /// Waits at a barrier for all workers.
    pub async fn barrier(
        &self,
        barrier_id: &str,
        worker_id: i32,
        num_workers: i32,
        timeout_ms: i64,
    ) -> PsResult<()> {
        self.barrier_on_shard(0, barrier_id, worker_id, num_workers, timeout_ms)
            .await
    }

    /// Waits at a barrier using a specific PS shard as coordinator.
    pub async fn barrier_on_shard(
        &self,
        shard_id: usize,
        barrier_id: &str,
        worker_id: i32,
        num_workers: i32,
        timeout_ms: i64,
    ) -> PsResult<()> {
        if timeout_ms <= 0 {
            return Err(PsError::InvalidConfig(
                "timeout_ms must be greater than zero".to_string(),
            ));
        }
        if num_workers <= 0 {
            return Err(PsError::InvalidConfig(
                "num_workers must be greater than zero".to_string(),
            ));
        }
        if worker_id < 0 || worker_id >= num_workers {
            return Err(PsError::InvalidConfig(format!(
                "worker_id {} out of range for num_workers={}",
                worker_id, num_workers
            )));
        }
        if self.clients.is_empty() {
            return Err(PsError::InvalidConfig("no PS clients configured".to_string()));
        }
        let mut client = self
            .clients
            .get(shard_id)
            .cloned()
            .ok_or_else(|| PsError::InvalidConfig(format!("invalid shard index: {}", shard_id)))?;

        let request = BarrierRequest {
            barrier_id: barrier_id.to_string(),
            worker_id,
            num_workers,
            timeout_ms,
        };

        let response = client
            .barrier(Request::new(request))
            .await?
            .into_inner();

        match response.status_code {
            0 => Ok(()),
            1 => {
                if response.error_message.to_lowercase().contains("timeout") {
                    Err(PsError::Timeout(Duration::from_millis(timeout_ms as u64)))
                } else {
                    Err(PsError::InvalidConfig(response.error_message))
                }
            }
            2 => Err(PsError::RpcError(Status::cancelled(response.error_message))),
            _ => Err(PsError::RpcError(Status::internal(response.error_message))),
        }
    }

    /// Returns the number of PS shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Performs health check against default coordinator shard (index 0).
    pub async fn health_check(&self, component: impl Into<String>) -> PsResult<HealthCheckResponse> {
        self.health_check_shard(0, component).await
    }

    /// Performs health check against one PS shard.
    pub async fn health_check_shard(
        &self,
        shard_id: usize,
        component: impl Into<String>,
    ) -> PsResult<HealthCheckResponse> {
        if self.clients.is_empty() {
            return Err(PsError::InvalidConfig("no PS clients configured".to_string()));
        }
        let mut client = self
            .clients
            .get(shard_id)
            .cloned()
            .ok_or_else(|| PsError::InvalidConfig(format!("invalid shard index: {}", shard_id)))?;
        let response = client
            .health_check(Request::new(HealthCheckRequest {
                component: component.into(),
            }))
            .await?
            .into_inner();
        Ok(response)
    }

    /// Performs health checks against all shards in parallel.
    pub async fn health_check_all(
        &self,
        component: impl Into<String>,
    ) -> PsResult<Vec<HealthCheckResponse>> {
        if self.clients.is_empty() {
            return Err(PsError::InvalidConfig("no PS clients configured".to_string()));
        }
        let component = component.into();
        let mut checks = Vec::with_capacity(self.clients.len());
        for mut client in self.clients.iter().cloned() {
            let component = component.clone();
            checks.push(async move {
                let response = client
                    .health_check(Request::new(HealthCheckRequest { component }))
                    .await?
                    .into_inner();
                Ok::<HealthCheckResponse, PsError>(response)
            });
        }
        try_join_all(checks).await
    }

    /// Gets stats from one PS shard.
    pub async fn get_stats_shard(
        &self,
        shard_id: usize,
        include_table_stats: bool,
    ) -> PsResult<GetStatsResponse> {
        if self.clients.is_empty() {
            return Err(PsError::InvalidConfig("no PS clients configured".to_string()));
        }
        let mut client = self
            .clients
            .get(shard_id)
            .cloned()
            .ok_or_else(|| PsError::InvalidConfig(format!("invalid shard index: {}", shard_id)))?;
        let response = client
            .get_stats(Request::new(GetStatsRequest { include_table_stats }))
            .await?
            .into_inner();
        Ok(response)
    }

    /// Gets stats from default coordinator shard (index 0).
    pub async fn get_stats(&self, include_table_stats: bool) -> PsResult<GetStatsResponse> {
        self.get_stats_shard(0, include_table_stats).await
    }

    /// Gets stats from all PS shards in parallel.
    pub async fn get_stats_all(&self, include_table_stats: bool) -> PsResult<Vec<GetStatsResponse>> {
        if self.clients.is_empty() {
            return Err(PsError::InvalidConfig("no PS clients configured".to_string()));
        }
        let mut stats_calls = Vec::with_capacity(self.clients.len());
        for mut client in self.clients.iter().cloned() {
            stats_calls.push(async move {
                let response = client
                    .get_stats(Request::new(GetStatsRequest { include_table_stats }))
                    .await?
                    .into_inner();
                Ok::<GetStatsResponse, PsError>(response)
            });
        }
        try_join_all(stats_calls).await
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Deduplicates IDs and returns mapping from original to unique indices.
pub fn dedup_ids(ids: &[i64]) -> (Vec<i64>, Vec<usize>) {
    let mut unique_ids = Vec::new();
    let mut id_to_idx: HashMap<i64, usize> = HashMap::new();
    let mut original_to_unique = Vec::with_capacity(ids.len());

    for &id in ids {
        let unique_idx = *id_to_idx.entry(id).or_insert_with(|| {
            let idx = unique_ids.len();
            unique_ids.push(id);
            idx
        });
        original_to_unique.push(unique_idx);
    }

    (unique_ids, original_to_unique)
}

/// Routes IDs to shards using modulo routing.
pub fn route_to_shards(ids: &[i64], num_shards: usize) -> Vec<Vec<i64>> {
    let mut shards = vec![Vec::new(); num_shards];
    for &id in ids {
        // Python uses `tf.math.floormod`, which matches Rust's `rem_euclid`.
        let shard = (id.rem_euclid(num_shards as i64)) as usize;
        shards[shard].push(id);
    }
    shards
}

/// Aggregates gradients for duplicate IDs by summing.
pub fn aggregate_gradients(
    fids: &[i64],
    gradients: &[f32],
    dim_size: usize,
) -> (Vec<i64>, Vec<f32>) {
    let mut unique_fids = Vec::new();
    let mut fid_to_idx: HashMap<i64, usize> = HashMap::new();
    let mut aggregated: Vec<f32> = Vec::new();

    for (i, &fid) in fids.iter().enumerate() {
        let grad_start = i * dim_size;
        let grad = &gradients[grad_start..grad_start + dim_size];

        if let Some(&idx) = fid_to_idx.get(&fid) {
            // Add to existing
            let agg_start = idx * dim_size;
            for (j, &g) in grad.iter().enumerate() {
                aggregated[agg_start + j] += g;
            }
        } else {
            // New unique ID
            let idx = unique_fids.len();
            fid_to_idx.insert(fid, idx);
            unique_fids.push(fid);
            aggregated.extend_from_slice(grad);
        }
    }

    (unique_fids, aggregated)
}

/// Gets the shard index for a given FID.
pub fn get_shard_for_id(fid: i64, num_shards: usize) -> usize {
    if num_shards == 0 {
        return 0;
    }
    (fid.rem_euclid(num_shards as i64)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use monolith_proto::monolith::ps_training::{
        ApplyGradientsRequest, GetStatsRequest, LookupRequest,
    };
    use tonic::Request;

    #[test]
    fn test_dedup_ids() {
        let ids = vec![1, 2, 1, 3, 2, 1];
        let (unique, mapping) = dedup_ids(&ids);

        assert_eq!(unique, vec![1, 2, 3]);
        assert_eq!(mapping, vec![0, 1, 0, 2, 1, 0]);
    }

    #[test]
    fn test_route_to_shards() {
        let ids = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let shards = route_to_shards(&ids, 3);

        assert_eq!(shards[0], vec![0, 3, 6]); // ids % 3 == 0
        assert_eq!(shards[1], vec![1, 4, 7]); // ids % 3 == 1
        assert_eq!(shards[2], vec![2, 5]); // ids % 3 == 2
    }

    #[test]
    fn test_aggregate_gradients() {
        let fids = vec![1, 2, 1]; // ID 1 appears twice
        let gradients = vec![
            1.0, 2.0, // grad for fid 1 (first)
            3.0, 4.0, // grad for fid 2
            0.5, 0.5, // grad for fid 1 (second, should be summed)
        ];
        let dim_size = 2;

        let (unique_fids, agg_grads) = aggregate_gradients(&fids, &gradients, dim_size);

        assert_eq!(unique_fids, vec![1, 2]);
        assert_eq!(
            agg_grads,
            vec![
                1.5, 2.5, // 1.0+0.5, 2.0+0.5 for fid 1
                3.0, 4.0, // unchanged for fid 2
            ]
        );
    }

    #[test]
    fn test_embedding_table() {
        let table = EmbeddingTable::new("test", 4);

        // Lookup with create
        let (emb, found) = table.lookup(&[1, 2, 3], true);
        assert_eq!(emb.len(), 12); // 3 * 4
                                   // Parity with Python: `found` indicates pre-existence. Newly created rows return `false`.
        assert_eq!(found, vec![false, false, false]);

        // Lookup existing
        let (emb2, found2) = table.lookup(&[1, 2], false);
        assert_eq!(emb2.len(), 8);
        assert_eq!(found2, vec![true, true]); // All found

        // Check embeddings are consistent
        assert_eq!(&emb[0..4], &emb2[0..4]); // Same embedding for fid 1
    }

    #[test]
    fn test_embedding_table_apply_gradients() {
        let table = EmbeddingTable::new("test", 2);

        // Create embeddings
        table.lookup(&[1], true);

        // Get initial values
        let (initial, _) = table.lookup(&[1], false);

        // Apply gradients
        let (updated, not_found) = table.apply_gradients(&[1], &[0.1, 0.2], 0.5);
        assert_eq!(updated, 1);
        assert_eq!(not_found, 0);

        // Check values changed
        let (after, _) = table.lookup(&[1], false);
        assert!((after[0] - (initial[0] - 0.05)).abs() < 1e-6);
        assert!((after[1] - (initial[1] - 0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_get_shard_for_id() {
        assert_eq!(get_shard_for_id(0, 3), 0);
        assert_eq!(get_shard_for_id(1, 3), 1);
        assert_eq!(get_shard_for_id(2, 3), 2);
        assert_eq!(get_shard_for_id(3, 3), 0);
        assert_eq!(get_shard_for_id(-1, 3), 2); // floormod(-1, 3) = 2
        assert_eq!(get_shard_for_id(100, 0), 0); // edge case
    }

    #[tokio::test]
    async fn test_ps_server_stats_tracks_average_latency() {
        let ps = PsServer::new(0, 2);
        let handle = PsServerHandle(ps.clone());

        let _ = handle
            .lookup(Request::new(LookupRequest {
                table_name: "latency_table".to_string(),
                fids: vec![1, 2],
                dim_size: 2,
                create_if_missing: true,
                timeout_ms: 1000,
            }))
            .await
            .unwrap();

        let _ = handle
            .apply_gradients(Request::new(ApplyGradientsRequest {
                table_name: "latency_table".to_string(),
                fids: vec![1],
                gradients: vec![0.1, 0.2],
                dim_size: 2,
                learning_rate: 0.5,
                global_step: 1,
                timeout_ms: 1000,
            }))
            .await
            .unwrap();

        let stats = handle
            .get_stats(Request::new(GetStatsRequest {
                include_table_stats: true,
            }))
            .await
            .unwrap()
            .into_inner();

        assert_eq!(stats.lookup_count, 1);
        assert_eq!(stats.apply_gradients_count, 1);
        assert!(stats.avg_lookup_latency_us >= 1);
        assert!(stats.avg_apply_latency_us >= 1);
        assert!(stats.total_embeddings >= 2);
    }

    #[tokio::test]
    async fn test_ps_server_stats_counts_failed_apply_requests() {
        let ps = PsServer::new(0, 2);
        let handle = PsServerHandle(ps);

        let bad = handle
            .apply_gradients(Request::new(ApplyGradientsRequest {
                table_name: "latency_table".to_string(),
                fids: vec![1],
                gradients: vec![0.1], // wrong dim (expected 2)
                dim_size: 2,
                learning_rate: 0.5,
                global_step: 1,
                timeout_ms: 1000,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(bad.status_code, 1);

        let stats = handle
            .get_stats(Request::new(GetStatsRequest {
                include_table_stats: false,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(stats.apply_gradients_count, 1);
        assert!(stats.avg_apply_latency_us >= 1);
    }

    #[tokio::test]
    async fn test_ps_server_barrier_success_and_reset() {
        let ps = PsServer::new(0, 2);
        let handle = PsServerHandle(ps);

        let req0 = BarrierRequest {
            barrier_id: "b0".to_string(),
            worker_id: 0,
            num_workers: 2,
            timeout_ms: 200,
        };
        let req1 = BarrierRequest {
            worker_id: 1,
            ..req0.clone()
        };
        let (r1, r2) = tokio::join!(
            handle.barrier(Request::new(req0.clone())),
            handle.barrier(Request::new(req1.clone()))
        );
        let r1 = r1.unwrap().into_inner();
        let r2 = r2.unwrap().into_inner();
        assert_eq!(r1.status_code, 0);
        assert_eq!(r2.status_code, 0);
        assert_eq!(r1.num_arrived, 2);
        assert_eq!(r2.num_arrived, 2);

        // Reuse same barrier id for next round; should still work.
        let (r3, r4) = tokio::join!(
            handle.barrier(Request::new(req0)),
            handle.barrier(Request::new(req1))
        );
        assert_eq!(r3.unwrap().into_inner().status_code, 0);
        assert_eq!(r4.unwrap().into_inner().status_code, 0);
    }

    #[tokio::test]
    async fn test_ps_server_barrier_num_workers_mismatch() {
        let ps = PsServer::new(0, 2);
        let handle = PsServerHandle(ps);

        // Initialize barrier id with num_workers=1 (immediate success).
        let ok = handle
            .barrier(Request::new(BarrierRequest {
                barrier_id: "bmismatch".to_string(),
                worker_id: 0,
                num_workers: 1,
                timeout_ms: 50,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(ok.status_code, 0);

        // Mismatched participant count should return explicit error.
        let bad = handle
            .barrier(Request::new(BarrierRequest {
                barrier_id: "bmismatch".to_string(),
                worker_id: 1,
                num_workers: 2,
                timeout_ms: 50,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(bad.status_code, 1);
        assert!(bad.error_message.contains("expects num_workers=1"));
    }

    #[tokio::test]
    async fn test_ps_server_barrier_worker_id_range_validation() {
        let ps = PsServer::new(0, 2);
        let handle = PsServerHandle(ps);

        let bad = handle
            .barrier(Request::new(BarrierRequest {
                barrier_id: "bwid".to_string(),
                worker_id: 2, // out of [0,2)
                num_workers: 2,
                timeout_ms: 50,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(bad.status_code, 1);
        assert!(bad.error_message.contains("out of range"));
    }

    #[tokio::test]
    async fn test_ps_server_barrier_duplicate_worker_rejected() {
        let ps = PsServer::new(0, 2);
        let handle = PsServerHandle(ps);

        let req = BarrierRequest {
            barrier_id: "bdup".to_string(),
            worker_id: 0,
            num_workers: 2,
            timeout_ms: 300,
        };
        let first = {
            let handle_bg = handle.clone();
            let req_bg = req.clone();
            tokio::spawn(async move { handle_bg.barrier(Request::new(req_bg)).await })
        };
        tokio::time::sleep(Duration::from_millis(30)).await;

        let dup = handle
            .barrier(Request::new(req))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(dup.status_code, 1);
        assert!(dup.error_message.contains("already arrived"));

        let peer = handle
            .barrier(Request::new(BarrierRequest {
                barrier_id: "bdup".to_string(),
                worker_id: 1,
                num_workers: 2,
                timeout_ms: 300,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(peer.status_code, 0);

        let first = first.await.unwrap().unwrap().into_inner();
        assert_eq!(first.status_code, 0);
    }

    #[tokio::test]
    async fn test_ps_server_barrier_timeout_cleanup_allows_retry() {
        let ps = PsServer::new(0, 2);
        let handle = PsServerHandle(ps);

        let timeout = handle
            .barrier(Request::new(BarrierRequest {
                barrier_id: "btimeout".to_string(),
                worker_id: 0,
                num_workers: 2,
                timeout_ms: 20,
            }))
            .await
            .unwrap()
            .into_inner();
        assert_eq!(timeout.status_code, 1);
        assert_eq!(timeout.num_arrived, 0);

        let req0 = BarrierRequest {
            barrier_id: "btimeout".to_string(),
            worker_id: 0,
            num_workers: 2,
            timeout_ms: 200,
        };
        let req1 = BarrierRequest {
            worker_id: 1,
            ..req0.clone()
        };
        let (r1, r2) = tokio::join!(
            handle.barrier(Request::new(req0)),
            handle.barrier(Request::new(req1))
        );
        assert_eq!(r1.unwrap().into_inner().status_code, 0);
        assert_eq!(r2.unwrap().into_inner().status_code, 0);
    }

    #[tokio::test]
    async fn test_ps_client_lookup_and_apply_across_shards() {
        let bind0 = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let bind1 = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();

        let ps0 = PsServer::new(0, 2);
        let ps1 = PsServer::new(1, 2);
        let h0 = tokio::spawn(async move {
            let _ = serve_ps(ps0, bind0).await;
        });
        let h1 = tokio::spawn(async move {
            let _ = serve_ps(ps1, bind1).await;
        });

        tokio::time::sleep(Duration::from_millis(80)).await;

        let addr0 = bind0.to_string();
        let addr1 = bind1.to_string();
        let client = PsClient::connect(&[&addr0, &addr1]).await.unwrap();

        let initial = client.lookup("emb", &[0, 1, 2, 3], 2, true).await.unwrap();
        assert_eq!(initial, vec![0.0; 8]);

        let (updated, not_found) = client
            .apply_gradients(
                "emb",
                &[0, 1, 0, 3],
                &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                2,
                0.5,
                1,
            )
            .await
            .unwrap();
        assert_eq!(updated, 3);
        assert_eq!(not_found, 0);

        let after = client.lookup("emb", &[0, 1, 3], 2, false).await.unwrap();
        assert_eq!(after, vec![-3.0, -4.0, -1.5, -2.0, -3.5, -4.0]);

        h0.abort();
        h1.abort();
    }

    #[tokio::test]
    async fn test_ps_client_detailed_lookup_and_apply_metadata() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();

        let first_lookup = client.lookup_detailed("meta", &[7, 7, 9], 2, true).await.unwrap();
        assert_eq!(first_lookup.status_code, 0);
        assert_eq!(first_lookup.found, vec![false, false, false]);
        assert_eq!(first_lookup.num_found, 0);
        assert_eq!(first_lookup.num_initialized, 3);

        let second_lookup = client.lookup_detailed("meta", &[7, 9], 2, false).await.unwrap();
        assert_eq!(second_lookup.status_code, 0);
        assert_eq!(second_lookup.found, vec![true, true]);
        assert_eq!(second_lookup.num_found, 2);
        assert_eq!(second_lookup.num_initialized, 0);

        let apply_resp = client
            .apply_gradients_detailed("meta", &[7, 9], &[1.0, 2.0, 3.0, 4.0], 2, 0.1, 1)
            .await
            .unwrap();
        assert_eq!(apply_resp.status_code, 0);
        assert_eq!(apply_resp.num_updated, 2);
        assert_eq!(apply_resp.num_not_found, 0);

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_connect_requires_addresses() {
        let res = PsClient::connect(&[]).await;
        assert!(matches!(res, Err(PsError::InvalidConfig(_))));
    }

    #[tokio::test]
    async fn test_ps_client_supports_parallel_immutable_lookups() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();
        client.lookup("immut", &[1, 2], 2, true).await.unwrap();

        let (left, right) = tokio::join!(
            client.lookup("immut", &[1], 2, false),
            client.lookup("immut", &[2], 2, false)
        );
        assert_eq!(left.unwrap(), vec![0.0, 0.0]);
        assert_eq!(right.unwrap(), vec![0.0, 0.0]);

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_lookup_errors_without_clients() {
        let client = PsClient {
            clients: Vec::new(),
            num_shards: 0,
        };
        let err = client.lookup("emb", &[1], 2, true).await.unwrap_err();
        assert!(matches!(err, PsError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_ps_client_lookup_rejects_zero_dim() {
        let client = PsClient {
            clients: Vec::new(),
            num_shards: 1,
        };
        let err = client.lookup("emb", &[1], 0, true).await.unwrap_err();
        assert!(matches!(err, PsError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_ps_client_apply_rejects_gradient_size_mismatch() {
        let client = PsClient {
            clients: Vec::new(),
            num_shards: 1,
        };
        let err = client
            .apply_gradients("emb", &[1, 2], &[0.1, 0.2, 0.3], 2, 0.1, 1)
            .await
            .unwrap_err();
        assert!(matches!(
            err,
            PsError::DimensionMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[tokio::test]
    async fn test_ps_client_barrier_rejects_invalid_worker_range() {
        let client = PsClient {
            clients: Vec::new(),
            num_shards: 1,
        };
        let err = client.barrier("b", 2, 2, 100).await.unwrap_err();
        assert!(matches!(err, PsError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_ps_client_barrier_rejects_non_positive_timeout() {
        let client = PsClient {
            clients: Vec::new(),
            num_shards: 0,
        };
        let err = client.barrier("b", 0, 1, 0).await.unwrap_err();
        assert!(matches!(err, PsError::InvalidConfig(_)));
    }

    #[tokio::test]
    async fn test_ps_client_barrier_on_shard_rejects_invalid_index() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();
        let err = client
            .barrier_on_shard(1, "bshard", 0, 1, 100)
            .await
            .unwrap_err();
        assert!(matches!(err, PsError::InvalidConfig(_)));
        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_barrier_on_shard_routes_to_selected_coordinator() {
        let bind0 = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let bind1 = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();

        let ps0 = PsServer::new(0, 2);
        let ps1 = PsServer::new(1, 2);
        let h0 = tokio::spawn(async move {
            let _ = serve_ps(ps0, bind0).await;
        });
        let h1 = tokio::spawn(async move {
            let _ = serve_ps(ps1, bind1).await;
        });
        tokio::time::sleep(Duration::from_millis(80)).await;

        let addr0 = bind0.to_string();
        let addr1 = bind1.to_string();
        let client = PsClient::connect(&[&addr0, &addr1]).await.unwrap();

        // Use shard 1 as explicit barrier coordinator.
        client
            .barrier_on_shard(1, "explicit_shard", 0, 1, 200)
            .await
            .unwrap();
        // Default barrier should still use shard 0 and succeed independently.
        client.barrier("default_shard0", 0, 1, 200).await.unwrap();

        h0.abort();
        h1.abort();
    }

    #[tokio::test]
    async fn test_ps_client_barrier_maps_timeout_error() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();
        let err = client.barrier("bt", 0, 2, 20).await.unwrap_err();
        assert!(matches!(err, PsError::Timeout(_)));
        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_barrier_maps_mismatch_to_invalid_config() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();
        // Initializes barrier "bm" with num_workers=1.
        client.barrier("bm", 0, 1, 200).await.unwrap();
        // Reusing same barrier id with incompatible num_workers should become InvalidConfig.
        let err = client.barrier("bm", 0, 2, 200).await.unwrap_err();
        assert!(matches!(err, PsError::InvalidConfig(_)));
        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_health_and_stats_shard_methods() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();

        let health = client.health_check_shard(0, "ps").await.unwrap();
        assert_eq!(
            health.status,
            monolith_proto::monolith::ps_training::health_check_response::Status::Healthy as i32
        );

        let _ = client.lookup("emb", &[1, 2], 2, true).await.unwrap();
        let stats = client.get_stats_shard(0, true).await.unwrap();
        assert_eq!(stats.shard_id, 0);
        assert!(stats.lookup_count >= 1);
        assert!(stats.total_embeddings >= 2);
        assert!(!stats.table_stats.is_empty());
        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_default_health_and_stats_methods() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();
        let health = client.health_check("ps").await.unwrap();
        assert_eq!(
            health.status,
            monolith_proto::monolith::ps_training::health_check_response::Status::Healthy as i32
        );

        let _ = client.lookup("emb", &[1], 2, true).await.unwrap();
        let stats = client.get_stats(false).await.unwrap();
        assert_eq!(stats.shard_id, 0);
        assert!(stats.lookup_count >= 1);

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_health_and_stats_all_methods() {
        let bind0 = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let bind1 = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();

        let ps0 = PsServer::new(0, 2);
        let ps1 = PsServer::new(1, 2);
        let h0 = tokio::spawn(async move {
            let _ = serve_ps(ps0, bind0).await;
        });
        let h1 = tokio::spawn(async move {
            let _ = serve_ps(ps1, bind1).await;
        });
        tokio::time::sleep(Duration::from_millis(80)).await;

        let addr0 = bind0.to_string();
        let addr1 = bind1.to_string();
        let client = PsClient::connect(&[&addr0, &addr1]).await.unwrap();

        let healths = client.health_check_all("ps").await.unwrap();
        assert_eq!(healths.len(), 2);
        assert!(healths.iter().all(|h| {
            h.status
                == monolith_proto::monolith::ps_training::health_check_response::Status::Healthy
                    as i32
        }));

        let stats = client.get_stats_all(false).await.unwrap();
        assert_eq!(stats.len(), 2);
        assert!(stats.iter().any(|s| s.shard_id == 0));
        assert!(stats.iter().any(|s| s.shard_id == 1));
        h0.abort();
        h1.abort();
    }

    #[tokio::test]
    async fn test_ps_client_stats_shard_index_validation() {
        let client = PsClient {
            clients: Vec::new(),
            num_shards: 0,
        };
        assert!(matches!(
            client.get_stats_shard(0, false).await,
            Err(PsError::InvalidConfig(_))
        ));
        assert!(matches!(
            client.health_check_shard(0, "ps").await,
            Err(PsError::InvalidConfig(_))
        ));
    }

    #[tokio::test]
    async fn test_ps_client_batch_lookup_and_apply() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();

        let lookup_batch = BatchLookupRequest {
            requests: vec![
                LookupRequest {
                    table_name: "t1".to_string(),
                    fids: vec![1, 2],
                    dim_size: 2,
                    create_if_missing: true,
                    timeout_ms: 1000,
                },
                LookupRequest {
                    table_name: "t2".to_string(),
                    fids: vec![3],
                    dim_size: 2,
                    create_if_missing: true,
                    timeout_ms: 1000,
                },
            ],
        };
        let lookup_resp = client.batch_lookup(lookup_batch).await.unwrap();
        assert_eq!(lookup_resp.responses.len(), 2);
        assert!(lookup_resp.responses.iter().all(|r| r.status_code == 0));
        assert_eq!(lookup_resp.responses[0].embeddings.len(), 4);
        assert_eq!(lookup_resp.responses[1].embeddings.len(), 2);

        let apply_batch = BatchApplyGradientsRequest {
            requests: vec![
                ApplyGradientsRequest {
                    table_name: "t1".to_string(),
                    fids: vec![1, 2],
                    gradients: vec![1.0, 2.0, 3.0, 4.0],
                    dim_size: 2,
                    learning_rate: 0.5,
                    global_step: 1,
                    timeout_ms: 1000,
                },
                // Intentionally malformed gradient shape.
                ApplyGradientsRequest {
                    table_name: "t2".to_string(),
                    fids: vec![3],
                    gradients: vec![1.0],
                    dim_size: 2,
                    learning_rate: 0.5,
                    global_step: 1,
                    timeout_ms: 1000,
                },
            ],
        };
        let apply_resp = client.batch_apply_gradients(apply_batch).await.unwrap();
        assert_eq!(apply_resp.responses.len(), 2);
        assert_eq!(apply_resp.responses[0].status_code, 0);
        assert_eq!(apply_resp.responses[1].status_code, 1);

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_batch_lookup_preserves_duplicate_found_flags() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();

        // First lookup initializes missing IDs and should report all as newly initialized.
        let first = client
            .batch_lookup(BatchLookupRequest {
                requests: vec![LookupRequest {
                    table_name: "dup".to_string(),
                    fids: vec![10, 10, 11],
                    dim_size: 2,
                    create_if_missing: true,
                    timeout_ms: 1000,
                }],
            })
            .await
            .unwrap();
        assert_eq!(first.responses.len(), 1);
        let first_resp = &first.responses[0];
        assert_eq!(first_resp.status_code, 0);
        assert_eq!(first_resp.found, vec![false, false, false]);
        assert_eq!(first_resp.num_found, 0);
        assert_eq!(first_resp.num_initialized, 3);

        // Second lookup should see all IDs (including duplicates) as found.
        let second = client
            .batch_lookup(BatchLookupRequest {
                requests: vec![LookupRequest {
                    table_name: "dup".to_string(),
                    fids: vec![10, 10, 11],
                    dim_size: 2,
                    create_if_missing: false,
                    timeout_ms: 1000,
                }],
            })
            .await
            .unwrap();
        let second_resp = &second.responses[0];
        assert_eq!(second_resp.status_code, 0);
        assert_eq!(second_resp.found, vec![true, true, true]);
        assert_eq!(second_resp.num_found, 3);
        assert_eq!(second_resp.num_initialized, 0);

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_batch_lookup_validates_dim_size_per_entry() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();

        let resp = client
            .batch_lookup(BatchLookupRequest {
                requests: vec![
                    LookupRequest {
                        table_name: "dim".to_string(),
                        fids: vec![1],
                        dim_size: 0,
                        create_if_missing: true,
                        timeout_ms: 1000,
                    },
                    LookupRequest {
                        table_name: "dim".to_string(),
                        fids: vec![2],
                        dim_size: 2,
                        create_if_missing: true,
                        timeout_ms: 1000,
                    },
                ],
            })
            .await
            .unwrap();

        assert_eq!(resp.responses.len(), 2);
        assert_eq!(resp.responses[0].status_code, 1);
        assert!(
            resp.responses[0]
                .error_message
                .contains("dim_size must be greater than zero")
        );
        assert_eq!(resp.responses[1].status_code, 0);
        assert_eq!(resp.responses[1].embeddings.len(), 2);

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_client_batch_apply_validates_dim_size_per_entry() {
        let bind = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap();
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr]).await.unwrap();

        // Initialize valid table entry so the second apply request can update it.
        client.lookup("dim_apply", &[9], 2, true).await.unwrap();

        let resp = client
            .batch_apply_gradients(BatchApplyGradientsRequest {
                requests: vec![
                    ApplyGradientsRequest {
                        table_name: "dim_apply".to_string(),
                        fids: vec![9],
                        gradients: vec![1.0, 2.0],
                        dim_size: -1,
                        learning_rate: 0.1,
                        global_step: 1,
                        timeout_ms: 1000,
                    },
                    ApplyGradientsRequest {
                        table_name: "dim_apply".to_string(),
                        fids: vec![9],
                        gradients: vec![1.0, 2.0],
                        dim_size: 2,
                        learning_rate: 0.1,
                        global_step: 1,
                        timeout_ms: 1000,
                    },
                ],
            })
            .await
            .unwrap();

        assert_eq!(resp.responses.len(), 2);
        assert_eq!(resp.responses[0].status_code, 1);
        assert!(
            resp.responses[0]
                .error_message
                .contains("dim_size must be greater than zero")
        );
        assert_eq!(resp.responses[1].status_code, 0);
        assert_eq!(resp.responses[1].num_updated, 1);

        server.abort();
    }
}
