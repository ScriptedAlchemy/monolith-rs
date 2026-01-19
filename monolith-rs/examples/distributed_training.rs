//! Distributed Training Simulation Example
//!
//! This example demonstrates the distributed training architecture of Monolith
//! using a single-process simulation with threads. It shows:
//!
//! - Parameter Server (PS) architecture for embedding storage
//! - Worker coordination for gradient computation
//! - Service discovery using InMemoryDiscovery
//! - Synchronous and asynchronous SGD patterns
//! - Model sharding with MultiHashTable
//! - Periodic checkpointing during training
//!
//! # Architecture
//!
//! ```text
//!                    ┌─────────────────────┐
//!                    │   Service Discovery │
//!                    │  (InMemoryDiscovery)│
//!                    └──────────┬──────────┘
//!                               │
//!          ┌────────────────────┼────────────────────┐
//!          │                    │                    │
//!          ▼                    ▼                    ▼
//!    ┌──────────┐        ┌──────────┐         ┌──────────┐
//!    │   PS 0   │        │   PS 1   │   ...   │  PS N-1  │
//!    │ (Shard)  │        │ (Shard)  │         │ (Shard)  │
//!    └────┬─────┘        └────┬─────┘         └────┬─────┘
//!         │                   │                    │
//!         └───────────────────┼────────────────────┘
//!                             │
//!          ┌──────────────────┼──────────────────┐
//!          │                  │                  │
//!          ▼                  ▼                  ▼
//!    ┌──────────┐       ┌──────────┐       ┌──────────┐
//!    │ Worker 0 │       │ Worker 1 │  ...  │ Worker M │
//!    └──────────┘       └──────────┘       └──────────┘
//! ```
//!
//! # Usage
//!
//! ```bash
//! cargo run --example distributed_training -- --help
//! cargo run --example distributed_training -- --num-workers 4 --num-ps 2 --sync-mode sync
//! cargo run --example distributed_training -- --num-workers 8 --sync-mode async --num-steps 1000
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Barrier, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

use clap::Parser;
use rand::prelude::*;

// We'll define our own types since we're simulating the distributed system

/// Command-line arguments for the distributed training example.
#[derive(Parser, Debug)]
#[command(name = "distributed_training")]
#[command(about = "Simulates distributed training with parameter servers and workers")]
struct Args {
    /// Number of worker threads
    #[arg(long, default_value = "4")]
    num_workers: usize,

    /// Number of parameter server shards
    #[arg(long, default_value = "2")]
    num_ps: usize,

    /// Synchronization mode: 'sync' or 'async'
    #[arg(long, default_value = "sync")]
    sync_mode: String,

    /// Batch size per worker
    #[arg(long, default_value = "32")]
    batch_size: usize,

    /// Number of training steps
    #[arg(long, default_value = "100")]
    num_steps: u64,

    /// Embedding dimension
    #[arg(long, default_value = "8")]
    embedding_dim: usize,

    /// Number of unique feature IDs to simulate
    #[arg(long, default_value = "1000")]
    num_features: i64,

    /// Checkpoint interval (steps)
    #[arg(long, default_value = "20")]
    checkpoint_interval: u64,

    /// Checkpoint directory
    #[arg(long, default_value = "/tmp/distributed_training_checkpoints")]
    checkpoint_dir: PathBuf,

    /// Learning rate
    #[arg(long, default_value = "0.01")]
    learning_rate: f32,

    /// Log interval (steps)
    #[arg(long, default_value = "10")]
    log_interval: u64,
}

// ============================================================================
// Service Discovery (simplified in-memory version)
// ============================================================================

/// Health status of a service.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HealthStatus {
    Healthy,
    Unhealthy,
    Starting,
}

/// Information about a registered service.
#[derive(Debug, Clone)]
struct ServiceInfo {
    id: String,
    name: String,
    service_type: String,
    shard_id: usize,
    health: HealthStatus,
}

impl ServiceInfo {
    fn new(id: &str, name: &str, service_type: &str, shard_id: usize) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            service_type: service_type.to_string(),
            shard_id,
            health: HealthStatus::Starting,
        }
    }
}

/// In-memory service discovery for the simulation.
struct InMemoryDiscovery {
    services: RwLock<HashMap<String, ServiceInfo>>,
}

impl InMemoryDiscovery {
    fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
        }
    }

    fn register(&self, service: ServiceInfo) {
        let mut services = self.services.write().unwrap();
        println!(
            "[Discovery] Registered {} ({}) - shard {}",
            service.name, service.service_type, service.shard_id
        );
        services.insert(service.id.clone(), service);
    }

    fn discover(&self, service_type: &str) -> Vec<ServiceInfo> {
        let services = self.services.read().unwrap();
        services
            .values()
            .filter(|s| s.service_type == service_type)
            .cloned()
            .collect()
    }

    fn update_health(&self, service_id: &str, health: HealthStatus) {
        let mut services = self.services.write().unwrap();
        if let Some(service) = services.get_mut(service_id) {
            service.health = health;
        }
    }

    #[allow(dead_code)]
    fn get_healthy_ps(&self) -> Vec<ServiceInfo> {
        self.discover("ps")
            .into_iter()
            .filter(|s| s.health == HealthStatus::Healthy)
            .collect()
    }
}

// ============================================================================
// Messages for PS-Worker Communication
// ============================================================================

/// Request types from workers to parameter servers.
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum PsRequest {
    /// Request embeddings for given IDs
    Lookup { worker_id: usize, ids: Vec<i64> },
    /// Apply gradients for given IDs
    ApplyGradients {
        worker_id: usize,
        ids: Vec<i64>,
        gradients: Vec<f32>,
    },
    /// Request to save checkpoint
    Checkpoint { step: u64 },
    /// Shutdown the PS
    Shutdown,
}

/// Response types from parameter servers to workers.
#[allow(dead_code)]
#[derive(Debug, Clone)]
enum PsResponse {
    /// Embeddings for requested IDs
    Embeddings { embeddings: Vec<f32> },
    /// Gradient application acknowledgment
    GradientsApplied { success: bool },
    /// Checkpoint saved acknowledgment
    CheckpointSaved { path: PathBuf },
}

// ============================================================================
// Sharded Embedding Table (simulating MultiHashTable)
// ============================================================================

/// A single shard of the embedding table.
#[allow(dead_code)]
struct EmbeddingShard {
    /// Embeddings stored as id -> vector
    embeddings: HashMap<i64, Vec<f32>>,
    /// Embedding dimension
    dim: usize,
    /// Learning rate for gradient updates
    learning_rate: f32,
    /// Shard index
    shard_id: usize,
    /// Number of lookups performed
    lookup_count: AtomicU64,
    /// Number of gradient updates applied
    update_count: AtomicU64,
}

impl EmbeddingShard {
    fn new(shard_id: usize, dim: usize, learning_rate: f32) -> Self {
        Self {
            embeddings: HashMap::new(),
            dim,
            learning_rate,
            shard_id,
            lookup_count: AtomicU64::new(0),
            update_count: AtomicU64::new(0),
        }
    }

    /// Initialize an embedding with random values if not present.
    fn get_or_init(&mut self, id: i64) -> Vec<f32> {
        self.lookup_count.fetch_add(1, Ordering::Relaxed);

        self.embeddings
            .entry(id)
            .or_insert_with(|| {
                let mut rng = rand::thread_rng();
                (0..self.dim).map(|_| rng.gen_range(-0.1..0.1)).collect()
            })
            .clone()
    }

    /// Apply gradient update: embedding -= lr * gradient
    fn apply_gradient(&mut self, id: i64, gradient: &[f32]) {
        self.update_count.fetch_add(1, Ordering::Relaxed);

        let embedding = self
            .embeddings
            .entry(id)
            .or_insert_with(|| vec![0.0; self.dim]);

        for (e, g) in embedding.iter_mut().zip(gradient.iter()) {
            *e -= self.learning_rate * g;
        }
    }

    /// Get shard statistics.
    fn stats(&self) -> (usize, u64, u64) {
        (
            self.embeddings.len(),
            self.lookup_count.load(Ordering::Relaxed),
            self.update_count.load(Ordering::Relaxed),
        )
    }

    /// Serialize shard state for checkpointing.
    fn to_checkpoint(&self) -> HashMap<i64, Vec<f32>> {
        self.embeddings.clone()
    }
}

/// Determines which shard an ID belongs to.
fn shard_for_id(id: i64, num_shards: usize) -> usize {
    let positive = if id >= 0 {
        id as usize
    } else {
        (id.wrapping_abs() as usize).wrapping_add(1)
    };
    positive % num_shards
}

// ============================================================================
// Parameter Server
// ============================================================================

/// A parameter server that holds a shard of embeddings.
struct ParameterServer {
    shard: Mutex<EmbeddingShard>,
    shard_id: usize,
    num_shards: usize,
    checkpoint_dir: PathBuf,
}

impl ParameterServer {
    fn new(
        shard_id: usize,
        num_shards: usize,
        dim: usize,
        lr: f32,
        checkpoint_dir: PathBuf,
    ) -> Self {
        Self {
            shard: Mutex::new(EmbeddingShard::new(shard_id, dim, lr)),
            shard_id,
            num_shards,
            checkpoint_dir,
        }
    }

    /// Handle a lookup request.
    fn handle_lookup(&self, ids: &[i64]) -> Vec<f32> {
        let mut shard = self.shard.lock().unwrap();
        let mut result = Vec::with_capacity(ids.len() * shard.dim);

        for &id in ids {
            // Only handle IDs that belong to this shard
            if shard_for_id(id, self.num_shards) == self.shard_id {
                let emb = shard.get_or_init(id);
                result.extend(emb);
            }
        }

        result
    }

    /// Handle gradient application.
    fn handle_gradients(&self, ids: &[i64], gradients: &[f32], dim: usize) {
        let mut shard = self.shard.lock().unwrap();

        let mut grad_idx = 0;
        for &id in ids {
            if shard_for_id(id, self.num_shards) == self.shard_id {
                let grad = &gradients[grad_idx..grad_idx + dim];
                shard.apply_gradient(id, grad);
                grad_idx += dim;
            }
        }
    }

    /// Save checkpoint.
    fn save_checkpoint(&self, step: u64) -> PathBuf {
        let shard = self.shard.lock().unwrap();
        let checkpoint = shard.to_checkpoint();

        let path = self.checkpoint_dir.join(format!(
            "checkpoint-step{}-shard{}.json",
            step, self.shard_id
        ));

        // Create directory if needed
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        // Serialize (simplified - just count entries in real impl)
        let data = serde_json::json!({
            "step": step,
            "shard_id": self.shard_id,
            "num_entries": checkpoint.len(),
            "sample_ids": checkpoint.keys().take(10).collect::<Vec<_>>(),
        });

        let _ = std::fs::write(&path, serde_json::to_string_pretty(&data).unwrap());

        path
    }

    /// Get statistics.
    fn stats(&self) -> (usize, u64, u64) {
        self.shard.lock().unwrap().stats()
    }
}

// ============================================================================
// Worker
// ============================================================================

/// Training metrics collected by a worker.
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
struct WorkerMetrics {
    steps_completed: u64,
    total_loss: f64,
    embeddings_fetched: u64,
    gradients_sent: u64,
    time_in_forward_ms: u64,
    time_in_backward_ms: u64,
    time_waiting_ps_ms: u64,
}

/// A training worker that computes gradients.
#[allow(dead_code)]
struct Worker {
    worker_id: usize,
    batch_size: usize,
    embedding_dim: usize,
    num_features: i64,
    metrics: Mutex<WorkerMetrics>,
}

impl Worker {
    fn new(worker_id: usize, batch_size: usize, embedding_dim: usize, num_features: i64) -> Self {
        Self {
            worker_id,
            batch_size,
            embedding_dim,
            num_features,
            metrics: Mutex::new(WorkerMetrics::default()),
        }
    }

    /// Generate a random batch of feature IDs.
    fn generate_batch(&self) -> Vec<i64> {
        let mut rng = rand::thread_rng();
        (0..self.batch_size)
            .map(|_| rng.gen_range(0..self.num_features))
            .collect()
    }

    /// Simulate forward pass and compute loss.
    fn forward(&self, embeddings: &[f32]) -> f64 {
        // Simulated loss computation
        let sum: f32 = embeddings.iter().map(|x| x * x).sum();
        (sum / embeddings.len() as f32) as f64
    }

    /// Simulate backward pass and compute gradients.
    fn backward(&self, embeddings: &[f32]) -> Vec<f32> {
        // Simulated gradient computation (just use embeddings * 2 as gradient)
        embeddings.iter().map(|x| x * 2.0).collect()
    }

    /// Record metrics.
    fn record_step(&self, loss: f64, emb_count: usize, grad_count: usize) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.steps_completed += 1;
        metrics.total_loss += loss;
        metrics.embeddings_fetched += emb_count as u64;
        metrics.gradients_sent += grad_count as u64;
    }

    fn get_metrics(&self) -> WorkerMetrics {
        self.metrics.lock().unwrap().clone()
    }
}

// ============================================================================
// Synchronization Coordinator (for sync SGD)
// ============================================================================

/// Coordinates synchronous training across workers.
struct SyncCoordinator {
    /// Barrier for synchronizing workers at each step
    step_barrier: Barrier,
    /// Current global step
    global_step: AtomicU64,
    /// Number of workers
    num_workers: usize,
    /// Aggregated gradients per step (for sync mode)
    gradient_buffer: Mutex<HashMap<i64, Vec<f32>>>,
    /// Number of workers that have submitted gradients
    workers_ready: AtomicU64,
}

impl SyncCoordinator {
    fn new(num_workers: usize) -> Self {
        Self {
            step_barrier: Barrier::new(num_workers),
            global_step: AtomicU64::new(0),
            num_workers,
            gradient_buffer: Mutex::new(HashMap::new()),
            workers_ready: AtomicU64::new(0),
        }
    }

    /// Wait for all workers at a barrier.
    fn sync_barrier(&self) {
        self.step_barrier.wait();
    }

    /// Increment the global step (called once per sync step).
    fn increment_step(&self) -> u64 {
        self.global_step.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn current_step(&self) -> u64 {
        self.global_step.load(Ordering::SeqCst)
    }

    /// Add gradients to the aggregation buffer (for sync averaging).
    fn add_gradients(&self, ids: &[i64], gradients: &[f32], dim: usize) {
        let mut buffer = self.gradient_buffer.lock().unwrap();

        for (i, &id) in ids.iter().enumerate() {
            let start = i * dim;
            let end = start + dim;
            let grad = &gradients[start..end];

            buffer
                .entry(id)
                .and_modify(|existing| {
                    for (e, g) in existing.iter_mut().zip(grad.iter()) {
                        *e += g;
                    }
                })
                .or_insert_with(|| grad.to_vec());
        }

        self.workers_ready.fetch_add(1, Ordering::SeqCst);
    }

    /// Get and clear the aggregated gradients (called after all workers ready).
    fn take_averaged_gradients(&self) -> HashMap<i64, Vec<f32>> {
        let mut buffer = self.gradient_buffer.lock().unwrap();
        let num_workers = self.num_workers as f32;

        // Average the gradients
        for grads in buffer.values_mut() {
            for g in grads.iter_mut() {
                *g /= num_workers;
            }
        }

        self.workers_ready.store(0, Ordering::SeqCst);
        std::mem::take(&mut *buffer)
    }

    #[allow(dead_code)]
    fn all_workers_ready(&self) -> bool {
        self.workers_ready.load(Ordering::SeqCst) >= self.num_workers as u64
    }
}

// ============================================================================
// Main Training Loop
// ============================================================================

fn run_parameter_server(
    ps: Arc<ParameterServer>,
    discovery: Arc<InMemoryDiscovery>,
    rx: mpsc::Receiver<(PsRequest, mpsc::Sender<PsResponse>)>,
    shutdown: Arc<AtomicBool>,
    dim: usize,
) {
    let service_id = format!("ps-{}", ps.shard_id);

    // Update health status
    discovery.update_health(&service_id, HealthStatus::Healthy);

    println!("[PS-{}] Started and ready to serve embeddings", ps.shard_id);

    while !shutdown.load(Ordering::Relaxed) {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok((request, response_tx)) => {
                let response = match request {
                    PsRequest::Lookup { worker_id: _, ids } => {
                        let embeddings = ps.handle_lookup(&ids);
                        PsResponse::Embeddings { embeddings }
                    }
                    PsRequest::ApplyGradients {
                        worker_id: _,
                        ids,
                        gradients,
                    } => {
                        ps.handle_gradients(&ids, &gradients, dim);
                        PsResponse::GradientsApplied { success: true }
                    }
                    PsRequest::Checkpoint { step } => {
                        let path = ps.save_checkpoint(step);
                        println!("[PS-{}] Saved checkpoint at step {}", ps.shard_id, step);
                        PsResponse::CheckpointSaved { path }
                    }
                    PsRequest::Shutdown => {
                        println!("[PS-{}] Shutting down", ps.shard_id);
                        break;
                    }
                };

                let _ = response_tx.send(response);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    // Final stats
    let (entries, lookups, updates) = ps.stats();
    println!(
        "[PS-{}] Final stats: {} entries, {} lookups, {} updates",
        ps.shard_id, entries, lookups, updates
    );
}

fn run_worker_sync(
    worker: Arc<Worker>,
    ps_channels: Vec<mpsc::Sender<(PsRequest, mpsc::Sender<PsResponse>)>>,
    coordinator: Arc<SyncCoordinator>,
    discovery: Arc<InMemoryDiscovery>,
    args: Arc<Args>,
    shutdown: Arc<AtomicBool>,
) {
    let service_id = format!("worker-{}", worker.worker_id);
    discovery.update_health(&service_id, HealthStatus::Healthy);

    println!("[Worker-{}] Started in SYNC mode", worker.worker_id);

    let num_ps = ps_channels.len();
    let dim = args.embedding_dim;

    while !shutdown.load(Ordering::Relaxed) {
        let current_step = coordinator.current_step();
        if current_step >= args.num_steps {
            break;
        }

        // Generate batch
        let batch = worker.generate_batch();

        // Group IDs by shard
        let mut ids_by_shard: Vec<Vec<i64>> = vec![Vec::new(); num_ps];
        for &id in &batch {
            let shard = shard_for_id(id, num_ps);
            ids_by_shard[shard].push(id);
        }

        // Fetch embeddings from each PS
        let mut all_embeddings: HashMap<i64, Vec<f32>> = HashMap::new();

        for (shard_id, ids) in ids_by_shard.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }

            let (resp_tx, resp_rx) = mpsc::channel();
            let request = PsRequest::Lookup {
                worker_id: worker.worker_id,
                ids: ids.clone(),
            };

            ps_channels[shard_id].send((request, resp_tx)).unwrap();

            if let Ok(PsResponse::Embeddings { embeddings }) = resp_rx.recv() {
                for (i, &id) in ids.iter().enumerate() {
                    let start = i * dim;
                    let end = start + dim;
                    if end <= embeddings.len() {
                        all_embeddings.insert(id, embeddings[start..end].to_vec());
                    }
                }
            }
        }

        // Construct embedding tensor in batch order
        let mut embedding_tensor: Vec<f32> = Vec::with_capacity(batch.len() * dim);
        for &id in &batch {
            if let Some(emb) = all_embeddings.get(&id) {
                embedding_tensor.extend(emb);
            } else {
                // Fallback to zeros
                embedding_tensor.extend(vec![0.0; dim]);
            }
        }

        // Forward pass
        let loss = worker.forward(&embedding_tensor);

        // Backward pass
        let gradients = worker.backward(&embedding_tensor);

        // Add gradients to coordinator for averaging
        coordinator.add_gradients(&batch, &gradients, dim);

        // Sync barrier - wait for all workers
        coordinator.sync_barrier();

        // Only worker 0 applies the averaged gradients and increments step
        if worker.worker_id == 0 {
            let averaged_grads = coordinator.take_averaged_gradients();

            // Group by shard and apply
            let mut grads_by_shard: Vec<(Vec<i64>, Vec<f32>)> =
                vec![(Vec::new(), Vec::new()); num_ps];

            for (id, grads) in averaged_grads {
                let shard = shard_for_id(id, num_ps);
                grads_by_shard[shard].0.push(id);
                grads_by_shard[shard].1.extend(grads);
            }

            for (shard_id, (ids, grads)) in grads_by_shard.into_iter().enumerate() {
                if ids.is_empty() {
                    continue;
                }

                let (resp_tx, resp_rx) = mpsc::channel();
                let request = PsRequest::ApplyGradients {
                    worker_id: worker.worker_id,
                    ids,
                    gradients: grads,
                };

                ps_channels[shard_id].send((request, resp_tx)).unwrap();
                let _ = resp_rx.recv();
            }

            let step = coordinator.increment_step();

            // Logging
            if step % args.log_interval == 0 {
                println!(
                    "[Step {}] loss = {:.6} (sync mode, {} workers)",
                    step, loss, args.num_workers
                );
            }

            // Checkpointing
            if step % args.checkpoint_interval == 0 {
                for (shard_id, ps_tx) in ps_channels.iter().enumerate() {
                    let (resp_tx, resp_rx) = mpsc::channel();
                    ps_tx
                        .send((PsRequest::Checkpoint { step }, resp_tx))
                        .unwrap();

                    if let Ok(PsResponse::CheckpointSaved { path }) = resp_rx.recv() {
                        println!("[Checkpoint] Shard {} saved to {:?}", shard_id, path);
                    }
                }
            }
        }

        worker.record_step(loss, batch.len(), batch.len());

        // Second barrier to ensure all workers wait for gradient application
        coordinator.sync_barrier();
    }

    let metrics = worker.get_metrics();
    println!(
        "[Worker-{}] Completed: {} steps, avg loss = {:.6}",
        worker.worker_id,
        metrics.steps_completed,
        if metrics.steps_completed > 0 {
            metrics.total_loss / metrics.steps_completed as f64
        } else {
            0.0
        }
    );
}

fn run_worker_async(
    worker: Arc<Worker>,
    ps_channels: Vec<mpsc::Sender<(PsRequest, mpsc::Sender<PsResponse>)>>,
    global_step: Arc<AtomicU64>,
    discovery: Arc<InMemoryDiscovery>,
    args: Arc<Args>,
    shutdown: Arc<AtomicBool>,
) {
    let service_id = format!("worker-{}", worker.worker_id);
    discovery.update_health(&service_id, HealthStatus::Healthy);

    println!("[Worker-{}] Started in ASYNC mode", worker.worker_id);

    let num_ps = ps_channels.len();
    let dim = args.embedding_dim;

    while !shutdown.load(Ordering::Relaxed) {
        let current_global = global_step.load(Ordering::Relaxed);
        if current_global >= args.num_steps {
            break;
        }

        // Generate batch
        let batch = worker.generate_batch();

        // Group IDs by shard
        let mut ids_by_shard: Vec<Vec<i64>> = vec![Vec::new(); num_ps];
        for &id in &batch {
            let shard = shard_for_id(id, num_ps);
            ids_by_shard[shard].push(id);
        }

        // Fetch embeddings from each PS
        let mut all_embeddings: HashMap<i64, Vec<f32>> = HashMap::new();

        for (shard_id, ids) in ids_by_shard.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }

            let (resp_tx, resp_rx) = mpsc::channel();
            let request = PsRequest::Lookup {
                worker_id: worker.worker_id,
                ids: ids.clone(),
            };

            ps_channels[shard_id].send((request, resp_tx)).unwrap();

            if let Ok(PsResponse::Embeddings { embeddings }) = resp_rx.recv() {
                for (i, &id) in ids.iter().enumerate() {
                    let start = i * dim;
                    let end = start + dim;
                    if end <= embeddings.len() {
                        all_embeddings.insert(id, embeddings[start..end].to_vec());
                    }
                }
            }
        }

        // Construct embedding tensor
        let mut embedding_tensor: Vec<f32> = Vec::with_capacity(batch.len() * dim);
        for &id in &batch {
            if let Some(emb) = all_embeddings.get(&id) {
                embedding_tensor.extend(emb);
            } else {
                embedding_tensor.extend(vec![0.0; dim]);
            }
        }

        // Forward pass
        let loss = worker.forward(&embedding_tensor);

        // Backward pass
        let gradients = worker.backward(&embedding_tensor);

        // Async: immediately apply gradients (no waiting for other workers)
        let mut grad_idx = 0;
        for (shard_id, ids) in ids_by_shard.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }

            let grad_count = ids.len() * dim;
            let grads = gradients[grad_idx..grad_idx + grad_count].to_vec();
            grad_idx += grad_count;

            let (resp_tx, resp_rx) = mpsc::channel();
            let request = PsRequest::ApplyGradients {
                worker_id: worker.worker_id,
                ids: ids.clone(),
                gradients: grads,
            };

            ps_channels[shard_id].send((request, resp_tx)).unwrap();
            let _ = resp_rx.recv();
        }

        let step = global_step.fetch_add(1, Ordering::SeqCst) + 1;

        worker.record_step(loss, batch.len(), batch.len());

        // Only worker 0 does logging and checkpointing
        if worker.worker_id == 0 {
            if step % args.log_interval == 0 {
                println!("[Step {}] Worker-0 loss = {:.6} (async mode)", step, loss);
            }

            if step % args.checkpoint_interval == 0 {
                for (shard_id, ps_tx) in ps_channels.iter().enumerate() {
                    let (resp_tx, resp_rx) = mpsc::channel();
                    ps_tx
                        .send((PsRequest::Checkpoint { step }, resp_tx))
                        .unwrap();

                    if let Ok(PsResponse::CheckpointSaved { path }) = resp_rx.recv() {
                        println!("[Checkpoint] Shard {} saved to {:?}", shard_id, path);
                    }
                }
            }
        }
    }

    let metrics = worker.get_metrics();
    println!(
        "[Worker-{}] Completed: {} local steps, avg loss = {:.6}",
        worker.worker_id,
        metrics.steps_completed,
        if metrics.steps_completed > 0 {
            metrics.total_loss / metrics.steps_completed as f64
        } else {
            0.0
        }
    );
}

fn main() {
    let args = Arc::new(Args::parse());

    println!("{}", "=".repeat(70));
    println!("Distributed Training Simulation");
    println!("{}", "=".repeat(70));
    println!("Configuration:");
    println!("  - Workers: {}", args.num_workers);
    println!("  - Parameter Servers: {}", args.num_ps);
    println!("  - Sync Mode: {}", args.sync_mode);
    println!("  - Batch Size: {}", args.batch_size);
    println!("  - Training Steps: {}", args.num_steps);
    println!("  - Embedding Dim: {}", args.embedding_dim);
    println!("  - Feature Space: {} unique IDs", args.num_features);
    println!("  - Learning Rate: {}", args.learning_rate);
    println!("  - Checkpoint Dir: {:?}", args.checkpoint_dir);
    println!("{}", "=".repeat(70));
    println!();

    let is_sync = args.sync_mode == "sync";

    // Create service discovery
    let discovery = Arc::new(InMemoryDiscovery::new());

    // Create shutdown flag
    let shutdown = Arc::new(AtomicBool::new(false));

    // Create parameter servers
    let mut ps_handles = Vec::new();
    let mut ps_channels: Vec<mpsc::Sender<(PsRequest, mpsc::Sender<PsResponse>)>> = Vec::new();

    for shard_id in 0..args.num_ps {
        let (tx, rx) = mpsc::channel();
        ps_channels.push(tx);

        let ps = Arc::new(ParameterServer::new(
            shard_id,
            args.num_ps,
            args.embedding_dim,
            args.learning_rate,
            args.checkpoint_dir.clone(),
        ));

        // Register with discovery
        let service = ServiceInfo::new(
            &format!("ps-{}", shard_id),
            &format!("Parameter Server {}", shard_id),
            "ps",
            shard_id,
        );
        discovery.register(service);

        let discovery_clone = discovery.clone();
        let shutdown_clone = shutdown.clone();
        let dim = args.embedding_dim;

        let handle = thread::spawn(move || {
            run_parameter_server(ps, discovery_clone, rx, shutdown_clone, dim);
        });

        ps_handles.push(handle);
    }

    // Wait for PS to be ready
    thread::sleep(Duration::from_millis(100));

    // Create workers
    let mut worker_handles = Vec::new();

    // For sync mode, create coordinator
    let coordinator = Arc::new(SyncCoordinator::new(args.num_workers));

    // For async mode, create shared global step
    let global_step = Arc::new(AtomicU64::new(0));

    let start_time = Instant::now();

    for worker_id in 0..args.num_workers {
        let worker = Arc::new(Worker::new(
            worker_id,
            args.batch_size,
            args.embedding_dim,
            args.num_features,
        ));

        // Register with discovery
        let service = ServiceInfo::new(
            &format!("worker-{}", worker_id),
            &format!("Worker {}", worker_id),
            "worker",
            worker_id,
        );
        discovery.register(service);

        let ps_channels_clone = ps_channels.clone();
        let discovery_clone = discovery.clone();
        let args_clone = args.clone();
        let shutdown_clone = shutdown.clone();

        let handle = if is_sync {
            let coordinator_clone = coordinator.clone();
            thread::spawn(move || {
                run_worker_sync(
                    worker,
                    ps_channels_clone,
                    coordinator_clone,
                    discovery_clone,
                    args_clone,
                    shutdown_clone,
                );
            })
        } else {
            let global_step_clone = global_step.clone();
            thread::spawn(move || {
                run_worker_async(
                    worker,
                    ps_channels_clone,
                    global_step_clone,
                    discovery_clone,
                    args_clone,
                    shutdown_clone,
                );
            })
        };

        worker_handles.push(handle);
    }

    // Wait for all workers to finish
    for handle in worker_handles {
        handle.join().unwrap();
    }

    let elapsed = start_time.elapsed();

    // Signal PS to shutdown
    shutdown.store(true, Ordering::Relaxed);

    // Send shutdown to all PS
    for ps_tx in &ps_channels {
        let (resp_tx, _) = mpsc::channel();
        let _ = ps_tx.send((PsRequest::Shutdown, resp_tx));
    }

    // Wait for PS threads
    for handle in ps_handles {
        handle.join().unwrap();
    }

    // Print final summary
    println!();
    println!("{}", "=".repeat(70));
    println!("Training Complete!");
    println!("{}", "=".repeat(70));
    println!("  - Total time: {:.2?}", elapsed);
    println!(
        "  - Steps/sec: {:.2}",
        args.num_steps as f64 / elapsed.as_secs_f64()
    );

    let final_step = if is_sync {
        coordinator.current_step()
    } else {
        global_step.load(Ordering::Relaxed)
    };
    println!("  - Final global step: {}", final_step);

    // Show discovered services
    println!();
    println!("Service Discovery Summary:");
    println!("  - PS services: {}", discovery.discover("ps").len());
    println!(
        "  - Worker services: {}",
        discovery.discover("worker").len()
    );

    println!();
    println!("Sync vs Async Trade-offs:");
    if is_sync {
        println!("  [SYNC MODE]");
        println!("  + Consistent gradients (averaged across all workers)");
        println!("  + Better convergence properties");
        println!("  - Workers wait for slowest worker (straggler problem)");
        println!("  - Lower throughput");
    } else {
        println!("  [ASYNC MODE]");
        println!("  + Higher throughput (no waiting)");
        println!("  + Better hardware utilization");
        println!("  - Stale gradients can hurt convergence");
        println!("  - Less deterministic");
    }

    println!();
    println!("{}", "=".repeat(70));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shard_for_id() {
        assert_eq!(shard_for_id(0, 4), 0);
        assert_eq!(shard_for_id(1, 4), 1);
        assert_eq!(shard_for_id(4, 4), 0);
        assert_eq!(shard_for_id(5, 4), 1);

        // Negative IDs should also work
        assert!(shard_for_id(-1, 4) < 4);
        assert!(shard_for_id(-100, 4) < 4);
    }

    #[test]
    fn test_embedding_shard() {
        let mut shard = EmbeddingShard::new(0, 4, 0.1);

        // Get or init should create new embedding
        let emb1 = shard.get_or_init(100);
        assert_eq!(emb1.len(), 4);

        // Same ID should return same embedding
        let emb2 = shard.get_or_init(100);
        assert_eq!(emb1, emb2);

        // Apply gradient should update embedding
        shard.apply_gradient(100, &[1.0, 1.0, 1.0, 1.0]);
        let emb3 = shard.get_or_init(100);

        // emb3 = emb1 - 0.1 * [1,1,1,1]
        for (e3, e1) in emb3.iter().zip(emb1.iter()) {
            assert!((e3 - (e1 - 0.1)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_service_discovery() {
        let discovery = InMemoryDiscovery::new();

        discovery.register(ServiceInfo::new("ps-0", "PS 0", "ps", 0));
        discovery.register(ServiceInfo::new("ps-1", "PS 1", "ps", 1));
        discovery.register(ServiceInfo::new("worker-0", "Worker 0", "worker", 0));

        let ps_services = discovery.discover("ps");
        assert_eq!(ps_services.len(), 2);

        let worker_services = discovery.discover("worker");
        assert_eq!(worker_services.len(), 1);

        // Initially not healthy
        assert_eq!(discovery.get_healthy_ps().len(), 0);

        // Update health
        discovery.update_health("ps-0", HealthStatus::Healthy);
        assert_eq!(discovery.get_healthy_ps().len(), 1);
    }

    #[test]
    fn test_worker_generate_batch() {
        let worker = Worker::new(0, 32, 8, 1000);
        let batch = worker.generate_batch();

        assert_eq!(batch.len(), 32);
        for &id in &batch {
            assert!(id >= 0 && id < 1000);
        }
    }

    #[test]
    fn test_sync_coordinator() {
        let coord = SyncCoordinator::new(2);

        // Add gradients from "workers"
        coord.add_gradients(&[1, 2], &[1.0, 2.0, 3.0, 4.0], 2);
        assert!(!coord.all_workers_ready());

        coord.add_gradients(&[1, 2], &[1.0, 2.0, 3.0, 4.0], 2);
        assert!(coord.all_workers_ready());

        let averaged = coord.take_averaged_gradients();

        // Each gradient should be averaged
        // id=1: (1.0 + 1.0) / 2 = 1.0, (2.0 + 2.0) / 2 = 2.0
        assert_eq!(averaged.get(&1), Some(&vec![1.0, 2.0]));
        assert_eq!(averaged.get(&2), Some(&vec![3.0, 4.0]));
    }
}
