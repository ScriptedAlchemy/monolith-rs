//! Distributed training runtime helpers.
//!
//! This module provides local-process distributed training components including
//! parameter servers, workers, and a lightweight local cluster coordinator used
//! by parity tests and runtime orchestration.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use thiserror::Error;

/// Errors that can occur in distributed training.
#[derive(Debug, Error)]
pub enum DistributedError {
    /// Failed to connect to a remote node.
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// Failed to send or receive a message.
    #[error("Communication error: {0}")]
    CommunicationError(String),

    /// The requested parameter was not found.
    #[error("Parameter not found: {0}")]
    ParameterNotFound(String),

    /// The cluster configuration is invalid.
    #[error("Invalid cluster configuration: {0}")]
    InvalidConfiguration(String),

    /// Timed out waiting for barrier synchronization.
    #[error("Barrier timeout at epoch {epoch} after {timeout_ms} ms")]
    BarrierTimeout { epoch: u64, timeout_ms: u64 },
}

/// Result type for distributed operations.
pub type DistributedResult<T> = Result<T, DistributedError>;

/// Outcome of a local-cluster barrier synchronization call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BarrierStatus {
    /// This worker has arrived at the barrier, but not all workers are present yet.
    Waiting {
        /// Barrier epoch (uses worker step for ordering).
        epoch: u64,
        /// Number of workers currently waiting at this epoch.
        arrived: usize,
        /// Total number of workers required to release the barrier.
        required: usize,
    },
    /// All workers reached the barrier for this epoch and it has been released.
    Released {
        /// Barrier epoch (uses worker step for ordering).
        epoch: u64,
        /// Number of participants that released this barrier.
        participants: usize,
    },
}

/// Configuration for a distributed training cluster.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterConfig {
    /// Addresses of parameter servers.
    pub ps_addrs: Vec<SocketAddr>,
    /// Addresses of workers.
    pub worker_addrs: Vec<SocketAddr>,
    /// Index of this node in its role (PS or worker).
    pub task_index: usize,
    /// Whether this node is a parameter server.
    pub is_ps: bool,
}

impl ClusterConfig {
    /// Creates a new cluster configuration.
    ///
    /// # Arguments
    ///
    /// * `ps_addrs` - Addresses of parameter servers.
    /// * `worker_addrs` - Addresses of workers.
    /// * `task_index` - Index of this node.
    /// * `is_ps` - Whether this node is a parameter server.
    pub fn new(
        ps_addrs: Vec<SocketAddr>,
        worker_addrs: Vec<SocketAddr>,
        task_index: usize,
        is_ps: bool,
    ) -> Self {
        Self {
            ps_addrs,
            worker_addrs,
            task_index,
            is_ps,
        }
    }

    /// Returns the number of parameter servers.
    pub fn num_ps(&self) -> usize {
        self.ps_addrs.len()
    }

    /// Returns the number of workers.
    pub fn num_workers(&self) -> usize {
        self.worker_addrs.len()
    }

    /// Validates the cluster configuration.
    pub fn validate(&self) -> DistributedResult<()> {
        if self.ps_addrs.is_empty() {
            return Err(DistributedError::InvalidConfiguration(
                "At least one parameter server is required".to_string(),
            ));
        }

        if self.worker_addrs.is_empty() {
            return Err(DistributedError::InvalidConfiguration(
                "At least one worker is required".to_string(),
            ));
        }

        let unique_ps: std::collections::HashSet<_> = self.ps_addrs.iter().collect();
        if unique_ps.len() != self.ps_addrs.len() {
            return Err(DistributedError::InvalidConfiguration(
                "Parameter server addresses must be unique".to_string(),
            ));
        }

        let unique_workers: std::collections::HashSet<_> = self.worker_addrs.iter().collect();
        if unique_workers.len() != self.worker_addrs.len() {
            return Err(DistributedError::InvalidConfiguration(
                "Worker addresses must be unique".to_string(),
            ));
        }

        let max_index = if self.is_ps {
            self.ps_addrs.len()
        } else {
            self.worker_addrs.len()
        };

        if self.task_index >= max_index {
            return Err(DistributedError::InvalidConfiguration(format!(
                "Task index {} is out of range (max: {})",
                self.task_index,
                max_index - 1
            )));
        }

        Ok(())
    }
}

/// State of a parameter on the parameter server.
#[derive(Debug, Clone)]
pub struct ParameterState {
    /// The parameter values.
    pub values: Vec<f32>,
    /// Version number for optimistic concurrency.
    pub version: u64,
}

/// A local parameter server implementation.
///
/// # Examples
///
/// ```
/// use monolith_training::distributed::ParameterServer;
///
/// let mut ps = ParameterServer::new(0);
/// ps.set_parameter("weights", vec![0.1, 0.2, 0.3]);
/// let weights = ps
///     .get_parameter("weights")
///     .expect("weights should be present after setting the parameter");
/// assert_eq!(weights.len(), 3);
/// ```
#[derive(Debug)]
pub struct ParameterServer {
    /// Index of this parameter server.
    server_index: usize,
    /// Stored parameters.
    parameters: HashMap<String, ParameterState>,
    /// Whether the server is running.
    running: bool,
}

impl ParameterServer {
    /// Creates a new parameter server.
    ///
    /// # Arguments
    ///
    /// * `server_index` - The index of this parameter server in the cluster.
    pub fn new(server_index: usize) -> Self {
        Self {
            server_index,
            parameters: HashMap::new(),
            running: false,
        }
    }

    /// Returns the server index.
    pub fn server_index(&self) -> usize {
        self.server_index
    }

    /// Returns whether the server is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Starts the parameter server.
    pub fn start(&mut self) -> DistributedResult<()> {
        if self.running {
            return Err(DistributedError::InvalidConfiguration(format!(
                "parameter server {} is already running",
                self.server_index
            )));
        }
        tracing::info!("Starting parameter server {}", self.server_index);
        self.running = true;
        Ok(())
    }

    /// Stops the parameter server.
    pub fn stop(&mut self) -> DistributedResult<()> {
        if !self.running {
            return Err(DistributedError::InvalidConfiguration(format!(
                "parameter server {} is not running",
                self.server_index
            )));
        }
        tracing::info!("Stopping parameter server {}", self.server_index);
        self.running = false;
        Ok(())
    }

    /// Sets a parameter value.
    ///
    /// # Arguments
    ///
    /// * `name` - The parameter name.
    /// * `values` - The parameter values.
    pub fn set_parameter(&mut self, name: impl Into<String>, values: Vec<f32>) {
        let name = name.into();
        let version = self
            .parameters
            .get(&name)
            .map(|p| p.version + 1)
            .unwrap_or(1);

        self.parameters
            .insert(name, ParameterState { values, version });
    }

    /// Gets a parameter value.
    ///
    /// # Arguments
    ///
    /// * `name` - The parameter name.
    ///
    /// # Returns
    ///
    /// The parameter values, or `None` if the parameter doesn't exist.
    pub fn get_parameter(&self, name: &str) -> Option<Vec<f32>> {
        self.parameters.get(name).map(|p| p.values.clone())
    }

    /// Applies a gradient update to a parameter.
    ///
    /// # Arguments
    ///
    /// * `name` - The parameter name.
    /// * `gradients` - The gradients to apply.
    /// * `learning_rate` - The learning rate.
    ///
    /// # Returns
    ///
    /// The updated parameter values.
    pub fn apply_gradients(
        &mut self,
        name: &str,
        gradients: &[f32],
        learning_rate: f32,
    ) -> DistributedResult<Vec<f32>> {
        let state = self
            .parameters
            .get_mut(name)
            .ok_or_else(|| DistributedError::ParameterNotFound(name.to_string()))?;

        if gradients.len() != state.values.len() {
            return Err(DistributedError::CommunicationError(format!(
                "Gradient size mismatch: expected {}, got {}",
                state.values.len(),
                gradients.len()
            )));
        }

        for (v, g) in state.values.iter_mut().zip(gradients.iter()) {
            *v -= learning_rate * g;
        }

        state.version += 1;
        Ok(state.values.clone())
    }

    /// Returns the number of parameters stored.
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
}

/// A local distributed training worker implementation.
///
/// # Examples
///
/// ```
/// use monolith_training::distributed::Worker;
///
/// let worker = Worker::new(0, 4);
/// assert_eq!(worker.worker_index(), 0);
/// assert_eq!(worker.num_workers(), 4);
/// ```
#[derive(Debug)]
pub struct Worker {
    /// Index of this worker.
    worker_index: usize,
    /// Total number of workers.
    num_workers: usize,
    /// Current training step.
    current_step: u64,
    /// Whether the worker is running.
    running: bool,
}

impl Worker {
    /// Creates a new worker.
    ///
    /// # Arguments
    ///
    /// * `worker_index` - The index of this worker in the cluster.
    /// * `num_workers` - The total number of workers.
    pub fn new(worker_index: usize, num_workers: usize) -> Self {
        Self {
            worker_index,
            num_workers,
            current_step: 0,
            running: false,
        }
    }

    /// Returns the worker index.
    pub fn worker_index(&self) -> usize {
        self.worker_index
    }

    /// Returns the total number of workers.
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Returns the current training step.
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Returns whether the worker is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Starts the worker.
    pub fn start(&mut self) -> DistributedResult<()> {
        if self.running {
            return Err(DistributedError::InvalidConfiguration(format!(
                "worker {} is already running",
                self.worker_index
            )));
        }
        tracing::info!(
            "Starting worker {} of {}",
            self.worker_index,
            self.num_workers
        );
        self.running = true;
        Ok(())
    }

    /// Stops the worker.
    pub fn stop(&mut self) -> DistributedResult<()> {
        if !self.running {
            return Err(DistributedError::InvalidConfiguration(format!(
                "worker {} is not running",
                self.worker_index
            )));
        }
        tracing::info!(
            "Stopping worker {} at step {}",
            self.worker_index,
            self.current_step
        );
        self.running = false;
        Ok(())
    }

    /// Simulates a training step.
    pub fn step(&mut self) -> DistributedResult<()> {
        if !self.running {
            return Err(DistributedError::CommunicationError(
                "Worker is not running".to_string(),
            ));
        }

        self.current_step += 1;
        tracing::debug!(
            "Worker {} completed step {}",
            self.worker_index,
            self.current_step
        );

        Ok(())
    }

    /// Synchronizes with other workers.
    pub fn sync_barrier(&self) -> DistributedResult<()> {
        if !self.running {
            return Err(DistributedError::CommunicationError(
                "Worker is not running".to_string(),
            ));
        }
        tracing::debug!(
            "Worker {} waiting at sync barrier (step {})",
            self.worker_index,
            self.current_step
        );
        // Stub: would actually wait for all workers
        Ok(())
    }
}

/// A local in-process distributed cluster simulator.
///
/// This provides a practical runtime for tests and local development where
/// parameter servers and workers run in a single process.
#[derive(Debug)]
pub struct LocalCluster {
    config: ClusterConfig,
    parameter_servers: Vec<ParameterServer>,
    workers: Vec<Worker>,
    learning_rate: f32,
    barrier_waiters: HashMap<u64, HashSet<usize>>,
    released_barriers: HashMap<u64, usize>,
}

impl LocalCluster {
    fn ensure_cluster_running(&self) -> DistributedResult<()> {
        let all_ps_running = self.parameter_servers.iter().all(ParameterServer::is_running);
        let all_workers_running = self.workers.iter().all(Worker::is_running);
        if all_ps_running && all_workers_running {
            Ok(())
        } else {
            Err(DistributedError::InvalidConfiguration(
                "local cluster is not fully running".to_string(),
            ))
        }
    }

    fn prune_released_barriers(&mut self) {
        // Keep released epochs that may still be observed by lagging workers.
        // Once all workers have advanced beyond an epoch (epoch < min step),
        // that release marker can be safely discarded.
        let min_step = self
            .workers
            .iter()
            .map(Worker::current_step)
            .min()
            .unwrap_or(0);
        self.released_barriers.retain(|&epoch, _| epoch >= min_step);
    }

    fn remove_barrier_waiter(&mut self, epoch: u64, worker_index: usize) {
        if let Some(waiters) = self.barrier_waiters.get_mut(&epoch) {
            waiters.remove(&worker_index);
            if waiters.is_empty() {
                self.barrier_waiters.remove(&epoch);
            }
        }
    }

    /// Creates a new local cluster from a validated cluster config.
    pub fn new(config: ClusterConfig, learning_rate: f32) -> DistributedResult<Self> {
        config.validate()?;
        let parameter_servers = (0..config.num_ps()).map(ParameterServer::new).collect();
        let workers = (0..config.num_workers())
            .map(|idx| Worker::new(idx, config.num_workers()))
            .collect();
        Ok(Self {
            config,
            parameter_servers,
            workers,
            learning_rate,
            barrier_waiters: HashMap::new(),
            released_barriers: HashMap::new(),
        })
    }

    /// Starts all PS and worker roles.
    pub fn start(&mut self) -> DistributedResult<()> {
        if self
            .parameter_servers
            .iter()
            .any(ParameterServer::is_running)
            || self.workers.iter().any(Worker::is_running)
        {
            return Err(DistributedError::InvalidConfiguration(
                "local cluster is already running".to_string(),
            ));
        }
        for ps in &mut self.parameter_servers {
            ps.start()?;
        }
        for worker in &mut self.workers {
            worker.start()?;
        }
        Ok(())
    }

    /// Stops all roles.
    pub fn stop(&mut self) -> DistributedResult<()> {
        if self
            .parameter_servers
            .iter()
            .all(|ps| !ps.is_running())
            && self.workers.iter().all(|w| !w.is_running())
        {
            return Err(DistributedError::InvalidConfiguration(
                "local cluster is not running".to_string(),
            ));
        }
        for worker in &mut self.workers {
            worker.stop()?;
        }
        for ps in &mut self.parameter_servers {
            ps.stop()?;
        }
        self.barrier_waiters.clear();
        self.released_barriers.clear();
        Ok(())
    }

    /// Registers a parameter in the routed parameter server shard.
    pub fn register_parameter(
        &mut self,
        name: impl Into<String>,
        values: Vec<f32>,
    ) -> DistributedResult<usize> {
        self.ensure_cluster_running()?;
        let name = name.into();
        let ps_idx = get_ps_index(&name, self.parameter_servers.len());
        let ps = self.parameter_servers.get_mut(ps_idx).ok_or_else(|| {
            DistributedError::InvalidConfiguration(format!("Invalid PS index {}", ps_idx))
        })?;
        ps.set_parameter(name, values);
        Ok(ps_idx)
    }

    /// Fetches a parameter from its routed PS shard.
    pub fn get_parameter(&self, name: &str) -> Option<Vec<f32>> {
        let ps_idx = get_ps_index(name, self.parameter_servers.len());
        self.parameter_servers.get(ps_idx)?.get_parameter(name)
    }

    /// Runs one worker training step and applies gradients to routed parameters.
    ///
    /// Returns the updated parameter values for each touched parameter.
    pub fn train_step(
        &mut self,
        worker_index: usize,
        gradients: &HashMap<String, Vec<f32>>,
    ) -> DistributedResult<HashMap<String, Vec<f32>>> {
        self.ensure_cluster_running()?;
        let worker = self.workers.get(worker_index).ok_or_else(|| {
            DistributedError::InvalidConfiguration(format!(
                "Worker index {} out of range",
                worker_index
            ))
        })?;
        if !worker.is_running() {
            return Err(DistributedError::CommunicationError(
                "Worker is not running".to_string(),
            ));
        }

        let mut updated = HashMap::new();
        for (param_name, grad) in gradients {
            let ps_idx = get_ps_index(param_name, self.parameter_servers.len());
            let ps = self.parameter_servers.get_mut(ps_idx).ok_or_else(|| {
                DistributedError::InvalidConfiguration(format!("Invalid PS index {}", ps_idx))
            })?;
            let next = ps.apply_gradients(param_name, grad, self.learning_rate)?;
            updated.insert(param_name.clone(), next);
        }

        self.workers
            .get_mut(worker_index)
            .expect("worker index was validated before applying gradients")
            .step()?;

        Ok(updated)
    }

    /// Returns current step of a specific worker.
    pub fn worker_step(&self, worker_index: usize) -> DistributedResult<u64> {
        let worker = self.workers.get(worker_index).ok_or_else(|| {
            DistributedError::InvalidConfiguration(format!(
                "Worker index {} out of range",
                worker_index
            ))
        })?;
        Ok(worker.current_step())
    }

    /// Returns total configured number of workers.
    pub fn num_workers(&self) -> usize {
        self.config.num_workers()
    }

    /// Returns total configured number of parameter servers.
    pub fn num_ps(&self) -> usize {
        self.config.num_ps()
    }

    /// Non-blocking barrier synchronization for local worker coordination.
    ///
    /// Each worker calls this method when it reaches a synchronization point.
    /// The barrier epoch is derived from the caller's current step. The method
    /// returns [`BarrierStatus::Waiting`] until all workers have arrived at the
    /// same epoch, then returns [`BarrierStatus::Released`] for the last caller.
    pub fn sync_barrier(&mut self, worker_index: usize) -> DistributedResult<BarrierStatus> {
        self.prune_released_barriers();

        let worker = self.workers.get(worker_index).ok_or_else(|| {
            DistributedError::InvalidConfiguration(format!(
                "Worker index {} out of range",
                worker_index
            ))
        })?;
        worker.sync_barrier()?;
        let epoch = worker.current_step();
        if let Some(&participants) = self.released_barriers.get(&epoch) {
            return Ok(BarrierStatus::Released {
                epoch,
                participants,
            });
        }
        let required = self.workers.len();

        let arrived = {
            let waiters = self.barrier_waiters.entry(epoch).or_default();
            waiters.insert(worker_index);
            waiters.len()
        };

        if arrived >= required {
            self.barrier_waiters.remove(&epoch);
            self.released_barriers.insert(epoch, required);
            Ok(BarrierStatus::Released {
                epoch,
                participants: required,
            })
        } else {
            Ok(BarrierStatus::Waiting {
                epoch,
                arrived,
                required,
            })
        }
    }

    /// Blocking barrier synchronization helper with timeout.
    ///
    /// This repeatedly checks local barrier state until the worker's current
    /// epoch has been released or the timeout elapses.
    pub fn wait_for_barrier(
        &mut self,
        worker_index: usize,
        timeout: std::time::Duration,
        poll_interval: std::time::Duration,
    ) -> DistributedResult<BarrierStatus> {
        let start = std::time::Instant::now();
        loop {
            match self.sync_barrier(worker_index)? {
                BarrierStatus::Released {
                    epoch,
                    participants,
                } => {
                    return Ok(BarrierStatus::Released {
                        epoch,
                        participants,
                    });
                }
                BarrierStatus::Waiting { epoch, .. } => {
                    if start.elapsed() >= timeout {
                        // Match robust distributed barrier semantics: a timed-out worker
                        // should not remain as a stale participant for future retries.
                        self.remove_barrier_waiter(epoch, worker_index);
                        return Err(DistributedError::BarrierTimeout {
                            epoch,
                            timeout_ms: timeout.as_millis() as u64,
                        });
                    }
                }
            }
            std::thread::sleep(poll_interval);
        }
    }
}

/// Determines which parameter server should store a given parameter.
///
/// Uses consistent hashing to distribute parameters across servers.
///
/// # Arguments
///
/// * `param_name` - The parameter name.
/// * `num_ps` - The number of parameter servers.
///
/// # Returns
///
/// The index of the parameter server that should store this parameter.
pub fn get_ps_index(param_name: &str, num_ps: usize) -> usize {
    if num_ps == 0 {
        return 0;
    }

    // Simple hash-based partitioning
    let hash: usize = param_name.bytes().fold(0usize, |acc, b| {
        acc.wrapping_mul(31).wrapping_add(b as usize)
    });

    hash % num_ps
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn make_addr(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
    }

    #[test]
    fn test_cluster_config_validation() {
        let config = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            true,
        );
        config
            .validate()
            .expect("valid cluster config should pass validation");

        // Invalid: no PS
        let config = ClusterConfig::new(vec![], vec![make_addr(6000)], 0, false);
        let err = config
            .validate()
            .expect_err("config without parameter servers should fail validation");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("At least one parameter server is required")),
            "expected InvalidConfiguration mentioning missing parameter servers, got {err:?}"
        );

        // Invalid: no workers
        let config = ClusterConfig::new(vec![make_addr(5000)], vec![], 0, true);
        let err = config
            .validate()
            .expect_err("config without workers should fail validation");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("At least one worker is required")),
            "expected InvalidConfiguration mentioning missing workers, got {err:?}"
        );

        // Invalid: task index out of range
        let config = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 5, false);
        let err = config
            .validate()
            .expect_err("out-of-range task index should fail validation");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("Task index 5 is out of range")),
            "expected InvalidConfiguration mentioning task-index range, got {err:?}"
        );

        // Invalid: duplicate PS addresses
        let config = ClusterConfig::new(
            vec![make_addr(5000), make_addr(5000)],
            vec![make_addr(6000)],
            0,
            true,
        );
        let err = config
            .validate()
            .expect_err("duplicate parameter-server addresses should fail validation");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("Parameter server addresses must be unique")),
            "expected InvalidConfiguration mentioning duplicate parameter-server addresses, got {err:?}"
        );

        // Invalid: duplicate worker addresses
        let config = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6000)],
            0,
            false,
        );
        let err = config
            .validate()
            .expect_err("duplicate worker addresses should fail validation");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("Worker addresses must be unique")),
            "expected InvalidConfiguration mentioning duplicate worker addresses, got {err:?}"
        );
    }

    #[test]
    fn test_parameter_server() {
        let mut ps = ParameterServer::new(0);
        assert!(!ps.is_running());

        ps.start()
            .expect("parameter server start should succeed in basic lifecycle test");
        assert!(ps.is_running());

        ps.set_parameter("weights", vec![1.0, 2.0, 3.0]);
        assert_eq!(ps.get_parameter("weights"), Some(vec![1.0, 2.0, 3.0]));
        assert!(ps.get_parameter("biases").is_none());

        ps.stop()
            .expect("parameter server stop should succeed in basic lifecycle test");
        assert!(!ps.is_running());
    }

    #[test]
    fn test_parameter_server_lifecycle_guards() {
        let mut ps = ParameterServer::new(1);
        let err = ps
            .stop()
            .expect_err("stopping an idle parameter server should fail");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("parameter server 1 is not running")),
            "expected InvalidConfiguration mentioning not-running parameter server, got {err:?}"
        );

        ps.start()
            .expect("parameter server start should succeed before duplicate-start check");
        let err = ps
            .start()
            .expect_err("starting an already running parameter server should fail");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("parameter server 1 is already running")),
            "expected InvalidConfiguration mentioning already-running parameter server, got {err:?}"
        );
        ps.stop()
            .expect("parameter server stop should succeed after duplicate-start check");
    }

    #[test]
    fn test_parameter_server_apply_gradients() {
        let mut ps = ParameterServer::new(0);
        ps.set_parameter("w", vec![1.0, 2.0, 3.0]);

        let updated = ps
            .apply_gradients("w", &[0.1, 0.2, 0.3], 1.0)
            .expect("apply_gradients should succeed for matching parameter and gradient sizes");
        assert_eq!(updated, vec![0.9, 1.8, 2.7]);

        // Wrong gradient size
        let err = ps
            .apply_gradients("w", &[0.1], 1.0)
            .expect_err("gradient-size mismatch should fail apply_gradients");
        assert!(
            matches!(err, DistributedError::CommunicationError(ref msg)
                if msg.contains("Gradient size mismatch")),
            "expected CommunicationError mentioning gradient-size mismatch, got {err:?}"
        );

        // Unknown parameter
        let err = ps
            .apply_gradients("unknown", &[0.1], 1.0)
            .expect_err("unknown parameter should fail apply_gradients");
        assert!(
            matches!(err, DistributedError::ParameterNotFound(ref name) if name == "unknown"),
            "expected ParameterNotFound(unknown), got {err:?}"
        );
    }

    #[test]
    fn test_worker() {
        let mut worker = Worker::new(0, 4);
        assert_eq!(worker.worker_index(), 0);
        assert_eq!(worker.num_workers(), 4);
        assert!(!worker.is_running());

        // Can't step when not running
        let err = worker
            .step()
            .expect_err("step should fail while worker is not running");
        assert!(
            matches!(err, DistributedError::CommunicationError(ref msg) if msg == "Worker is not running"),
            "expected CommunicationError(\"Worker is not running\"), got {err:?}"
        );

        worker
            .start()
            .expect("worker start should succeed in worker lifecycle test");
        assert!(worker.is_running());

        worker
            .step()
            .expect("first worker step should succeed while running");
        assert_eq!(worker.current_step(), 1);

        worker
            .step()
            .expect("second worker step should succeed while running");
        assert_eq!(worker.current_step(), 2);

        // Barrier works while running.
        worker
            .sync_barrier()
            .expect("worker barrier should succeed while running");

        worker
            .stop()
            .expect("worker stop should succeed in worker lifecycle test");
        assert!(!worker.is_running());

        // Barrier fails once stopped.
        let err = worker
            .sync_barrier()
            .expect_err("barrier should fail once worker is stopped");
        assert!(
            matches!(err, DistributedError::CommunicationError(ref msg) if msg == "Worker is not running"),
            "expected CommunicationError(\"Worker is not running\"), got {err:?}"
        );
    }

    #[test]
    fn test_worker_lifecycle_guards() {
        let mut worker = Worker::new(1, 2);
        let err = worker
            .stop()
            .expect_err("stopping an idle worker should fail");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("worker 1 is not running")),
            "expected InvalidConfiguration mentioning not-running worker, got {err:?}"
        );

        worker
            .start()
            .expect("worker start should succeed before duplicate-start check");
        let err = worker
            .start()
            .expect_err("starting an already running worker should fail");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("worker 1 is already running")),
            "expected InvalidConfiguration mentioning already-running worker, got {err:?}"
        );
        worker
            .stop()
            .expect("worker stop should succeed after duplicate-start check");
    }

    #[test]
    fn test_get_ps_index() {
        // Same parameter always goes to same PS
        let idx1 = get_ps_index("layer1/weights", 3);
        let idx2 = get_ps_index("layer1/weights", 3);
        assert_eq!(idx1, idx2);

        // Index is within range
        for i in 0..100 {
            let name = format!("param_{}", i);
            let idx = get_ps_index(&name, 5);
            assert!(idx < 5);
        }

        // Edge case: 0 PS
        assert_eq!(get_ps_index("test", 0), 0);
    }

    #[test]
    fn test_local_cluster_train_step() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000), make_addr(5001)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for train_step test");
        cluster
            .start()
            .expect("local cluster start should succeed for train_step test");

        let ps_idx = cluster
            .register_parameter("w", vec![1.0, 2.0])
            .expect("register_parameter should succeed in train_step test");
        assert_eq!(ps_idx, get_ps_index("w", cluster.num_ps()));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), vec![0.5, 1.0]);
        let updated = cluster
            .train_step(0, &grads)
            .expect("train_step should succeed for worker 0");
        assert_eq!(updated["w"], vec![0.95, 1.9]);
        assert_eq!(
            cluster
                .worker_step(0)
                .expect("worker_step(0) should be available after one train_step"),
            1
        );

        cluster
            .stop()
            .expect("local cluster stop should succeed for train_step test");
    }

    #[test]
    fn test_local_cluster_register_parameter_requires_running_cluster() {
        let cfg = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 0, false);
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed");
        let err = cluster
            .register_parameter("w", vec![1.0, 2.0])
            .expect_err("register_parameter should fail when cluster is not running");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("local cluster is not fully running")),
            "expected InvalidConfiguration mentioning not-fully-running cluster, got {err:?}"
        );

        cluster
            .start()
            .expect("local cluster start should succeed");
        cluster
            .register_parameter("w", vec![1.0, 2.0])
            .expect("register_parameter should succeed after cluster start");
    }

    #[test]
    fn test_local_cluster_bad_worker_index() {
        let cfg = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 0, false);
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed");
        cluster
            .start()
            .expect("local cluster start should succeed");
        let grads: HashMap<String, Vec<f32>> = HashMap::new();
        let err = cluster
            .train_step(5, &grads)
            .expect_err("train_step should fail for out-of-range worker index");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("Worker index 5 out of range")),
            "expected InvalidConfiguration mentioning out-of-range worker index, got {err:?}"
        );
    }

    #[test]
    fn test_local_cluster_train_step_requires_running_cluster() {
        let cfg = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 0, false);
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed");
        cluster
            .start()
            .expect("local cluster start should succeed");
        cluster
            .stop()
            .expect("local cluster stop should succeed");

        let grads: HashMap<String, Vec<f32>> = HashMap::new();
        let err = cluster
            .train_step(0, &grads)
            .expect_err("train_step should fail when cluster is not running");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("local cluster is not fully running")),
            "expected InvalidConfiguration mentioning not-fully-running cluster, got {err:?}"
        );
    }

    #[test]
    fn test_local_cluster_barrier_release() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for barrier-release test");
        cluster
            .start()
            .expect("local cluster start should succeed for barrier-release test");

        // Epoch 0 barrier: first caller waits, second caller releases.
        let first = cluster
            .sync_barrier(0)
            .expect("first worker barrier call should succeed at epoch 0");
        assert_eq!(
            first,
            BarrierStatus::Waiting {
                epoch: 0,
                arrived: 1,
                required: 2
            }
        );
        let second = cluster
            .sync_barrier(1)
            .expect("second worker barrier call should succeed and release epoch 0");
        assert_eq!(
            second,
            BarrierStatus::Released {
                epoch: 0,
                participants: 2
            }
        );

        // Advance both workers to step 1 and verify next barrier epoch.
        let grads: HashMap<String, Vec<f32>> = HashMap::new();
        cluster
            .train_step(0, &grads)
            .expect("worker 0 train_step should succeed before epoch-1 barrier");
        cluster
            .train_step(1, &grads)
            .expect("worker 1 train_step should succeed before epoch-1 barrier");

        let first = cluster
            .sync_barrier(0)
            .expect("first worker barrier call should succeed at epoch 1");
        assert!(matches!(first, BarrierStatus::Waiting { epoch: 1, .. }));
        let second = cluster
            .sync_barrier(1)
            .expect("second worker barrier call should succeed and release epoch 1");
        assert_eq!(
            second,
            BarrierStatus::Released {
                epoch: 1,
                participants: 2
            }
        );

        // Re-checking a released epoch from first worker now reports released.
        let again = cluster
            .sync_barrier(0)
            .expect("released barrier should be observable by prior participant");
        assert_eq!(
            again,
            BarrierStatus::Released {
                epoch: 1,
                participants: 2
            }
        );
    }

    #[test]
    fn test_local_cluster_wait_for_barrier_timeout() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for barrier-timeout test");
        cluster
            .start()
            .expect("local cluster start should succeed for barrier-timeout test");

        let err = cluster
            .wait_for_barrier(
                0,
                std::time::Duration::from_millis(10),
                std::time::Duration::from_millis(1),
            )
            .expect_err("wait_for_barrier should fail with timeout when peers never arrive");
        assert!(
            matches!(err, DistributedError::BarrierTimeout { epoch: 0, timeout_ms: 10 }),
            "expected BarrierTimeout {{ epoch: 0, timeout_ms: 10 }}, got {err:?}"
        );
    }

    #[test]
    fn test_local_cluster_wait_for_barrier_success() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for barrier-success test");
        cluster
            .start()
            .expect("local cluster start should succeed for barrier-success test");

        let first = cluster
            .sync_barrier(0)
            .expect("first worker barrier call should succeed at epoch 0");
        assert!(matches!(first, BarrierStatus::Waiting { epoch: 0, .. }));
        let second = cluster.wait_for_barrier(
            1,
            std::time::Duration::from_millis(50),
            std::time::Duration::from_millis(1),
        );
        assert!(matches!(
            second.expect("second worker wait_for_barrier should succeed"),
            BarrierStatus::Released {
                epoch: 0,
                participants: 2
            }
        ));
    }

    #[test]
    fn test_local_cluster_wait_for_barrier_timeout_cleanup_allows_retry() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for barrier-timeout cleanup test");
        cluster
            .start()
            .expect("local cluster start should succeed for barrier-timeout cleanup test");

        // Worker 0 times out waiting at epoch 0.
        let timeout_err = cluster
            .wait_for_barrier(
                0,
                std::time::Duration::from_millis(8),
                std::time::Duration::from_millis(1),
            )
            .expect_err("wait_for_barrier should fail with timeout before retry");
        assert!(
            matches!(timeout_err, DistributedError::BarrierTimeout { epoch: 0, timeout_ms: 8 }),
            "expected BarrierTimeout {{ epoch: 0, timeout_ms: 8 }}, got {timeout_err:?}"
        );

        // Cleanup should have removed worker 0 from waiter set, so worker 1 alone
        // should still be in waiting state instead of incorrectly releasing.
        let worker1_first = cluster
            .sync_barrier(1)
            .expect("worker 1 barrier call should succeed while waiting after timeout cleanup");
        assert_eq!(
            worker1_first,
            BarrierStatus::Waiting {
                epoch: 0,
                arrived: 1,
                required: 2
            }
        );

        // Worker 0 retries and now barrier can release correctly.
        let worker0_retry = cluster
            .sync_barrier(0)
            .expect("worker 0 retry barrier call should succeed and release");
        assert_eq!(
            worker0_retry,
            BarrierStatus::Released {
                epoch: 0,
                participants: 2
            }
        );
    }

    #[test]
    fn test_local_cluster_prunes_released_barriers_after_all_workers_advance() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for released-barrier pruning test");
        cluster
            .start()
            .expect("local cluster start should succeed for released-barrier pruning test");

        // Release epoch 0 barrier.
        assert!(matches!(
            cluster
                .sync_barrier(0)
                .expect("worker 0 barrier call should succeed at epoch 0"),
            BarrierStatus::Waiting { epoch: 0, .. }
        ));
        assert!(matches!(
            cluster
                .sync_barrier(1)
                .expect("worker 1 barrier call should succeed and release epoch 0"),
            BarrierStatus::Released { epoch: 0, .. }
        ));
        assert!(cluster.released_barriers.contains_key(&0));

        // Advance only worker 0; worker 1 should still observe epoch 0 as released.
        let grads: HashMap<String, Vec<f32>> = HashMap::new();
        cluster
            .train_step(0, &grads)
            .expect("worker 0 train_step should succeed before stale-release check");
        assert!(matches!(
            cluster
                .sync_barrier(1)
                .expect("worker 1 barrier call should observe released epoch 0"),
            BarrierStatus::Released { epoch: 0, .. }
        ));
        assert!(cluster.released_barriers.contains_key(&0));

        // Once worker 1 advances too, epoch 0 is obsolete and can be pruned.
        cluster
            .train_step(1, &grads)
            .expect("worker 1 train_step should succeed before prune");
        let _ = cluster
            .sync_barrier(0)
            .expect("worker 0 barrier call should trigger prune pass"); // triggers prune pass
        assert!(!cluster.released_barriers.contains_key(&0));
    }

    #[test]
    fn test_local_cluster_train_step_unknown_parameter_does_not_advance_worker_step() {
        let cfg = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 0, false);
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for unknown-parameter step test");
        cluster
            .start()
            .expect("local cluster start should succeed for unknown-parameter step test");
        cluster
            .register_parameter("known", vec![1.0, 2.0])
            .expect("register_parameter should seed known parameter before failure case");

        let mut grads = HashMap::new();
        grads.insert("unknown".to_string(), vec![0.5, 0.5]);
        let err = cluster
            .train_step(0, &grads)
            .expect_err("train_step should fail when gradient references unknown parameter");
        assert!(
            matches!(err, DistributedError::ParameterNotFound(ref name) if name == "unknown"),
            "expected ParameterNotFound(unknown), got {err:?}"
        );
        assert_eq!(
            cluster
                .worker_step(0)
                .expect("worker_step(0) should remain readable after failed train_step"),
            0,
            "failed train_step should not advance worker step for unknown-parameter errors"
        );
    }

    #[test]
    fn test_local_cluster_train_step_gradient_mismatch_does_not_advance_worker_step() {
        let cfg = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 0, false);
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for gradient-mismatch step test");
        cluster
            .start()
            .expect("local cluster start should succeed for gradient-mismatch step test");
        cluster
            .register_parameter("known", vec![1.0, 2.0])
            .expect("register_parameter should seed known parameter before mismatch case");

        let mut grads = HashMap::new();
        grads.insert("known".to_string(), vec![0.5]);
        let err = cluster
            .train_step(0, &grads)
            .expect_err("train_step should fail on gradient-size mismatch");
        assert!(
            matches!(err, DistributedError::CommunicationError(ref msg)
                if msg.contains("Gradient size mismatch")),
            "expected CommunicationError mentioning gradient-size mismatch, got {err:?}"
        );
        assert_eq!(
            cluster
                .worker_step(0)
                .expect("worker_step(0) should remain readable after failed train_step"),
            0,
            "failed train_step should not advance worker step for gradient-size mismatch errors"
        );
    }

    #[test]
    fn test_local_cluster_start_is_not_reentrant() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for non-reentrant start test");
        cluster
            .start()
            .expect("local cluster start should succeed for non-reentrant start test");

        let err = cluster
            .start()
            .expect_err("starting an already running local cluster should fail");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("local cluster is already running")),
            "expected InvalidConfiguration mentioning already-running local cluster, got {err:?}"
        );
    }

    #[test]
    fn test_local_cluster_stop_requires_running_cluster() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1)
            .expect("local cluster construction should succeed for stop-requires-running test");
        let err = cluster
            .stop()
            .expect_err("stopping a non-running local cluster should fail");
        assert!(
            matches!(err, DistributedError::InvalidConfiguration(ref msg)
                if msg.contains("local cluster is not running")),
            "expected InvalidConfiguration mentioning not-running local cluster, got {err:?}"
        );
    }
}
