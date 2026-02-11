//! Distributed training stubs.
//!
//! This module provides stub implementations for distributed training components
//! including parameter servers and workers. These are placeholders for future
//! integration with actual distributed training infrastructure.

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

/// A stub implementation of a parameter server.
///
/// In a real implementation, this would handle distributed parameter storage
/// and synchronization across workers.
///
/// # Examples
///
/// ```
/// use monolith_training::distributed::ParameterServer;
///
/// let mut ps = ParameterServer::new(0);
/// ps.set_parameter("weights", vec![0.1, 0.2, 0.3]);
/// let weights = ps.get_parameter("weights").unwrap();
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
    ///
    /// This is a stub that just sets the running flag.
    pub fn start(&mut self) -> DistributedResult<()> {
        tracing::info!("Starting parameter server {}", self.server_index);
        self.running = true;
        Ok(())
    }

    /// Stops the parameter server.
    pub fn stop(&mut self) -> DistributedResult<()> {
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

/// A stub implementation of a distributed training worker.
///
/// In a real implementation, this would handle training steps and
/// communication with parameter servers.
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
    ///
    /// This is a stub that just sets the running flag.
    pub fn start(&mut self) -> DistributedResult<()> {
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
        tracing::info!(
            "Stopping worker {} at step {}",
            self.worker_index,
            self.current_step
        );
        self.running = false;
        Ok(())
    }

    /// Simulates a training step.
    ///
    /// This is a stub that just increments the step counter.
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

    /// Synchronizes with other workers (stub).
    ///
    /// In a real implementation, this would perform an all-reduce or similar
    /// synchronization operation.
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
}

impl LocalCluster {
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
        })
    }

    /// Starts all PS and worker roles.
    pub fn start(&mut self) -> DistributedResult<()> {
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
        for worker in &mut self.workers {
            worker.stop()?;
        }
        for ps in &mut self.parameter_servers {
            ps.stop()?;
        }
        self.barrier_waiters.clear();
        Ok(())
    }

    /// Registers a parameter in the routed parameter server shard.
    pub fn register_parameter(
        &mut self,
        name: impl Into<String>,
        values: Vec<f32>,
    ) -> DistributedResult<usize> {
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
        let worker = self.workers.get_mut(worker_index).ok_or_else(|| {
            DistributedError::InvalidConfiguration(format!(
                "Worker index {} out of range",
                worker_index
            ))
        })?;
        worker.step()?;

        let mut updated = HashMap::new();
        for (param_name, grad) in gradients {
            let ps_idx = get_ps_index(param_name, self.parameter_servers.len());
            let ps = self.parameter_servers.get_mut(ps_idx).ok_or_else(|| {
                DistributedError::InvalidConfiguration(format!("Invalid PS index {}", ps_idx))
            })?;
            let next = ps.apply_gradients(param_name, grad, self.learning_rate)?;
            updated.insert(param_name.clone(), next);
        }

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
        let worker = self.workers.get(worker_index).ok_or_else(|| {
            DistributedError::InvalidConfiguration(format!(
                "Worker index {} out of range",
                worker_index
            ))
        })?;
        worker.sync_barrier()?;
        let epoch = worker.current_step();
        let required = self.workers.len();

        let arrived = {
            let waiters = self.barrier_waiters.entry(epoch).or_default();
            waiters.insert(worker_index);
            waiters.len()
        };

        if arrived >= required {
            self.barrier_waiters.remove(&epoch);
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
        assert!(config.validate().is_ok());

        // Invalid: no PS
        let config = ClusterConfig::new(vec![], vec![make_addr(6000)], 0, false);
        assert!(config.validate().is_err());

        // Invalid: task index out of range
        let config = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 5, false);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_parameter_server() {
        let mut ps = ParameterServer::new(0);
        assert!(!ps.is_running());

        ps.start().unwrap();
        assert!(ps.is_running());

        ps.set_parameter("weights", vec![1.0, 2.0, 3.0]);
        assert_eq!(ps.get_parameter("weights"), Some(vec![1.0, 2.0, 3.0]));
        assert!(ps.get_parameter("biases").is_none());

        ps.stop().unwrap();
        assert!(!ps.is_running());
    }

    #[test]
    fn test_parameter_server_apply_gradients() {
        let mut ps = ParameterServer::new(0);
        ps.set_parameter("w", vec![1.0, 2.0, 3.0]);

        let updated = ps.apply_gradients("w", &[0.1, 0.2, 0.3], 1.0).unwrap();
        assert_eq!(updated, vec![0.9, 1.8, 2.7]);

        // Wrong gradient size
        assert!(ps.apply_gradients("w", &[0.1], 1.0).is_err());

        // Unknown parameter
        assert!(ps.apply_gradients("unknown", &[0.1], 1.0).is_err());
    }

    #[test]
    fn test_worker() {
        let mut worker = Worker::new(0, 4);
        assert_eq!(worker.worker_index(), 0);
        assert_eq!(worker.num_workers(), 4);
        assert!(!worker.is_running());

        // Can't step when not running
        assert!(worker.step().is_err());

        worker.start().unwrap();
        assert!(worker.is_running());

        worker.step().unwrap();
        assert_eq!(worker.current_step(), 1);

        worker.step().unwrap();
        assert_eq!(worker.current_step(), 2);

        // Barrier works while running.
        worker.sync_barrier().unwrap();

        worker.stop().unwrap();
        assert!(!worker.is_running());

        // Barrier fails once stopped.
        assert!(worker.sync_barrier().is_err());
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
        let mut cluster = LocalCluster::new(cfg, 0.1).unwrap();
        cluster.start().unwrap();

        let ps_idx = cluster.register_parameter("w", vec![1.0, 2.0]).unwrap();
        assert_eq!(ps_idx, get_ps_index("w", cluster.num_ps()));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), vec![0.5, 1.0]);
        let updated = cluster.train_step(0, &grads).unwrap();
        assert_eq!(updated["w"], vec![0.95, 1.9]);
        assert_eq!(cluster.worker_step(0).unwrap(), 1);

        cluster.stop().unwrap();
    }

    #[test]
    fn test_local_cluster_bad_worker_index() {
        let cfg = ClusterConfig::new(vec![make_addr(5000)], vec![make_addr(6000)], 0, false);
        let mut cluster = LocalCluster::new(cfg, 0.1).unwrap();
        cluster.start().unwrap();
        let grads: HashMap<String, Vec<f32>> = HashMap::new();
        let err = cluster.train_step(5, &grads).unwrap_err();
        assert!(matches!(err, DistributedError::InvalidConfiguration(_)));
    }

    #[test]
    fn test_local_cluster_barrier_release() {
        let cfg = ClusterConfig::new(
            vec![make_addr(5000)],
            vec![make_addr(6000), make_addr(6001)],
            0,
            false,
        );
        let mut cluster = LocalCluster::new(cfg, 0.1).unwrap();
        cluster.start().unwrap();

        // Epoch 0 barrier: first caller waits, second caller releases.
        let first = cluster.sync_barrier(0).unwrap();
        assert_eq!(
            first,
            BarrierStatus::Waiting {
                epoch: 0,
                arrived: 1,
                required: 2
            }
        );
        let second = cluster.sync_barrier(1).unwrap();
        assert_eq!(
            second,
            BarrierStatus::Released {
                epoch: 0,
                participants: 2
            }
        );

        // Advance both workers to step 1 and verify next barrier epoch.
        let grads: HashMap<String, Vec<f32>> = HashMap::new();
        cluster.train_step(0, &grads).unwrap();
        cluster.train_step(1, &grads).unwrap();

        let first = cluster.sync_barrier(0).unwrap();
        assert!(matches!(first, BarrierStatus::Waiting { epoch: 1, .. }));
        let second = cluster.sync_barrier(1).unwrap();
        assert_eq!(
            second,
            BarrierStatus::Released {
                epoch: 1,
                participants: 2
            }
        );
    }
}
