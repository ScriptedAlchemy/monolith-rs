//! Training orchestration and distributed parameter server for Monolith.
//!
//! This crate provides the core training infrastructure for the Monolith
//! recommendation system, including:
//!
//! - **Estimator pattern**: A high-level API for training, evaluation, and prediction
//! - **Training hooks**: Customizable callbacks for logging, checkpointing, and early stopping
//! - **Metrics**: Collection and aggregation of training metrics
//! - **Distributed training**: Stubs for parameter server and worker coordination
//!
//! # Architecture
//!
//! The training system follows TensorFlow's Estimator pattern, providing a clean
//! separation between model definition, training orchestration, and infrastructure
//! concerns.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                        Estimator                            │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
//! │  │  ModelFn    │  │   Hooks     │  │  MetricsRecorder    │  │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘  │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!              ┌───────────────┼───────────────┐
//!              │               │               │
//!              ▼               ▼               ▼
//!         train()         evaluate()       predict()
//! ```
//!
//! # Example
//!
//! ```rust
//! use monolith_training::{
//!     Estimator, EstimatorConfig, ConstantModelFn,
//!     LoggingHook, CheckpointHook,
//! };
//! use std::path::PathBuf;
//!
//! // Configure the estimator
//! let config = EstimatorConfig::new(PathBuf::from("/tmp/model"))
//!     .with_train_steps(10000)
//!     .with_save_checkpoints_steps(1000);
//!
//! // Create the model function
//! let model_fn = ConstantModelFn::new(0.5);
//!
//! // Create the estimator with hooks
//! let mut estimator = Estimator::new(config.clone(), model_fn);
//! estimator.add_hook(LoggingHook::new(100));
//! estimator.add_hook(CheckpointHook::new(config.model_dir.clone(), 1000));
//!
//! // Run training
//! // let result = estimator.train().unwrap();
//! ```

pub mod barrier;
pub mod base_embedding_task;
pub mod discovery;
pub mod distributed;
pub mod distributed_ps;
pub mod distributed_serving_ops;
pub mod entry;
pub mod estimator;
pub mod file_ops;
pub mod hooks;
pub mod metrics;
pub mod native_task_context;
pub mod native_training;
pub mod parameter_sync_replicator;
pub mod prefetch_queue;
pub mod py_discovery;
pub mod runner;
pub mod runner_utils;

// Re-export main types for convenience
pub use barrier::{
    Barrier, BarrierError, BarrierResult, InMemoryBarrier, PsBarrier, SharedBarrier,
};
pub use base_embedding_task::{
    BaseEmbeddingTask, BaseEmbeddingTaskConfig, BaseTaskConfig, EmbeddingDataset, EvalConfig,
    InputConfig, TrainConfig,
};
#[cfg(feature = "consul")]
pub use discovery::{new_consul_discovery, ConsulDiscovery};
pub use discovery::{
    new_in_memory_discovery, DiscoveryError, DiscoveryEvent, HealthStatus, InMemoryDiscovery,
    Result as DiscoveryResult, ServiceDiscovery, ServiceDiscoveryAsync, ServiceInfo,
    SharedDiscovery,
};
#[cfg(feature = "zookeeper")]
pub use discovery::{new_zk_discovery, ZkDiscovery};
pub use distributed::{ClusterConfig, DistributedError, ParameterServer, Worker};
pub use distributed_ps::{
    aggregate_gradients, dedup_ids, get_shard_for_id, route_to_shards, EmbeddingTable, PsClient,
    PsError, PsResult, PsServer, PsServerHandle,
};
pub use distributed_serving_ops::{refresh_sync_config, DistributedServingOpsError};
pub use entry::{
    combine_as_segment, AdadeltaOptimizer, AdagradOptimizer, AdamOptimizer, AmsgradOptimizer,
    BatchSoftmaxInitializer, BatchSoftmaxOptimizer, ConstantsInitializer, CuckooHashTableConfig,
    DynamicWdAdagradOptimizer, EntryError, FixedR8Compressor, Fp16Compressor, Fp32Compressor,
    FtrlOptimizer, HashTableConfig, HashTableConfigInstance, Initializer, LearningRateFn,
    MomentumOptimizer, MovingAverageOptimizer, OneBitCompressor, Optimizer,
    RandomUniformInitializer, RmspropOptimizer, RmspropV2Optimizer, SgdOptimizer,
    StochasticRoundingFloat16OptimizerWrapper, ZerosInitializer,
};
pub use estimator::{
    ConstantModelFn, Estimator, EstimatorConfig, EstimatorError, EstimatorMode, EstimatorResult,
    EvalResult, ModelFn, PredictResult, TrainResult,
};
pub use file_ops::{FileCloseHook, WritableFile};
pub use hooks::{
    CheckpointHook, EarlyStoppingHook, Hook, HookAction, HookError, HookList, HookResult,
    LoggingHook,
};
pub use metrics::{Metrics, MetricsRecorder};
pub use native_task_context::{
    get as get_native_task_context, with_ctx as with_native_task_context,
};
pub use native_training::save_utils::{
    get_monolith_checkpoint_state, write_monolith_checkpoint_state, SaveUtilsError,
    MONOLITH_CKPT_STATE_FILE_NAME,
};
pub use parameter_sync_replicator::{DirtyTracker, ParameterSyncReplicator};
pub use prefetch_queue::{
    enqueue_dicts_with_queue_return, AsyncFunctionMgr, Device, DevicePlacement, EnqueueResult,
    FifoQueue, MultiFifoQueue, Nested, Placed,
};
pub use py_discovery::{
    HostFileDiscovery, MlpServiceDiscovery, PyServiceDiscovery, TfConfigServiceDiscovery,
};
pub use runner::{run_distributed, DistributedRunConfig, Role};
pub use runner_utils::{copy_checkpoint_from_restore_dir, CheckpointState, RunnerUtilsError};

/// Training configuration combining estimator and distributed settings.
///
/// This is a higher-level configuration that encompasses both the estimator
/// configuration and distributed training settings.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingConfig {
    /// Estimator configuration.
    pub estimator: EstimatorConfig,
    /// Whether to use distributed training.
    pub distributed: bool,
    /// Cluster configuration for distributed training.
    pub cluster: Option<ClusterConfig>,
    /// Batch size for training.
    pub batch_size: usize,
    /// Learning rate.
    pub learning_rate: f64,
    /// Number of epochs (if not using steps).
    pub num_epochs: Option<u64>,
}

impl TrainingConfig {
    /// Creates a new training configuration.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - The directory to save model checkpoints.
    pub fn new(model_dir: std::path::PathBuf) -> Self {
        Self {
            estimator: EstimatorConfig::new(model_dir),
            distributed: false,
            cluster: None,
            batch_size: 32,
            learning_rate: 0.001,
            num_epochs: None,
        }
    }

    /// Sets the number of training steps.
    pub fn with_train_steps(mut self, steps: u64) -> Self {
        self.estimator.train_steps = Some(steps);
        self
    }

    /// Sets the batch size.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the learning rate.
    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Enables distributed training with the given cluster configuration.
    pub fn with_distributed(mut self, cluster: ClusterConfig) -> Self {
        self.distributed = true;
        self.cluster = Some(cluster);
        self
    }

    /// Sets the number of epochs.
    pub fn with_num_epochs(mut self, epochs: u64) -> Self {
        self.num_epochs = Some(epochs);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new(PathBuf::from("/tmp/model"))
            .with_train_steps(10000)
            .with_batch_size(64)
            .with_learning_rate(0.01);

        assert_eq!(config.estimator.train_steps, Some(10000));
        assert_eq!(config.batch_size, 64);
        assert!((config.learning_rate - 0.01).abs() < 1e-10);
        assert!(!config.distributed);
    }

    #[test]
    fn test_training_config_distributed() {
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};

        let cluster = ClusterConfig::new(
            vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5000)],
            vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 6000)],
            0,
            false,
        );

        let config = TrainingConfig::new(PathBuf::from("/tmp/model")).with_distributed(cluster);

        assert!(config.distributed);
        assert!(config.cluster.is_some());
    }

    #[test]
    fn test_full_training_flow() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"))
            .with_train_steps(50)
            .with_eval_steps(5);

        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        // Add hooks
        estimator.add_hook(LoggingHook::new(10));

        // Train
        let train_result = estimator.train().unwrap();
        assert_eq!(train_result.global_step, 50);

        // Evaluate
        let eval_result = estimator.evaluate().unwrap();
        assert!((eval_result.metrics.loss - 0.5).abs() < 1e-10);

        // Predict
        let predict_result = estimator.predict(3).unwrap();
        assert_eq!(predict_result.num_examples, 3);
    }
}
