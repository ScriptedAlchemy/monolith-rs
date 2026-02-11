//! Train Command Implementation
//!
//! Provides distributed model training using the Monolith estimator.
//! Supports configuration via JSON files and command-line arguments.

use anyhow::{Context, Result};
use clap::Args;
use monolith_checkpoint::{Checkpointer, JsonCheckpointer, ModelState};
use monolith_data::example::total_fid_count;
use monolith_data::{CompressionType, TFRecordReader};
use monolith_training::discovery::ServiceDiscoveryAsync;
use monolith_training::runner::{run_distributed, DistributedRunConfig, Role};
use std::fs::File;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{info, warn};

/// Train a model using distributed training
///
/// This command initializes the Monolith estimator and runs training
/// for the specified number of steps. Training configuration can be
/// provided via a JSON config file or command-line arguments.
///
/// # Example
///
/// ```bash
/// monolith train \
///     --model-dir /path/to/model \
///     --config /path/to/config.json \
///     --train-steps 10000
/// ```
#[derive(Args, Debug, Clone)]
pub struct TrainCommand {
    /// Directory to save model checkpoints and logs
    #[arg(long, short = 'd', env = "MONOLITH_MODEL_DIR")]
    pub model_dir: PathBuf,

    /// Path to the training configuration file (JSON format)
    #[arg(long, short = 'c', env = "MONOLITH_CONFIG_PATH")]
    pub config_path: Option<PathBuf>,

    /// Optional TFRecord file to read training data from.
    #[arg(long, env = "MONOLITH_TFRECORD_PATH")]
    pub tfrecord_path: Option<PathBuf>,

    /// Train from CSV lines on stdin (mov,uid,label).
    #[arg(long, default_value = "false")]
    pub stdin: bool,

    /// Number of training steps to run
    #[arg(long, short = 's', default_value = "10000")]
    pub train_steps: u64,

    /// Batch size for training
    #[arg(long, short = 'b', default_value = "128")]
    pub batch_size: usize,

    /// Learning rate
    #[arg(long, default_value = "0.001")]
    pub learning_rate: f64,

    /// Checkpoint save interval (in steps)
    #[arg(long, default_value = "1000")]
    pub save_steps: u64,

    /// Evaluation interval (in steps)
    #[arg(long, default_value = "500")]
    pub eval_steps: u64,

    /// Resume training from the latest checkpoint
    #[arg(long, default_value = "true")]
    pub resume: bool,

    /// Number of data loading workers
    #[arg(long, default_value = "4")]
    pub num_workers: usize,

    // ---------------------------------------------------------------------
    // Distributed training settings (Rust parity with Python monolith)
    // ---------------------------------------------------------------------
    /// Run distributed orchestration (PS/worker roles with discovery).
    #[arg(long, default_value = "false")]
    pub distributed: bool,

    /// Role for this process in distributed mode.
    #[arg(long, value_enum, default_value = "worker")]
    pub role: TrainRole,

    /// Index for this role (e.g. worker-0, ps-1).
    #[arg(long, default_value = "0")]
    pub index: usize,

    /// Number of parameter servers in the cluster.
    #[arg(long, default_value = "1")]
    pub num_ps: usize,

    /// Number of workers in the cluster.
    #[arg(long, default_value = "1")]
    pub num_workers_cluster: usize,

    /// Bind address for this process (host:port). Use port 0 for ephemeral port.
    #[arg(long, default_value = "127.0.0.1:0")]
    pub bind_addr: String,

    /// Service discovery backend.
    ///
    /// - `in-memory`: local-only (single process / tests)
    /// - `tf-config`: read cluster from `TF_CONFIG` env (Primus-like)
    /// - `mlp`: environment-variable discovery (MLP_*)
    /// - `zookeeper`: ZooKeeper (feature: `zookeeper`)
    /// - `consul`: HashiCorp Consul (feature: `consul`)
    #[arg(long, value_enum, default_value = "in-memory")]
    pub discovery: DiscoveryBackend,

    /// ZooKeeper hosts (e.g. "127.0.0.1:2181") for `--discovery zookeeper`.
    #[arg(long)]
    pub zk_hosts: Option<String>,

    /// ZooKeeper base path (e.g. "/monolith/my_job") for `--discovery zookeeper`.
    #[arg(long, default_value = "/monolith/default")]
    pub zk_base_path: String,

    /// Consul address (e.g. "http://127.0.0.1:8500") for `--discovery consul`.
    #[arg(long)]
    pub consul_addr: Option<String>,

    /// Consul service name (stable job identifier) for `--discovery consul`.
    #[arg(long, default_value = "monolith")]
    pub consul_service_name: String,

    /// Discovery service type name for PS entries.
    #[arg(long, default_value = "ps")]
    pub discovery_service_type_ps: String,

    /// Discovery service type name for worker entries.
    #[arg(long, default_value = "worker")]
    pub discovery_service_type_worker: String,

    /// Embedding table name for PS training RPC.
    #[arg(long, default_value = "emb")]
    pub table_name: String,

    /// Embedding dimension.
    #[arg(long, default_value = "64")]
    pub dim: usize,

    /// Number of connect retries waiting for the PS set.
    #[arg(long, default_value = "6")]
    pub connect_retries: usize,

    /// Retry backoff in milliseconds.
    #[arg(long, default_value = "500")]
    pub retry_backoff_ms: u64,

    /// Worker barrier timeout in milliseconds.
    #[arg(long, default_value = "10000")]
    pub barrier_timeout_ms: i64,

    /// Disable discovery heartbeat task (if backend supports it).
    #[arg(long, default_value = "false")]
    pub disable_heartbeat: bool,

    /// Heartbeat interval in seconds (0 disables).
    #[arg(long, default_value = "10")]
    pub heartbeat_interval_secs: u64,

    /// Timeout for discovery setup/query operations in milliseconds.
    #[arg(long, default_value = "5000")]
    pub discovery_operation_timeout_ms: u64,

    /// Timeout for discovery cleanup operations in milliseconds.
    #[arg(long, default_value = "200")]
    pub discovery_cleanup_timeout_ms: u64,

    /// ParameterSync gRPC targets ("host:port") for training PS to push embedding deltas to online.
    #[arg(long = "parameter-sync-target")]
    pub parameter_sync_targets: Vec<String>,

    /// ParameterSync push interval in milliseconds.
    #[arg(long, default_value = "200")]
    pub parameter_sync_interval_ms: u64,

    /// ParameterSync model name.
    #[arg(long, default_value = "default")]
    pub parameter_sync_model_name: String,

    /// ParameterSync signature name.
    #[arg(long, default_value = "serving_default")]
    pub parameter_sync_signature_name: String,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum TrainRole {
    Ps,
    Worker,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum DiscoveryBackend {
    InMemory,
    TfConfig,
    Mlp,
    #[cfg(feature = "zookeeper")]
    Zookeeper,
    #[cfg(feature = "consul")]
    Consul,
}

impl TrainCommand {
    fn build_distributed_run_config(&self) -> Result<Option<DistributedRunConfig>> {
        if !self.distributed {
            return Ok(None);
        }

        let bind_addr: SocketAddr = self
            .bind_addr
            .parse()
            .context("Invalid --bind-addr (expected host:port)")?;

        let role = match self.role {
            TrainRole::Ps => Role::Ps,
            TrainRole::Worker => Role::Worker,
        };

        let heartbeat_interval = if self.disable_heartbeat || self.heartbeat_interval_secs == 0 {
            None
        } else {
            Some(std::time::Duration::from_secs(self.heartbeat_interval_secs))
        };

        Ok(Some(DistributedRunConfig {
            role,
            index: self.index,
            num_ps: self.num_ps,
            num_workers: self.num_workers_cluster,
            bind_addr,
            discovery_service_type_ps: self.discovery_service_type_ps.clone(),
            discovery_service_type_worker: self.discovery_service_type_worker.clone(),
            table_name: self.table_name.clone(),
            dim: self.dim,
            connect_retries: self.connect_retries,
            retry_backoff_ms: self.retry_backoff_ms,
            barrier_timeout_ms: self.barrier_timeout_ms,
            heartbeat_interval,
            discovery_operation_timeout: std::time::Duration::from_millis(
                self.discovery_operation_timeout_ms,
            ),
            discovery_cleanup_timeout: std::time::Duration::from_millis(
                self.discovery_cleanup_timeout_ms,
            ),
            parameter_sync_targets: self.parameter_sync_targets.clone(),
            parameter_sync_interval: std::time::Duration::from_millis(
                self.parameter_sync_interval_ms,
            ),
            parameter_sync_model_name: self.parameter_sync_model_name.clone(),
            parameter_sync_signature_name: self.parameter_sync_signature_name.clone(),
        }))
    }

    /// Execute the train command
    pub async fn run(&self) -> Result<()> {
        info!("Starting training...");
        info!("Model directory: {:?}", self.model_dir);
        info!("Training steps: {}", self.train_steps);

        // Ensure model directory exists
        if !self.model_dir.exists() {
            std::fs::create_dir_all(&self.model_dir).context("Failed to create model directory")?;
            info!("Created model directory: {:?}", self.model_dir);
        }

        // Load configuration if provided (JSON is parsed for parity, but we use CLI args here).
        if let Some(config_path) = &self.config_path {
            info!("Loading config from: {:?}", config_path);
            let config_str =
                std::fs::read_to_string(config_path).context("Failed to read config file")?;
            let _config: serde_json::Value =
                serde_json::from_str(&config_str).context("Failed to parse config JSON")?;
        } else {
            warn!("No config file provided, using default configuration");
        }

        info!(
            "Training configuration: batch_size={}, lr={}, workers={}",
            self.batch_size, self.learning_rate, self.num_workers
        );

        if let Some(cfg) = self.build_distributed_run_config()? {
            let discovery = self.build_discovery().await?;
            run_distributed(discovery, cfg).await?;
            info!("Distributed training completed successfully");
            return Ok(());
        }

        if self.stdin {
            run_stdin_training(self)?;
            info!("Training completed successfully");
            return Ok(());
        }

        if let Some(path) = &self.tfrecord_path {
            run_tfrecord_training(self, path)?;
            info!("Training completed successfully");
            return Ok(());
        }

        info!("Training completed successfully");
        Ok(())
    }

    async fn build_discovery(&self) -> Result<std::sync::Arc<dyn ServiceDiscoveryAsync>> {
        // Note: We return the async trait object used by the runner. Many backends also implement
        // the sync trait; the runner relies on the async API.
        match self.discovery {
            DiscoveryBackend::InMemory => Ok(std::sync::Arc::new(
                monolith_training::discovery::InMemoryDiscovery::new(),
            )),
            DiscoveryBackend::TfConfig => {
                let tf = std::env::var("TF_CONFIG")
                    .context("TF_CONFIG is required for --discovery tf-config")?;
                let d = monolith_training::TfConfigServiceDiscovery::new(&tf)
                    .context("Failed to parse TF_CONFIG")?;
                Ok(std::sync::Arc::new(d))
            }
            DiscoveryBackend::Mlp => Ok(std::sync::Arc::new(
                monolith_training::MlpServiceDiscovery::new(),
            )),
            #[cfg(feature = "zookeeper")]
            DiscoveryBackend::Zookeeper => {
                let hosts = self
                    .zk_hosts
                    .clone()
                    .context("--zk-hosts is required for --discovery zookeeper")?;
                let zk = monolith_training::ZkDiscovery::new(hosts, self.zk_base_path.clone());
                Ok(std::sync::Arc::new(zk))
            }
            #[cfg(feature = "consul")]
            DiscoveryBackend::Consul => {
                let addr = self
                    .consul_addr
                    .clone()
                    .context("--consul-addr is required for --discovery consul")?;
                let d = monolith_training::ConsulDiscovery::new(addr)
                    .with_service_name(self.consul_service_name.clone());
                Ok(std::sync::Arc::new(d))
            }
        }
    }
}

struct SimpleModel {
    weight: f32,
    bias: f32,
    lr: f32,
}

impl SimpleModel {
    fn new(lr: f32) -> Self {
        Self {
            weight: 0.01,
            bias: 0.0,
            lr,
        }
    }

    fn restore_from(state: &ModelState, lr: f32) -> Self {
        let weight = state
            .dense_params
            .get("linear.weight")
            .and_then(|v| v.first().copied())
            .unwrap_or(0.01);
        let bias = state
            .dense_params
            .get("linear.bias")
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        Self { weight, bias, lr }
    }

    fn predict(&self, feature_count: f32) -> f32 {
        self.weight * feature_count + self.bias
    }

    fn train_step(&mut self, feature_count: f32, label: f32) -> f32 {
        let pred = self.predict(feature_count);
        let err = pred - label;
        let loss = err * err;
        let grad_w = 2.0 * err * feature_count;
        let grad_b = 2.0 * err;
        self.weight -= self.lr * grad_w;
        self.bias -= self.lr * grad_b;
        loss
    }

    fn to_state(&self, step: u64) -> ModelState {
        let mut state = ModelState::new(step);
        state.add_dense_param("linear.weight", vec![self.weight]);
        state.add_dense_param("linear.bias", vec![self.bias]);
        state
    }
}

fn run_tfrecord_training(cmd: &TrainCommand, path: &PathBuf) -> Result<()> {
    let file = File::open(path).context("Failed to open TFRecord file")?;
    let compression = CompressionType::from_extension(path.to_string_lossy().as_ref());
    let mut reader = TFRecordReader::new(file, true).with_compression(compression);

    let checkpointer = JsonCheckpointer::new();
    let mut step = 0u64;
    let mut model = if cmd.resume {
        let latest = checkpointer
            .latest(&cmd.model_dir)
            .context("No checkpoint found to resume")?;
        let state = checkpointer.restore(&latest)?;
        step = state.global_step;
        SimpleModel::restore_from(&state, cmd.learning_rate as f32)
    } else {
        SimpleModel::new(cmd.learning_rate as f32)
    };

    let start = Instant::now();
    let mut batch_loss = 0.0f32;
    let mut batch_count = 0u64;

    while step < cmd.train_steps {
        let example = match reader.read_example()? {
            Some(ex) => ex,
            None => {
                let file = File::open(path).context("Failed to reopen TFRecord file")?;
                reader = TFRecordReader::new(file, true).with_compression(compression);
                continue;
            }
        };

        let feature_count = total_fid_count(&example) as f32;
        let label = example.label.first().copied().unwrap_or(0.0);
        let loss = model.train_step(feature_count, label);
        batch_loss += loss;
        batch_count += 1;
        step += 1;

        if step % cmd.eval_steps == 0 {
            let avg_loss = batch_loss / batch_count.max(1) as f32;
            info!("Step {}: loss={:.6}", step, avg_loss);
            batch_loss = 0.0;
            batch_count = 0;
        }

        if step % cmd.save_steps == 0 {
            let state = model.to_state(step);
            let path = cmd.model_dir.join(format!("checkpoint-{}.json", step));
            checkpointer.save(&path, &state)?;
            info!("Saved checkpoint to {:?}", path);
        }
    }

    let state = model.to_state(step);
    let path = cmd.model_dir.join(format!("checkpoint-{}.json", step));
    checkpointer.save(&path, &state)?;
    info!(
        "Training finished at step {} in {:.2}s",
        step,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

fn run_stdin_training(cmd: &TrainCommand) -> Result<()> {
    let checkpointer = JsonCheckpointer::new();
    let mut step = 0u64;
    let mut model = SimpleModel::new(cmd.learning_rate as f32);
    let start = Instant::now();

    for line in std::io::stdin().lines() {
        let line = line.context("Failed to read stdin")?;
        if line.trim().is_empty() {
            continue;
        }
        let tokens: Vec<_> = line.trim().split(',').collect();
        if tokens.len() != 3 {
            continue;
        }
        let label: f32 = tokens[2].parse().unwrap_or(0.0);
        let feature_count = 2.0;
        let _loss = model.train_step(feature_count, label);
        step += 1;

        if step % cmd.save_steps == 0 {
            let state = model.to_state(step);
            let path = cmd.model_dir.join(format!("checkpoint-{}.json", step));
            checkpointer.save(&path, &state)?;
            info!("Saved checkpoint to {:?}", path);
        }

        if step >= cmd.train_steps {
            break;
        }
    }

    let state = model.to_state(step);
    let path = cmd.model_dir.join(format!("checkpoint-{}.json", step));
    checkpointer.save(&path, &state)?;
    info!(
        "Stdin training finished at step {} in {:.2}s",
        step,
        start.elapsed().as_secs_f32()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use monolith_data::example::{add_feature, create_example};
    use monolith_data::TFRecordWriter;
    use std::fs::File;
    use std::time::Duration;
    use tempfile::tempdir;

    fn test_cmd_defaults() -> TrainCommand {
        TrainCommand {
            model_dir: PathBuf::from("/tmp/model"),
            config_path: None,
            tfrecord_path: None,
            stdin: false,
            train_steps: 10000,
            batch_size: 128,
            learning_rate: 0.001,
            save_steps: 1000,
            eval_steps: 500,
            resume: true,
            num_workers: 4,
            distributed: false,
            role: TrainRole::Worker,
            index: 0,
            num_ps: 1,
            num_workers_cluster: 1,
            bind_addr: "127.0.0.1:0".to_string(),
            discovery: DiscoveryBackend::InMemory,
            zk_hosts: None,
            zk_base_path: "/monolith/default".to_string(),
            consul_addr: None,
            consul_service_name: "monolith".to_string(),
            discovery_service_type_ps: "ps".to_string(),
            discovery_service_type_worker: "worker".to_string(),
            table_name: "emb".to_string(),
            dim: 64,
            connect_retries: 6,
            retry_backoff_ms: 500,
            barrier_timeout_ms: 10_000,
            disable_heartbeat: false,
            heartbeat_interval_secs: 10,
            discovery_operation_timeout_ms: 5000,
            discovery_cleanup_timeout_ms: 200,
            parameter_sync_targets: Vec::new(),
            parameter_sync_interval_ms: 200,
            parameter_sync_model_name: "default".to_string(),
            parameter_sync_signature_name: "serving_default".to_string(),
        }
    }

    #[test]
    fn test_train_command_defaults() {
        let cmd = test_cmd_defaults();

        assert_eq!(cmd.train_steps, 10000);
        assert_eq!(cmd.batch_size, 128);
        assert!(cmd.resume);
    }

    #[test]
    fn test_build_distributed_run_config_maps_timeout_and_heartbeat_fields() {
        let mut cmd = test_cmd_defaults();
        cmd.distributed = true;
        cmd.role = TrainRole::Ps;
        cmd.index = 3;
        cmd.num_ps = 2;
        cmd.num_workers_cluster = 4;
        cmd.bind_addr = "127.0.0.1:12345".to_string();
        cmd.discovery_operation_timeout_ms = 1234;
        cmd.discovery_cleanup_timeout_ms = 4321;
        cmd.heartbeat_interval_secs = 7;
        cmd.parameter_sync_targets = vec!["127.0.0.1:5100".to_string()];
        cmd.parameter_sync_interval_ms = 345;

        let cfg = cmd.build_distributed_run_config().unwrap().unwrap();
        assert!(matches!(cfg.role, Role::Ps));
        assert_eq!(cfg.index, 3);
        assert_eq!(cfg.num_ps, 2);
        assert_eq!(cfg.num_workers, 4);
        assert_eq!(cfg.bind_addr.to_string(), "127.0.0.1:12345");
        assert_eq!(cfg.discovery_operation_timeout, Duration::from_millis(1234));
        assert_eq!(cfg.discovery_cleanup_timeout, Duration::from_millis(4321));
        assert_eq!(cfg.heartbeat_interval, Some(Duration::from_secs(7)));
        assert_eq!(cfg.parameter_sync_targets, vec!["127.0.0.1:5100".to_string()]);
        assert_eq!(cfg.parameter_sync_interval, Duration::from_millis(345));
    }

    #[test]
    fn test_build_distributed_run_config_returns_none_for_non_distributed_mode() {
        let cmd = test_cmd_defaults();
        let cfg = cmd.build_distributed_run_config().unwrap();
        assert!(cfg.is_none());
    }

    #[test]
    fn test_build_distributed_run_config_disables_heartbeat_when_requested() {
        let mut cmd = test_cmd_defaults();
        cmd.distributed = true;
        cmd.disable_heartbeat = true;
        cmd.heartbeat_interval_secs = 9;
        let cfg = cmd.build_distributed_run_config().unwrap().unwrap();
        assert_eq!(cfg.heartbeat_interval, None);

        cmd.disable_heartbeat = false;
        cmd.heartbeat_interval_secs = 0;
        let cfg = cmd.build_distributed_run_config().unwrap().unwrap();
        assert_eq!(cfg.heartbeat_interval, None);
    }

    #[test]
    fn test_build_distributed_run_config_rejects_invalid_bind_addr() {
        let mut cmd = test_cmd_defaults();
        cmd.distributed = true;
        cmd.bind_addr = "bad-bind-addr".to_string();
        let err = cmd.build_distributed_run_config().unwrap_err().to_string();
        assert!(
            err.contains("Invalid --bind-addr"),
            "unexpected bind-addr parse error: {err}"
        );
    }

    #[test]
    fn test_tfrecord_training_creates_checkpoint() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path().join("model");
        std::fs::create_dir_all(&model_dir).unwrap();

        let tfrecord_path = dir.path().join("train.tfrecord");
        let mut writer = TFRecordWriter::new(File::create(&tfrecord_path).unwrap());
        for i in 0..5 {
            let mut ex = create_example();
            add_feature(&mut ex, "user_id", vec![i], vec![1.0]);
            ex.label = vec![1.0];
            writer.write_example(&ex).unwrap();
        }
        writer.flush().unwrap();

        let cmd = TrainCommand {
            model_dir: model_dir.clone(),
            config_path: None,
            tfrecord_path: Some(tfrecord_path),
            stdin: false,
            train_steps: 3,
            batch_size: 1,
            learning_rate: 0.1,
            save_steps: 2,
            eval_steps: 2,
            resume: false,
            num_workers: 1,
            distributed: false,
            role: TrainRole::Worker,
            index: 0,
            num_ps: 1,
            num_workers_cluster: 1,
            bind_addr: "127.0.0.1:0".to_string(),
            discovery: DiscoveryBackend::InMemory,
            zk_hosts: None,
            zk_base_path: "/monolith/default".to_string(),
            consul_addr: None,
            consul_service_name: "monolith".to_string(),
            discovery_service_type_ps: "ps".to_string(),
            discovery_service_type_worker: "worker".to_string(),
            table_name: "emb".to_string(),
            dim: 64,
            connect_retries: 6,
            retry_backoff_ms: 500,
            barrier_timeout_ms: 10_000,
            disable_heartbeat: false,
            heartbeat_interval_secs: 10,
            discovery_operation_timeout_ms: 5000,
            discovery_cleanup_timeout_ms: 200,
            parameter_sync_targets: Vec::new(),
            parameter_sync_interval_ms: 200,
            parameter_sync_model_name: "default".to_string(),
            parameter_sync_signature_name: "serving_default".to_string(),
        };

        run_tfrecord_training(&cmd, cmd.tfrecord_path.as_ref().unwrap()).unwrap();

        let checkpoint = model_dir.join("checkpoint-3.json");
        assert!(checkpoint.exists());
    }
}
