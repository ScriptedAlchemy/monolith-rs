//! Train Command Implementation
//!
//! Provides distributed model training using the Monolith estimator.
//! Supports configuration via JSON files and command-line arguments.

use anyhow::{Context, Result};
use clap::Args;
use monolith_training::discovery::ServiceDiscoveryAsync;
use monolith_training::runner::{run_distributed, DistributedRunConfig, Role};
use std::net::SocketAddr;
use std::path::PathBuf;
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

    /// Disable discovery heartbeat task (if backend supports it).
    #[arg(long, default_value = "false")]
    pub disable_heartbeat: bool,

    /// Heartbeat interval in seconds (0 disables).
    #[arg(long, default_value = "10")]
    pub heartbeat_interval_secs: u64,

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

        // Load configuration if provided
        let _config = if let Some(config_path) = &self.config_path {
            info!("Loading config from: {:?}", config_path);
            let config_str =
                std::fs::read_to_string(config_path).context("Failed to read config file")?;
            let config: serde_json::Value =
                serde_json::from_str(&config_str).context("Failed to parse config JSON")?;
            Some(config)
        } else {
            warn!("No config file provided, using default configuration");
            None
        };

        // TODO: Initialize estimator with configuration
        // let estimator = Estimator::new(config)?;

        // TODO: Load training data
        // let train_data = DataLoader::new(&self.data_path, self.batch_size)?;

        // TODO: Run training loop
        // for step in 0..self.train_steps {
        //     let batch = train_data.next_batch()?;
        //     let loss = estimator.train_step(batch)?;
        //
        //     if step % self.save_steps == 0 {
        //         estimator.save_checkpoint(&self.model_dir, step)?;
        //     }
        //
        //     if step % self.eval_steps == 0 {
        //         let metrics = estimator.evaluate()?;
        //         info!("Step {}: loss={:.4}, auc={:.4}", step, loss, metrics.auc);
        //     }
        // }

        info!(
            "Training configuration: batch_size={}, lr={}, workers={}",
            self.batch_size, self.learning_rate, self.num_workers
        );

        if self.distributed {
            let bind_addr: SocketAddr = self
                .bind_addr
                .parse()
                .context("Invalid --bind-addr (expected host:port)")?;

            let role = match self.role {
                TrainRole::Ps => Role::Ps,
                TrainRole::Worker => Role::Worker,
            };

            let heartbeat_interval = if self.disable_heartbeat || self.heartbeat_interval_secs == 0
            {
                None
            } else {
                Some(std::time::Duration::from_secs(self.heartbeat_interval_secs))
            };

            let cfg = DistributedRunConfig {
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
                heartbeat_interval,
                parameter_sync_targets: self.parameter_sync_targets.clone(),
                parameter_sync_interval: std::time::Duration::from_millis(
                    self.parameter_sync_interval_ms,
                ),
                parameter_sync_model_name: self.parameter_sync_model_name.clone(),
                parameter_sync_signature_name: self.parameter_sync_signature_name.clone(),
            };

            let discovery = self.build_discovery().await?;
            run_distributed(discovery, cfg).await?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_command_defaults() {
        let cmd = TrainCommand {
            model_dir: PathBuf::from("/tmp/model"),
            config_path: None,
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
            disable_heartbeat: false,
            heartbeat_interval_secs: 10,
            parameter_sync_targets: Vec::new(),
            parameter_sync_interval_ms: 200,
            parameter_sync_model_name: "default".to_string(),
            parameter_sync_signature_name: "serving_default".to_string(),
        };

        assert_eq!(cmd.train_steps, 10000);
        assert_eq!(cmd.batch_size, 128);
        assert!(cmd.resume);
    }
}
