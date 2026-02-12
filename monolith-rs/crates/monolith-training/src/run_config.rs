//! RunConfig / RunnerConfig parity layer for native training.
//!
//! Python's `native_training.estimator.RunConfig` carries user intent and merges
//! into an execution-time runner config while preserving CLI overrides.
//! This module implements that merge behavior for Rust-native training flows.

use crate::native_training::service_discovery::ServiceDiscoveryType;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RunConfigError {
    /// Parameter sync was requested but no compatible runner toggle exists.
    #[error("enable_parameter_sync requested, but runner config has no compatible sync field")]
    MissingSyncToggle,
}

pub type Result<T> = std::result::Result<T, RunConfigError>;

/// Execution-time runner configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunnerConfig {
    pub is_local: bool,
    pub index: usize,
    pub num_ps: usize,
    pub num_workers: usize,
    pub model_dir: PathBuf,
    pub restore_dir: Option<PathBuf>,
    pub restore_ckpt: Option<String>,
    pub restore_sync_timeout_secs: u64,
    pub restore_sync_poll_interval_ms: u64,
    pub connect_retries: usize,
    pub retry_backoff_ms: u64,
    pub barrier_timeout_ms: i64,
    pub discovery_operation_timeout_ms: u64,
    pub discovery_cleanup_timeout_ms: u64,
    pub log_step_count_steps: u64,
    pub discovery_type: ServiceDiscoveryType,
    pub discovery_service_type_ps: String,
    pub discovery_service_type_worker: String,
    pub tf_config: Option<String>,
    pub deep_insight_name: String,
    pub zk_server: String,
    pub table_name: String,
    pub dim: usize,
    pub enable_gpu_training: bool,
    pub embedding_prefetch_capacity: usize,
    pub enable_embedding_postpush: bool,
    pub enable_realtime_training: bool,
    pub enable_parameter_sync: bool,
    pub parameter_sync_targets: Vec<String>,
    pub parameter_sync_interval_ms: u64,
    pub parameter_sync_model_name: String,
    pub parameter_sync_signature_name: String,
    pub tf_grpc_worker_cache_threads: Option<usize>,
    pub monolith_grpc_worker_service_handler_multiplier: Option<usize>,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            is_local: true,
            index: 0,
            num_ps: 0,
            num_workers: 1,
            model_dir: PathBuf::from("./model"),
            restore_dir: None,
            restore_ckpt: None,
            restore_sync_timeout_secs: 3600,
            restore_sync_poll_interval_ms: 30_000,
            connect_retries: 6,
            retry_backoff_ms: 500,
            barrier_timeout_ms: 10_000,
            discovery_operation_timeout_ms: 5_000,
            discovery_cleanup_timeout_ms: 200,
            log_step_count_steps: 100,
            discovery_type: ServiceDiscoveryType::Primus,
            discovery_service_type_ps: "ps".to_string(),
            discovery_service_type_worker: "worker".to_string(),
            tf_config: None,
            deep_insight_name: "monolith".to_string(),
            zk_server: "127.0.0.1:2181".to_string(),
            table_name: "emb".to_string(),
            dim: 64,
            enable_gpu_training: false,
            embedding_prefetch_capacity: 1,
            enable_embedding_postpush: true,
            enable_realtime_training: false,
            enable_parameter_sync: false,
            parameter_sync_targets: Vec::new(),
            parameter_sync_interval_ms: 200,
            parameter_sync_model_name: "default".to_string(),
            parameter_sync_signature_name: "serving_default".to_string(),
            tf_grpc_worker_cache_threads: None,
            monolith_grpc_worker_service_handler_multiplier: None,
        }
    }
}

/// User-facing run configuration that can be merged into [`RunnerConfig`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunConfig {
    pub is_local: bool,
    pub index: usize,
    pub num_ps: usize,
    pub num_workers: usize,
    pub model_dir: PathBuf,
    pub restore_dir: Option<PathBuf>,
    pub restore_ckpt: Option<String>,
    pub restore_sync_timeout_secs: u64,
    pub restore_sync_poll_interval_ms: u64,
    pub connect_retries: usize,
    pub retry_backoff_ms: u64,
    pub barrier_timeout_ms: i64,
    pub discovery_operation_timeout_ms: u64,
    pub discovery_cleanup_timeout_ms: u64,
    pub log_step_count_steps: u64,
    pub discovery_type: ServiceDiscoveryType,
    pub discovery_service_type_ps: String,
    pub discovery_service_type_worker: String,
    pub tf_config: Option<String>,
    pub deep_insight_name: String,
    pub zk_server: String,
    pub table_name: String,
    pub dim: usize,
    pub enable_gpu_training: bool,
    pub embedding_prefetch_capacity: usize,
    pub enable_embedding_postpush: bool,
    pub enable_parameter_sync: bool,
    pub parameter_sync_targets: Vec<String>,
    pub parameter_sync_interval_ms: u64,
    pub parameter_sync_model_name: String,
    pub parameter_sync_signature_name: String,
    pub tf_grpc_worker_cache_threads: Option<usize>,
    pub monolith_grpc_worker_service_handler_multiplier: Option<usize>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            is_local: true,
            index: 0,
            num_ps: 0,
            num_workers: 1,
            model_dir: PathBuf::from("./model"),
            restore_dir: None,
            restore_ckpt: None,
            restore_sync_timeout_secs: 3600,
            restore_sync_poll_interval_ms: 30_000,
            connect_retries: 6,
            retry_backoff_ms: 500,
            barrier_timeout_ms: 10_000,
            discovery_operation_timeout_ms: 5_000,
            discovery_cleanup_timeout_ms: 200,
            log_step_count_steps: 100,
            discovery_type: ServiceDiscoveryType::Primus,
            discovery_service_type_ps: "ps".to_string(),
            discovery_service_type_worker: "worker".to_string(),
            tf_config: None,
            deep_insight_name: "monolith".to_string(),
            zk_server: "127.0.0.1:2181".to_string(),
            table_name: "emb".to_string(),
            dim: 64,
            enable_gpu_training: false,
            embedding_prefetch_capacity: 0,
            enable_embedding_postpush: false,
            enable_parameter_sync: false,
            parameter_sync_targets: Vec::new(),
            parameter_sync_interval_ms: 200,
            parameter_sync_model_name: "default".to_string(),
            parameter_sync_signature_name: "serving_default".to_string(),
            tf_grpc_worker_cache_threads: None,
            monolith_grpc_worker_service_handler_multiplier: None,
        }
    }
}

impl RunConfig {
    /// Builds a runner config while preserving values that likely came from CLI.
    ///
    /// Merge rule:
    /// - update a field only if this `RunConfig` value differs from `RunConfig::default()`
    /// - and also differs from the current runner value.
    pub fn to_runner_config(&self, base: Option<RunnerConfig>) -> Result<RunnerConfig> {
        let defaults = RunConfig::default();
        let mut conf = base.unwrap_or_default();

        macro_rules! merge_field {
            ($field:ident) => {
                if self.$field != defaults.$field && self.$field != conf.$field {
                    conf.$field = self.$field.clone();
                }
            };
        }

        merge_field!(is_local);
        merge_field!(index);
        merge_field!(num_ps);
        merge_field!(num_workers);
        merge_field!(model_dir);
        merge_field!(restore_dir);
        merge_field!(restore_ckpt);
        merge_field!(restore_sync_timeout_secs);
        merge_field!(restore_sync_poll_interval_ms);
        merge_field!(connect_retries);
        merge_field!(retry_backoff_ms);
        merge_field!(barrier_timeout_ms);
        merge_field!(discovery_operation_timeout_ms);
        merge_field!(discovery_cleanup_timeout_ms);
        merge_field!(log_step_count_steps);
        merge_field!(enable_gpu_training);
        merge_field!(embedding_prefetch_capacity);
        merge_field!(enable_embedding_postpush);
        merge_field!(parameter_sync_targets);
        merge_field!(parameter_sync_interval_ms);
        merge_field!(parameter_sync_model_name);
        merge_field!(parameter_sync_signature_name);
        merge_field!(tf_grpc_worker_cache_threads);
        merge_field!(monolith_grpc_worker_service_handler_multiplier);
        merge_field!(tf_config);
        merge_field!(discovery_service_type_ps);
        merge_field!(discovery_service_type_worker);
        merge_field!(deep_insight_name);
        merge_field!(zk_server);
        merge_field!(table_name);
        merge_field!(dim);

        if self.discovery_type != defaults.discovery_type && self.discovery_type != conf.discovery_type
        {
            conf.discovery_type = match self.discovery_type {
                ServiceDiscoveryType::Consul => ServiceDiscoveryType::Zk,
                other => other,
            };
        }

        // Python parity: when GPU training is disabled force minimum embedding prefetch
        // and enable postpush.
        if !conf.enable_gpu_training {
            conf.embedding_prefetch_capacity = conf.embedding_prefetch_capacity.max(1);
            conf.enable_embedding_postpush = true;
        }

        // Python parity: enabling parameter sync maps into runtime sync toggles.
        if self.enable_parameter_sync {
            conf.enable_realtime_training = true;
            conf.enable_parameter_sync = true;
        }

        Ok(conf)
    }

    /// Returns explicit user-provided overrides relative to defaults.
    pub fn user_overrides(&self) -> BTreeMap<String, serde_json::Value> {
        let defaults = RunConfig::default();
        let mut out = BTreeMap::new();

        macro_rules! push_override {
            ($field:ident) => {
                if self.$field != defaults.$field {
                    out.insert(
                        stringify!($field).to_string(),
                        serde_json::to_value(&self.$field).unwrap_or(serde_json::Value::Null),
                    );
                }
            };
        }

        push_override!(is_local);
        push_override!(index);
        push_override!(num_ps);
        push_override!(num_workers);
        push_override!(model_dir);
        push_override!(restore_dir);
        push_override!(restore_ckpt);
        push_override!(restore_sync_timeout_secs);
        push_override!(restore_sync_poll_interval_ms);
        push_override!(connect_retries);
        push_override!(retry_backoff_ms);
        push_override!(barrier_timeout_ms);
        push_override!(discovery_operation_timeout_ms);
        push_override!(discovery_cleanup_timeout_ms);
        push_override!(log_step_count_steps);
        push_override!(discovery_type);
        push_override!(discovery_service_type_ps);
        push_override!(discovery_service_type_worker);
        push_override!(tf_config);
        push_override!(deep_insight_name);
        push_override!(zk_server);
        push_override!(table_name);
        push_override!(dim);
        push_override!(enable_gpu_training);
        push_override!(embedding_prefetch_capacity);
        push_override!(enable_embedding_postpush);
        push_override!(enable_parameter_sync);
        push_override!(parameter_sync_targets);
        push_override!(parameter_sync_interval_ms);
        push_override!(parameter_sync_model_name);
        push_override!(parameter_sync_signature_name);
        push_override!(tf_grpc_worker_cache_threads);
        push_override!(monolith_grpc_worker_service_handler_multiplier);

        out
    }

    /// Exports runtime env vars expected by Python estimator initialization.
    ///
    /// Mirrors:
    /// - `TF_GRPC_WORKER_CACHE_THREADS`
    /// - `MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER`
    pub fn apply_runtime_env_exports(runner: &RunnerConfig) {
        if let Some(v) = runner.tf_grpc_worker_cache_threads {
            std::env::set_var("TF_GRPC_WORKER_CACHE_THREADS", v.to_string());
        }
        if let Some(v) = runner.monolith_grpc_worker_service_handler_multiplier {
            std::env::set_var(
                "MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER",
                v.to_string(),
            );
        }
    }

    /// Builds an estimator config directly from user-facing run config values.
    pub fn to_estimator_config(&self) -> crate::estimator::EstimatorConfig {
        let mut cfg = crate::estimator::EstimatorConfig::new(self.model_dir.clone())
            .with_log_step_count_steps(self.log_step_count_steps);
        if let Some(restore_ckpt) = &self.restore_ckpt {
            cfg = cfg.with_warm_start_from(PathBuf::from(restore_ckpt));
        }
        cfg
    }
}

impl RunnerConfig {
    /// Builds an estimator config from execution-time runner configuration.
    pub fn to_estimator_config(&self) -> crate::estimator::EstimatorConfig {
        let mut cfg = crate::estimator::EstimatorConfig::new(self.model_dir.clone())
            .with_log_step_count_steps(self.log_step_count_steps);
        if let Some(restore_ckpt) = &self.restore_ckpt {
            cfg = cfg.with_warm_start_from(PathBuf::from(restore_ckpt));
        }
        cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_config_merge_preserves_cli_values() {
        let rc = RunConfig::default();
        let base = RunnerConfig {
            num_ps: 3,
            num_workers: 8,
            ..RunnerConfig::default()
        };
        let merged = rc
            .to_runner_config(Some(base.clone()))
            .expect("run config should merge with base runner config");
        assert_eq!(merged.num_ps, 3);
        assert_eq!(merged.num_workers, 8);
    }

    #[test]
    fn test_run_config_merge_overrides_explicit_values() {
        let rc = RunConfig {
            index: 3,
            num_ps: 2,
            num_workers: 5,
            restore_dir: Some(PathBuf::from("/tmp/restore")),
            restore_ckpt: Some("model.ckpt-10".to_string()),
            restore_sync_timeout_secs: 99,
            restore_sync_poll_interval_ms: 1234,
            connect_retries: 9,
            retry_backoff_ms: 88,
            barrier_timeout_ms: 4321,
            discovery_operation_timeout_ms: 7654,
            discovery_cleanup_timeout_ms: 321,
            discovery_service_type_ps: "parameter_server".to_string(),
            discovery_service_type_worker: "trainer".to_string(),
            table_name: "item_emb".to_string(),
            dim: 128,
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_interval_ms: 345,
            parameter_sync_model_name: "my_model".to_string(),
            parameter_sync_signature_name: "my_signature".to_string(),
            ..RunConfig::default()
        };
        let base = RunnerConfig {
            index: 0,
            num_ps: 1,
            num_workers: 1,
            ..RunnerConfig::default()
        };
        let merged = rc
            .to_runner_config(Some(base))
            .expect("run config should override explicit fields");
        assert_eq!(merged.index, 3);
        assert_eq!(merged.num_ps, 2);
        assert_eq!(merged.num_workers, 5);
        assert_eq!(merged.restore_dir, Some(PathBuf::from("/tmp/restore")));
        assert_eq!(merged.restore_ckpt, Some("model.ckpt-10".to_string()));
        assert_eq!(merged.restore_sync_timeout_secs, 99);
        assert_eq!(merged.restore_sync_poll_interval_ms, 1234);
        assert_eq!(merged.connect_retries, 9);
        assert_eq!(merged.retry_backoff_ms, 88);
        assert_eq!(merged.barrier_timeout_ms, 4321);
        assert_eq!(merged.discovery_operation_timeout_ms, 7654);
        assert_eq!(merged.discovery_cleanup_timeout_ms, 321);
        assert_eq!(merged.discovery_service_type_ps, "parameter_server");
        assert_eq!(merged.discovery_service_type_worker, "trainer");
        assert_eq!(merged.table_name, "item_emb");
        assert_eq!(merged.dim, 128);
        assert_eq!(merged.parameter_sync_targets, vec!["127.0.0.1:8500".to_string()]);
        assert_eq!(merged.parameter_sync_interval_ms, 345);
        assert_eq!(merged.parameter_sync_model_name, "my_model");
        assert_eq!(merged.parameter_sync_signature_name, "my_signature");
    }

    #[test]
    fn test_run_config_consul_maps_to_zk() {
        let rc = RunConfig {
            discovery_type: ServiceDiscoveryType::Consul,
            ..RunConfig::default()
        };
        let merged = rc
            .to_runner_config(None)
            .expect("consul discovery type should map into runner config");
        assert_eq!(merged.discovery_type, ServiceDiscoveryType::Zk);
    }

    #[test]
    fn test_run_config_cpu_training_forces_prefetch_and_postpush() {
        let rc = RunConfig {
            enable_gpu_training: false,
            embedding_prefetch_capacity: 0,
            enable_embedding_postpush: false,
            ..RunConfig::default()
        };
        let merged = rc
            .to_runner_config(None)
            .expect("cpu training adjustments should apply in runner config");
        assert_eq!(merged.embedding_prefetch_capacity, 1);
        assert!(merged.enable_embedding_postpush);
    }

    #[test]
    fn test_run_config_enables_parameter_sync_flags() {
        let rc = RunConfig {
            enable_parameter_sync: true,
            ..RunConfig::default()
        };
        let base = RunnerConfig {
            enable_realtime_training: true,
            enable_parameter_sync: false,
            ..RunnerConfig::default()
        };
        let merged = rc
            .to_runner_config(Some(base))
            .expect("parameter-sync flags should be merged into runner config");
        assert!(merged.enable_realtime_training);
        assert!(merged.enable_parameter_sync);
    }

    #[test]
    fn test_user_overrides() {
        let rc = RunConfig {
            num_ps: 2,
            enable_parameter_sync: true,
            tf_grpc_worker_cache_threads: Some(8),
            connect_retries: 12,
            retry_backoff_ms: 345,
            barrier_timeout_ms: 9000,
            discovery_operation_timeout_ms: 6789,
            discovery_cleanup_timeout_ms: 321,
            discovery_service_type_ps: "parameter_server".to_string(),
            discovery_service_type_worker: "trainer".to_string(),
            table_name: "item_emb".to_string(),
            dim: 256,
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_interval_ms: 345,
            parameter_sync_model_name: "my_model".to_string(),
            parameter_sync_signature_name: "my_signature".to_string(),
            ..RunConfig::default()
        };
        let overrides = rc.user_overrides();
        assert_eq!(
            overrides
                .get("num_ps")
                .expect("num_ps override should be present"),
            &serde_json::json!(2)
        );
        assert_eq!(
            overrides
                .get("enable_parameter_sync")
                .expect("enable_parameter_sync override should be present"),
            &serde_json::json!(true)
        );
        assert_eq!(
            overrides
                .get("tf_grpc_worker_cache_threads")
                .expect("tf_grpc_worker_cache_threads override should be present"),
            &serde_json::json!(8)
        );
        assert_eq!(
            overrides
                .get("connect_retries")
                .expect("connect_retries override should be present"),
            &serde_json::json!(12)
        );
        assert_eq!(
            overrides
                .get("retry_backoff_ms")
                .expect("retry_backoff_ms override should be present"),
            &serde_json::json!(345)
        );
        assert_eq!(
            overrides
                .get("barrier_timeout_ms")
                .expect("barrier_timeout_ms override should be present"),
            &serde_json::json!(9000)
        );
        assert_eq!(
            overrides
                .get("discovery_operation_timeout_ms")
                .expect("discovery_operation_timeout_ms override should be present"),
            &serde_json::json!(6789)
        );
        assert_eq!(
            overrides
                .get("discovery_cleanup_timeout_ms")
                .expect("discovery_cleanup_timeout_ms override should be present"),
            &serde_json::json!(321)
        );
        assert_eq!(
            overrides
                .get("discovery_service_type_ps")
                .expect("discovery_service_type_ps override should be present"),
            &serde_json::json!("parameter_server")
        );
        assert_eq!(
            overrides
                .get("discovery_service_type_worker")
                .expect("discovery_service_type_worker override should be present"),
            &serde_json::json!("trainer")
        );
        assert_eq!(
            overrides
                .get("table_name")
                .expect("table_name override should be present"),
            &serde_json::json!("item_emb")
        );
        assert_eq!(
            overrides.get("dim").expect("dim override should be present"),
            &serde_json::json!(256)
        );
        assert_eq!(
            overrides
                .get("parameter_sync_targets")
                .expect("parameter_sync_targets override should be present"),
            &serde_json::json!(vec!["127.0.0.1:8500"])
        );
        assert_eq!(
            overrides
                .get("parameter_sync_interval_ms")
                .expect("parameter_sync_interval_ms override should be present"),
            &serde_json::json!(345)
        );
        assert_eq!(
            overrides
                .get("parameter_sync_model_name")
                .expect("parameter_sync_model_name override should be present"),
            &serde_json::json!("my_model")
        );
        assert_eq!(
            overrides
                .get("parameter_sync_signature_name")
                .expect("parameter_sync_signature_name override should be present"),
            &serde_json::json!("my_signature")
        );
    }

    #[test]
    fn test_apply_runtime_env_exports() {
        // Keep this test isolated from other env-var-dependent tests.
        static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let _guard = ENV_MUTEX
            .lock()
            .expect("run-config env mutex should not be poisoned");

        std::env::remove_var("TF_GRPC_WORKER_CACHE_THREADS");
        std::env::remove_var("MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER");
        let rc = RunnerConfig {
            tf_grpc_worker_cache_threads: Some(16),
            monolith_grpc_worker_service_handler_multiplier: Some(3),
            ..RunnerConfig::default()
        };
        RunConfig::apply_runtime_env_exports(&rc);
        assert_eq!(
            std::env::var("TF_GRPC_WORKER_CACHE_THREADS")
                .expect("TF_GRPC_WORKER_CACHE_THREADS should be exported"),
            "16"
        );
        assert_eq!(
            std::env::var("MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER")
                .expect("MONOLITH_GRPC_WORKER_SERVICE_HANDLER_MULTIPLIER should be exported"),
            "3"
        );
    }

    #[test]
    fn test_runner_config_to_estimator_config() {
        let rc = RunnerConfig {
            model_dir: PathBuf::from("/tmp/model_dir"),
            log_step_count_steps: 42,
            restore_ckpt: Some("model.ckpt-30".to_string()),
            ..RunnerConfig::default()
        };
        let est = rc.to_estimator_config();
        assert_eq!(est.model_dir, PathBuf::from("/tmp/model_dir"));
        assert_eq!(est.log_step_count_steps, 42);
        assert_eq!(est.warm_start_from, Some(PathBuf::from("model.ckpt-30")));
    }

    #[test]
    fn test_run_config_to_estimator_config() {
        let rc = RunConfig {
            model_dir: PathBuf::from("/tmp/model_dir2"),
            log_step_count_steps: 21,
            restore_ckpt: Some("model.ckpt-61".to_string()),
            ..RunConfig::default()
        };
        let est = rc.to_estimator_config();
        assert_eq!(est.model_dir, PathBuf::from("/tmp/model_dir2"));
        assert_eq!(est.log_step_count_steps, 21);
        assert_eq!(est.warm_start_from, Some(PathBuf::from("model.ckpt-61")));
    }
}
