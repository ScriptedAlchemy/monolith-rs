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
    pub num_ps: usize,
    pub num_workers: usize,
    pub model_dir: PathBuf,
    pub log_step_count_steps: u64,
    pub discovery_type: ServiceDiscoveryType,
    pub enable_gpu_training: bool,
    pub embedding_prefetch_capacity: usize,
    pub enable_embedding_postpush: bool,
    pub enable_realtime_training: bool,
    pub enable_parameter_sync: bool,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            is_local: true,
            num_ps: 0,
            num_workers: 1,
            model_dir: PathBuf::from("./model"),
            log_step_count_steps: 100,
            discovery_type: ServiceDiscoveryType::Primus,
            enable_gpu_training: false,
            embedding_prefetch_capacity: 1,
            enable_embedding_postpush: true,
            enable_realtime_training: false,
            enable_parameter_sync: false,
        }
    }
}

/// User-facing run configuration that can be merged into [`RunnerConfig`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunConfig {
    pub is_local: bool,
    pub num_ps: usize,
    pub num_workers: usize,
    pub model_dir: PathBuf,
    pub log_step_count_steps: u64,
    pub discovery_type: ServiceDiscoveryType,
    pub enable_gpu_training: bool,
    pub embedding_prefetch_capacity: usize,
    pub enable_embedding_postpush: bool,
    pub enable_parameter_sync: bool,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            is_local: true,
            num_ps: 0,
            num_workers: 1,
            model_dir: PathBuf::from("./model"),
            log_step_count_steps: 100,
            discovery_type: ServiceDiscoveryType::Primus,
            enable_gpu_training: false,
            embedding_prefetch_capacity: 0,
            enable_embedding_postpush: false,
            enable_parameter_sync: false,
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
        merge_field!(num_ps);
        merge_field!(num_workers);
        merge_field!(model_dir);
        merge_field!(log_step_count_steps);
        merge_field!(enable_gpu_training);
        merge_field!(embedding_prefetch_capacity);
        merge_field!(enable_embedding_postpush);

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
        push_override!(num_ps);
        push_override!(num_workers);
        push_override!(model_dir);
        push_override!(log_step_count_steps);
        push_override!(discovery_type);
        push_override!(enable_gpu_training);
        push_override!(embedding_prefetch_capacity);
        push_override!(enable_embedding_postpush);
        push_override!(enable_parameter_sync);

        out
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
        let merged = rc.to_runner_config(Some(base.clone())).unwrap();
        assert_eq!(merged.num_ps, 3);
        assert_eq!(merged.num_workers, 8);
    }

    #[test]
    fn test_run_config_merge_overrides_explicit_values() {
        let rc = RunConfig {
            num_ps: 2,
            num_workers: 5,
            ..RunConfig::default()
        };
        let base = RunnerConfig {
            num_ps: 1,
            num_workers: 1,
            ..RunnerConfig::default()
        };
        let merged = rc.to_runner_config(Some(base)).unwrap();
        assert_eq!(merged.num_ps, 2);
        assert_eq!(merged.num_workers, 5);
    }

    #[test]
    fn test_run_config_consul_maps_to_zk() {
        let rc = RunConfig {
            discovery_type: ServiceDiscoveryType::Consul,
            ..RunConfig::default()
        };
        let merged = rc.to_runner_config(None).unwrap();
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
        let merged = rc.to_runner_config(None).unwrap();
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
        let merged = rc.to_runner_config(Some(base)).unwrap();
        assert!(merged.enable_realtime_training);
        assert!(merged.enable_parameter_sync);
    }

    #[test]
    fn test_user_overrides() {
        let rc = RunConfig {
            num_ps: 2,
            enable_parameter_sync: true,
            ..RunConfig::default()
        };
        let overrides = rc.user_overrides();
        assert_eq!(overrides.get("num_ps").unwrap(), &serde_json::json!(2));
        assert_eq!(
            overrides.get("enable_parameter_sync").unwrap(),
            &serde_json::json!(true)
        );
    }
}
