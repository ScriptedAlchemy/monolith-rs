//! TF-free parity shims for Python `monolith.native_training.distributed_serving_ops`.
//!
//! The Python module provides TF custom ops for:
//! - remote predict (TFServing remote graph inference)
//! - parameter sync client/server handles (C++ op wrappers)
//! - building / refreshing parameter sync configs from agent-service backends
//!
//! Rust parity focuses on the wire-level pieces we can exercise without TensorFlow:
//! - building a `monolith.parameter_sync.ClientConfig` equivalent (from ZK backend)
//! - a small `ParameterSyncClient` wrapper around the Rust gRPC client
//!
//! This keeps TF runtime optional and avoids vendoring TF binaries.

use monolith_proto::monolith::parameter_sync::{client_config::TargetExtraInfo, ClientConfig};
use monolith_serving::backends::ZkBackend;

#[derive(Debug, thiserror::Error)]
pub enum DistributedServingOpsError {
    #[error("backend error: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, DistributedServingOpsError>;

/// Build a `ClientConfig` matching the Python `refresh_sync_config()` behavior (ZKBackend path).
///
/// Python variants:
/// - `SyncBackend.get_sync_targets("ps_{i}")` returns `(saved_model, targets)`
/// - config.model_name is set to `saved_model`
/// - config.signature_name is `"hashtable_assign"`
/// - config.timeout_in_ms is 3000
///
/// In Rust we reuse `monolith_serving::backends::ZkBackend` directly.
pub fn refresh_sync_config(backend: &ZkBackend, ps_index: i32) -> Result<ClientConfig> {
    let (model_name, targets) = backend
        .get_sync_targets(&format!("ps_{ps_index}"))
        .map_err(|e| DistributedServingOpsError::Backend(e.to_string()))?;

    let mut cfg = ClientConfig::default();
    cfg.model_name = Some(model_name);
    cfg.signature_name = Some("hashtable_assign".to_string());
    cfg.timeout_in_ms = Some(3000);
    cfg.targets = targets;
    // Rust ZkBackend currently only returns plain targets; extra info is optional.
    cfg.targets_extra_info = Vec::<TargetExtraInfo>::new();
    Ok(cfg)
}
