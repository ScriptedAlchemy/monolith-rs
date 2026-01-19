//! Model registry for mapping model keys to parameter factories.
//!
//! Mirrors Python `monolith/core/model_registry.py` behavior.

use std::collections::BTreeMap;
use std::sync::RwLock;

use once_cell::sync::Lazy;

use crate::base_model_params::SingleTaskModelParams;
use crate::error::{MonolithError, Result};
use crate::hyperparams::Params;

type ModelFactory = fn() -> Box<dyn SingleTaskModelParams>;

static REGISTRY: Lazy<RwLock<BTreeMap<String, ModelFactory>>> =
    Lazy::new(|| RwLock::new(BTreeMap::new()));

/// Registers a single-task model factory under a key.
pub fn register_single_task_model(key: &str, factory: ModelFactory) -> Result<()> {
    let mut registry = REGISTRY
        .write()
        .map_err(|_| MonolithError::ConfigError {
            message: "Model registry lock poisoned".to_string(),
        })?;
    if registry.contains_key(key) {
        return Err(MonolithError::ConfigError {
            message: format!("Duplicate model registered for key {}", key),
        });
    }
    registry.insert(key.to_string(), factory);
    Ok(())
}

/// Returns a snapshot of all registered model keys.
pub fn get_all_registered() -> Result<Vec<String>> {
    let registry = REGISTRY
        .read()
        .map_err(|_| MonolithError::ConfigError {
            message: "Model registry lock poisoned".to_string(),
        })?;
    Ok(registry.keys().cloned().collect())
}

/// Retrieves the factory for a given key.
pub fn get_class(key: &str) -> Result<ModelFactory> {
    let registry = REGISTRY
        .read()
        .map_err(|_| MonolithError::ConfigError {
            message: "Model registry lock poisoned".to_string(),
        })?;
    registry.get(key).copied().ok_or_else(|| MonolithError::ConfigError {
        message: format!("Model {} not found in registry", key),
    })
}

/// Instantiates params for a given model key.
pub fn get_params(key: &str) -> Result<Params> {
    let factory = get_class(key)?;
    let params = factory();
    Ok(params.task())
}

/// Helper macro to register a model type using its Default impl.
#[macro_export]
macro_rules! register_single_task_model {
    ($key:expr, $ty:ty) => {{
        fn factory() -> Box<dyn $crate::base_model_params::SingleTaskModelParams> {
            Box::new(<$ty as std::default::Default>::default())
        }
        $crate::model_registry::register_single_task_model($key, factory)
    }};
}
