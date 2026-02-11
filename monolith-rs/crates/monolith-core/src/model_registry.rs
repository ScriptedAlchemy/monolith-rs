//! Model registry for mapping model keys to parameter factories.
//!
//! Mirrors Python `monolith/core/model_registry.py` behavior.

use std::collections::BTreeMap;
use std::sync::RwLock;

use once_cell::sync::Lazy;

use crate::base_model_params::SingleTaskModelParams;
use crate::error::{MonolithError, Result};
use crate::hyperparams::Params;
use crate::model_imports;

type ModelFactory = fn() -> Box<dyn SingleTaskModelParams>;

static REGISTRY: Lazy<RwLock<BTreeMap<String, ModelFactory>>> =
    Lazy::new(|| RwLock::new(BTreeMap::new()));

/// Clears the global registry.
///
/// This is intended for tests to avoid cross-test pollution.
pub fn clear_registry_for_test() {
    if let Ok(mut reg) = REGISTRY.write() {
        reg.clear();
    }
}

/// Registers a single-task model factory under a key.
pub fn register_single_task_model(key: &str, factory: ModelFactory) -> Result<()> {
    let mut registry = REGISTRY.write().map_err(|_| MonolithError::ConfigError {
        message: "Model registry lock poisoned".to_string(),
    })?;
    if registry.contains_key(key) {
        // Python error: 'Duplicate model registered for key {}: {}.{}'.format(key, module, cls)
        // We don't have module/class info for a plain factory function; preserve the exact
        // message prefix and fill in placeholders deterministically.
        return Err(MonolithError::DuplicateModelRegistered {
            key: key.to_string(),
            module: "<unknown>".to_string(),
            class_name: "<unknown>".to_string(),
        });
    }
    registry.insert(key.to_string(), factory);
    Ok(())
}

/// Returns a snapshot of all registered model keys.
pub fn get_all_registered() -> Result<Vec<String>> {
    // Python triggers imports before listing registered classes.
    let _ = model_imports::import_all_params(model_imports::DEFAULT_TASK_ROOT, &[], false)?;
    let registry = REGISTRY.read().map_err(|_| MonolithError::ConfigError {
        message: "Model registry lock poisoned".to_string(),
    })?;
    Ok(registry.keys().cloned().collect())
}

/// Retrieves the factory for a given key.
pub fn get_class(key: &str) -> Result<ModelFactory> {
    // Python triggers imports for the requested class_key.
    let _ = model_imports::import_params(key, model_imports::DEFAULT_TASK_ROOT, &[], false)?;
    let registry = REGISTRY.read().map_err(|_| MonolithError::ConfigError {
        message: "Model registry lock poisoned".to_string(),
    })?;
    registry
        .get(key)
        .copied()
        .ok_or_else(|| MonolithError::ModelNotFound {
            key: key.to_string(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::base_model_params::SingleTaskModelParams;
    use crate::hyperparams::Params;

    #[derive(Default)]
    struct DummyParams;

    impl SingleTaskModelParams for DummyParams {
        fn task(&self) -> Params {
            let mut p = Params::new();
            let _ = p.define("dummy", 1_i64, "dummy param");
            p
        }
    }

    #[test]
    fn test_register_duplicate_error_message() {
        clear_registry_for_test();

        let key = "monolith.tasks.dummy.Dummy";
        register_single_task_model(key, || Box::new(DummyParams::default())).unwrap();

        let err = register_single_task_model(key, || Box::new(DummyParams::default())).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Duplicate model registered for key monolith.tasks.dummy.Dummy: <unknown>.<unknown>"
        );
    }

    #[test]
    fn test_get_class_not_found_error_message() {
        clear_registry_for_test();

        let err = get_class("does.not.Exist").unwrap_err();
        assert_eq!(
            err.to_string(),
            "Model does.not.Exist not found from list of above known models."
        );
    }
}
