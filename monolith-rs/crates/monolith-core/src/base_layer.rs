//! Base layer utilities mirroring Python `BaseLayer`.

use std::collections::BTreeMap;
use std::sync::RwLock;

use once_cell::sync::Lazy;

use crate::error::{MonolithError, Result};
use crate::hyperparams::InstantiableParams;
use crate::nested_map::{NestedMap, NestedValue};

static NAME_IN_USE: Lazy<RwLock<BTreeMap<String, usize>>> =
    Lazy::new(|| RwLock::new(BTreeMap::new()));
static LAYER_LOSS: Lazy<RwLock<BTreeMap<String, f32>>> = Lazy::new(|| RwLock::new(BTreeMap::new()));

/// Returns a unique name by appending an index if necessary.
pub fn get_uname(name: &str) -> String {
    let mut map = NAME_IN_USE.write().unwrap_or_else(|e| e.into_inner());
    // Mirror Python `monolith/core/base_layer.py::get_uname`:
    // it only appends an index if the name is already present in the counter map.
    if let Some(entry) = map.get_mut(name) {
        *entry += 1;
        format!("{}_{}", name, *entry)
    } else {
        name.to_string()
    }
}

/// Adds a layer loss value to the global registry.
pub fn add_layer_loss(name: &str, loss: f32) {
    let mut map = LAYER_LOSS.write().unwrap_or_else(|e| e.into_inner());
    let entry = map.entry(name.to_string()).or_insert(0.0);
    *entry += loss;
}

/// Returns a snapshot of all recorded layer losses.
pub fn get_layer_loss() -> BTreeMap<String, f32> {
    LAYER_LOSS.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Core container for managing child layers.
#[derive(Clone, Default)]
pub struct BaseLayerCore {
    name: String,
    children: NestedMap,
}

impl BaseLayerCore {
    /// Creates a new BaseLayerCore with a unique name.
    pub fn new(name: &str) -> Self {
        Self {
            name: get_uname(name),
            children: NestedMap::new(),
        }
    }

    /// Returns the unique name for this layer.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns child layers.
    pub fn children(&self) -> &NestedMap {
        &self.children
    }

    /// Returns mutable child layers.
    pub fn children_mut(&mut self) -> &mut NestedMap {
        &mut self.children
    }

    /// Creates a single child layer from params.
    pub fn create_child(&mut self, name: &str, params: &InstantiableParams) -> Result<()> {
        let child = params.instantiate()?;
        self.children
            .insert(name, NestedValue::Value(child))
            .map_err(|e| MonolithError::ConfigError {
                message: format!("Failed to create child {}: {}", name, e),
            })?;
        Ok(())
    }

    /// Creates a list of child layers from a list of params.
    pub fn create_children(&mut self, name: &str, params: &[InstantiableParams]) -> Result<()> {
        let mut list = Vec::with_capacity(params.len());
        for p in params {
            let child = p.instantiate()?;
            list.push(NestedValue::Value(child));
        }
        self.children
            .insert(name, NestedValue::List(list))
            .map_err(|e| MonolithError::ConfigError {
                message: format!("Failed to create children {}: {}", name, e),
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod python_parity_tests {
    use super::*;
    use crate::dyn_value::DynValue;
    use crate::hyperparams::Params;
    use crate::hyperparams::ParamsFactory;
    use std::sync::Arc;

    #[derive(Debug)]
    struct DummyLayer;

    struct DummyLayerFactory;

    impl ParamsFactory for DummyLayerFactory {
        fn type_name(&self) -> &'static str {
            "DummyLayer"
        }

        fn create(&self, _params: &Params) -> Result<Arc<dyn DynValue>> {
            Ok(Arc::new(DummyLayer))
        }
    }

    // Mirrors monolith/core/base_layer_test.py::BaseLayerTest::test_create_child
    #[test]
    fn test_create_child() {
        let mut core = BaseLayerCore::new("BaseLayer");
        let factory = Arc::new(DummyLayerFactory);
        let p = InstantiableParams::new(Some(factory));
        core.create_child("a", &p).unwrap();
        assert!(core.children().get("a").is_some());
    }

    // Mirrors monolith/core/base_layer_test.py::BaseLayerTest::test_create_children
    #[test]
    fn test_create_children() {
        let mut core = BaseLayerCore::new("BaseLayer");
        let factory = Arc::new(DummyLayerFactory);
        let p = InstantiableParams::new(Some(factory));
        core.create_children("a", &[p.clone(), p]).unwrap();

        let children = core.children().get("a").expect("child key `a` should exist");
        assert!(
            matches!(children, NestedValue::List(list) if list.len() == 2),
            "children for key `a` should be stored as a list with two elements"
        );
    }

    #[test]
    fn test_get_uname_parity() {
        // Mirrors current Python implementation in `monolith/core/base_layer.py`:
        // it only appends an index if the name is already present in the counter map.
        assert_eq!(get_uname("X"), "X");
        assert_eq!(get_uname("X"), "X");

        // If the name is pre-populated, `get_uname` increments and appends.
        {
            let mut m = super::NAME_IN_USE.write().unwrap_or_else(|e| e.into_inner());
            m.insert("X".to_string(), 0);
        }
        assert_eq!(get_uname("X"), "X_1");
    }
}
