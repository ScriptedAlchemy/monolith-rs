//! Python `monolith.native_training.graph_meta` parity (conceptual).
//!
//! Python stores per-graph metadata in TensorFlow collections. In Rust we don't
//! have a TF graph; we expose a tiny typed registry keyed by `(graph_id, key)`.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static GRAPH_META: OnceLock<Mutex<HashMap<String, HashMap<String, Box<dyn Any + Send + Sync>>>>> =
    OnceLock::new();

fn store() -> &'static Mutex<HashMap<String, HashMap<String, Box<dyn Any + Send + Sync>>>> {
    GRAPH_META.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Get-or-create a value stored under `(graph_id, key)` and return a cloned copy.
///
/// This avoids returning references tied to a global lock (and avoids unstable
/// `MutexGuard::map` APIs).
pub fn get_meta_cloned<T: Clone + Send + Sync + 'static>(
    graph_id: &str,
    key: &str,
    factory: impl FnOnce() -> T,
) -> T {
    let mut guard = store().lock().unwrap();
    let g = guard.entry(graph_id.to_string()).or_default();
    if !g.contains_key(key) {
        g.insert(key.to_string(), Box::new(factory()));
    }

    g.get(key)
        .and_then(|v| v.downcast_ref::<T>())
        .expect("graph_meta type mismatch")
        .clone()
}

/// Update a stored value (or create it) and return the new cloned value.
pub fn update_meta<T: Clone + Send + Sync + 'static>(
    graph_id: &str,
    key: &str,
    factory: impl FnOnce() -> T,
    update: impl FnOnce(&mut T),
) -> T {
    let mut guard = store().lock().unwrap();
    let g = guard.entry(graph_id.to_string()).or_default();
    if !g.contains_key(key) {
        g.insert(key.to_string(), Box::new(factory()));
    }
    let v = g
        .get_mut(key)
        .and_then(|b| b.downcast_mut::<T>())
        .expect("graph_meta type mismatch");
    update(v);
    v.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_meta_cloned_factory_only_once() {
        let graph_id = "g0";
        let key = "k";

        let v1 = get_meta_cloned(graph_id, key, || 123i32);
        assert_eq!(v1, 123);

        let v2 = update_meta(graph_id, key, || 999i32, |v| *v = 456);
        assert_eq!(v2, 456);

        let v3 = get_meta_cloned(graph_id, key, || 999i32);
        assert_eq!(v3, 456);
    }
}
