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

fn lock_store_recover(
) -> std::sync::MutexGuard<'static, HashMap<String, HashMap<String, Box<dyn Any + Send + Sync>>>> {
    match store().lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("graph meta store mutex was poisoned; continuing with recovered state");
            poisoned.into_inner()
        }
    }
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
    let mut guard = lock_store_recover();
    let g = guard.entry(graph_id.to_string()).or_default();
    if !g.contains_key(key) {
        g.insert(key.to_string(), Box::new(factory()));
    }

    g.get(key)
        .and_then(|v| v.downcast_ref::<T>())
        .unwrap_or_else(|| {
            panic!("graph_meta type mismatch for graph_id={graph_id}, key={key}")
        })
        .clone()
}

/// Update a stored value (or create it) and return the new cloned value.
pub fn update_meta<T: Clone + Send + Sync + 'static>(
    graph_id: &str,
    key: &str,
    factory: impl FnOnce() -> T,
    update: impl FnOnce(&mut T),
) -> T {
    let mut guard = lock_store_recover();
    let g = guard.entry(graph_id.to_string()).or_default();
    if !g.contains_key(key) {
        g.insert(key.to_string(), Box::new(factory()));
    }
    let v = g
        .get_mut(key)
        .and_then(|b| b.downcast_mut::<T>())
        .unwrap_or_else(|| {
            panic!("graph_meta type mismatch for graph_id={graph_id}, key={key}")
        });
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

    #[test]
    fn test_get_meta_cloned_recovers_after_poisoned_store_mutex() {
        let join_result = std::thread::spawn(|| {
            let _guard = super::store()
                .lock()
                .expect("graph meta mutex acquisition should succeed before poisoning");
            panic!("poisoning graph-meta store mutex for recovery-path regression");
        })
        .join();
        assert!(
            join_result.is_err(),
            "poisoning thread should panic to poison graph-meta store mutex"
        );

        let v = get_meta_cloned("poisoned-graph", "counter", || 17i32);
        assert_eq!(
            v, 17,
            "get_meta_cloned should recover from poisoned graph-meta mutex and return factory value"
        );
    }

    #[test]
    fn test_update_meta_recovers_after_poisoned_store_mutex() {
        let join_result = std::thread::spawn(|| {
            let _guard = super::store()
                .lock()
                .expect("graph meta mutex acquisition should succeed before poisoning");
            panic!("poisoning graph-meta store mutex for update recovery-path regression");
        })
        .join();
        assert!(
            join_result.is_err(),
            "poisoning thread should panic to poison graph-meta store mutex"
        );

        let updated = update_meta("poisoned-graph-update", "counter", || 5i32, |v| *v += 8);
        assert_eq!(
            updated, 13,
            "update_meta should recover from poisoned graph-meta mutex and apply update"
        );
    }
}
