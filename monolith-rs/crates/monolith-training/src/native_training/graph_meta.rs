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
    if let Some(existing) = g.get(key).and_then(|v| v.downcast_ref::<T>()) {
        return existing.clone();
    }

    if g.contains_key(key) {
        tracing::warn!(
            graph_id = %graph_id,
            key = %key,
            "graph_meta type mismatch encountered; replacing stored value with factory output"
        );
    }

    let created = factory();
    g.insert(key.to_string(), Box::new(created.clone()));
    created
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
    let mut value = if let Some(existing) = g.remove(key) {
        match existing.downcast::<T>() {
            Ok(existing) => *existing,
            Err(_) => {
                tracing::warn!(
                    graph_id = %graph_id,
                    key = %key,
                    "graph_meta type mismatch encountered during update; replacing stored value with factory output"
                );
                factory()
            }
        }
    } else {
        factory()
    };
    update(&mut value);
    let ret = value.clone();
    g.insert(key.to_string(), Box::new(value));
    ret
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
        join_result.expect_err("poisoning thread should panic to poison graph-meta store mutex");

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
        join_result.expect_err("poisoning thread should panic to poison graph-meta store mutex");

        let updated = update_meta("poisoned-graph-update", "counter", || 5i32, |v| *v += 8);
        assert_eq!(
            updated, 13,
            "update_meta should recover from poisoned graph-meta mutex and apply update"
        );
    }

    #[test]
    fn test_get_meta_cloned_type_mismatch_replaces_stored_value() {
        let graph_id = "g-mismatch";
        let key = "k";
        let _ = get_meta_cloned(graph_id, key, || 7i32);

        let repaired = get_meta_cloned(graph_id, key, || "ok".to_string());
        assert_eq!(
            repaired, "ok",
            "type mismatch should replace stale graph-meta value with factory output"
        );

        let stable = get_meta_cloned(graph_id, key, || "new".to_string());
        assert_eq!(
            stable, "ok",
            "subsequent reads should reuse replaced graph-meta value after type-mismatch repair"
        );
    }

    #[test]
    fn test_update_meta_type_mismatch_replaces_stored_value() {
        let graph_id = "g-mismatch-update";
        let key = "k";
        let _ = get_meta_cloned(graph_id, key, || 3i32);

        let updated = update_meta(graph_id, key, || "x".to_string(), |v| v.push('y'));
        assert_eq!(
            updated, "xy",
            "type mismatch during update should replace stale graph-meta value before applying update"
        );

        let stable = get_meta_cloned(graph_id, key, || "z".to_string());
        assert_eq!(
            stable, "xy",
            "graph-meta value should persist after mismatch-recovery update"
        );
    }
}
