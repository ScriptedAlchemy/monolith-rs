//! In-memory embedding store for serving / parameter sync.
//!
//! This is a minimal store used to apply `parameter_sync.proto` PushRequest deltas
//! into a queryable in-memory map. This provides the "online PS" side needed for
//! end-to-end parity with Python Monolith parameter sync behavior.

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

/// Thread-safe in-memory embedding store.
#[derive(Debug, Default)]
pub struct EmbeddingStore {
    // table_unique_id -> fid -> embedding
    tables: RwLock<HashMap<String, HashMap<i64, Vec<f32>>>>,
}

impl EmbeddingStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply a delta payload.
    ///
    /// `embeddings` is a flat buffer with length `fids.len() * dim`.
    pub fn apply_delta(
        &self,
        unique_id: &str,
        dim: usize,
        fids: &[i64],
        embeddings: &[f32],
    ) -> usize {
        if fids.is_empty() {
            return 0;
        }
        if embeddings.len() != fids.len() * dim {
            // Malformed request; drop.
            return 0;
        }

        let mut tables = self.tables.write();
        let table = tables.entry(unique_id.to_string()).or_default();

        for (i, &fid) in fids.iter().enumerate() {
            let start = i * dim;
            let end = start + dim;
            table.insert(fid, embeddings[start..end].to_vec());
        }

        fids.len()
    }

    /// Lookup embeddings for a set of fids. Missing fids are omitted.
    pub fn lookup(&self, unique_id: &str, fids: &[i64]) -> Vec<Vec<f32>> {
        let tables = self.tables.read();
        let Some(table) = tables.get(unique_id) else {
            return Vec::new();
        };
        fids.iter()
            .filter_map(|fid| table.get(fid).cloned())
            .collect()
    }

    /// Fetch a single embedding.
    pub fn get(&self, unique_id: &str, fid: i64) -> Option<Vec<f32>> {
        let tables = self.tables.read();
        tables.get(unique_id).and_then(|t| t.get(&fid).cloned())
    }
}

/// Shared embedding store handle.
pub type SharedEmbeddingStore = Arc<EmbeddingStore>;
