//! ParameterSync PushSink implementation that applies deltas to an EmbeddingStore.

use crate::embedding_store::SharedEmbeddingStore;
use crate::error::{ServingError, ServingResult};
use crate::parameter_sync_rpc::PushSink;
use monolith_proto::monolith::parameter_sync::push_request::DeltaEmbeddingHashTable;
use monolith_proto::monolith::parameter_sync::{PushRequest, PushResponse};
use std::sync::Arc;

#[derive(Debug, Clone)]
/// PushSink that applies pushed embeddings into an [`EmbeddingStore`].
pub struct EmbeddingStorePushSink {
    store: SharedEmbeddingStore,
}

impl EmbeddingStorePushSink {
    /// Create a new sink that writes into the provided store.
    pub fn new(store: SharedEmbeddingStore) -> Self {
        Self { store }
    }

    fn apply_delta_table(&self, t: &DeltaEmbeddingHashTable) -> ServingResult<usize> {
        let unique_id = t
            .unique_id
            .as_deref()
            .ok_or_else(|| ServingError::server("missing unique_id".to_string()))?;
        let dim = t
            .dim_size
            .map(|d| d as usize)
            .ok_or_else(|| ServingError::server("missing dim_size".to_string()))?;

        Ok(self
            .store
            .apply_delta(unique_id, dim, &t.fids, &t.embeddings))
    }
}

impl PushSink for EmbeddingStorePushSink {
    fn handle_push(&self, req: PushRequest) -> ServingResult<PushResponse> {
        let mut updated = 0usize;

        for t in &req.delta_hash_tables {
            updated += self.apply_delta_table(t)?;
        }
        for t in &req.delta_multi_hash_tables {
            updated += self.apply_delta_table(t)?;
        }

        Ok(PushResponse {
            target: None,
            status_code: Some(0),
            error_message: None,
            update_num: Some(updated as i32),
        })
    }
}

/// Shared sink handle.
pub type SharedEmbeddingStorePushSink = Arc<EmbeddingStorePushSink>;
