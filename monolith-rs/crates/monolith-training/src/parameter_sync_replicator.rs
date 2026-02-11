//! Training PS -> online ParameterSync replication.
//!
//! Python Monolith's realtime training path periodically pushes embedding deltas
//! to online serving via a ParameterSync client. This module provides a Rust
//! analogue: track which FIDs were updated and periodically push their latest
//! embeddings to `ParameterSyncService`.

use crate::distributed_ps::PsServer;
use monolith_proto::monolith::parameter_sync::push_request::DeltaEmbeddingHashTable;
use monolith_proto::monolith::parameter_sync::PushRequest;
use monolith_serving::parameter_sync_rpc::ParameterSyncRpcClient;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::watch;
use tokio::task::JoinHandle;

#[derive(Debug, Error)]
pub enum ReplicatorError {
    #[error("parameter sync rpc error: {0}")]
    Rpc(String),
    #[error("table not found: {0}")]
    TableNotFound(String),
}

pub type ReplicatorResult<T> = Result<T, ReplicatorError>;

/// Tracks dirty fids for a table on a single PS shard.
#[derive(Debug, Default)]
pub struct DirtyTracker {
    // table_name -> dirty fids
    dirty: RwLock<HashMap<String, HashSet<i64>>>,
}

impl DirtyTracker {
    pub fn mark_dirty(&self, table_name: &str, fids: &[i64]) {
        let mut dirty = self.dirty.write();
        let set = dirty
            .entry(table_name.to_string())
            .or_insert_with(HashSet::new);
        for &fid in fids {
            set.insert(fid);
        }
    }

    pub fn take_dirty(&self, table_name: &str, max: usize) -> Vec<i64> {
        let mut dirty = self.dirty.write();
        let Some(set) = dirty.get_mut(table_name) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for fid in set.iter().copied().take(max) {
            out.push(fid);
        }
        for fid in &out {
            set.remove(fid);
        }
        out
    }
}

/// Background replicator that periodically pushes updated embeddings to online.
pub struct ParameterSyncReplicator {
    ps: Arc<PsServer>,
    tracker: Arc<DirtyTracker>,
    targets: Vec<String>,
    model_name: String,
    signature_name: String,
    table_name: String,
    max_batch_fids: usize,
}

/// Handle for a running background replication task.
///
/// Dropping this handle signals the task to stop on the next loop select-cycle.
pub struct ParameterSyncReplicatorTask {
    stop_tx: watch::Sender<bool>,
    join_handle: Option<JoinHandle<()>>,
}

impl ParameterSyncReplicatorTask {
    /// Signals the replication task to stop and waits for completion.
    pub async fn stop(mut self) {
        let _ = self.stop_tx.send(true);
        if let Some(mut handle) = self.join_handle.take() {
            let _ = (&mut handle).await;
        }
    }
}

impl Drop for ParameterSyncReplicatorTask {
    fn drop(&mut self) {
        // Best-effort safety net for call sites that forget explicit `stop().await`.
        // We signal shutdown and abort the task to avoid lingering detached loops.
        let _ = self.stop_tx.send(true);
        if let Some(handle) = self.join_handle.take() {
            handle.abort();
        }
    }
}

impl ParameterSyncReplicator {
    pub fn new(
        ps: Arc<PsServer>,
        tracker: Arc<DirtyTracker>,
        targets: Vec<String>,
        model_name: String,
        signature_name: String,
        table_name: String,
    ) -> Self {
        Self {
            ps,
            tracker,
            targets,
            model_name,
            signature_name,
            table_name,
            max_batch_fids: 10_000,
        }
    }

    pub fn with_max_batch_fids(mut self, n: usize) -> Self {
        self.max_batch_fids = n;
        self
    }

    /// Spawn a background task that flushes deltas every `interval`.
    pub fn spawn(self, interval: Duration) -> ParameterSyncReplicatorTask {
        let (stop_tx, mut stop_rx) = watch::channel(false);
        let join_handle = tokio::spawn(async move {
            loop {
                tokio::select! {
                    stop_changed = stop_rx.changed() => {
                        if stop_changed.is_err() || *stop_rx.borrow() {
                            break;
                        }
                    }
                    _ = tokio::time::sleep(interval) => {
                        if let Err(e) = self.flush_once().await {
                            tracing::warn!(error = %e, "ParameterSyncReplicator flush failed");
                        }
                    }
                }
            }
        });
        ParameterSyncReplicatorTask {
            stop_tx,
            join_handle: Some(join_handle),
        }
    }

    async fn flush_once(&self) -> ReplicatorResult<()> {
        if self.targets.is_empty() {
            return Ok(());
        }

        let dirty_fids = self
            .tracker
            .take_dirty(&self.table_name, self.max_batch_fids);
        if dirty_fids.is_empty() {
            return Ok(());
        }

        let table = self
            .ps
            .get_or_create_table(&self.table_name, self.ps.default_dim());
        let dim = table.dim();
        let (fids, embeddings) = table.export_embeddings(&dirty_fids);

        if fids.is_empty() {
            return Ok(());
        }

        let delta = DeltaEmbeddingHashTable {
            unique_id: Some(self.table_name.clone()),
            dim_size: Some(dim as i32),
            fids,
            embeddings,
        };

        let req = PushRequest {
            model_name: Some(self.model_name.clone()),
            signature_name: Some(self.signature_name.clone()),
            delta_hash_tables: vec![delta],
            delta_multi_hash_tables: vec![],
            timeout_in_ms: Some(1000),
        };

        // Push to all targets.
        for target in &self.targets {
            let mut client = ParameterSyncRpcClient::connect(target)
                .await
                .map_err(|e| ReplicatorError::Rpc(e.to_string()))?;
            let _ = client
                .push(req.clone())
                .await
                .map_err(|e| ReplicatorError::Rpc(e.to_string()))?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_parameter_sync_replicator_task_stop() {
        let ps = PsServer::new(0, 8);
        let tracker = Arc::new(DirtyTracker::default());
        let task = ParameterSyncReplicator::new(
            ps,
            tracker,
            Vec::new(),
            "m".to_string(),
            "sig".to_string(),
            "emb".to_string(),
        )
        .spawn(Duration::from_millis(20));

        tokio::time::sleep(Duration::from_millis(30)).await;
        let stopped = tokio::time::timeout(Duration::from_secs(1), task.stop()).await;
        assert!(stopped.is_ok(), "replicator task stop should complete quickly");
    }

    #[tokio::test]
    async fn test_parameter_sync_replicator_task_drop_is_safe() {
        let ps = PsServer::new(0, 8);
        let tracker = Arc::new(DirtyTracker::default());
        let task = ParameterSyncReplicator::new(
            ps,
            tracker,
            Vec::new(),
            "m".to_string(),
            "sig".to_string(),
            "emb".to_string(),
        )
        .spawn(Duration::from_millis(20));

        drop(task);
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}
