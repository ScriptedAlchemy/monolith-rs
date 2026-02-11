//! Barrier abstraction for distributed training.
//!
//! Python Monolith uses TF barrier/allreduce primitives for synchronization.
//! In Rust we provide a small trait so we can swap implementations:
//! - In-process (for tests)
//! - Remote PS-coordinated (via gRPC)
//! - ZK/Consul-backed (future)

use crate::distributed_ps::{PsClient, PsError};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Barrier as TokioBarrier;

#[derive(Debug, Error)]
pub enum BarrierError {
    #[error("Barrier timeout")]
    Timeout,
    #[error("Barrier RPC error: {0}")]
    Rpc(#[from] PsError),
}

pub type BarrierResult<T> = Result<T, BarrierError>;

#[async_trait::async_trait]
pub trait Barrier: Send + Sync {
    async fn wait(&self, barrier_id: &str, worker_id: i32, num_workers: i32) -> BarrierResult<()>;
}

/// In-process barrier (single-process tests).
pub struct InMemoryBarrier {
    barrier: TokioBarrier,
}

impl InMemoryBarrier {
    pub fn new(num_workers: usize) -> Self {
        Self {
            barrier: TokioBarrier::new(num_workers),
        }
    }
}

#[async_trait::async_trait]
impl Barrier for InMemoryBarrier {
    async fn wait(
        &self,
        _barrier_id: &str,
        _worker_id: i32,
        _num_workers: i32,
    ) -> BarrierResult<()> {
        self.barrier.wait().await;
        Ok(())
    }
}

/// Barrier backed by the PS `Barrier` RPC.
pub struct PsBarrier {
    client: tokio::sync::Mutex<PsClient>,
    timeout_ms: i64,
}

impl PsBarrier {
    pub fn new(client: PsClient, timeout_ms: i64) -> Self {
        Self {
            client: tokio::sync::Mutex::new(client),
            timeout_ms,
        }
    }
}

#[async_trait::async_trait]
impl Barrier for PsBarrier {
    async fn wait(&self, barrier_id: &str, worker_id: i32, num_workers: i32) -> BarrierResult<()> {
        let client = self.client.lock().await;
        client
            .barrier(barrier_id, worker_id, num_workers, self.timeout_ms)
            .await?;
        Ok(())
    }
}

/// Convenience alias for shared barrier implementations.
pub type SharedBarrier = Arc<dyn Barrier>;
