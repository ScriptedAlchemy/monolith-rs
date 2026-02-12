//! Barrier abstraction for distributed training.
//!
//! Python Monolith uses TF barrier/allreduce primitives for synchronization.
//! In Rust we provide a small trait so we can swap implementations:
//! - In-process (for tests)
//! - Remote PS-coordinated (via gRPC)
//! - ZK/Consul-backed (future)

use crate::distributed_ps::{PsClient, PsError, PsResult};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::Barrier as TokioBarrier;

#[derive(Debug, Error)]
pub enum BarrierError {
    #[error("Barrier timeout")]
    Timeout,
    #[error("Barrier invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("Barrier RPC error: {0}")]
    Rpc(PsError),
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
    client: PsClient,
    timeout_ms: i64,
}

impl PsBarrier {
    pub fn new(client: PsClient, timeout_ms: i64) -> Self {
        Self { client, timeout_ms }
    }

    /// Connects a PS-backed barrier directly from shard addresses.
    pub async fn connect(addrs: &[&str], timeout_ms: i64) -> PsResult<Self> {
        let client = PsClient::connect(addrs).await?;
        Ok(Self::new(client, timeout_ms))
    }
}

#[async_trait::async_trait]
impl Barrier for PsBarrier {
    async fn wait(&self, barrier_id: &str, worker_id: i32, num_workers: i32) -> BarrierResult<()> {
        match self
            .client
            .barrier(barrier_id, worker_id, num_workers, self.timeout_ms)
            .await
        {
            Ok(()) => Ok(()),
            Err(PsError::Timeout(_)) => Err(BarrierError::Timeout),
            Err(PsError::InvalidConfig(msg)) => Err(BarrierError::InvalidConfig(msg)),
            Err(e) => Err(BarrierError::Rpc(e)),
        }
    }
}

/// Convenience alias for shared barrier implementations.
pub type SharedBarrier = Arc<dyn Barrier>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed_ps::{serve_ps, PsServer};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_in_memory_barrier_waits_for_all_workers() {
        let barrier = Arc::new(InMemoryBarrier::new(2));
        let b0 = barrier.clone();
        let b1 = barrier.clone();

        let (r0, r1) = tokio::join!(
            async move { b0.wait("s0", 0, 2).await },
            async move { b1.wait("s0", 1, 2).await }
        );
        r0.expect("worker 0 should pass in-memory barrier once worker 1 arrives");
        r1.expect("worker 1 should pass in-memory barrier once worker 0 arrives");
    }

    #[tokio::test]
    async fn test_ps_barrier_allows_parallel_waits() {
        let bind = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("ephemeral listener bind should succeed")
            .local_addr()
            .expect("ephemeral listener local_addr lookup should succeed");
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr])
            .await
            .expect("ps client connect should succeed for barrier parallel-wait test");
        let barrier = Arc::new(PsBarrier::new(client, 500));
        let b0 = barrier.clone();
        let b1 = barrier.clone();

        let (r0, r1) = tokio::join!(
            async move { b0.wait("parallel", 0, 2).await },
            async move { b1.wait("parallel", 1, 2).await }
        );
        r0.expect("worker 0 should pass PS-backed barrier in parallel wait");
        r1.expect("worker 1 should pass PS-backed barrier in parallel wait");

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_barrier_maps_timeout_to_barrier_timeout() {
        let bind = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("ephemeral listener bind should succeed")
            .local_addr()
            .expect("ephemeral listener local_addr lookup should succeed");
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr])
            .await
            .expect("ps client connect should succeed for timeout-mapping test");
        let barrier = PsBarrier::new(client, 20);
        let err = barrier
            .wait("timeout_case", 0, 2)
            .await
            .expect_err("barrier wait should map PS timeout to BarrierError::Timeout");
        assert!(matches!(err, BarrierError::Timeout));

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_barrier_maps_invalid_config_error() {
        let bind = std::net::TcpListener::bind("127.0.0.1:0")
            .expect("ephemeral listener bind should succeed")
            .local_addr()
            .expect("ephemeral listener local_addr lookup should succeed");
        let ps = PsServer::new(0, 2);
        let server = tokio::spawn(async move {
            let _ = serve_ps(ps, bind).await;
        });
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;

        let addr = bind.to_string();
        let client = PsClient::connect(&[&addr])
            .await
            .expect("ps client connect should succeed for invalid-config mapping test");
        let barrier = PsBarrier::new(client, 100);
        let err = barrier
            .wait("bad_cfg", -1, 2)
            .await
            .expect_err("barrier wait should map invalid worker id to BarrierError::InvalidConfig");
        assert!(matches!(err, BarrierError::InvalidConfig(_)));

        server.abort();
    }

    #[tokio::test]
    async fn test_ps_barrier_connect_requires_addresses() {
        let res = PsBarrier::connect(&[], 100).await;
        assert!(matches!(res, Err(PsError::InvalidConfig(_))));
    }
}
