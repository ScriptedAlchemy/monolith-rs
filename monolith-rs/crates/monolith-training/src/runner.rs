//! Distributed training runner.
//!
//! This is a Rust-native analogue of Python's `distributed_train` flow:
//! - role-based startup (ps vs worker)
//! - discovery-backed registration and cluster formation
//! - retries/backoff for worker connect
//! - optional metrics heartbeat
//!
//! The actual model math is still outside the scope of this runner; the runner
//! focuses on the orchestration and the parity-critical distributed plumbing.

use crate::barrier::{PsBarrier, SharedBarrier};
use crate::discovery::{ServiceDiscoveryAsync, ServiceInfo};
use crate::distributed_ps::{PsClient, PsServer};
use crate::parameter_sync_replicator::{DirtyTracker, ParameterSyncReplicator};
use crate::run_config::RunnerConfig;
use crate::runner_utils::initialize_restore_checkpoint_from_runner_defaults;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

/// Helper for tests: bind to `127.0.0.1:0` and return the chosen address.
async fn bind_ephemeral(
    addr: SocketAddr,
) -> std::io::Result<(tokio::net::TcpListener, SocketAddr)> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    Ok((listener, local))
}

#[derive(Debug, Clone)]
pub enum Role {
    Ps,
    Worker,
}

#[derive(Debug, Clone)]
pub struct DistributedRunConfig {
    pub role: Role,
    pub index: usize,
    pub num_ps: usize,
    pub num_workers: usize,
    pub bind_addr: SocketAddr,
    pub discovery_service_type_ps: String,
    pub discovery_service_type_worker: String,
    pub table_name: String,
    pub dim: usize,
    pub connect_retries: usize,
    pub retry_backoff_ms: u64,
    pub barrier_timeout_ms: i64,
    pub heartbeat_interval: Option<Duration>,
    /// If set, periodically pushes updated embeddings to online serving via ParameterSync.
    pub parameter_sync_targets: Vec<String>,
    pub parameter_sync_interval: Duration,
    pub parameter_sync_model_name: String,
    pub parameter_sync_signature_name: String,
}

impl Default for DistributedRunConfig {
    fn default() -> Self {
        Self {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "ps".to_string(),
            discovery_service_type_worker: "worker".to_string(),
            table_name: "emb".to_string(),
            dim: 64,
            connect_retries: 6,
            retry_backoff_ms: 500,
            barrier_timeout_ms: 10_000,
            heartbeat_interval: Some(Duration::from_secs(10)),
            parameter_sync_targets: Vec::new(),
            parameter_sync_interval: Duration::from_millis(200),
            parameter_sync_model_name: "default".to_string(),
            parameter_sync_signature_name: "serving_default".to_string(),
        }
    }
}

/// Builds a distributed-runner config from a higher-level runner config.
pub fn distributed_config_from_runner(
    runner_conf: &RunnerConfig,
    role: Role,
    bind_addr: SocketAddr,
) -> DistributedRunConfig {
    DistributedRunConfig {
        role,
        index: runner_conf.index,
        num_ps: runner_conf.num_ps.max(1),
        num_workers: runner_conf.num_workers.max(1),
        bind_addr,
        connect_retries: runner_conf.connect_retries,
        retry_backoff_ms: runner_conf.retry_backoff_ms,
        barrier_timeout_ms: runner_conf.barrier_timeout_ms,
        ..DistributedRunConfig::default()
    }
}

fn ordered_ps_addrs(ps_services: Vec<ServiceInfo>, required_num_ps: usize) -> Vec<String> {
    let mut indexed: std::collections::BTreeMap<usize, String> = std::collections::BTreeMap::new();
    let mut conflicting_index = false;
    for svc in &ps_services {
        let Some(idx_str) = svc.metadata.get("index") else {
            let mut addrs: Vec<String> = ps_services.into_iter().map(|s| s.address()).collect();
            addrs.sort();
            addrs.dedup();
            return addrs;
        };
        let Ok(idx) = idx_str.parse::<usize>() else {
            let mut addrs: Vec<String> = ps_services.into_iter().map(|s| s.address()).collect();
            addrs.sort();
            addrs.dedup();
            return addrs;
        };
        let addr = svc.address();
        if let Some(current) = indexed.get(&idx) {
            if current != &addr {
                conflicting_index = true;
                break;
            }
        } else {
            indexed.insert(idx, addr);
        }
    }

    if conflicting_index {
        // Multiple endpoints advertised the same shard index. Treat as
        // inconsistent discovery state and force worker retry.
        return Vec::new();
    }

    let mut ordered = Vec::with_capacity(required_num_ps);
    for idx in 0..required_num_ps {
        let Some(addr) = indexed.get(&idx) else {
            // All services reported index metadata, but contiguous shard set
            // is incomplete. Force caller to retry discovery instead of
            // connecting with a gap (e.g. 0,2 without 1).
            return Vec::new();
        };
        ordered.push(addr.clone());
    }
    ordered
}

/// Run a PS or worker process using the provided discovery backend.
pub async fn run_distributed<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    cfg: DistributedRunConfig,
) -> anyhow::Result<()> {
    discovery.connect().await?;

    let (service_type, service_id) = match cfg.role {
        Role::Ps => (
            cfg.discovery_service_type_ps.clone(),
            format!("ps-{}", cfg.index),
        ),
        Role::Worker => (
            cfg.discovery_service_type_worker.clone(),
            format!("worker-{}", cfg.index),
        ),
    };

    let role_res: anyhow::Result<()> = match cfg.role {
        Role::Ps => run_ps_role(Arc::clone(&discovery), &service_id, service_type, cfg).await,
        Role::Worker => {
            // Register worker address for parity (some backends rely on it for cluster formation).
            let mut service = ServiceInfo::new(
                service_id.clone(),
                service_id.clone(),
                service_type.clone(),
                cfg.bind_addr.ip().to_string(),
                cfg.bind_addr.port(),
            );
            service = service.with_metadata("addr", cfg.bind_addr.to_string());
            service = service.with_metadata("index", cfg.index.to_string());
            discovery.register_async(service).await?;
            run_worker_role(Arc::clone(&discovery), &service_id, cfg).await
        }
    };

    let deregister_result = discovery.deregister_async(&service_id).await;
    let disconnect_result = discovery.disconnect().await;

    if let Err(e) = role_res {
        if let Err(de) = deregister_result {
            tracing::warn!(service_id = %service_id, error = %de, "Failed to deregister service after role error");
        }
        if let Err(de) = disconnect_result {
            tracing::warn!(service_id = %service_id, error = %de, "Failed to disconnect discovery after role error");
        }
        return Err(e);
    }

    deregister_result?;
    disconnect_result?;
    Ok(())
}

/// RunnerConfig-driven distributed entrypoint.
///
/// This applies runner post-init restore/env semantics, then dispatches into
/// the role-based distributed runner.
pub async fn run_distributed_from_runner_config<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    runner_conf: &RunnerConfig,
    role: Role,
    bind_addr: SocketAddr,
) -> anyhow::Result<()> {
    let _ = initialize_restore_checkpoint_from_runner_defaults(runner_conf)?;
    let cfg = distributed_config_from_runner(runner_conf, role, bind_addr);
    run_distributed(discovery, cfg).await
}

/// Run distributed runtime directly from RunConfig by applying merge semantics first.
pub async fn run_distributed_from_run_config<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    run_conf: &crate::run_config::RunConfig,
    base: Option<RunnerConfig>,
    role: Role,
    bind_addr: SocketAddr,
) -> anyhow::Result<()> {
    let runner = run_conf.to_runner_config(base)?;
    run_distributed_from_runner_config(discovery, &runner, role, bind_addr).await
}

async fn run_ps_role<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    service_id: &str,
    service_type: String,
    cfg: DistributedRunConfig,
) -> anyhow::Result<()> {
    // PS: start serving and optionally heartbeat in discovery (backend-specific).
    let ps = PsServer::new(cfg.index as i32, cfg.dim);

    // If parameter sync replication is enabled, wire dirty tracking into the PS RPC path
    // and spawn a background replicator.
    if !cfg.parameter_sync_targets.is_empty() {
        let tracker = Arc::new(DirtyTracker::default());
        ps.set_dirty_tracker(Arc::clone(&tracker));

        // The replicator exports dirty FIDs from the PS table and pushes them to online.
        ParameterSyncReplicator::new(
            Arc::clone(&ps),
            tracker,
            cfg.parameter_sync_targets.clone(),
            cfg.parameter_sync_model_name.clone(),
            cfg.parameter_sync_signature_name.clone(),
            cfg.table_name.clone(),
        )
        .spawn(cfg.parameter_sync_interval);
    }

    // Bind early so we can register the real (possibly ephemeral) port. Python tests always pass
    // explicit ports; in Rust we also want to support `:0` for local tests.
    let (listener, actual_addr) = bind_ephemeral(cfg.bind_addr).await?;

    // Register with the real (possibly ephemeral) address.
    let mut service = ServiceInfo::new(
        service_id.to_string(),
        service_id.to_string(),
        service_type,
        actual_addr.ip().to_string(),
        actual_addr.port(),
    );
    service = service.with_metadata("addr", actual_addr.to_string());
    service = service.with_metadata("index", cfg.index.to_string());
    service = service.with_metadata("allow_update", "true");
    discovery.register_async(service).await?;

    tracing::info!(
        role = "ps",
        index = cfg.index,
        addr = %actual_addr,
        "Starting PS gRPC server"
    );
    let (heartbeat_stop_tx, mut heartbeat_task) = if let Some(interval) = cfg.heartbeat_interval {
        let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
        let discovery = Arc::clone(&discovery);
        let service_id = service_id.to_string();
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    stop_changed = stop_rx.changed() => {
                        if stop_changed.is_err() || *stop_rx.borrow() {
                            break;
                        }
                    }
                    _ = tokio::time::sleep(interval) => {
                        if let Err(e) = discovery.heartbeat_async(&service_id).await {
                            tracing::warn!(
                                service_id = %service_id,
                                error = %e,
                                "Discovery heartbeat failed"
                            );
                        }
                    }
                }
            }
        });
        (Some(stop_tx), Some(task))
    } else {
        (None, None)
    };

    let server_result = tonic::transport::Server::builder()
        .add_service(Arc::clone(&ps).into_service())
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await;

    if let Some(stop_tx) = heartbeat_stop_tx {
        let _ = stop_tx.send(true);
    }
    if let Some(task) = heartbeat_task.take() {
        let _ = task.await;
    }

    server_result?;
    Ok(())
}

async fn run_worker_role<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    _service_id: &str,
    cfg: DistributedRunConfig,
) -> anyhow::Result<()> {
    // Worker: wait until we discover the expected PS set, then connect client.
    tracing::info!(
        role = "worker",
        index = cfg.index,
        "Waiting for PS discovery"
    );

    let mut ps_addrs: Vec<String> = Vec::new();
    for attempt in 0..=cfg.connect_retries {
        let ps_services = discovery
            .discover_async(&cfg.discovery_service_type_ps)
            .await
            .unwrap_or_default();

        let addrs = ordered_ps_addrs(ps_services, cfg.num_ps);

        if addrs.len() >= cfg.num_ps {
            ps_addrs = addrs;
            break;
        }

        if attempt == cfg.connect_retries {
            anyhow::bail!(
                "Timed out waiting for PS discovery: got {} expected {}",
                addrs.len(),
                cfg.num_ps
            );
        }

        tokio::time::sleep(Duration::from_millis(cfg.retry_backoff_ms)).await;
    }

    // For now, take the first num_ps.
    ps_addrs.truncate(cfg.num_ps);
    let ps_addr_refs: Vec<&str> = ps_addrs.iter().map(|s| s.as_str()).collect();
    tracing::info!(role = "worker", index = cfg.index, ps = ?ps_addrs, "Connecting to PS shards");

    let ps_client = PsClient::connect(&ps_addr_refs).await?;
    let barrier: SharedBarrier = Arc::new(PsBarrier::new(ps_client.clone(), cfg.barrier_timeout_ms));

    // Minimal "training loop" skeleton proving that:
    // - lookup works
    // - barrier works
    // - apply_gradients works
    let my_ids = vec![cfg.index as i64, cfg.index as i64, 42];
    let _ = ps_client
        .lookup(&cfg.table_name, &my_ids, cfg.dim, true)
        .await?;

    // Barrier on step 0.
    barrier
        .wait("step0", cfg.index as i32, cfg.num_workers as i32)
        .await?;

    // Apply fake gradients (all ones).
    let grads = vec![1.0f32; my_ids.len() * cfg.dim];
    let _ = ps_client
        .apply_gradients(&cfg.table_name, &my_ids, &grads, cfg.dim, 0.01, 0)
        .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::{
        DiscoveryEvent, InMemoryDiscovery, Result as DiscoveryResult, ServiceDiscovery,
        ServiceDiscoveryAsync, ServiceInfo,
    };
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingDiscovery {
        services: std::sync::Mutex<HashMap<String, ServiceInfo>>,
        heartbeat_count: AtomicUsize,
        events_tx: tokio::sync::broadcast::Sender<DiscoveryEvent>,
    }

    impl CountingDiscovery {
        fn new() -> Self {
            let (events_tx, _) = tokio::sync::broadcast::channel(64);
            Self {
                services: std::sync::Mutex::new(HashMap::new()),
                heartbeat_count: AtomicUsize::new(0),
                events_tx,
            }
        }

        fn heartbeat_count(&self) -> usize {
            self.heartbeat_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for CountingDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn register_async(&self, service: ServiceInfo) -> DiscoveryResult<()> {
            self.services
                .lock()
                .unwrap()
                .insert(service.id.clone(), service.clone());
            let _ = self.events_tx.send(DiscoveryEvent::ServiceAdded(service));
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            let services = self
                .services
                .lock()
                .unwrap()
                .values()
                .filter(|s| s.service_type == service_type)
                .cloned()
                .collect();
            Ok(services)
        }

        async fn watch_async(
            &self,
            _service_type: &str,
        ) -> DiscoveryResult<tokio::sync::broadcast::Receiver<DiscoveryEvent>> {
            Ok(self.events_tx.subscribe())
        }

        async fn deregister_async(&self, service_id: &str) -> DiscoveryResult<()> {
            self.services.lock().unwrap().remove(service_id);
            let _ = self
                .events_tx
                .send(DiscoveryEvent::ServiceRemoved(service_id.to_string()));
            Ok(())
        }

        async fn heartbeat_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            self.heartbeat_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[test]
    fn test_distributed_config_from_runner_maps_fields() {
        let rc = RunnerConfig {
            index: 2,
            num_ps: 3,
            num_workers: 5,
            connect_retries: 11,
            retry_backoff_ms: 77,
            barrier_timeout_ms: 2222,
            ..RunnerConfig::default()
        };
        let cfg = distributed_config_from_runner(
            &rc,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        );
        assert_eq!(cfg.index, 2);
        assert_eq!(cfg.num_ps, 3);
        assert_eq!(cfg.num_workers, 5);
        assert_eq!(cfg.connect_retries, 11);
        assert_eq!(cfg.retry_backoff_ms, 77);
        assert_eq!(cfg.barrier_timeout_ms, 2222);
        assert!(matches!(cfg.role, Role::Worker));
    }

    #[test]
    fn test_ordered_ps_addrs_prefers_discovery_index_metadata() {
        let mut ps1 = ServiceInfo::new("ps-1", "ps-1", "ps", "127.0.0.1", 10001);
        ps1 = ps1.with_metadata("index", "1");
        let mut ps0 = ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 20001);
        ps0 = ps0.with_metadata("index", "0");

        let ordered = ordered_ps_addrs(vec![ps1, ps0], 2);
        assert_eq!(
            ordered,
            vec!["127.0.0.1:20001".to_string(), "127.0.0.1:10001".to_string()]
        );
    }

    #[test]
    fn test_ordered_ps_addrs_falls_back_to_address_sort_without_index() {
        let ps_b = ServiceInfo::new("ps-b", "ps-b", "ps", "127.0.0.1", 30001);
        let ps_a = ServiceInfo::new("ps-a", "ps-a", "ps", "127.0.0.1", 20001);

        let ordered = ordered_ps_addrs(vec![ps_b, ps_a], 2);
        assert_eq!(
            ordered,
            vec!["127.0.0.1:20001".to_string(), "127.0.0.1:30001".to_string()]
        );
    }

    #[test]
    fn test_ordered_ps_addrs_requires_contiguous_index_set() {
        let mut ps0 = ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 10001);
        ps0 = ps0.with_metadata("index", "0");
        let mut ps2 = ServiceInfo::new("ps-2", "ps-2", "ps", "127.0.0.1", 30001);
        ps2 = ps2.with_metadata("index", "2");

        let ordered = ordered_ps_addrs(vec![ps0, ps2], 2);
        assert!(
            ordered.is_empty(),
            "missing shard index should force discovery retry"
        );
    }

    #[test]
    fn test_ordered_ps_addrs_rejects_conflicting_duplicate_index() {
        let mut ps0_a = ServiceInfo::new("ps-0a", "ps-0a", "ps", "127.0.0.1", 10001);
        ps0_a = ps0_a.with_metadata("index", "0");
        let mut ps0_b = ServiceInfo::new("ps-0b", "ps-0b", "ps", "127.0.0.1", 20001);
        ps0_b = ps0_b.with_metadata("index", "0");
        let mut ps1 = ServiceInfo::new("ps-1", "ps-1", "ps", "127.0.0.1", 30001);
        ps1 = ps1.with_metadata("index", "1");

        let ordered = ordered_ps_addrs(vec![ps0_a, ps0_b, ps1], 2);
        assert!(
            ordered.is_empty(),
            "conflicting duplicate index should force discovery retry"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_from_runner_config_smoke() {
        let discovery = Arc::new(InMemoryDiscovery::new());

        let ps_rc = RunnerConfig {
            index: 0,
            num_ps: 1,
            num_workers: 1,
            ..RunnerConfig::default()
        };
        let worker_rc = RunnerConfig {
            index: 0,
            num_ps: 1,
            num_workers: 1,
            ..RunnerConfig::default()
        };

        let discovery_bg = Arc::clone(&discovery);
        let ps_rc_bg = ps_rc.clone();
        let ps_task = tokio::spawn(async move {
            run_distributed_from_runner_config(
                discovery_bg,
                &ps_rc_bg,
                Role::Ps,
                "127.0.0.1:0".parse().unwrap(),
            )
            .await
        });

        tokio::time::sleep(Duration::from_millis(50)).await;
        let worker_res = run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &worker_rc,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        )
        .await;
        assert!(worker_res.is_ok(), "worker failed: {worker_res:?}");
        assert!(
            discovery.discover("worker").unwrap().is_empty(),
            "worker service should be deregistered after completion"
        );

        ps_task.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_from_run_config_smoke() {
        let discovery = Arc::new(InMemoryDiscovery::new());
        let run = crate::run_config::RunConfig {
            is_local: true,
            num_ps: 1,
            num_workers: 1,
            ..crate::run_config::RunConfig::default()
        };
        let worker_res = run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        )
        .await;
        // With no PS role started we expect timeout from worker path.
        assert!(worker_res.is_err());
    }

    #[tokio::test]
    async fn test_ps_heartbeat_task_stops_after_ps_task_abort() {
        let discovery = Arc::new(CountingDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Some(Duration::from_millis(10)),
            ..DistributedRunConfig::default()
        };

        let task = tokio::spawn(run_distributed(Arc::clone(&discovery), cfg));
        tokio::time::sleep(Duration::from_millis(80)).await;
        assert!(
            discovery.heartbeat_count() > 0,
            "heartbeat should run while PS task is alive"
        );

        task.abort();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let after_abort = discovery.heartbeat_count();
        tokio::time::sleep(Duration::from_millis(40)).await;
        let stable = discovery.heartbeat_count();
        assert_eq!(
            stable, after_abort,
            "heartbeat task should stop after PS task cancellation"
        );
    }
}
