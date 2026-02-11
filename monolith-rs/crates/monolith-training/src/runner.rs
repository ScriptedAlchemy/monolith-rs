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

impl DistributedRunConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if self.num_ps == 0 {
            anyhow::bail!("distributed config requires num_ps > 0");
        }
        if self.num_workers == 0 {
            anyhow::bail!("distributed config requires num_workers > 0");
        }
        if self.dim == 0 {
            anyhow::bail!("distributed config requires dim > 0");
        }
        if self.barrier_timeout_ms <= 0 {
            anyhow::bail!("distributed config requires barrier_timeout_ms > 0");
        }
        Ok(())
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
        discovery_service_type_ps: runner_conf.discovery_service_type_ps.clone(),
        discovery_service_type_worker: runner_conf.discovery_service_type_worker.clone(),
        table_name: runner_conf.table_name.clone(),
        dim: runner_conf.dim,
        connect_retries: runner_conf.connect_retries,
        retry_backoff_ms: runner_conf.retry_backoff_ms,
        barrier_timeout_ms: runner_conf.barrier_timeout_ms,
        ..DistributedRunConfig::default()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PsAddrOrderError {
    MissingOrGappedIndexSet,
    ConflictingDuplicateIndex,
    MixedIndexMetadataPresence,
    InvalidIndexMetadata,
}

fn ordered_ps_addrs(
    ps_services: Vec<ServiceInfo>,
    required_num_ps: usize,
) -> Result<Vec<String>, PsAddrOrderError> {
    let mut saw_index_metadata = false;
    let mut saw_missing_index_metadata = false;
    let mut saw_invalid_index_metadata = false;

    let mut parsed = Vec::with_capacity(ps_services.len());
    for svc in &ps_services {
        match svc.metadata.get("index") {
            Some(idx_str) => match idx_str.parse::<usize>() {
                Ok(idx) => {
                    saw_index_metadata = true;
                    parsed.push((idx, svc.address()));
                }
                Err(_) => {
                    saw_index_metadata = true;
                    saw_invalid_index_metadata = true;
                }
            },
            None => {
                saw_missing_index_metadata = true;
            }
        }
    }

    if !saw_index_metadata {
        let mut addrs: Vec<String> = ps_services.into_iter().map(|s| s.address()).collect();
        addrs.sort();
        addrs.dedup();
        return Ok(addrs);
    }

    if saw_missing_index_metadata || saw_invalid_index_metadata {
        // Mixed/invalid metadata is treated as inconsistent discovery state:
        // force caller retry instead of silently switching to address ordering.
        return if saw_invalid_index_metadata {
            Err(PsAddrOrderError::InvalidIndexMetadata)
        } else {
            Err(PsAddrOrderError::MixedIndexMetadataPresence)
        };
    }

    let mut indexed: std::collections::BTreeMap<usize, String> = std::collections::BTreeMap::new();
    let mut conflicting_index = false;
    for (idx, addr) in parsed {
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
        return Err(PsAddrOrderError::ConflictingDuplicateIndex);
    }

    let mut ordered = Vec::with_capacity(required_num_ps);
    for idx in 0..required_num_ps {
        let Some(addr) = indexed.get(&idx) else {
            // All services reported index metadata, but contiguous shard set
            // is incomplete. Force caller to retry discovery instead of
            // connecting with a gap (e.g. 0,2 without 1).
            return Err(PsAddrOrderError::MissingOrGappedIndexSet);
        };
        ordered.push(addr.clone());
    }
    Ok(ordered)
}

/// Run a PS or worker process using the provided discovery backend.
pub async fn run_distributed<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    cfg: DistributedRunConfig,
) -> anyhow::Result<()> {
    cfg.validate()?;
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
            match discovery.register_async(service).await {
                Ok(()) => run_worker_role(Arc::clone(&discovery), &service_id, cfg).await,
                Err(e) => Err(anyhow::Error::from(e)),
            }
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
    let mut parameter_sync_task = if !cfg.parameter_sync_targets.is_empty() {
        let tracker = Arc::new(DirtyTracker::default());
        ps.set_dirty_tracker(Arc::clone(&tracker));

        // The replicator exports dirty FIDs from the PS table and pushes them to online.
        let task = ParameterSyncReplicator::new(
            Arc::clone(&ps),
            tracker,
            cfg.parameter_sync_targets.clone(),
            cfg.parameter_sync_model_name.clone(),
            cfg.parameter_sync_signature_name.clone(),
            cfg.table_name.clone(),
        )
        .spawn(cfg.parameter_sync_interval);
        Some(task)
    } else {
        None
    };

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
    if let Some(task) = parameter_sync_task.take() {
        task.stop().await;
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
    let mut last_ordering_issue: Option<PsAddrOrderError> = None;
    let mut last_discovery_error: Option<String> = None;
    let mut max_raw_ps_observed: usize = 0;
    let mut max_usable_ps_observed: usize = 0;
    for attempt in 0..=cfg.connect_retries {
        let ps_services = match discovery.discover_async(&cfg.discovery_service_type_ps).await {
            Ok(services) => {
                if last_discovery_error.is_some() {
                    last_discovery_error = None;
                }
                services
            }
            Err(e) => {
                last_discovery_error = Some(e.to_string());
                Vec::new()
            }
        };
        max_raw_ps_observed = max_raw_ps_observed.max(ps_services.len());

        let (addrs, ordering_issue) = match ordered_ps_addrs(ps_services, cfg.num_ps) {
            Ok(addrs) => (addrs, None),
            Err(issue) => (Vec::new(), Some(issue)),
        };
        max_usable_ps_observed = max_usable_ps_observed.max(addrs.len());
        if let Some(issue) = ordering_issue {
            last_ordering_issue = Some(issue);
        }

        if addrs.len() >= cfg.num_ps {
            ps_addrs = addrs;
            break;
        }

        if attempt == cfg.connect_retries {
            let attempts_made = attempt + 1;
            match (last_ordering_issue, last_discovery_error.as_deref()) {
                (Some(issue), Some(discovery_error)) => {
                    anyhow::bail!(
                        "Timed out waiting for PS discovery: got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {}; last ordering issue: {:?}; last discovery error: {})",
                        addrs.len(),
                        cfg.num_ps,
                        attempts_made,
                        max_raw_ps_observed,
                        max_usable_ps_observed,
                        issue,
                        discovery_error
                    );
                }
                (Some(issue), None) => {
                    anyhow::bail!(
                        "Timed out waiting for PS discovery: got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {}; last ordering issue: {:?})",
                        addrs.len(),
                        cfg.num_ps,
                        attempts_made,
                        max_raw_ps_observed,
                        max_usable_ps_observed,
                        issue
                    );
                }
                (None, Some(discovery_error)) => {
                    anyhow::bail!(
                        "Timed out waiting for PS discovery: got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {}; last discovery error: {})",
                        addrs.len(),
                        cfg.num_ps,
                        attempts_made,
                        max_raw_ps_observed,
                        max_usable_ps_observed,
                        discovery_error
                    );
                }
                (None, None) => {
                    anyhow::bail!(
                        "Timed out waiting for PS discovery: got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {})",
                        addrs.len(),
                        cfg.num_ps,
                        attempts_made,
                        max_raw_ps_observed,
                        max_usable_ps_observed
                    );
                }
            }
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

    struct FailingRegisterDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl FailingRegisterDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }

        fn deregister_count(&self) -> usize {
            self.deregister_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for FailingRegisterDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Err(crate::discovery::DiscoveryError::Internal(
                "forced register failure".to_string(),
            ))
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            Ok(Vec::new())
        }

        async fn watch_async(
            &self,
            _service_type: &str,
        ) -> DiscoveryResult<tokio::sync::broadcast::Receiver<DiscoveryEvent>> {
            let (_tx, rx) = tokio::sync::broadcast::channel(1);
            Ok(rx)
        }

        async fn deregister_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::NotFound(
                "missing".to_string(),
            ))
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
            discovery_service_type_ps: "parameter_server".to_string(),
            discovery_service_type_worker: "trainer".to_string(),
            table_name: "item_emb".to_string(),
            dim: 128,
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
        assert_eq!(cfg.discovery_service_type_ps, "parameter_server");
        assert_eq!(cfg.discovery_service_type_worker, "trainer");
        assert_eq!(cfg.table_name, "item_emb");
        assert_eq!(cfg.dim, 128);
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

        let ordered = ordered_ps_addrs(vec![ps1, ps0], 2).unwrap();
        assert_eq!(
            ordered,
            vec!["127.0.0.1:20001".to_string(), "127.0.0.1:10001".to_string()]
        );
    }

    struct SequencedDiscoverDiscovery {
        calls: AtomicUsize,
    }

    impl SequencedDiscoverDiscovery {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    struct SequencedDiscoverErrorDiscovery {
        calls: AtomicUsize,
    }

    impl SequencedDiscoverErrorDiscovery {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for SequencedDiscoverErrorDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            let _ = self.calls.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced discover failure".to_string(),
            ))
        }

        async fn watch_async(
            &self,
            _service_type: &str,
        ) -> DiscoveryResult<tokio::sync::broadcast::Receiver<DiscoveryEvent>> {
            let (_tx, rx) = tokio::sync::broadcast::channel(1);
            Ok(rx)
        }

        async fn deregister_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            Ok(())
        }
    }

    struct SequencedOrderAndErrorDiscovery {
        calls: AtomicUsize,
    }

    impl SequencedOrderAndErrorDiscovery {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for SequencedOrderAndErrorDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                let mut ps0 = ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 10001);
                ps0 = ps0.with_metadata("index", "0");
                let ps1 = ServiceInfo::new("ps-1", "ps-1", "ps", "127.0.0.1", 10002);
                Ok(vec![ps0, ps1])
            } else {
                Err(crate::discovery::DiscoveryError::Internal(
                    "forced discover failure".to_string(),
                ))
            }
        }

        async fn watch_async(
            &self,
            _service_type: &str,
        ) -> DiscoveryResult<tokio::sync::broadcast::Receiver<DiscoveryEvent>> {
            let (_tx, rx) = tokio::sync::broadcast::channel(1);
            Ok(rx)
        }

        async fn deregister_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            Ok(())
        }
    }

    struct SequencedPartialPsDiscovery {
        calls: AtomicUsize,
    }

    impl SequencedPartialPsDiscovery {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    struct SequencedDiscoverErrorThenPartialDiscovery {
        calls: AtomicUsize,
    }

    impl SequencedDiscoverErrorThenPartialDiscovery {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for SequencedDiscoverErrorThenPartialDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Err(crate::discovery::DiscoveryError::Internal(
                    "forced discover failure".to_string(),
                ))
            } else {
                Ok(vec![ServiceInfo::new(
                    "ps-0",
                    "ps-0",
                    "ps",
                    "127.0.0.1",
                    10001,
                )])
            }
        }

        async fn watch_async(
            &self,
            _service_type: &str,
        ) -> DiscoveryResult<tokio::sync::broadcast::Receiver<DiscoveryEvent>> {
            let (_tx, rx) = tokio::sync::broadcast::channel(1);
            Ok(rx)
        }

        async fn deregister_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for SequencedPartialPsDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Ok(vec![ServiceInfo::new(
                    "ps-0",
                    "ps-0",
                    "ps",
                    "127.0.0.1",
                    10001,
                )])
            } else {
                Ok(Vec::new())
            }
        }

        async fn watch_async(
            &self,
            _service_type: &str,
        ) -> DiscoveryResult<tokio::sync::broadcast::Receiver<DiscoveryEvent>> {
            let (_tx, rx) = tokio::sync::broadcast::channel(1);
            Ok(rx)
        }

        async fn deregister_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for SequencedDiscoverDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                let mut ps0 = ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 10001);
                ps0 = ps0.with_metadata("index", "0");
                let ps1 = ServiceInfo::new("ps-1", "ps-1", "ps", "127.0.0.1", 10002);
                Ok(vec![ps0, ps1])
            } else {
                Ok(Vec::new())
            }
        }

        async fn watch_async(
            &self,
            _service_type: &str,
        ) -> DiscoveryResult<tokio::sync::broadcast::Receiver<DiscoveryEvent>> {
            let (_tx, rx) = tokio::sync::broadcast::channel(1);
            Ok(rx)
        }

        async fn deregister_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_ordered_ps_addrs_falls_back_to_address_sort_without_index() {
        let ps_b = ServiceInfo::new("ps-b", "ps-b", "ps", "127.0.0.1", 30001);
        let ps_a = ServiceInfo::new("ps-a", "ps-a", "ps", "127.0.0.1", 20001);

        let ordered = ordered_ps_addrs(vec![ps_b, ps_a], 2).unwrap();
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
        assert_eq!(ordered, Err(PsAddrOrderError::MissingOrGappedIndexSet));
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
        assert_eq!(ordered, Err(PsAddrOrderError::ConflictingDuplicateIndex));
    }

    #[test]
    fn test_ordered_ps_addrs_rejects_mixed_index_metadata_presence() {
        let mut ps0 = ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 10001);
        ps0 = ps0.with_metadata("index", "0");
        let ps1 = ServiceInfo::new("ps-1", "ps-1", "ps", "127.0.0.1", 20001);

        let ordered = ordered_ps_addrs(vec![ps0, ps1], 2);
        assert_eq!(ordered, Err(PsAddrOrderError::MixedIndexMetadataPresence));
    }

    #[test]
    fn test_ordered_ps_addrs_rejects_invalid_index_metadata() {
        let mut ps0 = ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 10001);
        ps0 = ps0.with_metadata("index", "not-an-int");
        let mut ps1 = ServiceInfo::new("ps-1", "ps-1", "ps", "127.0.0.1", 20001);
        ps1 = ps1.with_metadata("index", "1");

        let ordered = ordered_ps_addrs(vec![ps0, ps1], 2);
        assert_eq!(ordered, Err(PsAddrOrderError::InvalidIndexMetadata));
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
    async fn test_run_distributed_rejects_invalid_runtime_config() {
        let discovery = Arc::new(InMemoryDiscovery::new());
        let bad_cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 0,
            ..DistributedRunConfig::default()
        };
        let err = run_distributed(discovery, bad_cfg).await.unwrap_err();
        assert!(err.to_string().contains("num_ps > 0"));
    }

    #[tokio::test]
    async fn test_run_worker_role_timeout_reports_ordering_issue() {
        let discovery = Arc::new(InMemoryDiscovery::new());
        discovery.connect().await.unwrap();

        let mut ps0 = ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 10001);
        ps0 = ps0.with_metadata("index", "0");
        let ps1 = ServiceInfo::new("ps-1", "ps-1", "ps", "127.0.0.1", 10002);
        discovery.register_async(ps0).await.unwrap();
        discovery.register_async(ps1).await.unwrap();

        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 2,
            num_workers: 1,
            index: 0,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("MixedIndexMetadataPresence"));
    }

    #[tokio::test]
    async fn test_run_worker_role_preserves_last_ordering_issue_across_retries() {
        let discovery = Arc::new(SequencedDiscoverDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 2,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("MixedIndexMetadataPresence"),
            "expected timeout to preserve last ordering issue, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_run_worker_role_preserves_last_discovery_error_across_retries() {
        let discovery = Arc::new(SequencedDiscoverErrorDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 1,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("forced discover failure"),
            "expected timeout to preserve last discovery error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_run_worker_role_timeout_reports_ordering_and_discovery_errors() {
        let discovery = Arc::new(SequencedOrderAndErrorDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 2,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("MixedIndexMetadataPresence"),
            "expected timeout to include ordering issue, got: {msg}"
        );
        assert!(
            msg.contains("forced discover failure"),
            "expected timeout to include discovery error, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_run_worker_role_timeout_reports_max_observed_ps_count() {
        let discovery = Arc::new(SequencedPartialPsDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 2,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max raw observed: 1"),
            "expected timeout to report max observed ps count, got: {msg}"
        );
        assert!(
            msg.contains("max usable observed: 1"),
            "expected timeout to report max usable ps count, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_run_worker_role_reports_raw_vs_usable_observed_counts() {
        let discovery = Arc::new(SequencedDiscoverDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 2,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("max raw observed: 2"),
            "expected raw observed count to include inconsistent endpoints, got: {msg}"
        );
        assert!(
            msg.contains("max usable observed: 0"),
            "expected usable observed count to remain zero under ordering inconsistency, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_run_worker_role_timeout_reports_attempt_count() {
        let discovery = Arc::new(SequencedPartialPsDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 2,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("attempts: 2"),
            "expected timeout to report attempt count, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_run_worker_role_clears_stale_discovery_error_after_successful_discover() {
        let discovery = Arc::new(SequencedDiscoverErrorThenPartialDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 2,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(discovery, "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            !msg.contains("forced discover failure"),
            "stale discovery error should be cleared after successful discover, got: {msg}"
        );
        assert!(
            msg.contains("max raw observed: 1"),
            "expected successful discover attempt to contribute to max observed diagnostics, got: {msg}"
        );
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

    #[tokio::test]
    async fn test_run_distributed_disconnects_when_worker_registration_fails() {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let res = run_distributed(Arc::clone(&discovery), cfg).await;
        assert!(res.is_err(), "expected worker registration failure");
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be called on worker registration failure"
        );
        assert_eq!(
            discovery.deregister_count(),
            1,
            "deregister should still be attempted on worker registration failure"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_disconnects_when_ps_registration_fails() {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let res = run_distributed(Arc::clone(&discovery), cfg).await;
        assert!(res.is_err(), "expected ps registration failure");
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be called on ps registration failure"
        );
        assert_eq!(
            discovery.deregister_count(),
            1,
            "deregister should still be attempted on ps registration failure"
        );
    }
}
