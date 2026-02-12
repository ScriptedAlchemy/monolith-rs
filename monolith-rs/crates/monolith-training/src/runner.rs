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
use std::collections::HashSet;
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_DISCOVERY_CLEANUP_TIMEOUT: Duration = Duration::from_millis(200);

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
    pub discovery_operation_timeout: Duration,
    pub discovery_cleanup_timeout: Duration,
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
            discovery_operation_timeout: Duration::from_secs(5),
            discovery_cleanup_timeout: DEFAULT_DISCOVERY_CLEANUP_TIMEOUT,
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
        match self.role {
            Role::Ps if self.index >= self.num_ps => {
                anyhow::bail!(
                    "distributed config requires index < num_ps for ps role (index={} num_ps={})",
                    self.index,
                    self.num_ps
                );
            }
            Role::Worker if self.index >= self.num_workers => {
                anyhow::bail!(
                    "distributed config requires index < num_workers for worker role (index={} num_workers={})",
                    self.index,
                    self.num_workers
                );
            }
            _ => {}
        }
        if self.dim == 0 {
            anyhow::bail!("distributed config requires dim > 0");
        }
        if self.barrier_timeout_ms <= 0 {
            anyhow::bail!("distributed config requires barrier_timeout_ms > 0");
        }
        if self.discovery_operation_timeout.is_zero() {
            anyhow::bail!("distributed config requires discovery_operation_timeout > 0");
        }
        if self.discovery_cleanup_timeout.is_zero() {
            anyhow::bail!("distributed config requires discovery_cleanup_timeout > 0");
        }
        if self.discovery_service_type_ps.trim().is_empty() {
            anyhow::bail!("distributed config requires non-empty discovery_service_type_ps");
        }
        if self.discovery_service_type_worker.trim().is_empty() {
            anyhow::bail!("distributed config requires non-empty discovery_service_type_worker");
        }
        if self.discovery_service_type_ps.trim() != self.discovery_service_type_ps {
            anyhow::bail!(
                "distributed config requires discovery_service_type_ps without leading/trailing whitespace"
            );
        }
        if self
            .discovery_service_type_ps
            .chars()
            .any(char::is_whitespace)
        {
            anyhow::bail!(
                "distributed config requires discovery_service_type_ps without whitespace characters"
            );
        }
        if self.discovery_service_type_worker.trim() != self.discovery_service_type_worker {
            anyhow::bail!(
                "distributed config requires discovery_service_type_worker without leading/trailing whitespace"
            );
        }
        if self
            .discovery_service_type_worker
            .chars()
            .any(char::is_whitespace)
        {
            anyhow::bail!(
                "distributed config requires discovery_service_type_worker without whitespace characters"
            );
        }
        if self
            .discovery_service_type_ps
            .trim()
            .eq_ignore_ascii_case(self.discovery_service_type_worker.trim())
        {
            anyhow::bail!(
                "distributed config requires distinct discovery_service_type_ps and discovery_service_type_worker"
            );
        }
        if self.table_name.trim().is_empty() {
            anyhow::bail!("distributed config requires non-empty table_name");
        }
        if self.table_name.trim() != self.table_name {
            anyhow::bail!(
                "distributed config requires table_name without leading/trailing whitespace"
            );
        }
        if self.table_name.chars().any(char::is_whitespace) {
            anyhow::bail!("distributed config requires table_name without whitespace characters");
        }
        if !self.parameter_sync_targets.is_empty() && self.parameter_sync_interval.is_zero() {
            anyhow::bail!(
                "distributed config requires parameter_sync_interval > 0 when parameter_sync_targets are configured"
            );
        }
        if self
            .parameter_sync_targets
            .iter()
            .any(|target| target.trim().is_empty())
        {
            anyhow::bail!(
                "distributed config requires non-empty parameter_sync_targets entries"
            );
        }
        if self
            .parameter_sync_targets
            .iter()
            .any(|target| target.trim() != target)
        {
            anyhow::bail!(
                "distributed config requires parameter_sync_targets entries without leading/trailing whitespace"
            );
        }
        let mut seen_parameter_sync_targets =
            HashSet::with_capacity(self.parameter_sync_targets.len());
        for target in &self.parameter_sync_targets {
            let has_http_prefix = target
                .get(..7)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("http://"));
            let has_https_prefix = target
                .get(..8)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("https://"));
            if target.contains("://") && !(has_http_prefix || has_https_prefix) {
                anyhow::bail!(
                    "distributed config has invalid parameter_sync_targets entry `{}`: endpoint scheme must be http or https",
                    target
                );
            }
            let endpoint_target = if has_http_prefix || has_https_prefix {
                target.clone()
            } else {
                format!("http://{target}")
            };
            tonic::transport::Endpoint::from_shared(endpoint_target.clone()).map_err(|e| {
                anyhow::anyhow!(
                    "distributed config has invalid parameter_sync_targets entry `{}`: {}",
                    target,
                    e
                )
            })?;
            let parsed_uri: tonic::codegen::http::Uri =
                endpoint_target.parse().map_err(|e| {
                    anyhow::anyhow!(
                        "distributed config has invalid parameter_sync_targets entry `{}`: {}",
                        target,
                        e
                    )
                })?;
            if parsed_uri.query().is_some()
                || (parsed_uri.path() != "/" && !parsed_uri.path().is_empty())
            {
                anyhow::bail!(
                    "distributed config has invalid parameter_sync_targets entry `{}`: endpoint must not include a URL path or query",
                    target
                );
            }
            let authority = parsed_uri.authority().ok_or_else(|| {
                anyhow::anyhow!(
                    "distributed config has invalid parameter_sync_targets entry `{}`: endpoint is missing host:port authority",
                    target
                )
            })?;
            if authority.as_str().contains('@') {
                anyhow::bail!(
                    "distributed config has invalid parameter_sync_targets entry `{}`: endpoint must not include userinfo",
                    target
                );
            }
            let scheme = parsed_uri
                .scheme_str()
                .unwrap_or("http")
                .to_ascii_lowercase();
            if scheme != "http" && scheme != "https" {
                anyhow::bail!(
                    "distributed config has invalid parameter_sync_targets entry `{}`: endpoint scheme must be http or https",
                    target
                );
            }
            let default_port = if scheme == "https" { 443 } else { 80 };
            let canonical_authority = if authority.port_u16().is_some() {
                authority.as_str().to_ascii_lowercase()
            } else {
                format!("{}:{default_port}", authority.as_str().to_ascii_lowercase())
            };
            let canonical_endpoint = format!("{scheme}://{canonical_authority}");
            if !seen_parameter_sync_targets.insert(canonical_endpoint) {
                anyhow::bail!("distributed config requires unique parameter_sync_targets entries");
            }
        }
        if !self.parameter_sync_targets.is_empty() && self.parameter_sync_model_name.trim().is_empty()
        {
            anyhow::bail!(
                "distributed config requires non-empty parameter_sync_model_name when parameter_sync_targets are configured"
            );
        }
        if !self.parameter_sync_targets.is_empty()
            && self.parameter_sync_model_name.trim() != self.parameter_sync_model_name
        {
            anyhow::bail!(
                "distributed config requires parameter_sync_model_name without leading/trailing whitespace when parameter_sync_targets are configured"
            );
        }
        if !self.parameter_sync_targets.is_empty()
            && self
                .parameter_sync_model_name
                .chars()
                .any(char::is_whitespace)
        {
            anyhow::bail!(
                "distributed config requires parameter_sync_model_name without whitespace characters when parameter_sync_targets are configured"
            );
        }
        if !self.parameter_sync_targets.is_empty()
            && self.parameter_sync_signature_name.trim().is_empty()
        {
            anyhow::bail!(
                "distributed config requires non-empty parameter_sync_signature_name when parameter_sync_targets are configured"
            );
        }
        if !self.parameter_sync_targets.is_empty()
            && self.parameter_sync_signature_name.trim() != self.parameter_sync_signature_name
        {
            anyhow::bail!(
                "distributed config requires parameter_sync_signature_name without leading/trailing whitespace when parameter_sync_targets are configured"
            );
        }
        if !self.parameter_sync_targets.is_empty()
            && self
                .parameter_sync_signature_name
                .chars()
                .any(char::is_whitespace)
        {
            anyhow::bail!(
                "distributed config requires parameter_sync_signature_name without whitespace characters when parameter_sync_targets are configured"
            );
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
        num_ps: runner_conf.num_ps,
        num_workers: runner_conf.num_workers,
        bind_addr,
        discovery_service_type_ps: runner_conf.discovery_service_type_ps.clone(),
        discovery_service_type_worker: runner_conf.discovery_service_type_worker.clone(),
        table_name: runner_conf.table_name.clone(),
        dim: runner_conf.dim,
        connect_retries: runner_conf.connect_retries,
        retry_backoff_ms: runner_conf.retry_backoff_ms,
        barrier_timeout_ms: runner_conf.barrier_timeout_ms,
        discovery_operation_timeout: Duration::from_millis(
            runner_conf.discovery_operation_timeout_ms,
        ),
        discovery_cleanup_timeout: Duration::from_millis(runner_conf.discovery_cleanup_timeout_ms),
        parameter_sync_targets: runner_conf.parameter_sync_targets.clone(),
        parameter_sync_interval: Duration::from_millis(runner_conf.parameter_sync_interval_ms),
        parameter_sync_model_name: runner_conf.parameter_sync_model_name.clone(),
        parameter_sync_signature_name: runner_conf.parameter_sync_signature_name.clone(),
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

fn spawn_heartbeat_task<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    service_id: &str,
    interval: Option<Duration>,
) -> (
    Option<tokio::sync::watch::Sender<bool>>,
    Option<tokio::task::JoinHandle<()>>,
) {
    let Some(interval) = interval else {
        return (None, None);
    };

    let (stop_tx, mut stop_rx) = tokio::sync::watch::channel(false);
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
                    tokio::select! {
                        stop_changed = stop_rx.changed() => {
                            if stop_changed.is_err() || *stop_rx.borrow() {
                                break;
                            }
                        }
                        hb_res = discovery.heartbeat_async(&service_id) => {
                            if let Err(e) = hb_res {
                                tracing::warn!(
                                    service_id = %service_id,
                                    error = %e,
                                    "Discovery heartbeat failed"
                                );
                            }
                        }
                    }
                }
            }
        }
    });
    (Some(stop_tx), Some(task))
}

async fn stop_heartbeat_task(
    heartbeat_stop_tx: Option<tokio::sync::watch::Sender<bool>>,
    heartbeat_task: Option<tokio::task::JoinHandle<()>>,
) {
    if let Some(stop_tx) = heartbeat_stop_tx {
        let _ = stop_tx.send(true);
    }
    if let Some(mut task) = heartbeat_task {
        let stop_timeout = Duration::from_millis(100);
        tokio::select! {
            joined = &mut task => {
                let _ = joined;
            }
            _ = tokio::time::sleep(stop_timeout) => {
                task.abort();
                let _ = task.await;
                tracing::warn!(
                    timeout_ms = stop_timeout.as_millis(),
                    "Heartbeat task did not stop promptly; forced abort"
                );
            }
        }
    }
}

async fn await_discovery_cleanup<Fut>(
    op_name: &str,
    timeout: Duration,
    fut: Fut,
) -> anyhow::Result<()>
where
    Fut: Future<Output = crate::discovery::Result<()>>,
{
    match tokio::time::timeout(timeout, fut).await {
        Ok(res) => res.map_err(anyhow::Error::from),
        Err(_) => Err(anyhow::anyhow!(
            "Timed out during discovery cleanup: {} after {}ms",
            op_name,
            timeout.as_millis()
        )),
    }
}

async fn await_discovery_operation<T, Fut>(
    op_name: &str,
    timeout: Duration,
    fut: Fut,
) -> anyhow::Result<T>
where
    Fut: Future<Output = crate::discovery::Result<T>>,
{
    match tokio::time::timeout(timeout, fut).await {
        Ok(res) => res.map_err(anyhow::Error::from),
        Err(_) => Err(anyhow::anyhow!(
            "Timed out during discovery operation: {} after {}ms",
            op_name,
            timeout.as_millis()
        )),
    }
}

/// Run a PS or worker process using the provided discovery backend.
pub async fn run_distributed<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    cfg: DistributedRunConfig,
) -> anyhow::Result<()> {
    cfg.validate()?;
    let operation_timeout = cfg.discovery_operation_timeout;
    let cleanup_timeout = cfg.discovery_cleanup_timeout;
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

    let connect_op = format!("connect {service_id} via {service_type}");
    if let Err(e) = await_discovery_operation(&connect_op, operation_timeout, discovery.connect())
    .await
    {
        let disconnect_op = format!("disconnect {service_id} via {service_type}");
        if let Err(disconnect_err) =
            await_discovery_cleanup(
                &disconnect_op,
                cleanup_timeout,
                discovery.disconnect(),
            )
            .await
        {
            tracing::warn!(
                error = %disconnect_err,
                "Failed to disconnect discovery after connect failure"
            );
            return Err(anyhow::anyhow!(
                "{} (discovery cleanup encountered issues after role error: {}: {})",
                e,
                disconnect_op,
                disconnect_err
            ));
        }
        return Err(anyhow::Error::from(e));
    }

    let role_res: anyhow::Result<()> = match cfg.role {
        Role::Ps => run_ps_role(
            Arc::clone(&discovery),
            &service_id,
            service_type.clone(),
            cfg,
        )
        .await,
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
            let register_op = format!("register {service_id} as {service_type}");
            match await_discovery_operation(
                &register_op,
                operation_timeout,
                discovery.register_async(service),
            )
            .await
            {
                Ok(()) => run_worker_role(Arc::clone(&discovery), &service_id, cfg).await,
                Err(e) => Err(e),
            }
        }
    };

    let deregister_op = format!("deregister {service_id} from {service_type}");
    let deregister_result = await_discovery_cleanup(
        &deregister_op,
        cleanup_timeout,
        discovery.deregister_async(&service_id),
    )
    .await;
    let disconnect_op = format!("disconnect {service_id} via {service_type}");
    let disconnect_result =
        await_discovery_cleanup(&disconnect_op, cleanup_timeout, discovery.disconnect())
            .await;

    if let Err(e) = role_res {
        let mut cleanup_issues = Vec::new();
        if let Err(de) = deregister_result {
            tracing::warn!(service_id = %service_id, error = %de, "Failed to deregister service after role error");
            cleanup_issues.push(format!("{deregister_op}: {de}"));
        }
        if let Err(de) = disconnect_result {
            tracing::warn!(service_id = %service_id, error = %de, "Failed to disconnect discovery after role error");
            cleanup_issues.push(format!("{disconnect_op}: {de}"));
        }
        if !cleanup_issues.is_empty() {
            return Err(anyhow::anyhow!(
                "{} (discovery cleanup encountered issues after role error: {})",
                e,
                cleanup_issues.join("; ")
            ));
        }
        return Err(e);
    }

    if let Err(de) = deregister_result {
        if let Err(disconnect_err) = disconnect_result {
            tracing::warn!(
                service_id = %service_id,
                error = %disconnect_err,
                "Failed to disconnect discovery after successful role with deregister failure"
            );
            return Err(anyhow::anyhow!(
                "{}: {} (discovery cleanup encountered issues after successful role completion: {}: {})",
                deregister_op,
                de,
                disconnect_op,
                disconnect_err
            ));
        }
        return Err(anyhow::anyhow!(
            "{} (discovery cleanup encountered issues after successful role completion: {}: {})",
            de,
            deregister_op,
            de
        ));
    }
    if let Err(disconnect_err) = disconnect_result {
        return Err(anyhow::anyhow!(
            "{} (discovery cleanup encountered issues after successful role completion: {}: {})",
            disconnect_err,
            disconnect_op,
            disconnect_err
        ));
    }
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
        service_type.clone(),
        actual_addr.ip().to_string(),
        actual_addr.port(),
    );
    service = service.with_metadata("addr", actual_addr.to_string());
    service = service.with_metadata("index", cfg.index.to_string());
    service = service.with_metadata("allow_update", "true");
    let register_op = format!("register {service_id} as {service_type}");
    await_discovery_operation(
        &register_op,
        cfg.discovery_operation_timeout,
        discovery.register_async(service),
    )
    .await?;

    tracing::info!(
        role = "ps",
        index = cfg.index,
        addr = %actual_addr,
        "Starting PS gRPC server"
    );
    let (heartbeat_stop_tx, heartbeat_task) =
        spawn_heartbeat_task(Arc::clone(&discovery), service_id, cfg.heartbeat_interval);

    let server_result = tonic::transport::Server::builder()
        .add_service(Arc::clone(&ps).into_service())
        .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
        .await;

    stop_heartbeat_task(heartbeat_stop_tx, heartbeat_task).await;
    if let Some(task) = parameter_sync_task.take() {
        task.stop().await;
    }

    server_result?;
    Ok(())
}

async fn run_worker_role<D: ServiceDiscoveryAsync + 'static + ?Sized>(
    discovery: Arc<D>,
    service_id: &str,
    cfg: DistributedRunConfig,
) -> anyhow::Result<()> {
    // Worker: wait until we discover the expected PS set, then connect client.
    tracing::info!(
        role = "worker",
        index = cfg.index,
        "Waiting for PS discovery"
    );

    let (heartbeat_stop_tx, heartbeat_task) =
        spawn_heartbeat_task(Arc::clone(&discovery), service_id, cfg.heartbeat_interval);

    let result: anyhow::Result<()> = async {
        let mut ps_addrs: Vec<String> = Vec::new();
        let mut last_ordering_issue: Option<PsAddrOrderError> = None;
        let mut last_discovery_error: Option<String> = None;
        let mut max_raw_ps_observed: usize = 0;
        let mut max_usable_ps_observed: usize = 0;
        for attempt in 0..=cfg.connect_retries {
            let discover_op = format!(
                "discover {service_id} for {}",
                cfg.discovery_service_type_ps
            );
            let ps_services = match await_discovery_operation(
                &discover_op,
                cfg.discovery_operation_timeout,
                discovery.discover_async(&cfg.discovery_service_type_ps),
            )
            .await
            {
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
            } else if !addrs.is_empty() {
                // If ordering recovered and we have at least one usable endpoint,
                // clear previously latched ordering inconsistency diagnostics.
                last_ordering_issue = None;
            }

            if addrs.len() >= cfg.num_ps {
                ps_addrs = addrs;
                break;
            }

            if attempt == cfg.connect_retries {
                let attempts_made = attempt + 1;
                match (last_ordering_issue, last_discovery_error.as_deref()) {
                    (Some(issue), Some(discovery_error)) => {
                        return Err(anyhow::anyhow!(
                            "Timed out waiting for PS discovery for {} (service type: {}): got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {}; last ordering issue: {:?}; last discovery error: {})",
                            service_id,
                            cfg.discovery_service_type_ps,
                            addrs.len(),
                            cfg.num_ps,
                            attempts_made,
                            max_raw_ps_observed,
                            max_usable_ps_observed,
                            issue,
                            discovery_error
                        ));
                    }
                    (Some(issue), None) => {
                        return Err(anyhow::anyhow!(
                            "Timed out waiting for PS discovery for {} (service type: {}): got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {}; last ordering issue: {:?})",
                            service_id,
                            cfg.discovery_service_type_ps,
                            addrs.len(),
                            cfg.num_ps,
                            attempts_made,
                            max_raw_ps_observed,
                            max_usable_ps_observed,
                            issue
                        ));
                    }
                    (None, Some(discovery_error)) => {
                        return Err(anyhow::anyhow!(
                            "Timed out waiting for PS discovery for {} (service type: {}): got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {}; last discovery error: {})",
                            service_id,
                            cfg.discovery_service_type_ps,
                            addrs.len(),
                            cfg.num_ps,
                            attempts_made,
                            max_raw_ps_observed,
                            max_usable_ps_observed,
                            discovery_error
                        ));
                    }
                    (None, None) => {
                        return Err(anyhow::anyhow!(
                            "Timed out waiting for PS discovery for {} (service type: {}): got {} expected {} (attempts: {}; max raw observed: {}; max usable observed: {})",
                            service_id,
                            cfg.discovery_service_type_ps,
                            addrs.len(),
                            cfg.num_ps,
                            attempts_made,
                            max_raw_ps_observed,
                            max_usable_ps_observed
                        ));
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
        let barrier: SharedBarrier =
            Arc::new(PsBarrier::new(ps_client.clone(), cfg.barrier_timeout_ms));

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
    .await;

    stop_heartbeat_task(heartbeat_stop_tx, heartbeat_task).await;
    result
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
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
        events_tx: tokio::sync::broadcast::Sender<DiscoveryEvent>,
        discover_delay: Option<Duration>,
    }

    impl CountingDiscovery {
        fn new() -> Self {
            let (events_tx, _) = tokio::sync::broadcast::channel(64);
            Self {
                services: std::sync::Mutex::new(HashMap::new()),
                heartbeat_count: AtomicUsize::new(0),
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
                events_tx,
                discover_delay: None,
            }
        }

        fn with_discover_delay(delay: Duration) -> Self {
            let mut d = Self::new();
            d.discover_delay = Some(delay);
            d
        }

        fn heartbeat_count(&self) -> usize {
            self.heartbeat_count.load(Ordering::SeqCst)
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

    struct FailingRegisterWithHangingCleanupDiscovery {
        connect_count: AtomicUsize,
        register_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl FailingRegisterWithHangingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                register_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn register_count(&self) -> usize {
            self.register_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }

        fn deregister_count(&self) -> usize {
            self.deregister_count.load(Ordering::SeqCst)
        }
    }

    struct FailingConnectDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
    }

    impl FailingConnectDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }
    }

    struct HangingConnectDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
    }

    impl HangingConnectDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }
    }

    struct HangingConnectWithHangingDisconnectDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
    }

    impl HangingConnectWithHangingDisconnectDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }
    }

    struct HangingConnectWithFailingDisconnectDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
    }

    impl HangingConnectWithFailingDisconnectDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }
    }

    struct FailingConnectWithHangingDisconnectDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
    }

    impl FailingConnectWithHangingDisconnectDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }
    }

    struct FailingConnectAndDisconnectDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
    }

    impl FailingConnectAndDisconnectDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
            }
        }

        fn connect_count(&self) -> usize {
            self.connect_count.load(Ordering::SeqCst)
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }
    }

    struct HangingRegisterDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl HangingRegisterDiscovery {
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

    struct HangingRegisterWithHangingCleanupDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl HangingRegisterWithHangingCleanupDiscovery {
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

    struct HangingRegisterWithFailingCleanupDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl HangingRegisterWithFailingCleanupDiscovery {
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

    struct HangingDiscoverDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
        discover_count: AtomicUsize,
    }

    impl HangingDiscoverDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct HangingDiscoverWithHangingCleanupDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
        discover_count: AtomicUsize,
    }

    impl HangingDiscoverWithHangingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct HangingDiscoverWithFailingCleanupDiscovery {
        connect_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
        discover_count: AtomicUsize,
    }

    impl HangingDiscoverWithFailingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct FailingDeregisterAfterSuccessDiscovery {
        ps_addr: String,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl FailingDeregisterAfterSuccessDiscovery {
        fn new(ps_addr: String) -> Self {
            Self {
                ps_addr,
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
            }
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }

        fn deregister_count(&self) -> usize {
            self.deregister_count.load(Ordering::SeqCst)
        }
    }

    struct HangingDeregisterAfterSuccessDiscovery {
        ps_addr: String,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl HangingDeregisterAfterSuccessDiscovery {
        fn new(ps_addr: String) -> Self {
            Self {
                ps_addr,
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
            }
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }

        fn deregister_count(&self) -> usize {
            self.deregister_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerTimeoutWithHangingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl WorkerTimeoutWithHangingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerDiscoverErrorWithHangingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl WorkerDiscoverErrorWithHangingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerDiscoverErrorWithFailingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl WorkerDiscoverErrorWithFailingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerTimeoutWithFailingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl WorkerTimeoutWithFailingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerOrderingIssueWithHangingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl WorkerOrderingIssueWithHangingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerOrderingIssueWithFailingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl WorkerOrderingIssueWithFailingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
        discover_attempt: AtomicUsize,
    }

    impl WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
                discover_attempt: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery {
        connect_count: AtomicUsize,
        discover_count: AtomicUsize,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
        discover_attempt: AtomicUsize,
    }

    impl WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery {
        fn new() -> Self {
            Self {
                connect_count: AtomicUsize::new(0),
                discover_count: AtomicUsize::new(0),
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
                discover_attempt: AtomicUsize::new(0),
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

        fn discover_count(&self) -> usize {
            self.discover_count.load(Ordering::SeqCst)
        }
    }

    struct FailingDisconnectAfterSuccessDiscovery {
        ps_addr: String,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl FailingDisconnectAfterSuccessDiscovery {
        fn new(ps_addr: String) -> Self {
            Self {
                ps_addr,
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
            }
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }

        fn deregister_count(&self) -> usize {
            self.deregister_count.load(Ordering::SeqCst)
        }
    }

    struct HangingDisconnectAfterSuccessDiscovery {
        ps_addr: String,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl HangingDisconnectAfterSuccessDiscovery {
        fn new(ps_addr: String) -> Self {
            Self {
                ps_addr,
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
            }
        }

        fn disconnect_count(&self) -> usize {
            self.disconnect_count.load(Ordering::SeqCst)
        }

        fn deregister_count(&self) -> usize {
            self.deregister_count.load(Ordering::SeqCst)
        }
    }

    struct FailingDeregisterAndDisconnectAfterSuccessDiscovery {
        ps_addr: String,
        disconnect_count: AtomicUsize,
        deregister_count: AtomicUsize,
    }

    impl FailingDeregisterAndDisconnectAfterSuccessDiscovery {
        fn new(ps_addr: String) -> Self {
            Self {
                ps_addr,
                disconnect_count: AtomicUsize::new(0),
                deregister_count: AtomicUsize::new(0),
            }
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
    impl ServiceDiscoveryAsync for FailingRegisterWithHangingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            self.register_count.fetch_add(1, Ordering::SeqCst);
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
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for FailingConnectDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::ConnectionFailed(
                "forced connect failure".to_string(),
            ))
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingConnectDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingConnectWithHangingDisconnectDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingConnectWithFailingDisconnectDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for FailingConnectWithHangingDisconnectDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::ConnectionFailed(
                "forced connect failure".to_string(),
            ))
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for FailingConnectAndDisconnectDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::ConnectionFailed(
                "forced connect failure".to_string(),
            ))
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingRegisterDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingRegisterWithHangingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
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
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingRegisterWithFailingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
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
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerDiscoverErrorWithHangingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerDiscoverErrorWithFailingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerTimeoutWithFailingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
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
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerOrderingIssueWithHangingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            let mut ps0 = ServiceInfo::new("ps-0", "ps-0", service_type, "127.0.0.1", 10001);
            ps0 = ps0.with_metadata("index", "0");
            let ps1 = ServiceInfo::new("ps-1", "ps-1", service_type, "127.0.0.1", 10002);
            Ok(vec![ps0, ps1])
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
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerOrderingIssueWithFailingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            let mut ps0 = ServiceInfo::new("ps-0", "ps-0", service_type, "127.0.0.1", 10001);
            ps0 = ps0.with_metadata("index", "0");
            let ps1 = ServiceInfo::new("ps-1", "ps-1", service_type, "127.0.0.1", 10002);
            Ok(vec![ps0, ps1])
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
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            let attempt = self.discover_attempt.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                let mut ps0 =
                    ServiceInfo::new("ps-0", "ps-0", service_type, "127.0.0.1", 10001);
                ps0 = ps0.with_metadata("index", "0");
                let ps1 = ServiceInfo::new("ps-1", "ps-1", service_type, "127.0.0.1", 10002);
                return Ok(vec![ps0, ps1]);
            }
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            let attempt = self.discover_attempt.fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                let mut ps0 =
                    ServiceInfo::new("ps-0", "ps-0", service_type, "127.0.0.1", 10001);
                ps0 = ps0.with_metadata("index", "0");
                let ps1 = ServiceInfo::new("ps-1", "ps-1", service_type, "127.0.0.1", 10002);
                return Ok(vec![ps0, ps1]);
            }
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingDiscoverDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
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
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingDiscoverWithHangingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
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
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingDiscoverWithFailingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
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
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for FailingDeregisterAfterSuccessDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            if service_type == "ps" {
                let (host, port) = self
                    .ps_addr
                    .split_once(':')
                    .ok_or_else(|| {
                        crate::discovery::DiscoveryError::Internal("invalid ps addr".to_string())
                    })?;
                let port: u16 = port
                    .parse()
                    .map_err(|_| {
                        crate::discovery::DiscoveryError::Internal(
                            "invalid ps port".to_string(),
                        )
                    })?;
                let mut ps = ServiceInfo::new("ps-0", "ps-0", "ps", host, port);
                ps = ps.with_metadata("index", "0");
                Ok(vec![ps])
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingDeregisterAfterSuccessDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            if service_type == "ps" {
                let (host, port) = self
                    .ps_addr
                    .split_once(':')
                    .ok_or_else(|| {
                        crate::discovery::DiscoveryError::Internal("invalid ps addr".to_string())
                    })?;
                let port: u16 = port
                    .parse()
                    .map_err(|_| {
                        crate::discovery::DiscoveryError::Internal(
                            "invalid ps port".to_string(),
                        )
                    })?;
                let mut ps = ServiceInfo::new("ps-0", "ps-0", "ps", host, port);
                ps = ps.with_metadata("index", "0");
                Ok(vec![ps])
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for FailingDisconnectAfterSuccessDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            if service_type == "ps" {
                let (host, port) = self
                    .ps_addr
                    .split_once(':')
                    .ok_or_else(|| {
                        crate::discovery::DiscoveryError::Internal("invalid ps addr".to_string())
                    })?;
                let port: u16 = port
                    .parse()
                    .map_err(|_| {
                        crate::discovery::DiscoveryError::Internal(
                            "invalid ps port".to_string(),
                        )
                    })?;
                let mut ps = ServiceInfo::new("ps-0", "ps-0", "ps", host, port);
                ps = ps.with_metadata("index", "0");
                Ok(vec![ps])
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingDisconnectAfterSuccessDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            if service_type == "ps" {
                let (host, port) = self
                    .ps_addr
                    .split_once(':')
                    .ok_or_else(|| {
                        crate::discovery::DiscoveryError::Internal("invalid ps addr".to_string())
                    })?;
                let port: u16 = port
                    .parse()
                    .map_err(|_| {
                        crate::discovery::DiscoveryError::Internal(
                            "invalid ps port".to_string(),
                        )
                    })?;
                let mut ps = ServiceInfo::new("ps-0", "ps-0", "ps", host, port);
                ps = ps.with_metadata("index", "0");
                Ok(vec![ps])
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for WorkerTimeoutWithHangingCleanupDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, _service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            self.discover_count.fetch_add(1, Ordering::SeqCst);
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
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for FailingDeregisterAndDisconnectAfterSuccessDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ))
        }

        async fn register_async(&self, _service: ServiceInfo) -> DiscoveryResult<()> {
            Ok(())
        }

        async fn discover_async(&self, service_type: &str) -> DiscoveryResult<Vec<ServiceInfo>> {
            if service_type == "ps" {
                let (host, port) = self
                    .ps_addr
                    .split_once(':')
                    .ok_or_else(|| {
                        crate::discovery::DiscoveryError::Internal("invalid ps addr".to_string())
                    })?;
                let port: u16 = port
                    .parse()
                    .map_err(|_| {
                        crate::discovery::DiscoveryError::Internal(
                            "invalid ps port".to_string(),
                        )
                    })?;
                let mut ps = ServiceInfo::new("ps-0", "ps-0", "ps", host, port);
                ps = ps.with_metadata("index", "0");
                Ok(vec![ps])
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
            Err(crate::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ))
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for CountingDiscovery {
        async fn connect(&self) -> DiscoveryResult<()> {
            self.connect_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        async fn disconnect(&self) -> DiscoveryResult<()> {
            self.disconnect_count.fetch_add(1, Ordering::SeqCst);
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
            if let Some(delay) = self.discover_delay {
                tokio::time::sleep(delay).await;
            }
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
            self.deregister_count.fetch_add(1, Ordering::SeqCst);
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
            discovery_operation_timeout_ms: 4321,
            discovery_cleanup_timeout_ms: 123,
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_interval_ms: 345,
            parameter_sync_model_name: "my_model".to_string(),
            parameter_sync_signature_name: "my_signature".to_string(),
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
        assert_eq!(cfg.discovery_operation_timeout, Duration::from_millis(4321));
        assert_eq!(cfg.discovery_cleanup_timeout, Duration::from_millis(123));
        assert_eq!(cfg.parameter_sync_targets, vec!["127.0.0.1:8500".to_string()]);
        assert_eq!(cfg.parameter_sync_interval, Duration::from_millis(345));
        assert_eq!(cfg.parameter_sync_model_name, "my_model");
        assert_eq!(cfg.parameter_sync_signature_name, "my_signature");
        assert!(matches!(cfg.role, Role::Worker));
    }

    #[test]
    fn test_distributed_config_validate_rejects_zero_discovery_operation_timeout() {
        let cfg = DistributedRunConfig {
            discovery_operation_timeout: Duration::from_millis(0),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires discovery_operation_timeout > 0"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_zero_num_ps() {
        let cfg = DistributedRunConfig {
            num_ps: 0,
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires num_ps > 0"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_zero_num_workers() {
        let cfg = DistributedRunConfig {
            num_workers: 0,
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires num_workers > 0"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_zero_discovery_cleanup_timeout() {
        let cfg = DistributedRunConfig {
            discovery_cleanup_timeout: Duration::from_millis(0),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires discovery_cleanup_timeout > 0"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_ps_index_out_of_range() {
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 2,
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires index < num_ps for ps role"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_worker_index_out_of_range() {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_workers: 3,
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires index < num_workers for worker role"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_empty_ps_service_type() {
        let cfg = DistributedRunConfig {
            discovery_service_type_ps: "   ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires non-empty discovery_service_type_ps"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_whitespace_padded_ps_service_type() {
        let cfg = DistributedRunConfig {
            discovery_service_type_ps: " ps ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires discovery_service_type_ps without leading/trailing whitespace"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_internal_whitespace_ps_service_type() {
        let cfg = DistributedRunConfig {
            discovery_service_type_ps: "ps cluster".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires discovery_service_type_ps without whitespace characters"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_empty_worker_service_type() {
        let cfg = DistributedRunConfig {
            discovery_service_type_worker: "".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires non-empty discovery_service_type_worker"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_whitespace_padded_worker_service_type() {
        let cfg = DistributedRunConfig {
            discovery_service_type_worker: " worker ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires discovery_service_type_worker without leading/trailing whitespace"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_internal_whitespace_worker_service_type() {
        let cfg = DistributedRunConfig {
            discovery_service_type_worker: "worker cluster".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires discovery_service_type_worker without whitespace characters"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_identical_ps_and_worker_service_types() {
        let cfg = DistributedRunConfig {
            discovery_service_type_ps: "service".to_string(),
            discovery_service_type_worker: "service".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires distinct discovery_service_type_ps and discovery_service_type_worker"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_case_insensitive_identical_ps_and_worker_service_types(
    ) {
        let cfg = DistributedRunConfig {
            discovery_service_type_ps: "Service".to_string(),
            discovery_service_type_worker: "service".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires distinct discovery_service_type_ps and discovery_service_type_worker"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_empty_table_name() {
        let cfg = DistributedRunConfig {
            table_name: "  ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires non-empty table_name"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_whitespace_padded_table_name() {
        let cfg = DistributedRunConfig {
            table_name: " emb ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires table_name without leading/trailing whitespace"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_internal_whitespace_table_name() {
        let cfg = DistributedRunConfig {
            table_name: "my table".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires table_name without whitespace characters"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_zero_parameter_sync_interval_when_targets_configured(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_interval: Duration::from_millis(0),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires parameter_sync_interval > 0 when parameter_sync_targets are configured"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_empty_parameter_sync_target_entry() {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["".to_string()],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires non-empty parameter_sync_targets entries"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_whitespace_padded_parameter_sync_target_entry() {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec![" 127.0.0.1:8500 ".to_string()],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires parameter_sync_targets entries without leading/trailing whitespace"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_invalid_parameter_sync_target_endpoint() {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["http://".to_string()],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config has invalid parameter_sync_targets entry `http://`"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_parameter_sync_target_endpoint_with_path_or_query(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["http://127.0.0.1:8500/v1?foo=bar".to_string()],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("endpoint must not include a URL path or query"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_parameter_sync_target_endpoint_with_unsupported_scheme(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["ftp://127.0.0.1:8500".to_string()],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("endpoint scheme must be http or https"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_parameter_sync_target_endpoint_with_userinfo() {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["http://user@127.0.0.1:8500".to_string()],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("endpoint must not include userinfo"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_accepts_case_insensitive_http_scheme_parameter_sync_target(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["HtTp://127.0.0.1:8500".to_string()],
            ..DistributedRunConfig::default()
        };
        cfg.validate().expect(
            "case-insensitive http scheme should be accepted for parameter_sync_targets endpoint parsing",
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_duplicate_parameter_sync_target_entries() {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec![
                "127.0.0.1:8500".to_string(),
                "127.0.0.1:8500".to_string(),
            ],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires unique parameter_sync_targets entries"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_duplicate_parameter_sync_target_entries_after_http_prefix_normalization(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec![
                "127.0.0.1:8500".to_string(),
                "http://127.0.0.1:8500".to_string(),
            ],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires unique parameter_sync_targets entries"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_duplicate_parameter_sync_target_entries_after_trailing_slash_normalization(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec![
                "127.0.0.1:8500".to_string(),
                "http://127.0.0.1:8500/".to_string(),
            ],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires unique parameter_sync_targets entries"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_duplicate_parameter_sync_target_entries_after_http_default_port_normalization(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec![
                "127.0.0.1".to_string(),
                "http://127.0.0.1:80".to_string(),
            ],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires unique parameter_sync_targets entries"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_duplicate_parameter_sync_target_entries_after_https_default_port_normalization(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec![
                "https://127.0.0.1".to_string(),
                "https://127.0.0.1:443".to_string(),
            ],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires unique parameter_sync_targets entries"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_duplicate_parameter_sync_target_entries_after_case_insensitive_http_prefix_and_host_normalization(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec![
                "EXAMPLE.com:8500".to_string(),
                "HtTp://example.COM:8500".to_string(),
            ],
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains("distributed config requires unique parameter_sync_targets entries"),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_empty_parameter_sync_model_name_when_targets_configured(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_model_name: " ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires non-empty parameter_sync_model_name when parameter_sync_targets are configured"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_whitespace_padded_parameter_sync_model_name_when_targets_configured(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_model_name: " model ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires parameter_sync_model_name without leading/trailing whitespace when parameter_sync_targets are configured"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_internal_whitespace_parameter_sync_model_name_when_targets_configured(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_model_name: "my model".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires parameter_sync_model_name without whitespace characters when parameter_sync_targets are configured"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_empty_parameter_sync_signature_when_targets_configured(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_signature_name: "".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires non-empty parameter_sync_signature_name when parameter_sync_targets are configured"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_whitespace_padded_parameter_sync_signature_when_targets_configured(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_signature_name: " signature ".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires parameter_sync_signature_name without leading/trailing whitespace when parameter_sync_targets are configured"
            ),
            "unexpected validation error: {err}"
        );
    }

    #[test]
    fn test_distributed_config_validate_rejects_internal_whitespace_parameter_sync_signature_when_targets_configured(
    ) {
        let cfg = DistributedRunConfig {
            parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
            parameter_sync_signature_name: "serving default".to_string(),
            ..DistributedRunConfig::default()
        };
        let err = cfg.validate().unwrap_err().to_string();
        assert!(
            err.contains(
                "distributed config requires parameter_sync_signature_name without whitespace characters when parameter_sync_targets are configured"
            ),
            "unexpected validation error: {err}"
        );
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

    struct HangingHeartbeatDiscovery;

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for HangingHeartbeatDiscovery {
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
            tokio::time::sleep(Duration::from_millis(30)).await;
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
            Ok(())
        }

        async fn heartbeat_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
        }
    }

    struct BlockingHeartbeatActiveCountDiscovery {
        active_heartbeats: AtomicUsize,
    }

    impl BlockingHeartbeatActiveCountDiscovery {
        fn new() -> Self {
            Self {
                active_heartbeats: AtomicUsize::new(0),
            }
        }

        fn active_heartbeats(&self) -> usize {
            self.active_heartbeats.load(Ordering::SeqCst)
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for BlockingHeartbeatActiveCountDiscovery {
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
            Ok(())
        }

        async fn heartbeat_async(&self, _service_id: &str) -> DiscoveryResult<()> {
            self.active_heartbeats.fetch_add(1, Ordering::SeqCst);
            struct ActiveGuard<'a>(&'a AtomicUsize);
            impl<'a> Drop for ActiveGuard<'a> {
                fn drop(&mut self) {
                    self.0.fetch_sub(1, Ordering::SeqCst);
                }
            }
            let _guard = ActiveGuard(&self.active_heartbeats);
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            Ok(())
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

    struct SequencedOrderingIssueThenPartialSuccessDiscovery {
        calls: AtomicUsize,
    }

    impl SequencedOrderingIssueThenPartialSuccessDiscovery {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl ServiceDiscoveryAsync for SequencedOrderingIssueThenPartialSuccessDiscovery {
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
                Ok(vec![ps0, ps1]) // mixed metadata => ordering issue
            } else {
                Ok(vec![ServiceInfo::new(
                    "ps-0",
                    "ps-0",
                    "ps",
                    "127.0.0.1",
                    10001,
                )]) // usable but insufficient count
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
        let err = worker_res.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "run-config distributed worker smoke should surface PS-discovery timeout context: {msg}"
        );
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
    async fn test_run_worker_role_retries_when_discover_operation_times_out() {
        let discovery = Arc::new(HangingDiscoverDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 1,
            num_workers: 1,
            index: 0,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(Arc::clone(&discovery), "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("attempts: 2"),
            "discover operation timeouts should still consume retry budget, got: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Timed out during discovery operation: discover worker-0 for ps"),
            "expected timeout operation context in worker discovery diagnostics, got: {msg}"
        );
        assert_eq!(
            discovery.discover_count(),
            2,
            "worker should retry discovery when discover operation times out"
        );
    }

    #[tokio::test]
    async fn test_run_worker_role_discover_timeout_includes_service_type_context() {
        let discovery = Arc::new(HangingDiscoverDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            num_ps: 1,
            num_workers: 1,
            index: 0,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        let err = run_worker_role(Arc::clone(&discovery), "worker-0", cfg)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-0 for parameter_server_custom after 20ms"
            ),
            "discover timeout should include queried service-type context, got: {msg}"
        );
        assert_eq!(
            discovery.discover_count(),
            1,
            "worker should perform one discover attempt when connect_retries=0"
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
    async fn test_run_worker_role_clears_stale_ordering_issue_after_usable_discovery() {
        let discovery = Arc::new(SequencedOrderingIssueThenPartialSuccessDiscovery::new());
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
            !msg.contains("MixedIndexMetadataPresence"),
            "stale ordering issue should be cleared after usable discovery, got: {msg}"
        );
        assert!(
            msg.contains("max usable observed: 1"),
            "expected usable discovery state to be reflected in diagnostics, got: {msg}"
        );
    }

    #[tokio::test]
    async fn test_worker_heartbeat_task_stops_after_worker_timeout() {
        let discovery = Arc::new(CountingDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 4,
            retry_backoff_ms: 25,
            heartbeat_interval: Some(Duration::from_millis(5)),
            ..DistributedRunConfig::default()
        };

        let res = run_worker_role(Arc::clone(&discovery), "worker-0", cfg).await;
        let err = res.expect_err("worker should time out with no PS services");
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should include PS-discovery timeout diagnostics: {msg}"
        );

        let after_timeout = discovery.heartbeat_count();
        assert!(
            after_timeout > 0,
            "worker heartbeat should run while waiting for discovery retries"
        );

        tokio::time::sleep(Duration::from_millis(40)).await;
        let stable = discovery.heartbeat_count();
        assert_eq!(
            stable, after_timeout,
            "worker heartbeat task should stop after worker role exits"
        );
    }

    #[test]
    fn test_worker_heartbeat_task_stops_after_worker_ordering_issue_timeout() {
        test_worker_heartbeat_task_stops_after_worker_timeout();
    }

    #[tokio::test]
    async fn test_worker_heartbeat_task_stops_after_worker_success() {
        let discovery = Arc::new(CountingDiscovery::with_discover_delay(Duration::from_millis(30)));
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let mut ps_service = ServiceInfo::new(
            "ps-0",
            "ps-0",
            "ps",
            actual_addr.ip().to_string(),
            actual_addr.port(),
        );
        ps_service = ps_service.with_metadata("index", "0");
        discovery.register_async(ps_service).await.unwrap();

        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 1,
            retry_backoff_ms: 1,
            heartbeat_interval: Some(Duration::from_millis(5)),
            ..DistributedRunConfig::default()
        };
        let res = run_worker_role(Arc::clone(&discovery), "worker-0", cfg).await;
        assert!(res.is_ok(), "worker should succeed with discoverable PS");

        let after_success = discovery.heartbeat_count();
        assert!(
            after_success > 0,
            "worker heartbeat should run while waiting on discovery before success"
        );

        tokio::time::sleep(Duration::from_millis(40)).await;
        let stable = discovery.heartbeat_count();
        assert_eq!(
            stable, after_success,
            "worker heartbeat task should stop after successful worker completion"
        );

        ps_server.abort();
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
    async fn test_run_worker_role_does_not_hang_when_heartbeat_blocks() {
        let discovery = Arc::new(HangingHeartbeatDiscovery);
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            heartbeat_interval: Some(Duration::from_millis(1)),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(300),
            run_worker_role(discovery, "worker-0", cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "worker role should return even when heartbeat task blocks"
        );
        assert!(
            res.unwrap().is_err(),
            "worker should still fail due to PS discovery timeout"
        );
    }

    #[tokio::test]
    async fn test_ps_abort_cancels_inflight_blocking_heartbeat() {
        let discovery = Arc::new(BlockingHeartbeatActiveCountDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            heartbeat_interval: Some(Duration::from_millis(1)),
            ..DistributedRunConfig::default()
        };

        let ps_task = tokio::spawn(run_distributed(Arc::clone(&discovery), cfg));

        let started = tokio::time::timeout(Duration::from_millis(250), async {
            loop {
                if discovery.active_heartbeats() > 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            started.is_ok(),
            "blocking heartbeat should have started before abort"
        );

        ps_task.abort();

        let stopped = tokio::time::timeout(Duration::from_millis(300), async {
            loop {
                if discovery.active_heartbeats() == 0 {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            stopped.is_ok(),
            "in-flight blocking heartbeat should be cancelled after PS abort"
        );
    }

    #[tokio::test]
    async fn test_stop_heartbeat_task_aborts_nonterminating_task() {
        let (stop_tx, _stop_rx) = tokio::sync::watch::channel(false);
        let stuck_task = tokio::spawn(async {
            std::future::pending::<()>().await;
        });

        let res = tokio::time::timeout(
            Duration::from_millis(300),
            stop_heartbeat_task(Some(stop_tx), Some(stuck_task)),
        )
        .await;
        assert!(
            res.is_ok(),
            "stop_heartbeat_task should return promptly by aborting stuck heartbeat tasks"
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

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected worker registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "worker register failure should remain primary when cleanup also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker register-failure diagnostics should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker") && msg.contains("missing"),
            "worker register-failure cleanup diagnostics should include worker deregister operation/failure context: {msg}"
        );
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
    async fn test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed worker registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed worker register failure should remain primary with default service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed worker register-failure diagnostics should include cleanup issue context with default service type: {msg}"
        );
        assert!(
            msg.contains("deregister worker-3 from worker") && msg.contains("missing"),
            "indexed worker register-failure cleanup diagnostics should include default worker service-type/index deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_registration_failure_with_default_service_type_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg).await.expect_err(
            "expected non-index worker registration failure with default service type",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "worker register failure should remain primary with default service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker register-failure diagnostics should include cleanup issue context with default service type: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker") && msg.contains("missing"),
            "worker register-failure cleanup diagnostics should include default worker service-type non-index deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
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

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected ps registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "ps register failure should remain primary when cleanup also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-failure diagnostics should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("deregister ps-0 from ps") && msg.contains("missing"),
            "ps register-failure cleanup diagnostics should include ps deregister operation/failure context: {msg}"
        );
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

    #[tokio::test]
    async fn test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed ps registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed ps register failure should remain primary with default service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed ps register-failure diagnostics should include cleanup issue context with default service type: {msg}"
        );
        assert!(
            msg.contains("deregister ps-2 from ps") && msg.contains("missing"),
            "indexed ps register-failure cleanup diagnostics should include default ps service-type/index deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_registration_failure_with_default_service_type_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg).await.expect_err(
            "expected non-index ps registration failure with default service type",
        );
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "ps register failure should remain primary with default service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-failure diagnostics should include cleanup issue context with default service type: {msg}"
        );
        assert!(
            msg.contains("deregister ps-0 from ps") && msg.contains("missing"),
            "ps register-failure cleanup diagnostics should include default ps service-type non-index deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_registration_failure_includes_custom_service_type_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected worker registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "worker register failure should remain primary with custom service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker register-failure diagnostics should include cleanup issue context with custom service type: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("missing"),
            "worker register-failure cleanup diagnostics should include custom worker service-type deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_registration_failure_with_custom_service_type_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected worker registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "worker register failure should remain primary with custom service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker register-failure diagnostics should include cleanup issue context with custom service type: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("missing"),
            "worker register-failure cleanup diagnostics should include custom worker service-type deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed worker registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed worker register failure should remain primary with custom service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed worker register-failure diagnostics should include cleanup issue context with custom service type: {msg}"
        );
        assert!(
            msg.contains("deregister worker-3 from trainer_custom")
                && msg.contains("missing"),
            "indexed worker register-failure cleanup diagnostics should include custom worker service-type/index deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_registration_failure_includes_custom_service_type_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected ps registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "ps register failure should remain primary with custom service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-failure diagnostics should include cleanup issue context with custom service type: {msg}"
        );
        assert!(
            msg.contains("deregister ps-0 from parameter_server_custom")
                && msg.contains("missing"),
            "ps register-failure cleanup diagnostics should include custom ps service-type deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_registration_failure_with_custom_service_type_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected ps registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "ps register failure should remain primary with custom service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-failure diagnostics should include cleanup issue context with custom service type: {msg}"
        );
        assert!(
            msg.contains("deregister ps-0 from parameter_server_custom")
                && msg.contains("missing"),
            "ps register-failure cleanup diagnostics should include custom ps service-type deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_cleanup_context(
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed ps registration failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed ps register failure should remain primary with custom service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed ps register-failure diagnostics should include cleanup issue context with custom service type: {msg}"
        );
        assert!(
            msg.contains("deregister ps-2 from parameter_server_custom")
                && msg.contains("missing"),
            "indexed ps register-failure cleanup diagnostics should include custom ps service-type/index deregister operation/failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_when_cleanup_steps_timeout() {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when worker register fails and cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "worker register failure should remain primary when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker register-failure diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"
            ),
            "worker register-failure cleanup diagnostics should include worker deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"
            ),
            "worker register-failure cleanup diagnostics should include worker disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed worker register fails and default-service cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed worker register failure should remain primary when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed worker register-failure diagnostics should include cleanup issue context when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from worker after 20ms"
            ),
            "indexed worker register-failure cleanup diagnostics should include default-service-type/index deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via worker after 20ms"
            ),
            "indexed worker register-failure cleanup diagnostics should include default-service-type/index disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index worker register fails and default-service cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "worker register failure should remain primary when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker register-failure diagnostics should include cleanup issue context when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"
            ),
            "worker register-failure cleanup diagnostics should include default-service-type non-index deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"
            ),
            "worker register-failure cleanup diagnostics should include default-service-type non-index disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when worker register fails and custom-service cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "worker register failure should remain primary when custom-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker register-failure diagnostics should include cleanup issue context when custom-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "worker register-failure cleanup diagnostics should include custom-service-type deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "worker register-failure cleanup diagnostics should include custom-service-type disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed worker register fails and custom-service cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed worker register failure should remain primary when custom-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed worker register-failure diagnostics should include cleanup issue context when custom-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from trainer_custom after 20ms"
            ),
            "indexed worker register-failure cleanup diagnostics should include custom-service-type deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "indexed worker register-failure cleanup diagnostics should include custom-service-type disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps register fails and cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "ps register failure should remain primary when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-failure diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister ps-0 from parameter_server_custom after 20ms"
            ),
            "ps register-failure cleanup diagnostics should include custom-service-type deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom after 20ms"
            ),
            "ps register-failure cleanup diagnostics should include custom-service-type disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed ps register fails and custom-service cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed ps register failure should remain primary when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed ps register-failure diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister ps-2 from parameter_server_custom after 20ms"
            ),
            "indexed ps register-failure cleanup diagnostics should include custom-service-type/index deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-2 via parameter_server_custom after 20ms"
            ),
            "indexed ps register-failure cleanup diagnostics should include custom-service-type/index disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_when_cleanup_steps_timeout() {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps register fails and cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "ps register failure should remain primary when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-failure diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister ps-0 from ps after 20ms"
            ),
            "ps register-failure cleanup diagnostics should include default-service-type deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via ps after 20ms"
            ),
            "ps register-failure cleanup diagnostics should include default-service-type disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed ps register fails and default-service cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "indexed ps register failure should remain primary when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed ps register-failure diagnostics should include cleanup issue context when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister ps-2 from ps after 20ms"
            ),
            "indexed ps register-failure cleanup diagnostics should include default-service-type/index deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-2 via ps after 20ms"
            ),
            "indexed ps register-failure cleanup diagnostics should include default-service-type/index disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(FailingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index ps register fails and default-service cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced register failure"),
            "ps register failure should remain primary when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-failure diagnostics should include cleanup issue context when default-service cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister ps-0 from ps after 20ms"
            ),
            "ps register-failure cleanup diagnostics should include default-service-type non-index deregister-timeout context: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via ps after 20ms"
            ),
            "ps register-failure cleanup diagnostics should include default-service-type non-index disconnect-timeout context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.register_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_when_cleanup_steps_fail() {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "worker-0", "worker").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "worker-0", "trainer_custom").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "worker-3", "trainer_custom").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "worker-3", "worker").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_register_failure_with_default_service_type_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "worker-0", "worker").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_when_cleanup_steps_fail() {
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "ps-0", "ps").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "ps-0", "parameter_server_custom").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "ps-2", "parameter_server_custom").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "ps-2", "ps").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_ps_register_failure_with_default_service_type_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };
        assert_register_failure_cleanup_fail_case(cfg, "ps-0", "ps").await;
    }

    async fn assert_register_failure_cleanup_fail_case(
        cfg: DistributedRunConfig,
        expected_service_id: &str,
        expected_service_type: &str,
    ) {
        let discovery = Arc::new(FailingRegisterDiscovery::new());
        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("register failure with failing cleanup should surface as a role error");
        let msg = err.to_string();
        assert!(
            msg.contains("forced register failure"),
            "register failure should remain primary over cleanup failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "register-failure diagnostics should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains(&format!(
                "deregister {expected_service_id} from {expected_service_type}"
            )) && msg.contains("missing"),
            "register-failure cleanup diagnostics should include expected deregister-failure context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_disconnects_when_worker_role_fails_after_registration() {
        let discovery = Arc::new(CountingDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected worker role timeout with no discovered ps services");
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "unexpected worker timeout error: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be called on worker role timeout failure"
        );
        assert_eq!(
            discovery.deregister_count(),
            1,
            "deregister should be attempted on worker role timeout failure"
        );
        let workers = discovery.discover_async("worker").await.unwrap();
        assert!(
            workers.is_empty(),
            "worker service should be removed during cleanup after role failure"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_attempts_disconnect_when_connect_fails() {
        let discovery = Arc::new(FailingConnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "connect failure should surface backend connect diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be attempted when connect fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_does_not_hang_and_attempts_disconnect() {
        let discovery = Arc::new(HangingConnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when discovery connect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-0 via worker"),
            "unexpected connect-timeout error: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "connect-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted after connect timeout"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_worker_connect_timeout_includes_custom_service_type_context() {
        let discovery = Arc::new(HangingConnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when worker connect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-0 via trainer_custom"),
            "worker connect-timeout diagnostics should include custom service-type context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "connect-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted after connect timeout"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_includes_custom_service_type_context() {
        let discovery = Arc::new(HangingConnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps connect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect ps-0 via parameter_server_custom"),
            "ps connect-timeout diagnostics should include custom service-type context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "connect-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted after connect timeout"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out()
    {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-0 via worker after 20ms"),
            "connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"
            ),
            "connect-timeout cleanup issue context should include disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails()
    {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-0 via worker after 20ms"),
            "connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "connect-timeout cleanup issue context should include disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-worker connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-2 via worker after 20ms"),
            "indexed default-worker connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-worker connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via worker")
                && msg.contains("forced disconnect failure"),
            "indexed default-worker connect-timeout cleanup issue context should include default-service-type/index disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-worker connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-0 via worker after 20ms"),
            "default-worker connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-worker connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "default-worker connect-timeout cleanup issue context should include default-service-type non-index disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-worker connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-3 via trainer_custom after 20ms"),
            "indexed custom-worker connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-worker connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "indexed custom-worker connect-timeout cleanup issue context should include custom-service-type/index disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index worker connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-0 via trainer_custom after 20ms"),
            "custom non-index worker connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index worker connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "custom non-index worker connect-timeout cleanup issue context should include custom-service-type disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index worker connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: connect worker-0 via trainer_custom after 20ms"
            ),
            "custom non-index worker connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index worker connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "custom non-index worker connect-timeout cleanup issue context should include custom-service-type disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-worker connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-3 via worker after 20ms"),
            "indexed default-worker connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-worker connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via worker after 20ms"
            ),
            "indexed default-worker connect-timeout cleanup issue context should include default-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-worker connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect worker-0 via worker after 20ms"),
            "default-worker connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-worker connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"
            ),
            "default-worker connect-timeout cleanup issue context should include default-service-type non-index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-worker connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: connect worker-3 via trainer_custom after 20ms"
            ),
            "indexed custom-worker connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-worker connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "indexed custom-worker connect-timeout cleanup issue context should include custom-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: connect ps-0 via parameter_server_custom after 20ms"
            ),
            "ps connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom after 20ms"
            ),
            "ps connect-timeout cleanup issue context should include custom-service-type disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-ps connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: connect ps-2 via parameter_server_custom after 20ms"
            ),
            "indexed custom-ps connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-ps connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-2 via parameter_server_custom")
                && msg.contains("forced disconnect failure"),
            "indexed custom-ps connect-timeout cleanup issue context should include custom-service-type/index disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index ps connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: connect ps-0 via parameter_server_custom after 20ms"
            ),
            "custom non-index ps connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index ps connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-0 via parameter_server_custom")
                && msg.contains("forced disconnect failure"),
            "custom non-index ps connect-timeout cleanup issue context should include custom-service-type disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-ps connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect ps-2 via ps after 20ms"),
            "indexed default-ps connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-ps connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-2 via ps") && msg.contains("forced disconnect failure"),
            "indexed default-ps connect-timeout cleanup issue context should include default-service-type/index disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithFailingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-ps connect is blocked and cleanup disconnect fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect ps-0 via ps after 20ms"),
            "default-ps connect timeout should remain primary even if cleanup disconnect fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-ps connect-timeout failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-0 via ps") && msg.contains("forced disconnect failure"),
            "default-ps connect-timeout cleanup issue context should include default-service-type non-index disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-ps connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect ps-2 via ps after 20ms"),
            "indexed default-ps connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-ps connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-2 via ps after 20ms"
            ),
            "indexed default-ps connect-timeout cleanup issue context should include default-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-ps connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: connect ps-0 via ps after 20ms"),
            "default-ps connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-ps connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via ps after 20ms"
            ),
            "default-ps connect-timeout cleanup issue context should include default-service-type non-index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-ps connect and cleanup disconnect are both blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: connect ps-2 via parameter_server_custom after 20ms"
            ),
            "indexed custom-ps connect timeout should remain primary even if cleanup disconnect also times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-ps connect-timeout failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-2 via parameter_server_custom after 20ms"
            ),
            "indexed custom-ps connect-timeout cleanup issue context should include custom-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_does_not_hang() {
        let discovery = Arc::new(HangingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when worker registration blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-0 as worker"),
            "unexpected worker register-timeout error: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "register-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_includes_custom_service_type_context()
    {
        let discovery = Arc::new(HangingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when worker registration blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-0 as trainer_custom"),
            "worker register-timeout diagnostics should include custom service-type context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "register-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out()
    {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-0 as worker after 20ms"),
            "register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-worker register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-2 as worker after 20ms"),
            "indexed default-worker register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-worker register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-2 from worker after 20ms"),
            "indexed default-worker register-timeout cleanup issue context should include default-service-type/index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-2 via worker after 20ms"),
            "indexed default-worker register-timeout cleanup issue context should include default-service-type/index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-worker register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-0 as worker after 20ms"),
            "default-worker register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-worker register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"),
            "default-worker register-timeout cleanup issue context should include default-service-type non-index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"),
            "default-worker register-timeout cleanup issue context should include default-service-type non-index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-worker register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register worker-3 as trainer_custom after 20ms"
            ),
            "indexed custom-worker register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-worker register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from trainer_custom after 20ms"
            ),
            "indexed custom-worker register-timeout cleanup issue context should include custom-service-type/index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "indexed custom-worker register-timeout cleanup issue context should include custom-service-type/index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index worker register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register worker-0 as trainer_custom after 20ms"
            ),
            "custom non-index worker register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index worker register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "custom non-index worker register-timeout cleanup issue context should include custom-service-type deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "custom non-index worker register-timeout cleanup issue context should include custom-service-type disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails() {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-0 as worker after 20ms"),
            "register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "register-timeout cleanup issue context should include worker deregister failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "register-timeout cleanup issue context should include worker disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-worker register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register worker-3 as trainer_custom after 20ms"
            ),
            "indexed custom-worker register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-worker register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister worker-3 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "indexed custom-worker register-timeout cleanup issue context should include custom-service-type/index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "indexed custom-worker register-timeout cleanup issue context should include custom-service-type/index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index worker register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register worker-0 as trainer_custom after 20ms"
            ),
            "custom non-index worker register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index worker register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "custom non-index worker register-timeout cleanup issue context should include custom-service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "custom non-index worker register-timeout cleanup issue context should include custom-service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-worker register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-2 as worker after 20ms"),
            "indexed default-worker register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-worker register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister worker-2 from worker")
                && msg.contains("forced deregister failure"),
            "indexed default-worker register-timeout cleanup issue context should include default-service-type/index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via worker")
                && msg.contains("forced disconnect failure"),
            "indexed default-worker register-timeout cleanup issue context should include default-service-type/index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-worker register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register worker-0 as worker after 20ms"),
            "default-worker register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-worker register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "default-worker register-timeout cleanup issue context should include default-service-type non-index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "default-worker register-timeout cleanup issue context should include default-service-type non-index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out() {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-0 as ps after 20ms"),
            "ps register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails() {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-0 as ps after 20ms"),
            "ps register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister ps-0 from ps") && msg.contains("forced deregister failure"),
            "ps register-timeout cleanup issue context should include default-service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-0 via ps") && msg.contains("forced disconnect failure"),
            "ps register-timeout cleanup issue context should include default-service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-ps register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-2 as ps after 20ms"),
            "indexed default-ps register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-ps register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister ps-2 from ps after 20ms"),
            "indexed default-ps register-timeout cleanup issue context should include default-service-type/index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect ps-2 via ps after 20ms"),
            "indexed default-ps register-timeout cleanup issue context should include default-service-type/index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-ps register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-0 as ps after 20ms"),
            "default-ps register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-ps register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister ps-0 from ps after 20ms"),
            "default-ps register-timeout cleanup issue context should include default-service-type non-index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps after 20ms"),
            "default-ps register-timeout cleanup issue context should include default-service-type non-index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-ps register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register ps-2 as parameter_server_custom after 20ms"
            ),
            "indexed custom-ps register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-ps register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister ps-2 from parameter_server_custom after 20ms"
            ),
            "indexed custom-ps register-timeout cleanup issue context should include custom-service-type/index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-2 via parameter_server_custom after 20ms"
            ),
            "indexed custom-ps register-timeout cleanup issue context should include custom-service-type/index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index ps register and cleanup operations are blocked"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register ps-0 as parameter_server_custom after 20ms"
            ),
            "custom non-index ps register timeout should remain primary over cleanup timeout failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index ps register-timeout failures should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister ps-0 from parameter_server_custom after 20ms"
            ),
            "custom non-index ps register-timeout cleanup issue context should include custom-service-type deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom after 20ms"
            ),
            "custom non-index ps register-timeout cleanup issue context should include custom-service-type disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-ps register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register ps-2 as parameter_server_custom after 20ms"
            ),
            "indexed custom-ps register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-ps register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister ps-2 from parameter_server_custom")
                && msg.contains("forced deregister failure"),
            "indexed custom-ps register-timeout cleanup issue context should include custom-service-type/index deregister failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-2 via parameter_server_custom")
                && msg.contains("forced disconnect failure"),
            "indexed custom-ps register-timeout cleanup issue context should include custom-service-type/index disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index ps register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "Timed out during discovery operation: register ps-0 as parameter_server_custom after 20ms"
            ),
            "custom non-index ps register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index ps register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister ps-0 from parameter_server_custom")
                && msg.contains("forced deregister failure"),
            "custom non-index ps register-timeout cleanup issue context should include custom-service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-0 via parameter_server_custom")
                && msg.contains("forced disconnect failure"),
            "custom non-index ps register-timeout cleanup issue context should include custom-service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-ps register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-2 as ps after 20ms"),
            "indexed default-ps register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed default-ps register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister ps-2 from ps") && msg.contains("forced deregister failure"),
            "indexed default-ps register-timeout cleanup issue context should include default-service-type/index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-2 via ps") && msg.contains("forced disconnect failure"),
            "indexed default-ps register-timeout cleanup issue context should include default-service-type/index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingRegisterWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index default-ps register is blocked and cleanup fails"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-0 as ps after 20ms"),
            "default-ps register timeout should remain primary over cleanup-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "default-ps register-timeout failures should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister ps-0 from ps") && msg.contains("forced deregister failure"),
            "default-ps register-timeout cleanup issue context should include default-service-type non-index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-0 via ps") && msg.contains("forced disconnect failure"),
            "default-ps register-timeout cleanup issue context should include default-service-type non-index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_does_not_hang() {
        let discovery = Arc::new(HangingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps registration blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-0 as ps"),
            "unexpected ps register-timeout error: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "register-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_ps_register_timeout_includes_custom_service_type_context() {
        let discovery = Arc::new(HangingRegisterDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(200),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps registration blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out during discovery operation: register ps-0 as parameter_server_custom"),
            "ps register-timeout diagnostics should include custom service-type context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "register-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_does_not_hang_and_cleans_up() {
        let discovery = Arc::new(HangingDiscoverDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when worker discovery operation blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("last discovery error: Timed out during discovery operation: discover worker-0 for ps"),
            "expected discover timeout context in worker timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("after 20ms"),
            "discover-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_includes_custom_service_type_context() {
        let discovery = Arc::new(HangingDiscoverDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when worker discover operation blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-0 for parameter_server_custom after 20ms"
            ),
            "discover timeout diagnostics should include queried custom service type: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingDiscoverWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-service discover times out and cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup timeouts: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "discover-timeout diagnostics should include custom PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "discover-timeout diagnostics should include indexed worker service-id context: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-3 for parameter_server_custom after 20ms"
            ),
            "discover-timeout diagnostics should preserve indexed custom discover operation-timeout details: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from trainer_custom after 20ms"
            ),
            "discover-timeout cleanup issue context should include indexed custom-worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "discover-timeout cleanup issue context should include indexed custom-worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingDiscoverWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default discover times out and cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup timeouts: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "discover-timeout diagnostics should include default PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "discover-timeout diagnostics should include indexed worker service-id context: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-2 for ps after 20ms"
            ),
            "discover-timeout diagnostics should preserve indexed default discover operation-timeout details: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-2 from worker after 20ms"
            ),
            "discover-timeout cleanup issue context should include indexed default-worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-2 via worker after 20ms"
            ),
            "discover-timeout cleanup issue context should include indexed default-worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingDiscoverWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when default-service non-index discover times out and cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup timeouts: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "discover-timeout diagnostics should include default PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "discover-timeout diagnostics should include worker service-id context for default-service non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-0 for ps after 20ms"
            ),
            "discover-timeout diagnostics should preserve default discover operation-timeout details for non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"
            ),
            "discover-timeout cleanup issue context should include default-worker deregister-timeout diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"
            ),
            "discover-timeout cleanup issue context should include default-worker disconnect-timeout diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingDiscoverWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed default-service discover times out and cleanup steps fail"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup failures: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "discover-timeout diagnostics should include default PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "discover-timeout diagnostics should include indexed worker service-id context: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-2 for ps after 20ms"
            ),
            "discover-timeout diagnostics should preserve indexed default discover operation-timeout details: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("deregister worker-2 from worker")
                && msg.contains("forced deregister failure"),
            "discover-timeout cleanup issue context should include indexed default-worker deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via worker")
                && msg.contains("forced disconnect failure"),
            "discover-timeout cleanup issue context should include indexed default-worker disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type(
    ) {
        let discovery = Arc::new(HangingDiscoverWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when default-service non-index discover times out and cleanup steps fail"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup failures: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "discover-timeout diagnostics should include default PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "discover-timeout diagnostics should include worker service-id context for default-service non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-0 for ps after 20ms"
            ),
            "discover-timeout diagnostics should preserve default discover operation-timeout details for non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "discover-timeout cleanup issue context should include default-worker deregister-failure diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "discover-timeout cleanup issue context should include default-worker disconnect-failure diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(HangingDiscoverWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom discover times out and cleanup steps fail"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup failures: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "discover-timeout diagnostics should include custom PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "discover-timeout diagnostics should include indexed worker service-id context: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-3 for parameter_server_custom after 20ms"
            ),
            "discover-timeout diagnostics should preserve indexed custom discover operation-timeout details: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("deregister worker-3 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "discover-timeout cleanup issue context should include indexed custom-worker deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "discover-timeout cleanup issue context should include indexed custom-worker disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingDiscoverWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index discover times out and cleanup steps block"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup timeouts with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "discover-timeout diagnostics should include custom PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "discover-timeout diagnostics should include worker service-id context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-0 for parameter_server_custom after 20ms"
            ),
            "discover-timeout diagnostics should preserve custom discover operation-timeout details for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "discover-timeout cleanup issue context should include custom worker service-type deregister-timeout diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "discover-timeout cleanup issue context should include custom worker service-type disconnect-timeout diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type(
    ) {
        let discovery = Arc::new(HangingDiscoverWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_operation_timeout: Duration::from_millis(20),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(700),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index discover times out and cleanup steps fail"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "discover timeout should remain primary over cleanup failures with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "discover-timeout diagnostics should include custom PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "discover-timeout diagnostics should include worker service-id context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "last discovery error: Timed out during discovery operation: discover worker-0 for parameter_server_custom after 20ms"
            ),
            "discover-timeout diagnostics should preserve custom discover operation-timeout details for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover-timeout diagnostics should include cleanup issue context when cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "discover-timeout cleanup issue context should include custom worker service-type deregister-failure diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "discover-timeout cleanup issue context should include custom worker service-type disconnect-failure diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks() {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when connect-failure cleanup disconnect blocks"
        );
        let err = res.unwrap().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("forced connect failure"), "unexpected error: {msg}");
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "connect failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
            "connect-failure cleanup issue context should include disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed worker connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed worker connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed worker connect failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via worker after 20ms"
            ),
            "indexed worker connect-failure cleanup issue context should include default-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index worker connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "worker connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker connect failures should include cleanup issue context when disconnect cleanup times out for non-index path: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"
            ),
            "worker connect-failure cleanup issue context should include default-service-type non-index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-worker connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed custom-worker connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-worker connect failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "indexed custom-worker connect-failure cleanup issue context should include custom-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when custom non-index worker connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "custom non-index worker connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index worker connect failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "custom non-index worker connect-failure cleanup issue context should include custom-service-type disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when ps connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "ps connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps connect failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom after 20ms"
            ),
            "ps connect-failure cleanup issue context should include custom-service-type disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed custom-ps connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed custom-ps connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-ps connect failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-2 via parameter_server_custom after 20ms"
            ),
            "indexed custom-ps connect-failure cleanup issue context should include custom-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when indexed ps connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed ps connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed ps connect failures should include cleanup issue context when disconnect cleanup times out: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-2 via ps after 20ms"
            ),
            "indexed ps connect-failure cleanup issue context should include default-service-type/index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_ps_connect_failure_does_not_hang_when_disconnect_blocks_with_default_service_type(
    ) {
        let discovery = Arc::new(FailingConnectWithHangingDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(900),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when non-index ps connect-failure cleanup disconnect blocks"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("forced connect failure"),
            "ps connect failure should remain primary when cleanup disconnect blocks: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps connect failures should include cleanup issue context when disconnect cleanup times out for non-index path: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect ps-0 via ps after 20ms"
            ),
            "ps connect-failure cleanup issue context should include default-service-type non-index disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if it blocks"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail() {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "connect failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "connect-failure cleanup issue context should include disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed connect failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via worker")
                && msg.contains("forced disconnect failure"),
            "indexed connect-failure cleanup issue context should include default-service-type/index disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_default_service_type(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected non-index connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "connect failures should include cleanup issue context when disconnect cleanup fails for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "connect-failure cleanup issue context should include default-service-type non-index disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed custom-worker connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-worker connect failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "indexed custom-worker connect-failure cleanup issue context should include custom-service-type/index disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_connect_error_when_connect_and_disconnect_fail_with_custom_service_type(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected custom non-index connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "custom non-index worker connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "custom non-index worker connect failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "custom non-index worker connect-failure cleanup issue context should include custom-service-type disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_custom_service_type(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "ps connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps connect failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-0 via parameter_server_custom")
                && msg.contains("forced disconnect failure"),
            "ps connect-failure cleanup issue context should include disconnect failure diagnostics with custom service-type operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_custom_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed custom-ps connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed custom-ps connect failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-2 via parameter_server_custom")
                && msg.contains("forced disconnect failure"),
            "indexed custom-ps connect-failure cleanup issue context should include custom-service-type/index disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_default_service_type_and_index(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 2,
            num_ps: 3,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected indexed connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "indexed ps connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "indexed ps connect failures should include cleanup issue context when disconnect cleanup fails: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-2 via ps") && msg.contains("forced disconnect failure"),
            "indexed ps connect-failure cleanup issue context should include default-service-type/index disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_returns_ps_connect_error_when_connect_and_disconnect_fail_with_default_service_type(
    ) {
        let discovery = Arc::new(FailingConnectAndDisconnectDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Ps,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            bind_addr: "127.0.0.1:0".parse().unwrap(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected non-index connect failure");
        let msg = err.to_string();
        assert!(
            msg.contains("forced connect failure"),
            "ps connect error should be returned even if disconnect also fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "ps connect failures should include cleanup issue context when disconnect cleanup fails for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("disconnect ps-0 via ps") && msg.contains("forced disconnect failure"),
            "ps connect-failure cleanup issue context should include default-service-type non-index disconnect failure diagnostics with operation context: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even when it also fails"
        );
    }

    #[tokio::test]
    async fn test_run_distributed_attempts_disconnect_when_deregister_fails_after_success() {
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let discovery = Arc::new(FailingDeregisterAfterSuccessDiscovery::new(
            actual_addr.to_string(),
        ));
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            dim: 8,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected deregister failure after successful worker run")
            .to_string();
        assert!(
            msg.contains("forced deregister failure"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after successful role completion"),
            "deregister failure after successful role should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker"),
            "deregister failure after successful role should include cleanup operation context: {msg}"
        );
        assert_eq!(
            discovery.deregister_count(),
            1,
            "deregister should be attempted exactly once"
        );
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be attempted even when deregister fails after success"
        );

        ps_server.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_surfaces_deregister_timeout_after_success() {
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let discovery = Arc::new(HangingDeregisterAfterSuccessDiscovery::new(
            actual_addr.to_string(),
        ));
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            dim: 8,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected deregister timeout after successful run");
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after successful role completion"),
            "deregister timeout after successful role should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "cleanup-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted when deregister times out"
        );

        ps_server.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_surfaces_deregister_timeout_with_custom_service_type_after_success(
    ) {
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let discovery = Arc::new(HangingDeregisterAfterSuccessDiscovery::new(
            actual_addr.to_string(),
        ));
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            dim: 8,
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected deregister timeout after successful run");
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from trainer_custom"),
            "deregister-timeout diagnostics should include custom service-type context: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after successful role completion"),
            "custom deregister timeout after successful role should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "cleanup-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted when deregister times out"
        );

        ps_server.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_surfaces_disconnect_failure_after_success() {
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let discovery = Arc::new(FailingDisconnectAfterSuccessDiscovery::new(
            actual_addr.to_string(),
        ));
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            dim: 8,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected disconnect failure after successful run");
        let msg = err.to_string();
        assert!(
            msg.contains("forced disconnect failure"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after successful role completion"),
            "disconnect failure after successful role should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker"),
            "disconnect failure after successful role should include cleanup operation context: {msg}"
        );
        assert_eq!(
            discovery.deregister_count(),
            1,
            "deregister should still be attempted before disconnect"
        );
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be attempted exactly once"
        );

        ps_server.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_surfaces_disconnect_timeout_after_success() {
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let discovery = Arc::new(HangingDisconnectAfterSuccessDiscovery::new(
            actual_addr.to_string(),
        ));
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            dim: 8,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected disconnect timeout after successful run");
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
            "unexpected error: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after successful role completion"),
            "disconnect timeout after successful role should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "cleanup-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be attempted exactly once before timing out"
        );

        ps_server.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_surfaces_disconnect_timeout_with_custom_service_type_after_success(
    ) {
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let discovery = Arc::new(HangingDisconnectAfterSuccessDiscovery::new(
            actual_addr.to_string(),
        ));
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            dim: 8,
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected disconnect timeout after successful run");
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via trainer_custom"),
            "disconnect-timeout diagnostics should include custom service-type context: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after successful role completion"),
            "custom disconnect-timeout after successful role should include cleanup issue context: {msg}"
        );
        assert!(
            msg.contains("after 200ms"),
            "cleanup-timeout diagnostics should include configured timeout duration: {msg}"
        );
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should be attempted exactly once before timing out"
        );

        ps_server.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_prefers_deregister_error_when_both_post_success_cleanup_steps_fail(
    ) {
        let ps = PsServer::new(0, 8);
        let (listener, actual_addr) = bind_ephemeral("127.0.0.1:0".parse().unwrap())
            .await
            .unwrap();
        let ps_server = tokio::spawn(
            tonic::transport::Server::builder()
                .add_service(Arc::clone(&ps).into_service())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
        );

        let discovery = Arc::new(FailingDeregisterAndDisconnectAfterSuccessDiscovery::new(
            actual_addr.to_string(),
        ));
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            dim: 8,
            ..DistributedRunConfig::default()
        };

        let err = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("expected cleanup failures after successful run");
        let msg = err.to_string();
        assert!(
            msg.contains("forced deregister failure"),
            "deregister failure should take precedence when both cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker"),
            "post-success cleanup issue diagnostics should include deregister operation context when both cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after successful role completion"),
            "post-success cleanup failures should include cleanup issue context when both cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "post-success cleanup issue context should include disconnect failure diagnostics: {msg}"
        );
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(
            discovery.disconnect_count(),
            1,
            "disconnect should still be attempted even if deregister already failed"
        );

        ps_server.abort();
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_when_cleanup_steps_timeout() {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang even when cleanup steps time out"
        );
        let err = res.unwrap().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role error should be preserved over cleanup timeout errors: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker-discovery timeout diagnostics should include default PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-discovery timeout diagnostics should include worker service-id context: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-role errors should include cleanup issue context when cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
            "cleanup issue context should include deregister timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
            "cleanup issue context should include disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_timeout()
    {
        let discovery = Arc::new(WorkerDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(1200),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when discover returns an error and cleanup steps time out"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"),
            "discover failure cleanup issue context should include worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"),
            "discover failure cleanup issue context should include worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(1200),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when discover returns an error with custom service types and cleanup steps time out"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker discover-failure diagnostics should include custom PS service type context: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "worker discover-failure diagnostics should include worker index propagated in service-id context: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with custom service types: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup steps time out with custom service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from trainer_custom after 20ms"
            ),
            "discover failure cleanup issue context should include custom worker service-type deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "discover failure cleanup issue context should include custom worker service-type disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(1200),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when discover returns an error with indexed default service types and cleanup steps time out"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with indexed default service types: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker discover-failure diagnostics should include default PS service type context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker discover-failure diagnostics should include indexed worker service-id context when cleanup steps time out: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with indexed default service types: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup steps time out with indexed default service types: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-2 from worker after 20ms"),
            "discover failure cleanup issue context should include indexed default-worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-2 via worker after 20ms"),
            "discover failure cleanup issue context should include indexed default-worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(1200),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when discover returns an error with default-service non-index path and cleanup steps time out"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with default service type: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker discover-failure diagnostics should include default PS service type context when cleanup steps time out for non-index path: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker discover-failure diagnostics should include worker service-id context when cleanup steps time out for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup steps time out for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"),
            "discover failure cleanup issue context should include default-worker non-index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"),
            "discover failure cleanup issue context should include default-worker non-index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(1200),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when discover returns an error with custom non-index service types and cleanup steps time out"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker discover-failure diagnostics should include custom PS service type context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker discover-failure diagnostics should include worker service-id context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup steps time out with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "discover failure cleanup issue context should include custom worker service-type deregister-timeout diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "discover failure cleanup issue context should include custom worker service-type disconnect-timeout diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let res = tokio::time::timeout(
            Duration::from_millis(1200),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await;
        assert!(
            res.is_ok(),
            "run_distributed should not hang when discover returns an error with custom non-index service types and cleanup steps time out"
        );
        let msg = res.unwrap().unwrap_err().to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker discover-failure diagnostics should include custom PS service type context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker discover-failure diagnostics should include worker service-id context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup steps time out with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "discover failure cleanup issue context should include custom worker service-type deregister-timeout diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "discover failure cleanup issue context should include custom worker service-type disconnect-timeout diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_fail() {
        let discovery = Arc::new(WorkerDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("discover failure with failing cleanup should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "discover failure cleanup issue context should include worker deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "discover failure cleanup issue context should include worker disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("discover failure with failing cleanup should surface as a role error with custom service types/index")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with custom service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker discover-failure diagnostics should include custom PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker discover-failure diagnostics should include worker index propagated in service-id context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with custom service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup fails with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("deregister worker-2 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "discover failure cleanup issue context should include custom worker service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "discover failure cleanup issue context should include custom worker service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("discover failure with failing cleanup should surface as a role error with indexed default service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with indexed default service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker discover-failure diagnostics should include default PS service type context when cleanup fails with indexed worker: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker discover-failure diagnostics should include indexed worker service-id context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with indexed default service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup fails with indexed default service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-2 from worker")
                && msg.contains("forced deregister failure"),
            "discover failure cleanup issue context should include indexed default-worker deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via worker")
                && msg.contains("forced disconnect failure"),
            "discover failure cleanup issue context should include indexed default-worker disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("discover failure with failing cleanup should surface as a role error with default-service non-index path")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with default service type when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker discover-failure diagnostics should include default PS service type context when cleanup fails for non-index path: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker discover-failure diagnostics should include worker service-id context when cleanup fails for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with default-service non-index path when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup fails for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "discover failure cleanup issue context should include default-worker non-index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "discover failure cleanup issue context should include default-worker non-index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("discover failure with failing cleanup should surface as a role error with custom non-index service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with custom non-index service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker discover-failure diagnostics should include custom PS service type context for custom non-index paths when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker discover-failure diagnostics should include worker service-id context for custom non-index paths when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with custom non-index service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup fails with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "discover failure cleanup issue context should include custom worker service-type deregister-failure diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "discover failure cleanup issue context should include custom worker service-type disconnect-failure diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("discover failure with failing cleanup should surface as a role error with custom non-index service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker discover failure should still surface as worker-role timeout diagnostic with custom non-index service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker discover-failure diagnostics should include custom PS service type context for custom non-index paths when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker discover-failure diagnostics should include worker service-id context for custom non-index paths when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker discover-failure diagnostics should preserve last discovery error details with custom non-index service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "discover failure diagnostics should include cleanup issue context when cleanup fails with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "discover failure cleanup issue context should include custom worker service-type deregister-failure diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "discover failure cleanup issue context should include custom worker service-type disconnect-failure diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_when_cleanup_steps_fail() {
        let discovery = Arc::new(WorkerTimeoutWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker timeout with failing cleanup should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should remain primary over cleanup failures: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker timeout diagnostics should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "worker-timeout cleanup issue context should include worker deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "worker-timeout cleanup issue context should include worker disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker timeout with failing cleanup should surface as a role error with custom service types/index")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should remain primary over cleanup failures with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker-timeout diagnostics should include custom PS service-type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "worker-timeout diagnostics should include propagated worker index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout diagnostics should include cleanup issue context when cleanup fails with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("deregister worker-3 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker-timeout cleanup issue context should include custom worker service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker-timeout cleanup issue context should include custom worker service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker timeout with failing cleanup should surface as a role error with custom non-index service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should remain primary over cleanup failures with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker-timeout diagnostics should include custom PS service-type context when cleanup fails for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-timeout diagnostics should include worker service-id context when cleanup fails for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout diagnostics should include cleanup issue context when cleanup fails with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker-timeout cleanup issue context should include custom worker service-type deregister-failure diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker-timeout cleanup issue context should include custom worker service-type disconnect-failure diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker timeout with failing cleanup should surface as a role error with custom non-index service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should remain primary over cleanup failures with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker-timeout diagnostics should include custom PS service-type context when cleanup fails for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-timeout diagnostics should include worker service-id context when cleanup fails for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout diagnostics should include cleanup issue context when cleanup fails with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker-timeout cleanup issue context should include custom worker service-type deregister-failure diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker-timeout cleanup issue context should include custom worker service-type disconnect-failure diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker timeout with failing cleanup should surface as a role error with default service type and index")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should remain primary over cleanup failures with default service type and index: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker-timeout diagnostics should include default PS service-type context when cleanup fails with indexed worker: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker-timeout diagnostics should include propagated indexed worker service-id when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout diagnostics should include cleanup issue context when cleanup fails with default service type and index: {msg}"
        );
        assert!(
            msg.contains("deregister worker-2 from worker")
                && msg.contains("forced deregister failure"),
            "worker-timeout cleanup issue context should include indexed default-worker deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via worker")
                && msg.contains("forced disconnect failure"),
            "worker-timeout cleanup issue context should include indexed default-worker disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_default_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker timeout with failing cleanup should surface as a role error with default-service non-index path")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should remain primary over cleanup failures with default service type: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker-timeout diagnostics should include default PS service-type context for non-index path: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-timeout diagnostics should include worker service-id context for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout diagnostics should include cleanup issue context when cleanup fails with default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "worker-timeout cleanup issue context should include default-worker non-index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "worker-timeout cleanup issue context should include default-worker non-index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_when_cleanup_steps_fail() {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        assert_worker_error_cleanup_fail_case(cfg, "ps", "worker-0", "worker").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_discovery_service_type_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_worker_error_cleanup_fail_case(cfg, "parameter_server_custom", "worker-0", "worker")
            .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_worker_error_cleanup_fail_case(
            cfg,
            "parameter_server_custom",
            "worker-3",
            "trainer_custom",
        )
        .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_worker_error_cleanup_fail_case(
            cfg,
            "parameter_server_custom",
            "worker-0",
            "trainer_custom",
        )
        .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };
        assert_worker_error_cleanup_fail_case(
            cfg,
            "parameter_server_custom",
            "worker-0",
            "trainer_custom",
        )
        .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        assert_worker_error_cleanup_fail_case(cfg, "ps", "worker-2", "worker").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_default_service_type_when_cleanup_steps_fail(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        assert_worker_error_cleanup_fail_case(cfg, "ps", "worker-0", "worker").await;
    }

    async fn assert_worker_error_cleanup_fail_case(
        cfg: DistributedRunConfig,
        expected_ps_service_type: &str,
        expected_worker_service_id: &str,
        expected_worker_service_type: &str,
    ) {
        let discovery = Arc::new(WorkerTimeoutWithFailingCleanupDiscovery::new());
        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker timeout with failing cleanup should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role error should be preserved over cleanup failures: {msg}"
        );
        assert!(
            msg.contains(&format!("service type: {expected_ps_service_type}")),
            "worker-discovery timeout diagnostics should include expected PS service-type context: {msg}"
        );
        assert!(
            msg.contains(&format!("for {expected_worker_service_id}")),
            "worker-discovery timeout diagnostics should include expected worker service-id context: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-role errors should include cleanup issue context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains(&format!(
                "deregister {expected_worker_service_id} from {expected_worker_service_type}"
            )) && msg.contains("forced deregister failure"),
            "cleanup issue context should include expected deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains(&format!(
                "disconnect {expected_worker_service_id} via {expected_worker_service_type}"
            )) && msg.contains("forced disconnect failure"),
            "cleanup issue context should include expected disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_when_cleanup_steps_timeout() {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };
        assert_worker_timeout_cleanup_timeout_case(cfg, "ps", "worker-0", "worker").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        assert_worker_timeout_cleanup_timeout_case(
            cfg,
            "parameter_server_custom",
            "worker-3",
            "trainer_custom",
        )
        .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        assert_worker_timeout_cleanup_timeout_case(
            cfg,
            "parameter_server_custom",
            "worker-0",
            "trainer_custom",
        )
        .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        assert_worker_timeout_cleanup_timeout_case(
            cfg,
            "parameter_server_custom",
            "worker-0",
            "trainer_custom",
        )
        .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        assert_worker_timeout_cleanup_timeout_case(cfg, "ps", "worker-2", "worker").await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_timeout_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        assert_worker_timeout_cleanup_timeout_case(cfg, "ps", "worker-0", "worker").await;
    }

    async fn assert_worker_timeout_cleanup_timeout_case(
        cfg: DistributedRunConfig,
        expected_ps_service_type: &str,
        expected_worker_service_id: &str,
        expected_worker_service_type: &str,
    ) {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker timeout cleanup steps time out");
        let msg = run_result
            .expect_err("worker timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker timeout should remain primary over cleanup timeout errors: {msg}"
        );
        assert!(
            msg.contains(&format!("service type: {expected_ps_service_type}")),
            "worker-timeout diagnostics should include expected PS service-type context: {msg}"
        );
        assert!(
            msg.contains(&format!("for {expected_worker_service_id}")),
            "worker-timeout diagnostics should include expected worker service-id context: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout diagnostics should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains(&format!(
                "Timed out during discovery cleanup: deregister {expected_worker_service_id} from {expected_worker_service_type}"
            )),
            "worker-timeout cleanup context should include expected deregister timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(&format!(
                "Timed out during discovery cleanup: disconnect {expected_worker_service_id} via {expected_worker_service_type}"
            )),
            "worker-timeout cleanup context should include expected disconnect timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    async fn assert_worker_ordering_issue_timeout_cleanup_timeout_case(
        cfg: DistributedRunConfig,
        expected_ps_service_type: &str,
        expected_worker_service_id: &str,
        expected_worker_service_type: &str,
    ) {
        assert_worker_timeout_cleanup_timeout_case(
            cfg,
            expected_ps_service_type,
            expected_worker_service_id,
            expected_worker_service_type,
        )
        .await;
    }

    #[tokio::test]
    async fn test_run_distributed_worker_ordering_issue_timeout_cleanup_timeout_case_alias() {
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };
        assert_worker_ordering_issue_timeout_cleanup_timeout_case(cfg, "ps", "worker-0", "worker")
            .await;
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 2,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out due to ordering issue and cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering-issue timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup timeout errors: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering-issue timeout should include default PS service type context: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker ordering-issue timeout should include indexed worker service-id context: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve last ordering issue diagnostics when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-2 from worker after 20ms"),
            "worker ordering-issue timeout cleanup context should include indexed default-worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-2 via worker after 20ms"),
            "worker ordering-issue timeout cleanup context should include indexed default-worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out due to ordering issue and default-service non-index cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering-issue timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup timeout errors with default service type: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering-issue timeout should include default PS service type context for non-index path: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering-issue timeout should include worker service-id context for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics for default-service non-index path when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup times out for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"),
            "worker ordering-issue timeout cleanup context should include default-worker non-index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"),
            "worker ordering-issue timeout cleanup context should include default-worker non-index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out due to ordering issue with custom service types and cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering-issue timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup timeout errors with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering-issue timeout should include custom PS service type context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering-issue timeout should include worker service-id context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics with custom service types when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "worker ordering-issue timeout cleanup context should include custom worker service-type deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "worker ordering-issue timeout cleanup context should include custom worker service-type disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out due to ordering issue with custom service types and cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering-issue timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup timeout errors with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering-issue timeout should include custom PS service type context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering-issue timeout should include worker service-id context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics with custom service types when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "worker ordering-issue timeout cleanup context should include custom worker service-type deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "worker ordering-issue timeout cleanup context should include custom worker service-type disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 2,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering-issue timeout with failing cleanup should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup failures with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering-issue timeout should include custom PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "worker ordering-issue timeout should include indexed worker service-id context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics with custom service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup fails with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("deregister worker-3 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker ordering-issue timeout cleanup context should include custom worker service-type/index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker ordering-issue timeout cleanup context should include custom worker service-type/index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 2,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering-issue timeout with failing cleanup should surface as a role error with default service types/index")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup failures with default service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering-issue timeout should include default PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker ordering-issue timeout should include indexed worker service-id context when cleanup fails with default service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics with default service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup fails with default service types/index: {msg}"
        );
        assert!(
            msg.contains("deregister worker-2 from worker")
                && msg.contains("forced deregister failure"),
            "worker ordering-issue timeout cleanup context should include default worker service-type/index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via worker")
                && msg.contains("forced disconnect failure"),
            "worker ordering-issue timeout cleanup context should include default worker service-type/index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering-issue timeout with failing cleanup should surface as a role error with default-service non-index path")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup failures with default service type: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering-issue timeout should include default PS service type context for non-index path when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering-issue timeout should include worker service-id context for default-service non-index path when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics for default-service non-index path when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup fails for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "worker ordering-issue timeout cleanup context should include default-worker non-index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "worker ordering-issue timeout cleanup context should include default-worker non-index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 2,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out due to ordering issue and custom cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering-issue timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup timeout errors with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering-issue timeout should include custom PS service type context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "worker ordering-issue timeout should include indexed worker service-id context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics with custom service types/index when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup times out with custom service types/index: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from trainer_custom after 20ms"
            ),
            "worker ordering-issue timeout cleanup context should include custom worker service-type/index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "worker ordering-issue timeout cleanup context should include custom worker service-type/index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 2,
            num_workers: 3,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out with ordering+discovery errors and cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering+discovery-error timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup timeout errors: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering+discovery-error timeout should include default PS service type context: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker ordering+discovery-error timeout should include indexed worker service-id context: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-2 from worker after 20ms"),
            "worker ordering+discovery-error timeout cleanup context should include indexed default-worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-2 via worker after 20ms"),
            "worker ordering+discovery-error timeout cleanup context should include indexed default-worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out with ordering+discovery errors and default-service non-index cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering+discovery-error timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup timeout errors with default service type: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering+discovery-error timeout should include default PS service type context: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering+discovery-error timeout should include worker service-id context for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with default service type when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with default service type when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup times out with default service type: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"),
            "worker ordering+discovery-error timeout cleanup context should include default-worker deregister-timeout diagnostics for non-index path: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"),
            "worker ordering+discovery-error timeout cleanup context should include default-worker disconnect-timeout diagnostics for non-index path: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out with ordering+discovery errors and custom non-index cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering+discovery-error timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup timeout errors with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering+discovery-error timeout should include custom PS service type context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering+discovery-error timeout should include worker service-id context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with custom service types when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with custom service types when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out with ordering+discovery errors and custom non-index cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering+discovery-error timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup timeout errors with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering+discovery-error timeout should include custom PS service type context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering+discovery-error timeout should include worker service-id context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with custom service types when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with custom service types when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 2,
            num_workers: 4,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering+discovery-error timeout with failing cleanup should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup failures with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering+discovery-error timeout should include custom PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "worker ordering+discovery-error timeout should include indexed worker service-id context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with custom service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with custom service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup fails with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("deregister worker-3 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type/index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-3 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type/index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 2,
            num_workers: 3,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering+discovery-error timeout with failing cleanup should surface as a role error with default service types/index")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup failures with default service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering+discovery-error timeout should include default PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker ordering+discovery-error timeout should include indexed worker service-id context when cleanup fails with default service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with default service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with default service types/index when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup fails with default service types/index: {msg}"
        );
        assert!(
            msg.contains("deregister worker-2 from worker")
                && msg.contains("forced deregister failure"),
            "worker ordering+discovery-error timeout cleanup context should include default worker service-type/index deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-2 via worker")
                && msg.contains("forced disconnect failure"),
            "worker ordering+discovery-error timeout cleanup context should include default worker service-type/index disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 1,
            retry_backoff_ms: 1,
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering+discovery-error timeout with failing cleanup should surface as a role error with default-service non-index path")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup failures with default service type: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker ordering+discovery-error timeout should include default PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering+discovery-error timeout should include worker service-id context for default-service non-index path when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with default service type when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with default service type when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup fails with default service type: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from worker")
                && msg.contains("forced deregister failure"),
            "worker ordering+discovery-error timeout cleanup context should include default-worker deregister-failure diagnostics for non-index path: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via worker")
                && msg.contains("forced disconnect failure"),
            "worker ordering+discovery-error timeout cleanup context should include default-worker disconnect-failure diagnostics for non-index path: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering+discovery-error timeout with failing cleanup should surface as a role error with custom non-index service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup failures with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering+discovery-error timeout should include custom PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering+discovery-error timeout should include worker service-id context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with custom service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with custom service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering+discovery-error timeout with failing cleanup should surface as a role error with custom non-index service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup failures with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering+discovery-error timeout should include custom PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering+discovery-error timeout should include worker service-id context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with custom service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with custom service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering-issue timeout with failing cleanup should surface as a role error with custom service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup failures with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering-issue timeout should include custom PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering-issue timeout should include worker service-id context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics with custom service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker ordering-issue timeout cleanup context should include custom worker service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker ordering-issue timeout cleanup context should include custom worker service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        let discovery = Arc::new(WorkerOrderingIssueWithFailingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 2,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let msg = run_distributed(Arc::clone(&discovery), cfg)
            .await
            .expect_err("worker ordering-issue timeout with failing cleanup should surface as a role error with custom service types")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering-issue timeout should remain primary over cleanup failures with custom service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering-issue timeout should include custom PS service type context when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker ordering-issue timeout should include worker service-id context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering-issue timeout should preserve ordering issue diagnostics with custom service types when cleanup fails: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering-issue timeout should include cleanup issue context when cleanup fails with custom service types: {msg}"
        );
        assert!(
            msg.contains("deregister worker-0 from trainer_custom")
                && msg.contains("forced deregister failure"),
            "worker ordering-issue timeout cleanup context should include custom worker service-type deregister-failure diagnostics: {msg}"
        );
        assert!(
            msg.contains("disconnect worker-0 via trainer_custom")
                && msg.contains("forced disconnect failure"),
            "worker ordering-issue timeout cleanup context should include custom worker service-type disconnect-failure diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerOrderingAndDiscoverErrorWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 2,
            num_workers: 4,
            connect_retries: 1,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when worker discovery times out with ordering+discovery errors and custom cleanup steps time out");
        let msg = run_result
            .expect_err("worker ordering+discovery-error timeout with cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker ordering+discovery-error timeout should remain primary over cleanup timeout errors with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker ordering+discovery-error timeout should include custom PS service type context when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "worker ordering+discovery-error timeout should include indexed worker service-id context when cleanup times out with custom service types: {msg}"
        );
        assert!(
            msg.contains("last ordering issue: MixedIndexMetadataPresence"),
            "worker ordering+discovery-error timeout should preserve ordering issue diagnostics with custom service types/index when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("last discovery error: Internal error: forced discover failure"),
            "worker ordering+discovery-error timeout should preserve discovery error diagnostics with custom service types/index when cleanup times out: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker ordering+discovery-error timeout should include cleanup issue context when cleanup times out with custom service types/index: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from trainer_custom after 20ms"
            ),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type/index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "worker ordering+discovery-error timeout cleanup context should include custom worker service-type/index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 2);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_discovery_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang even when cleanup steps time out with custom worker discovery service type");
        let msg = run_result
            .expect_err("worker timeout with custom discovery service type and cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role error should be preserved over cleanup timeout errors with custom worker discovery service type: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker-discovery timeout diagnostics should include configured custom PS service-type context: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-discovery timeout diagnostics should include worker service-id context with custom discover service type: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-role errors should include cleanup issue context with custom service type when cleanup steps fail: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
            "cleanup issue context should include worker deregister timeout diagnostics with custom discovery service type: {msg}"
        );
        assert!(
            msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
            "cleanup issue context should include worker disconnect timeout diagnostics with custom discovery service type: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 3,
            num_ps: 1,
            num_workers: 4,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when indexed custom worker discovery times out and cleanup steps time out");
        let msg = run_result
            .expect_err("worker timeout with custom service types/index and cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role timeout should remain primary over cleanup timeout errors with custom service types/index: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker-timeout diagnostics should include custom PS service-type context with indexed custom service types: {msg}"
        );
        assert!(
            msg.contains("for worker-3"),
            "worker-timeout diagnostics should include indexed worker service-id context: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout errors should include cleanup issue context when cleanup steps time out with custom service types/index: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-3 from trainer_custom after 20ms"
            ),
            "worker-timeout cleanup issue context should include indexed custom-worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-3 via trainer_custom after 20ms"
            ),
            "worker-timeout cleanup issue context should include indexed custom-worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when custom non-index worker discovery times out and cleanup steps time out");
        let msg = run_result
            .expect_err("worker timeout with custom non-index service type and cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role timeout should remain primary over cleanup timeout errors with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker-timeout diagnostics should include custom PS service-type context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-timeout diagnostics should include worker service-id context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout errors should include cleanup issue context when cleanup steps time out with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "worker-timeout cleanup issue context should include custom worker service-type deregister-timeout diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "worker-timeout cleanup issue context should include custom worker service-type disconnect-timeout diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_service_type_ps: "parameter_server_custom".to_string(),
            discovery_service_type_worker: "trainer_custom".to_string(),
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when custom non-index worker discovery times out and cleanup steps time out");
        let msg = run_result
            .expect_err("worker timeout with custom non-index service types and cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role timeout should remain primary over cleanup timeout errors with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains("service type: parameter_server_custom"),
            "worker-timeout diagnostics should include custom PS service-type context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-timeout diagnostics should include worker service-id context for custom non-index paths: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout errors should include cleanup issue context when cleanup steps time out with custom non-index service types: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
            ),
            "worker-timeout cleanup issue context should include custom worker service-type deregister-timeout diagnostics for non-index paths: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
            ),
            "worker-timeout cleanup issue context should include custom worker service-type disconnect-timeout diagnostics for non-index paths: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 2,
            num_ps: 1,
            num_workers: 3,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when indexed default worker discovery times out and cleanup steps time out");
        let msg = run_result
            .expect_err("worker timeout with default service type/index and cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role timeout should remain primary over cleanup timeout errors with default service type/index: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker-timeout diagnostics should include default PS service-type context with indexed default service types: {msg}"
        );
        assert!(
            msg.contains("for worker-2"),
            "worker-timeout diagnostics should include indexed worker service-id context for default service types: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout errors should include cleanup issue context when cleanup steps time out with default service type/index: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-2 from worker after 20ms"
            ),
            "worker-timeout cleanup issue context should include indexed default-worker deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-2 via worker after 20ms"
            ),
            "worker-timeout cleanup issue context should include indexed default-worker disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_preserves_worker_error_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let run_result = tokio::time::timeout(
            Duration::from_millis(1500),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should not hang when default-service non-index worker discovery times out and cleanup steps time out");
        let msg = run_result
            .expect_err("worker timeout with default non-index service type and cleanup timeouts should surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role timeout should remain primary over cleanup timeout errors with default service type: {msg}"
        );
        assert!(
            msg.contains("service type: ps"),
            "worker-timeout diagnostics should include default PS service-type context for non-index path: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-timeout diagnostics should include worker service-id context for default-service non-index path: {msg}"
        );
        assert!(
            msg.contains("discovery cleanup encountered issues after role error"),
            "worker-timeout errors should include cleanup issue context when cleanup steps time out with default-service non-index path: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"
            ),
            "worker-timeout cleanup issue context should include default-worker non-index deregister-timeout diagnostics: {msg}"
        );
        assert!(
            msg.contains(
                "Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"
            ),
            "worker-timeout cleanup issue context should include default-worker non-index disconnect-timeout diagnostics: {msg}"
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[tokio::test]
    async fn test_run_distributed_honors_configured_cleanup_timeout() {
        let discovery = Arc::new(WorkerTimeoutWithHangingCleanupDiscovery::new());
        let cfg = DistributedRunConfig {
            role: Role::Worker,
            index: 0,
            num_ps: 1,
            num_workers: 1,
            connect_retries: 0,
            retry_backoff_ms: 1,
            discovery_cleanup_timeout: Duration::from_millis(20),
            ..DistributedRunConfig::default()
        };

        let started = std::time::Instant::now();
        let run_result = tokio::time::timeout(
            Duration::from_millis(1200),
            run_distributed(Arc::clone(&discovery), cfg),
        )
        .await
        .expect("run_distributed should return promptly when cleanup timeout is reduced");
        let msg = run_result
            .expect_err("worker timeout with reduced cleanup timeout should still surface as a role error")
            .to_string();
        assert!(
            msg.contains("Timed out waiting for PS discovery"),
            "worker-role error should still be preserved: {msg}"
        );
        assert!(
            msg.contains("for worker-0"),
            "worker-discovery timeout diagnostics should include worker service-id context: {msg}"
        );

        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(320),
            "configured cleanup timeout should bound total cleanup delay (elapsed {:?})",
            elapsed
        );
        assert_eq!(discovery.connect_count(), 1);
        assert_eq!(discovery.discover_count(), 1);
        assert_eq!(discovery.deregister_count(), 1);
        assert_eq!(discovery.disconnect_count(), 1);
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_includes_custom_service_type_disconnect_failure_context(
    ) {
        test_run_distributed_ps_registration_failure_includes_custom_service_type_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_disconnect_failure_context(
    ) {
        test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_custom_service_type_includes_disconnect_failure_context(
    ) {
        test_run_distributed_ps_registration_failure_with_custom_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_disconnect_failure_context(
    ) {
        test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_default_service_type_includes_disconnect_failure_context(
    ) {
        test_run_distributed_ps_registration_failure_with_default_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_includes_custom_service_type_disconnect_failure_context(
    ) {
        test_run_distributed_worker_registration_failure_includes_custom_service_type_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_disconnect_failure_context(
    ) {
        test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_custom_service_type_includes_disconnect_failure_context(
    ) {
        test_run_distributed_worker_registration_failure_with_custom_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_disconnect_failure_context(
    ) {
        test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_default_service_type_includes_disconnect_failure_context(
    ) {
        test_run_distributed_worker_registration_failure_with_default_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_includes_custom_service_type_cleanup_timeout_context(
    ) {
        test_run_distributed_ps_registration_failure_includes_custom_service_type_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_custom_service_type_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_ps_registration_failure_with_custom_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_default_service_type_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_ps_registration_failure_with_default_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_includes_custom_service_type_cleanup_timeout_context(
    ) {
        test_run_distributed_worker_registration_failure_includes_custom_service_type_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_custom_service_type_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_worker_registration_failure_with_custom_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_default_service_type_includes_cleanup_timeout_context(
    ) {
        test_run_distributed_worker_registration_failure_with_default_service_type_includes_cleanup_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_includes_custom_service_type_when_cleanup_blocks(
    ) {
        test_run_distributed_ps_registration_failure_includes_custom_service_type_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_ps_registration_failure_with_custom_service_type_and_index_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_custom_service_type_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_ps_registration_failure_with_custom_service_type_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_ps_registration_failure_with_default_service_type_and_index_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_ps_registration_failure_with_default_service_type_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_ps_registration_failure_with_default_service_type_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_includes_custom_service_type_when_cleanup_blocks(
    ) {
        test_run_distributed_worker_registration_failure_includes_custom_service_type_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_worker_registration_failure_with_custom_service_type_and_index_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_custom_service_type_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_worker_registration_failure_with_custom_service_type_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_worker_registration_failure_with_default_service_type_and_index_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_worker_registration_failure_with_default_service_type_includes_when_cleanup_blocks(
    ) {
        test_run_distributed_worker_registration_failure_with_default_service_type_includes_disconnect_failure_context();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_deregister_timeout_with_default_service_type_after_success() {
        test_run_distributed_surfaces_deregister_timeout_with_custom_service_type_after_success();
    }

    #[test]
    fn test_run_distributed_surfaces_disconnect_timeout_with_default_service_type_after_success() {
        test_run_distributed_surfaces_disconnect_timeout_with_custom_service_type_after_success();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_and_index_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discover_failure_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_and_index_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_error_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_error_and_index_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_error_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_ordering_issue_timeout_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_and_index_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_timeout_and_index_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_does_not_hang_and_cleans_up() {
        test_run_distributed_worker_discover_timeout_does_not_hang_and_cleans_up();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_includes_custom_service_type_context() {
        test_run_distributed_worker_discover_timeout_includes_custom_service_type_context();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_custom_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_default_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_default_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_worker_role_last_discover_error_includes_service_type_context() {
        test_run_worker_role_discover_timeout_includes_service_type_context();
    }

    #[test]
    fn test_run_distributed_preserves_disconnect_timeout_with_custom_service_type_after_success() {
        test_run_distributed_surfaces_disconnect_timeout_with_custom_service_type_after_success();
    }

    #[test]
    fn test_run_distributed_preserves_disconnect_timeout_with_default_service_type_after_success() {
        test_run_distributed_surfaces_disconnect_timeout_with_default_service_type_after_success();
    }

    #[test]
    fn test_run_distributed_preserves_deregister_timeout_with_custom_service_type_after_success() {
        test_run_distributed_surfaces_deregister_timeout_with_custom_service_type_after_success();
    }

    #[test]
    fn test_run_distributed_preserves_deregister_timeout_with_default_service_type_after_success() {
        test_run_distributed_surfaces_deregister_timeout_with_default_service_type_after_success();
    }

    #[test]
    fn test_run_distributed_preserves_deregister_timeout_after_success() {
        test_run_distributed_surfaces_deregister_timeout_after_success();
    }

    #[test]
    fn test_run_distributed_preserves_disconnect_failure_after_success() {
        test_run_distributed_surfaces_disconnect_failure_after_success();
    }

    #[test]
    fn test_run_distributed_preserves_disconnect_timeout_after_success() {
        test_run_distributed_surfaces_disconnect_timeout_after_success();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_fails() {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_custom_service_type(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_default_service_type(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_default_service_type_and_index(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out() {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_custom_service_type(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_default_service_type(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index(
    ) {
        test_run_distributed_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_custom_service_type(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_default_service_type(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_fails_with_default_service_type_and_index(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_fails_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_custom_service_type(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_default_service_type(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_ps_connect_timeout_surfaces_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index(
    ) {
        test_run_distributed_ps_connect_timeout_preserves_error_when_disconnect_cleanup_times_out_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_when_cleanup_steps_fail() {
        test_run_distributed_preserves_ps_register_failure_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_ps_register_failure_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_ps_register_failure_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_ps_register_failure_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_and_index_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_timeout_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_and_index_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_timeout_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_timeout_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_timeout_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_timeout_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_timeout_with_default_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_ordering_issue_timeout_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_ordering_issue_timeout_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_issue_timeout_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_issue_timeout_with_default_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_ordering_and_discovery_error_timeout_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_ordering_and_discovery_error_timeout_with_default_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_fails_with_custom_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_fails_with_default_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_times_out_with_default_service_type(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_discover_timeout_surfaces_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_discover_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_fails_with_custom_service_type(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_fails_with_default_service_type(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_fails_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_times_out_with_default_service_type(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_last_discover_error_surfaces_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_last_discover_error_preserves_error_when_cleanup_times_out_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_and_index_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_discover_failure_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_and_index_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_discover_failure_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_and_index_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_error_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_and_index_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_error_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_error_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_error_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_discovery_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_discovery_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_discovery_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_discovery_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_types_when_cleanup_steps_fail()
    {
        test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_type_when_cleanup_steps_fail()
    {
        test_run_distributed_preserves_worker_error_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_types_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_error_with_default_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_error_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_error_with_default_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_register_failure_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_register_failure_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_register_failure_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_register_failure_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_fails() {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_fails_with_custom_service_type(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_fails_with_default_service_type(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_times_out() {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_times_out_with_default_service_type(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_ps_register_timeout_surfaces_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        test_run_distributed_ps_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_fails() {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_fails_with_custom_service_type(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_fails_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_fails_with_default_service_type(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_fails_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_fails_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_times_out() {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_times_out_with_custom_service_type(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_times_out_with_custom_service_type_and_index(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_custom_service_type_and_index();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_times_out_with_default_service_type(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type();
    }

    #[test]
    fn test_run_distributed_worker_register_timeout_surfaces_error_when_cleanup_times_out_with_default_service_type_and_index(
    ) {
        test_run_distributed_worker_register_timeout_preserves_error_when_cleanup_times_out_with_default_service_type_and_index();
    }

    #[test]
    fn test_run_worker_role_surfaces_last_discovery_error_across_retries() {
        test_run_worker_role_preserves_last_discovery_error_across_retries();
    }

    #[test]
    fn test_run_worker_role_surfaces_last_ordering_issue_across_retries() {
        test_run_worker_role_preserves_last_ordering_issue_across_retries();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_and_index_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_discover_failure_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_and_index_when_cleanup_steps_timeout()
    {
        test_run_distributed_preserves_worker_discover_failure_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_when_cleanup_steps_fail() {
        test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_when_cleanup_steps_timeout() {
        test_run_distributed_preserves_worker_discover_failure_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_preserves_worker_discovery_error_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_preserves_worker_discover_failure_with_default_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_and_index_when_cleanup_steps_fail() {
        test_run_distributed_surfaces_worker_discover_failure_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_and_index_when_cleanup_steps_timeout() {
        test_run_distributed_surfaces_worker_discover_failure_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_when_cleanup_steps_fail() {
        test_run_distributed_surfaces_worker_discover_failure_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_when_cleanup_steps_timeout() {
        test_run_distributed_surfaces_worker_discover_failure_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_custom_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_custom_service_types_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_type_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_type_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_type_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_type_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_type_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_types_and_index_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_types_and_index_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_and_index_when_cleanup_steps_timeout();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_types_when_cleanup_steps_fail(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_when_cleanup_steps_fail();
    }

    #[test]
    fn test_run_distributed_surfaces_worker_discovery_error_with_default_service_types_when_cleanup_steps_timeout(
    ) {
        test_run_distributed_surfaces_worker_discover_failure_with_default_service_types_when_cleanup_steps_timeout();
    }
}
