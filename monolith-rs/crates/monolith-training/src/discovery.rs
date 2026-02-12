//! Service discovery and distributed coordination.
//!
//! This module provides abstractions for service discovery in distributed training,
//! allowing workers and parameter servers to find each other dynamically.
//!
//! # Implementations
//!
//! - [`InMemoryDiscovery`]: An in-memory implementation for testing and single-node setups.
//! - [`ZkDiscovery`]: ZooKeeper-based discovery (feature: `zookeeper`).
//! - [`ConsulDiscovery`]: Consul-based discovery (feature: `consul`).
//!
//! # Example
//!
//! ```rust
//! use monolith_training::discovery::{
//!     ServiceDiscovery, ServiceInfo, HealthStatus, InMemoryDiscovery,
//! };
//! use std::collections::HashMap;
//!
//! let discovery = InMemoryDiscovery::new();
//!
//! // Register a service
//! let service = ServiceInfo::new(
//!     "ps-0",
//!     "parameter-server",
//!     "ps",
//!     "127.0.0.1",
//!     5000,
//! );
//! discovery.register(service).unwrap();
//!
//! // Discover services
//! let services = discovery.discover("ps").unwrap();
//! assert_eq!(services.len(), 1);
//! ```

use std::collections::HashMap;
#[cfg(any(feature = "zookeeper", feature = "consul", test))]
use std::future::Future;
#[cfg(any(feature = "zookeeper", feature = "consul"))]
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;
use tokio::sync::broadcast::{self, Receiver, Sender};

#[cfg(feature = "consul")]
use rs_consul as consul;
#[cfg(feature = "zookeeper")]
use zookeeper_client as zk;

/// Errors that can occur during service discovery operations.
#[derive(Debug, Error)]
pub enum DiscoveryError {
    /// The service was not found.
    #[error("Service not found: {0}")]
    NotFound(String),

    /// Failed to connect to the discovery backend.
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// The service is already registered.
    #[error("Service already registered: {0}")]
    AlreadyRegistered(String),

    /// A timeout occurred during the operation.
    #[error("Operation timed out: {0}")]
    Timeout(String),

    /// An internal error occurred.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Configuration error.
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Result type for discovery operations.
pub type Result<T> = std::result::Result<T, DiscoveryError>;

/// Health status of a service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum HealthStatus {
    /// The service is healthy and ready to receive traffic.
    Healthy,
    /// The service is unhealthy and should not receive traffic.
    Unhealthy,
    /// The health status is unknown.
    Unknown,
    /// The service is starting up.
    Starting,
    /// The service is shutting down.
    Stopping,
}

impl Default for HealthStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

/// Information about a registered service.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ServiceInfo {
    /// Unique identifier for this service instance.
    pub id: String,
    /// Human-readable name of the service.
    pub name: String,
    /// Type of service (e.g., "ps" for parameter server, "worker" for worker).
    pub service_type: String,
    /// Host address of the service.
    pub host: String,
    /// Port number of the service.
    pub port: u16,
    /// Additional metadata about the service.
    pub metadata: HashMap<String, String>,
    /// Current health status of the service.
    pub health: HealthStatus,
}

impl ServiceInfo {
    /// Creates a new service info with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `id` - Unique identifier for this service instance.
    /// * `name` - Human-readable name of the service.
    /// * `service_type` - Type of service.
    /// * `host` - Host address of the service.
    /// * `port` - Port number of the service.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        service_type: impl Into<String>,
        host: impl Into<String>,
        port: u16,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            service_type: service_type.into(),
            host: host.into(),
            port,
            metadata: HashMap::new(),
            health: HealthStatus::Unknown,
        }
    }

    /// Sets the health status.
    pub fn with_health(mut self, health: HealthStatus) -> Self {
        self.health = health;
        self
    }

    /// Adds metadata to the service info.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Returns the full address as a string.
    pub fn address(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Events emitted by service discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryEvent {
    /// A new service was added.
    ServiceAdded(ServiceInfo),
    /// A service was removed.
    ServiceRemoved(String),
    /// A service was updated.
    ServiceUpdated(ServiceInfo),
}

/// Trait for service discovery implementations.
///
/// This trait defines the interface for service discovery backends,
/// allowing services to register themselves, discover other services,
/// and watch for changes.
pub trait ServiceDiscovery: Send + Sync {
    /// Registers a service with the discovery backend.
    ///
    /// # Arguments
    ///
    /// * `service` - The service information to register.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the registration was successful, or an error otherwise.
    fn register(&self, service: ServiceInfo) -> Result<()>;

    /// Discovers all services of a given type.
    ///
    /// # Arguments
    ///
    /// * `service_type` - The type of services to discover.
    ///
    /// # Returns
    ///
    /// A list of services matching the given type.
    fn discover(&self, service_type: &str) -> Result<Vec<ServiceInfo>>;

    /// Watches for changes to services of a given type.
    ///
    /// # Arguments
    ///
    /// * `service_type` - The type of services to watch.
    ///
    /// # Returns
    ///
    /// A receiver that will receive discovery events.
    fn watch(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>>;

    /// Deregisters a service from the discovery backend.
    ///
    /// # Arguments
    ///
    /// * `service_id` - The ID of the service to deregister.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the deregistration was successful, or an error otherwise.
    fn deregister(&self, service_id: &str) -> Result<()>;
}

/// Async-friendly service discovery API (used by the distributed runner).
///
/// For local/in-memory discovery we can implement these methods directly.
/// For ZK/Consul, callers can either wrap blocking APIs in `spawn_blocking`
/// or provide native async implementations later.
#[async_trait::async_trait]
pub trait ServiceDiscoveryAsync: Send + Sync {
    async fn connect(&self) -> Result<()>;
    async fn disconnect(&self) -> Result<()>;

    async fn register_async(&self, service: ServiceInfo) -> Result<()>;
    async fn discover_async(&self, service_type: &str) -> Result<Vec<ServiceInfo>>;
    async fn watch_async(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>>;

    async fn deregister_async(&self, service_id: &str) -> Result<()>;

    /// Backend-specific keepalive/heartbeat. Default is a no-op for backends
    /// that do not require explicit heartbeats (e.g. in-memory, ZK ephemerals).
    async fn heartbeat_async(&self, _service_id: &str) -> Result<()> {
        Ok(())
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for InMemoryDiscovery {
    async fn connect(&self) -> Result<()> {
        Ok(())
    }

    async fn disconnect(&self) -> Result<()> {
        Ok(())
    }

    async fn register_async(&self, service: ServiceInfo) -> Result<()> {
        self.register(service)
    }

    async fn discover_async(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        self.discover(service_type)
    }

    async fn watch_async(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        self.watch(service_type)
    }

    async fn deregister_async(&self, service_id: &str) -> Result<()> {
        self.deregister(service_id)
    }
}

#[cfg(any(feature = "zookeeper", feature = "consul", test))]
fn spawn_watch_poll_loop<F, Fut, C>(
    sender: Sender<DiscoveryEvent>,
    backend: &'static str,
    poll_interval: std::time::Duration,
    should_continue: C,
    mut poll_discover: F,
) -> tokio::task::JoinHandle<()>
where
    F: FnMut() -> Fut + Send + 'static,
    Fut: Future<Output = Result<Vec<ServiceInfo>>> + Send + 'static,
    C: Fn() -> bool + Send + 'static,
{
    tokio::spawn(async move {
        let mut prev: HashMap<String, ServiceInfo> = HashMap::new();
        loop {
            // If no receivers are subscribed anymore, stop the poller to avoid
            // leaking long-lived background tasks.
            if sender.receiver_count() == 0 || !should_continue() {
                break;
            }

            let next_list = match poll_discover().await {
                Ok(v) => v,
                Err(e) => {
                    tracing::debug!(
                        backend = backend,
                        error = %e,
                        "Discovery watch poll discover failed"
                    );
                    if !should_continue() {
                        break;
                    }
                    tokio::time::sleep(poll_interval).await;
                    continue;
                }
            };

            let mut next: HashMap<String, ServiceInfo> = HashMap::new();
            for s in next_list {
                next.insert(s.id.clone(), s);
            }

            let mut should_stop = false;
            for (id, s) in next.iter() {
                let send_result = match prev.get(id) {
                    None => sender.send(DiscoveryEvent::ServiceAdded(s.clone())),
                    Some(prev_s) if prev_s != s => {
                        sender.send(DiscoveryEvent::ServiceUpdated(s.clone()))
                    }
                    Some(_) => continue,
                };
                if send_result.is_err() {
                    should_stop = true;
                    break;
                }
            }
            if !should_stop {
                for id in prev.keys() {
                    if !next.contains_key(id)
                        && sender.send(DiscoveryEvent::ServiceRemoved(id.clone())).is_err()
                    {
                        should_stop = true;
                        break;
                    }
                }
            }
            if should_stop {
                break;
            }

            prev = next;
            tokio::time::sleep(poll_interval).await;
        }
    })
}

/// In-memory service discovery implementation for testing.
///
/// This implementation stores all service information in memory and is
/// suitable for testing and single-node deployments.
///
/// # Example
///
/// ```rust
/// use monolith_training::discovery::{InMemoryDiscovery, ServiceInfo, ServiceDiscovery};
///
/// let discovery = InMemoryDiscovery::new();
///
/// let service = ServiceInfo::new("ps-0", "PS 0", "ps", "localhost", 5000);
/// discovery.register(service).unwrap();
///
/// let services = discovery.discover("ps").unwrap();
/// assert_eq!(services.len(), 1);
/// ```
pub struct InMemoryDiscovery {
    /// Registered services indexed by ID.
    services: RwLock<HashMap<String, ServiceInfo>>,
    /// Event senders for each service type.
    watchers: Mutex<HashMap<String, Sender<DiscoveryEvent>>>,
}

impl InMemoryDiscovery {
    /// Creates a new in-memory discovery instance.
    pub fn new() -> Self {
        Self {
            services: RwLock::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the number of registered services.
    pub fn len(&self) -> usize {
        self.services.read().unwrap().len()
    }

    /// Returns true if no services are registered.
    pub fn is_empty(&self) -> bool {
        self.services.read().unwrap().is_empty()
    }

    /// Clears all registered services.
    pub fn clear(&self) {
        let mut services = self.services.write().unwrap();
        let service_ids: Vec<String> = services.keys().cloned().collect();

        for id in service_ids {
            if let Some(service) = services.remove(&id) {
                self.notify_watchers(&service.service_type, DiscoveryEvent::ServiceRemoved(id));
            }
        }
    }

    /// Updates the health status of a service.
    pub fn update_health(&self, service_id: &str, health: HealthStatus) -> Result<()> {
        let mut services = self.services.write().unwrap();
        let service = services
            .get_mut(service_id)
            .ok_or_else(|| DiscoveryError::NotFound(service_id.to_string()))?;

        service.health = health;
        let updated_service = service.clone();
        let service_type = service.service_type.clone();
        drop(services);

        self.notify_watchers(
            &service_type,
            DiscoveryEvent::ServiceUpdated(updated_service),
        );
        Ok(())
    }

    /// Notifies watchers of an event.
    fn notify_watchers(&self, service_type: &str, event: DiscoveryEvent) {
        let mut watchers = self.watchers.lock().unwrap();
        if let Some(sender) = watchers.get(service_type) {
            if sender.receiver_count() == 0 || sender.send(event).is_err() {
                // No active subscribers for this service type anymore.
                watchers.remove(service_type);
            }
        }
    }

    /// Gets or creates a sender for a service type.
    fn get_or_create_sender(&self, service_type: &str) -> Sender<DiscoveryEvent> {
        let mut watchers = self.watchers.lock().unwrap();
        watchers
            .entry(service_type.to_string())
            .or_insert_with(|| broadcast::channel(100).0)
            .clone()
    }
}

impl Default for InMemoryDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl ServiceDiscovery for InMemoryDiscovery {
    fn register(&self, service: ServiceInfo) -> Result<()> {
        let mut services = self.services.write().unwrap();

        let is_update = services.contains_key(&service.id);
        // Default behavior: duplicate registration is an error. We only allow updates
        // when the caller explicitly marks the registration as idempotent.
        //
        // This keeps existing semantics/tests intact while still enabling the runner
        // to re-register after binding to an ephemeral port.
        let allow_update = service
            .metadata
            .get("allow_update")
            .map(|v| v == "true")
            .unwrap_or(false);
        if is_update && !allow_update {
            return Err(DiscoveryError::AlreadyRegistered(service.id.clone()));
        }

        let service_type = service.service_type.clone();
        let service_clone = service.clone();
        services.insert(service.id.clone(), service);
        drop(services);

        tracing::info!(
            service_id = %service_clone.id,
            service_type = %service_type,
            address = %service_clone.address(),
            "Registered service"
        );

        if is_update {
            self.notify_watchers(&service_type, DiscoveryEvent::ServiceUpdated(service_clone));
        } else {
            self.notify_watchers(&service_type, DiscoveryEvent::ServiceAdded(service_clone));
        }
        Ok(())
    }

    fn discover(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        let services = self.services.read().unwrap();
        let matching: Vec<ServiceInfo> = services
            .values()
            .filter(|s| s.service_type == service_type)
            .cloned()
            .collect();

        tracing::debug!(
            service_type = %service_type,
            count = matching.len(),
            "Discovered services"
        );

        Ok(matching)
    }

    fn watch(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        let sender = self.get_or_create_sender(service_type);
        Ok(sender.subscribe())
    }

    fn deregister(&self, service_id: &str) -> Result<()> {
        let mut services = self.services.write().unwrap();

        let service = services
            .remove(service_id)
            .ok_or_else(|| DiscoveryError::NotFound(service_id.to_string()))?;

        let service_type = service.service_type.clone();
        drop(services);

        tracing::info!(
            service_id = %service_id,
            service_type = %service_type,
            "Deregistered service"
        );

        self.notify_watchers(
            &service_type,
            DiscoveryEvent::ServiceRemoved(service_id.to_string()),
        );
        Ok(())
    }
}

/// ZooKeeper-based service discovery.
///
/// This is a stub implementation that provides the interface for ZooKeeper-based
/// service discovery. The actual implementation requires the `zookeeper` feature.
///
/// # Configuration
///
/// ZooKeeper discovery requires the following configuration:
/// - `hosts`: Comma-separated list of ZooKeeper hosts
/// - `base_path`: Base path for service registration (default: `/services`)
/// - `session_timeout`: Session timeout in milliseconds (default: 30000)
#[cfg(feature = "zookeeper")]
pub struct ZkDiscovery {
    /// ZooKeeper connection hosts.
    hosts: String,
    /// Base path for service registration.
    base_path: String,
    /// Session timeout in milliseconds.
    session_timeout_ms: u64,
    /// Connected ZooKeeper client (set on `connect()`).
    client: tokio::sync::Mutex<Option<zk::Client>>,
    /// Paths for ephemerals registered by this process (keyed by service_id).
    registered_paths: tokio::sync::Mutex<HashMap<String, String>>,
    /// In-memory cache of services.
    services: RwLock<HashMap<String, ServiceInfo>>,
    /// Event senders for watchers.
    watchers: Mutex<HashMap<String, Sender<DiscoveryEvent>>>,
    /// Generation counter for watch lifecycle control.
    watch_generation: Arc<AtomicU64>,
    /// Active watch-poll generations keyed by service type.
    watch_poll_generations: Mutex<HashMap<String, u64>>,
}

#[cfg(feature = "zookeeper")]
impl ZkDiscovery {
    /// Creates a new ZooKeeper discovery instance.
    ///
    /// # Arguments
    ///
    /// * `hosts` - Comma-separated list of ZooKeeper hosts.
    /// * `base_path` - Base path for service registration.
    pub fn new(hosts: impl Into<String>, base_path: impl Into<String>) -> Self {
        Self {
            hosts: hosts.into(),
            base_path: base_path.into(),
            session_timeout_ms: 30000,
            client: tokio::sync::Mutex::new(None),
            registered_paths: tokio::sync::Mutex::new(HashMap::new()),
            services: RwLock::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
            watch_generation: Arc::new(AtomicU64::new(0)),
            watch_poll_generations: Mutex::new(HashMap::new()),
        }
    }

    /// Sets the session timeout.
    pub fn with_session_timeout(mut self, timeout_ms: u64) -> Self {
        self.session_timeout_ms = timeout_ms;
        self
    }

    /// Connects to ZooKeeper.
    ///
    /// This is a placeholder that would establish a connection to ZooKeeper.
    pub async fn connect(&self) -> Result<()> {
        tracing::info!(hosts = %self.hosts, base_path = %self.base_path, "Connecting to ZooKeeper");

        let mut guard = self.client.lock().await;
        if guard.is_some() {
            return Ok(());
        }

        let client = zk::Client::connector()
            .with_session_timeout(std::time::Duration::from_millis(self.session_timeout_ms))
            .connect(&self.hosts)
            .await
            .map_err(|e| DiscoveryError::ConnectionFailed(format!("ZK connect failed: {e}")))?;

        // Ensure base_path exists.
        // ZK has no "mkdir -p" primitive; create parents best-effort.
        ensure_zk_path(&client, &self.base_path).await?;

        *guard = Some(client);
        Ok(())
    }

    /// Disconnects from ZooKeeper.
    pub async fn disconnect(&self) -> Result<()> {
        tracing::info!("Disconnecting from ZooKeeper");

        // Drop the client, which closes the session and cleans ephemerals.
        let mut guard = self.client.lock().await;
        *guard = None;
        self.registered_paths.lock().await.clear();
        self.watch_generation.fetch_add(1, Ordering::SeqCst);
        self.watch_poll_generations.lock().unwrap().clear();
        Ok(())
    }

    /// Gets or creates a sender for a service type.
    fn get_or_create_sender(&self, service_type: &str) -> Sender<DiscoveryEvent> {
        let mut watchers = self.watchers.lock().unwrap();
        watchers
            .entry(service_type.to_string())
            .or_insert_with(|| broadcast::channel(100).0)
            .clone()
    }

    /// Notifies watchers for a service type and removes dead sender entries.
    fn notify_watchers(&self, service_type: &str, event: DiscoveryEvent) {
        let mut watchers = self.watchers.lock().unwrap();
        if let Some(sender) = watchers.get(service_type) {
            if sender.receiver_count() == 0 || sender.send(event).is_err() {
                watchers.remove(service_type);
            }
        }
    }

    /// Returns true only when a new poll loop should be spawned for the service type.
    fn should_spawn_watch_poll(&self, service_type: &str, receiver_count: usize) -> bool {
        let generation = self.watch_generation.load(Ordering::SeqCst);
        let mut active = self.watch_poll_generations.lock().unwrap();
        match active.get(service_type).copied() {
            Some(g) if g == generation && receiver_count > 1 => false,
            _ => {
                active.insert(service_type.to_string(), generation);
                true
            }
        }
    }
}

#[cfg(feature = "zookeeper")]
impl ServiceDiscovery for ZkDiscovery {
    fn register(&self, service: ServiceInfo) -> Result<()> {
        tracing::info!(
            service_id = %service.id,
            service_type = %service.service_type,
            "Registering service with ZooKeeper"
        );

        // Keep sync API best-effort: update local cache only. The distributed runner uses
        // `ServiceDiscoveryAsync` for real ZK I/O.
        let mut services = self.services.write().unwrap();
        if services.contains_key(&service.id) {
            return Err(DiscoveryError::AlreadyRegistered(service.id.clone()));
        }
        services.insert(service.id.clone(), service.clone());

        let service_type = service.service_type.clone();
        self.notify_watchers(&service_type, DiscoveryEvent::ServiceAdded(service));
        Ok(())
    }

    fn discover(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        tracing::debug!(
            service_type = %service_type,
            "Discovering services from ZooKeeper"
        );

        // Sync API returns from cache only; see `discover_async` for real ZK query.
        let services = self.services.read().unwrap();
        Ok(services
            .values()
            .filter(|s| s.service_type == service_type)
            .cloned()
            .collect())
    }

    fn watch(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        tracing::debug!(
            service_type = %service_type,
            "Setting up ZooKeeper watch"
        );

        let sender = self.get_or_create_sender(service_type);
        Ok(sender.subscribe())
    }

    fn deregister(&self, service_id: &str) -> Result<()> {
        tracing::info!(
            service_id = %service_id,
            "Deregistering service from ZooKeeper"
        );

        // Cache-only removal for sync API.
        let mut services = self.services.write().unwrap();
        let service = services
            .remove(service_id)
            .ok_or_else(|| DiscoveryError::NotFound(service_id.to_string()))?;
        drop(services);

        self.notify_watchers(
            &service.service_type,
            DiscoveryEvent::ServiceRemoved(service_id.to_string()),
        );
        Ok(())
    }
}

#[cfg(feature = "zookeeper")]
#[async_trait::async_trait]
impl ServiceDiscoveryAsync for ZkDiscovery {
    async fn connect(&self) -> Result<()> {
        ZkDiscovery::connect(self).await
    }

    async fn disconnect(&self) -> Result<()> {
        ZkDiscovery::disconnect(self).await
    }

    async fn register_async(&self, service: ServiceInfo) -> Result<()> {
        self.connect().await?;

        let client =
            self.client.lock().await.as_ref().cloned().ok_or_else(|| {
                DiscoveryError::ConnectionFailed("ZK client not connected".into())
            })?;

        let idx = service
            .metadata
            .get("index")
            .and_then(|s| s.parse::<i32>().ok())
            .or_else(|| {
                service
                    .id
                    .rsplit_once('-')
                    .and_then(|(_, n)| n.parse::<i32>().ok())
            })
            .unwrap_or(0);

        // Match Python: /monolith/<job>/<service_type>.<index> with payload "host:port".
        let path = format!(
            "{}/{}.{}",
            self.base_path.trim_end_matches('/'),
            service.service_type,
            idx
        );
        let data = service.address().into_bytes();

        // Create ephemeral node with payload "host:port". If exists, set_data.
        let create_opts = zk::CreateMode::Ephemeral.with_acls(zk::Acls::anyone_all());
        match client.create(&path, &data, &create_opts).await {
            Ok(_) => {}
            Err(zk::Error::NodeExists) => {
                client
                    .set_data(&path, &data, None)
                    .await
                    .map_err(|e| DiscoveryError::Internal(format!("ZK set_data failed: {e}")))?;
            }
            Err(e) => {
                return Err(DiscoveryError::Internal(format!("ZK create failed: {e}")));
            }
        };

        self.registered_paths
            .lock()
            .await
            .insert(service.id.clone(), path.clone());

        // Update local cache (for quick discover + tests).
        self.services
            .write()
            .unwrap()
            .insert(service.id.clone(), service.clone());
        let service_type = service.service_type.clone();
        self.notify_watchers(&service_type, DiscoveryEvent::ServiceAdded(service));
        Ok(())
    }

    async fn discover_async(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        self.connect().await?;
        let client =
            self.client.lock().await.as_ref().cloned().ok_or_else(|| {
                DiscoveryError::ConnectionFailed("ZK client not connected".into())
            })?;

        let base = self.base_path.trim_end_matches('/');
        let children = client
            .list_children(base)
            .await
            .map_err(|e| DiscoveryError::Internal(format!("ZK list_children failed: {e}")))?;

        let mut out = Vec::new();
        for child in children {
            let name = child;
            if !name.starts_with(&format!("{}.", service_type)) {
                continue;
            }
            let path = format!("{}/{}", base, name);
            if let Ok((data, _stat)) = client.get_data(&path).await {
                if let Ok(addr) = String::from_utf8(data) {
                    if let Some((host, port_str)) = addr.split_once(':') {
                        if let Ok(port) = port_str.parse::<u16>() {
                            let idx = name
                                .trim_start_matches(&format!("{}.", service_type))
                                .to_string();
                            let id = format!("{}-{}", service_type, idx);
                            let mut svc =
                                ServiceInfo::new(id.clone(), id.clone(), service_type, host, port);
                            svc.metadata.insert("addr".into(), addr.clone());
                            svc.metadata.insert("index".into(), idx);
                            out.push(svc);
                        }
                    }
                }
            }
        }

        // Keep cache in sync.
        {
            let mut cache = self.services.write().unwrap();
            cache.retain(|_, v| v.service_type != service_type);
            for svc in &out {
                cache.insert(svc.id.clone(), svc.clone());
            }
        }

        Ok(out)
    }

    async fn watch_async(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        // Poll-based watcher: keeps parity with Python's callback-based watchers without
        // relying on ZK persistent watch semantics (which can be lossy during reconnect).
        let sender = self.get_or_create_sender(service_type);
        let rx = sender.subscribe();
        let receiver_count = sender.receiver_count();

        if self.should_spawn_watch_poll(service_type, receiver_count) {
            let svc_type = service_type.to_string();
            let this = Arc::new(self.clone_for_watch());
            let sender_for_poll = sender.clone();
            let watch_generation = Arc::clone(&self.watch_generation);
            let generation = watch_generation.load(Ordering::SeqCst);
            spawn_watch_poll_loop(
                sender_for_poll,
                "zk",
                std::time::Duration::from_secs(1),
                move || watch_generation.load(Ordering::SeqCst) == generation,
                move || {
                    let this = Arc::clone(&this);
                    let svc_type = svc_type.clone();
                    async move { this.discover_async(&svc_type).await }
                },
            );
        }

        Ok(rx)
    }

    async fn deregister_async(&self, service_id: &str) -> Result<()> {
        self.connect().await?;
        let client =
            self.client.lock().await.as_ref().cloned().ok_or_else(|| {
                DiscoveryError::ConnectionFailed("ZK client not connected".into())
            })?;

        let path_opt = self.registered_paths.lock().await.remove(service_id);
        if let Some(path) = path_opt {
            let _ = client.delete(&path, None).await;
        }

        self.services.write().unwrap().remove(service_id);
        Ok(())
    }
}

/// Consul-based service discovery.
///
/// This is a stub implementation that provides the interface for Consul-based
/// service discovery. The actual implementation requires the `consul` feature.
///
/// # Configuration
///
/// Consul discovery requires the following configuration:
/// - `address`: Consul agent address (default: `http://localhost:8500`)
/// - `datacenter`: Datacenter name (optional)
/// - `token`: ACL token for authentication (optional)
#[cfg(feature = "consul")]
pub struct ConsulDiscovery {
    /// Consul agent address.
    address: String,
    /// Datacenter name.
    datacenter: Option<String>,
    /// ACL token for authentication.
    token: Option<String>,
    /// Connected Consul client.
    client: tokio::sync::Mutex<Option<Arc<consul::Consul>>>,
    /// Consul "service name" used for discovery.
    service_name: String,
    /// In-memory cache of services.
    services: RwLock<HashMap<String, ServiceInfo>>,
    /// Event senders for watchers.
    watchers: Mutex<HashMap<String, Sender<DiscoveryEvent>>>,
    /// Generation counter for watch lifecycle control.
    watch_generation: Arc<AtomicU64>,
    /// Active watch-poll generations keyed by service type.
    watch_poll_generations: Mutex<HashMap<String, u64>>,
}

#[cfg(feature = "consul")]
impl ConsulDiscovery {
    /// Creates a new Consul discovery instance.
    ///
    /// # Arguments
    ///
    /// * `address` - Consul agent address.
    pub fn new(address: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            datacenter: None,
            token: None,
            client: tokio::sync::Mutex::new(None),
            service_name: "monolith".to_string(),
            services: RwLock::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
            watch_generation: Arc::new(AtomicU64::new(0)),
            watch_poll_generations: Mutex::new(HashMap::new()),
        }
    }

    /// Sets the datacenter.
    pub fn with_datacenter(mut self, datacenter: impl Into<String>) -> Self {
        self.datacenter = Some(datacenter.into());
        self
    }

    /// Sets the ACL token.
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Sets the Consul service name used for registration/discovery.
    pub fn with_service_name(mut self, service_name: impl Into<String>) -> Self {
        self.service_name = service_name.into();
        self
    }

    /// Connects to Consul.
    ///
    /// This is a placeholder that would establish a connection to Consul.
    pub async fn connect(&self) -> Result<()> {
        tracing::info!(
            address = %self.address,
            datacenter = ?self.datacenter,
            "Connecting to Consul"
        );
        let mut guard = self.client.lock().await;
        if guard.is_some() {
            return Ok(());
        }
        let mut cfg = consul::Config {
            address: self.address.clone(),
            token: self.token.clone(),
            ..Default::default()
        };
        // `rs-consul` uses QueryOptions for datacenter; keep here for parity.
        if cfg.address.is_empty() {
            cfg.address = "http://127.0.0.1:8500".to_string();
        }
        *guard = Some(Arc::new(consul::Consul::new(cfg)));
        Ok(())
    }

    /// Gets or creates a sender for a service type.
    fn get_or_create_sender(&self, service_type: &str) -> Sender<DiscoveryEvent> {
        let mut watchers = self.watchers.lock().unwrap();
        watchers
            .entry(service_type.to_string())
            .or_insert_with(|| broadcast::channel(100).0)
            .clone()
    }

    /// Notifies watchers for a service type and removes dead sender entries.
    fn notify_watchers(&self, service_type: &str, event: DiscoveryEvent) {
        let mut watchers = self.watchers.lock().unwrap();
        if let Some(sender) = watchers.get(service_type) {
            if sender.receiver_count() == 0 || sender.send(event).is_err() {
                watchers.remove(service_type);
            }
        }
    }

    /// Returns true only when a new poll loop should be spawned for the service type.
    fn should_spawn_watch_poll(&self, service_type: &str, receiver_count: usize) -> bool {
        let generation = self.watch_generation.load(Ordering::SeqCst);
        let mut active = self.watch_poll_generations.lock().unwrap();
        match active.get(service_type).copied() {
            Some(g) if g == generation && receiver_count > 1 => false,
            _ => {
                active.insert(service_type.to_string(), generation);
                true
            }
        }
    }
}

#[cfg(feature = "consul")]
impl ServiceDiscovery for ConsulDiscovery {
    fn register(&self, service: ServiceInfo) -> Result<()> {
        tracing::info!(
            service_id = %service.id,
            service_type = %service.service_type,
            "Registering service with Consul"
        );

        // Cache-only for sync API.
        let mut services = self.services.write().unwrap();
        if services.contains_key(&service.id) {
            return Err(DiscoveryError::AlreadyRegistered(service.id.clone()));
        }
        services.insert(service.id.clone(), service.clone());
        let service_type = service.service_type.clone();
        self.notify_watchers(&service_type, DiscoveryEvent::ServiceAdded(service));
        Ok(())
    }

    fn discover(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        tracing::debug!(
            service_type = %service_type,
            "Discovering services from Consul"
        );

        // Cache-only for sync API.
        let services = self.services.read().unwrap();
        Ok(services
            .values()
            .filter(|s| s.service_type == service_type)
            .cloned()
            .collect())
    }

    fn watch(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        tracing::debug!(
            service_type = %service_type,
            "Setting up Consul watch"
        );

        let sender = self.get_or_create_sender(service_type);
        Ok(sender.subscribe())
    }

    fn deregister(&self, service_id: &str) -> Result<()> {
        tracing::info!(
            service_id = %service_id,
            "Deregistering service from Consul"
        );

        let mut services = self.services.write().unwrap();
        let service = services
            .remove(service_id)
            .ok_or_else(|| DiscoveryError::NotFound(service_id.to_string()))?;
        drop(services);

        self.notify_watchers(
            &service.service_type,
            DiscoveryEvent::ServiceRemoved(service_id.to_string()),
        );
        Ok(())
    }
}

#[cfg(feature = "consul")]
#[async_trait::async_trait]
impl ServiceDiscoveryAsync for ConsulDiscovery {
    async fn connect(&self) -> Result<()> {
        ConsulDiscovery::connect(self).await
    }

    async fn disconnect(&self) -> Result<()> {
        let mut guard = self.client.lock().await;
        *guard = None;
        self.watch_generation.fetch_add(1, Ordering::SeqCst);
        self.watch_poll_generations.lock().unwrap().clear();
        Ok(())
    }

    async fn register_async(&self, service: ServiceInfo) -> Result<()> {
        self.connect().await?;
        let client = self.client.lock().await.as_ref().cloned().ok_or_else(|| {
            DiscoveryError::ConnectionFailed("Consul client not connected".into())
        })?;

        // Use the global catalog register endpoint (HashiCorp Consul).
        //
        // NOTE: Python's `monolith/native_training/consul.py` uses a ByteDance-specific
        // lookup API. This Rust implementation targets stock Consul deployments.
        let tags = vec![
            format!("name:{}", service.service_type),
            format!(
                "index:{}",
                service.metadata.get("index").cloned().unwrap_or_default()
            ),
            format!("ip:{}", service.host),
        ];

        let svc = consul::types::RegisterEntityService {
            ID: Some(service.id.clone()),
            Service: self.service_name.clone(),
            Tags: tags,
            TaggedAddresses: HashMap::new(),
            Meta: HashMap::new(),
            Port: Some(service.port),
            Namespace: None,
        };

        let payload = consul::types::RegisterEntityPayload {
            ID: None,
            Node: service.id.clone(),
            Address: service.host.clone(),
            Datacenter: self.datacenter.clone(),
            TaggedAddresses: HashMap::new(),
            NodeMeta: HashMap::new(),
            Service: Some(svc),
            Checks: Vec::new(),
            SkipNodeUpdate: Some(true),
        };

        client.register_entity(&payload).await.map_err(|e| {
            DiscoveryError::Internal(format!("Consul register_entity failed: {e:?}"))
        })?;

        self.services
            .write()
            .unwrap()
            .insert(service.id.clone(), service.clone());
        let service_type = service.service_type.clone();
        self.notify_watchers(&service_type, DiscoveryEvent::ServiceAdded(service));
        Ok(())
    }

    async fn discover_async(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        self.connect().await?;
        let client = self.client.lock().await.as_ref().cloned().ok_or_else(|| {
            DiscoveryError::ConnectionFailed("Consul client not connected".into())
        })?;

        let consul_service = self.service_name.as_str();
        let nodes = client
            .get_service_nodes(
                consul::types::GetServiceNodesRequest {
                    service: consul_service,
                    passing: true,
                    ..Default::default()
                },
                None,
            )
            .await
            .map_err(|e| {
                DiscoveryError::Internal(format!("Consul get_service_nodes failed: {e:?}"))
            })?;

        let mut out = Vec::new();
        for sn in nodes.response {
            // Filter by tags for this service_type (we stored as "name:<type>").
            if !sn
                .service
                .tags
                .iter()
                .any(|t| t == &format!("name:{}", service_type))
            {
                continue;
            }

            let host = if sn.service.address.is_empty() {
                sn.node.address
            } else {
                sn.service.address
            };
            let port = sn.service.port;
            let id = sn.service.id;

            let mut svc = ServiceInfo::new(id.clone(), id.clone(), service_type, host, port);
            svc.metadata.insert("addr".into(), svc.address());
            out.push(svc);
        }

        Ok(out)
    }

    async fn watch_async(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        // Poll-based watcher to avoid depending on Consul long-poll semantics here.
        let sender = self.get_or_create_sender(service_type);
        let rx = sender.subscribe();
        let receiver_count = sender.receiver_count();

        if self.should_spawn_watch_poll(service_type, receiver_count) {
            let svc_type = service_type.to_string();
            let this = Arc::new(self.clone_for_watch());
            let sender_for_poll = sender.clone();
            let watch_generation = Arc::clone(&self.watch_generation);
            let generation = watch_generation.load(Ordering::SeqCst);
            spawn_watch_poll_loop(
                sender_for_poll,
                "consul",
                std::time::Duration::from_secs(1),
                move || watch_generation.load(Ordering::SeqCst) == generation,
                move || {
                    let this = Arc::clone(&this);
                    let svc_type = svc_type.clone();
                    async move { this.discover_async(&svc_type).await }
                },
            );
        }

        Ok(rx)
    }

    async fn deregister_async(&self, service_id: &str) -> Result<()> {
        self.connect().await?;
        let client = self.client.lock().await.as_ref().cloned().ok_or_else(|| {
            DiscoveryError::ConnectionFailed("Consul client not connected".into())
        })?;

        let payload = consul::types::DeregisterEntityPayload {
            Node: None,
            Datacenter: self.datacenter.clone(),
            CheckID: None,
            ServiceID: Some(service_id.to_string()),
            Namespace: None,
        };

        let _ = client.deregister_entity(&payload).await;
        self.services.write().unwrap().remove(service_id);
        Ok(())
    }
}

// ============================================================================
// Helpers
// ============================================================================

#[cfg(feature = "zookeeper")]
async fn ensure_zk_path(client: &zk::Client, path: &str) -> Result<()> {
    let mut cur = String::new();
    for part in path.split('/').filter(|p| !p.is_empty()) {
        cur.push('/');
        cur.push_str(part);
        let opts = zk::CreateMode::Persistent.with_acls(zk::Acls::anyone_all());
        let _ = client.create(&cur, b"", &opts).await;
    }
    Ok(())
}

#[cfg(feature = "zookeeper")]
impl ZkDiscovery {
    // Small helper so we can spawn watch tasks without requiring `ZkDiscovery: Clone`.
    fn clone_for_watch(&self) -> Self {
        Self {
            hosts: self.hosts.clone(),
            base_path: self.base_path.clone(),
            session_timeout_ms: self.session_timeout_ms,
            client: tokio::sync::Mutex::new(None),
            registered_paths: tokio::sync::Mutex::new(HashMap::new()),
            services: RwLock::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
            watch_generation: Arc::clone(&self.watch_generation),
            watch_poll_generations: Mutex::new(HashMap::new()),
        }
    }
}

#[cfg(feature = "consul")]
impl ConsulDiscovery {
    fn clone_for_watch(&self) -> Self {
        Self {
            address: self.address.clone(),
            datacenter: self.datacenter.clone(),
            token: self.token.clone(),
            client: tokio::sync::Mutex::new(None),
            service_name: self.service_name.clone(),
            services: RwLock::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
            watch_generation: Arc::clone(&self.watch_generation),
            watch_poll_generations: Mutex::new(HashMap::new()),
        }
    }
}

/// A thread-safe wrapper around a service discovery implementation.
pub type SharedDiscovery = Arc<dyn ServiceDiscovery>;

/// Creates a new shared in-memory discovery instance.
pub fn new_in_memory_discovery() -> SharedDiscovery {
    Arc::new(InMemoryDiscovery::new())
}

/// Creates a new shared ZooKeeper discovery instance.
#[cfg(feature = "zookeeper")]
pub fn new_zk_discovery(hosts: impl Into<String>, base_path: impl Into<String>) -> SharedDiscovery {
    Arc::new(ZkDiscovery::new(hosts, base_path))
}

/// Creates a new shared Consul discovery instance.
#[cfg(feature = "consul")]
pub fn new_consul_discovery(address: impl Into<String>) -> SharedDiscovery {
    Arc::new(ConsulDiscovery::new(address))
}

/// Creates a new shared Consul discovery instance with a custom service name.
///
/// In practice, `service_name` should be a stable job identifier (e.g. `"monolith-job"`).
#[cfg(feature = "consul")]
pub fn new_consul_discovery_with_service_name(
    address: impl Into<String>,
    service_name: impl Into<String>,
) -> SharedDiscovery {
    Arc::new(ConsulDiscovery::new(address).with_service_name(service_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_service_info_creation() {
        let service = ServiceInfo::new("ps-0", "Parameter Server 0", "ps", "127.0.0.1", 5000);

        assert_eq!(service.id, "ps-0");
        assert_eq!(service.name, "Parameter Server 0");
        assert_eq!(service.service_type, "ps");
        assert_eq!(service.host, "127.0.0.1");
        assert_eq!(service.port, 5000);
        assert_eq!(service.health, HealthStatus::Unknown);
        assert!(service.metadata.is_empty());
    }

    #[test]
    fn test_service_info_with_metadata() {
        let service = ServiceInfo::new("worker-0", "Worker 0", "worker", "192.168.1.1", 6000)
            .with_health(HealthStatus::Healthy)
            .with_metadata("gpu_count", "2")
            .with_metadata("memory_gb", "32");

        assert_eq!(service.health, HealthStatus::Healthy);
        assert_eq!(service.metadata.get("gpu_count"), Some(&"2".to_string()));
        assert_eq!(service.metadata.get("memory_gb"), Some(&"32".to_string()));
    }

    #[test]
    fn test_service_info_address() {
        let service = ServiceInfo::new("test", "Test", "test", "10.0.0.1", 8080);
        assert_eq!(service.address(), "10.0.0.1:8080");
    }

    #[test]
    fn test_in_memory_register_and_discover() {
        let discovery = InMemoryDiscovery::new();

        // Register a parameter server
        let ps = ServiceInfo::new("ps-0", "PS 0", "ps", "localhost", 5000)
            .with_health(HealthStatus::Healthy);
        discovery.register(ps).unwrap();

        // Register workers
        let worker1 = ServiceInfo::new("worker-0", "Worker 0", "worker", "localhost", 6000);
        let worker2 = ServiceInfo::new("worker-1", "Worker 1", "worker", "localhost", 6001);
        discovery.register(worker1).unwrap();
        discovery.register(worker2).unwrap();

        // Discover by type
        let ps_services = discovery.discover("ps").unwrap();
        assert_eq!(ps_services.len(), 1);
        assert_eq!(ps_services[0].id, "ps-0");

        let worker_services = discovery.discover("worker").unwrap();
        assert_eq!(worker_services.len(), 2);

        // Discover non-existent type
        let empty = discovery.discover("nonexistent").unwrap();
        assert!(empty.is_empty());
    }

    #[test]
    fn test_in_memory_deregister() {
        let discovery = InMemoryDiscovery::new();

        let service = ServiceInfo::new("test-1", "Test 1", "test", "localhost", 8000);
        discovery.register(service).unwrap();

        assert_eq!(discovery.len(), 1);

        discovery.deregister("test-1").unwrap();
        assert_eq!(discovery.len(), 0);

        // Deregister non-existent service should fail
        let result = discovery.deregister("nonexistent");
        assert!(matches!(result, Err(DiscoveryError::NotFound(_))));
    }

    #[test]
    fn test_in_memory_duplicate_registration() {
        let discovery = InMemoryDiscovery::new();

        let service = ServiceInfo::new("dup-1", "Duplicate", "test", "localhost", 8000);
        discovery.register(service.clone()).unwrap();

        // Second registration should fail
        let result = discovery.register(service);
        assert!(matches!(result, Err(DiscoveryError::AlreadyRegistered(_))));
    }

    #[test]
    fn test_in_memory_update_health() {
        let discovery = InMemoryDiscovery::new();

        let service = ServiceInfo::new("health-test", "Health Test", "test", "localhost", 8000)
            .with_health(HealthStatus::Starting);
        discovery.register(service).unwrap();

        // Update health
        discovery
            .update_health("health-test", HealthStatus::Healthy)
            .unwrap();

        let services = discovery.discover("test").unwrap();
        assert_eq!(services[0].health, HealthStatus::Healthy);

        // Update to unhealthy
        discovery
            .update_health("health-test", HealthStatus::Unhealthy)
            .unwrap();

        let services = discovery.discover("test").unwrap();
        assert_eq!(services[0].health, HealthStatus::Unhealthy);

        // Update non-existent service should fail
        let result = discovery.update_health("nonexistent", HealthStatus::Healthy);
        assert!(matches!(result, Err(DiscoveryError::NotFound(_))));
    }

    #[test]
    fn test_in_memory_clear() {
        let discovery = InMemoryDiscovery::new();

        discovery
            .register(ServiceInfo::new("s1", "S1", "test", "localhost", 8001))
            .unwrap();
        discovery
            .register(ServiceInfo::new("s2", "S2", "test", "localhost", 8002))
            .unwrap();
        discovery
            .register(ServiceInfo::new("s3", "S3", "other", "localhost", 8003))
            .unwrap();

        assert_eq!(discovery.len(), 3);

        discovery.clear();

        assert_eq!(discovery.len(), 0);
        assert!(discovery.is_empty());
    }

    #[tokio::test]
    async fn test_in_memory_watch() {
        let discovery = InMemoryDiscovery::new();

        // Set up a watcher before registration
        let mut receiver = discovery.watch("ps").unwrap();

        // Register a service
        let service = ServiceInfo::new("ps-watch", "PS Watch", "ps", "localhost", 5000);
        discovery.register(service.clone()).unwrap();

        // Check that we received the event
        let event = receiver.recv().await.unwrap();
        match event {
            DiscoveryEvent::ServiceAdded(s) => {
                assert_eq!(s.id, "ps-watch");
            }
            _ => panic!("Expected ServiceAdded event"),
        }

        // Deregister and check for removal event
        discovery.deregister("ps-watch").unwrap();

        let event = receiver.recv().await.unwrap();
        match event {
            DiscoveryEvent::ServiceRemoved(id) => {
                assert_eq!(id, "ps-watch");
            }
            _ => panic!("Expected ServiceRemoved event"),
        }
    }

    #[tokio::test]
    async fn test_in_memory_watch_update() {
        let discovery = InMemoryDiscovery::new();

        let service = ServiceInfo::new("update-test", "Update Test", "worker", "localhost", 6000)
            .with_health(HealthStatus::Starting);
        discovery.register(service).unwrap();

        let mut receiver = discovery.watch("worker").unwrap();

        // Update health status
        discovery
            .update_health("update-test", HealthStatus::Healthy)
            .unwrap();

        let event = receiver.recv().await.unwrap();
        match event {
            DiscoveryEvent::ServiceUpdated(s) => {
                assert_eq!(s.id, "update-test");
                assert_eq!(s.health, HealthStatus::Healthy);
            }
            _ => panic!("Expected ServiceUpdated event"),
        }
    }

    #[test]
    fn test_in_memory_removes_dead_watchers_after_notification() {
        let discovery = InMemoryDiscovery::new();
        let rx = discovery.watch("ps").unwrap();
        assert!(
            discovery.watchers.lock().unwrap().contains_key("ps"),
            "watch sender should exist after subscribing"
        );

        drop(rx);

        let service = ServiceInfo::new("ps-0", "PS 0", "ps", "localhost", 5000);
        discovery.register(service).unwrap();
        assert!(
            !discovery.watchers.lock().unwrap().contains_key("ps"),
            "dead watcher sender should be removed after notification"
        );
    }

    #[tokio::test]
    async fn test_spawn_watch_poll_loop_emits_added_and_removed_events() {
        let (sender, _) = tokio::sync::broadcast::channel(16);
        let mut rx = sender.subscribe();
        let poll_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let poll_calls_for_loop = std::sync::Arc::clone(&poll_calls);

        let handle = spawn_watch_poll_loop(
            sender,
            "test",
            std::time::Duration::from_millis(5),
            || true,
            move || {
                let call = poll_calls_for_loop.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move {
                    if call == 0 {
                        Ok(vec![ServiceInfo::new(
                            "ps-0",
                            "ps-0",
                            "ps",
                            "127.0.0.1",
                            5000,
                        )])
                    } else {
                        Ok(Vec::new())
                    }
                }
            },
        );

        let added = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
            .await
            .expect("timed out waiting for ServiceAdded")
            .expect("watch channel closed unexpectedly");
        match added {
            DiscoveryEvent::ServiceAdded(s) => assert_eq!(s.id, "ps-0"),
            other => panic!("expected ServiceAdded, got {other:?}"),
        }

        let removed = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
            .await
            .expect("timed out waiting for ServiceRemoved")
            .expect("watch channel closed unexpectedly");
        match removed {
            DiscoveryEvent::ServiceRemoved(id) => assert_eq!(id, "ps-0"),
            other => panic!("expected ServiceRemoved, got {other:?}"),
        }

        drop(rx);
        tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("watch loop should stop when no receivers")
            .expect("watch task join failed");
    }

    #[tokio::test]
    async fn test_spawn_watch_poll_loop_emits_updated_events() {
        let (sender, _) = tokio::sync::broadcast::channel(16);
        let mut rx = sender.subscribe();
        let poll_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let poll_calls_for_loop = std::sync::Arc::clone(&poll_calls);

        let handle = spawn_watch_poll_loop(
            sender,
            "test",
            std::time::Duration::from_millis(5),
            || true,
            move || {
                let call = poll_calls_for_loop.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move {
                    let health = if call == 0 {
                        HealthStatus::Starting
                    } else {
                        HealthStatus::Healthy
                    };
                    Ok(vec![
                        ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 5000).with_health(health),
                    ])
                }
            },
        );

        let added = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
            .await
            .expect("timed out waiting for ServiceAdded")
            .expect("watch channel closed unexpectedly");
        match added {
            DiscoveryEvent::ServiceAdded(s) => {
                assert_eq!(s.id, "ps-0");
                assert_eq!(s.health, HealthStatus::Starting);
            }
            other => panic!("expected ServiceAdded, got {other:?}"),
        }

        let updated = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
            .await
            .expect("timed out waiting for ServiceUpdated")
            .expect("watch channel closed unexpectedly");
        match updated {
            DiscoveryEvent::ServiceUpdated(s) => {
                assert_eq!(s.id, "ps-0");
                assert_eq!(s.health, HealthStatus::Healthy);
            }
            other => panic!("expected ServiceUpdated, got {other:?}"),
        }

        drop(rx);
        tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("watch loop should stop when no receivers")
            .expect("watch task join failed");
    }

    #[tokio::test]
    async fn test_spawn_watch_poll_loop_stops_after_receivers_drop() {
        let (sender, _) = tokio::sync::broadcast::channel(16);
        let rx = sender.subscribe();
        let poll_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let poll_calls_for_loop = std::sync::Arc::clone(&poll_calls);

        let handle = spawn_watch_poll_loop(
            sender,
            "test",
            std::time::Duration::from_millis(5),
            || true,
            move || {
                poll_calls_for_loop.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move { Ok(Vec::new()) }
            },
        );

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert!(
            poll_calls.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "watch poll loop should execute while receiver is alive"
        );

        drop(rx);
        tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("watch loop should stop when receiver is dropped")
            .expect("watch task join failed");
    }

    #[tokio::test]
    async fn test_spawn_watch_poll_loop_stops_when_continue_predicate_false() {
        let (sender, _) = tokio::sync::broadcast::channel(16);
        let _rx = sender.subscribe();
        let poll_calls = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let poll_calls_for_loop = std::sync::Arc::clone(&poll_calls);
        let keep_running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
        let keep_running_for_loop = std::sync::Arc::clone(&keep_running);

        let handle = spawn_watch_poll_loop(
            sender,
            "test",
            std::time::Duration::from_millis(5),
            move || keep_running_for_loop.load(std::sync::atomic::Ordering::SeqCst),
            move || {
                poll_calls_for_loop.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move { Ok(Vec::new()) }
            },
        );

        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
        assert!(
            poll_calls.load(std::sync::atomic::Ordering::SeqCst) > 0,
            "watch poll loop should execute while continue predicate is true"
        );

        keep_running.store(false, std::sync::atomic::Ordering::SeqCst);
        tokio::time::timeout(std::time::Duration::from_millis(200), handle)
            .await
            .expect("watch loop should stop when continue predicate flips false")
            .expect("watch task join failed");
    }

    #[test]
    fn test_shared_discovery() {
        let discovery: SharedDiscovery = new_in_memory_discovery();

        let service = ServiceInfo::new("shared-test", "Shared Test", "test", "localhost", 9000);
        discovery.register(service).unwrap();

        let services = discovery.discover("test").unwrap();
        assert_eq!(services.len(), 1);
    }

    #[test]
    fn test_health_status_default() {
        let status: HealthStatus = Default::default();
        assert_eq!(status, HealthStatus::Unknown);
    }

    #[test]
    fn test_discovery_event_equality() {
        let service1 = ServiceInfo::new("s1", "S1", "test", "localhost", 8000);
        let service2 = ServiceInfo::new("s1", "S1", "test", "localhost", 8000);

        let event1 = DiscoveryEvent::ServiceAdded(service1);
        let event2 = DiscoveryEvent::ServiceAdded(service2);

        assert_eq!(event1, event2);

        let removed1 = DiscoveryEvent::ServiceRemoved("s1".to_string());
        let removed2 = DiscoveryEvent::ServiceRemoved("s1".to_string());

        assert_eq!(removed1, removed2);
    }

    #[test]
    fn test_multiple_service_types() {
        let discovery = InMemoryDiscovery::new();

        // Register services of different types
        for i in 0..3 {
            let ps = ServiceInfo::new(
                format!("ps-{}", i),
                format!("PS {}", i),
                "ps",
                "localhost",
                5000 + i as u16,
            );
            discovery.register(ps).unwrap();
        }

        for i in 0..5 {
            let worker = ServiceInfo::new(
                format!("worker-{}", i),
                format!("Worker {}", i),
                "worker",
                "localhost",
                6000 + i as u16,
            );
            discovery.register(worker).unwrap();
        }

        // Verify counts
        assert_eq!(discovery.discover("ps").unwrap().len(), 3);
        assert_eq!(discovery.discover("worker").unwrap().len(), 5);
        assert_eq!(discovery.len(), 8);
    }

    #[cfg(feature = "zookeeper")]
    #[test]
    fn test_zk_discovery_creation() {
        let _zk = ZkDiscovery::new("localhost:2181", "/services").with_session_timeout(60000);

        // Just test that it can be created
        assert!(true);
    }

    #[cfg(feature = "zookeeper")]
    #[tokio::test]
    async fn test_zk_disconnect_increments_watch_generation() {
        let zk = ZkDiscovery::new("localhost:2181", "/services");
        let before = zk.watch_generation.load(std::sync::atomic::Ordering::SeqCst);
        zk.disconnect().await.expect("disconnect should succeed");
        let after = zk.watch_generation.load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before + 1);
    }

    #[cfg(feature = "zookeeper")]
    #[tokio::test]
    async fn test_zk_should_spawn_watch_poll_once_per_generation() {
        let zk = ZkDiscovery::new("localhost:2181", "/services");

        assert!(
            zk.should_spawn_watch_poll("ps", 1),
            "first watch on service type should spawn poller"
        );
        assert!(
            !zk.should_spawn_watch_poll("ps", 2),
            "second watch on same service type and generation should not respawn poller"
        );
        assert!(
            zk.should_spawn_watch_poll("ps", 1),
            "single-receiver state should allow poller respawn after stale shutdown"
        );
        assert!(
            zk.should_spawn_watch_poll("worker", 1),
            "different service type should spawn its own poller"
        );

        zk.disconnect().await.expect("disconnect should succeed");
        assert!(
            zk.should_spawn_watch_poll("ps", 1),
            "after disconnect generation bump should allow respawn"
        );
    }

    #[cfg(feature = "zookeeper")]
    #[tokio::test]
    async fn test_zk_sync_watch_receives_removed_event_on_deregister() {
        let zk = ZkDiscovery::new("localhost:2181", "/services");
        zk.register(ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 5000))
            .expect("register should succeed");

        let mut rx = zk.watch("ps").expect("watch should succeed");
        zk.deregister("ps-0")
            .expect("deregister should succeed and notify watchers");

        let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
            .await
            .expect("timed out waiting for ServiceRemoved")
            .expect("watch channel closed unexpectedly");
        match event {
            DiscoveryEvent::ServiceRemoved(id) => assert_eq!(id, "ps-0"),
            other => panic!("expected ServiceRemoved, got {other:?}"),
        }
    }

    #[cfg(feature = "zookeeper")]
    #[test]
    fn test_zk_sync_register_removes_dead_watchers() {
        let zk = ZkDiscovery::new("localhost:2181", "/services");
        let rx = zk.watch("ps").expect("watch should succeed");
        assert!(
            zk.watchers.lock().unwrap().contains_key("ps"),
            "watch sender should exist after subscribing"
        );

        drop(rx);
        zk.register(ServiceInfo::new("ps-0", "ps-0", "ps", "127.0.0.1", 5000))
            .expect("register should succeed");
        assert!(
            !zk.watchers.lock().unwrap().contains_key("ps"),
            "dead watch sender should be removed after notification"
        );
    }

    #[cfg(feature = "consul")]
    #[test]
    fn test_consul_discovery_creation() {
        let _consul = ConsulDiscovery::new("http://localhost:8500")
            .with_datacenter("dc1")
            .with_token("secret-token");

        // Just test that it can be created
        assert!(true);
    }

    #[cfg(feature = "consul")]
    #[tokio::test]
    async fn test_consul_disconnect_increments_watch_generation() {
        let consul = ConsulDiscovery::new("http://localhost:8500");
        let before = consul
            .watch_generation
            .load(std::sync::atomic::Ordering::SeqCst);
        <ConsulDiscovery as ServiceDiscoveryAsync>::disconnect(&consul)
            .await
            .expect("disconnect should succeed");
        let after = consul
            .watch_generation
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(after, before + 1);
    }

    #[cfg(feature = "consul")]
    #[tokio::test]
    async fn test_consul_should_spawn_watch_poll_once_per_generation() {
        let consul = ConsulDiscovery::new("http://localhost:8500");

        assert!(
            consul.should_spawn_watch_poll("ps", 1),
            "first watch on service type should spawn poller"
        );
        assert!(
            !consul.should_spawn_watch_poll("ps", 2),
            "second watch on same service type and generation should not respawn poller"
        );
        assert!(
            consul.should_spawn_watch_poll("ps", 1),
            "single-receiver state should allow poller respawn after stale shutdown"
        );
        assert!(
            consul.should_spawn_watch_poll("worker", 1),
            "different service type should spawn its own poller"
        );

        <ConsulDiscovery as ServiceDiscoveryAsync>::disconnect(&consul)
            .await
            .expect("disconnect should succeed");
        assert!(
            consul.should_spawn_watch_poll("ps", 1),
            "after disconnect generation bump should allow respawn"
        );
    }

    #[cfg(feature = "consul")]
    #[tokio::test]
    async fn test_consul_sync_watch_receives_removed_event_on_deregister() {
        let consul = ConsulDiscovery::new("http://localhost:8500");
        consul
            .register(ServiceInfo::new(
                "worker-0",
                "worker-0",
                "worker",
                "127.0.0.1",
                6000,
            ))
            .expect("register should succeed");

        let mut rx = consul.watch("worker").expect("watch should succeed");
        consul
            .deregister("worker-0")
            .expect("deregister should succeed and notify watchers");

        let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
            .await
            .expect("timed out waiting for ServiceRemoved")
            .expect("watch channel closed unexpectedly");
        match event {
            DiscoveryEvent::ServiceRemoved(id) => assert_eq!(id, "worker-0"),
            other => panic!("expected ServiceRemoved, got {other:?}"),
        }
    }

    #[cfg(feature = "consul")]
    #[test]
    fn test_consul_sync_register_removes_dead_watchers() {
        let consul = ConsulDiscovery::new("http://localhost:8500");
        let rx = consul.watch("worker").expect("watch should succeed");
        assert!(
            consul.watchers.lock().unwrap().contains_key("worker"),
            "watch sender should exist after subscribing"
        );

        drop(rx);
        consul
            .register(ServiceInfo::new(
                "worker-0",
                "worker-0",
                "worker",
                "127.0.0.1",
                6000,
            ))
            .expect("register should succeed");
        assert!(
            !consul.watchers.lock().unwrap().contains_key("worker"),
            "dead watch sender should be removed after notification"
        );
    }
}
