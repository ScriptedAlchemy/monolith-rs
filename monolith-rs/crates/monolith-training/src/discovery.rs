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
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;
use tokio::sync::broadcast::{self, Receiver, Sender};

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
        let watchers = self.watchers.lock().unwrap();
        if let Some(sender) = watchers.get(service_type) {
            // Ignore send errors (no receivers)
            let _ = sender.send(event);
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

        if services.contains_key(&service.id) {
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

        self.notify_watchers(&service_type, DiscoveryEvent::ServiceAdded(service_clone));
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
    /// In-memory cache of services.
    services: RwLock<HashMap<String, ServiceInfo>>,
    /// Event senders for watchers.
    watchers: Mutex<HashMap<String, Sender<DiscoveryEvent>>>,
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
            services: RwLock::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
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
        tracing::info!(
            hosts = %self.hosts,
            base_path = %self.base_path,
            "Connecting to ZooKeeper (stub)"
        );
        // TODO: Implement actual ZooKeeper connection
        Ok(())
    }

    /// Disconnects from ZooKeeper.
    pub async fn disconnect(&self) -> Result<()> {
        tracing::info!("Disconnecting from ZooKeeper (stub)");
        // TODO: Implement actual ZooKeeper disconnection
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
}

#[cfg(feature = "zookeeper")]
impl ServiceDiscovery for ZkDiscovery {
    fn register(&self, service: ServiceInfo) -> Result<()> {
        tracing::info!(
            service_id = %service.id,
            service_type = %service.service_type,
            "Registering service with ZooKeeper (stub)"
        );

        // Stub: Store in local cache
        let mut services = self.services.write().unwrap();
        if services.contains_key(&service.id) {
            return Err(DiscoveryError::AlreadyRegistered(service.id.clone()));
        }
        services.insert(service.id.clone(), service);

        // TODO: Create ephemeral node in ZooKeeper
        Ok(())
    }

    fn discover(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        tracing::debug!(
            service_type = %service_type,
            "Discovering services from ZooKeeper (stub)"
        );

        // Stub: Return from local cache
        let services = self.services.read().unwrap();
        let matching: Vec<ServiceInfo> = services
            .values()
            .filter(|s| s.service_type == service_type)
            .cloned()
            .collect();

        // TODO: Query ZooKeeper for actual services
        Ok(matching)
    }

    fn watch(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        tracing::debug!(
            service_type = %service_type,
            "Setting up ZooKeeper watch (stub)"
        );

        let sender = self.get_or_create_sender(service_type);
        // TODO: Set up actual ZooKeeper watch
        Ok(sender.subscribe())
    }

    fn deregister(&self, service_id: &str) -> Result<()> {
        tracing::info!(
            service_id = %service_id,
            "Deregistering service from ZooKeeper (stub)"
        );

        // Stub: Remove from local cache
        let mut services = self.services.write().unwrap();
        services
            .remove(service_id)
            .ok_or_else(|| DiscoveryError::NotFound(service_id.to_string()))?;

        // TODO: Delete node from ZooKeeper
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
    /// In-memory cache of services.
    services: RwLock<HashMap<String, ServiceInfo>>,
    /// Event senders for watchers.
    watchers: Mutex<HashMap<String, Sender<DiscoveryEvent>>>,
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
            services: RwLock::new(HashMap::new()),
            watchers: Mutex::new(HashMap::new()),
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

    /// Connects to Consul.
    ///
    /// This is a placeholder that would establish a connection to Consul.
    pub async fn connect(&self) -> Result<()> {
        tracing::info!(
            address = %self.address,
            datacenter = ?self.datacenter,
            "Connecting to Consul (stub)"
        );
        // TODO: Implement actual Consul connection
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
}

#[cfg(feature = "consul")]
impl ServiceDiscovery for ConsulDiscovery {
    fn register(&self, service: ServiceInfo) -> Result<()> {
        tracing::info!(
            service_id = %service.id,
            service_type = %service.service_type,
            "Registering service with Consul (stub)"
        );

        // Stub: Store in local cache
        let mut services = self.services.write().unwrap();
        if services.contains_key(&service.id) {
            return Err(DiscoveryError::AlreadyRegistered(service.id.clone()));
        }
        services.insert(service.id.clone(), service);

        // TODO: Register with Consul API
        Ok(())
    }

    fn discover(&self, service_type: &str) -> Result<Vec<ServiceInfo>> {
        tracing::debug!(
            service_type = %service_type,
            "Discovering services from Consul (stub)"
        );

        // Stub: Return from local cache
        let services = self.services.read().unwrap();
        let matching: Vec<ServiceInfo> = services
            .values()
            .filter(|s| s.service_type == service_type)
            .cloned()
            .collect();

        // TODO: Query Consul for actual services
        Ok(matching)
    }

    fn watch(&self, service_type: &str) -> Result<Receiver<DiscoveryEvent>> {
        tracing::debug!(
            service_type = %service_type,
            "Setting up Consul watch (stub)"
        );

        let sender = self.get_or_create_sender(service_type);
        // TODO: Set up actual Consul blocking query
        Ok(sender.subscribe())
    }

    fn deregister(&self, service_id: &str) -> Result<()> {
        tracing::info!(
            service_id = %service_id,
            "Deregistering service from Consul (stub)"
        );

        // Stub: Remove from local cache
        let mut services = self.services.write().unwrap();
        services
            .remove(service_id)
            .ok_or_else(|| DiscoveryError::NotFound(service_id.to_string()))?;

        // TODO: Deregister from Consul API
        Ok(())
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
        let zk = ZkDiscovery::new("localhost:2181", "/services").with_session_timeout(60000);

        // Just test that it can be created
        assert!(true);
    }

    #[cfg(feature = "consul")]
    #[test]
    fn test_consul_discovery_creation() {
        let consul = ConsulDiscovery::new("http://localhost:8500")
            .with_datacenter("dc1")
            .with_token("secret-token");

        // Just test that it can be created
        assert!(true);
    }
}
