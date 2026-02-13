//! Python-compatible service discovery helpers.
//!
//! The Python implementation in `monolith/native_training/service_discovery.py` exposes a
//! simple interface:
//! - register(name, index, addr)
//! - deregister(name, index, addr)
//! - query(name) -> {index: addr}
//!
//! The Rust crate has a more general `ServiceDiscovery{,Async}` API based on `ServiceInfo`.
//! This module provides small adapters that match the Python behavior closely for
//! `TF_CONFIG` (Primus) and MLP environment discovery.

use crate::discovery::{DiscoveryError, Result};
use crate::discovery::{ServiceDiscovery, ServiceDiscoveryAsync, ServiceInfo};
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Python-style service discovery interface.
///
/// This is intentionally small and matches the Python surface area used by
/// `runner_utils.get_discovery(...)`.
pub trait PyServiceDiscovery: Send + Sync {
    fn register(&self, name: &str, index: i32, addr: &str) -> Result<()>;
    fn deregister(&self, name: &str, index: i32, addr: &str) -> Result<()>;
    fn query(&self, name: &str) -> Result<HashMap<i32, String>>;

    fn close(&self) -> Result<()> {
        Ok(())
    }
}

fn parse_addr(addr: &str) -> Result<(String, u16)> {
    let (host, port_str) = addr
        .split_once(':')
        .ok_or_else(|| DiscoveryError::ConfigError(format!("Invalid addr: {:?}", addr)))?;
    let port: u16 = port_str
        .parse()
        .map_err(|_| DiscoveryError::ConfigError(format!("Invalid port in addr: {:?}", addr)))?;
    Ok((host.to_string(), port))
}

// =============================================================================
// Local host file discovery (used by Python cpu_training_distributed_test_binary)
// =============================================================================

/// A file-backed discovery backend mirroring `cpu_training_distributed_test_binary.HostServiceDiscovery`.
///
/// It stores registrations under:
/// `base_path/<name>/<index>` (file contents: `<addr>`)
#[derive(Debug, Clone)]
pub struct HostFileDiscovery {
    base_path: std::path::PathBuf,
}

impl HostFileDiscovery {
    pub fn new(base_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            base_path: base_path.into(),
        }
    }

    fn named_path(&self, name: &str) -> std::path::PathBuf {
        self.base_path.join(name)
    }
}

impl PyServiceDiscovery for HostFileDiscovery {
    fn register(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        let dir = self.named_path(name);
        std::fs::create_dir_all(&dir).map_err(|e| {
            DiscoveryError::Internal(format!("Failed to create host discovery dir: {e}"))
        })?;
        std::fs::write(dir.join(index.to_string()), addr.as_bytes()).map_err(|e| {
            DiscoveryError::Internal(format!("Failed to write host discovery file: {e}"))
        })?;
        Ok(())
    }

    fn deregister(&self, _name: &str, _index: i32, _addr: &str) -> Result<()> {
        Ok(())
    }

    fn query(&self, name: &str) -> Result<HashMap<i32, String>> {
        let dir = self.named_path(name);
        if !dir.exists() {
            return Ok(HashMap::new());
        }
        let mut out = HashMap::new();
        for entry in std::fs::read_dir(&dir)
            .map_err(|e| DiscoveryError::Internal(format!("Failed to read dir: {e}")))?
        {
            let entry = entry.map_err(|e| DiscoveryError::Internal(format!("read_dir: {e}")))?;
            let file_name = entry.file_name().to_string_lossy().to_string();
            let idx: i32 = match file_name.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let addr = std::fs::read_to_string(entry.path())
                .map_err(|e| DiscoveryError::Internal(format!("read_to_string: {e}")))?;
            out.insert(idx, addr);
        }
        Ok(out)
    }
}

impl ServiceDiscovery for HostFileDiscovery {
    fn register(&self, service: ServiceInfo) -> crate::discovery::Result<()> {
        // Store by service_type and an index derived from "index" metadata or trailing "-<n>".
        let idx = service
            .metadata
            .get("index")
            .and_then(|s| s.parse::<i32>().ok())
            .or_else(|| {
                service
                    .id
                    .rsplit_once('-')
                    .and_then(|(_, i)| i.parse::<i32>().ok())
            })
            .unwrap_or(0);
        let addr = service
            .metadata
            .get("addr")
            .cloned()
            .unwrap_or_else(|| service.address());
        PyServiceDiscovery::register(self, &service.service_type, idx, &addr)
    }

    fn discover(&self, service_type: &str) -> crate::discovery::Result<Vec<ServiceInfo>> {
        let index_to_addr = PyServiceDiscovery::query(self, service_type)?;
        let mut out = Vec::with_capacity(index_to_addr.len());
        for (idx, addr) in index_to_addr {
            let (host, port) = parse_addr(&addr)?;
            let id = format!("{}-{}", service_type, idx);
            let mut svc = ServiceInfo::new(id.clone(), id.clone(), service_type, host, port);
            svc.metadata.insert("index".to_string(), idx.to_string());
            svc.metadata.insert("addr".to_string(), addr);
            out.push(svc);
        }
        Ok(out)
    }

    fn watch(
        &self,
        _service_type: &str,
    ) -> crate::discovery::Result<tokio::sync::broadcast::Receiver<crate::discovery::DiscoveryEvent>>
    {
        Err(DiscoveryError::ConfigError(
            "HostFileDiscovery does not support watch()".to_string(),
        ))
    }

    fn deregister(&self, _service_id: &str) -> crate::discovery::Result<()> {
        Ok(())
    }
}

#[async_trait]
impl ServiceDiscoveryAsync for HostFileDiscovery {
    async fn connect(&self) -> crate::discovery::Result<()> {
        Ok(())
    }

    async fn disconnect(&self) -> crate::discovery::Result<()> {
        Ok(())
    }

    async fn register_async(&self, service: ServiceInfo) -> crate::discovery::Result<()> {
        ServiceDiscovery::register(self, service)
    }

    async fn discover_async(
        &self,
        service_type: &str,
    ) -> crate::discovery::Result<Vec<ServiceInfo>> {
        self.discover(service_type)
    }

    async fn watch_async(
        &self,
        service_type: &str,
    ) -> crate::discovery::Result<tokio::sync::broadcast::Receiver<crate::discovery::DiscoveryEvent>>
    {
        self.watch(service_type)
    }

    async fn deregister_async(&self, _service_id: &str) -> crate::discovery::Result<()> {
        Ok(())
    }
}

// =============================================================================
// TF_CONFIG (Primus) discovery
// =============================================================================

#[derive(Debug, Clone, serde::Deserialize)]
struct TfTask {
    #[serde(rename = "type")]
    ty: String,
    index: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TfConfig {
    cluster: HashMap<String, Vec<String>>,
    task: TfTask,
}

/// `TF_CONFIG`-based discovery (Primus-style), matching Python's `TfConfigServiceDiscovery`.
#[derive(Debug, Clone)]
pub struct TfConfigServiceDiscovery {
    cfg: TfConfig,
}

impl TfConfigServiceDiscovery {
    /// Construct from a parsed JSON value.
    pub fn new(tf_config_json: &str) -> Result<Self> {
        let cfg: TfConfig = serde_json::from_str(tf_config_json)
            .map_err(|e| DiscoveryError::ConfigError(format!("Invalid TF_CONFIG JSON: {}", e)))?;
        Ok(Self { cfg })
    }

    /// Server type as used by Python (chief becomes worker).
    pub fn server_type(&self) -> &str {
        if self.cfg.task.ty == "chief" {
            "worker"
        } else {
            self.cfg.task.ty.as_str()
        }
    }

    /// Task index as used by Python: when chief exists, worker indices are shifted by +1.
    pub fn index(&self) -> usize {
        if self.cfg.cluster.contains_key("chief") {
            if self.cfg.task.ty == "worker" {
                self.cfg.task.index + 1
            } else {
                self.cfg.task.index
            }
        } else {
            self.cfg.task.index
        }
    }

    /// Address for this task.
    pub fn addr(&self) -> Result<&str> {
        let addrs = self
            .cfg
            .cluster
            .get(self.cfg.task.ty.as_str())
            .ok_or_else(|| {
                DiscoveryError::ConfigError(format!(
                    "TF_CONFIG cluster missing task type {:?}",
                    self.cfg.task.ty
                ))
            })?;
        addrs
            .get(self.cfg.task.index)
            .map(|s| s.as_str())
            .ok_or_else(|| {
                DiscoveryError::ConfigError(format!(
                    "TF_CONFIG cluster missing index {} for type {:?}",
                    self.cfg.task.index, self.cfg.task.ty
                ))
            })
    }
}

impl PyServiceDiscovery for TfConfigServiceDiscovery {
    fn register(&self, _name: &str, _index: i32, _addr: &str) -> Result<()> {
        // Python's TfConfigServiceDiscovery does not perform registration.
        Ok(())
    }

    fn deregister(&self, _name: &str, _index: i32, _addr: &str) -> Result<()> {
        // Python's TfConfigServiceDiscovery does not perform deregistration.
        Ok(())
    }

    fn query(&self, name: &str) -> Result<HashMap<i32, String>> {
        match name {
            "ps" => {
                let list =
                    self.cfg.cluster.get("ps").ok_or_else(|| {
                        DiscoveryError::ConfigError("TF_CONFIG missing ps".into())
                    })?;
                Ok(list
                    .iter()
                    .enumerate()
                    .map(|(i, addr)| (i as i32, addr.clone()))
                    .collect())
            }
            "worker" => {
                let mut list: Vec<String> = Vec::new();
                if let Some(chief) = self.cfg.cluster.get("chief") {
                    list.extend_from_slice(chief);
                }
                let workers = self.cfg.cluster.get("worker").ok_or_else(|| {
                    DiscoveryError::ConfigError("TF_CONFIG missing worker".into())
                })?;
                list.extend_from_slice(workers);
                Ok(list
                    .into_iter()
                    .enumerate()
                    .map(|(i, addr)| (i as i32, addr))
                    .collect())
            }
            other => Err(DiscoveryError::ConfigError(format!(
                "name must be ps/worker, got {:?}",
                other
            ))),
        }
    }
}

impl ServiceDiscovery for TfConfigServiceDiscovery {
    fn register(&self, _service: ServiceInfo) -> crate::discovery::Result<()> {
        // TF_CONFIG discovery is read-only in Python.
        Ok(())
    }

    fn discover(&self, service_type: &str) -> crate::discovery::Result<Vec<ServiceInfo>> {
        let index_to_addr = self.query(service_type)?;
        let mut out = Vec::with_capacity(index_to_addr.len());
        for (idx, addr) in index_to_addr {
            let (host, port) = parse_addr(&addr)?;
            let id = format!("{}-{}", service_type, idx);
            let mut svc = ServiceInfo::new(id.clone(), id.clone(), service_type, host, port);
            svc.metadata.insert("index".to_string(), idx.to_string());
            svc.metadata.insert("addr".to_string(), addr);
            out.push(svc);
        }
        Ok(out)
    }

    fn watch(
        &self,
        _service_type: &str,
    ) -> crate::discovery::Result<tokio::sync::broadcast::Receiver<crate::discovery::DiscoveryEvent>>
    {
        // TF_CONFIG is static for the lifetime of the process.
        Err(DiscoveryError::ConfigError(
            "TF_CONFIG discovery does not support watch()".to_string(),
        ))
    }

    fn deregister(&self, _service_id: &str) -> crate::discovery::Result<()> {
        Ok(())
    }
}

#[async_trait]
impl ServiceDiscoveryAsync for TfConfigServiceDiscovery {
    async fn connect(&self) -> crate::discovery::Result<()> {
        Ok(())
    }

    async fn disconnect(&self) -> crate::discovery::Result<()> {
        Ok(())
    }

    async fn register_async(&self, _service: ServiceInfo) -> crate::discovery::Result<()> {
        Ok(())
    }

    async fn discover_async(
        &self,
        service_type: &str,
    ) -> crate::discovery::Result<Vec<ServiceInfo>> {
        self.discover(service_type)
    }

    async fn watch_async(
        &self,
        service_type: &str,
    ) -> crate::discovery::Result<tokio::sync::broadcast::Receiver<crate::discovery::DiscoveryEvent>>
    {
        ServiceDiscovery::watch(self, service_type)
    }

    async fn deregister_async(&self, _service_id: &str) -> crate::discovery::Result<()> {
        Ok(())
    }
}

// =============================================================================
// MLP discovery
// =============================================================================

#[derive(Debug, Clone)]
struct MlpEnv {
    role: String,                      // uppercase
    index: usize,
    host: Option<String>,
    all_roles: HashMap<String, usize>, // uppercase role -> count
}

impl MlpEnv {
    fn from_env() -> Self {
        let role = std::env::var("MLP_ROLE")
            .unwrap_or_default()
            .trim()
            .to_uppercase();
        let index = std::env::var("MLP_ROLE_INDEX")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(0);
        let host = std::env::var("MLP_HOST")
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        // Parse `MLP_<ROLE>_NUM` keys.
        let mut all_roles: HashMap<String, usize> = HashMap::new();
        for (k, v) in std::env::vars() {
            if let Some(role_key) = k.strip_prefix("MLP_").and_then(|s| s.strip_suffix("_NUM")) {
                if role_key.is_empty() {
                    continue;
                }
                if let Ok(n) = v.parse::<usize>() {
                    all_roles.insert(role_key.to_uppercase(), n);
                }
            }
        }

        Self {
            role,
            index,
            host,
            all_roles,
        }
    }

    fn num_replicas(&self, role: &str) -> usize {
        let key = role.trim().to_uppercase();
        *self.all_roles.get(&key).unwrap_or(&0)
    }

    fn get_host(&self, role: &str, index: usize, is_primary: bool) -> Option<String> {
        let role_u = role.trim().to_uppercase();
        let key = if is_primary {
            format!("MLP_{}_{}_PRIMARY_HOST", role_u, index)
        } else {
            format!("MLP_{}_{}_HOST", role_u, index)
        };
        std::env::var(key).ok().map(|s| s.trim().to_string())
    }

    fn get_port(&self, role: &str, index: usize) -> Option<u16> {
        let role_u = role.trim().to_uppercase();
        let key = format!("MLP_{}_{}_PORT", role_u, index);
        std::env::var(key)
            .ok()
            .and_then(|s| s.trim().parse::<u16>().ok())
    }

    fn get_addr(&self, role: &str, index: usize, is_primary: bool) -> Option<String> {
        let host = self.get_host(role, index, is_primary)?;
        let port = self.get_port(role, index)?;
        Some(format!("{}:{}", host, port))
    }

    fn current_host_candidates(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.role.is_empty() {
            if let Some(h) = self.get_host(&self.role, self.index, true) {
                out.push(h);
            }
            if let Some(h) = self.get_host(&self.role, self.index, false) {
                out.push(h);
            }
        }
        if let Some(h) = self.host.clone() {
            out.push(h);
        }
        out.sort();
        out.dedup();
        out
    }
}

/// Environment-variable based discovery matching Python's `MLPServiceDiscovery`.
///
/// This implementation is intentionally non-networking: it does not attempt the
/// long-running port health checks that Python performs for PS roles.
#[derive(Debug)]
pub struct MlpServiceDiscovery {
    env: MlpEnv,
    // filters store entries like "ps:0" or "worker:3" (lowercase name)
    filters: Mutex<HashSet<String>>,
    closed: AtomicBool,
}

impl MlpServiceDiscovery {
    pub fn new() -> Self {
        Self {
            env: MlpEnv::from_env(),
            filters: Mutex::new(HashSet::new()),
            closed: AtomicBool::new(false),
        }
    }

    pub fn server_type(&self) -> Option<String> {
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        if self.env.role.is_empty() {
            None
        } else {
            Some(self.env.role.to_lowercase())
        }
    }

    pub fn index(&self) -> usize {
        self.env.index
    }

    pub fn addr(&self) -> Option<String> {
        if self.closed.load(Ordering::SeqCst) {
            return None;
        }
        self.server_type()
            .and_then(|role| self.env.get_addr(&role, self.index(), true))
    }

    fn key(name: &str, index: i32) -> String {
        format!("{}:{}", name.trim().to_lowercase(), index)
    }

    fn lock_filters_recover(&self) -> std::sync::MutexGuard<'_, HashSet<String>> {
        match self.filters.lock() {
            Ok(filters) => filters,
            Err(poisoned) => {
                tracing::warn!(
                    "mlp discovery filters mutex was poisoned; continuing with recovered state"
                );
                poisoned.into_inner()
            }
        }
    }

    pub fn deregister_all(&self) {
        if self.closed.load(Ordering::SeqCst) {
            return;
        }
        let mut filters = self.lock_filters_recover();
        for (name, num) in &self.env.all_roles {
            for idx in 0..*num {
                filters.insert(Self::key(&name.to_lowercase(), idx as i32));
            }
        }
    }

    pub fn query_all(&self) -> Result<HashMap<String, HashMap<i32, String>>> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(HashMap::new());
        }
        let mut out = HashMap::new();
        for name in self.env.all_roles.keys() {
            let lower = name.to_lowercase();
            if matches!(lower.as_str(), "ps" | "worker" | "chief") {
                out.insert(lower.clone(), self.query(&lower)?);
            }
        }
        Ok(out)
    }

    fn validate(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        let role_name = name.trim().to_uppercase();
        if !self.env.all_roles.contains_key(&role_name) {
            return Err(DiscoveryError::ConfigError(format!(
                "Unknown MLP role {:?}",
                name
            )));
        }

        if index < 0 {
            return Err(DiscoveryError::ConfigError(format!(
                "index must be >= 0, got {}",
                index
            )));
        }
        let n = self.env.num_replicas(&role_name);
        if (index as usize) >= n {
            return Err(DiscoveryError::ConfigError(format!(
                "index {} out of bounds for role {:?} (num_replicas={})",
                index, name, n
            )));
        }

        // Match Python-style host/port checks:
        // - host must be one of local aliases, expected primary/non-primary host,
        //   or current process host aliases.
        // - port must match MLP_<ROLE>_<IDX>_PORT.
        let (real_host, port_str) = addr
            .split_once(':')
            .ok_or_else(|| DiscoveryError::ConfigError(format!("Invalid addr: {:?}", addr)))?;
        let mut allowed_hosts: HashSet<String> = ["local", "localhost", "127.0.0.1", "0.0.0.0"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        if let Some(h) = self.env.get_host(&role_name, index as usize, true) {
            allowed_hosts.insert(h);
        }
        if let Some(h) = self.env.get_host(&role_name, index as usize, false) {
            allowed_hosts.insert(h);
        }
        for h in self.env.current_host_candidates() {
            allowed_hosts.insert(h);
        }
        if !allowed_hosts.contains(real_host) {
            return Err(DiscoveryError::ConfigError(format!(
                "Host mismatch for {:?}.{}: got {:?}",
                name, index, real_host
            )));
        }

        let port: u16 = port_str.parse().map_err(|_| {
            DiscoveryError::ConfigError(format!("Invalid port in addr: {:?}", addr))
        })?;

        let exp_port = self
            .env
            .get_port(name, index as usize)
            .ok_or_else(|| DiscoveryError::ConfigError("Missing MLP_*_PORT env".into()))?;
        if port != exp_port {
            return Err(DiscoveryError::ConfigError(format!(
                "Port mismatch for {:?}.{}: expected {}, got {}",
                name, index, exp_port, port
            )));
        }
        Ok(())
    }
}

impl Default for MlpServiceDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl PyServiceDiscovery for MlpServiceDiscovery {
    fn register(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.validate(name, index, addr)?;
        let mut filters = self.lock_filters_recover();
        filters.remove(&Self::key(name, index));
        Ok(())
    }

    fn deregister(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.validate(name, index, addr)?;
        let mut filters = self.lock_filters_recover();
        filters.insert(Self::key(name, index));
        Ok(())
    }

    fn query(&self, name: &str) -> Result<HashMap<i32, String>> {
        if self.closed.load(Ordering::SeqCst) {
            return Ok(HashMap::new());
        }
        if name.trim().is_empty() {
            return Err(DiscoveryError::ConfigError(
                "name must be non-empty".to_string(),
            ));
        }

        let num = self.env.num_replicas(name);
        if num == 0 {
            return Ok(HashMap::new());
        }

        let filters = self.lock_filters_recover();
        let mut out = HashMap::new();

        for idx in 0..num {
            let key = Self::key(name, idx as i32);
            if filters.contains(&key) {
                continue;
            }
            if let Some(addr) = self.env.get_addr(name, idx, true) {
                out.insert(idx as i32, addr);
            }
        }

        Ok(out)
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        self.lock_filters_recover().clear();
        Ok(())
    }
}

impl ServiceDiscovery for MlpServiceDiscovery {
    fn register(&self, service: ServiceInfo) -> crate::discovery::Result<()> {
        // Map ServiceInfo -> python-ish name/index/addr.
        let idx = service
            .metadata
            .get("index")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        PyServiceDiscovery::register(self, &service.service_type, idx, &service.address())
    }

    fn discover(&self, service_type: &str) -> crate::discovery::Result<Vec<ServiceInfo>> {
        let index_to_addr = self.query(service_type)?;
        let mut out = Vec::with_capacity(index_to_addr.len());
        for (idx, addr) in index_to_addr {
            let (host, port) = parse_addr(&addr)?;
            let id = format!("{}-{}", service_type, idx);
            let mut svc = ServiceInfo::new(id.clone(), id.clone(), service_type, host, port);
            svc.metadata.insert("index".to_string(), idx.to_string());
            svc.metadata.insert("addr".to_string(), addr);
            out.push(svc);
        }
        Ok(out)
    }

    fn watch(
        &self,
        _service_type: &str,
    ) -> crate::discovery::Result<tokio::sync::broadcast::Receiver<crate::discovery::DiscoveryEvent>>
    {
        // Python MLP discovery is local and effectively static aside from filters.
        Err(DiscoveryError::ConfigError(
            "MLP discovery does not support watch()".to_string(),
        ))
    }

    fn deregister(&self, service_id: &str) -> crate::discovery::Result<()> {
        // Best-effort: accept either "<name>-<idx>" or raw service_id.
        if let Some((name, idx_str)) = service_id.rsplit_once('-') {
            if let Ok(idx) = idx_str.parse::<i32>() {
                // We don't know the exact addr here; use the env-derived addr.
                let addr = self
                    .query(name)?
                    .get(&idx)
                    .cloned()
                    .unwrap_or_else(|| "0.0.0.0:0".to_string());
                return PyServiceDiscovery::deregister(self, name, idx, &addr);
            }
        }
        Err(DiscoveryError::ConfigError(format!(
            "Cannot infer name/index from service_id {:?}",
            service_id
        )))
    }
}

#[async_trait]
impl ServiceDiscoveryAsync for MlpServiceDiscovery {
    async fn connect(&self) -> crate::discovery::Result<()> {
        Ok(())
    }

    async fn disconnect(&self) -> crate::discovery::Result<()> {
        Ok(())
    }

    async fn register_async(&self, service: ServiceInfo) -> crate::discovery::Result<()> {
        let idx = service
            .metadata
            .get("index")
            .and_then(|s| s.parse::<i32>().ok())
            .unwrap_or(0);
        PyServiceDiscovery::register(self, &service.service_type, idx, &service.address())
    }

    async fn discover_async(
        &self,
        service_type: &str,
    ) -> crate::discovery::Result<Vec<ServiceInfo>> {
        self.discover(service_type)
    }

    async fn watch_async(
        &self,
        service_type: &str,
    ) -> crate::discovery::Result<tokio::sync::broadcast::Receiver<crate::discovery::DiscoveryEvent>>
    {
        ServiceDiscovery::watch(self, service_type)
    }

    async fn deregister_async(&self, service_id: &str) -> crate::discovery::Result<()> {
        ServiceDiscovery::deregister(self, service_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static MLP_ENV_TEST_MUTEX: Mutex<()> = Mutex::new(());

    struct EnvSnapshot {
        saved: Vec<(String, Option<String>)>,
    }

    impl EnvSnapshot {
        fn install(vars: &[(&str, &str)], managed_keys: &[&str]) -> Self {
            let mut saved = Vec::with_capacity(managed_keys.len());
            for key in managed_keys {
                saved.push(((*key).to_string(), std::env::var(key).ok()));
                std::env::remove_var(key);
            }
            for (k, v) in vars {
                std::env::set_var(k, v);
            }
            Self { saved }
        }
    }

    impl Drop for EnvSnapshot {
        fn drop(&mut self) {
            for (k, v) in &self.saved {
                if let Some(v) = v {
                    std::env::set_var(k, v);
                } else {
                    std::env::remove_var(k);
                }
            }
        }
    }

    const MLP_MANAGED_KEYS: &[&str] = &[
        "MLP_ROLE",
        "MLP_ROLE_INDEX",
        "MLP_HOST",
        "MLP_WORKER_NUM",
        "MLP_PS_NUM",
        "MLP_CHIEF_NUM",
        "MLP_WORKER_0_PRIMARY_HOST",
        "MLP_WORKER_0_HOST",
        "MLP_WORKER_0_PORT",
        "MLP_WORKER_1_PRIMARY_HOST",
        "MLP_WORKER_1_HOST",
        "MLP_WORKER_1_PORT",
        "MLP_PS_0_PRIMARY_HOST",
        "MLP_PS_0_HOST",
        "MLP_PS_0_PORT",
        "MLP_CHIEF_0_PRIMARY_HOST",
        "MLP_CHIEF_0_HOST",
        "MLP_CHIEF_0_PORT",
        "MLP_TRAINER_NUM",
    ];

    fn install_default_mlp_env() -> EnvSnapshot {
        EnvSnapshot::install(
            &[
                ("MLP_ROLE", "worker"),
                ("MLP_ROLE_INDEX", "0"),
                ("MLP_HOST", "worker0"),
                ("MLP_WORKER_NUM", "2"),
                ("MLP_PS_NUM", "1"),
                ("MLP_CHIEF_NUM", "1"),
                ("MLP_WORKER_0_PRIMARY_HOST", "worker0"),
                ("MLP_WORKER_0_HOST", "worker0"),
                ("MLP_WORKER_0_PORT", "2222"),
                ("MLP_WORKER_1_PRIMARY_HOST", "worker1"),
                ("MLP_WORKER_1_HOST", "worker1"),
                ("MLP_WORKER_1_PORT", "2223"),
                ("MLP_PS_0_PRIMARY_HOST", "ps0"),
                ("MLP_PS_0_HOST", "ps0"),
                ("MLP_PS_0_PORT", "3333"),
                ("MLP_CHIEF_0_PRIMARY_HOST", "chief0"),
                ("MLP_CHIEF_0_HOST", "chief0"),
                ("MLP_CHIEF_0_PORT", "4444"),
            ],
            MLP_MANAGED_KEYS,
        )
    }

    fn install_mlp_env_without_chief() -> EnvSnapshot {
        EnvSnapshot::install(
            &[
                ("MLP_ROLE", "worker"),
                ("MLP_ROLE_INDEX", "0"),
                ("MLP_HOST", "worker0"),
                ("MLP_WORKER_NUM", "1"),
                ("MLP_PS_NUM", "1"),
                ("MLP_TRAINER_NUM", "2"),
                ("MLP_WORKER_0_PRIMARY_HOST", "worker0"),
                ("MLP_WORKER_0_HOST", "worker0"),
                ("MLP_WORKER_0_PORT", "2222"),
                ("MLP_PS_0_PRIMARY_HOST", "ps0"),
                ("MLP_PS_0_HOST", "ps0"),
                ("MLP_PS_0_PORT", "3333"),
            ],
            MLP_MANAGED_KEYS,
        )
    }

    #[test]
    fn test_mlp_service_discovery_query_all_and_filters() {
        let _guard = MLP_ENV_TEST_MUTEX
            .lock()
            .expect("mlp env test mutex should not be poisoned");
        let _env = install_default_mlp_env();

        let d = MlpServiceDiscovery::new();
        assert_eq!(d.server_type().as_deref(), Some("worker"));
        assert_eq!(d.index(), 0);
        assert_eq!(d.addr().as_deref(), Some("worker0:2222"));

        let ps = d.query("ps").expect("mlp query(ps) should succeed");
        assert_eq!(ps.get(&0).expect("ps[0] should be present"), "ps0:3333");

        let workers = d.query("worker").expect("mlp query(worker) should succeed");
        assert_eq!(
            workers.get(&0).expect("worker[0] should be present"),
            "worker0:2222"
        );
        assert_eq!(
            workers.get(&1).expect("worker[1] should be present"),
            "worker1:2223"
        );

        PyServiceDiscovery::deregister(&d, "worker", 1, "worker1:2223")
            .expect("mlp deregister(worker,1) should succeed");
        let workers = d
            .query("worker")
            .expect("mlp query(worker) should succeed after deregister");
        assert_eq!(workers.len(), 1);
        assert_eq!(
            workers.get(&0).expect("worker[0] should remain after deregister"),
            "worker0:2222"
        );

        let all = d.query_all().expect("mlp query_all should succeed");
        assert_eq!(
            all.get("ps")
                .expect("query_all should include ps role")
                .get(&0)
                .expect("ps[0] should be present in query_all"),
            "ps0:3333"
        );
        assert_eq!(
            all.get("chief")
                .expect("query_all should include chief role")
                .get(&0)
                .expect("chief[0] should be present in query_all"),
            "chief0:4444"
        );

        d.deregister_all();
        let all = d
            .query_all()
            .expect("mlp query_all should succeed after deregister_all");
        assert!(
            all.get("ps")
                .expect("query_all should include ps role")
                .is_empty()
        );
        assert!(
            all.get("worker")
                .expect("query_all should include worker role")
                .is_empty()
        );
        assert!(
            all.get("chief")
                .expect("query_all should include chief role")
                .is_empty()
        );
    }

    #[test]
    fn test_mlp_register_rejects_unexpected_host() {
        let _guard = MLP_ENV_TEST_MUTEX
            .lock()
            .expect("mlp env test mutex should not be poisoned");
        let _env = install_default_mlp_env();

        let d = MlpServiceDiscovery::new();
        let err = PyServiceDiscovery::register(&d, "ps", 0, "untrusted-host:3333")
            .expect_err("MLP register should reject host mismatches");
        assert!(
            err.to_string().contains("Host mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_mlp_query_requires_non_empty_name() {
        let _guard = MLP_ENV_TEST_MUTEX
            .lock()
            .expect("mlp env test mutex should not be poisoned");
        let _env = install_default_mlp_env();

        let d = MlpServiceDiscovery::new();
        let err = d
            .query("")
            .expect_err("MLP query should reject empty service names");
        assert!(err.to_string().contains("name must be non-empty"));
    }

    #[test]
    fn test_mlp_close_disables_discovery_and_clears_filters() {
        let _guard = MLP_ENV_TEST_MUTEX
            .lock()
            .expect("mlp env test mutex should not be poisoned");
        let _env = install_default_mlp_env();

        let d = MlpServiceDiscovery::new();
        assert!(
            !d.query("worker")
                .expect("mlp query(worker) should succeed before close")
                .is_empty()
        );
        PyServiceDiscovery::deregister(&d, "worker", 1, "worker1:2223")
            .expect("mlp deregister(worker,1) should succeed before close");
        assert_eq!(
            d.query("worker")
                .expect("mlp query(worker) should succeed before close")
                .len(),
            1
        );

        PyServiceDiscovery::close(&d).expect("mlp close should succeed");
        assert!(d.server_type().is_none());
        assert!(d.addr().is_none());
        assert!(
            d.query("worker")
                .expect("mlp query(worker) should succeed after close")
                .is_empty()
        );
        assert!(
            d.query_all()
                .expect("mlp query_all should succeed after close")
                .is_empty()
        );

        // Post-close operations are no-op and should stay non-failing.
        PyServiceDiscovery::register(&d, "worker", 1, "worker1:2223")
            .expect("mlp register should no-op successfully after close");
        PyServiceDiscovery::deregister(&d, "worker", 1, "worker1:2223")
            .expect("mlp deregister should no-op successfully after close");
        assert!(
            d.query("worker")
                .expect("mlp query(worker) should succeed after post-close ops")
                .is_empty()
        );
    }

    #[test]
    fn test_mlp_query_all_only_includes_supported_configured_roles() {
        let _guard = MLP_ENV_TEST_MUTEX
            .lock()
            .expect("mlp env test mutex should not be poisoned");
        let _env = install_mlp_env_without_chief();

        let d = MlpServiceDiscovery::new();
        let all = d
            .query_all()
            .expect("mlp query_all should succeed for configured-role filter test");
        assert!(all.contains_key("worker"));
        assert!(all.contains_key("ps"));
        assert!(
            !all.contains_key("chief"),
            "chief should be omitted when not configured"
        );
        assert!(
            !all.contains_key("trainer"),
            "unsupported role names should not be included"
        );
    }

    #[test]
    fn test_mlp_register_recovers_after_poisoned_filters_mutex() {
        let _guard = MLP_ENV_TEST_MUTEX
            .lock()
            .expect("mlp env test mutex should not be poisoned");
        let _env = install_default_mlp_env();

        let d = std::sync::Arc::new(MlpServiceDiscovery::new());
        let poison_target = std::sync::Arc::clone(&d);
        let join_result = std::thread::spawn(move || {
            let _guard = poison_target
                .filters
                .lock()
                .expect("mlp filters mutex acquisition should succeed before poisoning");
            panic!("poisoning mlp filters mutex for register recovery-path regression");
        })
        .join();
        assert!(
            join_result.is_err(),
            "poisoning thread should panic to poison mlp filters mutex"
        );

        PyServiceDiscovery::register(&*d, "worker", 1, "worker1:2223")
            .expect("mlp register should recover from poisoned filters mutex");
        let workers = d
            .query("worker")
            .expect("mlp query(worker) should succeed after poisoned-mutex register recovery");
        assert_eq!(
            workers.get(&1).expect("worker[1] should remain visible after register recovery"),
            "worker1:2223"
        );
    }

    #[test]
    fn test_mlp_close_recovers_after_poisoned_filters_mutex() {
        let _guard = MLP_ENV_TEST_MUTEX
            .lock()
            .expect("mlp env test mutex should not be poisoned");
        let _env = install_default_mlp_env();

        let d = std::sync::Arc::new(MlpServiceDiscovery::new());
        PyServiceDiscovery::deregister(&*d, "worker", 1, "worker1:2223")
            .expect("mlp deregister should seed filters before poisoned-close recovery test");

        let poison_target = std::sync::Arc::clone(&d);
        let join_result = std::thread::spawn(move || {
            let _guard = poison_target
                .filters
                .lock()
                .expect("mlp filters mutex acquisition should succeed before poisoning");
            panic!("poisoning mlp filters mutex for close recovery-path regression");
        })
        .join();
        assert!(
            join_result.is_err(),
            "poisoning thread should panic to poison mlp filters mutex"
        );

        PyServiceDiscovery::close(&*d)
            .expect("mlp close should recover from poisoned filters mutex and succeed");
        assert!(
            d.query("worker")
                .expect("mlp query(worker) should succeed after poisoned-mutex close recovery")
                .is_empty(),
            "closed MLP discovery should remain empty after poisoned-mutex close recovery"
        );
    }
}
