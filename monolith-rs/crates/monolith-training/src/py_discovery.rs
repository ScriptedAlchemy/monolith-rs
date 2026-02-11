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
    all_roles: HashMap<String, usize>, // uppercase role -> count
}

impl MlpEnv {
    fn from_env() -> Self {
        let role = std::env::var("MLP_ROLE")
            .unwrap_or_default()
            .trim()
            .to_uppercase();

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

        Self { role, all_roles }
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
}

impl MlpServiceDiscovery {
    pub fn new() -> Self {
        Self {
            env: MlpEnv::from_env(),
            filters: Mutex::new(HashSet::new()),
        }
    }

    fn key(name: &str, index: i32) -> String {
        format!("{}:{}", name.trim().to_lowercase(), index)
    }

    fn validate(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        if index < 0 {
            return Err(DiscoveryError::ConfigError(format!(
                "index must be >= 0, got {}",
                index
            )));
        }
        let n = self.env.num_replicas(name);
        if (index as usize) >= n {
            return Err(DiscoveryError::ConfigError(format!(
                "index {} out of bounds for role {:?} (num_replicas={})",
                index, name, n
            )));
        }

        // Best-effort address validation: ensure the port matches the configured port.
        let (_host, port_str) = addr
            .split_once(':')
            .ok_or_else(|| DiscoveryError::ConfigError(format!("Invalid addr: {:?}", addr)))?;
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
        self.validate(name, index, addr)?;
        let mut filters = self.filters.lock().unwrap();
        filters.remove(&Self::key(name, index));
        Ok(())
    }

    fn deregister(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        self.validate(name, index, addr)?;
        let mut filters = self.filters.lock().unwrap();
        filters.insert(Self::key(name, index));
        Ok(())
    }

    fn query(&self, name: &str) -> Result<HashMap<i32, String>> {
        let num = self.env.num_replicas(name);
        if num == 0 {
            return Ok(HashMap::new());
        }

        let filters = self.filters.lock().unwrap();
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
