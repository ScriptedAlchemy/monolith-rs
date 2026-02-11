//! Python `monolith.native_training.service_discovery` parity adapters.
//!
//! The Python code provides:
//! - Consul-based registration/query keyed by a `consul_id` and user `name/index`.
//! - TF_CONFIG (Primus) discovery.
//! - ZooKeeper-based registration/query under `/monolith/<job_name>`.
//!
//! The Rust crate also has a more general `discovery` module. This file focuses
//! on matching the Python semantics + error strings as exercised by tests.

use std::collections::{BTreeMap, HashMap};
use std::sync::{atomic::AtomicBool, atomic::AtomicU64, atomic::Ordering, Arc, Mutex};
use std::time::Duration;

use crate::native_training::consul as bd_consul;

#[derive(Debug, thiserror::Error)]
pub enum ServiceDiscoveryError {
    #[error("Invalid addr: {0}")]
    InvalidAddr(String),

    #[error("This machine is blacklisted by consul.")]
    ConsulBlacklisted,

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, ServiceDiscoveryError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ServiceDiscoveryType {
    Primus = 1,
    Consul = 2,
    Zk = 3,
    Mlp = 4,
}

pub trait ServiceDiscovery: Send + Sync {
    fn register(&self, name: &str, index: i32, addr: &str) -> Result<()>;
    fn deregister(&self, name: &str, index: i32, addr: &str) -> Result<()>;
    fn query(&self, name: &str) -> Result<BTreeMap<i32, String>>;
    fn close(&self) -> Result<()> {
        Ok(())
    }
}

fn parse_host_port(addr: &str) -> Result<(String, u16)> {
    let parts: Vec<&str> = addr.split(':').collect();
    if parts.len() != 2 {
        return Err(ServiceDiscoveryError::InvalidAddr(addr.to_string()));
    }
    let host = parts[0].to_string();
    let port: u16 = parts[1]
        .parse()
        .map_err(|_| ServiceDiscoveryError::InvalidAddr(addr.to_string()))?;
    Ok((host, port))
}

// =============================================================================
// Consul discovery
// =============================================================================

/// Maximum random backoff in Python. We keep this a configurable global so
/// tests can set it to 0 (mirroring unittest mocks).
static RETRY_MAX_BACKOFF_SECS: AtomicU64 = AtomicU64::new(5);

pub fn set_retry_max_backoff_secs_for_tests(secs: u64) {
    RETRY_MAX_BACKOFF_SECS.store(secs, Ordering::SeqCst);
}

fn retry_with_socket_error<T>(mut f: impl FnMut() -> std::io::Result<T>) -> std::io::Result<T> {
    let tries = 5;
    for i in 0..tries {
        match f() {
            Ok(v) => return Ok(v),
            Err(e) => {
                if i < tries - 1 {
                    let backoff = RETRY_MAX_BACKOFF_SECS.load(Ordering::SeqCst);
                    if backoff > 0 {
                        std::thread::sleep(Duration::from_secs(backoff));
                    }
                    continue;
                }
                return Err(e);
            }
        }
    }
    unreachable!()
}

pub trait ConsulLike: Send + Sync {
    fn lookup(&self, name: &str, timeout_secs: u64) -> std::io::Result<Vec<serde_json::Value>>;
    fn register(
        &self,
        name: &str,
        port: u16,
        tags: &HashMap<String, String>,
    ) -> std::io::Result<()>;
    fn deregister(&self, name: &str, port: u16) -> std::io::Result<()>;
}

#[derive(Debug, Clone)]
struct RealConsul {
    inner: bd_consul::Client,
}

impl RealConsul {
    fn new() -> Self {
        Self {
            inner: bd_consul::Client::new(),
        }
    }
}

impl ConsulLike for RealConsul {
    fn lookup(&self, name: &str, timeout_secs: u64) -> std::io::Result<Vec<serde_json::Value>> {
        self.inner.lookup(name, timeout_secs, 0)
    }

    fn register(
        &self,
        name: &str,
        port: u16,
        tags: &HashMap<String, String>,
    ) -> std::io::Result<()> {
        self.inner
            .register(name, port, Some(tags), None, None)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }

    fn deregister(&self, name: &str, port: u16) -> std::io::Result<()> {
        self.inner
            .deregister(name, port, None)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))
    }
}

#[derive(Clone)]
pub struct ConsulServiceDiscovery {
    consul_id: String,
    client: Arc<dyn ConsulLike>,
    closed: Arc<AtomicBool>,
    retry_time_secs: f64,
    max_replace_retries: usize,
}

impl std::fmt::Debug for ConsulServiceDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsulServiceDiscovery")
            .field("consul_id", &self.consul_id)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("retry_time_secs", &self.retry_time_secs)
            .field("max_replace_retries", &self.max_replace_retries)
            .finish_non_exhaustive()
    }
}

impl ConsulServiceDiscovery {
    pub fn new(consul_id: impl Into<String>) -> Self {
        Self {
            consul_id: consul_id.into(),
            client: Arc::new(RealConsul::new()),
            closed: Arc::new(AtomicBool::new(false)),
            retry_time_secs: 3.0,
            max_replace_retries: 60,
        }
    }

    pub fn with_retry_time_secs(mut self, secs: f64) -> Self {
        self.retry_time_secs = secs;
        self
    }

    pub fn with_client(mut self, client: Arc<dyn ConsulLike>) -> Self {
        self.client = client;
        self
    }

    pub fn with_max_replace_retries(mut self, retries: usize) -> Self {
        self.max_replace_retries = retries.max(1);
        self
    }

    pub fn query_all(&self) -> Result<HashMap<String, BTreeMap<i32, String>>> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ServiceDiscoveryError::Other(
                "ConsulServiceDiscovery is closed".to_string(),
            ));
        }
        let elements = retry_with_socket_error(|| self.client.lookup(&self.consul_id, 15))?;
        let mut addrs: HashMap<String, BTreeMap<i32, String>> = HashMap::new();

        for (entry_idx, element) in elements.into_iter().enumerate() {
            let port = element
                .get("Port")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| {
                    ServiceDiscoveryError::Other(format!(
                        "Malformed consul lookup response at entry {entry_idx}: missing Port"
                    ))
                })? as u16;

            let tags = element.get("Tags").and_then(|v| v.as_object()).ok_or_else(|| {
                ServiceDiscoveryError::Other(format!(
                    "Malformed consul lookup response at entry {entry_idx}: missing Tags"
                ))
            })?;

            let name = tags
                .get("name")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    ServiceDiscoveryError::Other(format!(
                        "Malformed consul lookup response at entry {entry_idx}: missing tag 'name'"
                    ))
                })?
                .to_string();
            let ip = tags.get("ip").and_then(|v| v.as_str()).ok_or_else(|| {
                ServiceDiscoveryError::Other(format!(
                    "Malformed consul lookup response at entry {entry_idx}: missing tag 'ip'"
                ))
            })?;
            let index: i32 = tags
                .get("index")
                .and_then(|v| v.as_i64())
                .or_else(|| tags.get("index").and_then(|v| v.as_str()?.parse::<i64>().ok()))
                .ok_or_else(|| {
                    ServiceDiscoveryError::Other(format!(
                        "Malformed consul lookup response at entry {entry_idx}: missing/invalid tag 'index'"
                    ))
                })? as i32;
            let addr = format!("{ip}:{port}");
            addrs.entry(name).or_default().insert(index, addr);
        }

        Ok(addrs)
    }
}

impl ServiceDiscovery for ConsulServiceDiscovery {
    fn register(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ServiceDiscoveryError::Other(
                "ConsulServiceDiscovery is closed".to_string(),
            ));
        }
        // Best-effort: deregister any existing address for the same (name, index).
        let mut replace_retries = 0usize;
        loop {
            let map = self.query(name)?;
            if let Some(old) = map.get(&index).cloned() {
                if old == addr {
                    // Idempotent register: desired address is already visible.
                    return Ok(());
                }
                replace_retries += 1;
                if replace_retries > self.max_replace_retries {
                    return Err(ServiceDiscoveryError::Other(format!(
                        "Timed out clearing existing consul registration for {name}.{index}"
                    )));
                }
                self.deregister(name, index, &old)?;
            } else {
                break;
            }
            if self.retry_time_secs > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(self.retry_time_secs));
            }
        }

        let (host, port) = parse_host_port(addr)?;
        let mut tags: HashMap<String, String> = HashMap::new();
        tags.insert("index".to_string(), index.to_string());
        tags.insert("name".to_string(), name.to_string());
        tags.insert("ip".to_string(), host);

        retry_with_socket_error(|| self.client.register(&self.consul_id, port, &tags))?;

        // Wait until the registration is visible, or treat as "blacklisted".
        let backoff = RETRY_MAX_BACKOFF_SECS.load(Ordering::SeqCst).max(1);
        let max_retries = std::cmp::max(2, 180 / backoff);
        let mut retries = 0_u64;
        loop {
            let map = self.query(name)?;
            if map.contains_key(&index) {
                break;
            }
            retries += 1;
            if retries > max_retries {
                return Err(ServiceDiscoveryError::ConsulBlacklisted);
            }
            let sleep_s = RETRY_MAX_BACKOFF_SECS.load(Ordering::SeqCst);
            if sleep_s > 0 {
                std::thread::sleep(Duration::from_secs(sleep_s));
            }
        }

        Ok(())
    }

    fn deregister(&self, _name: &str, _index: i32, addr: &str) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ServiceDiscoveryError::Other(
                "ConsulServiceDiscovery is closed".to_string(),
            ));
        }
        let (_host, port) = parse_host_port(addr)?;
        retry_with_socket_error(|| self.client.deregister(&self.consul_id, port))?;
        Ok(())
    }

    fn query(&self, name: &str) -> Result<BTreeMap<i32, String>> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ServiceDiscoveryError::Other(
                "ConsulServiceDiscovery is closed".to_string(),
            ));
        }
        let all = self.query_all()?;
        Ok(all.get(name).cloned().unwrap_or_default())
    }

    fn close(&self) -> Result<()> {
        self.closed.store(true, Ordering::SeqCst);
        Ok(())
    }
}

pub fn deregister_all(consul_id: &str) -> Result<()> {
    let discovery = ConsulServiceDiscovery::new(consul_id.to_string());
    let named = discovery.query_all()?;
    for (name, addrs) in named {
        for (idx, addr) in addrs {
            discovery.deregister(&name, idx, &addr)?;
        }
    }
    Ok(())
}

// =============================================================================
// TF_CONFIG discovery: reuse existing adapter
// =============================================================================

pub use crate::py_discovery::TfConfigServiceDiscovery;

// =============================================================================
// ZK discovery (in-memory / pluggable)
// =============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkState {
    Lost,
    Suspended,
    Connected,
}

pub trait ZkClientLike: Send + Sync {
    fn ensure_path(&self, path: &str) -> std::io::Result<()>;
    fn start(&self) -> std::io::Result<()>;
    fn stop(&self);
    fn close(&self);
    fn add_listener(&self, f: Arc<dyn Fn(ZkState) + Send + Sync>);

    fn create_ephemeral(&self, path: &str, value: &[u8], makepath: bool) -> std::io::Result<()>;
    fn set(&self, path: &str, value: &[u8]) -> std::io::Result<()>;
    fn delete_recursive(&self, path: &str) -> std::io::Result<()>;
    fn get(&self, path: &str) -> std::io::Result<Option<Vec<u8>>>;
    fn get_children(&self, path: &str) -> std::io::Result<Vec<String>>;
}

pub struct ZkServiceDiscovery {
    path_prefix: String,
    client: Arc<dyn ZkClientLike>,
    threads: Arc<Mutex<HashMap<(String, i32), Arc<ZkRegThread>>>>,
    closed: AtomicBool,
    max_tries: usize,
    delay_secs: u64,
}

impl std::fmt::Debug for ZkServiceDiscovery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZkServiceDiscovery")
            .field("path_prefix", &self.path_prefix)
            .field("closed", &self.closed.load(Ordering::SeqCst))
            .field("max_tries", &self.max_tries)
            .field("delay_secs", &self.delay_secs)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct ZkRegThread {
    stop: std::sync::atomic::AtomicBool,
    wakeup: std::sync::Condvar,
    mu: Mutex<bool>,
    handle: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl ZkRegThread {
    fn new() -> Self {
        Self {
            stop: std::sync::atomic::AtomicBool::new(false),
            wakeup: std::sync::Condvar::new(),
            mu: Mutex::new(false),
            handle: Mutex::new(None),
        }
    }

    fn request_wakeup(&self) {
        let mut g = self.mu.lock().unwrap();
        *g = true;
        self.wakeup.notify_all();
    }

    fn stop_and_join(&self) {
        self.stop.store(true, Ordering::SeqCst);
        self.request_wakeup();
        if let Some(h) = self.handle.lock().unwrap().take() {
            let _ = h.join();
        }
    }
}

static ZK_REGISTRATION_PERIOD_MS: AtomicU64 = AtomicU64::new(30 * 60 * 1000);

pub fn set_zk_registration_period_ms_for_tests(ms: u64) {
    ZK_REGISTRATION_PERIOD_MS.store(ms, Ordering::SeqCst);
}

impl ZkServiceDiscovery {
    pub fn new(job_name: &str, client: Arc<dyn ZkClientLike>) -> Result<Self> {
        let path_prefix = format!("/monolith/{job_name}");
        client.start()?;
        client.ensure_path(&path_prefix)?;

        let sd = Self {
            path_prefix,
            client,
            threads: Arc::new(Mutex::new(HashMap::new())),
            closed: AtomicBool::new(false),
            max_tries: 3,
            delay_secs: 5,
        };
        sd.install_listener();
        Ok(sd)
    }

    pub fn with_retry(mut self, max_tries: usize, delay_secs: u64) -> Self {
        self.max_tries = max_tries;
        self.delay_secs = delay_secs;
        self
    }

    fn install_listener(&self) {
        let threads = Arc::clone(&self.threads);
        let f = Arc::new(move |state: ZkState| {
            // Match Python's semantics used by tests: on reconnect after lost,
            // best-effort wake periodic registration threads.
            if state == ZkState::Connected {
                let guard = threads.lock().unwrap();
                for ts in guard.values() {
                    ts.request_wakeup();
                }
            }
        });
        self.client.add_listener(f);
    }

    fn node_name(server_type: &str, index: i32) -> String {
        format!("{server_type}.{index}")
    }

    fn path(&self, server_type: &str, index: i32) -> String {
        format!(
            "{}/{}",
            self.path_prefix,
            Self::node_name(server_type, index)
        )
    }

    fn try_create_or_set(&self, path: &str, value: &str) -> std::io::Result<()> {
        let v = value.as_bytes();
        match self.client.create_ephemeral(path, v, true) {
            Ok(_) => Ok(()),
            Err(_) => self.client.set(path, v),
        }
    }

    fn internal_register(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        let path = self.path(name, index);
        let mut last_err: Option<std::io::Error> = None;
        for _ in 0..self.max_tries {
            match self.try_create_or_set(&path, addr) {
                Ok(_) => return Ok(()),
                Err(e) => {
                    last_err = Some(e);
                    if self.delay_secs > 0 {
                        std::thread::sleep(Duration::from_secs(self.delay_secs));
                    }
                }
            }
        }
        Err(ServiceDiscoveryError::Io(last_err.unwrap_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::Other, "zk register failed")
        })))
    }
}

impl ServiceDiscovery for ZkServiceDiscovery {
    fn register(&self, name: &str, index: i32, addr: &str) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ServiceDiscoveryError::Other(
                "ZkServiceDiscovery is closed".to_string(),
            ));
        }
        self.internal_register(name, index, addr)?;

        // Spawn periodic re-registration thread (best-effort), mirroring Python.
        let key = (name.to_string(), index);
        let old = {
            let mut threads = self.threads.lock().unwrap();
            threads.remove(&key)
        };
        if let Some(old) = old {
            // Replace existing without holding the shared map lock across join.
            old.stop_and_join();
        }

        let ts = Arc::new(ZkRegThread::new());
        let client = Arc::clone(&self.client);
        let path = self.path(name, index);
        let addr = addr.to_string();
        let max_tries = self.max_tries;
        let delay_secs = self.delay_secs;
        let ts2 = Arc::clone(&ts);
        let h = std::thread::spawn(move || loop {
            // Wake up periodically or on explicit wakeup.
            let period_ms = ZK_REGISTRATION_PERIOD_MS.load(Ordering::SeqCst);
            let mut w = ts2.mu.lock().unwrap();
            if !*w {
                if period_ms == 0 {
                    // Busy-looping can be expensive; sleep a tiny bit between attempts.
                    let (wg, _) = ts2
                        .wakeup
                        .wait_timeout(w, Duration::from_millis(10))
                        .unwrap();
                    w = wg;
                } else {
                    // Guard against spurious wakeups: only proceed on explicit wake or timeout.
                    let mut timed_out = false;
                    while !*w && !ts2.stop.load(Ordering::SeqCst) && !timed_out {
                        let (wg, res) = ts2
                            .wakeup
                            .wait_timeout(w, Duration::from_millis(period_ms))
                            .unwrap();
                        w = wg;
                        timed_out = res.timed_out();
                    }
                }
            }
            *w = false;
            if ts2.stop.load(Ordering::SeqCst) {
                break;
            }

            // Best-effort retry loop.
            let mut last: Option<std::io::Error> = None;
            for _ in 0..max_tries {
                let v = addr.as_bytes();
                let res = match client.create_ephemeral(&path, v, true) {
                    Ok(_) => Ok(()),
                    Err(_) => client.set(&path, v),
                };
                match res {
                    Ok(_) => {
                        last = None;
                        break;
                    }
                    Err(e) => {
                        last = Some(e);
                        if delay_secs > 0 {
                            std::thread::sleep(Duration::from_secs(delay_secs));
                        }
                    }
                }
            }
            let _ = last;
        });
        *ts.handle.lock().unwrap() = Some(h);
        self.threads.lock().unwrap().insert(key, ts);
        Ok(())
    }

    fn deregister(&self, name: &str, index: i32, _addr: &str) -> Result<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ServiceDiscoveryError::Other(
                "ZkServiceDiscovery is closed".to_string(),
            ));
        }
        let path = self.path(name, index);
        let _ = self.client.delete_recursive(&path);

        let key = (name.to_string(), index);
        let removed = self.threads.lock().unwrap().remove(&key);
        if let Some(ts) = removed {
            ts.stop_and_join();
        }
        Ok(())
    }

    fn query(&self, name: &str) -> Result<BTreeMap<i32, String>> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(ServiceDiscoveryError::Other(
                "ZkServiceDiscovery is closed".to_string(),
            ));
        }
        let children = self.client.get_children(&self.path_prefix)?;
        let mut out = BTreeMap::new();
        for child in children {
            if !child.starts_with(&format!("{name}.")) {
                continue;
            }
            let idx = child
                .split_once('.')
                .and_then(|(_, n)| n.parse::<i32>().ok())
                .unwrap_or(0);
            let path = format!("{}/{}", self.path_prefix, child);
            if let Some(data) = self.client.get(&path)? {
                if !data.is_empty() {
                    if let Ok(addr) = String::from_utf8(data) {
                        out.insert(idx, addr);
                    }
                }
            }
        }
        Ok(out)
    }

    fn close(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(());
        }
        let drained = {
            let mut threads = self.threads.lock().unwrap();
            threads.drain().map(|(_, ts)| ts).collect::<Vec<_>>()
        };
        for ts in drained {
            ts.stop_and_join();
        }
        self.client.stop();
        self.client.close();
        Ok(())
    }
}

impl Drop for ZkServiceDiscovery {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::py_discovery::PyServiceDiscovery;
    use std::sync::Mutex;
    use std::sync::RwLock;
    // ZK registration period is a global; keep tests that modify it serialized.
    static ZK_PERIOD_TEST_MUTEX: Mutex<()> = Mutex::new(());

    // Mirrors Python `FakeConsul`.
    #[derive(Debug)]
    struct FakeConsul {
        name_to_dict: Mutex<HashMap<String, HashMap<String, serde_json::Value>>>,
        blacklist: Vec<String>,
    }

    impl FakeConsul {
        fn new(blacklist: Vec<String>) -> Self {
            Self {
                name_to_dict: Mutex::new(HashMap::new()),
                blacklist,
            }
        }
    }

    impl ConsulLike for FakeConsul {
        fn lookup(
            &self,
            name: &str,
            _timeout_secs: u64,
        ) -> std::io::Result<Vec<serde_json::Value>> {
            let guard = self.name_to_dict.lock().unwrap();
            Ok(guard
                .get(name)
                .map(|m| m.values().cloned().collect())
                .unwrap_or_default())
        }

        fn register(
            &self,
            name: &str,
            port: u16,
            tags: &HashMap<String, String>,
        ) -> std::io::Result<()> {
            if self
                .blacklist
                .iter()
                .any(|ip| ip == tags.get("ip").unwrap_or(&"".to_string()))
            {
                return Ok(());
            }
            let host = tags.get("ip").cloned().unwrap_or_default();
            let key = format!("{host}:{port}");
            let mut map = self.name_to_dict.lock().unwrap();
            let entry = map.entry(name.to_string()).or_default();
            entry.insert(
                key.clone(),
                serde_json::json!({"Host": host, "Port": port, "Tags": tags}),
            );
            Ok(())
        }

        fn deregister(&self, name: &str, port: u16) -> std::io::Result<()> {
            // Host isn't passed; match FakeConsul's behavior by removing any key with port.
            let mut map = self.name_to_dict.lock().unwrap();
            if let Some(d) = map.get_mut(name) {
                let to_remove: Vec<String> = d
                    .keys()
                    .filter(|k| {
                        k.rsplit_once(':').and_then(|(_, p)| p.parse::<u16>().ok()) == Some(port)
                    })
                    .cloned()
                    .collect();
                for k in to_remove {
                    d.remove(&k);
                }
            }
            Ok(())
        }
    }

    #[test]
    fn consul_service_discovery_basic() {
        let c = Arc::new(FakeConsul::new(vec![]));
        let d = ConsulServiceDiscovery::new("unique_id").with_client(c);
        d.register("server", 0, "192.168.0.1:1001").unwrap();
        d.register("server", 1, "192.168.0.2:1002").unwrap();
        assert_eq!(
            d.query("server").unwrap(),
            BTreeMap::from([
                (0, "192.168.0.1:1001".to_string()),
                (1, "192.168.0.2:1002".to_string())
            ])
        );
        d.deregister("server", 0, "192.168.0.1:1001").unwrap();
        d.deregister("server", 1, "192.168.0.2:1002").unwrap();
        assert_eq!(d.query("server").unwrap(), BTreeMap::new());
    }

    #[test]
    fn consul_duplicate_registration_overwrites() {
        let c = Arc::new(FakeConsul::new(vec![]));
        let d = ConsulServiceDiscovery::new("unique_id")
            .with_retry_time_secs(0.0)
            .with_client(c);
        d.register("server", 0, "192.168.0.1:1001").unwrap();
        d.register("server", 0, "192.168.0.1:1002").unwrap();
        assert_eq!(
            d.query("server").unwrap(),
            BTreeMap::from([(0, "192.168.0.1:1002".to_string())])
        );
    }

    #[test]
    fn consul_idempotent_registration_short_circuits_when_addr_already_visible() {
        let c = Arc::new(FakeConsul::new(vec![]));
        let d = ConsulServiceDiscovery::new("unique_id")
            .with_retry_time_secs(0.0)
            .with_client(c);
        d.register("server", 0, "192.168.0.1:1001").unwrap();
        d.register("server", 0, "192.168.0.1:1001").unwrap();
        assert_eq!(
            d.query("server").unwrap(),
            BTreeMap::from([(0, "192.168.0.1:1001".to_string())])
        );
    }

    #[test]
    fn consul_multi_names() {
        let c = Arc::new(FakeConsul::new(vec![]));
        let d = ConsulServiceDiscovery::new("unique_id").with_client(c);
        d.register("ps", 0, "192.168.0.1:1001").unwrap();
        d.register("worker", 0, "192.168.0.1:1002").unwrap();
        assert_eq!(
            d.query("worker").unwrap(),
            BTreeMap::from([(0, "192.168.0.1:1002".to_string())])
        );
    }

    #[test]
    fn consul_retry_propagates_error() {
        struct ErrClient;
        impl ConsulLike for ErrClient {
            fn lookup(
                &self,
                _name: &str,
                _timeout_secs: u64,
            ) -> std::io::Result<Vec<serde_json::Value>> {
                Ok(Vec::new())
            }
            fn register(
                &self,
                _name: &str,
                _port: u16,
                _tags: &HashMap<String, String>,
            ) -> std::io::Result<()> {
                Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "timeout"))
            }
            fn deregister(&self, _name: &str, _port: u16) -> std::io::Result<()> {
                Ok(())
            }
        }

        set_retry_max_backoff_secs_for_tests(0);
        let d = ConsulServiceDiscovery::new("unique_id").with_client(Arc::new(ErrClient));
        let err = d.register("ps", 0, "192.168.0.1:1001").unwrap_err();
        match err {
            ServiceDiscoveryError::Io(e) => assert_eq!(e.kind(), std::io::ErrorKind::TimedOut),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn consul_registration_failed_blacklisted_message() {
        set_retry_max_backoff_secs_for_tests(0);
        let c = Arc::new(FakeConsul::new(vec!["192.168.0.1".to_string()]));
        let d = ConsulServiceDiscovery::new("unique_id").with_client(c);
        let err = d.register("ps", 0, "192.168.0.1:1001").unwrap_err();
        assert_eq!(err.to_string(), "This machine is blacklisted by consul.");
    }

    #[test]
    fn consul_query_all_rejects_malformed_entries() {
        struct MalformedLookupConsul;
        impl ConsulLike for MalformedLookupConsul {
            fn lookup(
                &self,
                _name: &str,
                _timeout_secs: u64,
            ) -> std::io::Result<Vec<serde_json::Value>> {
                Ok(vec![serde_json::json!({
                    "Port": 1001,
                    "Tags": {"name": "ps", "ip": "127.0.0.1"}
                })])
            }

            fn register(
                &self,
                _name: &str,
                _port: u16,
                _tags: &HashMap<String, String>,
            ) -> std::io::Result<()> {
                Ok(())
            }

            fn deregister(&self, _name: &str, _port: u16) -> std::io::Result<()> {
                Ok(())
            }
        }

        let d = ConsulServiceDiscovery::new("unique_id").with_client(Arc::new(MalformedLookupConsul));
        let err = d.query_all().unwrap_err();
        assert!(
            err.to_string().contains("missing/invalid tag 'index'"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn consul_register_times_out_when_old_registration_never_clears() {
        #[derive(Debug)]
        struct StickyLookupConsul;
        impl ConsulLike for StickyLookupConsul {
            fn lookup(
                &self,
                _name: &str,
                _timeout_secs: u64,
            ) -> std::io::Result<Vec<serde_json::Value>> {
                let tags = serde_json::json!({
                    "name": "ps",
                    "index": "0",
                    "ip": "192.168.0.99"
                });
                Ok(vec![serde_json::json!({
                    "Host": "192.168.0.99",
                    "Port": 1001,
                    "Tags": tags
                })])
            }

            fn register(
                &self,
                _name: &str,
                _port: u16,
                _tags: &HashMap<String, String>,
            ) -> std::io::Result<()> {
                Ok(())
            }

            fn deregister(&self, _name: &str, _port: u16) -> std::io::Result<()> {
                // Simulate stale visibility / eventual consistency that never resolves.
                Ok(())
            }
        }

        let d = ConsulServiceDiscovery::new("unique_id")
            .with_retry_time_secs(0.0)
            .with_max_replace_retries(2)
            .with_client(Arc::new(StickyLookupConsul));
        let err = d.register("ps", 0, "192.168.0.1:1001").unwrap_err();
        assert!(
            err.to_string()
                .contains("Timed out clearing existing consul registration"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn consul_close_is_idempotent_and_blocks_operations() {
        let c = Arc::new(FakeConsul::new(vec![]));
        let d = ConsulServiceDiscovery::new("unique_id")
            .with_retry_time_secs(0.0)
            .with_client(c);
        d.close().unwrap();
        d.close().unwrap();

        let register_err = d.register("ps", 0, "192.168.0.1:1001").unwrap_err();
        assert!(register_err.to_string().contains("closed"));

        let query_err = d.query("ps").unwrap_err();
        assert!(query_err.to_string().contains("closed"));

        let query_all_err = d.query_all().unwrap_err();
        assert!(query_all_err.to_string().contains("closed"));

        let deregister_err = d.deregister("ps", 0, "192.168.0.1:1001").unwrap_err();
        assert!(deregister_err.to_string().contains("closed"));
    }

    #[test]
    fn consul_close_state_is_shared_across_clones() {
        let c = Arc::new(FakeConsul::new(vec![]));
        let d1 = ConsulServiceDiscovery::new("unique_id")
            .with_retry_time_secs(0.0)
            .with_client(c);
        let d2 = d1.clone();

        d1.close().unwrap();
        let err = d2.query_all().unwrap_err();
        assert!(err.to_string().contains("closed"));
    }

    #[test]
    fn tf_config_service_discovery_matches_python_test() {
        let tf_conf = serde_json::json!({
          "cluster": {
            "chief": ["host0:2222"],
            "ps": ["host1:2222", "host2:2222"],
            "worker": ["host3:2222", "host4:2222", "host5:2222"]
          },
          "task": {"type": "worker", "index": 1}
        });
        let d = TfConfigServiceDiscovery::new(&tf_conf.to_string()).unwrap();
        let ps = d.query("ps").unwrap();
        assert_eq!(ps.get(&0).unwrap(), "host1:2222");
        assert_eq!(ps.get(&1).unwrap(), "host2:2222");
        let wk = d.query("worker").unwrap();
        assert_eq!(wk.get(&0).unwrap(), "host0:2222");
        assert_eq!(wk.get(&1).unwrap(), "host3:2222");
        assert_eq!(wk.get(&2).unwrap(), "host4:2222");
        assert_eq!(d.addr().unwrap(), "host4:2222");
        assert_eq!(d.server_type(), "worker");
        assert_eq!(d.index(), 2);
    }

    // Mirrors Python FakeKazooClient with a minimal feature set needed by our ZkServiceDiscovery.
    struct FakeZk {
        data: RwLock<HashMap<String, Vec<u8>>>,
        listeners: Mutex<Vec<Arc<dyn Fn(ZkState) + Send + Sync>>>,
    }

    impl FakeZk {
        fn new() -> Self {
            Self {
                data: RwLock::new(HashMap::new()),
                listeners: Mutex::new(Vec::new()),
            }
        }

        fn emit(&self, st: ZkState) {
            for l in self.listeners.lock().unwrap().iter() {
                l(st);
            }
        }
    }

    impl ZkClientLike for FakeZk {
        fn ensure_path(&self, path: &str) -> std::io::Result<()> {
            let mut d = self.data.write().unwrap();
            d.entry(path.to_string()).or_default();
            Ok(())
        }

        fn start(&self) -> std::io::Result<()> {
            Ok(())
        }

        fn stop(&self) {}
        fn close(&self) {}

        fn add_listener(&self, f: Arc<dyn Fn(ZkState) + Send + Sync>) {
            self.listeners.lock().unwrap().push(f);
        }

        fn create_ephemeral(
            &self,
            path: &str,
            value: &[u8],
            makepath: bool,
        ) -> std::io::Result<()> {
            if makepath {
                if let Some(parent) = std::path::Path::new(path).parent().and_then(|p| p.to_str()) {
                    self.ensure_path(parent)?;
                }
            }
            let mut d = self.data.write().unwrap();
            if d.contains_key(path) {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    "exists",
                ));
            }
            d.insert(path.to_string(), value.to_vec());
            Ok(())
        }

        fn set(&self, path: &str, value: &[u8]) -> std::io::Result<()> {
            let mut d = self.data.write().unwrap();
            if !d.contains_key(path) {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
            }
            d.insert(path.to_string(), value.to_vec());
            Ok(())
        }

        fn delete_recursive(&self, path: &str) -> std::io::Result<()> {
            let mut d = self.data.write().unwrap();
            let keys: Vec<String> = d.keys().filter(|k| k.starts_with(path)).cloned().collect();
            if keys.is_empty() {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
            }
            for k in keys {
                d.remove(&k);
            }
            Ok(())
        }

        fn get(&self, path: &str) -> std::io::Result<Option<Vec<u8>>> {
            let d = self.data.read().unwrap();
            Ok(d.get(path).cloned())
        }

        fn get_children(&self, path: &str) -> std::io::Result<Vec<String>> {
            let d = self.data.read().unwrap();
            if !d.contains_key(path) {
                return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "missing"));
            }
            let prefix = format!("{}/", path.trim_end_matches('/'));
            let mut out = Vec::new();
            for k in d.keys() {
                if k.starts_with(&prefix) {
                    let rest = &k[prefix.len()..];
                    let child = rest.split('/').next().unwrap_or("");
                    if !child.is_empty() {
                        out.push(child.to_string());
                    }
                }
            }
            out.sort();
            out.dedup();
            Ok(out)
        }
    }

    // Make the trait-object coercion used by tests work: provide a default `as_any`
    // for all implementors via a helper trait.

    #[test]
    fn zk_basic_and_duplicate_and_multi_names() {
        let c = Arc::new(FakeZk::new());
        let d =
            ZkServiceDiscovery::new("test_model", Arc::clone(&c) as Arc<dyn ZkClientLike>).unwrap();
        d.register("server", 0, "192.168.0.1:1001").unwrap();
        d.register("server", 1, "192.168.0.2:1002").unwrap();
        assert_eq!(
            d.query("server").unwrap(),
            BTreeMap::from([
                (0, "192.168.0.1:1001".to_string()),
                (1, "192.168.0.2:1002".to_string())
            ])
        );
        d.deregister("server", 0, "192.168.0.1:1001").unwrap();
        d.deregister("server", 1, "192.168.0.2:1002").unwrap();
        assert_eq!(d.query("server").unwrap(), BTreeMap::new());

        // Duplicate overwrites.
        d.register("server", 0, "192.168.0.1:1001").unwrap();
        d.register("server", 0, "192.168.0.1:1002").unwrap();
        assert_eq!(
            d.query("server").unwrap(),
            BTreeMap::from([(0, "192.168.0.1:1002".to_string())])
        );

        // Multi names.
        d.register("ps", 0, "192.168.0.1:1001").unwrap();
        d.register("worker", 0, "192.168.0.1:1002").unwrap();
        assert_eq!(
            d.query("worker").unwrap(),
            BTreeMap::from([(0, "192.168.0.1:1002".to_string())])
        );

        d.close().unwrap();
    }

    #[test]
    fn zk_periodic_registration_repairs_data() {
        let _guard = ZK_PERIOD_TEST_MUTEX.lock().unwrap();
        set_zk_registration_period_ms_for_tests(10);
        let c = Arc::new(FakeZk::new());
        let d =
            ZkServiceDiscovery::new("test_model", Arc::clone(&c) as Arc<dyn ZkClientLike>).unwrap();
        d.register("ps", 0, "192.168.0.1:1001").unwrap();
        // Corrupt the node.
        c.set("/monolith/test_model/ps.0", b"wrongdata").unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(
            d.query("ps").unwrap(),
            BTreeMap::from([(0, "192.168.0.1:1001".to_string())])
        );
        d.close().unwrap();
    }

    #[test]
    fn zk_listener_wakeup_does_not_break_registration() {
        let _guard = ZK_PERIOD_TEST_MUTEX.lock().unwrap();
        set_zk_registration_period_ms_for_tests(60_000);
        let c = Arc::new(FakeZk::new());
        let d =
            ZkServiceDiscovery::new("test_model", Arc::clone(&c) as Arc<dyn ZkClientLike>).unwrap();
        d.register("ps", 0, "192.168.0.1:1001").unwrap();
        c.emit(ZkState::Lost);
        c.emit(ZkState::Connected);
        assert_eq!(
            d.query("ps").unwrap(),
            BTreeMap::from([(0, "192.168.0.1:1001".to_string())])
        );
        d.close().unwrap();
    }

    #[test]
    fn zk_close_is_idempotent_after_deregister() {
        let c = Arc::new(FakeZk::new());
        let d =
            ZkServiceDiscovery::new("test_model", Arc::clone(&c) as Arc<dyn ZkClientLike>).unwrap();
        d.register("ps", 0, "192.168.0.1:1001").unwrap();
        d.deregister("ps", 0, "192.168.0.1:1001").unwrap();
        d.close().unwrap();
        d.close().unwrap();
    }

    #[test]
    fn zk_operations_fail_after_close() {
        let c = Arc::new(FakeZk::new());
        let d =
            ZkServiceDiscovery::new("test_model", Arc::clone(&c) as Arc<dyn ZkClientLike>).unwrap();
        d.close().unwrap();

        let register_err = d.register("ps", 0, "192.168.0.1:1001").unwrap_err();
        assert!(register_err.to_string().contains("closed"));

        let query_err = d.query("ps").unwrap_err();
        assert!(query_err.to_string().contains("closed"));

        let deregister_err = d.deregister("ps", 0, "192.168.0.1:1001").unwrap_err();
        assert!(deregister_err.to_string().contains("closed"));
    }
}
