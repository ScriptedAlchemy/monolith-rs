//! Replica discovery + status updater (Python parity for `monolith/agent_service/replica_manager.py`).
//!
//! The upstream Python agent maintains two cooperating loops:
//! - `ReplicaWatcher`: ZK watches to mirror all replicas for a model, plus a periodic poll
//!   to reconcile stale watch state.
//! - `ReplicaUpdater`: registers this replica's ephemeral nodes, polls TFServing for model
//!   status, and updates its own znode with the current state.
//!
//! For Rust parity we implement the minimal surface used by tests and future integration:
//! - `ReplicaWatcher::watch_data/stop/get_*`
//! - `ReplicaUpdater::register/start/stop`
//! - `ReplicaManager::start/stop` as a small façade.

#![cfg(feature = "grpc")]

use crate::backends::{DataWatchCallback, WatchEventType, ZkClient, ZkError};
use crate::data_def::{AddressFamily, ModelState, ReplicaMeta};
use crate::grpc::ServerType;
use crate::tfs_monitor::{
    AgentConfig as TfsAgentConfig, DeployType as TfsDeployType, TfServerType, TfsMonitor,
};
use monolith_proto::monolith::parameter_sync::client_config::TargetExtraInfo;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;
use tracing::error;

// Python default: DEFAULT_USE_ARCHON = False.
const DEFAULT_USE_ARCHON: bool = false;

fn parse_task_path(path: &str) -> (Option<String>, Option<String>, Option<i32>) {
    // Expect: /{bzid}/service/{base_name}/({idc}:{cluster}/)?{server_type}:{task}
    // We only need server_type and task.
    let p = path.trim_end_matches('/');
    let task_part = p.rsplit('/').next().unwrap_or("");
    let mut it = task_part.split(':');
    let server_type = it.next().map(|s| s.to_string());
    let task = it.next().and_then(|s| s.parse::<i32>().ok());

    // Optional location: the segment before task can be "{idc}:{cluster}" in dc-aware mode.
    let segs = p.split('/').filter(|s| !s.is_empty()).collect::<Vec<_>>();
    if segs.len() < 4 {
        return (None, server_type, task);
    }

    // ... service, base_name, maybe location, task or replica_id
    // Distinguish location from task segment by checking whether the suffix is numeric.
    // - task: "{server}:{index}" where index is all digits
    // - location: "{idc}:{cluster}" where cluster is typically non-numeric
    let location = if segs.len() >= 5 {
        let candidate = segs[3];
        let is_task_like = candidate
            .split_once(':')
            .map(|(_, rhs)| !rhs.is_empty() && rhs.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false);
        if candidate.contains(':') && !is_task_like {
            Some(candidate.to_string())
        } else {
            None
        }
    } else {
        None
    };
    (location, server_type, task)
}

fn should_ship_in(location: &Option<String>, idc: Option<&str>, cluster: Option<&str>) -> bool {
    if idc.is_none() || cluster.is_none() {
        return true;
    }
    let Some(loc) = location.as_deref() else {
        return true;
    };
    let want = format!("{}:{}", idc.unwrap(), cluster.unwrap());
    loc == want
}

/// ZooKeeper connection states (subset) used by the Python agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkConnectionState {
    /// Connection is lost (session expired).
    Lost,
    /// Connection is temporarily suspended.
    Suspended,
    /// Connection is active.
    Connected,
}

/// Mirror remote replicas via watches + periodic reconciliation.
pub struct ReplicaWatcher {
    zk: Arc<dyn ZkClient>,
    conf: WatcherConfig,
    use_archon: bool,
    zk_watch_address_family: &'static str,
    path_prefix: String,

    replicas: Arc<RwLock<HashMap<String, HashMap<String, ReplicaMeta>>>>,
    watch_keys: Arc<Mutex<HashSet<String>>>,
    has_stop: Arc<AtomicBool>,
    should_poll: Arc<AtomicBool>,
    poll_thread: Mutex<Option<JoinHandle<()>>>,
}

/// Configuration for [`ReplicaWatcher`].
#[derive(Clone)]
pub struct WatcherConfig {
    /// Business ID used for ZK paths.
    pub bzid: String,
    /// Base model name under the service subtree.
    pub base_name: String,
    /// Whether the service tree includes a `{idc}:{cluster}` segment.
    pub dc_aware: bool,
    /// Deploy type used to filter tasks.
    pub deploy_type: TfsDeployType,
    /// Whether dense runs as standalone in addition to entry/ps.
    pub dense_alone: bool,
}

impl WatcherConfig {
    /// Watcher prefix is always `/{bzid}/service/{base_name}` (Python parity).
    ///
    /// In dc-aware mode, this contains per-DC subtrees like `{idc}:{cluster}` as children.
    pub fn path_prefix(&self) -> String {
        format!("/{}/service/{}", self.bzid, self.base_name)
    }
}

impl ReplicaWatcher {
    /// Create a new watcher instance.
    pub fn new(
        zk: Arc<dyn ZkClient>,
        conf: WatcherConfig,
        use_archon: bool,
        zk_watch_address_family: &'static str,
    ) -> Self {
        let family = if zk_watch_address_family == AddressFamily::IPV4 {
            AddressFamily::IPV4
        } else {
            AddressFamily::IPV6
        };
        let path_prefix = conf.path_prefix();
        Self {
            zk,
            conf,
            use_archon,
            zk_watch_address_family: family,
            path_prefix,
            replicas: Arc::new(RwLock::new(HashMap::new())),
            watch_keys: Arc::new(Mutex::new(HashSet::new())),
            has_stop: Arc::new(AtomicBool::new(false)),
            should_poll: Arc::new(AtomicBool::new(true)),
            poll_thread: Mutex::new(None),
        }
    }

    /// Install ZK watches and start the background poll loop.
    pub fn watch_data(&self) -> Result<(), ZkError> {
        self.zk.ensure_path(&self.path_prefix)?;
        if self.conf.dc_aware {
            let base = self.path_prefix.clone();
            let zk = self.zk.clone();
            let watcher = self.clone_for_watch();
            self.zk.children_watch(
                &self.path_prefix,
                Arc::new(move |children| {
                    // idc_cluster children
                    for ic in children {
                        let ic_path = Path::new(&base).join(&ic).to_string_lossy().to_string();
                        let ic_path_cb = ic_path.clone();
                        let watcher2 = watcher.clone();
                        let zk2 = zk.clone();
                        // Avoid duplicate watches per idc_cluster.
                        {
                            let mut keys = watcher2.watch_keys.lock();
                            if !keys.insert(format!("idc:{ic_path}")) {
                                continue;
                            }
                        }
                        let _ = zk.children_watch(
                            &ic_path,
                            Arc::new(move |tasks| {
                                watcher2.register_task_watches(&ic_path_cb, tasks, &zk2);
                                true
                            }),
                        );
                    }
                    true
                }),
            )?;
        } else {
            let base = self.path_prefix.clone();
            let zk = self.zk.clone();
            let watcher = self.clone_for_watch();
            self.zk.children_watch(
                &self.path_prefix,
                Arc::new(move |tasks| {
                    watcher.register_task_watches(&base, tasks, &zk);
                    true
                }),
            )?;
        }

        // periodic poll loop (Python: 60s)
        let has_stop = self.has_stop.clone();
        let should_poll = self.should_poll.clone();
        let zk = self.zk.clone();
        let replicas = self.replicas.clone();
        let conf = self.conf.clone();
        let path_prefix = self.path_prefix.clone();
        let family = self.zk_watch_address_family;
        let use_archon = self.use_archon;
        let poll = thread::spawn(move || {
            while !has_stop.load(Ordering::Relaxed) {
                // Python sleeps for 60s per loop. Split it so `stop()` doesn't
                // block for up to a minute on join.
                for _ in 0..60 {
                    if has_stop.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(Duration::from_secs(1));
                }
                if !should_poll.load(Ordering::Relaxed) {
                    continue;
                }
                let _ = poll_once(&zk, &conf, &path_prefix, &replicas, use_archon, family);
            }
        });
        *self.poll_thread.lock() = Some(poll);
        Ok(())
    }

    /// Enable/disable the periodic reconciliation poll loop.
    ///
    /// Python parity: set by `ZKListener` on LOST/CONNECTED transitions.
    pub fn set_should_poll(&self, v: bool) {
        self.should_poll.store(v, Ordering::Relaxed);
    }

    fn clone_for_watch(&self) -> ReplicaWatcherForWatch {
        ReplicaWatcherForWatch {
            replicas: self.replicas.clone(),
            watch_keys: self.watch_keys.clone(),
        }
    }

    /// Stop watches/polling and clear cached replica state.
    pub fn stop(&self) {
        self.has_stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.poll_thread.lock().take() {
            let _ = h.join();
        }
        self.replicas.write().clear();
    }

    /// Return all available replicas grouped by task key.
    pub fn get_all_replicas(
        &self,
        server_type: ServerType,
        idc: Option<&str>,
        cluster: Option<&str>,
    ) -> HashMap<String, Vec<String>> {
        let st = match server_type {
            ServerType::Ps => "ps",
            ServerType::Entry => "entry",
            ServerType::Dense => "dense",
        };

        let mut out: HashMap<String, Vec<String>> = HashMap::new();
        let map = self.replicas.read();
        for (path, replicas) in map.iter() {
            let (location, server_type2, task) = parse_task_path(path);
            if server_type2.as_deref() != Some(st) {
                continue;
            }
            if !should_ship_in(&location, idc, cluster) {
                continue;
            }
            let task_key = task.map(|t| t.to_string()).unwrap_or_default();
            let key = if self.conf.dc_aware {
                let loc = location.clone().unwrap_or_default();
                format!("{}/{}:{}", loc, st, task_key)
            } else {
                format!("{}:{}", st, task_key)
            };

            for rm in replicas.values() {
                if rm.stat == ModelState::Available as i32 {
                    if let Some(addr) =
                        rm.get_address(self.use_archon, self.zk_watch_address_family)
                    {
                        out.entry(key.clone()).or_default().push(addr);
                    }
                }
            }
        }
        if (server_type == ServerType::Ps && out.is_empty())
            || (server_type == ServerType::Dense && self.conf.dense_alone && out.is_empty())
        {
            error!("empty replicas {}-{}", self.path_prefix, st);
        }
        out
    }

    /// Return available replica addresses for a specific task.
    pub fn get_replicas(
        &self,
        server_type: ServerType,
        task: i32,
        idc: Option<&str>,
        cluster: Option<&str>,
    ) -> Vec<String> {
        let st = match server_type {
            ServerType::Ps => "ps",
            ServerType::Entry => "entry",
            ServerType::Dense => "dense",
        };
        let map = self.replicas.read();
        let mut out = Vec::new();
        for (path, replicas) in map.iter() {
            let (location, server_type2, task2) = parse_task_path(path);
            if server_type2.as_deref() != Some(st) {
                continue;
            }
            if task2.unwrap_or(-1) != task {
                continue;
            }
            if !should_ship_in(&location, idc, cluster) {
                continue;
            }
            for rm in replicas.values() {
                if rm.stat == ModelState::Available as i32 {
                    if let Some(addr) =
                        rm.get_address(self.use_archon, self.zk_watch_address_family)
                    {
                        out.push(addr);
                    }
                }
            }
        }
        out
    }

    /// Return addresses for a specific replica (when available).
    pub fn get_replica(
        &self,
        server_type: ServerType,
        task: i32,
        replica: i32,
        idc: Option<&str>,
        cluster: Option<&str>,
    ) -> Option<Vec<String>> {
        let st = match server_type {
            ServerType::Ps => "ps",
            ServerType::Entry => "entry",
            ServerType::Dense => "dense",
        };
        let map = self.replicas.read();
        let mut out = Vec::new();
        for (path, replicas) in map.iter() {
            let (location, server_type2, task2) = parse_task_path(path);
            if server_type2.as_deref() != Some(st) || task2.unwrap_or(-1) != task {
                continue;
            }
            if !should_ship_in(&location, idc, cluster) {
                continue;
            }
            if let Some(rm) = replicas.get(&replica.to_string()) {
                if rm.stat == ModelState::Available as i32 {
                    if let Some(addr) =
                        rm.get_address(self.use_archon, self.zk_watch_address_family)
                    {
                        out.push(addr);
                    }
                }
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Return a mapping of `address -> TargetExtraInfo` for parameter sync.
    ///
    /// Python parity: `ReplicaWatcher.get_replicas_with_extra_info()`.
    pub fn get_replicas_with_extra_info(
        &self,
        server_type: ServerType,
        task: i32,
        idc: Option<&str>,
        cluster: Option<&str>,
    ) -> HashMap<String, TargetExtraInfo> {
        let st = match server_type {
            ServerType::Ps => "ps",
            ServerType::Entry => "entry",
            ServerType::Dense => "dense",
        };
        let map = self.replicas.read();
        let mut out: HashMap<String, TargetExtraInfo> = HashMap::new();
        for (path, replicas) in map.iter() {
            let (location, server_type2, task2) = parse_task_path(path);
            if server_type2.as_deref() != Some(st) || task2.unwrap_or(-1) != task {
                continue;
            }
            if !should_ship_in(&location, idc, cluster) {
                continue;
            }

            let (idc_val, cluster_val) = location
                .as_deref()
                .and_then(|loc| loc.split_once(':'))
                .map(|(i, c)| (Some(i.to_string()), Some(c.to_string())))
                .unwrap_or((None, None));

            for (replica_id, rm) in replicas.iter() {
                if rm.stat != ModelState::Available as i32 {
                    continue;
                }
                let Some(addr) = rm.get_address(self.use_archon, self.zk_watch_address_family)
                else {
                    continue;
                };
                let rid = replica_id.parse::<i64>().unwrap_or(-1);
                out.insert(
                    addr,
                    TargetExtraInfo {
                        idc: idc_val.clone(),
                        cluster: cluster_val.clone(),
                        replica_id: Some(rid),
                    },
                );
            }
        }
        out
    }
}

#[derive(Clone)]
struct ReplicaWatcherForWatch {
    replicas: Arc<RwLock<HashMap<String, HashMap<String, ReplicaMeta>>>>,
    watch_keys: Arc<Mutex<HashSet<String>>>,
}

impl ReplicaWatcherForWatch {
    fn register_task_watches(&self, base: &str, tasks: Vec<String>, zk: &Arc<dyn ZkClient>) {
        for task in tasks {
            let task_path = Path::new(base).join(&task).to_string_lossy().to_string();
            // Avoid duplicate watches.
            {
                let mut keys = self.watch_keys.lock();
                if !keys.insert(format!("task:{task_path}")) {
                    continue;
                }
            }
            let replicas_map = self.replicas.clone();
            let task_path_for_cb = task_path.clone();
            let watch_keys = self.watch_keys.clone();
            let zk2 = zk.clone();
            let _ = zk.children_watch(
                &task_path,
                Arc::new(move |replicas| {
                    for replica in replicas {
                        let replica_path = Path::new(&task_path_for_cb)
                            .join(&replica)
                            .to_string_lossy()
                            .to_string();
                        {
                            let mut keys = watch_keys.lock();
                            if !keys.insert(format!("replica:{replica_path}")) {
                                continue;
                            }
                        }
                        let replicas_map2 = replicas_map.clone();
                        let task_path2 = task_path_for_cb.clone();
                        let replica_id = replica.parse::<i32>().ok().unwrap_or(0);
                        let cb: DataWatchCallback = Arc::new(move |data, _stat, event| {
                            on_data_watch(&replicas_map2, &task_path2, replica_id, data, event);
                            true
                        });
                        let _ = zk2.data_watch(&replica_path, cb);
                    }
                    true
                }),
            );
        }
    }
}

fn on_data_watch(
    replicas_map: &Arc<RwLock<HashMap<String, HashMap<String, ReplicaMeta>>>>,
    task_path: &str,
    replica_id: i32,
    data: Option<Vec<u8>>,
    event: Option<crate::backends::WatchedEvent>,
) {
    let key = replica_id.to_string();
    let mut map = replicas_map.write();
    let ev = event.as_ref().map(|e| &e.event_type);

    // Python parity: if `data` is missing, only mutate existing entries by setting UNKNOWN.
    if data.as_deref().map(|b| b.is_empty()).unwrap_or(true) {
        match ev {
            Some(WatchEventType::Deleted) => {
                if let Some(r) = map.get_mut(task_path) {
                    r.remove(&key);
                    if r.is_empty() {
                        map.remove(task_path);
                    }
                }
            }
            _ => {
                if let Some(r) = map.get_mut(task_path) {
                    if let Some(meta) = r.get_mut(&key) {
                        meta.stat = ModelState::Unknown as i32;
                    }
                }
            }
        }
        return;
    }

    match ev {
        None | Some(WatchEventType::Created) | Some(WatchEventType::Changed) => {
            let meta = data
                .as_deref()
                .and_then(|b| ReplicaMeta::deserialize(b).ok())
                .unwrap_or_default();
            map.entry(task_path.to_string())
                .or_default()
                .insert(key, meta);
        }
        Some(WatchEventType::Deleted) => {
            if let Some(r) = map.get_mut(task_path) {
                r.remove(&key);
                if r.is_empty() {
                    map.remove(task_path);
                }
            }
        }
        _ => {}
    }
}

fn poll_once(
    zk: &Arc<dyn ZkClient>,
    conf: &WatcherConfig,
    path_prefix: &str,
    replicas: &Arc<RwLock<HashMap<String, HashMap<String, ReplicaMeta>>>>,
    _use_archon: bool,
    _family: &'static str,
) -> Result<(), ZkError> {
    // This is a best-effort reconciliation; it refreshes `replicas` with the latest data in ZK.
    let mut tasks = Vec::new();
    if conf.dc_aware {
        let idc_clusters = zk.get_children(path_prefix).unwrap_or_default();
        for ic in idc_clusters {
            let ic_path = Path::new(path_prefix)
                .join(&ic)
                .to_string_lossy()
                .to_string();
            let ts = zk.get_children(&ic_path).unwrap_or_default();
            for t in ts {
                tasks.push(format!("{ic}/{t}"));
            }
        }
    } else {
        tasks = zk.get_children(path_prefix).unwrap_or_default();
    }

    let mut tmp: HashMap<String, HashMap<String, ReplicaMeta>> = HashMap::new();
    for task in tasks {
        let task_path = Path::new(path_prefix)
            .join(&task)
            .to_string_lossy()
            .to_string();
        let reps = zk.get_children(&task_path).unwrap_or_default();
        let mut reps_map = HashMap::new();
        for r in reps {
            let replica_path = Path::new(&task_path).join(&r).to_string_lossy().to_string();
            if let Ok(bytes) = zk.get(&replica_path) {
                if let Ok(meta) = ReplicaMeta::deserialize(&bytes) {
                    let rid = r.parse::<i32>().ok().unwrap_or(0).to_string();
                    reps_map.insert(rid, meta);
                }
            }
        }
        tmp.insert(task_path, reps_map);
    }
    *replicas.write() = tmp;
    Ok(())
}

/// Register/update this replica's znode based on TFServing model status.
pub struct ReplicaUpdater {
    zk: Arc<dyn ZkClient>,
    conf: UpdaterConfig,
    path_prefix: String,
    model_monitor: TfsMonitor,

    meta: Arc<Mutex<HashMap<String, ReplicaMeta>>>,
    has_stop: Arc<AtomicBool>,
    should_update: Arc<AtomicBool>,
    should_reregister: Arc<AtomicBool>,
    thread: Mutex<Option<JoinHandle<()>>>,
    reregister_thread: Mutex<Option<JoinHandle<()>>>,
    replica_id: Arc<std::sync::atomic::AtomicI32>,
}

/// Configuration for [`ReplicaUpdater`].
#[derive(Clone)]
pub struct UpdaterConfig {
    /// Business ID used for ZK paths.
    pub bzid: String,
    /// Base model name under the service subtree.
    pub base_name: String,
    /// Local base path for model files.
    pub base_path: String,
    /// Number of PS tasks in the deployment.
    pub num_ps: i32,
    /// Total number of shards.
    pub num_shard: i32,
    /// Current shard ID.
    pub shard_id: i32,
    /// Local replica ID (or -1 to allocate).
    pub replica_id: i32,
    /// Deploy type used to determine which tasks to register.
    pub deploy_type: TfsDeployType,
    /// Whether dense runs as standalone in addition to entry/ps.
    pub dense_alone: bool,
    /// Entry server port.
    pub tfs_entry_port: u16,
    /// PS server port.
    pub tfs_ps_port: u16,
    /// Dense server port.
    pub tfs_dense_port: u16,
    /// Entry server archon port.
    pub tfs_entry_archon_port: u16,
    /// PS server archon port.
    pub tfs_ps_archon_port: u16,
    /// Dense server archon port.
    pub tfs_dense_archon_port: u16,
    /// Whether the service tree includes a `{idc}:{cluster}` segment.
    pub dc_aware: bool,
    /// IDC to use when dc-aware.
    pub idc: Option<String>,
    /// Cluster to use when dc-aware.
    pub cluster: Option<String>,
}

impl UpdaterConfig {
    /// Return the base service path for this config.
    pub fn path_prefix(&self) -> String {
        if self.dc_aware {
            let loc = match (&self.idc, &self.cluster) {
                (Some(i), Some(c)) => format!("{}:{}", i, c),
                _ => "unknown:unknown".to_string(),
            };
            format!("/{}/service/{}/{}", self.bzid, self.base_name, loc)
        } else {
            format!("/{}/service/{}", self.bzid, self.base_name)
        }
    }

    fn scheduled_ps_tasks(&self) -> Vec<i32> {
        let mut out = Vec::new();
        if matches!(self.deploy_type, TfsDeployType::Mixed | TfsDeployType::Ps) {
            for i in 0..self.num_ps {
                if i % self.num_shard == self.shard_id {
                    out.push(i);
                }
            }
        }
        out
    }

    fn model_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if matches!(self.deploy_type, TfsDeployType::Mixed | TfsDeployType::Ps) {
            for task_id in self.scheduled_ps_tasks() {
                names.push(format!("{}_{}", TfServerType::PS, task_id));
            }
        }
        if matches!(
            self.deploy_type,
            TfsDeployType::Mixed | TfsDeployType::Entry
        ) {
            names.push(TfServerType::ENTRY.to_string());
        }
        if self.dense_alone
            && matches!(
                self.deploy_type,
                TfsDeployType::Mixed | TfsDeployType::Dense
            )
        {
            names.push(format!("{}_0", TfServerType::DENSE));
        }
        names
    }
}

impl ReplicaUpdater {
    /// Create a new updater instance.
    pub fn new(zk: Arc<dyn ZkClient>, conf: UpdaterConfig) -> Self {
        let path_prefix = conf.path_prefix();
        let replica_id = conf.replica_id;
        let monitor_conf = TfsAgentConfig {
            deploy_type: conf.deploy_type,
            dense_alone: conf.dense_alone,
            tfs_entry_port: conf.tfs_entry_port,
            tfs_ps_port: conf.tfs_ps_port,
            tfs_dense_port: conf.tfs_dense_port,
        };
        Self {
            zk,
            path_prefix,
            model_monitor: TfsMonitor::new(monitor_conf),
            conf,
            meta: Arc::new(Mutex::new(HashMap::new())),
            has_stop: Arc::new(AtomicBool::new(false)),
            should_update: Arc::new(AtomicBool::new(true)),
            should_reregister: Arc::new(AtomicBool::new(false)),
            thread: Mutex::new(None),
            reregister_thread: Mutex::new(None),
            replica_id: Arc::new(std::sync::atomic::AtomicI32::new(replica_id)),
        }
    }

    fn current_replica_id(&self) -> i32 {
        self.replica_id.load(Ordering::Relaxed)
    }

    fn entry_path(&self) -> String {
        let rid = self.current_replica_id();
        if rid == -1 {
            // Python parity: use `/entry:0/0` as a base when replica_id is unknown.
            return format!("{}/{:}:0/0", self.path_prefix, TfServerType::ENTRY);
        }
        format!(
            "{}/{:}:0/{:011}",
            self.path_prefix,
            TfServerType::ENTRY,
            rid
        )
    }

    fn ps_path(&self, task_id: i32) -> String {
        let rid = self.current_replica_id();
        format!(
            "{}/{}:{}/{}",
            self.path_prefix,
            TfServerType::PS,
            task_id,
            rid
        )
    }

    fn dense_path(&self) -> String {
        let rid = self.current_replica_id();
        format!("{}/{}:0/{}", self.path_prefix, TfServerType::DENSE, rid)
    }

    fn do_register(&self, replica_path: &str, grpc_port: u16, archon_port: u16) {
        // Python uses env host discovery, falling back to local hostname resolution.
        // For test stability, prefer loopback if envs are unset.
        let host = std::env::var("MY_HOST_IP").unwrap_or_else(|_| "127.0.0.1".to_string());
        let host_ipv6 = std::env::var("MY_HOST_IPV6").unwrap_or_else(|_| "::1".to_string());
        let host_ipv6 = format!("[{}]", host_ipv6.trim_matches(['[', ']']));

        let rm = ReplicaMeta {
            address: Some(format!("{host}:{grpc_port}")),
            address_ipv6: Some(format!("{host_ipv6}:{grpc_port}")),
            stat: ModelState::Unknown as i32,
            archon_address: Some(format!("{host}:{archon_port}")),
            archon_address_ipv6: Some(format!("{host_ipv6}:{archon_port}")),
            ..Default::default()
        };

        self.meta
            .lock()
            .insert(replica_path.to_string(), rm.clone());
        let value = rm.serialize();
        match self.zk.create(replica_path, value.clone(), true, true) {
            Ok(_) => {}
            Err(ZkError::NodeExists(_)) => {
                let _ = self.zk.set(replica_path, value);
            }
            Err(_) => {}
        }
    }

    fn allocate_replica_id_for_entry(&self) -> i32 {
        // Best-effort emulation of Kazoo `sequence=True` behavior when replica_id is unknown.
        // We pick `max(child)+1` under `{path_prefix}/entry:0`.
        let entry_dir = format!("{}/{:}:0", self.path_prefix, TfServerType::ENTRY);
        let mut max_id = 0i32;
        if let Ok(children) = self.zk.get_children(&entry_dir) {
            for c in children {
                if let Ok(v) = c.parse::<i32>() {
                    max_id = max_id.max(v);
                }
            }
        }
        max_id.saturating_add(1)
    }

    /// Create/refresh this replica's ephemeral znodes.
    pub fn register(&self) {
        // If the replica_id is unknown, allocate it before registering.
        if self.current_replica_id() == -1
            && matches!(
                self.conf.deploy_type,
                TfsDeployType::Mixed | TfsDeployType::Entry
            )
        {
            let rid = self.allocate_replica_id_for_entry();
            self.replica_id.store(rid, Ordering::Relaxed);
        }

        if matches!(
            self.conf.deploy_type,
            TfsDeployType::Mixed | TfsDeployType::Entry
        ) {
            self.do_register(
                &self.entry_path(),
                self.conf.tfs_entry_port,
                self.conf.tfs_entry_archon_port,
            );
        }
        if matches!(
            self.conf.deploy_type,
            TfsDeployType::Mixed | TfsDeployType::Ps
        ) {
            for task_id in self.conf.scheduled_ps_tasks() {
                self.do_register(
                    &self.ps_path(task_id),
                    self.conf.tfs_ps_port,
                    self.conf.tfs_ps_archon_port,
                );
            }
        }
        if self.conf.dense_alone
            && matches!(
                self.conf.deploy_type,
                TfsDeployType::Mixed | TfsDeployType::Dense
            )
        {
            self.do_register(
                &self.dense_path(),
                self.conf.tfs_dense_port,
                self.conf.tfs_dense_archon_port,
            );
        }
    }

    /// Start background threads for status updates and periodic re-registration.
    pub fn start(&self) {
        self.has_stop.store(false, Ordering::Relaxed);
        if self.thread.lock().is_none() {
            let updater = self.clone_for_thread();
            *self.thread.lock() = Some(thread::spawn(move || updater.updater_loop()));
        }
        if self.reregister_thread.lock().is_none() {
            let updater = self.clone_for_thread();
            *self.reregister_thread.lock() = Some(thread::spawn(move || updater.reregister_loop()));
        }
    }

    fn clone_for_thread(&self) -> ReplicaUpdaterForThread {
        ReplicaUpdaterForThread {
            zk: self.zk.clone(),
            path_prefix: self.path_prefix.clone(),
            conf: self.conf.clone(),
            meta: self.meta.clone(),
            has_stop: self.has_stop.clone(),
            should_update: self.should_update.clone(),
            should_reregister: self.should_reregister.clone(),
            model_monitor: self.model_monitor.clone(),
            replica_id: self.replica_id.clone(),
        }
    }

    /// Enable/disable updating znode state based on TFServing.
    ///
    /// Python parity: set by `ZKListener` on LOST/CONNECTED transitions.
    pub fn set_should_update(&self, v: bool) {
        self.should_update.store(v, Ordering::Relaxed);
    }

    /// Trigger periodic re-registration of ephemeral znodes.
    ///
    /// Python parity: `ReplicaUpdater._should_reregister`.
    pub fn trigger_reregister(&self) {
        self.should_reregister.store(true, Ordering::Relaxed);
    }

    /// Stop background threads and clear cached meta state.
    pub fn stop(&self) {
        self.has_stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.thread.lock().take() {
            let _ = h.join();
        }
        if let Some(h) = self.reregister_thread.lock().take() {
            let _ = h.join();
        }
        self.meta.lock().clear();
    }
}

#[derive(Clone)]
struct ReplicaUpdaterForThread {
    zk: Arc<dyn ZkClient>,
    path_prefix: String,
    conf: UpdaterConfig,
    meta: Arc<Mutex<HashMap<String, ReplicaMeta>>>,
    has_stop: Arc<AtomicBool>,
    should_update: Arc<AtomicBool>,
    should_reregister: Arc<AtomicBool>,
    model_monitor: TfsMonitor,
    replica_id: Arc<std::sync::atomic::AtomicI32>,
}

impl ReplicaUpdaterForThread {
    fn updater_loop(&self) {
        // Use a dedicated runtime in this background thread and keep it alive for the
        // lifetime of the gRPC channels created by `TfsMonitor::connect()`.
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        let _ = rt.block_on(self.model_monitor.connect());

        let mut ensure_ticks = 0u32;
        while !self.has_stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(1));
            if !self.should_update.load(Ordering::Relaxed) {
                continue;
            }
            for name in self.conf.model_names() {
                if let Err(e) = self.do_update(&rt, &name) {
                    error!("replica updater update failed for {name}: {e}");
                }
            }

            // Python parity: watcher poll runs every 60s and re-registers missing znodes.
            // We perform a similar best-effort existence check periodically from the updater.
            ensure_ticks = ensure_ticks.saturating_add(1);
            if ensure_ticks >= 60 {
                self.ensure_registered();
                ensure_ticks = 0;
            }
        }

        let _ = rt.block_on(self.model_monitor.stop());
    }

    fn reregister_loop(&self) {
        // Python parity: `_reregister` checks every 10 seconds.
        while !self.has_stop.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_secs(10));
            if self.should_reregister.load(Ordering::Relaxed) {
                self.register_all();
                self.should_update.store(true, Ordering::Relaxed);
            }
        }
    }

    fn current_replica_id(&self) -> i32 {
        self.replica_id.load(Ordering::Relaxed)
    }

    fn register_all(&self) {
        // Minimal reimplementation of `ReplicaUpdater.register()` for the background thread,
        // reusing the meta data stored at construction time.
        //
        // We avoid duplicating all port logic here by simply re-creating the znodes based on
        // currently cached `self.meta` values.
        let metas = self.meta.lock().clone();
        for (path, meta) in metas {
            let value = meta.serialize();
            match self.zk.create(&path, value.clone(), true, true) {
                Ok(_) => {}
                Err(ZkError::NodeExists(_)) => {
                    let _ = self.zk.set(&path, value);
                }
                Err(_) => {}
            }
        }
    }

    fn ensure_registered(&self) {
        let metas = self.meta.lock().clone();
        for (path, meta) in metas {
            if !self.zk.exists(&path) {
                let value = meta.serialize();
                let _ = self.zk.create(&path, value, true, true);
            }
        }
    }

    fn do_update(&self, rt: &tokio::runtime::Runtime, name: &str) -> Result<(), String> {
        let replica_path = if name.starts_with(TfServerType::ENTRY) {
            let rid = self.current_replica_id();
            format!(
                "{}/{:}:0/{:011}",
                self.path_prefix,
                TfServerType::ENTRY,
                rid
            )
        } else if name.starts_with(TfServerType::PS) {
            let task: i32 = name
                .split('_')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            format!(
                "{}/{}:{}/{}",
                self.path_prefix,
                TfServerType::PS,
                task,
                self.current_replica_id()
            )
        } else {
            format!(
                "{}/{}:0/{}",
                self.path_prefix,
                TfServerType::DENSE,
                self.current_replica_id()
            )
        };

        let model_status = match rt.block_on(
            self.model_monitor
                .get_model_status_for_name(name, None, None),
        ) {
            Ok(v) => v,
            Err(_) => {
                // Python: on exception, set to UNKNOWN if it was not already.
                let mut meta = self.meta.lock();
                let rm = meta.entry(replica_path.clone()).or_default();
                if rm.stat != ModelState::Unknown as i32 {
                    rm.stat = ModelState::Unknown as i32;
                    let value = rm.serialize();
                    match self.zk.set(&replica_path, value.clone()) {
                        Ok(_) => {}
                        Err(ZkError::NoNode(_)) => {
                            let _ = self.zk.create(&replica_path, value, true, true);
                        }
                        Err(_) => {}
                    }
                }
                return Ok(());
            }
        };

        if model_status.is_empty() {
            return Ok(());
        }

        let mut statuses = model_status.clone();
        statuses.sort_by_key(|mvs| mvs.version);
        let mut selected = statuses.last().cloned().unwrap();
        if statuses.len() > 1 {
            // Pick newest AVAILABLE if present, otherwise latest.
            for s in statuses.iter().rev() {
                if s.state == ModelState::Available as i32 {
                    selected = s.clone();
                    break;
                }
            }
        }

        let status = selected.status.clone().unwrap_or_default();
        if status.error_code != monolith_proto::tensorflow_serving::error::Code::Ok as i32 {
            return Err(status.error_message);
        }

        // Update znode if state changed.
        let mut meta = self.meta.lock();
        let rm = meta.entry(replica_path.clone()).or_default();
        if rm.stat != selected.state {
            rm.stat = selected.state;
            let value = rm.serialize();
            match self.zk.set(&replica_path, value.clone()) {
                Ok(_) => {}
                Err(ZkError::NoNode(_)) => {
                    let _ = self.zk.create(&replica_path, value, true, true);
                }
                Err(_) => {}
            }
        }
        Ok(())
    }
}

/// Python parity for `ZKListener` in `replica_manager.py`.
///
/// This isn't wired to an actual Kazoo listener because `ZkClient` is an abstract trait
/// and the crate's production integration may provide connection-state events elsewhere.
/// Call `on_state_change()` when the caller observes ZK connection transitions.
pub struct ZkListener {
    watcher_should_poll: Arc<AtomicBool>,
    updater_should_update: Arc<AtomicBool>,
    updater_should_reregister: Arc<AtomicBool>,
    has_lost: AtomicBool,
}

impl ZkListener {
    /// Create a listener wiring watcher/updater flags.
    pub fn new(watcher: &ReplicaWatcher, updater: &ReplicaUpdater) -> Self {
        Self {
            watcher_should_poll: watcher.should_poll.clone(),
            updater_should_update: updater.should_update.clone(),
            updater_should_reregister: updater.should_reregister.clone(),
            has_lost: AtomicBool::new(false),
        }
    }

    /// Return value matches Kazoo listener expectations (always false for "keep").
    pub fn on_state_change(&self, state: ZkConnectionState) -> bool {
        match state {
            ZkConnectionState::Lost => {
                self.has_lost.store(true, Ordering::Relaxed);
                self.watcher_should_poll.store(false, Ordering::Relaxed);
                self.updater_should_update.store(false, Ordering::Relaxed);
            }
            ZkConnectionState::Suspended => {
                return false;
            }
            ZkConnectionState::Connected => {
                if self.has_lost.load(Ordering::Relaxed) {
                    self.updater_should_reregister
                        .store(true, Ordering::Relaxed);
                    thread::sleep(Duration::from_secs(5));
                    self.watcher_should_poll.store(true, Ordering::Relaxed);
                    self.has_lost.store(false, Ordering::Relaxed);
                }
            }
        }
        false
    }
}

/// Convenience wrapper tying watcher + updater together.
pub struct ReplicaManager {
    /// Replica watcher component (discovery).
    pub watcher: ReplicaWatcher,
    /// Replica updater component (registration + status updates).
    pub updater: ReplicaUpdater,
    /// Listener mapping ZK connection events to watcher/updater behavior.
    pub zk_listener: ZkListener,
}

impl ReplicaManager {
    /// Create a manager combining watcher + updater.
    pub fn new(
        zk: Arc<dyn ZkClient>,
        watcher_conf: WatcherConfig,
        updater_conf: UpdaterConfig,
    ) -> Self {
        let watcher = ReplicaWatcher::new(
            zk.clone(),
            watcher_conf,
            DEFAULT_USE_ARCHON,
            AddressFamily::IPV4,
        );
        let updater = ReplicaUpdater::new(zk, updater_conf);
        let zk_listener = ZkListener::new(&watcher, &updater);
        Self {
            watcher,
            updater,
            zk_listener,
        }
    }

    /// Start watcher + updater.
    pub fn start(&self) -> Result<(), ZkError> {
        self.updater.register();
        self.watcher.watch_data()?;
        self.updater.start();
        Ok(())
    }

    /// Stop watcher + updater.
    pub fn stop(&self) {
        self.updater.stop();
        self.watcher.stop();
    }

    /// Return true when all PS tasks in `[0, num_ps)` have at least one available replica.
    pub fn is_ps_set_started(&self, num_ps: i32, idc: Option<&str>, cluster: Option<&str>) -> bool {
        for i in 0..num_ps {
            let replicas = self.watcher.get_replicas(ServerType::Ps, i, idc, cluster);
            if replicas.is_empty() {
                return false;
            }
        }
        true
    }

    /// Return true when dense task 0 has at least one available replica.
    pub fn is_dense_set_started(&self) -> bool {
        !self
            .watcher
            .get_replicas(ServerType::Dense, 0, None, None)
            .is_empty()
    }
}
