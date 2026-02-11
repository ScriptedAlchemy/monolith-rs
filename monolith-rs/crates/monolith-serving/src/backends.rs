//! Agent-service backends (Python parity for `monolith/agent_service/backends.py`).
//!
//! The Python agent-service stores service discovery and layout state in ZooKeeper using
//! JSON-serialized dataclasses. This module ports the relevant pieces for Rust-side parity,
//! including a minimal in-memory fake ZooKeeper client used by tests.

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// ZooKeeper-like error type (subset used by the Python backend).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ZkError {
    /// Path does not exist.
    NoNode(String),
    /// Node already exists.
    NodeExists(String),
    /// Non-empty node deletion without recursive flag.
    NotEmpty(String),
    /// Generic error.
    Other(String),
}

impl fmt::Display for ZkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZkError::NoNode(p) => write!(f, "NoNode: {p}"),
            ZkError::NodeExists(p) => write!(f, "NodeExists: {p}"),
            ZkError::NotEmpty(p) => write!(f, "NotEmpty: {p}"),
            ZkError::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for ZkError {}

/// A minimal znode stat (Python parity for `kazoo.protocol.states.ZnodeStat`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ZnodeStat {
    /// Creation time (unix seconds).
    pub ctime: i64,
    /// Modification time (unix seconds).
    pub mtime: i64,
    /// Version counter for the node data.
    pub version: i32,
    /// Data length in bytes.
    pub data_length: usize,
    /// Number of direct children.
    pub num_children: usize,
}

/// Watch event type (Python parity for `kazoo.protocol.states.EventType` subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchEventType {
    /// Node was created.
    Created,
    /// Node was deleted.
    Deleted,
    /// Node data was changed.
    Changed,
    /// Child list changed.
    Child,
}

/// Watched event (Python parity for `kazoo.protocol.states.WatchedEvent` subset).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WatchedEvent {
    /// Event type.
    pub event_type: WatchEventType,
    /// Path that triggered the event.
    pub path: String,
}

/// Handle returned by `children_watch`.
#[derive(Debug, Clone)]
pub struct ChildrenWatchHandle {
    stopped: Arc<AtomicBool>,
}

impl ChildrenWatchHandle {
    /// Mark the watch as stopped.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }
}

/// Callback signature for children watches.
pub type ChildrenWatchCallback = Arc<dyn Fn(Vec<String>) -> bool + Send + Sync + 'static>;

/// Callback signature for children watches with event details.
pub type ChildrenWatchEventCallback =
    Arc<dyn Fn(Vec<String>, Option<WatchedEvent>) -> bool + Send + Sync + 'static>;

/// Handle returned by `data_watch`.
#[derive(Debug, Clone)]
pub struct DataWatchHandle {
    stopped: Arc<AtomicBool>,
}

impl DataWatchHandle {
    /// Mark the watch as stopped.
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::Relaxed);
    }

    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }
}

/// Callback signature for data watches.
pub type DataWatchCallback =
    Arc<dyn Fn(Option<Vec<u8>>, ZnodeStat, Option<WatchedEvent>) -> bool + Send + Sync + 'static>;

/// Minimal ZooKeeper client API needed for `ZkBackend` parity.
pub trait ZkClient: Send + Sync {
    /// Start the client.
    fn start(&self) -> Result<(), ZkError>;
    /// Stop the client.
    fn stop(&self) -> Result<(), ZkError>;
    /// Restart the client (Python parity: `MonolithKazooClient.restart()`).
    fn restart(&self) -> Result<(), ZkError> {
        self.stop()?;
        self.start()?;
        Ok(())
    }

    /// Ensure that a path exists (creating parents if needed).
    fn ensure_path(&self, path: &str) -> Result<(), ZkError>;

    /// Create a znode.
    fn create(
        &self,
        path: &str,
        value: Vec<u8>,
        ephemeral: bool,
        makepath: bool,
    ) -> Result<(), ZkError>;
    /// Delete a znode.
    fn delete(&self, path: &str, recursive: bool) -> Result<(), ZkError>;
    /// Set a znode's value.
    fn set(&self, path: &str, value: Vec<u8>) -> Result<(), ZkError>;
    /// Get a znode's value.
    fn get(&self, path: &str) -> Result<Vec<u8>, ZkError>;
    /// Check if a path exists.
    fn exists(&self, path: &str) -> bool;
    /// List children of a path.
    fn get_children(&self, path: &str) -> Result<Vec<String>, ZkError>;

    /// Register a watch on children changes for a path.
    fn children_watch(
        &self,
        path: &str,
        callback: ChildrenWatchCallback,
    ) -> Result<ChildrenWatchHandle, ZkError>;

    /// Register a watch on children changes for a path, including event details.
    ///
    /// Default impl degrades to `children_watch` and passes `None` for events.
    fn children_watch_event(
        &self,
        path: &str,
        callback: ChildrenWatchEventCallback,
    ) -> Result<ChildrenWatchHandle, ZkError> {
        let cb: ChildrenWatchCallback = Arc::new(move |children| callback(children, None));
        self.children_watch(path, cb)
    }

    /// Register a watch on data changes for a path.
    ///
    /// Implementations that don't support data watches can leave the default behavior.
    fn data_watch(
        &self,
        _path: &str,
        _callback: DataWatchCallback,
    ) -> Result<DataWatchHandle, ZkError> {
        Err(ZkError::Other("data_watch not implemented".to_string()))
    }
}

/// A saved-model identifier: `{model_name}:{sub_graph}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SavedModel {
    /// Model name.
    pub model_name: String,
    /// Sub-graph name.
    pub sub_graph: String,
}

impl SavedModel {
    /// Construct a new saved model identifier.
    pub fn new(model_name: impl Into<String>, sub_graph: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            sub_graph: sub_graph.into(),
        }
    }
}

impl fmt::Display for SavedModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.model_name, self.sub_graph)
    }
}

/// Python parity for `SavedModelDeployConfig` (JSON bytes stored in ZK).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SavedModelDeployConfig {
    /// Model base path.
    #[serde(default)]
    pub model_base_path: Option<String>,
    /// Version policy string.
    #[serde(default)]
    pub version_policy: Option<String>,
}

impl SavedModelDeployConfig {
    /// Serialize to UTF-8 JSON bytes (Python parity: `dataclasses_json.to_json()`).
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("SavedModelDeployConfig must be JSON-serializable")
    }

    /// Deserialize from UTF-8 JSON bytes.
    pub fn deserialize(serialized: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(serialized)
    }
}

/// A serving container identifier: `{ctx_cluster}:{ctx_id}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Container {
    /// Cluster name.
    pub ctx_cluster: String,
    /// Container id.
    pub ctx_id: String,
}

impl Container {
    /// Construct a new container identifier.
    pub fn new(ctx_cluster: impl Into<String>, ctx_id: impl Into<String>) -> Self {
        Self {
            ctx_cluster: ctx_cluster.into(),
            ctx_id: ctx_id.into(),
        }
    }
}

impl fmt::Display for Container {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.ctx_cluster, self.ctx_id)
    }
}

/// Python parity for `ContainerServiceInfo` (JSON bytes stored in ZK).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContainerServiceInfo {
    /// gRPC address (`ip:port`).
    #[serde(default)]
    pub grpc: Option<String>,
    /// HTTP address (`ip:port`).
    #[serde(default)]
    pub http: Option<String>,
    /// Archon address (`ip:port`).
    #[serde(default)]
    pub archon: Option<String>,
    /// Agent address (`ip:port`).
    #[serde(default)]
    pub agent: Option<String>,
    /// IDC/DC name.
    #[serde(default)]
    pub idc: Option<String>,
    /// Debug string (opaque).
    #[serde(default)]
    pub debug_info: Option<String>,
}

impl ContainerServiceInfo {
    /// Serialize to UTF-8 JSON bytes (Python parity: `dataclasses_json.to_json()`).
    pub fn serialize(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("ContainerServiceInfo must be JSON-serializable")
    }

    /// Deserialize from UTF-8 JSON bytes.
    pub fn deserialize(serialized: &[u8]) -> Result<Self, serde_json::Error> {
        serde_json::from_slice(serialized)
    }
}

#[derive(Debug, Default)]
struct ZkBackendState {
    available_saved_model: HashSet<SavedModel>,
    /// `{model_name -> {sub_graph -> [service_info...]}}`
    service_info_map: HashMap<String, HashMap<String, Vec<ContainerServiceInfo>>>,
    children_watcher_map: HashMap<String, WatchEntry>,
    sync_model_name: Option<String>,
}

#[derive(Debug, Clone)]
struct WatchEntry {
    /// True while registration is in-flight (prevents re-entrant double registration).
    registering: bool,
    handle: Option<ChildrenWatchHandle>,
}

struct ZkBackendShared {
    lock: Mutex<ZkBackendState>,
    bzid: String,
    zk: Arc<dyn ZkClient>,
    is_lost: AtomicBool,
}

impl ZkBackendShared {
    fn children_watch(
        self: &Arc<Self>,
        path: &str,
        callback: ChildrenWatchCallback,
    ) -> Result<(), ZkError> {
        {
            let mut guard = self.lock.lock();
            if let Some(entry) = guard.children_watcher_map.get(path) {
                if entry.registering {
                    return Ok(());
                }
                if let Some(handle) = &entry.handle {
                    if !handle.is_stopped() {
                        return Ok(());
                    }
                }
            }
            guard.children_watcher_map.insert(
                path.to_string(),
                WatchEntry {
                    registering: true,
                    handle: None,
                },
            );
        }

        self.zk.ensure_path(path)?;
        let handle = self.zk.children_watch(path, callback)?;

        let mut guard = self.lock.lock();
        guard.children_watcher_map.insert(
            path.to_string(),
            WatchEntry {
                registering: false,
                handle: Some(handle),
            },
        );
        Ok(())
    }

    fn get_znode(&self, path: &str) -> Result<Option<Vec<u8>>, ZkError> {
        match self.zk.get(path) {
            Ok(v) => Ok(Some(v)),
            Err(ZkError::NoNode(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    fn create_znode(
        &self,
        path: &str,
        value: Vec<u8>,
        ephemeral: bool,
        makepath: bool,
    ) -> Result<(), ZkError> {
        // Python parity (`ZKBackend.create_znode`): swallow most ZK errors after logging.
        match self.zk.create(path, value.clone(), ephemeral, makepath) {
            Ok(()) => Ok(()),
            Err(ZkError::NodeExists(_)) => {
                if let Err(e) = self.zk.set(path, value) {
                    tracing::error!("exception in create_znode (set): {e}");
                }
                Ok(())
            }
            Err(e) => {
                tracing::error!("exception in create_znode: {e}");
                Ok(())
            }
        }
    }

    fn delete_znode(&self, path: &str) -> Result<(), ZkError> {
        // Python parity (`ZKBackend.delete_znode`): swallow errors after logging.
        if let Err(e) = self.zk.delete(path, true) {
            tracing::error!("exception in delete_znode: {e}");
        }
        Ok(())
    }

    fn get_service_info(
        &self,
        container: &Container,
    ) -> Result<Option<ContainerServiceInfo>, ZkError> {
        let path = format!("/{}/container_service/{}", self.bzid, container);
        match self.get_znode(&path)? {
            None => Ok(None),
            Some(data) => Ok(Some(
                ContainerServiceInfo::deserialize(&data)
                    .map_err(|e| ZkError::Other(e.to_string()))?,
            )),
        }
    }

    fn bind_callback(self: &Arc<Self>, model_name: &str, children: Vec<String>) {
        {
            let guard = self.lock.lock();
            if !guard.service_info_map.contains_key(model_name) {
                return;
            }
        }

        let mut new_binding: HashMap<String, Vec<ContainerServiceInfo>> = HashMap::new();
        for child in children {
            let parts: Vec<&str> = child.split(':').collect();
            if parts.len() < 3 {
                continue;
            }
            let sub_graph = parts[0].to_string();
            let container = Container::new(parts[1], parts[2]);
            let service_info = match self.get_service_info(&container) {
                Ok(Some(v)) => v,
                _ => continue,
            };
            new_binding.entry(sub_graph).or_default().push(service_info);
        }

        let mut guard = self.lock.lock();
        if !guard.service_info_map.contains_key(model_name) {
            return;
        }
        guard
            .service_info_map
            .insert(model_name.to_string(), new_binding);
    }
}

/// Python parity for `monolith.agent_service.backends.ZKBackend`.
#[derive(Clone)]
pub struct ZkBackend {
    shared: Arc<ZkBackendShared>,
}

impl ZkBackend {
    /// Create a new backend.
    pub fn new(bzid: impl Into<String>, zk: Arc<dyn ZkClient>) -> Self {
        Self {
            shared: Arc::new(ZkBackendShared {
                lock: Mutex::new(ZkBackendState::default()),
                bzid: bzid.into(),
                zk,
                is_lost: AtomicBool::new(false),
            }),
        }
    }

    /// Start the underlying ZK client.
    pub fn start(&self) -> Result<(), ZkError> {
        self.shared.zk.start()
    }

    /// Stop the underlying ZK client.
    pub fn stop(&self) -> Result<(), ZkError> {
        self.shared.zk.stop()
    }

    /// Test hook: set the "lost" flag (Python parity: KazooState.LOST listener).
    pub fn set_lost_for_test(&self, lost: bool) {
        self.shared.is_lost.store(lost, Ordering::Relaxed);
    }

    /// Report locally available saved models serving in TensorFlow Serving.
    pub fn sync_available_saved_models(
        &self,
        container: &Container,
        saved_models: HashSet<SavedModel>,
    ) -> Result<(), ZkError> {
        let mut guard = self.shared.lock.lock();
        if self.shared.is_lost.load(Ordering::Relaxed) {
            guard.available_saved_model.clear();
            let _ = self.shared.zk.restart();
            return Ok(());
        }

        let add_saved_models: Vec<_> = saved_models
            .difference(&guard.available_saved_model)
            .cloned()
            .collect();
        let remove_saved_models: Vec<_> = guard
            .available_saved_model
            .difference(&saved_models)
            .cloned()
            .collect();
        drop(guard);

        for saved_model in add_saved_models {
            let bind_path = format!(
                "/{}/binding/{}/{}:{}",
                self.shared.bzid, saved_model.model_name, saved_model.sub_graph, container
            );
            self.shared
                .create_znode(&bind_path, Vec::new(), true, true)?;
        }
        for saved_model in remove_saved_models {
            let bind_path = format!(
                "/{}/binding/{}/{}:{}",
                self.shared.bzid, saved_model.model_name, saved_model.sub_graph, container
            );
            let _ = self.shared.delete_znode(&bind_path);
        }

        let mut guard = self.shared.lock.lock();
        guard.available_saved_model = saved_models;
        Ok(())
    }

    /// Register a callback invoked when a layout path updates.
    pub fn register_layout_callback(
        &self,
        layout_path: &str,
        callback: impl Fn(Vec<(SavedModel, SavedModelDeployConfig)>) -> bool + Send + Sync + 'static,
    ) -> Result<(), ZkError> {
        let callback = Arc::new(callback);
        let layout_path = layout_path.to_string();
        let shared = self.shared.clone();

        let callback_wrap: ChildrenWatchCallback = Arc::new(move |children: Vec<String>| {
            let mut model_names: HashSet<String> = HashSet::new();
            let mut saved_models_with_cfg: Vec<(SavedModel, SavedModelDeployConfig)> = Vec::new();
            for child in children {
                let mut it = child.split(':');
                let model_name = match it.next() {
                    Some(v) => v.to_string(),
                    None => continue,
                };
                let sub_graph = match it.next() {
                    Some(v) => v.to_string(),
                    None => continue,
                };
                let saved_model = SavedModel::new(&model_name, &sub_graph);
                let fetch_path =
                    format!("/{}/saved_models/{}/{}", shared.bzid, model_name, sub_graph);
                let data = match shared.get_znode(&fetch_path) {
                    Ok(Some(v)) => v,
                    _ => continue,
                };
                let cfg = match SavedModelDeployConfig::deserialize(&data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                saved_models_with_cfg.push((saved_model, cfg));
                model_names.insert(model_name);
            }

            {
                let mut guard = shared.lock.lock();
                guard.service_info_map = model_names
                    .iter()
                    .map(|name| {
                        (
                            name.clone(),
                            guard
                                .service_info_map
                                .get(name)
                                .cloned()
                                .unwrap_or_default(),
                        )
                    })
                    .collect();
            }

            for model_name in model_names {
                let binding_watch_path = format!("/{}/binding/{}", shared.bzid, model_name);
                let shared_for_cb = shared.clone();
                let model_name_for_cb = model_name.clone();
                let _ = shared.children_watch(
                    &binding_watch_path,
                    Arc::new(move |children| {
                        shared_for_cb.bind_callback(&model_name_for_cb, children);
                        true
                    }),
                );
            }

            callback(saved_models_with_cfg)
        });

        self.shared.zk.ensure_path(&layout_path)?;
        self.shared.children_watch(&layout_path, callback_wrap)?;
        Ok(())
    }

    /// Get service info map.
    pub fn get_service_map(&self) -> HashMap<String, HashMap<String, Vec<ContainerServiceInfo>>> {
        self.shared.lock.lock().service_info_map.clone()
    }

    /// Report service info for a container.
    pub fn report_service_info(
        &self,
        container: &Container,
        service_info: &ContainerServiceInfo,
    ) -> Result<(), ZkError> {
        let path = format!("/{}/container_service/{}", self.shared.bzid, container);
        self.shared
            .create_znode(&path, service_info.serialize(), true, true)
    }

    /// Get service info for a container.
    pub fn get_service_info(
        &self,
        container: &Container,
    ) -> Result<Option<ContainerServiceInfo>, ZkError> {
        self.shared.get_service_info(container)
    }

    /// List declared saved models for `model_name`.
    pub fn list_saved_models(&self, model_name: &str) -> Result<Vec<SavedModel>, ZkError> {
        let model_path = format!("/{}/saved_models/{}", self.shared.bzid, model_name);
        match self.shared.zk.get_children(&model_path) {
            Ok(sub_graphs) => Ok(sub_graphs
                .into_iter()
                .map(|sg| SavedModel::new(model_name, sg))
                .collect()),
            Err(ZkError::NoNode(_)) => Ok(Vec::new()),
            Err(e) => Err(e),
        }
    }

    /// Declare a saved model and its deploy config.
    pub fn decl_saved_model(
        &self,
        saved_model: &SavedModel,
        deploy_config: &SavedModelDeployConfig,
    ) -> Result<(), ZkError> {
        let path = format!(
            "/{}/saved_models/{}/{}",
            self.shared.bzid, saved_model.model_name, saved_model.sub_graph
        );
        self.shared
            .create_znode(&path, deploy_config.serialize(), false, true)
    }

    /// Add a saved model to a layout.
    pub fn add_to_layout(&self, layout: &str, saved_model: &SavedModel) -> Result<(), ZkError> {
        let path = format!("{}/{}", layout, saved_model);
        self.shared.zk.ensure_path(&path)
    }

    /// Remove a saved model from a layout.
    pub fn remove_from_layout(
        &self,
        layout: &str,
        saved_model: &SavedModel,
    ) -> Result<(), ZkError> {
        let path = format!("{}/{}", layout, saved_model);
        match self.shared.zk.delete(&path, true) {
            Ok(()) => Ok(()),
            Err(ZkError::NoNode(_)) => Ok(()),
            Err(e) => Err(e),
        }
    }

    /// Return a JSON map similar to Python `bzid_info()` for debugging.
    pub fn bzid_info(&self) -> Result<serde_json::Value, ZkError> {
        let mut model_info: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        let mut container_info: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        let mut layout_info: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();

        let saved_models_root = format!("/{}/saved_models", self.shared.bzid);
        if self.shared.zk.exists(&saved_models_root) {
            for model_name in self.shared.zk.get_children(&saved_models_root)? {
                let model_path = format!("{}/{}", saved_models_root, model_name);
                let sub_graphs = self.shared.zk.get_children(&model_path)?;
                let mut m = serde_json::Map::new();
                m.insert(
                    "sub_graphs_total".to_string(),
                    serde_json::Value::from(sub_graphs.len() as i64),
                );
                for sub_graph in sub_graphs {
                    let cfg_path = format!("{}/{}", model_path, sub_graph);
                    if let Ok(data) = self.shared.zk.get(&cfg_path) {
                        let cfg_str = String::from_utf8_lossy(&data).to_string();
                        let mut sg = serde_json::Map::new();
                        sg.insert(
                            "deploy_config".to_string(),
                            serde_json::Value::from(cfg_str),
                        );
                        m.insert(sub_graph, serde_json::Value::Object(sg));
                    }
                }
                model_info.insert(model_name, serde_json::Value::Object(m));
            }
        }

        let container_root = format!("/{}/container_service", self.shared.bzid);
        if self.shared.zk.exists(&container_root) {
            for container in self.shared.zk.get_children(&container_root)? {
                let parts: Vec<&str> = container.split(':').collect();
                if parts.len() < 2 {
                    continue;
                }
                let cluster = parts[0].to_string();
                let container_id = parts[1].to_string();
                let path = format!("{}/{}", container_root, container);
                let data = self.shared.zk.get(&path)?;
                let svc_str = String::from_utf8_lossy(&data).to_string();
                let entry = container_info
                    .entry(cluster)
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                let obj = entry.as_object_mut().expect("must be object");
                let mut ci = serde_json::Map::new();
                ci.insert("service_info".to_string(), serde_json::Value::from(svc_str));
                obj.insert(container_id, serde_json::Value::Object(ci));
            }
        }

        let layouts_root = format!("/{}/layouts", self.shared.bzid);
        if self.shared.zk.exists(&layouts_root) {
            for layout in self.shared.zk.get_children(&layouts_root)? {
                let path = format!("{}/{}", layouts_root, layout);
                let saved_models = match self.shared.zk.get_children(&path) {
                    Ok(v) => v,
                    Err(ZkError::NoNode(_)) => Vec::new(),
                    Err(e) => return Err(e),
                };
                let mut saved_models = saved_models;
                saved_models.sort();
                layout_info.insert(layout, serde_json::Value::from(saved_models));
            }
        }

        // Bindings: enrich both `model_info` and `container_info` with cross references.
        //
        // Python parity: `monolith/agent_service/backends.py::ZKBackend.bzid_info`.
        let bindings_root = format!("/{}/binding", self.shared.bzid);
        if self.shared.zk.exists(&bindings_root) {
            for model_name in self.shared.zk.get_children(&bindings_root)? {
                let model_entry = model_info
                    .entry(model_name.clone())
                    .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                let model_obj = model_entry
                    .as_object_mut()
                    .ok_or_else(|| ZkError::Other("model_info entry must be object".to_string()))?;

                let binding_path = format!("{}/{}", bindings_root, model_name);
                let bindings = match self.shared.zk.get_children(&binding_path) {
                    Ok(v) => v,
                    Err(ZkError::NoNode(_)) => Vec::new(),
                    Err(e) => return Err(e),
                };

                for binding in bindings {
                    let parts: Vec<&str> = binding.split(':').collect();
                    if parts.len() < 3 {
                        continue;
                    }
                    let sub_graph = parts[0];
                    let cluster = parts[1];
                    let container_id = parts[2];

                    // model_info[model_name][sub_graph]['bindings'] += ["{cluster}:{container_id}"]
                    let mut need_incr_subgraphs_available = false;
                    {
                        let sg_entry = model_obj
                            .entry(sub_graph.to_string())
                            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                        let sg_obj = sg_entry.as_object_mut().ok_or_else(|| {
                            ZkError::Other("model_info sub_graph entry must be object".to_string())
                        })?;

                        if !sg_obj.contains_key("bindings") {
                            sg_obj.insert(
                                "bindings".to_string(),
                                serde_json::Value::Array(Vec::new()),
                            );
                            need_incr_subgraphs_available = true;
                        }
                        let arr = sg_obj
                            .get_mut("bindings")
                            .and_then(|v| v.as_array_mut())
                            .ok_or_else(|| {
                                ZkError::Other("model_info.bindings must be array".to_string())
                            })?;
                        arr.push(serde_json::Value::from(format!("{cluster}:{container_id}")));
                    }
                    if need_incr_subgraphs_available {
                        let cur = model_obj
                            .get("sub_graphs_available")
                            .and_then(|v| v.as_i64())
                            .unwrap_or(0);
                        model_obj.insert(
                            "sub_graphs_available".to_string(),
                            serde_json::Value::from(cur + 1),
                        );
                    }

                    // container_info[cluster][container_id]['saved_models'] += ["{model}:{sub_graph}"]
                    let cluster_entry = container_info
                        .entry(cluster.to_string())
                        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                    let cluster_obj = cluster_entry.as_object_mut().ok_or_else(|| {
                        ZkError::Other("container_info cluster entry must be object".to_string())
                    })?;
                    let container_entry = cluster_obj
                        .entry(container_id.to_string())
                        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
                    let container_obj = container_entry.as_object_mut().ok_or_else(|| {
                        ZkError::Other("container_info container entry must be object".to_string())
                    })?;
                    let saved_models = container_obj
                        .entry("saved_models".to_string())
                        .or_insert_with(|| serde_json::Value::Array(Vec::new()));
                    let saved_models = saved_models.as_array_mut().ok_or_else(|| {
                        ZkError::Other("container_info.saved_models must be array".to_string())
                    })?;
                    saved_models.push(serde_json::Value::from(format!("{model_name}:{sub_graph}")));
                }
            }
        }

        Ok(serde_json::json!({
            "model_info": serde_json::Value::Object(model_info),
            "container_info": serde_json::Value::Object(container_info),
            "layout_info": serde_json::Value::Object(layout_info),
        }))
    }

    /// Sync backend: subscribe to a model name.
    pub fn subscribe_model(&self, model_name: &str) -> Result<(), ZkError> {
        {
            let mut guard = self.shared.lock.lock();
            if guard.sync_model_name.as_deref() == Some(model_name) {
                return Ok(());
            }
            if guard.sync_model_name.is_some() {
                return Err(ZkError::Other(
                    "subscribe_model called more than once".to_string(),
                ));
            }
            guard.sync_model_name = Some(model_name.to_string());
            guard
                .service_info_map
                .entry(model_name.to_string())
                .or_default();
        }

        let binding_watch_path = format!("/{}/binding/{}", self.shared.bzid, model_name);
        let shared_for_cb = self.shared.clone();
        let model_name_for_cb = model_name.to_string();
        self.shared.children_watch(
            &binding_watch_path,
            Arc::new(move |children| {
                shared_for_cb.bind_callback(&model_name_for_cb, children);
                true
            }),
        )?;
        Ok(())
    }

    /// Sync backend: return (tfs_model_name, grpc_targets) for the subscribed model.
    pub fn get_sync_targets(&self, sub_graph: &str) -> Result<(String, Vec<String>), ZkError> {
        if self.shared.is_lost.load(Ordering::Relaxed) {
            let mut guard = self.shared.lock.lock();
            guard.available_saved_model.clear();
            drop(guard);
            let _ = self.shared.zk.restart();
        }

        let (model_name, sub_graph_map) = {
            let state = self.shared.lock.lock();
            let model_name = state
                .sync_model_name
                .clone()
                .ok_or_else(|| ZkError::Other("subscribe_model not called".to_string()))?;
            let sub_graph_map = state
                .service_info_map
                .get(&model_name)
                .cloned()
                .unwrap_or_default();
            (model_name, sub_graph_map)
        };
        let service_infos = sub_graph_map.get(sub_graph).cloned().unwrap_or_default();
        let grpc_targets = service_infos
            .into_iter()
            .filter_map(|si| si.grpc)
            .collect::<Vec<_>>();
        Ok((format!("{model_name}:{sub_graph}"), grpc_targets))
    }
}

// ---------------------------
// In-memory fake Kazoo client
// ---------------------------

#[derive(Clone)]
struct Node {
    value: Vec<u8>,
    ephemeral: bool,
    children: HashMap<String, String>, // name -> full path
    children_order: Vec<String>,       // insertion order (Python dict parity)
    ctime: i64,
    mtime: i64,
    version: i32,
}

struct Catalog {
    nodes: HashMap<String, Node>, // path -> node
    children_watches: HashMap<String, Vec<(ChildrenWatchHandle, ChildrenWatchEventCallback)>>,
    data_watches: HashMap<String, Vec<(DataWatchHandle, DataWatchCallback)>>,
    notify: Vec<Notify>,
}

enum Notify {
    Children {
        path: String,
        event: Option<WatchEventType>,
    },
    Data {
        path: String,
        event: Option<WatchEventType>,
        data: Option<Vec<u8>>,
        stat: ZnodeStat,
    },
}

enum NotifyCall {
    Children {
        watches: Vec<(ChildrenWatchHandle, ChildrenWatchEventCallback)>,
        children: Vec<String>,
        event: Option<WatchedEvent>,
    },
    Data {
        watches: Vec<(DataWatchHandle, DataWatchCallback)>,
        data: Option<Vec<u8>>,
        stat: ZnodeStat,
        event: Option<WatchedEvent>,
    },
}

impl Catalog {
    fn new() -> Self {
        let mut nodes = HashMap::new();
        let now = now_secs();
        nodes.insert(
            "/".to_string(),
            Node {
                value: Vec::new(),
                ephemeral: false,
                children: HashMap::new(),
                children_order: Vec::new(),
                ctime: now,
                mtime: now,
                version: 0,
            },
        );
        Self {
            nodes,
            children_watches: HashMap::new(),
            data_watches: HashMap::new(),
            notify: Vec::new(),
        }
    }

    fn normalize(path: &str) -> String {
        if path.is_empty() {
            return "/".to_string();
        }
        if path == "/" {
            return "/".to_string();
        }
        // Keep leading slash, drop trailing slashes.
        let mut p = path.to_string();
        while p.ends_with('/') {
            p.pop();
        }
        if !p.starts_with('/') {
            p.insert(0, '/');
        }
        p
    }

    fn parent(path: &str) -> String {
        let p = Self::normalize(path);
        if p == "/" {
            return "/".to_string();
        }
        Path::new(&p)
            .parent()
            .map(|v| v.to_string_lossy().to_string())
            .unwrap_or_else(|| "/".to_string())
    }

    fn basename(path: &str) -> String {
        let p = Self::normalize(path);
        if p == "/" {
            return "/".to_string();
        }
        Path::new(&p)
            .file_name()
            .map(|v| v.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    fn ensure_path(&mut self, path: &str) -> Result<(), ZkError> {
        let path = Self::normalize(path);
        if path == "/" {
            return Ok(());
        }
        let mut current = "/".to_string();
        for comp in path.trim_start_matches('/').split('/') {
            if comp.is_empty() {
                continue;
            }
            let next = if current == "/" {
                format!("/{comp}")
            } else {
                format!("{current}/{comp}")
            };
            if !self.nodes.contains_key(&next) {
                self.create(&next, Vec::new(), false, true)?;
            }
            current = next;
        }
        Ok(())
    }

    fn create(
        &mut self,
        path: &str,
        value: Vec<u8>,
        ephemeral: bool,
        makepath: bool,
    ) -> Result<(), ZkError> {
        let path = Self::normalize(path);
        if self.nodes.contains_key(&path) {
            return Err(ZkError::NodeExists(path));
        }
        let parent = Self::parent(&path);
        if makepath {
            self.ensure_path(&parent)?;
        } else if !self.nodes.contains_key(&parent) {
            return Err(ZkError::NoNode(parent));
        }

        let now = now_secs();
        self.nodes.insert(
            path.clone(),
            Node {
                value: value.clone(),
                ephemeral,
                children: HashMap::new(),
                children_order: Vec::new(),
                ctime: now,
                mtime: now,
                version: 0,
            },
        );

        // Attach to parent.
        if parent != path {
            let name = Self::basename(&path);
            let pnode = self
                .nodes
                .get_mut(&parent)
                .ok_or_else(|| ZkError::NoNode(parent.clone()))?;
            if !pnode.children.contains_key(&name) {
                pnode.children.insert(name.clone(), path.clone());
                pnode.children_order.push(name);
            }
            self.mark_children_notify(&parent, Some(WatchEventType::Child));
        }

        // Node-level notifications.
        self.mark_children_notify(&path, Some(WatchEventType::Created));
        let stat = self.node_stat(&path).unwrap_or_default();
        self.mark_data_notify(&path, Some(WatchEventType::Created), Some(value), stat);
        Ok(())
    }

    fn delete(&mut self, path: &str, recursive: bool) -> Result<(), ZkError> {
        let path = Self::normalize(path);
        if path == "/" {
            return Err(ZkError::Other("cannot delete root".to_string()));
        }
        if !self.nodes.contains_key(&path) {
            return Err(ZkError::NoNode(path));
        }
        // Remove children first if recursive.
        let children: Vec<String> = {
            let node = self.nodes.get(&path).unwrap();
            node.children_order
                .iter()
                .filter_map(|name| node.children.get(name).cloned())
                .collect()
        };
        if !children.is_empty() && !recursive {
            return Err(ZkError::NotEmpty(path));
        }
        for child_path in children {
            self.delete(&child_path, recursive)?;
        }

        let old_data = self.nodes.get(&path).map(|n| n.value.clone());
        let old_stat = self.node_stat(&path).unwrap_or_default();

        let parent = Self::parent(&path);
        let name = Self::basename(&path);
        self.nodes.remove(&path);
        if let Some(pnode) = self.nodes.get_mut(&parent) {
            pnode.children.remove(&name);
            pnode.children_order.retain(|v| v != &name);
            self.mark_children_notify(&parent, Some(WatchEventType::Child));
        }
        self.mark_children_notify(&path, Some(WatchEventType::Deleted));
        self.mark_data_notify(&path, Some(WatchEventType::Deleted), old_data, old_stat);
        Ok(())
    }

    fn set(&mut self, path: &str, value: Vec<u8>) -> Result<(), ZkError> {
        let path = Self::normalize(path);
        let (data, stat) = {
            let node = self
                .nodes
                .get_mut(&path)
                .ok_or_else(|| ZkError::NoNode(path.clone()))?;
            node.value = value;
            node.version += 1;
            node.mtime = now_secs();
            let stat = ZnodeStat {
                ctime: node.ctime,
                mtime: node.mtime,
                version: node.version,
                data_length: node.value.len(),
                num_children: node.children.len(),
            };
            (node.value.clone(), stat)
        };
        self.mark_data_notify(&path, Some(WatchEventType::Changed), Some(data), stat);
        Ok(())
    }

    fn get(&self, path: &str) -> Result<Vec<u8>, ZkError> {
        let path = Self::normalize(path);
        let node = self.nodes.get(&path).ok_or_else(|| ZkError::NoNode(path))?;
        Ok(node.value.clone())
    }

    fn exists(&self, path: &str) -> bool {
        let path = Self::normalize(path);
        self.nodes.contains_key(&path)
    }

    fn get_children(&self, path: &str) -> Result<Vec<String>, ZkError> {
        let path = Self::normalize(path);
        let node = self.nodes.get(&path).ok_or_else(|| ZkError::NoNode(path))?;
        Ok(node.children_order.clone())
    }

    fn register_children_watch_event(
        &mut self,
        path: &str,
        callback: ChildrenWatchEventCallback,
    ) -> Result<ChildrenWatchHandle, ZkError> {
        let path = Self::normalize(path);
        let handle = ChildrenWatchHandle {
            stopped: Arc::new(AtomicBool::new(false)),
        };
        self.children_watches
            .entry(path.clone())
            .or_default()
            .push((handle.clone(), callback));
        // Initial callback if the node exists (Python parity: first call with event None).
        if self.nodes.contains_key(&path) {
            self.mark_children_notify(&path, None);
        }
        Ok(handle)
    }

    fn register_data_watch(
        &mut self,
        path: &str,
        callback: DataWatchCallback,
    ) -> Result<DataWatchHandle, ZkError> {
        let path = Self::normalize(path);
        let handle = DataWatchHandle {
            stopped: Arc::new(AtomicBool::new(false)),
        };
        self.data_watches
            .entry(path.clone())
            .or_default()
            .push((handle.clone(), callback));
        // Initial callback if the node exists.
        if let Some(stat) = self.node_stat(&path) {
            let data = self.nodes.get(&path).map(|n| n.value.clone());
            self.mark_data_notify(&path, None, data, stat);
        }
        Ok(handle)
    }

    fn node_stat(&self, path: &str) -> Option<ZnodeStat> {
        let path = Self::normalize(path);
        let node = self.nodes.get(&path)?;
        Some(ZnodeStat {
            ctime: node.ctime,
            mtime: node.mtime,
            version: node.version,
            data_length: node.value.len(),
            num_children: node.children.len(),
        })
    }

    fn mark_children_notify(&mut self, path: &str, event: Option<WatchEventType>) {
        self.notify.push(Notify::Children {
            path: Self::normalize(path),
            event,
        });
    }

    fn mark_data_notify(
        &mut self,
        path: &str,
        event: Option<WatchEventType>,
        data: Option<Vec<u8>>,
        stat: ZnodeStat,
    ) {
        self.notify.push(Notify::Data {
            path: Self::normalize(path),
            event,
            data,
            stat,
        });
    }

    fn drain_notifications(&mut self) -> Vec<NotifyCall> {
        let mut out = Vec::new();
        let notify = std::mem::take(&mut self.notify);
        for n in notify {
            match n {
                Notify::Children { path, event } => {
                    let watches = self
                        .children_watches
                        .get(&path)
                        .cloned()
                        .unwrap_or_default();
                    if watches.is_empty() {
                        continue;
                    }
                    let children = self.get_children(&path).unwrap_or_default();
                    let event = event.map(|event_type| WatchedEvent {
                        event_type,
                        path: path.clone(),
                    });
                    out.push(NotifyCall::Children {
                        watches,
                        children,
                        event,
                    });
                }
                Notify::Data {
                    path,
                    event,
                    data,
                    stat,
                } => {
                    let watches = self.data_watches.get(&path).cloned().unwrap_or_default();
                    if watches.is_empty() {
                        continue;
                    }
                    let event = event.map(|event_type| WatchedEvent {
                        event_type,
                        path: path.clone(),
                    });
                    out.push(NotifyCall::Data {
                        watches,
                        data,
                        stat,
                        event,
                    });
                }
            }
        }
        out
    }
}

/// In-memory fake Kazoo/ZooKeeper client (used by Rust parity tests).
///
/// This is intentionally minimal: only the methods used by `ZkBackend` and its tests
/// are implemented.
#[derive(Default)]
pub struct FakeKazooClient {
    catalog: parking_lot::Mutex<Option<Catalog>>,
}

impl FakeKazooClient {
    /// Create a new fake client.
    pub fn new() -> Self {
        Self {
            catalog: parking_lot::Mutex::new(None),
        }
    }

    fn with_catalog_mut<T>(
        &self,
        f: impl FnOnce(&mut Catalog) -> Result<T, ZkError>,
    ) -> Result<T, ZkError> {
        let mut guard = self.catalog.lock();
        let cat = guard
            .as_mut()
            .ok_or_else(|| ZkError::Other("client not started".to_string()))?;
        let res = f(cat);
        let notifications = cat.drain_notifications();
        drop(guard);
        Self::run_notifications(notifications);
        res
    }

    fn with_catalog<T>(
        &self,
        f: impl FnOnce(&Catalog) -> Result<T, ZkError>,
    ) -> Result<T, ZkError> {
        let guard = self.catalog.lock();
        let cat = guard
            .as_ref()
            .ok_or_else(|| ZkError::Other("client not started".to_string()))?;
        f(cat)
    }

    fn run_notifications(notifications: Vec<NotifyCall>) {
        for n in notifications {
            match n {
                NotifyCall::Children {
                    watches,
                    children,
                    event,
                } => {
                    for (h, cb) in watches {
                        if h.is_stopped() {
                            continue;
                        }
                        let _ = cb(children.clone(), event.clone());
                    }
                }
                NotifyCall::Data {
                    watches,
                    data,
                    stat,
                    event,
                } => {
                    for (h, cb) in watches {
                        if h.is_stopped() {
                            continue;
                        }
                        let _ = cb(data.clone(), stat, event.clone());
                    }
                }
            }
        }
    }
}

impl ZkClient for FakeKazooClient {
    fn start(&self) -> Result<(), ZkError> {
        *self.catalog.lock() = Some(Catalog::new());
        Ok(())
    }

    fn stop(&self) -> Result<(), ZkError> {
        *self.catalog.lock() = None;
        Ok(())
    }

    fn ensure_path(&self, path: &str) -> Result<(), ZkError> {
        self.with_catalog_mut(|c| c.ensure_path(path))
    }

    fn create(
        &self,
        path: &str,
        value: Vec<u8>,
        ephemeral: bool,
        makepath: bool,
    ) -> Result<(), ZkError> {
        self.with_catalog_mut(|c| c.create(path, value, ephemeral, makepath))
    }

    fn delete(&self, path: &str, recursive: bool) -> Result<(), ZkError> {
        self.with_catalog_mut(|c| c.delete(path, recursive))
    }

    fn set(&self, path: &str, value: Vec<u8>) -> Result<(), ZkError> {
        self.with_catalog_mut(|c| c.set(path, value))
    }

    fn get(&self, path: &str) -> Result<Vec<u8>, ZkError> {
        self.with_catalog(|c| c.get(path))
    }

    fn exists(&self, path: &str) -> bool {
        self.with_catalog(|c| Ok(c.exists(path))).unwrap_or(false)
    }

    fn get_children(&self, path: &str) -> Result<Vec<String>, ZkError> {
        self.with_catalog(|c| c.get_children(path))
    }

    fn children_watch(
        &self,
        path: &str,
        callback: ChildrenWatchCallback,
    ) -> Result<ChildrenWatchHandle, ZkError> {
        let cb: ChildrenWatchEventCallback = Arc::new(move |children, _event| callback(children));
        self.with_catalog_mut(|c| c.register_children_watch_event(path, cb))
    }

    fn children_watch_event(
        &self,
        path: &str,
        callback: ChildrenWatchEventCallback,
    ) -> Result<ChildrenWatchHandle, ZkError> {
        self.with_catalog_mut(|c| c.register_children_watch_event(path, callback))
    }

    fn data_watch(
        &self,
        path: &str,
        callback: DataWatchCallback,
    ) -> Result<DataWatchHandle, ZkError> {
        self.with_catalog_mut(|c| c.register_data_watch(path, callback))
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
