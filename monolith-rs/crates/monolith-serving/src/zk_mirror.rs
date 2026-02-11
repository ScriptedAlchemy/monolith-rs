//! ZooKeeper mirror (Python parity for `monolith/agent_service/zk_mirror.py`).
//!
//! This module mirrors key subtrees in ZooKeeper into an in-memory map and provides helper
//! functions for CRUD operations and querying replica/service state.
//!
//! It also provides watch-driven event emission (portal/publish/service/resource) into a queue,
//! matching the Python agent-service semantics closely enough for parity tests.

use crate::backends::{DataWatchCallback, WatchEventType, ZkClient, ZkError};
use crate::constants::HOST_SHARD_ENV;
use crate::data_def::{
    Event, EventType, ModelMeta, ModelName, ModelState, PublishMeta, PublishType, ReplicaMeta,
    ResourceSpec,
};
use std::collections::VecDeque;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::env;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Mirror of ZK state and helper APIs (Python parity).
#[derive(Clone)]
pub struct ZkMirror {
    zk: Arc<dyn ZkClient>,
    bzid: String,
    sep: String,
    data: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    queue: Arc<Mutex<VecDeque<Event>>>,
    local_host: String,
    is_leader: Arc<std::sync::atomic::AtomicBool>,

    /// Local shard ID used for routing publish/service paths.
    pub tce_shard_id: i32,
    /// Total number of shards used by the deployment.
    pub num_tce_shard: i32,

    /// Base path for resource nodes: `/{bzid}/resource`.
    pub resource_path: String,
    /// Base path for portal nodes: `/{bzid}/portal`.
    pub portal_base_path: String,
    /// Base path for publish nodes: `/{bzid}/publish`.
    pub publish_base_path: String,
    /// Base path for service nodes: `/{bzid}/service`.
    pub service_base_path: String,
}

impl ZkMirror {
    /// Create a new mirror for a given business ID (`bzid`) and shard configuration.
    pub fn new(
        zk: Arc<dyn ZkClient>,
        bzid: impl Into<String>,
        tce_shard_id: i32,
        num_tce_shard: i32,
    ) -> Self {
        let bzid = bzid.into();
        let sep = "/".to_string();
        Self {
            zk,
            bzid: bzid.clone(),
            sep,
            data: Arc::new(Mutex::new(HashMap::new())),
            queue: Arc::new(Mutex::new(VecDeque::new())),
            local_host: "127.0.0.1".to_string(),
            is_leader: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            tce_shard_id,
            num_tce_shard,
            resource_path: format!("/{bzid}/resource"),
            portal_base_path: format!("/{bzid}/portal"),
            publish_base_path: format!("/{bzid}/publish"),
            service_base_path: format!("/{bzid}/service"),
        }
    }

    /// Override the host string used to identify "local" replicas (tests only).
    pub fn set_local_host_for_test(&mut self, host: impl Into<String>) {
        self.local_host = host.into();
    }

    /// Start the underlying ZK client and install the required watches.
    ///
    /// When `is_client` is true, publish watches are skipped (Python parity).
    pub fn start(&self, is_client: bool) -> Result<(), ZkError> {
        self.zk.start()?;
        self.watch_service()?;
        if !is_client {
            self.watch_publish()?;
        }
        Ok(())
    }

    /// Stop the underlying ZK client.
    pub fn stop(&self) -> Result<(), ZkError> {
        self.zk.stop()
    }

    /// Return whether this mirror instance is considered leader.
    pub fn is_leader(&self) -> bool {
        self.is_leader.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Mark this mirror instance as leader.
    pub fn set_leader(&self) {
        self.is_leader
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// Pop the next pending watch event (if any).
    pub fn pop_event(&self) -> Option<Event> {
        self.queue.lock().unwrap().pop_front()
    }

    /// Drain and return all pending watch events.
    pub fn drain_events(&self) -> Vec<Event> {
        let mut q = self.queue.lock().unwrap();
        q.drain(..).collect()
    }

    fn push_event(&self, ev: Event) {
        self.queue.lock().unwrap().push_back(ev);
    }

    // ----------------
    // CRUD wrappers
    // ----------------

    /// Ensure a ZK path exists (creates missing parent components as needed).
    pub fn ensure_path(&self, path: &str) -> Result<(), ZkError> {
        self.zk.ensure_path(path)
    }

    /// Create a node or update it if it already exists (Python parity).
    pub fn create(
        &self,
        path: &str,
        value: Vec<u8>,
        ephemeral: bool,
        makepath: bool,
    ) -> Result<(), ZkError> {
        match self.zk.create(path, value.clone(), ephemeral, makepath) {
            Ok(_) => Ok(()),
            Err(ZkError::NodeExists(_)) => self.zk.set(path, value),
            Err(e) => Err(e),
        }
    }

    /// Set a node's value, creating it when missing (Python parity).
    pub fn set(&self, path: &str, value: Vec<u8>) -> Result<(), ZkError> {
        match self.zk.set(path, value.clone()) {
            Ok(_) => Ok(()),
            Err(ZkError::NoNode(_)) => self.zk.create(path, value, false, true),
            Err(e) => Err(e),
        }
    }

    /// Check whether a node exists.
    ///
    /// Falls back to the in-memory cache when the underlying client returns false, which
    /// matches the behavior relied upon by the parity tests.
    pub fn exists(&self, path: &str) -> bool {
        if self.zk.exists(path) {
            return true;
        }
        // Python parity fallback when the underlying client errors: check in-memory map.
        self.data.lock().unwrap().contains_key(path)
    }

    /// Delete a node.
    ///
    /// Deleting a missing node is treated as success (Python parity).
    pub fn delete(&self, path: &str, recursive: bool) -> Result<(), ZkError> {
        match self.zk.delete(path, recursive) {
            Ok(_) => Ok(()),
            Err(ZkError::NoNode(_)) => Ok(()),
            Err(ZkError::NotEmpty(_)) => self.zk.delete(path, true),
            Err(e) => Err(e),
        }
    }

    /// Retrieve cached bytes for a given path (as seen by watches).
    pub fn get_cached(&self, path: &str) -> Option<Vec<u8>> {
        self.data.lock().unwrap().get(path).cloned()
    }

    fn get_children_cached(&self, path: &str) -> Vec<String> {
        let data = self.data.lock().unwrap();
        let length = path.split(&self.sep).count();
        let mut out: BTreeSet<String> = BTreeSet::new();
        for p in data.keys() {
            if p.starts_with(path) {
                let parts: Vec<&str> = p.split(&self.sep).collect();
                if parts.len() > length {
                    out.insert(parts[length].to_string());
                }
            }
        }
        out.into_iter().collect()
    }

    // ----------------
    // Properties
    // ----------------

    /// Replica ID from `REPLICA_ID` (or -1 when unset/invalid).
    pub fn tce_replica_id(&self) -> i32 {
        env::var("REPLICA_ID")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
            .unwrap_or(-1)
    }

    /// Publish this process's resource information under the resource subtree.
    pub fn report_resource(&self, resource: &ResourceSpec) -> Result<(), ZkError> {
        let path = resource.get_path(&self.resource_path);
        let value = resource.serialize();
        self.create(&path, value, true, true)
    }

    /// Return all cached resource specs currently visible under the resource subtree.
    pub fn resources(&self) -> Vec<ResourceSpec> {
        let children = self.get_children_cached(&self.resource_path);
        let mut out = Vec::new();
        for child in children {
            let p = Path::new(&self.resource_path)
                .join(child)
                .to_string_lossy()
                .to_string();
            if let Some(bytes) = self.get_cached(&p) {
                if let Ok(rs) = ResourceSpec::deserialize(&bytes) {
                    out.push(rs);
                }
            }
        }
        out
    }

    // ----------------
    // Publish handling
    // ----------------

    /// Write publish metas for models being loaded, updating only when values changed.
    pub fn publish_loading(&self, pms: &[PublishMeta]) -> Result<(), ZkError> {
        for pm in pms {
            let path = pm.get_path(&self.publish_base_path);
            let value = pm.serialize();
            let loc = self.get_cached(&path);
            if loc.as_deref() != Some(&value) {
                self.create(&path, value, false, true)?;
            }
        }
        Ok(())
    }

    /// Compute the expected set of load operations for this shard/replica.
    ///
    /// This mirrors the Python agent-service selection logic, including the handling of
    /// shard/replica-specific and non-specific publish entries.
    pub fn expected_loading(&self) -> HashMap<ModelName, PublishMeta> {
        // Mirror Python logic: compute "select" once total publish count arrived,
        // then derive expected for current shard/replica.
        let nodes = self.get_children_cached(&self.publish_base_path);
        let mut models: HashMap<String, i32> = HashMap::new();
        let mut select: Vec<PublishMeta> = Vec::new();
        let mut shortest_sub_model_pm: HashMap<String, PublishMeta> = HashMap::new();

        for node in nodes {
            let path = Path::new(&self.publish_base_path)
                .join(&node)
                .to_string_lossy()
                .to_string();
            let data = match self.get_cached(&path) {
                Some(d) => d,
                None => continue,
            };
            let pm = match PublishMeta::deserialize(&data) {
                Ok(v) => v,
                Err(_) => continue,
            };
            let parts: Vec<&str> = node.split(':').collect();
            if parts.len() < 3 {
                continue;
            }
            let model_name = parts[2].to_string();
            *models.entry(model_name.clone()).or_insert(0) += 1;

            let sub_len = pm.sub_models.as_ref().map(|m| m.len()).unwrap_or_default();
            let cur_len = shortest_sub_model_pm
                .get(&model_name)
                .and_then(|p| p.sub_models.as_ref().map(|m| m.len()))
                .unwrap_or(usize::MAX);
            if !shortest_sub_model_pm.contains_key(&model_name) || cur_len > sub_len {
                shortest_sub_model_pm.insert(model_name.clone(), pm.clone());
            }

            if let Some(total) = Some(pm.total_publish_num) {
                if models.get(&model_name).copied().unwrap_or_default() == total {
                    if let Some(sel) = shortest_sub_model_pm.get(&model_name) {
                        select.push(sel.clone());
                    }
                }
            }
        }

        let mut expected: HashMap<String, PublishMeta> = HashMap::new();
        for mut pm in select {
            // Load the publish meta for our replica if present.
            let self_path = Path::new(&self.publish_base_path)
                .join(format!(
                    "{}:{}:{}",
                    self.tce_shard_id,
                    self.tce_replica_id(),
                    pm.model_name.clone().unwrap_or_default()
                ))
                .to_string_lossy()
                .to_string();
            if let Some(bytes) = self.get_cached(&self_path) {
                if let Ok(v) = PublishMeta::deserialize(&bytes) {
                    pm = v;
                }
            }
            let model_name = pm.model_name.clone().unwrap_or_default();
            if pm.ptype != PublishType::Load {
                continue;
            }

            let pm_shard = pm.shard_id.unwrap_or_default();
            let pm_replica = pm.replica_id;
            if pm_shard == self.tce_shard_id && pm_replica == self.tce_replica_id() {
                expected.insert(model_name, pm);
            } else if pm_shard == self.tce_shard_id && !pm.is_spec {
                if !expected.contains_key(&model_name) {
                    pm.replica_id = self.tce_replica_id();
                    expected.insert(model_name, pm);
                }
            } else if !expected.contains_key(&model_name) {
                pm.shard_id = Some(self.tce_shard_id);
                pm.replica_id = self.tce_replica_id();
                if let Some(sm) = pm.sub_models.take() {
                    let filtered = sm
                        .into_iter()
                        .filter(|(k, _)| k.starts_with("entry"))
                        .collect::<HashMap<_, _>>();
                    pm.sub_models = Some(filtered);
                }
                expected.insert(model_name, pm);
            }
        }
        expected
    }

    // ----------------
    // Service handling
    // ----------------

    /// Update the service subtree to match `replicas` for the local host/replica.
    pub fn update_service(&self, replicas: &[ReplicaMeta]) -> Result<(), ZkError> {
        let mut need_create_or_update: HashMap<String, Vec<u8>> = HashMap::new();
        let mut local_load_paths: HashSet<String> = HashSet::new();

        for rm in replicas {
            let path = rm.get_path(&self.bzid, &self.sep);
            let value = rm.serialize();
            local_load_paths.insert(path.clone());
            let loc = self.get_cached(&path);
            if loc.as_deref() != Some(&value) {
                need_create_or_update.insert(path, value);
            }
        }

        // Remove local replicas that are no longer present.
        let local_replica_paths = self.local_replica_paths();
        for p in local_replica_paths.difference(&local_load_paths) {
            let _ = self.delete(p, false);
        }

        for (path, value) in need_create_or_update {
            let _ = self.create(&path, value, true, true);
        }
        Ok(())
    }

    /// Return ZK paths for replicas that belong to this host and replica ID.
    pub fn local_replica_paths(&self) -> HashSet<String> {
        let data = self.data.lock().unwrap();
        let mut out = HashSet::new();
        for (path, value) in data.iter() {
            if path.starts_with(&self.service_base_path) {
                if let Ok(rm) = ReplicaMeta::deserialize(value) {
                    let host = rm
                        .address
                        .as_deref()
                        .and_then(|a| a.split(':').next())
                        .unwrap_or_default()
                        .to_string();
                    if host == self.local_host && rm.replica == self.tce_replica_id() {
                        out.insert(path.clone());
                    }
                }
            }
        }
        out
    }

    /// Return available replicas for all models/tasks of a given `server_type`.
    pub fn get_all_replicas(&self, server_type: &str) -> HashMap<String, Vec<ReplicaMeta>> {
        let data = self.data.lock().unwrap();
        let mut out: HashMap<String, Vec<ReplicaMeta>> = HashMap::new();
        for (path, value) in data.iter() {
            if path.starts_with(&self.service_base_path) {
                let raw_key = path[self.service_base_path.len()..].trim_matches('/');
                let replaced = raw_key.replace('/', ":");
                let parts: Vec<&str> = replaced.split(':').collect();
                if parts.len() < 4 {
                    continue;
                }
                let (model, st, task) = (parts[0], parts[1], parts[2]);
                if st == server_type {
                    if let Ok(rm) = ReplicaMeta::deserialize(value) {
                        if rm.stat == ModelState::Available as i32 {
                            out.entry(format!("{model}:{server_type}:{task}"))
                                .or_default()
                                .push(rm);
                        }
                    }
                }
            }
        }
        out
    }

    /// Return available replicas grouped by task for a model and server type.
    pub fn get_model_replicas(
        &self,
        model_name: &str,
        server_type: &str,
    ) -> HashMap<String, Vec<ReplicaMeta>> {
        let base_path = Path::new(&self.service_base_path)
            .join(model_name)
            .to_string_lossy()
            .to_string();
        let tasks = self.get_children_cached(&base_path);
        let mut out: HashMap<String, Vec<ReplicaMeta>> = HashMap::new();
        for task in tasks {
            if !task.starts_with(&server_type.to_lowercase()) {
                continue;
            }
            let task_path = Path::new(&base_path)
                .join(&task)
                .to_string_lossy()
                .to_string();
            let replicas = self.get_children_cached(&task_path);
            for replica in replicas {
                let p = Path::new(&task_path)
                    .join(&replica)
                    .to_string_lossy()
                    .to_string();
                if let Some(bytes) = self.get_cached(&p) {
                    if let Ok(rm) = ReplicaMeta::deserialize(&bytes) {
                        if rm.stat == ModelState::Available as i32 {
                            out.entry(format!("{model_name}:{task}"))
                                .or_default()
                                .push(rm);
                        }
                    }
                }
            }
        }
        out
    }

    /// Return available replicas for a specific model/task.
    pub fn get_task_replicas(
        &self,
        model_name: &str,
        server_type: &str,
        task: i32,
    ) -> Vec<ReplicaMeta> {
        let path = Path::new(&self.service_base_path)
            .join(model_name)
            .join(format!("{}:{task}", server_type.to_lowercase()))
            .to_string_lossy()
            .to_string();
        let mut out = Vec::new();
        for child in self.get_children_cached(&path) {
            let p = Path::new(&path).join(child).to_string_lossy().to_string();
            if let Some(bytes) = self.get_cached(&p) {
                if let Ok(rm) = ReplicaMeta::deserialize(&bytes) {
                    if rm.stat == ModelState::Available as i32 {
                        out.push(rm);
                    }
                }
            }
        }
        out
    }

    /// Return a specific available replica (if present and `Available`).
    pub fn get_replica(
        &self,
        model_name: &str,
        server_type: &str,
        task: i32,
        replica: i32,
    ) -> Option<ReplicaMeta> {
        let path = Path::new(&self.service_base_path)
            .join(model_name)
            .join(format!("{}:{task}", server_type.to_lowercase()))
            .join(replica.to_string())
            .to_string_lossy()
            .to_string();
        let bytes = self.get_cached(&path)?;
        let rm = ReplicaMeta::deserialize(&bytes).ok()?;
        if rm.stat == ModelState::Available as i32 {
            Some(rm)
        } else {
            None
        }
    }

    // ----------------
    // Watches
    // ----------------

    /// Watch the portal subtree and emit portal events into the queue.
    pub fn watch_portal(&self) -> Result<(), ZkError> {
        self.zk.ensure_path(&self.portal_base_path)?;
        self.zk.ensure_path(&self.publish_base_path)?;

        // portal/publish conscience check (Python parity)
        let models_in_portal: HashSet<String> = self
            .zk
            .get_children(&self.portal_base_path)
            .unwrap_or_default()
            .into_iter()
            .collect();
        let models_in_publish: HashSet<String> = self
            .zk
            .get_children(&self.publish_base_path)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|item| item.split(':').last().map(|s| s.to_string()))
            .collect();
        if !models_in_publish.is_empty() {
            let remove: HashSet<String> = if models_in_portal.is_empty() {
                models_in_publish
            } else {
                models_in_publish
                    .difference(&models_in_portal)
                    .cloned()
                    .collect()
            };
            for model in remove {
                let keys = self
                    .zk
                    .get_children(&self.publish_base_path)
                    .unwrap_or_default();
                for key in keys {
                    if key.ends_with(&model) {
                        let p = Path::new(&self.publish_base_path)
                            .join(key)
                            .to_string_lossy()
                            .to_string();
                        let _ = self.delete(&p, true);
                    }
                }
            }
        }

        let models_seen: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
        let models_seen_cb = models_seen.clone();
        let this = self.clone();
        self.zk.children_watch(
            &self.portal_base_path,
            Arc::new(move |children| {
                if children.is_empty() {
                    return true;
                }
                for model in children {
                    let mut seen = models_seen_cb.lock().unwrap();
                    if seen.contains(&model) {
                        continue;
                    }
                    seen.insert(model.clone());
                    drop(seen);

                    let this2 = this.clone();
                    let data_path = Path::new(&this2.portal_base_path)
                        .join(&model)
                        .to_string_lossy()
                        .to_string();
                    let data_path_for_cb = data_path.clone();
                    let this_for_cb = this2.clone();
                    let model_name_for_cb = model.clone();
                    let cb: DataWatchCallback = Arc::new(move |data, _stat, event| {
                        let action_str = if event.is_none() {
                            if data.is_none() {
                                "DELETED".to_string()
                            } else {
                                "NONE".to_string()
                            }
                        } else {
                            match event.as_ref().unwrap().event_type {
                                WatchEventType::Created => "CREATED".to_string(),
                                WatchEventType::Deleted => "DELETED".to_string(),
                                _ => "NONE".to_string(),
                            }
                        };

                        let mut mm = if let Some(bytes) = data.as_ref().filter(|b| !b.is_empty()) {
                            ModelMeta::deserialize(bytes).unwrap_or_default()
                        } else {
                            ModelMeta::default()
                        };
                        if mm.model_name.is_none() {
                            mm.model_name = Some(model_name_for_cb.clone());
                        }
                        mm.action = action_str;
                        this_for_cb.push_event(Event {
                            path: Some(data_path_for_cb.clone()),
                            data: mm.serialize(),
                            etype: EventType::Portal,
                        });
                        true
                    });
                    let _ = this2.zk.data_watch(&data_path, cb);
                }
                true
            }),
        )?;

        Ok(())
    }

    /// Watch the publish subtree and emit publish events into the queue.
    pub fn watch_publish(&self) -> Result<(), ZkError> {
        self.zk.ensure_path(&self.publish_base_path)?;
        let publish_seen: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
        let publish_seen_cb = publish_seen.clone();
        let this = self.clone();

        let get_publish_cnt = move |this: &ZkMirror, model_name: &str| -> i32 {
            let data = this.data.lock().unwrap();
            data.keys()
                .filter(|p| p.starts_with(&this.publish_base_path) && p.ends_with(model_name))
                .count() as i32
        };

        self.zk.children_watch(
            &self.publish_base_path,
            Arc::new(move |children| {
                if children.is_empty() {
                    return true;
                }
                for pub_node in children {
                    let mut seen = publish_seen_cb.lock().unwrap();
                    if seen.contains(&pub_node) {
                        continue;
                    }
                    seen.insert(pub_node.clone());
                    drop(seen);

                    let this2 = this.clone();
                    let data_path = Path::new(&this2.publish_base_path)
                        .join(&pub_node)
                        .to_string_lossy()
                        .to_string();
                    let data_path_for_cb = data_path.clone();
                    let this_for_cb = this2.clone();
                    let cb: DataWatchCallback = Arc::new(move |data, _stat, event| {
                        let data = data.or_else(|| this_for_cb.get_cached(&data_path_for_cb));
                        let Some(bytes) = data.clone() else {
                            return true;
                        };
                        if bytes.is_empty() {
                            return true;
                        }
                        let pm = match PublishMeta::deserialize(&bytes) {
                            Ok(v) => v,
                            Err(_) => return true,
                        };

                        {
                            let mut m = this_for_cb.data.lock().unwrap();
                            match event.as_ref().map(|e| &e.event_type) {
                                None | Some(WatchEventType::Created) => {
                                    if pm.ptype == PublishType::Load {
                                        m.insert(data_path_for_cb.clone(), bytes.clone());
                                    } else {
                                        m.remove(&data_path_for_cb);
                                    }
                                }
                                Some(WatchEventType::Deleted) => {
                                    m.remove(&data_path_for_cb);
                                }
                                _ => {}
                            }
                        }

                        let cnt = get_publish_cnt(
                            &this_for_cb,
                            pm.model_name.as_deref().unwrap_or_default(),
                        );
                        if cnt == 0 || cnt == pm.total_publish_num {
                            // Send event to unload/load when all publish arrived or all removed.
                            let event_data = this_for_cb
                                .get_cached(&pm.get_path(&this_for_cb.publish_base_path))
                                .unwrap_or(bytes);
                            this_for_cb.push_event(Event {
                                path: Some(data_path_for_cb.clone()),
                                data: event_data,
                                etype: EventType::Publish,
                            });
                        }
                        true
                    });
                    let _ = this2.zk.data_watch(&data_path, cb);
                }
                true
            }),
        )?;
        Ok(())
    }

    /// Watch the resource subtree and update the in-memory cache.
    pub fn watch_resource(&self) -> Result<(), ZkError> {
        let instances_seen: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
        let instances_seen_cb = instances_seen.clone();
        let this = self.clone();
        self.zk.children_watch(
            &self.resource_path,
            Arc::new(move |children| {
                if children.is_empty() {
                    return true;
                }
                for inst in children {
                    let mut seen = instances_seen_cb.lock().unwrap();
                    if seen.contains(&inst) {
                        continue;
                    }
                    seen.insert(inst.clone());
                    drop(seen);

                    let this2 = this.clone();
                    let data_path = Path::new(&this2.resource_path)
                        .join(&inst)
                        .to_string_lossy()
                        .to_string();
                    let data_path_for_cb = data_path.clone();
                    let this_for_cb = this2.clone();
                    let cb: DataWatchCallback = Arc::new(move |data, _stat, event| {
                        let data = data.or_else(|| this_for_cb.get_cached(&data_path_for_cb));
                        let mut m = this_for_cb.data.lock().unwrap();
                        match event.as_ref().map(|e| &e.event_type) {
                            None
                            | Some(WatchEventType::Created)
                            | Some(WatchEventType::Changed) => {
                                if let Some(bytes) = data {
                                    m.insert(data_path_for_cb.clone(), bytes);
                                }
                            }
                            Some(WatchEventType::Deleted) => {
                                m.remove(&data_path_for_cb);
                            }
                            _ => {}
                        }
                        true
                    });
                    let _ = this2.zk.data_watch(&data_path, cb);
                }
                true
            }),
        )?;
        Ok(())
    }

    /// Watch the service subtree and update the in-memory cache.
    pub fn watch_service(&self) -> Result<(), ZkError> {
        self.zk.ensure_path(&self.service_base_path)?;
        let children_set: Arc<Mutex<HashSet<String>>> = Arc::new(Mutex::new(HashSet::new()));
        let this = self.clone();
        let children_set_model = children_set.clone();

        // model -> tasks -> replicas -> data watch
        let service_base_path = this.service_base_path.clone();
        let zk_root = this.zk.clone();
        self.zk.children_watch(
            &self.service_base_path,
            Arc::new(move |models| {
                if models.is_empty() {
                    return true;
                }
                for model in models {
                    let mut set = children_set_model.lock().unwrap();
                    if set.contains(&model) {
                        continue;
                    }
                    set.insert(model.clone());
                    drop(set);

                    let model_path = Path::new(&service_base_path)
                        .join(&model)
                        .to_string_lossy()
                        .to_string();
                    let this2 = this.clone();
                    let model_path_for_watch = model_path.clone();
                    let children_set_task = children_set.clone();
                    let model_name = model.clone();
                    let zk_model_outer = zk_root.clone();
                    let _ = zk_root.children_watch(
                        &model_path_for_watch,
                        Arc::new(move |tasks| {
                            if tasks.is_empty() {
                                return true;
                            }
                            let zk_model = zk_model_outer.clone();
                            for task in tasks {
                                let key = format!("{model_name}:{task}");
                                let mut set = children_set_task.lock().unwrap();
                                if set.contains(&key) {
                                    continue;
                                }
                                set.insert(key);
                                drop(set);

                                let task_path = Path::new(&model_path)
                                    .join(&task)
                                    .to_string_lossy()
                                    .to_string();
                                let this3 = this2.clone();
                                let task_path_for_watch = task_path.clone();
                                let children_set_replica = children_set_task.clone();
                                let model_name2 = model_name.clone();
                                let task_name2 = task.clone();
                                let zk_task_outer = zk_model.clone();
                                let task_path_for_watch_cb = task_path_for_watch.clone();
                                let _ = zk_model.children_watch(
                                    &task_path_for_watch,
                                    Arc::new(move |replicas| {
                                        if replicas.is_empty() {
                                            return true;
                                        }
                                        let zk_task = zk_task_outer.clone();
                                        for replica in replicas {
                                            let key =
                                                format!("{model_name2}:{task_name2}:{replica}");
                                            let mut set = children_set_replica.lock().unwrap();
                                            if set.contains(&key) {
                                                continue;
                                            }
                                            set.insert(key);
                                            drop(set);

                                            let data_path = Path::new(&task_path_for_watch_cb)
                                                .join(&replica)
                                                .to_string_lossy()
                                                .to_string();
                                            let data_path_for_cb = data_path.clone();
                                            let this_for_cb = this3.clone();
                                            let this_for_watch = this3.clone();
                                            let cb: DataWatchCallback =
                                                Arc::new(move |data, _stat, event| {
                                                    let data = data.or_else(|| {
                                                        this_for_cb.get_cached(&data_path_for_cb)
                                                    });
                                                    let mut m = this_for_cb.data.lock().unwrap();
                                                    match event.as_ref().map(|e| &e.event_type) {
                                                        None
                                                        | Some(WatchEventType::Created)
                                                        | Some(WatchEventType::Changed) => {
                                                            if let Some(bytes) = data {
                                                                m.insert(
                                                                    data_path_for_cb.clone(),
                                                                    bytes,
                                                                );
                                                            }
                                                        }
                                                        Some(WatchEventType::Deleted) => {
                                                            m.remove(&data_path_for_cb);
                                                        }
                                                        _ => {}
                                                    }
                                                    true
                                                });
                                            let _ = zk_task.data_watch(&data_path, cb);
                                            let _ = this_for_watch; // keep mirror alive for callback captures
                                        }
                                        true
                                    }),
                                );
                            }
                            true
                        }),
                    );
                }
                true
            }),
        )?;
        Ok(())
    }
}

/// Ensure the shard count env var is present for parity tests (mirrors Python).
pub fn set_host_shard_env(num_shards: i32) {
    env::set_var(HOST_SHARD_ENV, num_shards.to_string());
}
