//! Unified container agent (Python parity for `monolith/agent_service/agent_v3.py`).
//!
//! This module implements the minimal AgentV3 flow exercised by the Python tests:
//! - registers `ContainerServiceInfo` to ZK on start
//! - subscribes to a layout path and writes TF Serving model_config pbtxt
//! - reports available models back to ZK binding paths
//! - exposes an address map via AgentService (discovery) using `AgentDataProvider`

#![cfg(feature = "grpc")]

use crate::agent_service_discovery::{
    AgentDataProvider, AgentDiscoveryServer, AgentServiceDiscoveryImpl,
};
use crate::backends::{
    Container, ContainerServiceInfo, SavedModel, SavedModelDeployConfig, ZkBackend, ZkClient,
    ZkError,
};
use crate::error::{ServingError, ServingResult};
use crate::tfs_wrapper::{FakeTfsWrapper, TfsWrapper};
use crate::utils::{get_local_ip, normalize_regex, write_to_tmp_file, AgentConfig, DeployType};
use monolith_proto::descriptor_pool::descriptor_pool;
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use parking_lot::Mutex;
use prost_reflect::text_format::FormatOptions;
use prost_reflect::DynamicMessage;
use regex::Regex;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::task::JoinHandle;
use tokio::time::{sleep, Duration};

/// A small interface for the TFServing wrapper used by AgentV3.
#[tonic::async_trait]
pub trait TfsWrapperApi: Send + Sync {
    /// Whether remote operations should use the gRPC address (vs archon).
    fn is_grpc_remote_op(&self) -> bool {
        true
    }
    /// Start the underlying TF Serving process (no-op for the fake wrapper).
    fn start(&self) -> ServingResult<()> {
        Ok(())
    }
    /// Stop the underlying TF Serving process (no-op for the fake wrapper).
    fn stop(&self) -> ServingResult<()> {
        Ok(())
    }
    /// Poll the underlying process exit status, if applicable.
    fn poll(&self) -> Option<i32> {
        None
    }

    /// List saved model names referenced by the active model_config.
    fn list_saved_models(&self) -> ServingResult<Vec<String>>;
    /// Return a best-effort status per saved model.
    async fn list_saved_models_status(
        &self,
    ) -> ServingResult<HashMap<String, tfserving_apis::ModelVersionStatus>>;
}

#[tonic::async_trait]
impl TfsWrapperApi for TfsWrapper {
    fn list_saved_models(&self) -> ServingResult<Vec<String>> {
        self.list_saved_models()
    }

    async fn list_saved_models_status(
        &self,
    ) -> ServingResult<HashMap<String, tfserving_apis::ModelVersionStatus>> {
        self.list_saved_models_status().await
    }
}

#[tonic::async_trait]
impl TfsWrapperApi for FakeTfsWrapper {
    fn list_saved_models(&self) -> ServingResult<Vec<String>> {
        self.list_saved_models()
    }

    async fn list_saved_models_status(
        &self,
    ) -> ServingResult<HashMap<String, tfserving_apis::ModelVersionStatus>> {
        Ok(self.list_saved_models_status()?)
    }
}

#[derive(Clone)]
struct LayoutFilter {
    re: Regex,
    cond: String,
}

impl LayoutFilter {
    fn parse(raw: &str, shard_id: i32, shard_num: i32) -> ServingResult<Self> {
        // Python replaces `${shard_id}` and `${shard_num}` first.
        let mut raw = raw.replace("${shard_id}", &shard_id.to_string());
        raw = raw.replace("${shard_num}", &shard_num.to_string());

        // Python expects `match;cond`. Be tolerant: treat "ps_0" as "ps_0;True".
        let (m, cond) = raw
            .split_once(';')
            .map(|(a, b)| (a.trim(), b.trim()))
            .unwrap_or((raw.trim(), "True"));

        let re = Regex::new(&normalize_regex(m))
            .map_err(|e| ServingError::ConfigError(format!("invalid layout filter regex: {e}")))?;
        Ok(Self {
            re,
            cond: cond.to_string(),
        })
    }

    fn accepts(&self, sub_graph: &str) -> bool {
        let Some(caps) = self.re.captures(sub_graph) else {
            return false;
        };

        // Minimal "safe eval" subset: support `True`/`False` and integer comparisons.
        // For more complex expressions, default to false to avoid accidental over-acceptance.
        let cond = self.cond.trim();
        if cond.eq_ignore_ascii_case("true") {
            return true;
        }
        if cond.eq_ignore_ascii_case("false") {
            return false;
        }

        // Support patterns like `i % 2 == 1` where identifiers come from named groups.
        // Very small evaluator: only handles `{var} % {int} == {int}`.
        // This is sufficient for the documented layout_filters use cases.
        let parts: Vec<&str> = cond.split_whitespace().collect();
        if parts.len() == 5 && parts[1] == "%" && parts[3] == "==" {
            let var = parts[0];
            let m = parts[2].parse::<i64>().ok().unwrap_or(0);
            let rhs = parts[4].parse::<i64>().ok().unwrap_or(0);
            let v = caps
                .name(var)
                .and_then(|m| m.as_str().parse::<i64>().ok())
                .unwrap_or(0);
            if m == 0 {
                return false;
            }
            return v % m == rhs;
        }

        false
    }
}

fn gen_empty_model_config_file() -> ServingResult<PathBuf> {
    write_to_tmp_file(b"model_config_list {}")
}

fn model_server_config_pbtxt(cfg: &tfserving_apis::ModelServerConfig) -> ServingResult<String> {
    let pool = descriptor_pool();
    let desc = pool
        .get_message_by_name("tensorflow.serving.ModelServerConfig")
        .ok_or_else(|| {
            ServingError::ConfigError("missing ModelServerConfig descriptor".to_string())
        })?;
    let mut dyn_msg = DynamicMessage::new(desc);
    dyn_msg.transcode_from(cfg).map_err(|e| {
        ServingError::ConfigError(format!("failed to transcode ModelServerConfig: {e}"))
    })?;
    Ok(dyn_msg.to_text_format_with_options(&FormatOptions::new().pretty(true)))
}

fn gen_model_config_compat(
    name: &str,
    base_path: &str,
    version_policy: &str,
) -> tfserving_apis::ModelConfig {
    use tfserving_apis::file_system_storage_path_source_config::servable_version_policy as vp;
    use tfserving_apis::file_system_storage_path_source_config::ServableVersionPolicy;

    // Upstream TF Serving protos in this repo do not include `latest_once`.
    // Map it to `latest` for best-effort compatibility.
    let policy_choice = match version_policy.to_ascii_lowercase().as_str() {
        "all" => Some(vp::PolicyChoice::All(vp::All {})),
        "specific" => Some(vp::PolicyChoice::Specific(vp::Specific {
            versions: vec![1],
        })),
        // "latest_once" -> latest
        _ => Some(vp::PolicyChoice::Latest(vp::Latest { num_versions: 1 })),
    };

    tfserving_apis::ModelConfig {
        name: name.to_string(),
        base_path: base_path.to_string(),
        model_type: 0, // deprecated
        model_platform: "tensorflow".to_string(),
        model_version_policy: Some(ServableVersionPolicy { policy_choice }),
        version_labels: HashMap::new(),
        logging_config: None,
    }
}

/// Rust AgentV3 (unified container agent).
pub struct AgentV3 {
    config: AgentConfig,
    model_config_path: PathBuf,
    tfs_wrapper: Arc<dyn TfsWrapperApi>,
    backend: ZkBackend,
    container: Container,
    service_info: ContainerServiceInfo,
    layout_filters: Vec<LayoutFilter>,

    agent_server: Mutex<Option<AgentDiscoveryServer>>,
    bg_tasks: Mutex<Vec<JoinHandle<()>>>,
    stop: tokio::sync::watch::Sender<bool>,
}

impl AgentV3 {
    /// Create a new AgentV3 instance.
    ///
    /// The caller provides a ZK client implementation (real or fake). The TFServing wrapper
    /// defaults to a file-based fake wrapper pointing at the generated model_config file; callers
    /// can override it with [`Self::set_tfs_wrapper`].
    pub fn new(config: AgentConfig, zk: Arc<dyn ZkClient>) -> ServingResult<Self> {
        if config.deploy_type != DeployType::Unified {
            return Err(ServingError::ConfigError(
                "agent v3 only supports unified deploy_type".to_string(),
            ));
        }
        if config.agent_version != 3 {
            return Err(ServingError::ConfigError(format!(
                "agent version {} unexpected",
                config.agent_version
            )));
        }

        let model_config_path = gen_empty_model_config_file()?;
        let tfs_wrapper: Arc<dyn TfsWrapperApi> =
            Arc::new(FakeTfsWrapper::new(model_config_path.clone()));

        let shard_id = AgentConfig::shard_id().max(0);
        let shard_num = config.num_shard.max(1);
        let mut layout_filters = Vec::new();
        for raw in config.layout_filters.iter() {
            layout_filters.push(LayoutFilter::parse(raw, shard_id, shard_num)?);
        }

        let container = Container::new(config.container_cluster(), config.container_id());
        let local_ip = get_local_ip();
        let layout_path = config.layout_path().unwrap_or_default();
        let debug_info = json!({
            "layout_path": layout_path,
            "layout_filters": layout_filters
                .iter()
                .map(|f| format!("{};{}", f.re.as_str(), f.cond))
                .collect::<Vec<_>>(),
        })
        .to_string();

        let service_info = ContainerServiceInfo {
            grpc: Some(format!("{local_ip}:{}", config.tfs_port_grpc)),
            http: Some(format!("{local_ip}:{}", config.tfs_port_http)),
            archon: Some(format!("{local_ip}:{}", config.tfs_port_archon)),
            agent: Some(format!("{local_ip}:{}", config.agent_port)),
            idc: AgentConfig::idc(),
            debug_info: Some(debug_info),
        };

        let backend = ZkBackend::new(config.bzid.clone(), zk);
        let (tx, _rx) = tokio::sync::watch::channel(false);

        Ok(Self {
            config,
            model_config_path,
            tfs_wrapper,
            backend,
            container,
            service_info,
            layout_filters,
            agent_server: Mutex::new(None),
            bg_tasks: Mutex::new(Vec::new()),
            stop: tx,
        })
    }

    /// Return the path to the generated TF Serving `model_config` file.
    pub fn model_config_path(&self) -> &Path {
        &self.model_config_path
    }

    /// Return the ZK backend used by the agent.
    pub fn backend(&self) -> &ZkBackend {
        &self.backend
    }

    /// Return this agent's container identity.
    pub fn container(&self) -> &Container {
        &self.container
    }

    /// Return the service-info payload reported to ZooKeeper.
    pub fn service_info(&self) -> &ContainerServiceInfo {
        &self.service_info
    }

    /// Override the TFServing wrapper implementation used by this agent.
    pub fn set_tfs_wrapper(&mut self, wrapper: Arc<dyn TfsWrapperApi>) {
        self.tfs_wrapper = wrapper;
    }

    fn gen_addrs_map(&self) -> HashMap<String, Vec<String>> {
        let service_map = self.backend.get_service_map();
        let mut addrs_map: HashMap<String, Vec<String>> = HashMap::new();
        for (model_name, sub_map) in service_map {
            for (sub_graph, infos) in sub_map {
                let key = format!("{model_name}:{sub_graph}");
                let addrs = infos
                    .into_iter()
                    .filter_map(|si| {
                        if self.tfs_wrapper.is_grpc_remote_op() {
                            si.grpc
                        } else {
                            si.archon
                        }
                    })
                    .collect::<Vec<_>>();
                addrs_map.insert(key, addrs);
            }
        }
        addrs_map
    }

    /// Query TF Serving and sync the set of available saved models back to ZooKeeper.
    pub async fn sync_available_saved_models(&self) -> Result<(), ZkError> {
        let status = self
            .tfs_wrapper
            .list_saved_models_status()
            .await
            .map_err(|e| ZkError::Other(format!("list_saved_models_status failed: {e}")))?;
        let mut available: HashSet<SavedModel> = HashSet::new();
        for (saved_model_name, st) in status {
            if st.state == tfserving_apis::model_version_status::State::Available as i32 {
                let mut it = saved_model_name.split(':');
                let model_name = it.next().unwrap_or_default();
                let sub_graph = it.next().unwrap_or_default();
                if !model_name.is_empty() && !sub_graph.is_empty() {
                    available.insert(SavedModel::new(model_name, sub_graph));
                }
            }
        }
        self.backend
            .sync_available_saved_models(&self.container, available)
    }

    fn layout_update_callback(
        model_config_path: PathBuf,
        layout_filters: Vec<LayoutFilter>,
        saved_models: Vec<(SavedModel, SavedModelDeployConfig)>,
    ) -> bool {
        let mut cfg_list: Vec<tfserving_apis::ModelConfig> = Vec::new();
        for (saved_model, deploy) in saved_models {
            let accepted = if layout_filters.is_empty() {
                true
            } else {
                layout_filters
                    .iter()
                    .any(|f| f.accepts(&saved_model.sub_graph))
            };
            if !accepted {
                continue;
            }

            let base_path = deploy.model_base_path.unwrap_or_default();
            let version_policy = deploy
                .version_policy
                .unwrap_or_else(|| "latest".to_string());
            cfg_list.push(gen_model_config_compat(
                &saved_model.to_string(),
                &base_path,
                &version_policy,
            ));
        }

        let cfg = tfserving_apis::ModelServerConfig {
            config: Some(
                tfserving_apis::model_server_config::Config::ModelConfigList(
                    tfserving_apis::ModelConfigList { config: cfg_list },
                ),
            ),
        };

        let pbtxt = match model_server_config_pbtxt(&cfg) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("failed to format ModelServerConfig pbtxt: {e}");
                return false;
            }
        };

        if let Err(e) = std::fs::write(&model_config_path, pbtxt) {
            tracing::error!("failed to write model_config pbtxt: {e}");
            return false;
        }
        true
    }

    /// Start the agent: connect backend/wrapper, serve discovery, and register callbacks/tasks.
    pub async fn start(&self) -> ServingResult<()> {
        self.tfs_wrapper.start()?;
        self.backend
            .start()
            .map_err(|e| ServingError::server(format!("backend start failed: {e}")))?;

        // Start discovery service (AgentService) for v3 using an address provider.
        // This mirrors Python's `AgentService(AgentDataProvider(addrs_fn=_gen_addrs_map))`.
        let provider = AgentDataProvider::new({
            let this = self.clone_for_provider();
            move || this.gen_addrs_map()
        });
        let svc = AgentServiceDiscoveryImpl::from_provider(provider, self.config.clone());

        let addr: SocketAddr = format!("127.0.0.1:{}", self.config.agent_port)
            .parse()
            .map_err(|e| ServingError::ConfigError(format!("invalid agent listen addr: {e}")))?;
        let server = AgentDiscoveryServer::serve(addr, svc).await?;
        *self.agent_server.lock() = Some(server);

        // Start periodic tasks.
        let mut rx = self.stop.subscribe();
        let backend = self.backend.clone();
        let container = self.container.clone();
        let service_info = self.service_info.clone();
        let t1 = tokio::spawn(async move {
            // First call is immediate (Python parity).
            let _ = backend.report_service_info(&container, &service_info);
            loop {
                if *rx.borrow() {
                    break;
                }
                sleep(Duration::from_secs(60)).await;
                if *rx.borrow() {
                    break;
                }
                let _ = backend.report_service_info(&container, &service_info);
            }
        });
        self.bg_tasks.lock().push(t1);

        let mut rx2 = self.stop.subscribe();
        let this = self.clone_for_provider();
        let t2 = tokio::spawn(async move {
            // Immediate first run.
            let _ = this.sync_available_saved_models().await;
            loop {
                if *rx2.borrow() {
                    break;
                }
                sleep(Duration::from_secs(30)).await;
                if *rx2.borrow() {
                    break;
                }
                let _ = this.sync_available_saved_models().await;
            }
        });
        self.bg_tasks.lock().push(t2);

        // Layout callback that writes the model_config file.
        let layout_path = self.config.layout_path().ok_or_else(|| {
            ServingError::ConfigError("layout_pattern is required for agent v3".to_string())
        })?;
        let model_config_path = self.model_config_path.clone();
        let filters = self.layout_filters.clone();
        self.backend
            .register_layout_callback(&layout_path, move |saved_models| {
                Self::layout_update_callback(
                    model_config_path.clone(),
                    filters.clone(),
                    saved_models,
                )
            })
            .map_err(|e| ServingError::server(format!("register_layout_callback failed: {e}")))?;

        Ok(())
    }

    /// Stop the agent and all background tasks.
    pub async fn stop(&self) {
        let _ = self.stop.send(true);

        if let Some(server) = self.agent_server.lock().take() {
            server.shutdown();
        }
        for t in self.bg_tasks.lock().drain(..) {
            t.abort();
        }
        let _ = self.backend.stop();
        let _ = self.tfs_wrapper.stop();
    }

    fn clone_for_provider(&self) -> AgentV3Handle {
        AgentV3Handle {
            tfs_wrapper: Arc::clone(&self.tfs_wrapper),
            backend: self.backend.clone(),
            container: self.container.clone(),
        }
    }
}

#[derive(Clone)]
struct AgentV3Handle {
    tfs_wrapper: Arc<dyn TfsWrapperApi>,
    backend: ZkBackend,
    container: Container,
}

impl AgentV3Handle {
    fn gen_addrs_map(&self) -> HashMap<String, Vec<String>> {
        let service_map = self.backend.get_service_map();
        let mut addrs_map: HashMap<String, Vec<String>> = HashMap::new();
        for (model_name, sub_map) in service_map {
            for (sub_graph, infos) in sub_map {
                let key = format!("{model_name}:{sub_graph}");
                let addrs = infos
                    .into_iter()
                    .filter_map(|si| {
                        if self.tfs_wrapper.is_grpc_remote_op() {
                            si.grpc
                        } else {
                            si.archon
                        }
                    })
                    .collect::<Vec<_>>();
                addrs_map.insert(key, addrs);
            }
        }
        addrs_map
    }

    async fn sync_available_saved_models(&self) -> Result<(), ZkError> {
        let status = self
            .tfs_wrapper
            .list_saved_models_status()
            .await
            .map_err(|e| ZkError::Other(format!("list_saved_models_status failed: {e}")))?;
        let mut available: HashSet<SavedModel> = HashSet::new();
        for (saved_model_name, st) in status {
            if st.state == tfserving_apis::model_version_status::State::Available as i32 {
                let mut it = saved_model_name.split(':');
                let model_name = it.next().unwrap_or_default();
                let sub_graph = it.next().unwrap_or_default();
                if !model_name.is_empty() && !sub_graph.is_empty() {
                    available.insert(SavedModel::new(model_name, sub_graph));
                }
            }
        }
        self.backend
            .sync_available_saved_models(&self.container, available)
    }
}
