//! In-process fake TFServing gRPC server (Python parity).
//!
//! This is a Rust port of `monolith/agent_service/mocked_tfserving.py` used by
//! parity tests for `TFSMonitor`.
//!
//! The fake implements:
//! - ModelService/GetModelStatus
//! - ModelService/HandleReloadConfigRequest
//! - PredictionService/GetModelMetadata
//!
//! The internal "model manager" simulates asynchronous state progression:
//! UNKNOWN -> START -> LOADING -> AVAILABLE for new models/versions, and
//! UNLOADING -> END for removals.

#![cfg(feature = "grpc")]

use crate::error::ServingResult;
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use monolith_proto::tensorflow_serving::error::Code as TfServingCode;
use prost_types::Any;
use std::collections::{HashMap, HashSet, VecDeque};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::sleep;
use tonic::{Request, Response, Status};

type ModelName = String;

#[derive(Debug, Clone)]
struct ModelConf {
    model_name: ModelName,
    base_path: String,
    version_policy: String, // "latest" | "specific"
    version_data: Vec<i64>, // for latest: [num_versions]; for specific: list of versions
    model_platform: String,
    signature_name: Vec<String>,
}

#[derive(Debug, Clone)]
struct ModelVersion {
    version: i64,
    version_label: Option<String>,
    state: tfserving_apis::model_version_status::State,
}

#[derive(Debug, Clone)]
struct ModelMeta {
    conf: ModelConf,
    versions: Vec<ModelVersion>,
    unloading: bool,
}

impl ModelMeta {
    fn is_unloading(&self) -> bool {
        self.unloading
    }
}

#[derive(Debug, Clone)]
struct Event {
    model_name: ModelName,
    version: i64,
    state: tfserving_apis::model_version_status::State,
}

fn ok_status() -> tfserving_apis::StatusProto {
    tfserving_apis::StatusProto {
        error_code: TfServingCode::Ok as i32,
        error_message: "".to_string(),
    }
}

fn not_found_status(msg: String) -> tfserving_apis::StatusProto {
    tfserving_apis::StatusProto {
        error_code: TfServingCode::NotFound as i32,
        error_message: msg,
    }
}

fn gen_model_version_status(
    version: i64,
    state: tfserving_apis::model_version_status::State,
    status: tfserving_apis::StatusProto,
) -> tfserving_apis::ModelVersionStatus {
    tfserving_apis::ModelVersionStatus {
        version,
        state: state as i32,
        status: Some(status),
    }
}

struct ModelMgrInner {
    models: HashMap<ModelName, ModelMeta>,
    queue: VecDeque<Event>,
    last_auto_add: Instant,
}

impl Default for ModelMgrInner {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            queue: VecDeque::new(),
            last_auto_add: Instant::now(),
        }
    }
}

/// Simulated model manager similar to Python `ModelMgr`.
#[derive(Clone, Default)]
struct ModelMgr {
    inner: Arc<Mutex<ModelMgrInner>>,
    notify: Arc<Notify>,
    stopped: Arc<Notify>,
    stop_flag: Arc<Mutex<bool>>,
}

impl ModelMgr {
    async fn load_from_config_list(&self, configs: &[tfserving_apis::ModelConfig]) {
        let mut inner = self.inner.lock().await;
        for config in configs {
            let (version_policy, versions, version_data_vec) = match config
                .model_version_policy
                .as_ref()
                .and_then(|p| p.policy_choice.as_ref())
            {
                Some(tfserving_apis::file_system_storage_path_source_config::servable_version_policy::PolicyChoice::Latest(latest)) => {
                    let n = latest.num_versions.max(1) as i64;
                    (
                        "latest".to_string(),
                        (1..=n).map(|v| ModelVersion { version: v, version_label: None, state: tfserving_apis::model_version_status::State::Unknown }).collect(),
                        vec![n],
                    )
                }
                Some(tfserving_apis::file_system_storage_path_source_config::servable_version_policy::PolicyChoice::Specific(spec)) => {
                    let mut vs = spec.versions.clone();
                    vs.sort();
                    (
                        "specific".to_string(),
                        vs.iter().map(|&v| ModelVersion { version: v, version_label: None, state: tfserving_apis::model_version_status::State::Unknown }).collect(),
                        vs,
                    )
                }
                Some(tfserving_apis::file_system_storage_path_source_config::servable_version_policy::PolicyChoice::All(_)) | None => {
                    (
                        "latest".to_string(),
                        vec![ModelVersion { version: 1, version_label: None, state: tfserving_apis::model_version_status::State::Unknown }],
                        vec![],
                    )
                }
            };

            let conf = ModelConf {
                model_name: config.name.clone(),
                base_path: config.base_path.clone(),
                version_policy,
                version_data: version_data_vec,
                model_platform: config.model_platform.clone(),
                signature_name: vec!["update".to_string(), "predict".to_string()],
            };
            let versions = versions;
            inner.models.insert(
                config.name.clone(),
                ModelMeta {
                    conf,
                    versions: versions.clone(),
                    unloading: false,
                },
            );
            for v in versions {
                inner.queue.push_back(Event {
                    model_name: config.name.clone(),
                    version: v.version,
                    state: tfserving_apis::model_version_status::State::Start,
                });
            }
        }
        self.notify.notify_one();
    }

    async fn remove(&self, model_names: HashSet<ModelName>) {
        let mut inner = self.inner.lock().await;
        for name in model_names {
            if let Some(model) = inner.models.get_mut(&name) {
                model.unloading = true;
                let versions: Vec<i64> = model.versions.iter().map(|v| v.version).collect();
                for version in versions {
                    inner.queue.push_back(Event {
                        model_name: name.clone(),
                        version,
                        state: tfserving_apis::model_version_status::State::Unloading,
                    });
                }
            }
        }
        self.notify.notify_one();
    }

    async fn alive_model_names(&self) -> HashSet<ModelName> {
        let inner = self.inner.lock().await;
        inner
            .models
            .iter()
            .filter_map(|(k, v)| (!v.is_unloading()).then_some(k.clone()))
            .collect()
    }

    async fn get_status(
        &self,
        model_spec: &tfserving_apis::ModelSpec,
    ) -> Vec<tfserving_apis::ModelVersionStatus> {
        let inner = self.inner.lock().await;
        let mut out = Vec::new();
        if let Some(model) = inner.models.get(&model_spec.name) {
            match model_spec.version_choice.as_ref() {
                None => {
                    for v in &model.versions {
                        out.push(gen_model_version_status(v.version, v.state, ok_status()));
                    }
                }
                Some(tfserving_apis::model_spec::VersionChoice::Version(v)) => {
                    for mv in &model.versions {
                        if mv.version == *v {
                            out.push(gen_model_version_status(mv.version, mv.state, ok_status()));
                            break;
                        }
                    }
                }
                Some(tfserving_apis::model_spec::VersionChoice::VersionLabel(label)) => {
                    for mv in &model.versions {
                        if mv.version_label.as_deref() == Some(label.as_str()) {
                            out.push(gen_model_version_status(mv.version, mv.state, ok_status()));
                            break;
                        }
                    }
                }
            }
        }

        if out.is_empty() {
            out.push(gen_model_version_status(
                -1,
                tfserving_apis::model_version_status::State::Unknown,
                not_found_status(format!("{} is not found", model_spec.name)),
            ));
        }
        out
    }

    async fn get_metadata(
        &self,
        model_spec: &tfserving_apis::ModelSpec,
        metadata_fields: &HashSet<String>,
    ) -> HashMap<String, Any> {
        let inner = self.inner.lock().await;
        let mut out: HashMap<String, Any> = HashMap::new();
        let Some(model) = inner.models.get(&model_spec.name) else {
            return out;
        };

        // Python returns bytes(repr(v)); do the same for basic fields.
        for field in metadata_fields {
            let value = match field.as_str() {
                "base_path" => Some(model.conf.base_path.clone()),
                "num_versions" => {
                    if model.conf.version_policy == "latest" {
                        Some(
                            model
                                .conf
                                .version_data
                                .first()
                                .copied()
                                .unwrap_or(1)
                                .to_string(),
                        )
                    } else {
                        Some(model.versions.len().to_string())
                    }
                }
                "signature_name" => Some(format!("{:?}", model.conf.signature_name)),
                _ => None,
            };
            if let Some(v) = value {
                out.insert(
                    field.clone(),
                    Any {
                        type_url: "".to_string(),
                        value: format!("{v:?}").into_bytes(),
                    },
                );
            }
        }

        // Version-specific fields are ignored by parity tests; Python checks version only if set.
        out
    }

    async fn run(self) {
        let tick = Duration::from_millis(10);
        loop {
            if *self.stop_flag.lock().await {
                self.stopped.notify_one();
                return;
            }

            let mut handled = false;
            {
                let mut inner = self.inner.lock().await;
                if let Some(ev) = inner.queue.pop_front() {
                    handled = true;
                    Self::event_handler_locked(&mut inner, ev);
                } else {
                    // Auto-add a new version for non-specific models every ~30s, like Python.
                    let now = Instant::now();
                    if now.duration_since(inner.last_auto_add) > Duration::from_secs(30) {
                        inner.last_auto_add = now;
                        // Avoid borrowing `inner` twice by copying the first key and then looking up.
                        let first_name = inner.models.keys().next().cloned();
                        if let Some(model_name) = first_name {
                            if let Some(meta) = inner.models.get_mut(&model_name) {
                                if meta.conf.version_policy != "specific" && !meta.unloading {
                                    let next_ver =
                                        meta.versions.last().map(|v| v.version + 1).unwrap_or(1);
                                    meta.versions.push(ModelVersion {
                                        version: next_ver,
                                        version_label: None,
                                        state: tfserving_apis::model_version_status::State::Unknown,
                                    });
                                    inner.queue.push_back(Event {
                                        model_name,
                                        version: next_ver,
                                        state: tfserving_apis::model_version_status::State::Start,
                                    });
                                }
                            }
                        }
                    }
                }
            }

            if !handled {
                // Wait for new work but also tick periodically.
                tokio::select! {
                    _ = self.notify.notified() => {},
                    _ = sleep(tick) => {},
                }
            }
        }
    }

    fn event_handler_locked(inner: &mut ModelMgrInner, ev: Event) {
        let Some(model) = inner.models.get_mut(&ev.model_name) else {
            return;
        };
        use tfserving_apis::model_version_status::State as S;
        match ev.state {
            S::Start => {
                if let Some(v) = model.versions.iter_mut().find(|v| v.version == ev.version) {
                    if v.state == S::Unknown {
                        v.state = S::Start;
                        inner.queue.push_back(Event {
                            state: S::Loading,
                            ..ev
                        });
                    }
                }
            }
            S::Loading => {
                if let Some(v) = model.versions.iter_mut().find(|v| v.version == ev.version) {
                    if v.state == S::Start {
                        v.state = S::Loading;
                        inner.queue.push_back(Event {
                            state: S::Available,
                            ..ev
                        });
                    }
                }
            }
            S::Available => {
                if let Some(v) = model.versions.iter_mut().find(|v| v.version == ev.version) {
                    if v.state == S::Loading {
                        v.state = S::Available;
                        if model.conf.version_policy == "latest" {
                            // Keep at most N latest versions.
                            let keep =
                                model.conf.version_data.first().copied().unwrap_or(1) as usize;
                            while model.versions.len() > keep {
                                let unload_ver = model.versions[0].version;
                                inner.queue.push_back(Event {
                                    model_name: ev.model_name.clone(),
                                    version: unload_ver,
                                    state: S::Unloading,
                                });
                                break;
                            }
                        }
                    }
                }
            }
            S::Unloading => {
                if let Some(v) = model.versions.iter_mut().find(|v| v.version == ev.version) {
                    if v.state != S::Unloading && v.state != S::End {
                        v.state = S::Unloading;
                        inner.queue.push_back(Event {
                            state: S::End,
                            ..ev
                        });
                    }
                }
            }
            S::End => {
                if let Some(idx) = model.versions.iter().position(|v| v.version == ev.version) {
                    if model.versions[idx].state == S::Unloading {
                        model.versions[idx].state = S::End;
                    }
                    model.versions.remove(idx);
                }
                if model.versions.is_empty() {
                    inner.models.remove(&ev.model_name);
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
struct ModelServiceImpl {
    mgr: ModelMgr,
}

#[tonic::async_trait]
impl tfserving_apis::model_service_server::ModelService for ModelServiceImpl {
    async fn get_model_status(
        &self,
        request: Request<tfserving_apis::GetModelStatusRequest>,
    ) -> Result<Response<tfserving_apis::GetModelStatusResponse>, Status> {
        let req = request.into_inner();
        let spec = req
            .model_spec
            .ok_or_else(|| Status::invalid_argument("model_spec is required"))?;
        let statuses = self.mgr.get_status(&spec).await;
        Ok(Response::new(tfserving_apis::GetModelStatusResponse {
            model_version_status: statuses,
        }))
    }

    async fn handle_reload_config_request(
        &self,
        request: Request<tfserving_apis::ReloadConfigRequest>,
    ) -> Result<Response<tfserving_apis::ReloadConfigResponse>, Status> {
        let req = request.into_inner();
        let cfg = req
            .config
            .ok_or_else(|| Status::invalid_argument("config is required"))?;
        let list = match cfg.config {
            Some(tfserving_apis::model_server_config::Config::ModelConfigList(list)) => list.config,
            _ => Vec::new(),
        };

        let old = self.mgr.alive_model_names().await;
        let new: HashSet<ModelName> = list.iter().map(|c| c.name.clone()).collect();

        let to_remove: HashSet<ModelName> = old.difference(&new).cloned().collect();
        self.mgr.remove(to_remove).await;

        let to_load: Vec<tfserving_apis::ModelConfig> = list
            .into_iter()
            .filter(|c| !old.contains(&c.name))
            .collect();
        self.mgr.load_from_config_list(&to_load).await;

        Ok(Response::new(tfserving_apis::ReloadConfigResponse {
            status: Some(ok_status()),
            metric: Vec::new(),
        }))
    }
}

#[derive(Clone)]
struct PredictionServiceImpl {
    mgr: ModelMgr,
}

#[tonic::async_trait]
impl tfserving_apis::prediction_service_server::PredictionService for PredictionServiceImpl {
    async fn predict(
        &self,
        _request: Request<tfserving_apis::PredictRequest>,
    ) -> Result<Response<tfserving_apis::PredictResponse>, Status> {
        Err(Status::unimplemented(
            "Predict is not implemented in FakeTFServing",
        ))
    }

    async fn classify(
        &self,
        _request: Request<tfserving_apis::ClassificationRequest>,
    ) -> Result<Response<tfserving_apis::ClassificationResponse>, Status> {
        Err(Status::unimplemented("classify is not implemented"))
    }

    async fn regress(
        &self,
        _request: Request<tfserving_apis::RegressionRequest>,
    ) -> Result<Response<tfserving_apis::RegressionResponse>, Status> {
        Err(Status::unimplemented("regress is not implemented"))
    }

    async fn multi_inference(
        &self,
        _request: Request<tfserving_apis::MultiInferenceRequest>,
    ) -> Result<Response<tfserving_apis::MultiInferenceResponse>, Status> {
        Err(Status::unimplemented("multi_inference is not implemented"))
    }

    async fn get_model_metadata(
        &self,
        request: Request<tfserving_apis::GetModelMetadataRequest>,
    ) -> Result<Response<tfserving_apis::GetModelMetadataResponse>, Status> {
        let req = request.into_inner();
        let spec = req
            .model_spec
            .ok_or_else(|| Status::invalid_argument("model_spec is required"))?;
        let fields: HashSet<String> = req.metadata_field.into_iter().collect();
        let meta = self.mgr.get_metadata(&spec, &fields).await;
        Ok(Response::new(tfserving_apis::GetModelMetadataResponse {
            model_spec: Some(spec),
            metadata: meta,
        }))
    }
}

/// Test-only fake TFServing gRPC server.
pub struct FakeTfServing {
    addr: SocketAddr,
    mgr: ModelMgr,
    server_handle: Option<JoinHandle<()>>,
    mgr_handle: Option<JoinHandle<()>>,
}

impl FakeTfServing {
    /// Create a fake TF Serving instance bound to `addr` (not started yet).
    pub fn new(addr: SocketAddr) -> Self {
        Self {
            addr,
            mgr: ModelMgr::default(),
            server_handle: None,
            mgr_handle: None,
        }
    }

    /// Build a `ModelConfig` using a "latest N versions" policy.
    pub fn default_model_config(
        model_name: &str,
        base_path: &str,
        num_versions: i64,
    ) -> tfserving_apis::ModelConfig {
        let initial = tfserving_apis::ModelConfig {
            name: model_name.to_string(),
            base_path: base_path.to_string(),
            model_type: 0,
            model_platform: "tensorflow".to_string(),
            model_version_policy: Some(tfserving_apis::file_system_storage_path_source_config::ServableVersionPolicy {
                policy_choice: Some(tfserving_apis::file_system_storage_path_source_config::servable_version_policy::PolicyChoice::Latest(
                    tfserving_apis::file_system_storage_path_source_config::servable_version_policy::Latest { num_versions: num_versions.max(1) as u32 }
                )),
            }),
            version_labels: HashMap::new(),
            logging_config: None,
        };
        initial
    }

    /// Start the fake server and seed it with the provided model configs.
    pub async fn start_with_configs(
        &mut self,
        configs: Vec<tfserving_apis::ModelConfig>,
    ) -> ServingResult<()> {
        // Seed initial models.
        self.mgr.load_from_config_list(&configs).await;

        let mgr = self.mgr.clone();
        let mgr_handle = tokio::spawn(mgr.run());
        self.mgr_handle = Some(mgr_handle);

        let model_svc = ModelServiceImpl {
            mgr: self.mgr.clone(),
        };
        let pred_svc = PredictionServiceImpl {
            mgr: self.mgr.clone(),
        };

        let addr = self.addr;
        let handle = tokio::spawn(async move {
            let _ = tonic::transport::Server::builder()
                .add_service(
                    tfserving_apis::model_service_server::ModelServiceServer::new(model_svc),
                )
                .add_service(
                    tfserving_apis::prediction_service_server::PredictionServiceServer::new(
                        pred_svc,
                    ),
                )
                .serve(addr)
                .await;
        });
        self.server_handle = Some(handle);
        // Give the server a moment to bind.
        sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    /// Stop the fake server and background tasks.
    pub async fn stop(&mut self) {
        // Stop the model mgr loop.
        {
            let mut flag = self.mgr.stop_flag.lock().await;
            *flag = true;
        }
        self.mgr.notify.notify_one();
        let _ = self.mgr.stopped.notified().await;

        if let Some(h) = self.server_handle.take() {
            h.abort();
        }
        if let Some(h) = self.mgr_handle.take() {
            h.abort();
        }
    }

    /// Return the socket address the server is configured to listen on.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

/// Pick an ephemeral local port.
pub async fn find_free_port() -> u16 {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

/// Blocking helper for tests that are not async.
pub fn find_free_port_blocking() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .expect("bind")
        .local_addr()
        .expect("local_addr")
        .port()
}
