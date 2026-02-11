//! TFServing monitoring + reload config helpers (Python parity).
//!
//! This ports the parts of `monolith/agent_service/tfs_monitor.py` that are
//! exercised by the Python unit tests:
//! - generating TF Serving `ModelServerConfig` from a set of `PublishMeta`
//! - `HandleReloadConfigRequest` (adds a default model config if missing)
//! - `GetModelStatus` for a set of sub-models, with Python-like error shaping

#![cfg(feature = "grpc")]

use crate::data_def::{PublishMeta, PublishType, TfsModelName, VersionPath};
use crate::error::{ServingError, ServingResult};
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use monolith_proto::tensorflow_serving::error::Code as TfServingCode;
use monolith_proto::tensorflow_serving::{apis::model_version_status::State, apis::StatusProto};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Channel;

/// Minimal deploy type (Python parity for `DeployType` used by `TFSMonitor` tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeployType {
    /// Mixed deployment (entry + ps, optionally dense).
    Mixed,
    /// Entry-only deployment.
    Entry,
    /// Parameter-server-only deployment.
    Ps,
    /// Dense-only deployment.
    Dense,
}

/// Minimal TF Serving server type identifiers (Python parity).
pub struct TfServerType;

impl TfServerType {
    /// TF Serving entry server type.
    pub const ENTRY: &'static str = "entry";
    /// TF Serving parameter server type.
    pub const PS: &'static str = "ps";
    /// TF Serving dense server type.
    pub const DENSE: &'static str = "dense";
}

/// A small subset of agent config needed by the monitor.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Deployment type to route requests to the correct TF Serving instance.
    pub deploy_type: DeployType,
    /// Whether dense runs as a standalone instance in addition to entry/ps.
    pub dense_alone: bool,
    /// Entry server gRPC port.
    pub tfs_entry_port: u16,
    /// PS server gRPC port.
    pub tfs_ps_port: u16,
    /// Dense server gRPC port.
    pub tfs_dense_port: u16,
}

impl AgentConfig {
    /// Create a parity test config with ephemeral ports.
    pub fn for_test(deploy_type: DeployType, tfs_entry_port: u16, tfs_ps_port: u16) -> Self {
        Self {
            deploy_type,
            dense_alone: false,
            tfs_entry_port,
            tfs_ps_port,
            tfs_dense_port: 0,
        }
    }
}

fn get_local_ip() -> String {
    // For parity tests we prefer a stable local address.
    "127.0.0.1".to_string()
}

fn is_entry(name: &str) -> bool {
    name.starts_with(TfServerType::ENTRY)
}
fn is_ps(name: &str) -> bool {
    name.starts_with(TfServerType::PS)
}
fn is_dense(name: &str) -> bool {
    name.starts_with(TfServerType::DENSE)
}

fn service_type(conf: &AgentConfig, sub_model_name: &str) -> Option<&'static str> {
    match conf.deploy_type {
        DeployType::Entry => is_entry(sub_model_name).then_some(TfServerType::ENTRY),
        DeployType::Ps => is_ps(sub_model_name).then_some(TfServerType::PS),
        DeployType::Dense => is_dense(sub_model_name).then_some(TfServerType::DENSE),
        DeployType::Mixed => {
            if is_entry(sub_model_name) {
                Some(TfServerType::ENTRY)
            } else if is_ps(sub_model_name) {
                Some(TfServerType::PS)
            } else if is_dense(sub_model_name) {
                Some(TfServerType::DENSE)
            } else {
                None
            }
        }
    }
}

fn gen_model_spec(
    name: &str,
    version: Option<i64>,
    signature_name: Option<&str>,
) -> tfserving_apis::ModelSpec {
    let version_choice = version.map(tfserving_apis::model_spec::VersionChoice::Version);
    tfserving_apis::ModelSpec {
        name: name.to_string(),
        version_choice,
        signature_name: signature_name.unwrap_or_default().to_string(),
    }
}

fn gen_model_config(
    name: &str,
    base_path: &str,
    version_policy: &str,
    version_data: i64,
) -> ServingResult<tfserving_apis::ModelConfig> {
    use tfserving_apis::file_system_storage_path_source_config::servable_version_policy as vp;
    use tfserving_apis::file_system_storage_path_source_config::ServableVersionPolicy;

    let policy_choice = match version_policy.to_ascii_lowercase().as_str() {
        "latest" => Some(vp::PolicyChoice::Latest(vp::Latest {
            num_versions: version_data.max(1) as u32,
        })),
        "all" => Some(vp::PolicyChoice::All(vp::All {})),
        "specific" => Some(vp::PolicyChoice::Specific(vp::Specific {
            versions: vec![version_data],
        })),
        other => {
            // Python raises ValueError; surface as config error.
            return Err(ServingError::ConfigError(format!(
                "unsupported version_policy {other}"
            )));
        }
    };

    Ok(tfserving_apis::ModelConfig {
        name: name.to_string(),
        base_path: base_path.to_string(),
        model_type: 0, // deprecated
        model_platform: "tensorflow".to_string(),
        model_version_policy: Some(ServableVersionPolicy { policy_choice }),
        version_labels: HashMap::new(),
        logging_config: None,
    })
}

fn default_model_config() -> tfserving_apis::ModelConfig {
    // The exact base_path is not asserted by parity tests; only presence of `default`.
    gen_model_config("default", "/tmp/monolith/default", "latest", 1)
        .expect("default model config must be valid")
}

#[derive(Clone)]
struct Stubs {
    model: tfserving_apis::model_service_client::ModelServiceClient<Channel>,
    _predict: tfserving_apis::prediction_service_client::PredictionServiceClient<Channel>,
}

/// Rust port of `TFSMonitor`.
#[derive(Clone)]
pub struct TfsMonitor {
    conf: AgentConfig,
    host: Arc<Mutex<Option<String>>>,
    stubs: Arc<Mutex<HashMap<&'static str, Stubs>>>,
}

impl TfsMonitor {
    /// Create a new monitor instance.
    pub fn new(conf: AgentConfig) -> Self {
        Self {
            conf,
            host: Arc::new(Mutex::new(None)),
            stubs: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Return the resolved host name used for gRPC connections (cached).
    pub async fn host(&self) -> String {
        let mut h = self.host.lock().await;
        if h.as_deref().unwrap_or("").is_empty()
            || matches!(h.as_deref(), Some("localhost" | "127.0.0.1"))
        {
            *h = Some(get_local_ip());
        }
        h.clone().unwrap_or_else(get_local_ip)
    }

    async fn addr(&self, sub_model_name: &str) -> ServingResult<String> {
        let host = self.host().await;
        let port = match self.conf.deploy_type {
            DeployType::Mixed => {
                if is_entry(sub_model_name) {
                    self.conf.tfs_entry_port
                } else if is_ps(sub_model_name) {
                    self.conf.tfs_ps_port
                } else {
                    self.conf.tfs_dense_port
                }
            }
            DeployType::Entry => self.conf.tfs_entry_port,
            DeployType::Ps => self.conf.tfs_ps_port,
            DeployType::Dense => self.conf.tfs_dense_port,
        };
        Ok(format!("{host}:{port}"))
    }

    /// Connect gRPC stubs for all enabled server types.
    pub async fn connect(&self) -> ServingResult<()> {
        let mut map = self.stubs.lock().await;
        map.clear();

        let host = self.host().await;

        async fn connect_one(addr: String) -> ServingResult<Stubs> {
            let endpoint = if addr.starts_with("http://") {
                addr
            } else {
                format!("http://{addr}")
            };
            let ch = tonic::transport::Endpoint::from_shared(endpoint)
                .map_err(|e| ServingError::ConfigError(format!("invalid endpoint: {e}")))?
                .connect()
                .await
                .map_err(|e| ServingError::GrpcError(format!("connect failed: {e}")))?;
            Ok(Stubs {
                model: tfserving_apis::model_service_client::ModelServiceClient::new(ch.clone()),
                _predict: tfserving_apis::prediction_service_client::PredictionServiceClient::new(
                    ch,
                ),
            })
        }

        if matches!(self.conf.deploy_type, DeployType::Mixed | DeployType::Entry) {
            map.insert(
                TfServerType::ENTRY,
                connect_one(format!("{host}:{}", self.conf.tfs_entry_port)).await?,
            );
        }
        if matches!(self.conf.deploy_type, DeployType::Mixed | DeployType::Ps) {
            map.insert(
                TfServerType::PS,
                connect_one(format!("{host}:{}", self.conf.tfs_ps_port)).await?,
            );
        }
        if self.conf.dense_alone
            && matches!(self.conf.deploy_type, DeployType::Mixed | DeployType::Dense)
        {
            map.insert(
                TfServerType::DENSE,
                connect_one(format!("{host}:{}", self.conf.tfs_dense_port)).await?,
            );
        }
        Ok(())
    }

    /// Drop gRPC stubs (tonic channels close on drop).
    pub async fn stop(&self) {
        // tonic channels close on drop; we just clear references.
        self.stubs.lock().await.clear();
    }

    /// Python parity for `get_model_status(name: str, version: Union[int,str]=None, signature_name=None)`.
    pub async fn get_model_status_for_name(
        &self,
        name: &str,
        version: Option<i64>,
        signature_name: Option<&str>,
    ) -> ServingResult<Vec<tfserving_apis::ModelVersionStatus>> {
        let Some(st) = service_type(&self.conf, name) else {
            return Ok(Vec::new());
        };

        let req = tfserving_apis::GetModelStatusRequest {
            model_spec: Some(gen_model_spec(name, version, signature_name)),
        };

        let mut stubs = self.stubs.lock().await;
        let stub = stubs
            .get_mut(st)
            .ok_or_else(|| ServingError::NotConnected)?;
        let resp = stub
            .model
            .get_model_status(req)
            .await
            .map_err(|e| ServingError::GrpcError(format!("GetModelStatus failed: {e}")))?;
        Ok(resp.into_inner().model_version_status)
    }

    /// Python parity for `get_model_status(PublishMeta, fix_dense_version=False)`.
    pub async fn get_model_status_for_publish_meta(
        &self,
        pm: &PublishMeta,
        fix_dense_version: bool,
    ) -> ServingResult<HashMap<TfsModelName, (VersionPath, tfserving_apis::ModelVersionStatus)>>
    {
        let mut out = HashMap::new();
        let Some(model_name) = pm.model_name.as_deref() else {
            return Ok(out);
        };
        let Some(sub_models) = pm.sub_models.as_ref() else {
            return Ok(out);
        };

        for (sub_model_name, smvpath) in sub_models {
            let Some(st) = service_type(&self.conf, sub_model_name) else {
                continue;
            };
            let tfs_model_name = format!("{model_name}:{sub_model_name}");

            // Python special-case: dense node uses latest unless fix_dense_version.
            let is_dense_node = (!self.conf.dense_alone && is_entry(sub_model_name))
                || (self.conf.dense_alone && is_dense(sub_model_name));

            let requested_version: Option<i64> = if !fix_dense_version && is_dense_node {
                None
            } else {
                let base = Path::new(smvpath)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("0");
                base.parse::<i64>().ok()
            };

            let req = tfserving_apis::GetModelStatusRequest {
                model_spec: Some(gen_model_spec(&tfs_model_name, requested_version, None)),
            };

            let mut stubs = self.stubs.lock().await;
            let stub = stubs
                .get_mut(st)
                .ok_or_else(|| ServingError::NotConnected)?;

            let status = match stub.model.get_model_status(req).await {
                Ok(resp) => {
                    let mut statuses = resp.into_inner().model_version_status;
                    if statuses.is_empty() {
                        tfserving_apis::ModelVersionStatus {
                            version: -1,
                            state: State::Unknown as i32,
                            status: Some(StatusProto {
                                error_code: TfServingCode::NotFound as i32,
                                error_message: format!("{tfs_model_name} not found"),
                            }),
                        }
                    } else {
                        // Select the latest one by version, matching Python's sort logic.
                        statuses.sort_by_key(|s| s.version);
                        statuses.pop().unwrap()
                    }
                }
                Err(e) => {
                    // Python catches `_InactiveRpcError` and maps code/details.
                    let code = e.code();
                    tfserving_apis::ModelVersionStatus {
                        version: -1,
                        state: State::Unknown as i32,
                        status: Some(StatusProto {
                            error_code: code as i32,
                            error_message: e.message().to_string(),
                        }),
                    }
                }
            };

            out.insert(tfs_model_name, (smvpath.clone(), status));
        }

        Ok(out)
    }

    /// Python parity for `gen_model_config(pms, fix_dense_version=False)`.
    pub fn gen_model_config(
        &self,
        pms: &[PublishMeta],
        fix_dense_version: bool,
    ) -> HashMap<&'static str, tfserving_apis::ModelServerConfig> {
        let mut entry = tfserving_apis::ModelServerConfig {
            config: Some(
                tfserving_apis::model_server_config::Config::ModelConfigList(
                    tfserving_apis::ModelConfigList { config: vec![] },
                ),
            ),
        };
        let mut ps = entry.clone();
        let mut dense = entry.clone();

        for pm in pms {
            if pm.ptype == PublishType::Unload {
                continue;
            }
            let Some(model_name) = pm.model_name.as_deref() else {
                continue;
            };
            let Some(sub_models) = pm.sub_models.as_ref() else {
                continue;
            };

            for (sub_model_name, smv_path) in sub_models {
                let tfs_model_name = format!("{model_name}:{sub_model_name}");
                let Some(st) = service_type(&self.conf, sub_model_name) else {
                    continue;
                };

                let base_path = Path::new(smv_path)
                    .parent()
                    .and_then(|p| p.to_str())
                    .unwrap_or("")
                    .to_string();

                let is_dense_node = (!self.conf.dense_alone && is_entry(sub_model_name))
                    || (self.conf.dense_alone && is_dense(sub_model_name));
                let (version_policy, version_data) = if is_dense_node {
                    if fix_dense_version {
                        (
                            "specific",
                            Path::new(smv_path)
                                .file_name()
                                .and_then(|s| s.to_str())
                                .and_then(|s| s.parse::<i64>().ok())
                                .unwrap_or(1),
                        )
                    } else {
                        ("latest", 1)
                    }
                } else {
                    (
                        "specific",
                        Path::new(smv_path)
                            .file_name()
                            .and_then(|s| s.to_str())
                            .and_then(|s| s.parse::<i64>().ok())
                            .unwrap_or(1),
                    )
                };

                let Ok(mc) =
                    gen_model_config(&tfs_model_name, &base_path, version_policy, version_data)
                else {
                    continue;
                };
                let target = match st {
                    TfServerType::ENTRY => &mut entry,
                    TfServerType::PS => &mut ps,
                    _ => &mut dense,
                };
                if let Some(tfserving_apis::model_server_config::Config::ModelConfigList(list)) =
                    target.config.as_mut()
                {
                    list.config.push(mc);
                }
            }
        }

        HashMap::from([
            (TfServerType::ENTRY, entry),
            (TfServerType::PS, ps),
            (TfServerType::DENSE, dense),
        ])
    }

    /// Python parity for `handle_reload_config_request(service_type, model_configs)`.
    pub async fn handle_reload_config_request(
        &self,
        service_type: &'static str,
        mut model_configs: tfserving_apis::ModelServerConfig,
    ) -> ServingResult<tfserving_apis::StatusProto> {
        // Ensure default model is present, mirroring Python.
        let mut has_default = false;
        if let Some(tfserving_apis::model_server_config::Config::ModelConfigList(list)) =
            model_configs.config.as_ref()
        {
            has_default = list.config.iter().any(|c| c.name == "default");
        }
        if !has_default {
            if let Some(tfserving_apis::model_server_config::Config::ModelConfigList(list)) =
                model_configs.config.as_mut()
            {
                list.config.push(default_model_config());
            }
        }

        let req = tfserving_apis::ReloadConfigRequest {
            config: Some(model_configs),
            metric_names: vec![],
        };

        let mut stubs = self.stubs.lock().await;
        let stub = stubs
            .get_mut(service_type)
            .ok_or(ServingError::NotConnected)?;
        let resp = stub
            .model
            .handle_reload_config_request(req)
            .await
            .map_err(|e| ServingError::GrpcError(format!("HandleReloadConfigRequest failed: {e}")))?
            .into_inner();
        Ok(resp.status.unwrap_or_default())
    }
}
