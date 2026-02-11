//! TFServing process wrapper (Python parity).
//!
//! The Python agent uses `TFSWrapper` to spawn a TFServing binary and query it
//! via `ModelService/GetModelStatus`. In Rust we implement a small subset
//! needed by parity tests and future integration:
//! - model_config_text()
//! - list_saved_models() (pbtxt parsing)
//! - list_saved_models_status() (best-effort GetModelStatus)
//!
//! Process spawning is intentionally out of scope for parity tests; callers can
//! implement their own lifecycle management or use a container runtime.

#![cfg(feature = "grpc")]

use crate::error::{ServingError, ServingResult};
use crate::tfserving::{parse_model_server_config_pbtxt, TfServingClient};
use monolith_proto::descriptor_pool::descriptor_pool;
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use std::collections::HashMap;
use std::path::PathBuf;

/// Lightweight wrapper around a model_config pbtxt file.
pub struct TfsWrapper {
    model_config_file: PathBuf,
    grpc_endpoint: String,
}

impl TfsWrapper {
    /// Create a wrapper pointing at a TFServing endpoint and a model_config file path.
    ///
    /// `grpc_endpoint` should be something like `http://127.0.0.1:8500`.
    pub fn new(model_config_file: impl Into<PathBuf>, grpc_endpoint: impl Into<String>) -> Self {
        Self {
            model_config_file: model_config_file.into(),
            grpc_endpoint: grpc_endpoint.into(),
        }
    }

    /// Read the model_config pbtxt file contents.
    pub fn model_config_text(&self) -> ServingResult<String> {
        Ok(std::fs::read_to_string(&self.model_config_file)?)
    }

    /// List all saved model names present in the model_config.
    pub fn list_saved_models(&self) -> ServingResult<Vec<String>> {
        let pbtxt = self.model_config_text()?;
        let config = parse_model_server_config_pbtxt(&descriptor_pool(), &pbtxt)?;
        let list = match config.config {
            Some(tfserving_apis::model_server_config::Config::ModelConfigList(list)) => list.config,
            _ => Vec::new(),
        };
        Ok(list.into_iter().map(|c| c.name).collect())
    }

    /// Query TF Serving and return a best-effort status per saved model.
    pub async fn list_saved_models_status(
        &self,
    ) -> ServingResult<HashMap<String, tfserving_apis::ModelVersionStatus>> {
        let saved = self.list_saved_models()?;
        let mut client = TfServingClient::connect(&self.grpc_endpoint).await?;

        let mut out = HashMap::new();
        for name in saved {
            let resp = client.get_model_status(&name, None).await;
            let status = match resp {
                Ok(r) => {
                    let mut mvs = r.model_version_status;
                    if mvs.is_empty() {
                        tfserving_apis::ModelVersionStatus {
                            version: -1,
                            state: tfserving_apis::model_version_status::State::Unknown as i32,
                            status: None,
                        }
                    } else {
                        // Python picks available one if present, otherwise latest.
                        mvs.sort_by_key(|s| s.version);
                        let available = mvs
                            .iter()
                            .filter(|s| {
                                s.state
                                    == tfserving_apis::model_version_status::State::Available as i32
                            })
                            .last()
                            .cloned();
                        available.unwrap_or_else(|| mvs.last().cloned().unwrap())
                    }
                }
                Err(e) => {
                    // Keep error mapping lightweight.
                    return Err(ServingError::GrpcError(format!(
                        "GetModelStatus failed: {e}"
                    )));
                }
            };
            out.insert(name, status);
        }
        Ok(out)
    }
}

/// Fake wrapper used by tests for Agent components (Python parity).
pub struct FakeTfsWrapper {
    model_config_file: PathBuf,
}

impl FakeTfsWrapper {
    /// Create a fake wrapper backed by a model_config pbtxt file.
    pub fn new(model_config_file: impl Into<PathBuf>) -> Self {
        Self {
            model_config_file: model_config_file.into(),
        }
    }

    /// Read the model_config pbtxt file contents.
    pub fn model_config_text(&self) -> ServingResult<String> {
        Ok(std::fs::read_to_string(&self.model_config_file)?)
    }

    /// List all saved model names present in the model_config.
    pub fn list_saved_models(&self) -> ServingResult<Vec<String>> {
        let pbtxt = self.model_config_text()?;
        let config = parse_model_server_config_pbtxt(&descriptor_pool(), &pbtxt)?;
        let list = match config.config {
            Some(tfserving_apis::model_server_config::Config::ModelConfigList(list)) => list.config,
            _ => Vec::new(),
        };
        Ok(list.into_iter().map(|c| c.name).collect())
    }

    /// Return a synthetic "Available" status for each model (tests only).
    pub fn list_saved_models_status(
        &self,
    ) -> ServingResult<HashMap<String, tfserving_apis::ModelVersionStatus>> {
        let saved = self.list_saved_models()?;
        let mut out = HashMap::new();
        for name in saved {
            out.insert(
                name,
                tfserving_apis::ModelVersionStatus {
                    version: 1,
                    state: tfserving_apis::model_version_status::State::Available as i32,
                    status: Some(tfserving_apis::StatusProto::default()),
                },
            );
        }
        Ok(out)
    }
}
