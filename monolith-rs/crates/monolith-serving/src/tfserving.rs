#![cfg(feature = "grpc")]
//! TensorFlow Serving (TFS) client utilities.
//!
//! This module provides a small tonic/prost client for a subset of TF Serving APIs
//! that Monolith's Python tooling uses:
//! - ModelService/GetModelStatus
//! - ModelService/HandleReloadConfigRequest
//! - PredictionService/Predict
//! - pbtxt parsing for `ModelServerConfig`
//!
//! We intentionally keep this wrapper lightweight and focused on common Monolith
//! workflows rather than attempting to expose the entire TF Serving surface area.

use crate::error::{ServingError, ServingResult};
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use monolith_proto::tensorflow_serving::config as tfserving_config;
use prost::Message as ProstMessage;
use prost_reflect::prost::Message as ReflectMessage;
use prost_reflect::{DescriptorPool, DynamicMessage};
use tonic::transport::{Channel, Endpoint};

/// A TF Serving client (tonic).
#[derive(Clone)]
pub struct TfServingClient {
    model: tfserving_apis::model_service_client::ModelServiceClient<Channel>,
    predict: tfserving_apis::prediction_service_client::PredictionServiceClient<Channel>,
}

impl TfServingClient {
    /// Connect to a TF Serving gRPC endpoint (e.g. `http://127.0.0.1:8500`).
    pub async fn connect(endpoint: &str) -> ServingResult<Self> {
        let ep = Endpoint::from_shared(endpoint.to_string()).map_err(|e| {
            ServingError::ConfigError(format!("Invalid TF Serving endpoint {endpoint:?}: {e}"))
        })?;
        let ch = ep.connect().await.map_err(|e| {
            ServingError::GrpcError(format!("Failed to connect to TF Serving {endpoint:?}: {e}"))
        })?;
        Ok(Self {
            model: tfserving_apis::model_service_client::ModelServiceClient::new(ch.clone()),
            predict: tfserving_apis::prediction_service_client::PredictionServiceClient::new(ch),
        })
    }

    /// Call `ModelService/GetModelStatus`.
    pub async fn get_model_status(
        &mut self,
        model_name: &str,
        signature_name: Option<&str>,
    ) -> ServingResult<tfserving_apis::GetModelStatusResponse> {
        let spec = tfserving_apis::ModelSpec {
            name: model_name.to_string(),
            version_choice: None,
            signature_name: signature_name.unwrap_or_default().to_string(),
        };

        let request = tfserving_apis::GetModelStatusRequest {
            model_spec: Some(spec),
        };

        let resp: tonic::Response<tfserving_apis::GetModelStatusResponse> = self
            .model
            .get_model_status(request)
            .await
            .map_err(|e| ServingError::GrpcError(format!("GetModelStatus failed: {e}")))?;
        Ok(resp.into_inner())
    }

    /// Call `PredictionService/Predict`.
    pub async fn predict(
        &mut self,
        request: tfserving_apis::PredictRequest,
    ) -> ServingResult<tfserving_apis::PredictResponse> {
        let resp: tonic::Response<tfserving_apis::PredictResponse> = self
            .predict
            .predict(request)
            .await
            .map_err(|e| ServingError::GrpcError(format!("Predict failed: {e}")))?;
        Ok(resp.into_inner())
    }

    /// Call `ModelService/HandleReloadConfigRequest`.
    pub async fn reload_config(
        &mut self,
        config: tfserving_config::ModelServerConfig,
    ) -> ServingResult<tfserving_apis::ReloadConfigResponse> {
        let request = tfserving_apis::ReloadConfigRequest {
            config: Some(config),
            metric_names: vec![],
        };
        let resp: tonic::Response<tfserving_apis::ReloadConfigResponse> = self
            .model
            .handle_reload_config_request(request)
            .await
            .map_err(|e| {
                ServingError::GrpcError(format!("HandleReloadConfigRequest failed: {e}"))
            })?;
        Ok(resp.into_inner())
    }
}

/// Parse a TF Serving `ModelServerConfig` pbtxt string.
///
/// Python uses `google.protobuf.text_format.Parse` against `ModelServerConfig`.
/// In Rust, we use `prost-reflect` dynamic text-format parsing and then decode
/// into the concrete generated `ModelServerConfig`.
pub fn parse_model_server_config_pbtxt(
    descriptor_pool: &DescriptorPool,
    pbtxt: &str,
) -> ServingResult<tfserving_config::ModelServerConfig> {
    let msg_desc = descriptor_pool
        .get_message_by_name("tensorflow.serving.ModelServerConfig")
        .ok_or_else(|| {
            ServingError::ConfigError(
                "DescriptorPool missing tensorflow.serving.ModelServerConfig".to_string(),
            )
        })?;

    let dynamic = DynamicMessage::parse_text_format(msg_desc, pbtxt).map_err(|e| {
        ServingError::ConfigError(format!("Failed to parse ModelServerConfig pbtxt: {e}"))
    })?;

    let bytes = ReflectMessage::encode_to_vec(&dynamic);
    tfserving_config::ModelServerConfig::decode(bytes.as_slice()).map_err(|e| {
        ServingError::ConfigError(format!(
            "Failed to decode ModelServerConfig from pbtxt: {e}"
        ))
    })
}
