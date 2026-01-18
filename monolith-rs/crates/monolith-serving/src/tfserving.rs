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
use monolith_proto::tensorflow_serving::apis::{
    model_service_client::ModelServiceClient,
    prediction_service_client::PredictionServiceClient,
    GetModelStatusRequest, PredictRequest, ReloadConfigRequest,
};
use monolith_proto::tensorflow_serving::config::ModelServerConfig;
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage};
use tonic::transport::{Channel, Endpoint};

/// A TF Serving client (tonic).
#[derive(Clone)]
pub struct TfServingClient {
    model: ModelServiceClient<Channel>,
    predict: PredictionServiceClient<Channel>,
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
            model: ModelServiceClient::new(ch.clone()),
            predict: PredictionServiceClient::new(ch),
        })
    }

    /// Call `ModelService/GetModelStatus`.
    pub async fn get_model_status(
        &mut self,
        model_name: &str,
        signature_name: Option<&str>,
    ) -> ServingResult<monolith_proto::tensorflow_serving::apis::GetModelStatusResponse> {
        let spec = monolith_proto::tensorflow_serving::apis::ModelSpec {
            name: model_name.to_string(),
            // `ModelSpec` in TF Serving uses wrappers.Int64Value for version.
            version: None,
            version_label: "".to_string(),
            signature_name: signature_name.unwrap_or_default().to_string(),
        };

        let request = GetModelStatusRequest {
            model_spec: Some(spec),
        };

        let resp = self
            .model
            .get_model_status(request)
            .await
            .map_err(|e| ServingError::GrpcError(format!("GetModelStatus failed: {e}")))?;
        Ok(resp.into_inner())
    }

    /// Call `PredictionService/Predict`.
    pub async fn predict(
        &mut self,
        request: PredictRequest,
    ) -> ServingResult<monolith_proto::tensorflow_serving::apis::PredictResponse> {
        let resp = self
            .predict
            .predict(request)
            .await
            .map_err(|e| ServingError::GrpcError(format!("Predict failed: {e}")))?;
        Ok(resp.into_inner())
    }

    /// Call `ModelService/HandleReloadConfigRequest`.
    pub async fn reload_config(
        &mut self,
        config: ModelServerConfig,
    ) -> ServingResult<monolith_proto::tensorflow_serving::apis::ReloadConfigResponse> {
        let request = ReloadConfigRequest { config: Some(config) };
        let resp = self
            .model
            .handle_reload_config_request(request)
            .await
            .map_err(|e| ServingError::GrpcError(format!("HandleReloadConfigRequest failed: {e}")))?;
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
) -> ServingResult<ModelServerConfig> {
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

    let bytes = dynamic.encode_to_vec();
    ModelServerConfig::decode(bytes.as_slice()).map_err(|e| {
        ServingError::ConfigError(format!("Failed to decode ModelServerConfig from pbtxt: {e}"))
    })
}
