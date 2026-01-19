#![cfg(feature = "grpc")]
//! TensorFlow Serving-compatible prediction service (server side).
//!
//! This provides a minimal PredictionService implementation that accepts
//! serialized Monolith Example/ExampleBatchRowMajor payloads via a TF Serving
//! PredictRequest input tensor.

use crate::agent_service::{AgentServiceImpl, FeatureInput, PredictRequest};
use crate::error::{ServingError, ServingResult};
use monolith_data::example::extract_feature_data;
use monolith_data::instance::extract_slot;
use monolith_proto::tensorflow_core as tf_core;
use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tonic::{Request, Response, Status};

/// Name of the input tensor carrying serialized ExampleBatchRowMajor bytes.
pub const INPUT_EXAMPLE_BATCH: &str = "example_batch_row_major";

/// Name of the input tensor carrying a single Example bytes.
pub const INPUT_EXAMPLE: &str = "example";

/// Name of the output tensor for scores.
pub const OUTPUT_SCORES: &str = "scores";

/// TensorFlow Serving-compatible PredictionService implementation.
#[derive(Clone)]
pub struct TfServingPredictionServer {
    agent: Arc<AgentServiceImpl>,
}

impl TfServingPredictionServer {
    /// Create a new prediction server backed by the given agent.
    pub fn new(agent: Arc<AgentServiceImpl>) -> Self {
        Self { agent }
    }
}

#[tonic::async_trait]
impl tfserving_apis::prediction_service_server::PredictionService for TfServingPredictionServer {
    async fn predict(
        &self,
        request: Request<tfserving_apis::PredictRequest>,
    ) -> Result<Response<tfserving_apis::PredictResponse>, Status> {
        let req = request.into_inner();
        let examples: Vec<monolith_proto::monolith::io::proto::Example> = extract_examples(&req)
            .map_err(|e| Status::invalid_argument(format!("Invalid PredictRequest: {e}")))?;

        let mut batch_scores: Vec<Vec<f32>> = Vec::with_capacity(examples.len());
        for (idx, example) in examples.iter().enumerate() {
            let features = example_to_features(example).map_err(|e| *e)?;
            let predict_req = PredictRequest {
                request_id: format!("tfserving-{}", idx),
                features,
                return_embeddings: false,
                context: None,
            };
            let resp = self
                .agent
                .predict(predict_req)
                .await
                .map_err(|e| Status::internal(format!("Predict failed: {e}")))?;
            batch_scores.push(resp.scores);
        }

        let outputs = scores_to_outputs(&batch_scores)
            .map_err(|e| Status::internal(format!("Output conversion failed: {e}")))?;

        Ok(Response::new(tfserving_apis::PredictResponse {
            outputs,
            model_spec: req.model_spec,
        }))
    }

    async fn classify(
        &self,
        _request: Request<monolith_proto::tensorflow_serving::apis::ClassificationRequest>,
    ) -> Result<Response<monolith_proto::tensorflow_serving::apis::ClassificationResponse>, Status>
    {
        Err(Status::unimplemented("classify is not implemented"))
    }

    async fn regress(
        &self,
        _request: Request<monolith_proto::tensorflow_serving::apis::RegressionRequest>,
    ) -> Result<Response<monolith_proto::tensorflow_serving::apis::RegressionResponse>, Status>
    {
        Err(Status::unimplemented("regress is not implemented"))
    }

    async fn multi_inference(
        &self,
        _request: Request<monolith_proto::tensorflow_serving::apis::MultiInferenceRequest>,
    ) -> Result<Response<monolith_proto::tensorflow_serving::apis::MultiInferenceResponse>, Status>
    {
        Err(Status::unimplemented("multi_inference is not implemented"))
    }

    async fn get_model_metadata(
        &self,
        _request: Request<monolith_proto::tensorflow_serving::apis::GetModelMetadataRequest>,
    ) -> Result<Response<monolith_proto::tensorflow_serving::apis::GetModelMetadataResponse>, Status>
    {
        Err(Status::unimplemented(
            "get_model_metadata is not implemented",
        ))
    }
}

fn extract_examples(
    req: &tfserving_apis::PredictRequest,
) -> ServingResult<Vec<monolith_proto::monolith::io::proto::Example>> {
    if let Some(tensor) = req.inputs.get(INPUT_EXAMPLE_BATCH) {
        let bytes = tensor_to_bytes(tensor)?;
        let batch: monolith_proto::monolith::io::proto::ExampleBatchRowMajor =
            monolith_proto::monolith::io::proto::ExampleBatchRowMajor::decode(bytes.as_slice())
                .map_err(|e| {
                ServingError::PredictionError(format!("Failed to decode ExampleBatchRowMajor: {e}"))
            })?;
        if batch.example.is_empty() {
            return Err(ServingError::PredictionError(
                "example_batch_row_major is empty".to_string(),
            ));
        }
        return Ok(batch.example);
    }

    if let Some(tensor) = req.inputs.get(INPUT_EXAMPLE) {
        let bytes = tensor_to_bytes(tensor)?;
        let example = monolith_proto::monolith::io::proto::Example::decode(bytes.as_slice())
            .map_err(|e| ServingError::PredictionError(format!("Failed to decode Example: {e}")))?;
        return Ok(vec![example]);
    }

    Err(ServingError::PredictionError(
        "Missing input tensor: example_batch_row_major or example".to_string(),
    ))
}

fn tensor_to_bytes(tensor: &tf_core::TensorProto) -> ServingResult<Vec<u8>> {
    if tensor.dtype != tf_core::DataType::DtString as i32 {
        return Err(ServingError::PredictionError(
            "Input tensor must be DT_STRING".to_string(),
        ));
    }
    if let Some(first) = tensor.string_val.first() {
        return Ok(first.clone());
    }
    if !tensor.tensor_content.is_empty() {
        return Ok(tensor.tensor_content.clone());
    }
    Err(ServingError::PredictionError(
        "Input tensor contains no bytes".to_string(),
    ))
}

fn example_to_features(
    example: &monolith_proto::monolith::io::proto::Example,
) -> Result<Vec<FeatureInput>, Box<Status>> {
    let mut out = Vec::new();
    for nf in &example.named_feature {
        let Some(feature) = &nf.feature else {
            continue;
        };
        let data = extract_feature_data(feature);
        if data.fid.is_empty() {
            continue;
        }
        let slot_id = extract_slot(data.fid[0]);
        let values = if data.value.is_empty() {
            None
        } else {
            Some(data.value.clone())
        };
        out.push(FeatureInput {
            name: nf.name.clone(),
            slot_id,
            fids: data.fid,
            values,
        });
    }
    if out.is_empty() {
        return Err(Box::new(Status::invalid_argument(
            "Example has no usable features",
        )));
    }
    Ok(out)
}

fn scores_to_outputs(scores: &[Vec<f32>]) -> ServingResult<HashMap<String, tf_core::TensorProto>> {
    let batch = scores.len();
    if batch == 0 {
        return Err(ServingError::PredictionError("Empty batch".to_string()));
    }
    let dim = scores[0].len();
    if scores.iter().any(|s| s.len() != dim) {
        return Err(ServingError::PredictionError(
            "Inconsistent score vector lengths".to_string(),
        ));
    }

    let mut flat = Vec::with_capacity(batch * dim);
    for row in scores {
        flat.extend_from_slice(row);
    }

    let shape = tf_core::TensorShapeProto {
        dim: vec![
            tf_core::tensor_shape_proto::Dim {
                size: batch as i64,
                name: "".to_string(),
            },
            tf_core::tensor_shape_proto::Dim {
                size: dim as i64,
                name: "".to_string(),
            },
        ],
        ..Default::default()
    };

    let tensor = tf_core::TensorProto {
        dtype: tf_core::DataType::DtFloat as i32,
        tensor_shape: Some(shape),
        float_val: flat,
        ..Default::default()
    };

    let mut outputs = HashMap::new();
    outputs.insert(OUTPUT_SCORES.to_string(), tensor);
    Ok(outputs)
}
