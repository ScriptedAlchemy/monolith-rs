use monolith_proto::monolith::io::proto::{feature, Example, Feature, FidList, NamedFeature};
use monolith_proto::tensorflow_core::{DataType, TensorProto};
use monolith_proto::tensorflow_serving::apis::prediction_service_server::PredictionService;
use monolith_proto::tensorflow_serving::apis::PredictRequest as TfPredictRequest;
use monolith_serving::config::ModelLoaderConfig;
use monolith_serving::tfserving_server::{TfServingPredictionServer, INPUT_EXAMPLE, OUTPUT_SCORES};
use monolith_serving::{AgentServiceImpl, ModelLoader};
use prost::Message;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use tonic::Request;

fn make_example() -> Example {
    let feature = Feature {
        r#type: Some(feature::Type::FidV2List(FidList { value: vec![123] })),
    };
    Example {
        named_feature: vec![NamedFeature {
            id: 0,
            name: "user_id".to_string(),
            feature: Some(feature),
            sorted_id: 0,
        }],
        named_raw_feature: vec![],
        line_id: None,
        label: vec![1.0],
        instance_weight: 1.0,
        data_source_key: 0,
    }
}

#[tokio::test]
async fn test_tfserving_predict_single_example() {
    let dir = tempdir().unwrap();
    let model_path = dir.path().join("model");
    std::fs::create_dir_all(&model_path).unwrap();

    let loader = Arc::new(ModelLoader::new(ModelLoaderConfig::default()));
    loader.load(&model_path).await.unwrap();

    let agent = Arc::new(AgentServiceImpl::new(loader, None));
    let service = TfServingPredictionServer::new(agent);

    let example = make_example();
    let bytes = example.encode_to_vec();

    let mut inputs = HashMap::new();
    inputs.insert(
        INPUT_EXAMPLE.to_string(),
        TensorProto {
            dtype: DataType::DtString as i32,
            string_val: vec![bytes],
            ..Default::default()
        },
    );

    let req = TfPredictRequest {
        model_spec: None,
        inputs,
        output_filter: vec![],
        predict_streamed_options: None,
        client_id: None,
        request_options: None,
    };

    let resp = service
        .predict(Request::new(req))
        .await
        .unwrap()
        .into_inner();
    assert!(resp.outputs.contains_key(OUTPUT_SCORES));
}
