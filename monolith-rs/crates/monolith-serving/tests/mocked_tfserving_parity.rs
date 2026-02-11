#![cfg(feature = "grpc")]

use monolith_proto::tensorflow_serving::apis as tfserving_apis;
use monolith_serving::mocked_tfserving::{find_free_port_blocking, FakeTfServing};
use monolith_serving::ServingResult;
use std::net::SocketAddr;
use tokio::time::{sleep, Duration};

// Mirrors `monolith/agent_service/mocked_tfserving_test.py`.

const MODEL_NAME: &str = "test_model_test";
const BASE_PATH: &str = "/tmp/test_model/monolith";

#[tokio::test]
async fn mocked_tfserving_get_model_metadata_status_reload() -> ServingResult<()> {
    let port = find_free_port_blocking();
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let mut tfs = FakeTfServing::new(addr);
    let initial = FakeTfServing::default_model_config(MODEL_NAME, BASE_PATH, 2);
    tfs.start_with_configs(vec![initial]).await?;
    sleep(Duration::from_millis(50)).await;

    let endpoint = format!("http://127.0.0.1:{port}");

    // GetModelMetadata
    let mut pred = tfserving_apis::prediction_service_client::PredictionServiceClient::connect(
        endpoint.clone(),
    )
    .await
    .unwrap();
    let req = tfserving_apis::GetModelMetadataRequest {
        model_spec: Some(tfserving_apis::ModelSpec {
            name: MODEL_NAME.to_string(),
            version_choice: Some(tfserving_apis::model_spec::VersionChoice::Version(2)),
            signature_name: "predict".to_string(),
        }),
        metadata_field: vec![
            "base_path".to_string(),
            "num_versions".to_string(),
            "signature_name".to_string(),
        ],
    };
    let resp = pred.get_model_metadata(req).await.unwrap().into_inner();
    assert!(resp.metadata.contains_key("base_path"));

    // GetModelStatus
    let mut model =
        tfserving_apis::model_service_client::ModelServiceClient::connect(endpoint.clone())
            .await
            .unwrap();
    let status_req = tfserving_apis::GetModelStatusRequest {
        model_spec: Some(tfserving_apis::ModelSpec {
            name: MODEL_NAME.to_string(),
            version_choice: Some(tfserving_apis::model_spec::VersionChoice::Version(1)),
            signature_name: "predict".to_string(),
        }),
    };
    let status_resp = model
        .get_model_status(status_req)
        .await
        .unwrap()
        .into_inner();
    assert!(!status_resp.model_version_status.is_empty());

    // HandleReloadConfigRequest
    let mut cfgs = Vec::new();
    cfgs.push(FakeTfServing::default_model_config(
        "test_model",
        "/tmp/test_model/ctr/saved_model",
        2,
    ));
    cfgs.push(FakeTfServing::default_model_config(
        "test_model",
        "/tmp/test_model/cvr/saved_model",
        1,
    ));

    let request = tfserving_apis::ReloadConfigRequest {
        config: Some(tfserving_apis::ModelServerConfig {
            config: Some(
                tfserving_apis::model_server_config::Config::ModelConfigList(
                    tfserving_apis::ModelConfigList { config: cfgs },
                ),
            ),
        }),
        metric_names: vec![],
    };
    let reload_resp = model
        .handle_reload_config_request(request)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        reload_resp.status.unwrap().error_code,
        monolith_proto::tensorflow_serving::error::Code::Ok as i32
    );

    tfs.stop().await;
    Ok(())
}
