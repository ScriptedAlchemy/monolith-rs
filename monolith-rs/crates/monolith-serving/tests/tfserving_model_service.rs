use monolith_proto::tensorflow_serving::apis::{
    model_server_config, model_service_server::ModelService, GetModelStatusRequest, ModelConfig,
    ModelConfigList, ModelServerConfig, ModelSpec, ReloadConfigRequest,
};
use monolith_serving::tfserving_server::TfServingModelServer;
use tonic::Request;

#[tokio::test]
async fn test_tfserving_model_service_status() {
    let service = TfServingModelServer::new();

    let config = ModelServerConfig {
        config: Some(model_server_config::Config::ModelConfigList(
            ModelConfigList {
                config: vec![ModelConfig {
                    name: "demo".to_string(),
                    base_path: "/models/demo".to_string(),
                    ..Default::default()
                }],
            },
        )),
    };

    let reload_req = ReloadConfigRequest {
        config: Some(config),
        metric_names: vec![],
    };
    service
        .handle_reload_config_request(Request::new(reload_req))
        .await
        .unwrap();

    let status_req = GetModelStatusRequest {
        model_spec: Some(ModelSpec {
            name: "demo".to_string(),
            version_choice: None,
            signature_name: "".to_string(),
        }),
    };

    let response = service
        .get_model_status(Request::new(status_req))
        .await
        .unwrap()
        .into_inner();

    assert_eq!(response.model_version_status.len(), 1);
}
