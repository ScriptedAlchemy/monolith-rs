use monolith_serving::agent_v3::AgentV3;
use monolith_serving::backends::{FakeKazooClient, SavedModelDeployConfig, ZkBackend};
use monolith_serving::tfs_wrapper::FakeTfsWrapper;
use monolith_serving::utils::AgentConfig;
use monolith_serving::{TfsWrapperApi, ZkClient};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

// Mirrors `monolith/agent_service/agent_v3_test.py`.
#[tokio::test]
async fn agent_v3_service_info_and_publish_flow() {
    std::env::set_var("MY_HOST_IP", "127.0.0.1");

    let zk = Arc::new(FakeKazooClient::new());
    zk.start().unwrap();

    let mut agent_conf = AgentConfig {
        bzid: "gip".to_string(),
        base_name: None,
        base_path: None,
        num_ps: 1,
        num_shard: 1,
        deploy_type: monolith_serving::DeployType::Unified,
        stand_alone_serving: false,
        zk_servers: Some("127.0.0.1:8888".to_string()),
        dc_aware: false,
        agent_version: 3,
        agent_port: monolith_serving::find_free_port().unwrap(),
        layout_pattern: Some("/gip/layout".to_string()),
        layout_filters: vec![],
        tfs_port_archon: monolith_serving::find_free_port().unwrap(),
        tfs_port_grpc: monolith_serving::find_free_port().unwrap(),
        tfs_port_http: monolith_serving::find_free_port().unwrap(),
    };

    let mut agent = AgentV3::new(agent_conf.clone(), zk.clone()).unwrap();

    // Replace TFS wrapper with a fake that just reads the model_config file.
    let model_config_path = agent.model_config_path().to_path_buf();
    let fake_tfs = Arc::new(FakeTfsWrapper::new(model_config_path.clone()));
    agent.set_tfs_wrapper(fake_tfs.clone() as Arc<dyn TfsWrapperApi>);

    agent.start().await.unwrap();

    // Populate saved_model deploy configs under /gip/saved_models/test_ffm_model/*
    let base_path = std::env::temp_dir().join("test_ffm_model/exported_models");
    for sub_graph in ["entry", "ps_0", "ps_1", "ps_2"] {
        let cfg = SavedModelDeployConfig {
            model_base_path: Some(base_path.join(sub_graph).to_string_lossy().to_string()),
            version_policy: Some("latest".to_string()),
        };
        let path = format!("/gip/saved_models/test_ffm_model/{sub_graph}");
        zk.create(&path, cfg.serialize(), false, true).unwrap();
    }

    // Background service-info reporter is immediate; allow one tick.
    sleep(Duration::from_millis(50)).await;

    // service_info test
    let backend: &ZkBackend = agent.backend();
    let got = backend
        .get_service_info(agent.container())
        .unwrap()
        .expect("service_info must exist");
    assert_eq!(got, agent.service_info().clone());

    // publish models
    assert_eq!(fake_tfs.list_saved_models().unwrap(), Vec::<String>::new());

    zk.ensure_path("/gip/layout/test_ffm_model:entry").unwrap();
    zk.ensure_path("/gip/layout/test_ffm_model:ps_0").unwrap();

    // Layout callback writes model config; allow it to run.
    sleep(Duration::from_millis(50)).await;
    assert_eq!(
        fake_tfs.list_saved_models().unwrap(),
        vec![
            "test_ffm_model:entry".to_string(),
            "test_ffm_model:ps_0".to_string()
        ]
    );

    agent.sync_available_saved_models().await.unwrap();
    let service_map = backend.get_service_map();
    assert_eq!(
        service_map
            .get("test_ffm_model")
            .unwrap()
            .get("entry")
            .unwrap(),
        &vec![agent.service_info().clone()]
    );
    assert_eq!(
        service_map
            .get("test_ffm_model")
            .unwrap()
            .get("ps_0")
            .unwrap(),
        &vec![agent.service_info().clone()]
    );

    // Unpublish one model.
    zk.delete("/gip/layout/test_ffm_model:ps_0", true).unwrap();
    sleep(Duration::from_millis(50)).await;
    assert_eq!(
        fake_tfs.list_saved_models().unwrap(),
        vec!["test_ffm_model:entry".to_string()]
    );

    agent.sync_available_saved_models().await.unwrap();
    let service_map = backend.get_service_map();
    assert!(service_map
        .get("test_ffm_model")
        .unwrap()
        .get("ps_0")
        .is_none());
    assert_eq!(
        service_map
            .get("test_ffm_model")
            .unwrap()
            .get("entry")
            .unwrap(),
        &vec![agent.service_info().clone()]
    );

    agent.stop().await;
}
