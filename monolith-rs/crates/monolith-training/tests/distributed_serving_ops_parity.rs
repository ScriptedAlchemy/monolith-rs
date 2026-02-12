use monolith_serving::backends::{
    Container, ContainerServiceInfo, FakeKazooClient, SavedModel, ZkBackend,
};
use monolith_training::refresh_sync_config;
use std::collections::HashSet;

#[test]
fn refresh_sync_config_from_zk_backend_matches_python() {
    // Mirrors `monolith/native_training/distributed_serving_ops_test.py::test_refresh_sync_config_2`.
    let zk = std::sync::Arc::new(FakeKazooClient::new());
    let bd = ZkBackend::new("demo", zk);
    bd.start()
        .expect("ZkBackend start should succeed for fake client");

    let container = Container::new("default", "asdf");
    let service_info = ContainerServiceInfo {
        grpc: Some("localhost:8888".to_string()),
        http: Some("localhost:8889".to_string()),
        archon: Some("localhost:8890".to_string()),
        agent: Some("localhost:8891".to_string()),
        idc: Some("lf".to_string()),
        debug_info: None,
    };
    bd.report_service_info(&container, &service_info)
        .expect("report_service_info should succeed");
    bd.sync_available_saved_models(
        &container,
        HashSet::from([
            SavedModel::new("test_ffm_model", "ps_0"),
            SavedModel::new("test_ffm_model", "ps_1"),
            SavedModel::new("test_ffm_model", "ps_2"),
        ]),
    )
    .expect("sync_available_saved_models should succeed");

    bd.subscribe_model("test_ffm_model")
        .expect("subscribe_model should succeed");

    let cfg = refresh_sync_config(&bd, 1).expect("refresh_sync_config should succeed");
    assert_eq!(cfg.model_name.as_deref(), Some("test_ffm_model:ps_1"));
    assert_eq!(cfg.targets, vec!["localhost:8888".to_string()]);
    assert_eq!(cfg.signature_name.as_deref(), Some("hashtable_assign"));
    assert_eq!(cfg.timeout_in_ms, Some(3000));

    bd.stop().expect("ZkBackend stop should succeed");
}
