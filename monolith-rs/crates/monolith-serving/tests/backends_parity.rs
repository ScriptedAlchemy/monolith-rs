use monolith_serving::backends::{
    Container, ContainerServiceInfo, FakeKazooClient, SavedModel, SavedModelDeployConfig, ZkBackend,
};
use monolith_serving::ZkClient;
use std::collections::HashSet;
use std::sync::Arc;

// Mirrors `monolith/agent_service/backends_test.py` (ZKBackendTest).
#[test]
fn zk_backend_service_register_and_layout_and_sync() {
    let bzid = "gip".to_string();
    let container = Container::new("default", "asdf");
    let service_info = ContainerServiceInfo {
        grpc: Some("localhost:8888".to_string()),
        http: Some("localhost:8889".to_string()),
        archon: Some("localhost:8890".to_string()),
        agent: Some("localhost:8891".to_string()),
        idc: Some("IDC".to_string()),
        debug_info: None,
    };

    let zk = Arc::new(FakeKazooClient::new());
    zk.start().unwrap();

    let mut backend = ZkBackend::new(bzid.clone(), zk.clone());
    backend.start().unwrap();
    backend
        .report_service_info(&container, &service_info)
        .unwrap();

    // test_register_service
    let got = backend.get_service_info(&container).unwrap().unwrap();
    assert_eq!(got, service_info);

    // test_layout_callback
    let layout_path = format!("/{bzid}/layouts/test_layout/mixed");
    let record: Arc<std::sync::Mutex<Option<Vec<(SavedModel, SavedModelDeployConfig)>>>> =
        Arc::new(std::sync::Mutex::new(None));
    let record_cb = record.clone();
    backend
        .register_layout_callback(&layout_path, move |saved_models| {
            *record_cb.lock().unwrap() = Some(saved_models);
            true
        })
        .unwrap();

    let base_path = |sub_graph: &str| format!("/tmp/test_ffm_model/exported_models/{sub_graph}");
    for sub_graph in ["entry", "ps_0", "ps_1", "ps_2"] {
        let sm = SavedModel::new("test_ffm_model", sub_graph);
        let cfg = SavedModelDeployConfig {
            model_base_path: Some(base_path(sub_graph)),
            version_policy: Some("latest".to_string()),
        };
        backend.decl_saved_model(&sm, &cfg).unwrap();
        backend.add_to_layout(&layout_path, &sm).unwrap();
    }

    let expected = vec![
        (
            SavedModel::new("test_ffm_model", "entry"),
            SavedModelDeployConfig {
                model_base_path: Some(base_path("entry")),
                version_policy: Some("latest".to_string()),
            },
        ),
        (
            SavedModel::new("test_ffm_model", "ps_0"),
            SavedModelDeployConfig {
                model_base_path: Some(base_path("ps_0")),
                version_policy: Some("latest".to_string()),
            },
        ),
        (
            SavedModel::new("test_ffm_model", "ps_1"),
            SavedModelDeployConfig {
                model_base_path: Some(base_path("ps_1")),
                version_policy: Some("latest".to_string()),
            },
        ),
        (
            SavedModel::new("test_ffm_model", "ps_2"),
            SavedModelDeployConfig {
                model_base_path: Some(base_path("ps_2")),
                version_policy: Some("latest".to_string()),
            },
        ),
    ];
    assert_eq!(record.lock().unwrap().clone().unwrap(), expected);

    backend
        .remove_from_layout(&layout_path, &SavedModel::new("test_ffm_model", "entry"))
        .unwrap();

    let expected_after = vec![
        (
            SavedModel::new("test_ffm_model", "ps_0"),
            SavedModelDeployConfig {
                model_base_path: Some(base_path("ps_0")),
                version_policy: Some("latest".to_string()),
            },
        ),
        (
            SavedModel::new("test_ffm_model", "ps_1"),
            SavedModelDeployConfig {
                model_base_path: Some(base_path("ps_1")),
                version_policy: Some("latest".to_string()),
            },
        ),
        (
            SavedModel::new("test_ffm_model", "ps_2"),
            SavedModelDeployConfig {
                model_base_path: Some(base_path("ps_2")),
                version_policy: Some("latest".to_string()),
            },
        ),
    ];
    assert_eq!(record.lock().unwrap().clone().unwrap(), expected_after);

    // test_sync_available_models
    let saved_models = HashSet::from([
        SavedModel::new("test_ffm_model", "entry"),
        SavedModel::new("test_ffm_model", "ps_0"),
        SavedModel::new("test_ffm_model", "ps_1"),
    ]);
    backend
        .sync_available_saved_models(&container, saved_models)
        .unwrap();
    assert!(zk.exists(&format!("/{bzid}/binding/test_ffm_model/entry:{container}")));
    assert!(zk.exists(&format!("/{bzid}/binding/test_ffm_model/ps_0:{container}")));
    assert!(zk.exists(&format!("/{bzid}/binding/test_ffm_model/ps_1:{container}")));

    // `bzid_info()` also includes binding cross-references in Python.
    let info = backend.bzid_info().unwrap();
    let entry_bindings = info
        .get("model_info")
        .and_then(|v| v.get("test_ffm_model"))
        .and_then(|v| v.get("entry"))
        .and_then(|v| v.get("bindings"))
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect::<HashSet<_>>();
    assert_eq!(
        entry_bindings,
        HashSet::from([format!("{}:{}", container.ctx_cluster, container.ctx_id)])
    );
    let saved_models = info
        .get("container_info")
        .and_then(|v| v.get("default"))
        .and_then(|v| v.get("asdf"))
        .and_then(|v| v.get("saved_models"))
        .and_then(|v| v.as_array())
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect::<HashSet<_>>();
    assert!(saved_models.contains("test_ffm_model:entry"));

    // test_service_map: sync two models and ensure map contains service_info for each.
    let saved_models = HashSet::from([
        SavedModel::new("test_ffm_model", "entry"),
        SavedModel::new("test_ffm_model", "ps_0"),
    ]);
    backend
        .sync_available_saved_models(&container, saved_models)
        .unwrap();
    let service_map = backend.get_service_map();
    assert_eq!(
        service_map
            .get("test_ffm_model")
            .unwrap()
            .get("ps_0")
            .unwrap(),
        &vec![service_info.clone()]
    );
    assert_eq!(
        service_map
            .get("test_ffm_model")
            .unwrap()
            .get("entry")
            .unwrap(),
        &vec![service_info.clone()]
    );

    // test_sync_backend
    backend.subscribe_model("test_ffm_model").unwrap();
    let saved_models = HashSet::from([
        SavedModel::new("test_ffm_model", "ps_0"),
        SavedModel::new("test_ffm_model", "ps_1"),
        SavedModel::new("test_ffm_model", "ps_2"),
    ]);
    backend
        .sync_available_saved_models(&container, saved_models)
        .unwrap();
    let (model_name, targets) = backend.get_sync_targets("ps_1").unwrap();
    assert_eq!(model_name, "test_ffm_model:ps_1");
    assert_eq!(targets, vec!["localhost:8888".to_string()]);

    backend.stop().unwrap();
}
