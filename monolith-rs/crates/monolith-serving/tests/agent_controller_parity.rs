use monolith_serving::agent_controller::{declare_saved_model, map_model_to_layout};
use monolith_serving::backends::{FakeKazooClient, SavedModel, ZkBackend};
use monolith_serving::ZkClient;
use std::sync::Arc;

// Mirrors `monolith/agent_service/agent_controller_test.py`.
#[test]
fn agent_controller_decl_saved_models_and_pub_unpub() {
    let bzid = "gip";
    let zk = Arc::new(FakeKazooClient::new());
    zk.start().unwrap();
    let backend = ZkBackend::new(bzid.to_string(), zk.clone());
    backend.start().unwrap();

    // `TEST_SRCDIR/TEST_WORKSPACE` is Bazel-specific in Python tests. In Rust parity tests,
    // use repo-relative testdata.
    let repo_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../..")
        .canonicalize()
        .unwrap();
    let export_base = repo_root.join("monolith/native_training/model_export/testdata/saved_model");

    // decl
    let model_name = declare_saved_model(
        &backend,
        &export_base,
        Some("test_ffm_model"),
        true,
        "entry_ps",
    )
    .unwrap();
    assert_eq!(model_name, "test_ffm_model");
    let saved_models = backend.list_saved_models("test_ffm_model").unwrap();
    let got = saved_models
        .into_iter()
        .collect::<std::collections::HashSet<_>>();
    let expected = ["ps_0", "ps_1", "ps_2", "ps_3", "ps_4", "entry"]
        .into_iter()
        .map(|sg| SavedModel::new("test_ffm_model", sg))
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(got, expected);

    // pub/unpub
    map_model_to_layout(
        &backend,
        "test_ffm_model:entry",
        "/gip/layouts/test_layout1",
        "pub",
    )
    .unwrap();
    assert_eq!(
        backend
            .bzid_info()
            .unwrap()
            .get("layout_info")
            .unwrap()
            .get("test_layout1")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
        vec!["test_ffm_model:entry".to_string()]
    );

    map_model_to_layout(
        &backend,
        "test_ffm_model:ps_*",
        "/gip/layouts/test_layout1",
        "pub",
    )
    .unwrap();
    assert_eq!(
        backend
            .bzid_info()
            .unwrap()
            .get("layout_info")
            .unwrap()
            .get("test_layout1")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
        vec![
            "test_ffm_model:entry",
            "test_ffm_model:ps_0",
            "test_ffm_model:ps_1",
            "test_ffm_model:ps_2",
            "test_ffm_model:ps_3",
            "test_ffm_model:ps_4",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
    );

    map_model_to_layout(
        &backend,
        "test_ffm_model:ps_*",
        "/gip/layouts/test_layout1",
        "unpub",
    )
    .unwrap();
    assert_eq!(
        backend
            .bzid_info()
            .unwrap()
            .get("layout_info")
            .unwrap()
            .get("test_layout1")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect::<Vec<_>>(),
        vec!["test_ffm_model:entry".to_string()]
    );

    map_model_to_layout(
        &backend,
        "test_ffm_model:entry",
        "/gip/layouts/test_layout1",
        "unpub",
    )
    .unwrap();
    assert_eq!(
        backend
            .bzid_info()
            .unwrap()
            .get("layout_info")
            .unwrap()
            .get("test_layout1")
            .unwrap()
            .as_array()
            .unwrap()
            .len(),
        0
    );
}
