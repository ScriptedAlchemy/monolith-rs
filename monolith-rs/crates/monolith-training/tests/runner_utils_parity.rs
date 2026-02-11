use monolith_training::native_training::service_discovery::ServiceDiscoveryType;
use monolith_training::{
    get_discovery, monolith_discovery, prepare_restore_checkpoint, RunConfig, RunnerConfig,
};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tempfile::tempdir;

#[test]
fn test_run_config_to_discovery_selection_primus() {
    let tf_config = serde_json::json!({
      "cluster": {
        "chief": ["chief:2222"],
        "ps": ["ps0:2222"],
        "worker": ["worker0:2222"]
      },
      "task": {"type": "worker", "index": 0}
    })
    .to_string();

    let run = RunConfig {
        is_local: false,
        num_ps: 1,
        num_workers: 2,
        discovery_type: ServiceDiscoveryType::Primus,
        tf_config: Some(tf_config),
        ..RunConfig::default()
    };
    let runner = run
        .to_runner_config(Some(RunnerConfig {
            is_local: false,
            ..RunnerConfig::default()
        }))
        .unwrap();

    let d = get_discovery(&runner, None).unwrap().expect("discovery");
    assert_eq!(d.kind(), "tf_config");
}

#[test]
fn test_monolith_discovery_local_returns_none() {
    let rc = RunnerConfig {
        is_local: true,
        ..RunnerConfig::default()
    };
    let guard = monolith_discovery(&rc, None).unwrap();
    assert!(guard.discovery().is_none());
}

#[test]
fn test_prepare_restore_checkpoint_chief_then_worker() {
    let tmp = tempdir().unwrap();
    let restore_dir = tmp.path().join("restore");
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&restore_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();

    let pbtxt = r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#;
    fs::write(restore_dir.join("checkpoint"), pbtxt).unwrap();

    let chief = prepare_restore_checkpoint(
        &restore_dir,
        &model_dir,
        Some("model.ckpt-30"),
        true,
        Duration::from_secs(1),
        Duration::from_millis(10),
    )
    .unwrap();
    assert_eq!(
        Path::new(&chief.model_checkpoint_path).file_name().unwrap(),
        "model.ckpt-30"
    );

    let worker = prepare_restore_checkpoint(
        &restore_dir,
        &model_dir,
        None,
        false,
        Duration::from_secs(1),
        Duration::from_millis(10),
    )
    .unwrap();
    assert_eq!(
        Path::new(&worker.model_checkpoint_path).file_name().unwrap(),
        "model.ckpt-30"
    );
}
