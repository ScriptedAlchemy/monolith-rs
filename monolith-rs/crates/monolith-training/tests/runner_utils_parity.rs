use monolith_training::native_training::service_discovery::ServiceDiscoveryType;
use monolith_training::{
    get_checkpoint_state_with_restore_override, get_discovery,
    get_discovery_from_run_config,
    initialize_restore_checkpoint_from_runner, monolith_discovery, prepare_restore_checkpoint,
    RunConfig, RunnerConfig, RunnerMode,
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
fn test_get_discovery_from_run_config_primus() {
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
        discovery_type: ServiceDiscoveryType::Primus,
        tf_config: Some(tf_config),
        ..RunConfig::default()
    };
    let d = get_discovery_from_run_config(&run, None, None)
        .unwrap()
        .expect("discovery");
    assert_eq!(d.kind(), "tf_config");
}

#[test]
fn test_run_config_to_discovery_selection_zk() {
    let run = RunConfig {
        is_local: false,
        discovery_type: ServiceDiscoveryType::Zk,
        deep_insight_name: "job_for_zk".to_string(),
        zk_server: "zkhost:2181".to_string(),
        ..RunConfig::default()
    };
    let runner = run
        .to_runner_config(Some(RunnerConfig {
            is_local: false,
            discovery_type: ServiceDiscoveryType::Zk,
            ..RunnerConfig::default()
        }))
        .unwrap();

    let d = get_discovery(&runner, None).unwrap().expect("discovery");
    assert_eq!(d.kind(), "zk");
    let (job, zk) = d.zk_config().expect("zk config");
    assert_eq!(job, "job_for_zk");
    assert_eq!(zk, "zkhost:2181");
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
fn test_monolith_discovery_consul_auto_psm() {
    let rc = RunnerConfig {
        is_local: false,
        discovery_type: ServiceDiscoveryType::Consul,
        deep_insight_name: "job-uuid".to_string(),
        ..RunnerConfig::default()
    };
    let guard = monolith_discovery(&rc, None).unwrap();
    assert_eq!(guard.discovery().unwrap().kind(), "consul");
}

#[test]
fn test_monolith_discovery_guard_query_primus() {
    let tf_config = serde_json::json!({
      "cluster": {
        "chief": ["chief:2222"],
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222"]
      },
      "task": {"type": "worker", "index": 0}
    })
    .to_string();
    let rc = RunnerConfig {
        is_local: false,
        discovery_type: ServiceDiscoveryType::Primus,
        tf_config: Some(tf_config),
        ..RunnerConfig::default()
    };
    let guard = monolith_discovery(&rc, None).unwrap();
    assert_eq!(guard.kind(), Some("tf_config"));
    let ps = guard.query("ps").unwrap();
    assert_eq!(ps.get(&0).unwrap(), "ps0:2222");
    assert_eq!(ps.get(&1).unwrap(), "ps1:2222");
}

#[test]
fn test_monolith_discovery_guard_local_register_error() {
    let rc = RunnerConfig {
        is_local: true,
        ..RunnerConfig::default()
    };
    let guard = monolith_discovery(&rc, None).unwrap();
    let err = guard.register("ps", 0, "127.0.0.1:1000").unwrap_err();
    assert!(matches!(
        err,
        monolith_training::RunnerUtilsError::LocalModeNoDiscovery
    ));
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

#[test]
fn test_checkpoint_state_restore_override_non_train() {
    let tmp = tempdir().unwrap();
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        model_dir.join("checkpoint"),
        r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
    )
    .unwrap();

    let st = get_checkpoint_state_with_restore_override(
        &model_dir,
        "checkpoint",
        Some("model.ckpt-30"),
        RunnerMode::Predict,
        1,
        Duration::from_millis(1),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        Path::new(&st.model_checkpoint_path).file_name().unwrap(),
        "model.ckpt-30"
    );
}

#[test]
fn test_runner_config_restore_init_chief_path() {
    let tmp = tempdir().unwrap();
    let restore_dir = tmp.path().join("restore");
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&restore_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();
    fs::write(
        restore_dir.join("checkpoint"),
        r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
    )
    .unwrap();

    let rc = RunnerConfig {
        is_local: false,
        index: 0,
        model_dir: model_dir.clone(),
        restore_dir: Some(restore_dir),
        restore_ckpt: Some("model.ckpt-30".to_string()),
        ..RunnerConfig::default()
    };

    let st = initialize_restore_checkpoint_from_runner(
        &rc,
        Duration::from_secs(1),
        Duration::from_millis(10),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        Path::new(&st.model_checkpoint_path).file_name().unwrap(),
        "model.ckpt-30"
    );
}
