use monolith_training::native_training::service_discovery::ServiceDiscoveryType;
use monolith_training::{
    get_checkpoint_state_with_restore_override, get_discovery,
    get_discovery_from_run_config,
    initialize_restore_checkpoint_from_run_config,
    initialize_restore_checkpoint_from_run_config_defaults,
    initialize_restore_checkpoint_from_runner, monolith_discovery, prepare_restore_checkpoint,
    monolith_discovery_from_run_config, RunConfig, RunnerConfig, RunnerMode,
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
        .expect("to_runner_config should succeed for primus config");

    let d = get_discovery(&runner, None)
        .expect("get_discovery should succeed for primus config")
        .expect("discovery");
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
        .expect("get_discovery_from_run_config should succeed for primus config")
        .expect("discovery");
    assert_eq!(d.kind(), "tf_config");
}

#[test]
fn test_monolith_discovery_from_run_config_primus() {
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
    let guard = monolith_discovery_from_run_config(&run, None, None)
        .expect("monolith_discovery_from_run_config should succeed");
    assert_eq!(guard.kind(), Some("tf_config"));
    assert_eq!(
        guard
            .query("ps")
            .expect("query(ps) should succeed")
            .get(&0)
            .expect("ps index 0 should exist"),
        "ps0:2222"
    );
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
        .expect("to_runner_config should succeed for zk config");

    let d = get_discovery(&runner, None)
        .expect("get_discovery should succeed for zk config")
        .expect("discovery");
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
    let guard = monolith_discovery(&rc, None).expect("monolith_discovery should succeed");
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
    let guard = monolith_discovery(&rc, None).expect("monolith_discovery should succeed");
    assert_eq!(
        guard
            .discovery()
            .expect("discovery should be present in non-local mode")
            .kind(),
        "consul"
    );
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
    let guard = monolith_discovery(&rc, None).expect("monolith_discovery should succeed");
    assert_eq!(guard.kind(), Some("tf_config"));
    let ps = guard.query("ps").expect("query(ps) should succeed");
    assert_eq!(ps.get(&0).expect("ps index 0 should exist"), "ps0:2222");
    assert_eq!(ps.get(&1).expect("ps index 1 should exist"), "ps1:2222");
}

#[test]
fn test_monolith_discovery_guard_manual_close_is_idempotent() {
    let tf_config = serde_json::json!({
      "cluster": {
        "chief": ["chief:2222"],
        "ps": ["ps0:2222"],
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
    let mut guard = monolith_discovery(&rc, None).expect("monolith_discovery should succeed");
    assert_eq!(guard.kind(), Some("tf_config"));

    guard.close().expect("explicit discovery close should succeed");
    assert_eq!(guard.kind(), None);
    assert!(matches!(
        guard.query("ps"),
        Err(monolith_training::RunnerUtilsError::LocalModeNoDiscovery)
    ));

    // Explicit close remains idempotent for lifecycle parity with repeated teardown calls.
    guard
        .close()
        .expect("repeated discovery close should remain idempotent");
}

#[test]
fn test_monolith_discovery_guard_local_register_error() {
    let rc = RunnerConfig {
        is_local: true,
        ..RunnerConfig::default()
    };
    let guard = monolith_discovery(&rc, None).expect("monolith_discovery should succeed");
    let err = guard
        .register("ps", 0, "127.0.0.1:1000")
        .expect_err("local-mode discovery guard should reject register calls");
    assert!(matches!(
        err,
        monolith_training::RunnerUtilsError::LocalModeNoDiscovery
    ));
}

#[test]
fn test_prepare_restore_checkpoint_chief_then_worker() {
    let tmp = tempdir().expect("tempdir creation should succeed");
    let restore_dir = tmp.path().join("restore");
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&restore_dir).expect("create_dir_all(restore_dir) should succeed");
    fs::create_dir_all(&model_dir).expect("create_dir_all(model_dir) should succeed");

    let pbtxt = r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#;
    fs::write(restore_dir.join("checkpoint"), pbtxt)
        .expect("writing restore checkpoint file should succeed");

    let chief = prepare_restore_checkpoint(
        &restore_dir,
        &model_dir,
        Some("model.ckpt-30"),
        true,
        Duration::from_secs(1),
        Duration::from_millis(10),
    )
    .expect("prepare_restore_checkpoint should succeed for chief path");
    assert_eq!(
        Path::new(&chief.model_checkpoint_path)
            .file_name()
            .expect("chief model_checkpoint_path should contain filename"),
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
    .expect("prepare_restore_checkpoint should succeed for worker path");
    assert_eq!(
        Path::new(&worker.model_checkpoint_path)
            .file_name()
            .expect("worker model_checkpoint_path should contain filename"),
        "model.ckpt-30"
    );
}

#[test]
fn test_checkpoint_state_restore_override_non_train() {
    let tmp = tempdir().expect("tempdir creation should succeed");
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&model_dir).expect("create_dir_all(model_dir) should succeed");
    fs::write(
        model_dir.join("checkpoint"),
        r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
    )
    .expect("writing model_dir checkpoint file should succeed");

    let st = get_checkpoint_state_with_restore_override(
        &model_dir,
        "checkpoint",
        Some("model.ckpt-30"),
        RunnerMode::Predict,
        1,
        Duration::from_millis(1),
    )
    .expect("get_checkpoint_state_with_restore_override should succeed")
    .expect("checkpoint state should be present");
    assert_eq!(
        Path::new(&st.model_checkpoint_path)
            .file_name()
            .expect("restore override model_checkpoint_path should contain filename"),
        "model.ckpt-30"
    );
}

#[test]
fn test_runner_config_restore_init_chief_path() {
    let tmp = tempdir().expect("tempdir creation should succeed");
    let restore_dir = tmp.path().join("restore");
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&restore_dir).expect("create_dir_all(restore_dir) should succeed");
    fs::create_dir_all(&model_dir).expect("create_dir_all(model_dir) should succeed");
    fs::write(
        restore_dir.join("checkpoint"),
        r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
    )
    .expect("writing restore checkpoint file should succeed");

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
    .expect("initialize_restore_checkpoint_from_runner should succeed")
    .expect("restore checkpoint state should be present");
    assert_eq!(
        Path::new(&st.model_checkpoint_path)
            .file_name()
            .expect("runner restore model_checkpoint_path should contain filename"),
        "model.ckpt-30"
    );
}

#[test]
fn test_run_config_restore_init_chief_path() {
    let tmp = tempdir().expect("tempdir creation should succeed");
    let restore_dir = tmp.path().join("restore");
    let model_dir = tmp.path().join("model");
    fs::create_dir_all(&restore_dir).expect("create_dir_all(restore_dir) should succeed");
    fs::create_dir_all(&model_dir).expect("create_dir_all(model_dir) should succeed");
    fs::write(
        restore_dir.join("checkpoint"),
        r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
"#,
    )
    .expect("writing restore checkpoint file should succeed");

    let run = RunConfig {
        is_local: true,
        model_dir: model_dir.clone(),
        restore_dir: Some(restore_dir),
        restore_ckpt: Some("model.ckpt-30".to_string()),
        ..RunConfig::default()
    };

    let st = initialize_restore_checkpoint_from_run_config(
        &run,
        None,
        Duration::from_secs(1),
        Duration::from_millis(10),
    )
    .expect("initialize_restore_checkpoint_from_run_config should succeed")
    .expect("restore checkpoint state should be present");
    assert_eq!(
        Path::new(&st.model_checkpoint_path)
            .file_name()
            .expect("run_config restore model_checkpoint_path should contain filename"),
        "model.ckpt-30"
    );
}

#[test]
fn test_run_config_restore_init_defaults_none_when_restore_missing() {
    let run = RunConfig {
        is_local: true,
        ..RunConfig::default()
    };
    let st = initialize_restore_checkpoint_from_run_config_defaults(&run, None)
        .expect("initialize_restore_checkpoint_from_run_config_defaults should succeed");
    assert!(st.is_none());
}
