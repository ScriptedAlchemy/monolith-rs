use monolith_training::py_discovery::TfConfigServiceDiscovery;
use monolith_training::{
    copy_checkpoint_from_restore_dir, get_discovery, ConstantModelFn, EntryError, Estimator,
    RunConfig, RunnerConfig,
};
use std::fs;
use tempfile::tempdir;

#[test]
fn tf_config_discovery_matches_python_indexing() {
    // Mirrors monolith/native_training/runner_utils_test.py::test_get_discovery_primus
    let tf_config = r#"
{
  "cluster": {
    "ps": ["localhost:1111", "localhost:1112"],
    "worker": ["localhost:1113", "localhost:1114"],
    "chief": ["localhost:1115"]
  },
  "task": {"type":"chief", "index":0}
}
"#;
    let d = TfConfigServiceDiscovery::new(tf_config).unwrap();
    assert_eq!(d.server_type(), "worker");
    assert_eq!(d.index(), 0);
    assert_eq!(d.addr().unwrap(), "localhost:1115");

    let tf_config_worker0 = r#"
{
  "cluster": {
    "ps": ["localhost:1111", "localhost:1112"],
    "worker": ["localhost:1113", "localhost:1114"],
    "chief": ["localhost:1115"]
  },
  "task": {"type":"worker", "index":0}
}
"#;
    let d = TfConfigServiceDiscovery::new(tf_config_worker0).unwrap();
    assert_eq!(d.server_type(), "worker");
    // Worker indices shift by +1 when chief exists.
    assert_eq!(d.index(), 1);
    assert_eq!(d.addr().unwrap(), "localhost:1113");
}

#[test]
fn runner_utils_copy_ckpt_creates_expected_files() {
    // Mirrors monolith/native_training/runner_utils_test.py::test_copy_ckpt
    let tmp = tempdir().unwrap();
    let restore_dir = tmp.path().join("restore_dir");
    let model_dir = tmp.path().join("model_dir");
    fs::create_dir_all(&restore_dir).unwrap();
    fs::create_dir_all(&model_dir).unwrap();

    let pbtxt = r#"
model_checkpoint_path: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-61"
all_model_checkpoint_paths: "model.ckpt-30"
all_model_checkpoint_paths: "model.ckpt-0"
"#;
    fs::write(restore_dir.join("checkpoint"), pbtxt).unwrap();

    let st =
        copy_checkpoint_from_restore_dir(&restore_dir, &model_dir, Some("model.ckpt-30")).unwrap();
    assert!(model_dir.join("monolith_checkpoint").exists());
    assert!(model_dir.join("restore_ckpt").exists());
    assert!(model_dir.join("checkpoint").exists());
    assert!(
        st.model_checkpoint_path.ends_with("model.ckpt-30"),
        "picked {}",
        st.model_checkpoint_path
    );
}

#[test]
fn entry_batch_softmax_initializer_errors_like_python() {
    // Python raises ValueError when init_step_interval < 1.
    let err = monolith_training::BatchSoftmaxInitializer::new(0.9).unwrap_err();
    match err {
        EntryError::InvalidInitStepInterval(v) => assert!((v - 0.9).abs() < 1e-6),
        other => panic!("unexpected error: {other:?}"),
    }
}

#[tokio::test]
async fn distributed_runner_in_memory_ps_and_worker() {
    // Mirrors "distributed" smoke tests: can start PS + worker, discover, lookup, barrier, apply.
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed, DistributedRunConfig, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());

    let ps_cfg = DistributedRunConfig {
        role: Role::Ps,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        ..Default::default()
    };

    let worker_cfg = DistributedRunConfig {
        role: Role::Worker,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        bind_addr: "127.0.0.1:0".parse().unwrap(),
        connect_retries: 50,
        retry_backoff_ms: 20,
        ..Default::default()
    };

    let ps_task = tokio::spawn(run_distributed(Arc::clone(&discovery), ps_cfg));
    // Give PS time to bind and re-register actual port.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let worker_res = run_distributed(Arc::clone(&discovery), worker_cfg).await;
    assert!(worker_res.is_ok(), "worker failed: {worker_res:?}");

    // PS runs forever; abort for test shutdown.
    ps_task.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_smoke() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use monolith_training::RunnerConfig;
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let ps_rc = RunnerConfig {
        index: 0,
        num_ps: 1,
        num_workers: 1,
        ..RunnerConfig::default()
    };
    let worker_rc = RunnerConfig {
        index: 0,
        num_ps: 1,
        num_workers: 1,
        ..RunnerConfig::default()
    };

    let discovery_bg = Arc::clone(&discovery);
    let ps_rc_bg = ps_rc.clone();
    let ps_task = tokio::spawn(async move {
        run_distributed_from_runner_config(
            discovery_bg,
            &ps_rc_bg,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        )
        .await
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let worker_res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &worker_rc,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    assert!(worker_res.is_ok(), "worker failed: {worker_res:?}");
    ps_task.abort();
}

#[test]
fn estimator_from_run_config_roundtrip() {
    let run = RunConfig {
        model_dir: std::path::PathBuf::from("/tmp/parity_estimator"),
        log_step_count_steps: 15,
        restore_ckpt: Some("model.ckpt-22".to_string()),
        ..RunConfig::default()
    };
    let estimator = Estimator::from_run_config(&run, None, ConstantModelFn::new(0.1)).unwrap();
    assert_eq!(
        estimator.config().model_dir,
        std::path::PathBuf::from("/tmp/parity_estimator")
    );
    assert_eq!(estimator.config().log_step_count_steps, 15);
    assert_eq!(
        estimator.config().warm_start_from,
        Some(std::path::PathBuf::from("model.ckpt-22"))
    );
}

#[test]
fn runner_discovery_query_primus_roundtrip() {
    let tf_config = serde_json::json!({
      "cluster": {
        "chief": ["chief:2222"],
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222"]
      },
      "task": {"type": "worker", "index": 0}
    })
    .to_string();
    let runner = RunnerConfig {
        is_local: false,
        discovery_type: monolith_training::native_training::service_discovery::ServiceDiscoveryType::Primus,
        tf_config: Some(tf_config),
        ..RunnerConfig::default()
    };
    let d = get_discovery(&runner, None).unwrap().expect("discovery");
    let ps = d.query("ps").unwrap();
    assert_eq!(ps.get(&0).unwrap(), "ps0:2222");
    assert_eq!(ps.get(&1).unwrap(), "ps1:2222");
}
