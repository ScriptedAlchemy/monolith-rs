use monolith_training::py_discovery::TfConfigServiceDiscovery;
use monolith_training::discovery::{DiscoveryEvent, ServiceDiscoveryAsync, ServiceInfo};
use monolith_training::{
    copy_checkpoint_from_restore_dir, get_discovery, ConstantModelFn, EntryError, Estimator,
    RunConfig, RunnerConfig,
};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
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

#[tokio::test]
async fn distributed_runner_from_run_config_smoke() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        ..RunConfig::default()
    };

    let discovery_bg = Arc::clone(&discovery);
    let run_bg = run.clone();
    let ps_task = tokio::spawn(async move {
        run_distributed_from_run_config(
            discovery_bg,
            &run_bg,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        )
        .await
    });

    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    let worker_res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    assert!(worker_res.is_ok(), "worker failed: {worker_res:?}");
    ps_task.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_barrier_timeout_controls() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 2,
        barrier_timeout_ms: 40,
        connect_retries: 50,
        retry_backoff_ms: 10,
        ..RunConfig::default()
    };

    let discovery_bg = Arc::clone(&discovery);
    let run_bg = run.clone();
    let ps_task = tokio::spawn(async move {
        run_distributed_from_run_config(
            discovery_bg,
            &run_bg,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;
    let started = Instant::now();
    let worker_res = tokio::time::timeout(
        Duration::from_millis(1500),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    ps_task.abort();

    assert!(
        worker_res.is_ok(),
        "worker run should return promptly when barrier timeout is configured"
    );
    let elapsed = started.elapsed();
    let msg = worker_res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Barrier timeout"),
        "worker should fail with barrier timeout when only one of two workers runs: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(700),
        "barrier_timeout_ms from RunConfig should bound worker barrier wait duration (elapsed: {:?})",
        elapsed
    );
}

struct HangingDiscoverFromRunConfigDiscovery {
    connect_count: AtomicUsize,
    disconnect_count: AtomicUsize,
    discover_count: AtomicUsize,
    deregister_count: AtomicUsize,
}

impl HangingDiscoverFromRunConfigDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
            discover_count: AtomicUsize::new(0),
            deregister_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for HangingDiscoverFromRunConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        Ok(())
    }

    async fn discover_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        self.discover_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        self.deregister_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn distributed_runner_from_run_config_honors_discover_timeout_controls() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingDiscoverFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when discover blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: discover worker-0 for ps after 20ms"),
        "run-config timeout controls should propagate into discover timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_discover_service_type_into_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingDiscoverFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when discover blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery operation: discover worker-0 for parameter_server_custom after 20ms"
        ),
        "custom run-config ps service type should propagate into discover timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_discover_retry_controls() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingDiscoverFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 2,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(1000),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when discover repeatedly times out"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: discover worker-0 for ps after 20ms"),
        "discover timeout diagnostics should include configured operation timeout: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        discovery.discover_count.load(Ordering::SeqCst),
        3,
        "connect_retries=2 should yield exactly 3 discover attempts"
    );
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_retry_backoff_controls() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(EmptyDiscoverFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 2,
        retry_backoff_ms: 40,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(1200),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when PS list remains empty"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "empty-discover retry path should fail with PS discovery timeout: {msg}"
    );
    assert!(
        elapsed >= Duration::from_millis(60),
        "retry_backoff_ms from RunConfig should be reflected in elapsed retry delay (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 3);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_worker_discovery_error_when_cleanup_times_out(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(EmptyDiscoverWithHangingCleanupFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when worker discovery fails and cleanup steps block"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "worker discovery timeout should remain primary over cleanup timeout failures when configured via RunConfig: {msg}"
    );
    assert!(
        msg.contains("service type: ps"),
        "worker discovery timeout diagnostics should include default PS service-type context when invoked from RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("for worker-0"),
        "worker discovery timeout diagnostics should include worker service-id context when invoked from RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "worker discovery timeout errors should include cleanup issue context when cleanup also times out via RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
        "run-config cleanup issue context should include deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "run-config cleanup issue context should include disconnect timeout diagnostics: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(260),
        "cleanup timeout from RunConfig should bound blocked worker-cleanup duration after discovery failure (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_custom_discover_service_type_into_worker_discovery_error_when_cleanup_times_out(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(EmptyDiscoverWithHangingCleanupFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when worker discovery fails and cleanup steps block with custom discover service type"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "worker discovery timeout should remain primary over cleanup timeout failures for custom discover service type via RunConfig: {msg}"
    );
    assert!(
        msg.contains("service type: parameter_server_custom"),
        "worker discovery timeout diagnostics should include configured custom PS service type via RunConfig: {msg}"
    );
    assert!(
        msg.contains("for worker-0"),
        "worker discovery timeout diagnostics should include worker service-id context for custom PS service type via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "worker discovery timeout errors should include cleanup issue context for custom PS service type via RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
        "run-config cleanup issue context with custom PS service type should include deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "run-config cleanup issue context with custom PS service type should include disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_worker_index_into_ps_discovery_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(EmptyDiscoverFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 3,
        num_ps: 1,
        num_workers: 4,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when PS discovery remains empty for worker index propagation diagnostics"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "worker discovery should fail with discovery timeout when no PS endpoints are returned: {msg}"
    );
    assert!(
        msg.contains("for worker-3"),
        "worker index from RunConfig should propagate into worker discovery timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_zero_operation_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 0,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_operation_timeout > 0"),
        "zero run-config operation timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_zero_cleanup_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_cleanup_timeout_ms: 0,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_cleanup_timeout > 0"),
        "zero run-config cleanup timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_zero_barrier_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        barrier_timeout_ms: 0,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires barrier_timeout_ms > 0"),
        "zero run-config barrier timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_negative_barrier_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        barrier_timeout_ms: -1,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires barrier_timeout_ms > 0"),
        "negative run-config barrier timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_zero_num_ps() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 0,
        num_workers: 1,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires num_ps > 0"),
        "zero run-config num_ps should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_zero_num_workers() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 0,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires num_workers > 0"),
        "zero run-config num_workers should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_zero_dim() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        dim: 0,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires dim > 0"),
        "zero run-config dim should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_worker_index_out_of_range() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 2,
        num_ps: 1,
        num_workers: 2,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires index < num_workers for worker role"),
        "out-of-range run-config worker index should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_ps_index_out_of_range() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 1,
        num_ps: 1,
        num_workers: 1,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires index < num_ps for ps role"),
        "out-of-range run-config ps index should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_empty_ps_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "   ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty discovery_service_type_ps"),
        "empty run-config ps service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_whitespace_padded_ps_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: " ps ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires discovery_service_type_ps without leading/trailing whitespace"
        ),
        "whitespace-padded run-config ps service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_internal_whitespace_ps_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "ps cluster".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_service_type_ps without whitespace characters"),
        "internal-whitespace run-config ps service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_empty_worker_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty discovery_service_type_worker"),
        "empty run-config worker service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_whitespace_padded_worker_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: " worker ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires discovery_service_type_worker without leading/trailing whitespace"
        ),
        "whitespace-padded run-config worker service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_internal_whitespace_worker_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "worker cluster".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_service_type_worker without whitespace characters"),
        "internal-whitespace run-config worker service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_identical_ps_and_worker_service_types() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "service".to_string(),
        discovery_service_type_worker: "service".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires distinct discovery_service_type_ps and discovery_service_type_worker"
        ),
        "identical run-config discovery service types should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_case_insensitive_identical_ps_and_worker_service_types(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "Service".to_string(),
        discovery_service_type_worker: "service".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires distinct discovery_service_type_ps and discovery_service_type_worker"
        ),
        "case-insensitive identical run-config discovery service types should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_empty_table_name() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        table_name: " ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty table_name"),
        "empty run-config table name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_whitespace_padded_table_name() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        table_name: " emb ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires table_name without leading/trailing whitespace"),
        "whitespace-padded run-config table name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_internal_whitespace_table_name() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        table_name: "my table".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires table_name without whitespace characters"),
        "internal-whitespace run-config table name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_zero_parameter_sync_interval_with_targets() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_interval_ms: 0,
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_interval > 0 when parameter_sync_targets are configured"
        ),
        "zero run-config parameter-sync interval with targets should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_empty_parameter_sync_target_entry() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![" ".to_string()],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty parameter_sync_targets entries"),
        "empty run-config parameter-sync target entry should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_accepts_case_insensitive_http_scheme_parameter_sync_target(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(EmptyDiscoverFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["HtTp://127.0.0.1:8500".to_string()],
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang while validating case-insensitive parameter-sync target schemes"
    );
    let err = res.unwrap().unwrap_err().to_string();
    assert!(
        err.contains("Timed out waiting for PS discovery"),
        "case-insensitive parameter-sync target scheme should pass config validation and reach worker discovery path: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_whitespace_padded_parameter_sync_target_entry(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![" 127.0.0.1:8500 ".to_string()],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_targets entries without leading/trailing whitespace"
        ),
        "whitespace-padded run-config parameter-sync target entry should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_invalid_parameter_sync_target_endpoint() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["http://".to_string()],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config has invalid parameter_sync_targets entry `http://`"),
        "invalid run-config parameter-sync target endpoint should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_parameter_sync_target_endpoint_with_path_or_query(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["http://127.0.0.1:8500/v1?foo=bar".to_string()],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("endpoint must not include a URL path or query"),
        "run-config parameter-sync target with URL path/query should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_parameter_sync_target_endpoint_with_unsupported_scheme(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["ftp://127.0.0.1:8500".to_string()],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("endpoint scheme must be http or https"),
        "run-config parameter-sync target with unsupported scheme should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_parameter_sync_target_endpoint_with_userinfo()
{
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["http://user@127.0.0.1:8500".to_string()],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("endpoint must not include userinfo"),
        "run-config parameter-sync target with userinfo should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1:8500".to_string(),
            "127.0.0.1:8500".to_string(),
        ],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate run-config parameter-sync target entries should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry_after_http_prefix_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1:8500".to_string(),
            "http://127.0.0.1:8500".to_string(),
        ],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate run-config parameter-sync target entries after http-prefix normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry_after_trailing_slash_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1:8500".to_string(),
            "http://127.0.0.1:8500/".to_string(),
        ],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate run-config parameter-sync target entries after trailing-slash normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry_after_http_default_port_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1".to_string(),
            "http://127.0.0.1:80".to_string(),
        ],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate run-config parameter-sync target entries after http default-port normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry_after_https_default_port_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "https://127.0.0.1".to_string(),
            "https://127.0.0.1:443".to_string(),
        ],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate run-config parameter-sync target entries after https default-port normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_duplicate_parameter_sync_target_entry_after_case_insensitive_http_prefix_and_host_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "EXAMPLE.com:8500".to_string(),
            "HtTp://example.COM:8500".to_string(),
        ],
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate run-config parameter-sync target entries after case-insensitive http-prefix and host normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_empty_parameter_sync_model_name_with_targets() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_model_name: "".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires non-empty parameter_sync_model_name when parameter_sync_targets are configured"
        ),
        "empty run-config parameter-sync model name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_whitespace_padded_parameter_sync_model_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_model_name: " model ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_model_name without leading/trailing whitespace when parameter_sync_targets are configured"
        ),
        "whitespace-padded run-config parameter-sync model name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_internal_whitespace_parameter_sync_model_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_model_name: "my model".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_model_name without whitespace characters when parameter_sync_targets are configured"
        ),
        "internal-whitespace run-config parameter-sync model name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_empty_parameter_sync_signature_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_signature_name: "  ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires non-empty parameter_sync_signature_name when parameter_sync_targets are configured"
        ),
        "empty run-config parameter-sync signature name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_whitespace_padded_parameter_sync_signature_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_signature_name: " signature ".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_signature_name without leading/trailing whitespace when parameter_sync_targets are configured"
        ),
        "whitespace-padded run-config parameter-sync signature name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_rejects_internal_whitespace_parameter_sync_signature_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_signature_name: "serving default".to_string(),
        ..RunConfig::default()
    };

    let err = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_signature_name without whitespace characters when parameter_sync_targets are configured"
        ),
        "internal-whitespace run-config parameter-sync signature name should be rejected by distributed config validation: {err}"
    );
}


#[tokio::test]
async fn distributed_runner_from_run_config_propagates_custom_service_type_fields() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(RecordingServiceTypePropagationDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let worker_res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let worker_msg = worker_res.unwrap_err().to_string();
    assert!(
        worker_msg.contains("Timed out waiting for PS discovery"),
        "worker run should fail in discover loop with empty discovery backend: {worker_msg}"
    );

    discovery.set_fail_register(true);
    let ps_res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let ps_msg = ps_res.unwrap_err().to_string();
    assert!(
        ps_msg.contains("forced register failure"),
        "ps run should fail via forced register error to avoid server startup in service-type propagation test: {ps_msg}"
    );

    let registered = discovery.registered_types_snapshot();
    assert!(
        registered.contains(&"trainer_custom".to_string()),
        "worker should register with custom worker service type: {registered:?}"
    );
    assert!(
        registered.contains(&"parameter_server_custom".to_string()),
        "ps should register with custom ps service type: {registered:?}"
    );
    let discovered = discovery.discovered_types_snapshot();
    assert_eq!(
        discovered,
        vec!["parameter_server_custom".to_string()],
        "worker discovery should query custom ps discovery service type"
    );
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_deregister_timeout_with_custom_service_type_after_success(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when deregister cleanup blocks after successful ps run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
        ),
        "deregister timeout diagnostics should include custom worker service type from RunConfig after successful worker run: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "deregister-timeout diagnostics after successful worker run should include cleanup issue context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_disconnect_timeout_with_custom_service_type_after_success(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when disconnect cleanup blocks after successful ps run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
        ),
        "disconnect timeout diagnostics should include custom worker service type from RunConfig after successful worker run: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "disconnect-timeout diagnostics after successful worker run should include cleanup issue context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_deregister_timeout_after_success() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when default-service-type deregister cleanup blocks after successful worker run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"),
        "default-service-type deregister timeout diagnostics should be preserved after successful worker run via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type deregister timeout diagnostics should include successful-role cleanup issue context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_disconnect_timeout_after_success() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when default-service-type disconnect cleanup blocks after successful worker run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"),
        "default-service-type disconnect timeout diagnostics should be preserved after successful worker run via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type disconnect timeout diagnostics should include successful-role cleanup issue context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_deregister_failure_after_success() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced deregister failure"),
        "default-service-type deregister failure should be preserved after successful worker run via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type deregister failure should include successful-role cleanup issue context via RunConfig: {msg}"
    );
    assert!(
        msg.contains("deregister worker-0 from worker"),
        "default-service-type deregister failure should include cleanup operation context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_disconnect_failure_after_success() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced disconnect failure"),
        "default-service-type disconnect failure should be preserved after successful worker run via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type disconnect failure should include successful-role cleanup issue context via RunConfig: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via worker"),
        "default-service-type disconnect failure should include cleanup operation context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_deregister_failure_with_disconnect_failure_context_after_success(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        true,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("deregister worker-0 from worker") && msg.contains("forced deregister failure"),
        "run-config post-success both-failure diagnostics should preserve deregister failure with operation context: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "run-config post-success both-failure diagnostics should include cleanup issue context: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via worker") && msg.contains("forced disconnect failure"),
        "run-config post-success both-failure diagnostics should include disconnect failure with operation context: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_custom_worker_deregister_failure_after_success(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced deregister failure"),
        "custom-worker deregister failure should be preserved after successful worker run via RunConfig: {msg}"
    );
    assert!(
        msg.contains("deregister worker-0 from trainer_custom"),
        "custom-worker deregister failure should include cleanup operation context via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "custom-worker deregister failure should include successful-role cleanup issue context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_surfaces_custom_worker_disconnect_failure_after_success(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced disconnect failure"),
        "custom-worker disconnect failure should be preserved after successful worker run via RunConfig: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via trainer_custom"),
        "custom-worker disconnect failure should include cleanup operation context via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "custom-worker disconnect failure should include successful-role cleanup issue context via RunConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_deregister_timeout_with_disconnect_timeout_context_after_success(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        true,
    ));
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when both cleanup steps block after successful worker run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
        ),
        "deregister timeout should remain primary in run-config post-success cleanup path when both cleanup steps block: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "run-config post-success cleanup path should append cleanup issue context when both cleanup steps block: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
        ),
        "run-config post-success cleanup issue context should include disconnect timeout diagnostics when both cleanup steps block: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

struct HangingConnectAndCleanupFromRunConfigDiscovery {
    connect_count: AtomicUsize,
    disconnect_count: AtomicUsize,
}

impl HangingConnectAndCleanupFromRunConfigDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for HangingConnectAndCleanupFromRunConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        Ok(())
    }

    async fn discover_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        Ok(())
    }
}

struct FailingConnectWithHangingDisconnectFromConfigDiscovery {
    connect_count: AtomicUsize,
    disconnect_count: AtomicUsize,
}

impl FailingConnectWithHangingDisconnectFromConfigDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for FailingConnectWithHangingDisconnectFromConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Err(monolith_training::discovery::DiscoveryError::ConnectionFailed(
            "forced connect failure".to_string(),
        ))
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        Ok(())
    }

    async fn discover_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        Ok(())
    }
}

struct FailingConnectWithFailingDisconnectFromConfigDiscovery {
    connect_count: AtomicUsize,
    disconnect_count: AtomicUsize,
}

impl FailingConnectWithFailingDisconnectFromConfigDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for FailingConnectWithFailingDisconnectFromConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Err(monolith_training::discovery::DiscoveryError::ConnectionFailed(
            "forced connect failure".to_string(),
        ))
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        Err(monolith_training::discovery::DiscoveryError::Internal(
            "forced disconnect failure".to_string(),
        ))
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        Ok(())
    }

    async fn discover_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_connect_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-0 via worker after 20ms"),
        "connect timeout should remain primary over cleanup timeout when configured via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "connect-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "connect-timeout cleanup context via RunConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_connect_failure_with_cleanup_timeout_context()
{
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithHangingDisconnectFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when connect fails and disconnect cleanup blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "connect failure should remain primary when cleanup disconnect blocks via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "connect-failure diagnostics via RunConfig should include cleanup issue context when cleanup disconnect blocks: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
        ),
        "connect-failure cleanup context via RunConfig should include disconnect-timeout diagnostics with custom worker service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_connect_failure_with_disconnect_failure_context(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithFailingDisconnectFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "connect failure should remain primary when cleanup disconnect fails via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "connect-failure diagnostics via RunConfig should include cleanup issue context when cleanup disconnect fails: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via trainer_custom")
            && msg.contains("forced disconnect failure"),
        "connect-failure cleanup context via RunConfig should include disconnect-failure diagnostics with custom service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_cleanup_timeout_context(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithHangingDisconnectFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when default-ps connect fails and disconnect cleanup blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "default-ps connect failure should remain primary when cleanup disconnect blocks via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "default-ps connect-failure diagnostics via RunConfig should include cleanup issue context when cleanup disconnect blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps after 20ms"),
        "default-ps connect-failure cleanup context via RunConfig should include default-service-type disconnect-timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_default_ps_connect_failure_with_disconnect_failure_context(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithFailingDisconnectFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "default-ps connect failure should remain primary when cleanup disconnect fails via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "default-ps connect-failure diagnostics via RunConfig should include cleanup issue context when cleanup disconnect fails: {msg}"
    );
    assert!(
        msg.contains("disconnect ps-0 via ps") && msg.contains("forced disconnect failure"),
        "default-ps connect-failure cleanup context via RunConfig should include default-service-type disconnect-failure diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_ps_connect_failure_with_cleanup_timeout_context(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithHangingDisconnectFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when ps connect fails and disconnect cleanup blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "ps connect failure should remain primary when cleanup disconnect blocks via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps connect-failure diagnostics via RunConfig should include cleanup issue context when cleanup disconnect blocks: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom after 20ms"
        ),
        "ps connect-failure cleanup context via RunConfig should include disconnect-timeout diagnostics with custom ps service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_ps_connect_failure_with_disconnect_failure_context(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithFailingDisconnectFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = run_distributed_from_run_config(
        Arc::clone(&discovery),
        &run,
        None,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "ps connect failure should remain primary when cleanup disconnect fails via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps connect-failure diagnostics via RunConfig should include cleanup issue context when cleanup disconnect fails: {msg}"
    );
    assert!(
        msg.contains("disconnect ps-0 via parameter_server_custom")
            && msg.contains("forced disconnect failure"),
        "ps connect-failure cleanup context via RunConfig should include disconnect-failure diagnostics with custom ps service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_worker_index_into_connect_timeout_diagnostics() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 3,
        num_ps: 1,
        num_workers: 4,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-3 via worker after 20ms"),
        "worker index from RunConfig should propagate into timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "worker-index connect-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-3 via worker"),
        "worker-index connect-timeout cleanup context via RunConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_worker_service_type_into_connect_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-0 via trainer_custom after 20ms"),
        "worker service type from RunConfig should propagate into connect-timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom worker connect-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via trainer_custom"),
        "custom worker connect-timeout cleanup context via RunConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_ps_connect_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when ps connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect ps-0 via ps after 20ms"),
        "ps connect timeout should remain primary over cleanup timeout when configured via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps connect-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps"),
        "ps connect-timeout cleanup context via RunConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_ps_index_into_connect_timeout_diagnostics() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 2,
        num_ps: 3,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when ps connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect ps-2 via ps after 20ms"),
        "ps index from RunConfig should propagate into timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps-index connect-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-2 via ps"),
        "ps-index connect-timeout cleanup context via RunConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_ps_service_type_into_connect_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when ps connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect ps-0 via parameter_server_custom after 20ms"),
        "ps service type from RunConfig should propagate into connect-timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom ps connect-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom"
        ),
        "custom ps connect-timeout cleanup context via RunConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_honors_cleanup_timeout_with_blocked_cleanup() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 10,
        discovery_cleanup_timeout_ms: 10,
        ..RunConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when connect and cleanup are blocked"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-0 via worker after 10ms"),
        "operation timeout diagnostics should include configured operation timeout: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "cleanup-timeout bounded connect path should include cleanup issue context via RunConfig: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 10ms"),
        "cleanup-timeout bounded connect path via RunConfig should include disconnect timeout diagnostics with configured cleanup timeout: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(150),
        "cleanup timeout from RunConfig should bound blocked cleanup duration (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

struct HangingRegisterAndCleanupFromConfigDiscovery {
    connect_count: AtomicUsize,
    register_count: AtomicUsize,
    disconnect_count: AtomicUsize,
    deregister_count: AtomicUsize,
}

impl HangingRegisterAndCleanupFromConfigDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            register_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
            deregister_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for HangingRegisterAndCleanupFromConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        self.register_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }

    async fn discover_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        self.deregister_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }
}

struct EmptyDiscoverFromConfigDiscovery {
    connect_count: AtomicUsize,
    discover_count: AtomicUsize,
    disconnect_count: AtomicUsize,
    deregister_count: AtomicUsize,
}

impl EmptyDiscoverFromConfigDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            discover_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
            deregister_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for EmptyDiscoverFromConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        Ok(())
    }

    async fn discover_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        self.discover_count.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        self.deregister_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct EmptyDiscoverWithHangingCleanupFromConfigDiscovery {
    connect_count: AtomicUsize,
    register_count: AtomicUsize,
    discover_count: AtomicUsize,
    disconnect_count: AtomicUsize,
    deregister_count: AtomicUsize,
}

impl EmptyDiscoverWithHangingCleanupFromConfigDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            register_count: AtomicUsize::new(0),
            discover_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
            deregister_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for EmptyDiscoverWithHangingCleanupFromConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        self.register_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn discover_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        self.discover_count.fetch_add(1, Ordering::SeqCst);
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        self.deregister_count.fetch_add(1, Ordering::SeqCst);
        std::future::pending::<()>().await;
        #[allow(unreachable_code)]
        Ok(())
    }
}

struct RecordingServiceTypePropagationDiscovery {
    connect_count: AtomicUsize,
    register_count: AtomicUsize,
    discover_count: AtomicUsize,
    disconnect_count: AtomicUsize,
    deregister_count: AtomicUsize,
    fail_register: AtomicBool,
    registered_types: Mutex<Vec<String>>,
    discovered_types: Mutex<Vec<String>>,
}

impl RecordingServiceTypePropagationDiscovery {
    fn new() -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            register_count: AtomicUsize::new(0),
            discover_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
            deregister_count: AtomicUsize::new(0),
            fail_register: AtomicBool::new(false),
            registered_types: Mutex::new(Vec::new()),
            discovered_types: Mutex::new(Vec::new()),
        }
    }

    fn set_fail_register(&self, should_fail: bool) {
        self.fail_register.store(should_fail, Ordering::SeqCst);
    }

    fn registered_types_snapshot(&self) -> Vec<String> {
        self.registered_types.lock().unwrap().clone()
    }

    fn discovered_types_snapshot(&self) -> Vec<String> {
        self.discovered_types.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for RecordingServiceTypePropagationDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn register_async(
        &self,
        service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        self.register_count.fetch_add(1, Ordering::SeqCst);
        self.registered_types
            .lock()
            .unwrap()
            .push(service.service_type.clone());
        if self.fail_register.load(Ordering::SeqCst) {
            return Err(monolith_training::discovery::DiscoveryError::Internal(
                "forced register failure".to_string(),
            ));
        }
        Ok(())
    }

    async fn discover_async(
        &self,
        service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        self.discover_count.fetch_add(1, Ordering::SeqCst);
        self.discovered_types
            .lock()
            .unwrap()
            .push(service_type.to_string());
        Ok(Vec::new())
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        self.deregister_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

struct HangingCleanupAfterSuccessFromConfigDiscovery {
    connect_count: AtomicUsize,
    register_count: AtomicUsize,
    discover_count: AtomicUsize,
    disconnect_count: AtomicUsize,
    deregister_count: AtomicUsize,
    ps_addr: String,
    hang_deregister: bool,
    hang_disconnect: bool,
}

impl HangingCleanupAfterSuccessFromConfigDiscovery {
    fn new(ps_addr: String, hang_deregister: bool, hang_disconnect: bool) -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            register_count: AtomicUsize::new(0),
            discover_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
            deregister_count: AtomicUsize::new(0),
            ps_addr,
            hang_deregister,
            hang_disconnect,
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for HangingCleanupAfterSuccessFromConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        if self.hang_disconnect {
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            return Ok(());
        }
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        self.register_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn discover_async(
        &self,
        service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        self.discover_count.fetch_add(1, Ordering::SeqCst);
        let ps_addr: std::net::SocketAddr = self.ps_addr.parse().map_err(|e| {
            monolith_training::discovery::DiscoveryError::Internal(format!(
                "invalid ps_addr in HangingCleanupAfterSuccessFromConfigDiscovery: {e}"
            ))
        })?;
        let mut service = ServiceInfo::new(
            "ps-0".to_string(),
            "ps-0".to_string(),
            service_type.to_string(),
            ps_addr.ip().to_string(),
            ps_addr.port(),
        );
        service = service.with_metadata("addr", self.ps_addr.clone());
        service = service.with_metadata("index", "0");
        Ok(vec![service])
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        self.deregister_count.fetch_add(1, Ordering::SeqCst);
        if self.hang_deregister {
            std::future::pending::<()>().await;
            #[allow(unreachable_code)]
            return Ok(());
        }
        Ok(())
    }
}

struct FailingCleanupAfterSuccessFromConfigDiscovery {
    connect_count: AtomicUsize,
    register_count: AtomicUsize,
    discover_count: AtomicUsize,
    disconnect_count: AtomicUsize,
    deregister_count: AtomicUsize,
    ps_addr: String,
    fail_deregister: bool,
    fail_disconnect: bool,
}

impl FailingCleanupAfterSuccessFromConfigDiscovery {
    fn new(ps_addr: String, fail_deregister: bool, fail_disconnect: bool) -> Self {
        Self {
            connect_count: AtomicUsize::new(0),
            register_count: AtomicUsize::new(0),
            discover_count: AtomicUsize::new(0),
            disconnect_count: AtomicUsize::new(0),
            deregister_count: AtomicUsize::new(0),
            ps_addr,
            fail_deregister,
            fail_disconnect,
        }
    }
}

#[async_trait::async_trait]
impl ServiceDiscoveryAsync for FailingCleanupAfterSuccessFromConfigDiscovery {
    async fn connect(&self) -> monolith_training::discovery::Result<()> {
        self.connect_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn disconnect(&self) -> monolith_training::discovery::Result<()> {
        self.disconnect_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_disconnect {
            return Err(monolith_training::discovery::DiscoveryError::Internal(
                "forced disconnect failure".to_string(),
            ));
        }
        Ok(())
    }

    async fn register_async(
        &self,
        _service: ServiceInfo,
    ) -> monolith_training::discovery::Result<()> {
        self.register_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    async fn discover_async(
        &self,
        service_type: &str,
    ) -> monolith_training::discovery::Result<Vec<ServiceInfo>> {
        self.discover_count.fetch_add(1, Ordering::SeqCst);
        let ps_addr: std::net::SocketAddr = self.ps_addr.parse().map_err(|e| {
            monolith_training::discovery::DiscoveryError::Internal(format!(
                "invalid ps_addr in FailingCleanupAfterSuccessFromConfigDiscovery: {e}"
            ))
        })?;
        let mut service = ServiceInfo::new(
            "ps-0".to_string(),
            "ps-0".to_string(),
            service_type.to_string(),
            ps_addr.ip().to_string(),
            ps_addr.port(),
        );
        service = service.with_metadata("addr", self.ps_addr.clone());
        service = service.with_metadata("index", "0");
        Ok(vec![service])
    }

    async fn watch_async(
        &self,
        _service_type: &str,
    ) -> monolith_training::discovery::Result<tokio::sync::broadcast::Receiver<DiscoveryEvent>>
    {
        let (_tx, rx) = tokio::sync::broadcast::channel(1);
        Ok(rx)
    }

    async fn deregister_async(&self, _service_id: &str) -> monolith_training::discovery::Result<()> {
        self.deregister_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_deregister {
            return Err(monolith_training::discovery::DiscoveryError::Internal(
                "forced deregister failure".to_string(),
            ));
        }
        Ok(())
    }
}

async fn spawn_worker_success_ps_server(
    dim: usize,
) -> (
    tokio::task::JoinHandle<std::result::Result<(), tonic::transport::Error>>,
    std::net::SocketAddr,
) {
    let ps = monolith_training::distributed_ps::PsServer::new(0, dim);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral listener for worker-success parity tests");
    let actual_addr = listener
        .local_addr()
        .expect("resolve local addr for worker-success parity tests");
    let ps_server = tokio::spawn(
        tonic::transport::Server::builder()
            .add_service(monolith_training::distributed_ps::PsServer::into_service(ps.clone()))
            .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener)),
    );
    (ps_server, actual_addr)
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_register_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register worker-0 as worker after 20ms"),
        "register timeout should remain primary over cleanup timeout when configured via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "register timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
        "register-timeout cleanup context via RunConfig should include worker deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "register-timeout cleanup context via RunConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "register-timeout errors should include cleanup issue context via RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
        "run-config register-timeout cleanup issue context should include worker deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "run-config register-timeout cleanup issue context should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_worker_service_type_into_register_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when worker register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register worker-0 as trainer_custom after 20ms"),
        "custom worker service type from RunConfig should appear in timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom worker register-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from trainer_custom"),
        "custom worker register-timeout cleanup context via RunConfig should include worker deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via trainer_custom"),
        "custom worker register-timeout cleanup context via RunConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom worker register-timeout errors should include cleanup issue context via RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from trainer_custom"),
        "run-config custom worker register-timeout cleanup issue context should include deregister timeout diagnostics with custom worker service type: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via trainer_custom"),
        "run-config custom worker register-timeout cleanup issue context should include disconnect timeout diagnostics with custom worker service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_honors_cleanup_timeout_after_register_timeout() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 10,
        discovery_cleanup_timeout_ms: 10,
        ..RunConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when register and cleanup are blocked"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register worker-0 as worker after 10ms"),
        "operation timeout diagnostics should include configured operation timeout: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "cleanup timeout bounding path should still include cleanup issue context via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "register-timeout errors should include cleanup issue context for reduced cleanup timeout via RunConfig entrypoint: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(150),
        "cleanup timeout from RunConfig should bound blocked register-cleanup duration (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_preserves_ps_register_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when ps register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register ps-0 as ps after 20ms"),
        "ps register timeout should remain primary over cleanup timeout when configured via RunConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps register-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister ps-0 from ps"),
        "ps register-timeout cleanup context via RunConfig should include ps deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps"),
        "ps register-timeout cleanup context via RunConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps register-timeout errors should include cleanup issue context via RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister ps-0 from ps"),
        "run-config ps register-timeout cleanup issue context should include ps deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps"),
        "run-config ps register-timeout cleanup issue context should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_run_config_propagates_ps_service_type_into_register_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_run_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let run = RunConfig {
        is_local: true,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "ps_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_run_config(
            Arc::clone(&discovery),
            &run,
            None,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_run_config should not hang when ps register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register ps-0 as ps_custom after 20ms"),
        "custom ps service type from RunConfig should appear in timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom ps register-timeout diagnostics should include cleanup issue context via RunConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister ps-0 from ps_custom"),
        "custom ps register-timeout cleanup context via RunConfig should include ps deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps_custom"),
        "custom ps register-timeout cleanup context via RunConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom ps register-timeout errors should include cleanup issue context via RunConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister ps-0 from ps_custom"),
        "run-config custom ps register-timeout cleanup issue context should include deregister timeout diagnostics with custom ps service type: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps_custom"),
        "run-config custom ps register-timeout cleanup issue context should include disconnect timeout diagnostics with custom ps service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_connect_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-0 via worker after 20ms"),
        "connect timeout should remain primary over cleanup timeout when configured via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "connect-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "connect-timeout cleanup context via RunnerConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_connect_failure_with_cleanup_timeout_context(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithHangingDisconnectFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when connect fails and disconnect cleanup blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "connect failure should remain primary when cleanup disconnect blocks via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "connect-failure diagnostics via RunnerConfig should include cleanup issue context when cleanup disconnect blocks: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
        ),
        "connect-failure cleanup context via RunnerConfig should include disconnect-timeout diagnostics with custom worker service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_connect_failure_with_disconnect_failure_context(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithFailingDisconnectFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "connect failure should remain primary when cleanup disconnect fails via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "connect-failure diagnostics via RunnerConfig should include cleanup issue context when cleanup disconnect fails: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via trainer_custom")
            && msg.contains("forced disconnect failure"),
        "connect-failure cleanup context via RunnerConfig should include disconnect-failure diagnostics with custom service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_cleanup_timeout_context(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithHangingDisconnectFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when default-ps connect fails and disconnect cleanup blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "default-ps connect failure should remain primary when cleanup disconnect blocks via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "default-ps connect-failure diagnostics via RunnerConfig should include cleanup issue context when cleanup disconnect blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps after 20ms"),
        "default-ps connect-failure cleanup context via RunnerConfig should include default-service-type disconnect-timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_default_ps_connect_failure_with_disconnect_failure_context(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithFailingDisconnectFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "default-ps connect failure should remain primary when cleanup disconnect fails via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "default-ps connect-failure diagnostics via RunnerConfig should include cleanup issue context when cleanup disconnect fails: {msg}"
    );
    assert!(
        msg.contains("disconnect ps-0 via ps") && msg.contains("forced disconnect failure"),
        "default-ps connect-failure cleanup context via RunnerConfig should include default-service-type disconnect-failure diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_ps_connect_failure_with_cleanup_timeout_context(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithHangingDisconnectFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when ps connect fails and disconnect cleanup blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "ps connect failure should remain primary when cleanup disconnect blocks via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps connect-failure diagnostics via RunnerConfig should include cleanup issue context when cleanup disconnect blocks: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom after 20ms"
        ),
        "ps connect-failure cleanup context via RunnerConfig should include disconnect-timeout diagnostics with custom ps service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_ps_connect_failure_with_disconnect_failure_context(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(FailingConnectWithFailingDisconnectFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced connect failure"),
        "ps connect failure should remain primary when cleanup disconnect fails via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps connect-failure diagnostics via RunnerConfig should include cleanup issue context when cleanup disconnect fails: {msg}"
    );
    assert!(
        msg.contains("disconnect ps-0 via parameter_server_custom")
            && msg.contains("forced disconnect failure"),
        "ps connect-failure cleanup context via RunnerConfig should include disconnect-failure diagnostics with custom ps service type: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_worker_index_into_connect_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 4,
        num_ps: 1,
        num_workers: 5,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-4 via worker after 20ms"),
        "worker index from RunnerConfig should propagate into timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "worker-index connect-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-4 via worker"),
        "worker-index connect-timeout cleanup context via RunnerConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_worker_service_type_into_connect_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-0 via trainer_custom after 20ms"),
        "worker service type from RunnerConfig should propagate into connect-timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom worker connect-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via trainer_custom"),
        "custom worker connect-timeout cleanup context via RunnerConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_barrier_timeout_controls() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 2,
        barrier_timeout_ms: 40,
        connect_retries: 50,
        retry_backoff_ms: 10,
        ..RunnerConfig::default()
    };

    let discovery_bg = Arc::clone(&discovery);
    let runner_bg = runner.clone();
    let ps_task = tokio::spawn(async move {
        run_distributed_from_runner_config(
            discovery_bg,
            &runner_bg,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        )
        .await
    });

    tokio::time::sleep(Duration::from_millis(50)).await;
    let started = Instant::now();
    let worker_res = tokio::time::timeout(
        Duration::from_millis(1500),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    ps_task.abort();

    assert!(
        worker_res.is_ok(),
        "worker run should return promptly when barrier timeout is configured"
    );
    let elapsed = started.elapsed();
    let msg = worker_res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Barrier timeout"),
        "worker should fail with barrier timeout when only one of two workers runs: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(700),
        "barrier_timeout_ms from RunnerConfig should bound worker barrier wait duration (elapsed: {:?})",
        elapsed
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_ps_connect_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when ps connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect ps-0 via ps after 20ms"),
        "ps connect timeout should remain primary over cleanup timeout when configured via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps connect-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps"),
        "ps connect-timeout cleanup context via RunnerConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_ps_index_into_connect_timeout_diagnostics()
{
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 2,
        num_ps: 3,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when ps connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect ps-2 via ps after 20ms"),
        "ps index from RunnerConfig should propagate into timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps-index connect-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-2 via ps"),
        "ps-index connect-timeout cleanup context via RunnerConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_ps_service_type_into_connect_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when ps connect and cleanup disconnect block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect ps-0 via parameter_server_custom after 20ms"),
        "ps service type from RunnerConfig should propagate into connect-timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom ps connect-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect ps-0 via parameter_server_custom"
        ),
        "custom ps connect-timeout cleanup context via RunnerConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_honors_cleanup_timeout_with_blocked_cleanup() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(HangingConnectAndCleanupFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 10,
        discovery_cleanup_timeout_ms: 10,
        ..RunnerConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when connect and cleanup are blocked"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: connect worker-0 via worker after 10ms"),
        "operation timeout diagnostics should include configured operation timeout: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "cleanup-timeout bounded connect path should include cleanup issue context via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 10ms"),
        "cleanup-timeout bounded connect path via RunnerConfig should include disconnect timeout diagnostics with configured cleanup timeout: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(150),
        "cleanup timeout from RunnerConfig should bound blocked cleanup duration (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_honors_discover_timeout_controls() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingDiscoverFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when discover blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: discover worker-0 for ps after 20ms"),
        "runner-config timeout controls should propagate into discover timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_discover_service_type_into_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingDiscoverFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when discover blocks"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery operation: discover worker-0 for parameter_server_custom after 20ms"
        ),
        "custom runner-config ps service type should propagate into discover timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_discover_retry_controls() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingDiscoverFromRunConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 2,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(1000),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when discover repeatedly times out"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: discover worker-0 for ps after 20ms"),
        "discover timeout diagnostics should include configured operation timeout: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(
        discovery.discover_count.load(Ordering::SeqCst),
        3,
        "connect_retries=2 should yield exactly 3 discover attempts"
    );
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_retry_backoff_controls() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(EmptyDiscoverFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 2,
        retry_backoff_ms: 40,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(1200),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when PS list remains empty"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "empty-discover retry path should fail with PS discovery timeout: {msg}"
    );
    assert!(
        elapsed >= Duration::from_millis(60),
        "retry_backoff_ms from RunnerConfig should be reflected in elapsed retry delay (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 3);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_worker_discovery_error_when_cleanup_times_out(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(EmptyDiscoverWithHangingCleanupFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when worker discovery fails and cleanup steps block"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "worker discovery timeout should remain primary over cleanup timeout failures when configured via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("service type: ps"),
        "worker discovery timeout diagnostics should include default PS service-type context when invoked from RunnerConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("for worker-0"),
        "worker discovery timeout diagnostics should include worker service-id context when invoked from RunnerConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "worker discovery timeout errors should include cleanup issue context when cleanup also times out via RunnerConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
        "runner-config cleanup issue context should include deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "runner-config cleanup issue context should include disconnect timeout diagnostics: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(260),
        "cleanup timeout from RunnerConfig should bound blocked worker-cleanup duration after discovery failure (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_custom_discover_service_type_into_worker_discovery_error_when_cleanup_times_out(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(EmptyDiscoverWithHangingCleanupFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when worker discovery fails and cleanup steps block with custom discover service type"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "worker discovery timeout should remain primary over cleanup timeout failures for custom discover service type via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("service type: parameter_server_custom"),
        "worker discovery timeout diagnostics should include configured custom PS service type via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("for worker-0"),
        "worker discovery timeout diagnostics should include worker service-id context for custom PS service type via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "worker discovery timeout errors should include cleanup issue context for custom PS service type via RunnerConfig entrypoint: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
        "runner-config cleanup issue context with custom PS service type should include deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "runner-config cleanup issue context with custom PS service type should include disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_worker_index_into_ps_discovery_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(EmptyDiscoverFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 4,
        num_ps: 1,
        num_workers: 5,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when PS discovery remains empty for worker index propagation diagnostics"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out waiting for PS discovery"),
        "worker discovery should fail with discovery timeout when no PS endpoints are returned: {msg}"
    );
    assert!(
        msg.contains("for worker-4"),
        "worker index from RunnerConfig should propagate into worker discovery timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_zero_operation_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 0,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_operation_timeout > 0"),
        "zero runner-config operation timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_zero_cleanup_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_cleanup_timeout_ms: 0,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_cleanup_timeout > 0"),
        "zero runner-config cleanup timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_zero_barrier_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        barrier_timeout_ms: 0,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires barrier_timeout_ms > 0"),
        "zero runner-config barrier timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_negative_barrier_timeout() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        barrier_timeout_ms: -1,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires barrier_timeout_ms > 0"),
        "negative runner-config barrier timeout should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_zero_num_ps() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 0,
        num_workers: 1,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires num_ps > 0"),
        "zero runner-config num_ps should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_zero_num_workers() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 0,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires num_workers > 0"),
        "zero runner-config num_workers should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_zero_dim() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        dim: 0,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires dim > 0"),
        "zero runner-config dim should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_worker_index_out_of_range() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 3,
        num_ps: 1,
        num_workers: 3,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires index < num_workers for worker role"),
        "out-of-range runner-config worker index should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_ps_index_out_of_range() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 1,
        num_ps: 1,
        num_workers: 1,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires index < num_ps for ps role"),
        "out-of-range runner-config ps index should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_empty_ps_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "  ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty discovery_service_type_ps"),
        "empty runner-config ps service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_whitespace_padded_ps_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: " ps ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires discovery_service_type_ps without leading/trailing whitespace"
        ),
        "whitespace-padded runner-config ps service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_internal_whitespace_ps_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "ps cluster".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_service_type_ps without whitespace characters"),
        "internal-whitespace runner-config ps service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_empty_worker_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty discovery_service_type_worker"),
        "empty runner-config worker service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_whitespace_padded_worker_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: " worker ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires discovery_service_type_worker without leading/trailing whitespace"
        ),
        "whitespace-padded runner-config worker service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_internal_whitespace_worker_service_type() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "worker cluster".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires discovery_service_type_worker without whitespace characters"),
        "internal-whitespace runner-config worker service type should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_identical_ps_and_worker_service_types() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "service".to_string(),
        discovery_service_type_worker: "service".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires distinct discovery_service_type_ps and discovery_service_type_worker"
        ),
        "identical runner-config discovery service types should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_case_insensitive_identical_ps_and_worker_service_types(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "Service".to_string(),
        discovery_service_type_worker: "service".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires distinct discovery_service_type_ps and discovery_service_type_worker"
        ),
        "case-insensitive identical runner-config discovery service types should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_empty_table_name() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        table_name: "  ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty table_name"),
        "empty runner-config table name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_whitespace_padded_table_name() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        table_name: " emb ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires table_name without leading/trailing whitespace"),
        "whitespace-padded runner-config table name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_internal_whitespace_table_name() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        table_name: "my table".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires table_name without whitespace characters"),
        "internal-whitespace runner-config table name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_zero_parameter_sync_interval_with_targets() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_interval_ms: 0,
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_interval > 0 when parameter_sync_targets are configured"
        ),
        "zero runner-config parameter-sync interval with targets should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_empty_parameter_sync_target_entry() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["".to_string()],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires non-empty parameter_sync_targets entries"),
        "empty runner-config parameter-sync target entry should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_accepts_case_insensitive_http_scheme_parameter_sync_target(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(EmptyDiscoverFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["HtTp://127.0.0.1:8500".to_string()],
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang while validating case-insensitive parameter-sync target schemes"
    );
    let err = res.unwrap().unwrap_err().to_string();
    assert!(
        err.contains("Timed out waiting for PS discovery"),
        "case-insensitive parameter-sync target scheme should pass config validation and reach worker discovery path: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_whitespace_padded_parameter_sync_target_entry(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![" 127.0.0.1:8500 ".to_string()],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_targets entries without leading/trailing whitespace"
        ),
        "whitespace-padded runner-config parameter-sync target entry should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_invalid_parameter_sync_target_endpoint() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["http://".to_string()],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config has invalid parameter_sync_targets entry `http://`"),
        "invalid runner-config parameter-sync target endpoint should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_parameter_sync_target_endpoint_with_path_or_query(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["http://127.0.0.1:8500/v1?foo=bar".to_string()],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("endpoint must not include a URL path or query"),
        "runner-config parameter-sync target with URL path/query should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_parameter_sync_target_endpoint_with_unsupported_scheme(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["ftp://127.0.0.1:8500".to_string()],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("endpoint scheme must be http or https"),
        "runner-config parameter-sync target with unsupported scheme should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_parameter_sync_target_endpoint_with_userinfo(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["http://user@127.0.0.1:8500".to_string()],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("endpoint must not include userinfo"),
        "runner-config parameter-sync target with userinfo should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry() {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1:8500".to_string(),
            "127.0.0.1:8500".to_string(),
        ],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate runner-config parameter-sync target entries should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry_after_http_prefix_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1:8500".to_string(),
            "http://127.0.0.1:8500".to_string(),
        ],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate runner-config parameter-sync target entries after http-prefix normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry_after_trailing_slash_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1:8500".to_string(),
            "http://127.0.0.1:8500/".to_string(),
        ],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate runner-config parameter-sync target entries after trailing-slash normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry_after_http_default_port_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "127.0.0.1".to_string(),
            "http://127.0.0.1:80".to_string(),
        ],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate runner-config parameter-sync target entries after http default-port normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry_after_https_default_port_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "https://127.0.0.1".to_string(),
            "https://127.0.0.1:443".to_string(),
        ],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate runner-config parameter-sync target entries after https default-port normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_duplicate_parameter_sync_target_entry_after_case_insensitive_http_prefix_and_host_normalization(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec![
            "EXAMPLE.com:8500".to_string(),
            "HtTp://example.COM:8500".to_string(),
        ],
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("distributed config requires unique parameter_sync_targets entries"),
        "duplicate runner-config parameter-sync target entries after case-insensitive http-prefix and host normalization should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_empty_parameter_sync_model_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_model_name: " ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires non-empty parameter_sync_model_name when parameter_sync_targets are configured"
        ),
        "empty runner-config parameter-sync model name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_whitespace_padded_parameter_sync_model_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_model_name: " model ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_model_name without leading/trailing whitespace when parameter_sync_targets are configured"
        ),
        "whitespace-padded runner-config parameter-sync model name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_internal_whitespace_parameter_sync_model_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_model_name: "my model".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_model_name without whitespace characters when parameter_sync_targets are configured"
        ),
        "internal-whitespace runner-config parameter-sync model name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_empty_parameter_sync_signature_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_signature_name: "".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires non-empty parameter_sync_signature_name when parameter_sync_targets are configured"
        ),
        "empty runner-config parameter-sync signature name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_whitespace_padded_parameter_sync_signature_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_signature_name: " signature ".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_signature_name without leading/trailing whitespace when parameter_sync_targets are configured"
        ),
        "whitespace-padded runner-config parameter-sync signature name should be rejected by distributed config validation: {err}"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_rejects_internal_whitespace_parameter_sync_signature_name_with_targets(
) {
    use monolith_training::discovery::InMemoryDiscovery;
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(InMemoryDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        enable_parameter_sync: true,
        parameter_sync_targets: vec!["127.0.0.1:8500".to_string()],
        parameter_sync_signature_name: "serving default".to_string(),
        ..RunnerConfig::default()
    };

    let err = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await
    .unwrap_err()
    .to_string();
    assert!(
        err.contains(
            "distributed config requires parameter_sync_signature_name without whitespace characters when parameter_sync_targets are configured"
        ),
        "internal-whitespace runner-config parameter-sync signature name should be rejected by distributed config validation: {err}"
    );
}


#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_custom_service_type_fields() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(RecordingServiceTypePropagationDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let worker_res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let worker_msg = worker_res.unwrap_err().to_string();
    assert!(
        worker_msg.contains("Timed out waiting for PS discovery"),
        "worker run should fail in discover loop with empty discovery backend: {worker_msg}"
    );

    discovery.set_fail_register(true);
    let ps_res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Ps,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let ps_msg = ps_res.unwrap_err().to_string();
    assert!(
        ps_msg.contains("forced register failure"),
        "ps run should fail via forced register error to avoid server startup in service-type propagation test: {ps_msg}"
    );

    let registered = discovery.registered_types_snapshot();
    assert!(
        registered.contains(&"trainer_custom".to_string()),
        "worker should register with custom worker service type: {registered:?}"
    );
    assert!(
        registered.contains(&"parameter_server_custom".to_string()),
        "ps should register with custom ps service type: {registered:?}"
    );
    let discovered = discovery.discovered_types_snapshot();
    assert_eq!(
        discovered,
        vec!["parameter_server_custom".to_string()],
        "worker discovery should query custom ps discovery service type"
    );
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_deregister_timeout_with_custom_service_type_after_success(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when deregister cleanup blocks after successful ps run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
        ),
        "deregister timeout diagnostics should include custom worker service type from RunnerConfig after successful worker run: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "deregister-timeout diagnostics after successful worker run should include cleanup issue context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_disconnect_timeout_with_custom_service_type_after_success(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when disconnect cleanup blocks after successful ps run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
        ),
        "disconnect timeout diagnostics should include custom worker service type from RunnerConfig after successful worker run: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "disconnect-timeout diagnostics after successful worker run should include cleanup issue context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_deregister_timeout_after_success() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when default-service-type deregister cleanup blocks after successful worker run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker after 20ms"),
        "default-service-type deregister timeout diagnostics should be preserved after successful worker run via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type deregister timeout diagnostics should include successful-role cleanup issue context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_disconnect_timeout_after_success() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when default-service-type disconnect cleanup blocks after successful worker run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker after 20ms"),
        "default-service-type disconnect timeout diagnostics should be preserved after successful worker run via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type disconnect timeout diagnostics should include successful-role cleanup issue context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_deregister_failure_after_success() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced deregister failure"),
        "default-service-type deregister failure should be preserved after successful worker run via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type deregister failure should include successful-role cleanup issue context via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("deregister worker-0 from worker"),
        "default-service-type deregister failure should include cleanup operation context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_disconnect_failure_after_success() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced disconnect failure"),
        "default-service-type disconnect failure should be preserved after successful worker run via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "default-service-type disconnect failure should include successful-role cleanup issue context via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via worker"),
        "default-service-type disconnect failure should include cleanup operation context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_deregister_failure_with_disconnect_failure_context_after_success(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        true,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("deregister worker-0 from worker") && msg.contains("forced deregister failure"),
        "runner-config post-success both-failure diagnostics should preserve deregister failure with operation context: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "runner-config post-success both-failure diagnostics should include cleanup issue context: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via worker") && msg.contains("forced disconnect failure"),
        "runner-config post-success both-failure diagnostics should include disconnect failure with operation context: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_custom_worker_deregister_failure_after_success(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        false,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced deregister failure"),
        "custom-worker deregister failure should be preserved after successful worker run via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("deregister worker-0 from trainer_custom"),
        "custom-worker deregister failure should include cleanup operation context via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "custom-worker deregister failure should include successful-role cleanup issue context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_surfaces_custom_worker_disconnect_failure_after_success(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(FailingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        false,
        true,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = run_distributed_from_runner_config(
        Arc::clone(&discovery),
        &runner,
        Role::Worker,
        "127.0.0.1:0".parse().unwrap(),
    )
    .await;
    let msg = res.unwrap_err().to_string();
    assert!(
        msg.contains("forced disconnect failure"),
        "custom-worker disconnect failure should be preserved after successful worker run via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("disconnect worker-0 via trainer_custom"),
        "custom-worker disconnect failure should include cleanup operation context via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "custom-worker disconnect failure should include successful-role cleanup issue context via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_deregister_timeout_with_disconnect_timeout_context_after_success(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let (ps_server, ps_addr) = spawn_worker_success_ps_server(8).await;
    let discovery = Arc::new(HangingCleanupAfterSuccessFromConfigDiscovery::new(
        ps_addr.to_string(),
        true,
        true,
    ));
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        connect_retries: 0,
        retry_backoff_ms: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_service_type_ps: "parameter_server_custom".to_string(),
        dim: 8,
        discovery_operation_timeout_ms: 200,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when both cleanup steps block after successful worker run"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: deregister worker-0 from trainer_custom after 20ms"
        ),
        "deregister timeout should remain primary in runner-config post-success cleanup path when both cleanup steps block: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after successful role completion"),
        "runner-config post-success cleanup path should append cleanup issue context when both cleanup steps block: {msg}"
    );
    assert!(
        msg.contains(
            "Timed out during discovery cleanup: disconnect worker-0 via trainer_custom after 20ms"
        ),
        "runner-config post-success cleanup issue context should include disconnect timeout diagnostics when both cleanup steps block: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
    ps_server.abort();
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_register_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register worker-0 as worker after 20ms"),
        "register timeout should remain primary over cleanup timeout when configured via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "register timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from worker"),
        "register-timeout cleanup context via RunnerConfig should include worker deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via worker"),
        "register-timeout cleanup context via RunnerConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_worker_service_type_into_register_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_worker: "trainer_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when worker register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register worker-0 as trainer_custom after 20ms"),
        "custom worker service type from RunnerConfig should appear in timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom worker register-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister worker-0 from trainer_custom"),
        "custom worker register-timeout cleanup context via RunnerConfig should include worker deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect worker-0 via trainer_custom"),
        "custom worker register-timeout cleanup context via RunnerConfig should include worker disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_honors_cleanup_timeout_after_register_timeout() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 10,
        discovery_cleanup_timeout_ms: 10,
        ..RunnerConfig::default()
    };

    let started = Instant::now();
    let res = tokio::time::timeout(
        Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Worker,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when register and cleanup are blocked"
    );
    let elapsed = started.elapsed();
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register worker-0 as worker after 10ms"),
        "operation timeout diagnostics should include configured operation timeout: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "cleanup timeout bounding path should still include cleanup issue context via RunnerConfig: {msg}"
    );
    assert!(
        elapsed < Duration::from_millis(150),
        "cleanup timeout from RunnerConfig should bound blocked register-cleanup duration (elapsed: {:?})",
        elapsed
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_preserves_ps_register_timeout_when_cleanup_blocks() {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when ps register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register ps-0 as ps after 20ms"),
        "ps register timeout should remain primary over cleanup timeout when configured via RunnerConfig: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "ps register-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister ps-0 from ps"),
        "ps register-timeout cleanup context via RunnerConfig should include ps deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps"),
        "ps register-timeout cleanup context via RunnerConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn distributed_runner_from_runner_config_propagates_ps_service_type_into_register_timeout_diagnostics(
) {
    use monolith_training::runner::{run_distributed_from_runner_config, Role};
    use std::sync::Arc;

    let discovery = Arc::new(HangingRegisterAndCleanupFromConfigDiscovery::new());
    let runner = RunnerConfig {
        is_local: true,
        index: 0,
        num_ps: 1,
        num_workers: 1,
        discovery_service_type_ps: "ps_custom".to_string(),
        discovery_operation_timeout_ms: 20,
        discovery_cleanup_timeout_ms: 20,
        ..RunnerConfig::default()
    };

    let res = tokio::time::timeout(
        std::time::Duration::from_millis(700),
        run_distributed_from_runner_config(
            Arc::clone(&discovery),
            &runner,
            Role::Ps,
            "127.0.0.1:0".parse().unwrap(),
        ),
    )
    .await;
    assert!(
        res.is_ok(),
        "run_distributed_from_runner_config should not hang when ps register and cleanup block"
    );
    let msg = res.unwrap().unwrap_err().to_string();
    assert!(
        msg.contains("Timed out during discovery operation: register ps-0 as ps_custom after 20ms"),
        "custom ps service type from RunnerConfig should appear in timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("discovery cleanup encountered issues after role error"),
        "custom ps register-timeout diagnostics should include cleanup issue context via RunnerConfig when cleanup also blocks: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: deregister ps-0 from ps_custom"),
        "custom ps register-timeout cleanup context via RunnerConfig should include ps deregister timeout diagnostics: {msg}"
    );
    assert!(
        msg.contains("Timed out during discovery cleanup: disconnect ps-0 via ps_custom"),
        "custom ps register-timeout cleanup context via RunnerConfig should include ps disconnect timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.register_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
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
