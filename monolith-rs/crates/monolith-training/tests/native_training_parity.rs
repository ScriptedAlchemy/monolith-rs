use monolith_training::py_discovery::TfConfigServiceDiscovery;
use monolith_training::discovery::{DiscoveryEvent, ServiceDiscoveryAsync, ServiceInfo};
use monolith_training::{
    copy_checkpoint_from_restore_dir, get_discovery, ConstantModelFn, EntryError, Estimator,
    RunConfig, RunnerConfig,
};
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
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
        msg.contains("Timed out during discovery operation: discover worker-0 after 20ms"),
        "run-config timeout controls should propagate into discover timeout diagnostics: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.discover_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.deregister_count.load(Ordering::SeqCst), 1);
    assert_eq!(discovery.disconnect_count.load(Ordering::SeqCst), 1);
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
        msg.contains("Timed out during discovery operation: connect worker-0 after 20ms"),
        "connect timeout should remain primary over cleanup timeout when configured via RunConfig: {msg}"
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
        msg.contains("Timed out during discovery operation: register worker-0 after 20ms"),
        "register timeout should remain primary over cleanup timeout when configured via RunConfig: {msg}"
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
        msg.contains("Timed out during discovery operation: connect worker-0 after 20ms"),
        "connect timeout should remain primary over cleanup timeout when configured via RunnerConfig: {msg}"
    );
    assert_eq!(discovery.connect_count.load(Ordering::SeqCst), 1);
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
