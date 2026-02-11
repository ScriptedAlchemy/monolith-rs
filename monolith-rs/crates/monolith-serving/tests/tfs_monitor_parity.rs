#![cfg(feature = "grpc")]

use monolith_serving::data_def::{PublishMeta, PublishType};
use monolith_serving::mocked_tfserving::{find_free_port_blocking, FakeTfServing};
use monolith_serving::tfs_monitor::{AgentConfig, DeployType, TfServerType, TfsMonitor};
use monolith_serving::ServingResult;
use std::collections::HashMap;
use std::net::SocketAddr;
use tokio::time::{sleep, Duration};

// Mirrors `monolith/agent_service/tfs_monitor_test.py` but uses a deterministic set of publish metas.

const VERSION: &str = "1634631496";
const PATH_TEMPLATE: &str = "/tmp/monolith/agent_service/test_data/ckpt/exported_models";

fn make_version_path(sub_model: &str) -> String {
    format!("{}/{}/{}", PATH_TEMPLATE, sub_model, VERSION)
}

#[tokio::test]
async fn tfs_monitor_reload_config_and_status_transitions() -> ServingResult<()> {
    let entry_port = find_free_port_blocking();
    let ps_port = find_free_port_blocking();

    let entry_addr: SocketAddr = format!("127.0.0.1:{entry_port}").parse().unwrap();
    let ps_addr: SocketAddr = format!("127.0.0.1:{ps_port}").parse().unwrap();

    let mut tfs_entry = FakeTfServing::new(entry_addr);
    let mut tfs_ps = FakeTfServing::new(ps_addr);

    // Start with empty configs (no models loaded).
    tfs_entry.start_with_configs(vec![]).await?;
    tfs_ps.start_with_configs(vec![]).await?;

    // Give model mgr loop a moment.
    sleep(Duration::from_millis(100)).await;

    let conf = AgentConfig::for_test(DeployType::Mixed, entry_port, ps_port);
    let monitor = TfsMonitor::new(conf);
    monitor.connect().await?;

    let sub_models = HashMap::from([
        ("entry".to_string(), make_version_path("entry")),
        ("ps_0".to_string(), make_version_path("ps_0")),
        ("ps_3".to_string(), make_version_path("ps_3")),
        ("ps_5".to_string(), make_version_path("ps_5")),
    ]);

    let pm = PublishMeta {
        shard_id: Some(1),
        replica_id: 2,
        model_name: Some("test_1".to_string()),
        num_ps: Some(5),
        sub_models: Some(sub_models),
        ptype: PublishType::Load,
        ..Default::default()
    };

    let before = monitor
        .get_model_status_for_publish_meta(&pm, false)
        .await?;
    assert_eq!(before.len(), 4);
    // With no loaded models, should be NOT_FOUND.
    assert!(before.values().all(|(_p, s)| s.version == -1));

    // Load a model config covering all entries/ps, matching monitor.gen_model_config.
    let cfgs = monitor.gen_model_config(&[pm.clone()], false);
    for (service_type, cfg) in cfgs {
        // Only entry+ps are active for this test.
        if matches!(service_type, TfServerType::ENTRY | TfServerType::PS) {
            let _ = monitor
                .handle_reload_config_request(service_type, cfg)
                .await?;
        }
    }

    // Allow state machine to progress to AVAILABLE.
    sleep(Duration::from_millis(200)).await;

    let after = monitor
        .get_model_status_for_publish_meta(&pm, false)
        .await?;
    assert_eq!(after.len(), before.len());

    // Dense node logic: entry uses latest => version 1, others are specific => parsed from path.
    for (tfs_model_name, (path, status)) in after {
        if status.version == -1 {
            // Allow eventual consistency in the fake state machine.
            continue;
        }
        if tfs_model_name.ends_with(":entry") {
            assert_eq!(status.version, 1);
        } else {
            let expected = path.split('/').last().unwrap().parse::<i64>().unwrap();
            assert_eq!(status.version, expected);
        }
    }

    monitor.stop().await;
    tfs_entry.stop().await;
    tfs_ps.stop().await;
    Ok(())
}
