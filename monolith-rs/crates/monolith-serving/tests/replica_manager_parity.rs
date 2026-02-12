use monolith_serving::grpc::ServerType;
use monolith_serving::tfs_monitor::{DeployType as TfsDeployType, TfServerType};
use monolith_serving::{
    find_free_port_async, FakeKazooClient, FakeTfServing, ReplicaManager, UpdaterConfig,
    WatcherConfig, ZkClient,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

// Mirrors the core interactions exercised by the Python replica manager tests.
// The upstream `replica_manager_test.py` is mostly setup code; here we assert that:
// - updater registers znodes under /{bzid}/service/{base_name}
// - watcher observes those znodes and can resolve AVAILABLE replicas after updater updates state

#[test]
fn replica_manager_registers_and_watches_replicas() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
    let bzid = "bzid".to_string();
    let base_name = "test_model".to_string();

    let shard_id = 1;
    let replica_id = 2;
    let num_ps = 20;
    let num_shard = 5;

    // Start two fake TFServing servers: entry + ps.
    let entry_port = find_free_port_async().await;
    let ps_port = find_free_port_async().await;

    let mut tfs_entry = FakeTfServing::new(SocketAddr::from(([127, 0, 0, 1], entry_port)));
    let mut tfs_ps = FakeTfServing::new(SocketAddr::from(([127, 0, 0, 1], ps_port)));

    let entry_cfg = FakeTfServing::default_model_config(TfServerType::ENTRY, "/tmp/entry", 1);
    let ps_cfg =
        FakeTfServing::default_model_config(&format!("{}_{}", TfServerType::PS, 1), "/tmp/ps", 1);
    tfs_entry.start_with_configs(vec![entry_cfg]).await.unwrap();
    tfs_ps.start_with_configs(vec![ps_cfg]).await.unwrap();

    // Fake ZK.
    let zk = Arc::new(FakeKazooClient::new());
    zk.start().unwrap();

    let watcher_conf = WatcherConfig {
        bzid: bzid.clone(),
        base_name: base_name.clone(),
        dc_aware: true,
        deploy_type: TfsDeployType::Mixed,
        dense_alone: false,
    };
    let updater_conf = UpdaterConfig {
        bzid: bzid.clone(),
        base_name: base_name.clone(),
        base_path: "/tmp/model/saved_models".to_string(),
        num_ps,
        num_shard,
        shard_id,
        replica_id,
        deploy_type: TfsDeployType::Mixed,
        dense_alone: false,
        tfs_entry_port: entry_port,
        tfs_ps_port: ps_port,
        tfs_dense_port: 0,
        tfs_entry_archon_port: entry_port,
        tfs_ps_archon_port: ps_port,
        tfs_dense_archon_port: 0,
        dc_aware: true,
        idc: Some("lf".to_string()),
        cluster: Some("default".to_string()),
    };

    let mgr = ReplicaManager::new(zk.clone(), watcher_conf, updater_conf);
    mgr.start().unwrap();

    // Wait for updater loop to transition model status to AVAILABLE and push to ZK.
    tokio::time::sleep(Duration::from_millis(200)).await;

    // The updater should have registered: entry + all scheduled PS tasks for this shard.
    // Verify at least entry exists.
    let prefix = format!("/{}/service/{}", bzid, base_name);
    let children = zk.get_children(&prefix).unwrap();
    assert!(children.contains(&"lf:default".to_string()));

    // Watcher should see AVAILABLE entry replica when we query with matching idc/cluster.
    // Note: fake tfserving might not have reached AVAILABLE yet; retry briefly.
    let mut found = false;
    for _ in 0..20 {
        let reps = mgr
            .watcher
            .get_replicas(ServerType::Entry, 0, Some("lf"), Some("default"));
        if !reps.is_empty() {
            found = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    if !found {
        let entry_path = format!(
            "/{}/service/{}/lf:default/entry:0/{:011}",
            bzid, base_name, replica_id
        );
        let raw = zk.get(&entry_path).ok();
        let parsed = raw
            .as_deref()
            .and_then(|b| monolith_serving::ReplicaMeta::deserialize(b).ok());
        assert!(
            found,
            "expected entry replicas to become AVAILABLE; zk entry node: path={} raw_present={} parsed={:?}",
            entry_path,
            raw.is_some(),
            parsed
        );
    }

    mgr.stop();
    tfs_entry.stop().await;
    tfs_ps.stop().await;
    });
}
