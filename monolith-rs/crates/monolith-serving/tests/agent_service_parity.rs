use monolith_proto::monolith::serving::agent_service::{
    GetReplicasRequest, HeartBeatRequest, ServerType,
};
use monolith_serving::agent_service_discovery::{
    connect_agent_service_client, AgentDiscoveryServer, AgentServiceDiscoveryImpl,
};
use monolith_serving::backends::FakeKazooClient;
use monolith_serving::data_def::ReplicaMeta;
use monolith_serving::tfs_monitor::DeployType as TfsDeployType;
use monolith_serving::{find_free_port, ModelState, ReplicaWatcher, WatcherConfig, ZkClient};
use std::sync::Arc;
use tokio::time::{sleep, Duration};

// Mirrors `monolith/agent_service/agent_service_test.py` (v1 watcher-backed service).
#[tokio::test]
async fn agent_service_v1_heartbeat_and_get_replicas() {
    std::env::set_var("TCE_INTERNAL_IDC", "lf");
    std::env::set_var("TCE_LOGICAL_CLUSTER", "default");

    let bzid = "test_model";
    let base_name = "test_model_ctr";

    let zk = Arc::new(FakeKazooClient::new());
    zk.start().unwrap();

    // Register replica znodes similar to the Python test.
    let path_prefix = format!("/{}/service/{}/lf:default", bzid, base_name);

    let num_ps = 20;
    let num_ps_replicas = 2;
    let num_entry_replicas = 2;

    let mut idx = 2;
    for task_id in 0..num_ps {
        for replica_id in 0..num_ps_replicas {
            let meta = ReplicaMeta {
                address: Some(format!("192.168.1.{}:{}", idx, find_free_port().unwrap())),
                stat: ModelState::Available as i32,
                ..Default::default()
            };
            let replica_path = format!("{}/ps:{}/{}", path_prefix, task_id, replica_id);
            zk.create(&replica_path, meta.serialize(), true, true)
                .unwrap();
            idx += 1;
        }
    }

    for replica_id in 0..num_entry_replicas {
        let meta = ReplicaMeta {
            address: Some(format!("192.168.1.{}:{}", idx, find_free_port().unwrap())),
            stat: ModelState::Available as i32,
            ..Default::default()
        };
        let replica_path = format!("{}/entry:0/{}", path_prefix, replica_id);
        zk.create(&replica_path, meta.serialize(), true, true)
            .unwrap();
        idx += 1;
    }

    // Start watcher (mirrors the Python ReplicaWatcher.watch_data()).
    let watcher_conf = WatcherConfig {
        bzid: bzid.to_string(),
        base_name: base_name.to_string(),
        dc_aware: true,
        deploy_type: TfsDeployType::Ps,
        dense_alone: false,
    };
    let watcher = Arc::new(ReplicaWatcher::new(
        zk.clone(),
        watcher_conf,
        false,
        monolith_serving::AddressFamily::IPV4,
    ));
    watcher.watch_data().unwrap();

    // Serve discovery service on an ephemeral port.
    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let local_addr = listener.local_addr().unwrap();
    drop(listener);

    let svc = AgentServiceDiscoveryImpl::from_watcher(Arc::clone(&watcher), None);
    let server = AgentDiscoveryServer::serve(local_addr, svc).await.unwrap();

    // Give watcher + server time to start.
    sleep(Duration::from_millis(50)).await;

    let mut client = connect_agent_service_client(local_addr.to_string())
        .await
        .unwrap();

    // HeartBeat should return 20 ps tasks.
    let hb = client
        .heart_beat(HeartBeatRequest {
            server_type: ServerType::Ps as i32,
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(hb.addresses.len(), num_ps as usize);

    // GetReplicas should return `num_ps_replicas` addresses for the requested task.
    let gr = client
        .get_replicas(GetReplicasRequest {
            server_type: ServerType::Ps as i32,
            task: num_ps_replicas - 1,
            model_name: base_name.to_string(),
        })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        gr.address_list.unwrap().address.len(),
        num_ps_replicas as usize
    );

    server.shutdown();
    watcher.stop();
}
