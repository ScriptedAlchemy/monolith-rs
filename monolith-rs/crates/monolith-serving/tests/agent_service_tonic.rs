use monolith_proto::monolith::serving::agent_service::{
    GetReplicasRequest, HeartBeatRequest, ServerType,
};
use monolith_serving::grpc_agent::{connect_agent_client, AgentGrpcServer, AgentServiceRealImpl};
use tokio::time::{sleep, Duration};

#[tokio::test]
async fn agent_service_tonic_roundtrip() {
    let svc = AgentServiceRealImpl::new();
    svc.register_replica(ServerType::Ps, "ps1:9000");
    svc.register_replica(ServerType::Ps, "ps2:9000");

    let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    let local_addr = listener.local_addr().unwrap();
    drop(listener);

    let server = AgentGrpcServer::serve(local_addr, svc).await.unwrap();

    // Give the server a moment to start.
    sleep(Duration::from_millis(50)).await;

    let mut client = connect_agent_client(local_addr.to_string()).await.unwrap();

    let replicas = client
        .get_replicas(GetReplicasRequest {
            server_type: ServerType::Ps as i32,
            task: 0,
            model_name: "m".to_string(),
        })
        .await
        .unwrap()
        .into_inner();

    assert_eq!(replicas.address_list.unwrap().address.len(), 2);

    let hb = client
        .heart_beat(HeartBeatRequest {
            server_type: ServerType::Ps as i32,
        })
        .await
        .unwrap()
        .into_inner();

    assert!(hb.addresses.contains_key("PS"));

    server.shutdown();
}
