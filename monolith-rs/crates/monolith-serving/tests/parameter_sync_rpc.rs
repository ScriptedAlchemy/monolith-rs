use monolith_proto::monolith::parameter_sync::push_request::DeltaEmbeddingHashTable;
use monolith_proto::monolith::parameter_sync::PushRequest;
use monolith_serving::parameter_sync_rpc::{ParameterSyncGrpcServer, ParameterSyncRpcClient};
use monolith_serving::{EmbeddingStore, EmbeddingStorePushSink};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

fn localhost_port(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_parameter_sync_push_roundtrip() {
    let addr = localhost_port(55251);

    let store = Arc::new(EmbeddingStore::new());
    let server =
        ParameterSyncGrpcServer::new(Arc::new(EmbeddingStorePushSink::new(Arc::clone(&store))));
    let handle = tokio::spawn(async move { server.serve(addr).await });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut client = ParameterSyncRpcClient::connect("127.0.0.1:55251")
        .await
        .unwrap();

    let req = PushRequest {
        model_name: Some("m".to_string()),
        signature_name: Some("serving_default".to_string()),
        delta_hash_tables: vec![DeltaEmbeddingHashTable {
            unique_id: Some("t".to_string()),
            dim_size: Some(2),
            fids: vec![1, 2],
            embeddings: vec![0.1, 0.2, 0.3, 0.4],
        }],
        delta_multi_hash_tables: vec![],
        timeout_in_ms: Some(1000),
    };

    let resp = client.push(req).await.unwrap();
    assert_eq!(resp.status_code, Some(0));
    assert_eq!(resp.update_num, Some(2));

    // Store should reflect the pushed values.
    let e1 = store.get("t", 1).unwrap();
    assert_eq!(e1, vec![0.1, 0.2]);
    let e2 = store.get("t", 2).unwrap();
    assert_eq!(e2, vec![0.3, 0.4]);

    handle.abort();
}
