use monolith_serving::{EmbeddingStore, EmbeddingStorePushSink, ParameterSyncGrpcServer};
use monolith_training::distributed_ps::PsServer;
use monolith_training::parameter_sync_replicator::{DirtyTracker, ParameterSyncReplicator};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

fn localhost_port(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_training_ps_pushes_deltas_to_online_store() {
    // Online ParameterSync server with an embedding store sink.
    let online_store = Arc::new(EmbeddingStore::new());
    let sink = Arc::new(EmbeddingStorePushSink::new(Arc::clone(&online_store)));
    let psync_addr = localhost_port(55351);
    let psync_srv = ParameterSyncGrpcServer::new(sink);
    let h = tokio::spawn(async move { psync_srv.serve(psync_addr).await });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Training PS in-process (we don't need to serve its gRPC for this test).
    let ps = PsServer::new(0, 2);
    let tracker = Arc::new(DirtyTracker::default());

    // Create table + initialize fid 7.
    let table = ps.get_or_create_table("t", 2);
    let (_emb, _found) = table.lookup(&[7], true);

    // Apply a gradient update so embedding changes.
    let (_updated, _nf) = table.apply_gradients(&[7], &[1.0, 2.0], 0.1);
    tracker.mark_dirty("t", &[7]);

    // Replicator pushes to online store.
    let rep = ParameterSyncReplicator::new(
        Arc::clone(&ps),
        Arc::clone(&tracker),
        vec!["127.0.0.1:55351".to_string()],
        "m".to_string(),
        "serving_default".to_string(),
        "t".to_string(),
    );

    // Flush by running one interval via spawn + sleep.
    rep.spawn(std::time::Duration::from_millis(50));
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // Online store should now contain fid 7 for table "t".
    let pushed = online_store.get("t", 7);
    assert!(pushed.is_some());
    assert_eq!(pushed.unwrap().len(), 2);

    h.abort();
}
