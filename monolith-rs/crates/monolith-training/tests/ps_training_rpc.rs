use monolith_training::distributed_ps::{serve_ps, PsClient, PsServer};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

fn localhost_port(port: u16) -> SocketAddr {
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rpc_lookup_dedup_and_remap_across_shards() {
    // 2-shard PS cluster.
    let ps0 = PsServer::new(0, 4);
    let ps1 = PsServer::new(1, 4);

    // Bind fixed ports for simplicity in tests.
    let addr0 = localhost_port(55051);
    let addr1 = localhost_port(55052);

    let t0 = tokio::spawn(async move { serve_ps(ps0, addr0).await });
    let t1 = tokio::spawn(async move { serve_ps(ps1, addr1).await });

    // Give servers a moment to bind.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut client = PsClient::connect(&["127.0.0.1:55051", "127.0.0.1:55052"])
        .await
        .unwrap();

    // Includes duplicates and cross-shard ids.
    let fids = vec![1i64, 2, 1, 3, 2, 4];
    let out = client
        .lookup("t", &fids, 4, /*create_if_missing=*/ true)
        .await
        .unwrap();

    assert_eq!(out.len(), fids.len() * 4);

    // Duplicates should map to identical embeddings in their respective rows.
    assert_eq!(&out[0..4], &out[8..12]); // fid 1 at positions 0 and 2
    assert_eq!(&out[4..8], &out[16..20]); // fid 2 at positions 1 and 4

    // Shutdown the servers to avoid hanging tasks.
    t0.abort();
    t1.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_rpc_apply_gradients_aggregates_duplicates() {
    let ps0 = PsServer::new(0, 2);
    let ps1 = PsServer::new(1, 2);

    let addr0 = localhost_port(55151);
    let addr1 = localhost_port(55152);

    let t0 = tokio::spawn(async move { serve_ps(ps0, addr0).await });
    let t1 = tokio::spawn(async move { serve_ps(ps1, addr1).await });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    let mut client = PsClient::connect(&["127.0.0.1:55151", "127.0.0.1:55152"])
        .await
        .unwrap();

    // Initialize fid=7 and read initial value.
    let fids = vec![7i64];
    let before = client.lookup("t", &fids, 2, true).await.unwrap();

    // Apply gradients where fid=7 appears twice; client should aggregate.
    // grad vectors: [1,2] and [0.5,0.5] -> aggregated [1.5,2.5]
    let dup_fids = vec![7i64, 7i64];
    let grads = vec![1.0f32, 2.0, 0.5, 0.5];
    client
        .apply_gradients("t", &dup_fids, &grads, 2, 0.1, 1)
        .await
        .unwrap();

    let after = client.lookup("t", &fids, 2, false).await.unwrap();

    // Expected: w_new = w_old - lr * sum_grad
    assert!((after[0] - (before[0] - 0.1 * 1.5)).abs() < 1e-5);
    assert!((after[1] - (before[1] - 0.1 * 2.5)).abs() < 1e-5);

    t0.abort();
    t1.abort();
}
