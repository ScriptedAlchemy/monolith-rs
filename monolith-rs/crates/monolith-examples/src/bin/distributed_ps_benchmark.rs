//! Rust analogue of `monolith/native_training/distributed_ps_benchmark.py`.
//!
//! This benchmark is TF-free and targets the Rust-native PS gRPC service:
//! - start N PS shards
//! - run lookup/apply_gradients for large ID batches
//! - optional dedup on client side (to reflect Python's enable_dedup behavior)

use anyhow::Context;
use clap::Parser;
use monolith_training::distributed_ps::{serve_ps, PsClient, PsServer};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 4)]
    ps_num: usize,

    #[arg(long, default_value_t = 100_000)]
    num_elements: usize,

    #[arg(long, default_value_t = 16)]
    dim: usize,

    /// Run lookup benchmark
    #[arg(long, default_value_t = true)]
    lookup: bool,

    /// Run apply_gradients benchmark
    #[arg(long, default_value_t = true)]
    apply: bool,

    /// Enable client-side dedup for lookup/apply
    #[arg(long, default_value_t = true)]
    dedup: bool,
}

fn make_ids(n: usize) -> Vec<i64> {
    // Similar to Python benchmark: lots of duplicates.
    (0..n).map(|x| (x / 2) as i64).collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    let args = Args::parse();

    let mut addrs = Vec::with_capacity(args.ps_num);
    let mut handles = Vec::with_capacity(args.ps_num);

    for shard in 0..args.ps_num {
        let ps = PsServer::new(shard as i32, args.dim);
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0);
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;
        addrs.push(local_addr);

        let ps = Arc::clone(&ps);
        handles.push(tokio::spawn(async move {
            serve_ps(ps, local_addr)
                .await
                .map_err(|e| anyhow::anyhow!("ps serve failed: {}", e))
        }));
        drop(listener); // allow tonic to bind
    }

    // Give servers a moment to start.
    tokio::time::sleep(Duration::from_millis(100)).await;

    let addr_strs: Vec<String> = addrs.iter().map(|a| a.to_string()).collect();
    let addr_refs: Vec<&str> = addr_strs.iter().map(|s| s.as_str()).collect();
    let mut client = PsClient::connect(&addr_refs)
        .await
        .context("connect PsClient")?;

    let ids = make_ids(args.num_elements);
    let table = "bench";

    if args.dedup {
        tracing::warn!("--dedup is a no-op for this benchmark binary right now; PsClient always deduplicates internally for lookup/apply.");
    }

    if args.lookup {
        let start = Instant::now();
        let _ = client
            .lookup(table, &ids, args.dim, true)
            .await
            .context("lookup")?;
        let elapsed = start.elapsed();
        tracing::info!(
            ps_num = args.ps_num,
            num_elements = args.num_elements,
            dim = args.dim,
            dedup = args.dedup,
            ms = elapsed.as_millis(),
            "lookup benchmark done"
        );
    }

    if args.apply {
        // Apply constant grads.
        let grads = vec![0.3f32; ids.len() * args.dim];
        let start = Instant::now();
        let _ = client
            .apply_gradients(table, &ids, &grads, args.dim, 0.01, 0)
            .await
            .context("apply_gradients")?;
        let elapsed = start.elapsed();
        tracing::info!(
            ps_num = args.ps_num,
            num_elements = args.num_elements,
            dim = args.dim,
            dedup = args.dedup,
            ms = elapsed.as_millis(),
            "apply_gradients benchmark done"
        );
    }

    // Keep servers alive until done; then abort tasks.
    for h in handles {
        h.abort();
    }

    Ok(())
}
