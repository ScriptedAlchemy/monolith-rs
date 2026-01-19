//! Rust analogue of `monolith/native_training/distribution_ops_benchmark.py`.
//!
//! Focuses on the core routing/remap behavior used by distributed embedding lookups:
//! - shard routing (`id % num_ps`)
//! - dedup + remap correctness
//!
//! This is TF-free: we benchmark the Rust implementations used by `monolith-training`.

use clap::Parser;
use monolith_training::distributed_ps::{dedup_ids, route_to_shards};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1_000_000)]
    num_elements: usize,

    #[arg(long, default_value_t = 16)]
    dim: usize,

    #[arg(long, default_value_t = 10)]
    num_shards: usize,
}

fn main_ids(n: usize) -> Vec<i64> {
    (0..n).map(|x| (x as i64)).collect()
}

fn main() {
    tracing_subscriber::fmt().with_target(false).init();
    let args = Args::parse();

    let ids = main_ids(args.num_elements);
    let start = Instant::now();
    let (unique, map) = dedup_ids(&ids);
    let elapsed = start.elapsed();
    tracing::info!(
        num_elements = args.num_elements,
        unique = unique.len(),
        ms = elapsed.as_millis(),
        "dedup_ids"
    );

    let start = Instant::now();
    let shards = route_to_shards(&unique, args.num_shards);
    let elapsed = start.elapsed();
    let total_routed: usize = shards.iter().map(|v| v.len()).sum();
    tracing::info!(
        num_shards = args.num_shards,
        total_routed,
        ms = elapsed.as_millis(),
        "route_to_shards"
    );

    // Minimal remap exercise (shape-only): pretend we have embeddings for unique ids.
    let start = Instant::now();
    let mut out = vec![0.0f32; ids.len() * args.dim];
    for (orig_idx, &u) in map.iter().enumerate() {
        let dst = orig_idx * args.dim;
        let val = (u as f32) * 0.001;
        for j in 0..args.dim {
            out[dst + j] = val;
        }
    }
    let elapsed = start.elapsed();
    tracing::info!(
        num_elements = args.num_elements,
        dim = args.dim,
        ms = elapsed.as_millis(),
        "remap (synthetic)"
    );

    // Keep `out` alive
    tracing::debug!(last = out.last().copied().unwrap_or(0.0), "done");
}
