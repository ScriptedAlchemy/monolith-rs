//! Rust analogue of `monolith/native_training/hash_table_ops_benchmark.py`.
//!
//! This benchmark exercises the embedding hash table implementation directly
//! (no TensorFlow dependency) and focuses on:
//! - lookup throughput
//! - apply/update throughput

use clap::Parser;
use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable};
use std::time::Instant;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1_000_00)]
    len: usize,

    #[arg(long, default_value_t = 32)]
    dim: usize,

    #[arg(long, default_value_t = 50)]
    iters: usize,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    let args = Args::parse();

    let mut ht = CuckooEmbeddingHashTable::new(args.len * 2, args.dim);
    let ids: Vec<i64> = (0..args.len as i64).collect();

    // init embeddings (flat buffer)
    let init = vec![1.0f32; ids.len() * args.dim];
    ht.assign(&ids, &init)?;

    let start = Instant::now();
    let mut last = vec![0.0f32; ids.len() * args.dim];
    for _ in 0..args.iters {
        ht.lookup(&ids, &mut last)?;
    }
    let elapsed = start.elapsed();
    tracing::info!(
        len = args.len,
        dim = args.dim,
        iters = args.iters,
        avg_us = (elapsed.as_micros() as f64) / (args.iters as f64),
        last_len = last.len(),
        "lookup benchmark"
    );

    // Apply constant grads
    let grads = vec![0.01f32; ids.len() * args.dim];
    let start = Instant::now();
    for _ in 0..args.iters {
        ht.apply_gradients(&ids, &grads)?;
    }
    let elapsed = start.elapsed();
    tracing::info!(
        len = args.len,
        dim = args.dim,
        iters = args.iters,
        avg_us = (elapsed.as_micros() as f64) / (args.iters as f64),
        "apply_gradients benchmark"
    );

    Ok(())
}
