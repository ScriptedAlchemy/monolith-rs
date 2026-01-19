//! Rust analogue of `monolith/native_training/demo.py`.
//!
//! This is intentionally TF-free: it exercises the Rust-native training stack
//! (data parsing + layers + optimizer + checkpoint/export hooks if enabled).

use clap::Parser;
use monolith_training::{ConstantModelFn, Estimator, EstimatorConfig, LoggingHook};
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Where to write checkpoints / artifacts.
    #[arg(long, default_value = "/tmp/monolith_rs_demo_model")]
    model_dir: PathBuf,

    /// Steps to run.
    #[arg(long, default_value_t = 100)]
    steps: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_target(false).init();
    let args = Args::parse();

    let cfg = EstimatorConfig::new(args.model_dir).with_train_steps(args.steps);
    let model_fn = ConstantModelFn::new(0.5);
    let mut estimator = Estimator::new(cfg, model_fn);
    estimator.add_hook(LoggingHook::new(10));

    let r = estimator.train()?;
    tracing::info!(global_step = r.global_step, "demo_train finished");
    Ok(())
}
