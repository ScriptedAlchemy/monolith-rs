//! Monolith CLI Library
//!
//! This crate provides the command-line interface for Monolith, including:
//!
//! - **Train**: Distributed training with the Monolith estimator
//! - **Serve**: gRPC model serving for inference
//! - **Export**: Checkpoint export for deployment
//!
//! # Example
//!
//! ```bash
//! # Train a model
//! monolith train --model-dir /path/to/model --config /path/to/config.json --train-steps 10000
//!
//! # Serve a model
//! monolith serve --model-dir /path/to/model --port 8500 --workers 4
//!
//! # Export a checkpoint
//! monolith export --checkpoint-path /path/to/ckpt --output-path /path/to/export
//! ```

pub mod commands;
pub mod gflags_utils;

use clap::{Parser, Subcommand};

pub use commands::{
    AgentServiceCommand, ExportCommand, ServeCommand, TfRunnerCommand, TrainCommand,
};

/// Monolith - A high-performance recommendation system
///
/// Provides tools for training, serving, and exporting deep learning models
/// optimized for large-scale recommendation workloads.
#[derive(Parser, Debug)]
#[command(name = "monolith")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// The subcommand to execute
    #[command(subcommand)]
    pub command: Commands,
}

/// Available CLI commands
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Train a model using distributed training
    Train(TrainCommand),

    /// Serve a model for inference via gRPC
    Serve(ServeCommand),

    /// Export a checkpoint for deployment
    Export(ExportCommand),

    /// Agent-service utilities (controller, discovery client, and agent runner)
    AgentService(AgentServiceCommand),

    /// TensorFlow runner-style entrypoints (Python parity for gpu_runner/tpu_runner).
    TfRunner(TfRunnerCommand),
}

/// Common configuration options shared across commands
#[derive(Debug, Clone)]
pub struct CommonConfig {
    /// Verbosity level for logging
    pub verbosity: u8,

    /// Whether to run in dry-run mode
    pub dry_run: bool,
}

impl Default for CommonConfig {
    fn default() -> Self {
        Self {
            verbosity: 1,
            dry_run: false,
        }
    }
}

/// Result type alias for CLI operations
pub type CliResult<T> = anyhow::Result<T>;
