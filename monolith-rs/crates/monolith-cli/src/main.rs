//! Monolith CLI - Command-line interface for training, serving, and exporting models.
//!
//! This binary provides a unified interface to the Monolith recommendation system,
//! supporting distributed training, model serving, and checkpoint export operations.

use anyhow::Result;
use clap::Parser;
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use monolith_cli::{Cli, Commands};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing subscriber with environment filter
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive("monolith=info".parse()?))
        .init();

    // Parse command-line arguments
    let cli = Cli::parse();

    info!("Monolith CLI starting...");

    // Dispatch to appropriate subcommand
    match cli.command {
        Commands::Train(cmd) => cmd.run().await?,
        Commands::Serve(cmd) => cmd.run().await?,
        Commands::Export(cmd) => cmd.run().await?,
    }

    info!("Monolith CLI completed successfully");
    Ok(())
}
