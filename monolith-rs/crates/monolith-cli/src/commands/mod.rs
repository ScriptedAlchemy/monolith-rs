//! CLI Command Implementations
//!
//! This module contains the implementations for all CLI subcommands:
//!
//! - [`train`]: Model training with the Monolith estimator
//! - [`serve`]: gRPC model serving
//! - [`export`]: Checkpoint export for deployment

mod agent_service;
mod export;
mod serve;
mod tf_runner;
mod train;

pub use agent_service::AgentServiceCommand;
pub use export::ExportCommand;
pub use serve::ServeCommand;
pub use tf_runner::TfRunnerCommand;
pub use train::TrainCommand;
