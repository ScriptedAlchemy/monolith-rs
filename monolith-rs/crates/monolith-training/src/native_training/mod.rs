//! Parity utilities for Python `monolith.native_training.*`.
//!
//! NOTE: Many Python modules here depend on TensorFlow runtime ops. The Rust
//! implementation focuses on TF-free semantics and provides small equivalents
//! where they are used by other Rust crates and parity tests.

pub mod consul;
pub mod device_utils;
pub mod env_utils;
pub mod gen_seq_mask;
pub mod graph_meta;
pub mod graph_utils;
pub mod hvd_lib;
pub mod learning_rate_functions;
pub mod logging_ops;
pub mod nested_tensors;
pub mod ragged_utils;
pub mod save_utils;
pub mod service_discovery;
