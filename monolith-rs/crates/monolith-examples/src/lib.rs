//! Example applications for Monolith-RS.
//!
//! This crate contains example binaries demonstrating various Monolith-RS features:
//!
//! - `stream_training` - Streaming training with Kafka (or mock mode)
//!
//! # Running Examples
//!
//! ```bash
//! # Run stream training with mock data
//! cargo run -p monolith-examples --bin stream_training -- --use-mock
//!
//! # Run with real Kafka
//! cargo run -p monolith-examples --bin stream_training -- \
//!     --kafka-brokers localhost:9092 \
//!     --topic movie-train \
//!     --group-id stream-trainer
//! ```

// This crate is primarily for examples, no lib code needed.
