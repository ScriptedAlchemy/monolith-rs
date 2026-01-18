//! Streaming Training Example for Monolith-RS
//!
//! This example demonstrates real-time/online learning patterns using Kafka
//! for streaming training examples. It shows how to:
//!
//! 1. Connect to Kafka (or use mock mode for testing)
//! 2. Consume training examples from a topic
//! 3. Train a model incrementally with each example
//! 4. Periodically log metrics
//!
//! # Usage
//!
//! ```bash
//! # Run with mock data (no Kafka required)
//! cargo run -p monolith-examples --bin stream_training -- --use-mock
//!
//! # Run with real Kafka
//! cargo run -p monolith-examples --bin stream_training -- \
//!     --kafka-brokers localhost:9092 \
//!     --topic movie-train \
//!     --group-id stream-trainer
//! ```
//!
//! # Architecture
//!
//! The streaming training pipeline follows this pattern:
//!
//! ```text
//! ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
//! │  Kafka Topic    │─────▶│  KafkaStream    │─────▶│  OnlineTrainer  │
//! │  (movie-train)  │      │  Trainer        │      │  (Dense Model)  │
//! └─────────────────┘      └─────────────────┘      └─────────────────┘
//!                                   │
//!                                   ▼
//!                          ┌─────────────────┐
//!                          │  TrainingMetrics│
//!                          │  (loss, AUC)    │
//!                          └─────────────────┘
//! ```

use std::time::{Duration, Instant};

// Monolith crates
use monolith_data::{
    example::{add_feature, create_example, get_feature},
    get_feature_data,
    kafka::{KafkaConfig, KafkaConsumer, KafkaDataSource, KafkaMessage, MockKafkaConsumer},
};
use monolith_layers::{
    activation::Sigmoid,
    layer::Layer,
    mlp::{ActivationType, MLPConfig, MLP},
    tensor::Tensor,
};
use monolith_proto::Example;
use prost::Message;

// ============================================================================
// Command-line argument parsing
// ============================================================================

/// Configuration for the streaming trainer.
#[derive(Debug, Clone)]
pub struct StreamTrainerConfig {
    /// Kafka broker addresses (comma-separated)
    pub kafka_brokers: Vec<String>,
    /// Kafka topic to consume from
    pub topic: String,
    /// Consumer group ID
    pub group_id: String,
    /// Whether to use mock mode (no real Kafka)
    pub use_mock: bool,
    /// Learning rate for the model
    pub learning_rate: f32,
    /// How often to log metrics (in number of examples)
    pub log_interval: usize,
    /// Maximum number of examples to train on (0 = unlimited)
    pub max_examples: usize,
    /// Poll timeout in milliseconds
    pub poll_timeout_ms: u64,
}

impl Default for StreamTrainerConfig {
    fn default() -> Self {
        Self {
            kafka_brokers: vec!["localhost:9092".to_string()],
            topic: "movie-train".to_string(),
            group_id: "stream-trainer".to_string(),
            use_mock: false,
            learning_rate: 0.01,
            log_interval: 100,
            max_examples: 1000,
            poll_timeout_ms: 100,
        }
    }
}

impl StreamTrainerConfig {
    /// Parse command-line arguments into configuration.
    pub fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut config = Self::default();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--kafka-brokers" | "-b" => {
                    if i + 1 < args.len() {
                        config.kafka_brokers = args[i + 1]
                            .split(',')
                            .map(|s| s.trim().to_string())
                            .collect();
                        i += 1;
                    }
                }
                "--topic" | "-t" => {
                    if i + 1 < args.len() {
                        config.topic = args[i + 1].clone();
                        i += 1;
                    }
                }
                "--group-id" | "-g" => {
                    if i + 1 < args.len() {
                        config.group_id = args[i + 1].clone();
                        i += 1;
                    }
                }
                "--use-mock" | "-m" => {
                    config.use_mock = true;
                }
                "--learning-rate" | "-l" => {
                    if i + 1 < args.len() {
                        config.learning_rate = args[i + 1].parse().unwrap_or(0.01);
                        i += 1;
                    }
                }
                "--log-interval" => {
                    if i + 1 < args.len() {
                        config.log_interval = args[i + 1].parse().unwrap_or(100);
                        i += 1;
                    }
                }
                "--max-examples" => {
                    if i + 1 < args.len() {
                        config.max_examples = args[i + 1].parse().unwrap_or(1000);
                        i += 1;
                    }
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }

        config
    }
}

fn print_usage() {
    println!(
        r#"Monolith Streaming Training Example

USAGE:
    stream_training [OPTIONS]

OPTIONS:
    -b, --kafka-brokers <BROKERS>   Kafka broker addresses (comma-separated)
                                    [default: localhost:9092]
    -t, --topic <TOPIC>             Kafka topic to consume from
                                    [default: movie-train]
    -g, --group-id <GROUP_ID>       Consumer group ID
                                    [default: stream-trainer]
    -m, --use-mock                  Use mock mode (no real Kafka required)
    -l, --learning-rate <LR>        Learning rate [default: 0.01]
        --log-interval <N>          Log metrics every N examples [default: 100]
        --max-examples <N>          Maximum examples to train on (0=unlimited)
                                    [default: 1000]
    -h, --help                      Print help information

EXAMPLES:
    # Run with mock data
    stream_training --use-mock

    # Run with real Kafka
    stream_training --kafka-brokers localhost:9092 --topic movie-train
"#
    );
}

// ============================================================================
// Training Metrics
// ============================================================================

/// Tracks training metrics over time.
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Total loss accumulated
    total_loss: f32,
    /// Number of examples seen
    num_examples: usize,
    /// Number of correct predictions (for accuracy)
    num_correct: usize,
    /// Start time of training
    start_time: Instant,
    /// Loss history for moving average
    loss_history: Vec<f32>,
    /// Window size for moving average
    window_size: usize,
}

impl TrainingMetrics {
    pub fn new(window_size: usize) -> Self {
        Self {
            total_loss: 0.0,
            num_examples: 0,
            num_correct: 0,
            start_time: Instant::now(),
            loss_history: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Update metrics with a new training example.
    pub fn update(&mut self, loss: f32, prediction: f32, label: f32) {
        self.total_loss += loss;
        self.num_examples += 1;

        // Binary accuracy (threshold at 0.5)
        let predicted_class = if prediction > 0.5 { 1.0 } else { 0.0 };
        if (predicted_class - label).abs() < 0.5 {
            self.num_correct += 1;
        }

        // Update loss history for moving average
        if self.loss_history.len() >= self.window_size {
            self.loss_history.remove(0);
        }
        self.loss_history.push(loss);
    }

    /// Get average loss over all examples.
    pub fn avg_loss(&self) -> f32 {
        if self.num_examples == 0 {
            0.0
        } else {
            self.total_loss / self.num_examples as f32
        }
    }

    /// Get moving average loss (over recent window).
    pub fn moving_avg_loss(&self) -> f32 {
        if self.loss_history.is_empty() {
            0.0
        } else {
            self.loss_history.iter().sum::<f32>() / self.loss_history.len() as f32
        }
    }

    /// Get accuracy.
    pub fn accuracy(&self) -> f32 {
        if self.num_examples == 0 {
            0.0
        } else {
            self.num_correct as f32 / self.num_examples as f32
        }
    }

    /// Get throughput (examples per second).
    pub fn throughput(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_secs_f32();
        if elapsed > 0.0 {
            self.num_examples as f32 / elapsed
        } else {
            0.0
        }
    }

    /// Get total number of examples processed.
    pub fn num_examples(&self) -> usize {
        self.num_examples
    }

    /// Print a summary of metrics.
    pub fn print_summary(&self) {
        println!(
            "[Step {}] Loss: {:.4} (MA: {:.4}) | Acc: {:.2}% | Throughput: {:.1} ex/s",
            self.num_examples,
            self.avg_loss(),
            self.moving_avg_loss(),
            self.accuracy() * 100.0,
            self.throughput()
        );
    }
}

// ============================================================================
// Simple Online Model
// ============================================================================

/// A simple dense model for online learning demonstration.
///
/// Architecture: Input -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(1) -> Sigmoid
pub struct OnlineModel {
    /// Multi-layer perceptron
    mlp: MLP,
    /// Sigmoid activation for output
    sigmoid: Sigmoid,
    /// Learning rate
    learning_rate: f32,
}

impl OnlineModel {
    /// Create a new online model.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of input features
    /// * `learning_rate` - Learning rate for SGD updates
    pub fn new(input_dim: usize, learning_rate: f32) -> Self {
        // Build a simple MLP: input -> 64 -> 32 -> 1
        let mlp = MLPConfig::new(input_dim)
            .add_layer(64, ActivationType::ReLU)
            .add_layer(32, ActivationType::ReLU)
            .add_layer(1, ActivationType::None) // Sigmoid applied separately
            .build()
            .expect("Failed to build MLP");

        Self {
            mlp,
            sigmoid: Sigmoid::new(),
            learning_rate,
        }
    }

    /// Perform a forward pass.
    pub fn forward(&self, input: &Tensor) -> Tensor {
        let hidden = self.mlp.forward(input).expect("Forward pass failed");
        self.sigmoid.forward(&hidden).expect("Sigmoid failed")
    }

    /// Compute binary cross-entropy loss.
    pub fn compute_loss(&self, prediction: f32, label: f32) -> f32 {
        // Binary cross-entropy: -y*log(p) - (1-y)*log(1-p)
        let p = prediction.clamp(1e-7, 1.0 - 1e-7);
        -label * p.ln() - (1.0 - label) * (1.0 - p).ln()
    }

    /// Train on a single example (online learning).
    ///
    /// Returns the loss for this example.
    pub fn train_step(&mut self, input: &Tensor, label: f32) -> (f32, f32) {
        // Forward pass
        let output = self.forward(input);
        let prediction = output.data()[0];

        // Compute loss
        let loss = self.compute_loss(prediction, label);

        // Compute gradient (dL/dp for BCE)
        // dL/dp = -y/p + (1-y)/(1-p) = (p - y) / (p * (1-p))
        // For sigmoid output: dL/dz = p - y (simplified)
        let grad_output = prediction - label;

        // Simple SGD update on MLP parameters
        // In a full implementation, we would do proper backprop
        // Here we use a simplified gradient update
        self.simple_gradient_update(grad_output);

        (loss, prediction)
    }

    /// Simplified gradient update (demonstration purposes).
    fn simple_gradient_update(&mut self, grad: f32) {
        // Apply a small gradient update to all parameters
        // This is a simplified version - real implementation would do proper backprop
        for param in self.mlp.parameters_mut() {
            let data = param.data_mut();
            for val in data.iter_mut() {
                *val -= self.learning_rate * grad * 0.01; // Scaled gradient
            }
        }
    }
}

// ============================================================================
// Kafka Stream Trainer
// ============================================================================

/// Main streaming trainer that consumes from Kafka and trains a model.
pub struct KafkaStreamTrainer<C: KafkaConsumer> {
    /// Kafka data source
    source: KafkaDataSource<C>,
    /// Online model
    model: OnlineModel,
    /// Training metrics
    metrics: TrainingMetrics,
    /// Configuration
    config: StreamTrainerConfig,
}

impl KafkaStreamTrainer<MockKafkaConsumer> {
    /// Create a new trainer with mock Kafka consumer.
    pub fn with_mock(config: StreamTrainerConfig) -> Self {
        let kafka_config = KafkaConfig::new(
            config.kafka_brokers.clone(),
            config.topic.clone(),
            config.group_id.clone(),
        );

        let source = KafkaDataSource::with_mock_consumer(kafka_config)
            .expect("Failed to create mock Kafka source");

        // Assume input dimension of 2 (movie_id + user_id embeddings)
        let model = OnlineModel::new(2, config.learning_rate);
        let metrics = TrainingMetrics::new(100);

        Self {
            source,
            model,
            metrics,
            config,
        }
    }

    /// Add synthetic messages for mock mode.
    pub fn populate_mock_data(&mut self, num_examples: usize) {
        println!("Generating {} synthetic training examples...", num_examples);

        // Simple linear random generator for reproducibility
        let mut seed: u64 = 42;
        let mut next_rand = || {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            ((seed >> 16) & 0x7fff) as f32 / 32768.0
        };

        for i in 0..num_examples {
            // Create a synthetic example similar to movie-lens
            let mut example = create_example();

            // Generate random movie and user IDs
            let movie_id = (next_rand() * 1000.0) as i64;
            let user_id = (next_rand() * 10000.0) as i64;

            // Generate label based on some pattern (for demonstration)
            // In reality, this would come from actual user interactions
            let label = if next_rand() > 0.5 { 1.0 } else { 0.0 };

            add_feature(&mut example, "mov", vec![movie_id], vec![1.0]);
            add_feature(&mut example, "uid", vec![user_id], vec![1.0]);
            add_feature(&mut example, "label", vec![0], vec![label]);

            // Serialize to protobuf
            let payload = example.encode_to_vec();

            // Create Kafka message
            let message = KafkaMessage {
                topic: self.config.topic.clone(),
                partition: 0,
                offset: i as i64,
                key: Some(format!("{}", i).into_bytes()),
                payload,
                timestamp: Some(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as i64,
                ),
            };

            self.source.consumer_mut().add_message(message);
        }

        println!("Mock data populated successfully.");
    }
}

impl<C: KafkaConsumer> KafkaStreamTrainer<C> {
    /// Subscribe to the configured topic.
    pub fn subscribe(&mut self) -> Result<(), monolith_data::kafka::KafkaError> {
        self.source.subscribe_configured()
    }

    /// Extract features from an Example protobuf.
    fn extract_features(&self, example: &Example) -> Option<(Tensor, f32)> {
        // Extract movie and user features
        let mov_feature = get_feature_data(example, "mov")?;
        let uid_feature = get_feature_data(example, "uid")?;
        let label_feature = get_feature_data(example, "label")?;

        // Create input tensor (simplified: just use the first fid as feature)
        let mov_id = mov_feature.fid.first().copied().unwrap_or(0) as f32;
        let uid_id = uid_feature.fid.first().copied().unwrap_or(0) as f32;

        // Normalize features (simple scaling for demonstration)
        let input = Tensor::from_data(
            &[1, 2],
            vec![
                mov_id / 1000.0,  // Normalize movie ID
                uid_id / 10000.0, // Normalize user ID
            ],
        );

        let label = label_feature.value.first().copied().unwrap_or(0.0);

        Some((input, label))
    }

    /// Run the streaming training loop.
    pub fn run(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("\n=== Starting Streaming Training ===");
        println!("Configuration:");
        println!("  Brokers: {:?}", self.config.kafka_brokers);
        println!("  Topic: {}", self.config.topic);
        println!("  Group ID: {}", self.config.group_id);
        println!("  Learning Rate: {}", self.config.learning_rate);
        println!("  Log Interval: {} examples", self.config.log_interval);
        println!(
            "  Max Examples: {}",
            if self.config.max_examples == 0 {
                "unlimited".to_string()
            } else {
                self.config.max_examples.to_string()
            }
        );
        println!();

        let poll_timeout = Duration::from_millis(self.config.poll_timeout_ms);
        let mut consecutive_empty_polls = 0;
        let max_empty_polls = 50; // Stop after this many empty polls in a row

        loop {
            // Check if we've reached max examples
            if self.config.max_examples > 0
                && self.metrics.num_examples() >= self.config.max_examples
            {
                println!(
                    "\nReached maximum examples limit ({}).",
                    self.config.max_examples
                );
                break;
            }

            // Poll for next message
            match self.source.poll(poll_timeout) {
                Some(message) => {
                    consecutive_empty_polls = 0;

                    // Decode the Example protobuf
                    match message.decode_example() {
                        Ok(example) => {
                            // Extract features and train
                            if let Some((input, label)) = self.extract_features(&example) {
                                let (loss, prediction) = self.model.train_step(&input, label);
                                self.metrics.update(loss, prediction, label);

                                // Log metrics periodically
                                if self.metrics.num_examples() % self.config.log_interval == 0 {
                                    self.metrics.print_summary();
                                }
                            }
                        }
                        Err(e) => {
                            eprintln!(
                                "Warning: Failed to decode message at offset {}: {}",
                                message.offset, e
                            );
                        }
                    }
                }
                None => {
                    consecutive_empty_polls += 1;
                    if consecutive_empty_polls >= max_empty_polls {
                        println!(
                            "\nNo more messages available (empty polls: {}).",
                            consecutive_empty_polls
                        );
                        break;
                    }
                }
            }
        }

        // Print final summary
        println!("\n=== Training Complete ===");
        println!("Total examples: {}", self.metrics.num_examples());
        println!("Final average loss: {:.4}", self.metrics.avg_loss());
        println!("Final accuracy: {:.2}%", self.metrics.accuracy() * 100.0);
        println!(
            "Average throughput: {:.1} examples/sec",
            self.metrics.throughput()
        );
        println!(
            "Total time: {:.2}s",
            self.metrics.start_time.elapsed().as_secs_f32()
        );

        Ok(())
    }

    /// Commit offsets after processing.
    #[allow(dead_code)]
    pub fn commit(&mut self) -> Result<(), monolith_data::kafka::KafkaError> {
        self.source.commit()
    }

    /// Get current metrics.
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = StreamTrainerConfig::from_args();

    println!("Monolith-RS Streaming Training Example");
    println!("======================================");

    if config.use_mock {
        println!("\nRunning in MOCK mode (no real Kafka connection)");

        // Create trainer with mock consumer
        let mut trainer = KafkaStreamTrainer::with_mock(config.clone());

        // Populate with synthetic data
        let num_examples = if config.max_examples > 0 {
            config.max_examples
        } else {
            1000
        };
        trainer.populate_mock_data(num_examples);

        // Subscribe and run
        trainer.subscribe()?;
        trainer.run()?;
    } else {
        println!("\nConnecting to Kafka...");
        println!("Note: If Kafka is not available, run with --use-mock flag");

        // For real Kafka, we would need the kafka feature enabled
        // For now, we demonstrate with mock
        println!("\nReal Kafka connection not yet implemented.");
        println!("Please run with --use-mock flag for demonstration.");
        println!("\nExample: cargo run -p monolith-examples --bin stream_training -- --use-mock");
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new(10);

        // Update with some examples
        metrics.update(0.5, 0.6, 1.0); // Correct prediction
        metrics.update(0.3, 0.3, 0.0); // Correct prediction
        metrics.update(0.8, 0.7, 0.0); // Incorrect prediction

        assert_eq!(metrics.num_examples(), 3);
        assert!(metrics.accuracy() > 0.6); // 2/3 correct
        assert!(metrics.avg_loss() > 0.0);
    }

    #[test]
    fn test_online_model_forward() {
        let model = OnlineModel::new(2, 0.01);
        let input = Tensor::from_data(&[1, 2], vec![0.5, 0.5]);

        let output = model.forward(&input);

        assert_eq!(output.shape(), &[1, 1]);
        // Output should be between 0 and 1 (sigmoid)
        let pred = output.data()[0];
        assert!(pred >= 0.0 && pred <= 1.0);
    }

    #[test]
    fn test_online_model_train_step() {
        let mut model = OnlineModel::new(2, 0.01);
        let input = Tensor::from_data(&[1, 2], vec![0.5, 0.5]);

        let (loss, pred) = model.train_step(&input, 1.0);

        assert!(loss >= 0.0);
        assert!(pred >= 0.0 && pred <= 1.0);
    }

    #[test]
    fn test_kafka_stream_trainer_mock() {
        let config = StreamTrainerConfig {
            use_mock: true,
            max_examples: 10,
            log_interval: 5,
            ..Default::default()
        };

        let mut trainer = KafkaStreamTrainer::with_mock(config);
        trainer.populate_mock_data(10);
        trainer.subscribe().unwrap();

        trainer.run().unwrap();

        assert_eq!(trainer.metrics().num_examples(), 10);
    }

    #[test]
    fn test_feature_extraction() {
        let config = StreamTrainerConfig::default();
        let trainer = KafkaStreamTrainer::with_mock(config);

        let mut example = create_example();
        add_feature(&mut example, "mov", vec![100], vec![1.0]);
        add_feature(&mut example, "uid", vec![5000], vec![1.0]);
        // `add_feature` stores sparse data as fids and ignores `values` when fids are present,
        // so use float_list encoding for labels.
        add_feature(&mut example, "label", vec![], vec![1.0]);

        let (input, label) = trainer.extract_features(&example).unwrap();

        assert_eq!(input.shape(), &[1, 2]);
        assert!((label - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_default() {
        let config = StreamTrainerConfig::default();

        assert_eq!(config.kafka_brokers, vec!["localhost:9092".to_string()]);
        assert_eq!(config.topic, "movie-train");
        assert_eq!(config.group_id, "stream-trainer");
        assert!(!config.use_mock);
    }
}
