//! Local predictor example for Monolith-RS.
//!
//! This example demonstrates how to:
//! 1. Load a saved model checkpoint
//! 2. Create a Predictor struct for inference
//! 3. Generate synthetic test data with sparse features
//! 4. Run predictions using embedding lookups and forward pass
//! 5. Benchmark inference latency and throughput
//!
//! # Usage
//!
//! ```bash
//! # Run with default settings (generates demo checkpoint)
//! cargo run --example local_predictor
//!
//! # Run with custom model path
//! cargo run --example local_predictor -- --model-path /path/to/checkpoint
//!
//! # Run benchmark mode
//! cargo run --example local_predictor -- --benchmark --num-batches 1000
//!
//! # Specify batch size
//! cargo run --example local_predictor -- --batch-size 256 --benchmark
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

// Import from monolith crates
use monolith_checkpoint::{HashTableState, ModelExporter, ModelState};
use monolith_layers::embedding::EmbeddingHashTable;
use monolith_layers::layer::Layer;
use monolith_layers::mlp::{ActivationType, MLPConfig, MLP};
use monolith_layers::tensor::Tensor;
use monolith_layers::Sigmoid;

// ============================================================================
// Configuration
// ============================================================================

/// Model configuration constants.
mod config {
    /// Number of feature slots (similar to Python's _NUM_SLOTS).
    pub const NUM_SLOTS: usize = 10;

    /// Vocabulary size per slot.
    pub const VOCAB_SIZE: u64 = 10000;

    /// Embedding dimension.
    pub const EMBEDDING_DIM: usize = 16;

    /// MLP hidden layer sizes.
    pub const MLP_HIDDEN: &[usize] = &[64, 32];

    /// Number of sparse features per instance.
    pub const FEATURES_PER_INSTANCE: usize = 5;
}

/// Command-line arguments.
#[derive(Debug, Clone)]
struct Args {
    /// Path to the model checkpoint directory.
    model_path: Option<PathBuf>,

    /// Batch size for inference.
    batch_size: usize,

    /// Number of batches to process.
    num_batches: usize,

    /// Whether to run in benchmark mode.
    benchmark: bool,

    /// Whether to generate a demo checkpoint if none exists.
    generate_demo: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            model_path: None,
            batch_size: 128,
            num_batches: 100,
            benchmark: false,
            generate_demo: true,
        }
    }
}

impl Args {
    /// Parse command-line arguments.
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut result = Self::default();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--model-path" | "-m" => {
                    i += 1;
                    if i < args.len() {
                        result.model_path = Some(PathBuf::from(&args[i]));
                    }
                }
                "--batch-size" | "-b" => {
                    i += 1;
                    if i < args.len() {
                        result.batch_size = args[i].parse().unwrap_or(128);
                    }
                }
                "--num-batches" | "-n" => {
                    i += 1;
                    if i < args.len() {
                        result.num_batches = args[i].parse().unwrap_or(100);
                    }
                }
                "--benchmark" => {
                    result.benchmark = true;
                }
                "--no-demo" => {
                    result.generate_demo = false;
                }
                "--help" | "-h" => {
                    println!("Monolith Local Predictor Example");
                    println!();
                    println!("USAGE:");
                    println!("    cargo run --example local_predictor [OPTIONS]");
                    println!();
                    println!("OPTIONS:");
                    println!("    -m, --model-path <PATH>    Path to model checkpoint directory");
                    println!(
                        "    -b, --batch-size <SIZE>    Batch size for inference (default: 128)"
                    );
                    println!("    -n, --num-batches <NUM>    Number of batches to process (default: 100)");
                    println!(
                        "    --benchmark                Run benchmark mode with timing statistics"
                    );
                    println!(
                        "    --no-demo                  Don't generate demo checkpoint if missing"
                    );
                    println!("    -h, --help                 Print this help message");
                    std::process::exit(0);
                }
                _ => {}
            }
            i += 1;
        }

        result
    }
}

// ============================================================================
// Feature ID (FID) Generation
// ============================================================================

/// Creates a feature ID (FID) from slot ID and feature value.
///
/// This mirrors the Python `make_fid_v1` function:
/// `fid = (slot_id << 54) | feature_value`
#[inline]
fn make_fid_v1(slot_id: u32, feature_value: u64) -> u64 {
    ((slot_id as u64) << 54) | (feature_value & ((1u64 << 54) - 1))
}

/// Extracts the slot ID from a FID.
#[inline]
fn extract_slot_from_fid(fid: u64) -> u32 {
    (fid >> 54) as u32
}

/// Extracts the feature value from a FID.
#[inline]
fn extract_feature_from_fid(fid: u64) -> u64 {
    fid & ((1u64 << 54) - 1)
}

// ============================================================================
// Instance Generation
// ============================================================================

/// Represents a training/inference instance with sparse features.
#[derive(Debug, Clone)]
struct Instance {
    /// Feature IDs (FIDs) for sparse features.
    fids: Vec<u64>,
}

impl Instance {
    /// Generates a demo instance with random sparse features.
    ///
    /// Similar to Python's `generate_demo_instance()`:
    /// - Creates FIDs for each slot
    /// - Uses random feature values within vocabulary bounds
    fn generate_random(rng_seed: u64) -> Self {
        let mut seed = rng_seed;
        let mut fids = Vec::with_capacity(config::NUM_SLOTS * config::FEATURES_PER_INSTANCE);

        for slot_id in 0..config::NUM_SLOTS {
            for _ in 0..config::FEATURES_PER_INSTANCE {
                // Simple LCG random number generator
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let feature_value = (seed >> 16) % config::VOCAB_SIZE;
                let fid = make_fid_v1(slot_id as u32, feature_value);
                fids.push(fid);
            }
        }

        Instance { fids }
    }
}

/// Generates a batch of random instances.
fn generate_batch(batch_size: usize, batch_idx: usize) -> Vec<Instance> {
    (0..batch_size)
        .map(|i| {
            let seed = ((batch_idx * batch_size + i) as u64).wrapping_mul(0x5DEECE66D);
            Instance::generate_random(seed)
        })
        .collect()
}

// ============================================================================
// Predictor
// ============================================================================

/// Predictor for running inference on exported Monolith models.
///
/// The Predictor loads model state from a checkpoint and provides
/// forward-only inference capabilities.
struct Predictor {
    /// Embedding tables for sparse feature lookup.
    embedding_tables: HashMap<String, EmbeddingHashTable>,

    /// MLP network for scoring.
    mlp: MLP,

    /// Sigmoid activation for final output.
    sigmoid: Sigmoid,

    /// Model metadata.
    global_step: u64,
}

impl Predictor {
    /// Creates a new Predictor from a model checkpoint directory.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the exported model directory
    ///
    /// # Returns
    ///
    /// A Result containing the Predictor or an error message.
    fn from_checkpoint(model_path: &Path) -> Result<Self, String> {
        println!("Loading model from: {}", model_path.display());

        // Load manifest to get model info
        let manifest = ModelExporter::load_manifest(model_path)
            .map_err(|e| format!("Failed to load manifest: {}", e))?;

        println!("  Model version: {}", manifest.version);
        println!("  Global step: {}", manifest.global_step);
        println!("  Embedding tables: {:?}", manifest.embedding_tables);
        println!("  Dense params: {:?}", manifest.dense_params);

        // Load embedding tables
        let mut embedding_tables = HashMap::new();
        for table_name in &manifest.embedding_tables {
            let table_state = ModelExporter::load_embedding_table(model_path, table_name)
                .map_err(|e| format!("Failed to load embedding table '{}': {}", table_name, e))?;

            let mut hash_table = EmbeddingHashTable::new(table_state.dim);
            for (fid, embedding) in table_state.entries {
                hash_table.insert(fid as u64, embedding);
            }

            println!(
                "  Loaded table '{}': {} entries, dim={}",
                table_name,
                hash_table.len(),
                table_state.dim
            );
            embedding_tables.insert(table_name.clone(), hash_table);
        }

        // Load dense parameters and reconstruct MLP
        // Note: In production, these would be used to initialize the MLP weights
        let _dense_params = ModelExporter::load_dense_params(model_path)
            .map_err(|e| format!("Failed to load dense params: {}", e))?;

        // Calculate input dimension from embedding tables
        let total_embedding_dim = config::NUM_SLOTS * config::EMBEDDING_DIM;

        // Create MLP with the same architecture
        let mlp = MLPConfig::new(total_embedding_dim)
            .add_layer(config::MLP_HIDDEN[0], ActivationType::ReLU)
            .add_layer(config::MLP_HIDDEN[1], ActivationType::ReLU)
            .add_layer(1, ActivationType::None)
            .build()
            .map_err(|e| format!("Failed to create MLP: {:?}", e))?;

        // Note: In a production system, you would load the actual weights
        // from dense_params and initialize the MLP with them.
        // For this example, we use the randomly initialized weights.

        println!(
            "  MLP created: {} -> {:?} -> 1",
            total_embedding_dim,
            config::MLP_HIDDEN
        );

        Ok(Predictor {
            embedding_tables,
            mlp,
            sigmoid: Sigmoid::new(),
            global_step: manifest.global_step,
        })
    }

    /// Creates a Predictor from a ModelState directly (for demo purposes).
    fn from_model_state(state: &ModelState) -> Result<Self, String> {
        println!("Creating predictor from model state...");
        println!("  Global step: {}", state.global_step);

        // Load embedding tables
        let mut embedding_tables = HashMap::new();
        for table_state in &state.hash_tables {
            let mut hash_table = EmbeddingHashTable::new(table_state.dim);
            for (fid, embedding) in &table_state.entries {
                hash_table.insert(*fid as u64, embedding.clone());
            }

            println!(
                "  Loaded table '{}': {} entries, dim={}",
                table_state.name,
                hash_table.len(),
                table_state.dim
            );
            embedding_tables.insert(table_state.name.clone(), hash_table);
        }

        // Calculate input dimension from embedding tables
        let total_embedding_dim = config::NUM_SLOTS * config::EMBEDDING_DIM;

        // Create MLP
        let mlp = MLPConfig::new(total_embedding_dim)
            .add_layer(config::MLP_HIDDEN[0], ActivationType::ReLU)
            .add_layer(config::MLP_HIDDEN[1], ActivationType::ReLU)
            .add_layer(1, ActivationType::None)
            .build()
            .map_err(|e| format!("Failed to create MLP: {:?}", e))?;

        Ok(Predictor {
            embedding_tables,
            mlp,
            sigmoid: Sigmoid::new(),
            global_step: state.global_step,
        })
    }

    /// Looks up embeddings for a batch of instances.
    ///
    /// # Arguments
    ///
    /// * `instances` - Batch of instances with sparse features
    ///
    /// # Returns
    ///
    /// Tensor of shape [batch_size, total_embedding_dim]
    fn lookup_embeddings(&self, instances: &[Instance]) -> Tensor {
        let batch_size = instances.len();
        let total_dim = config::NUM_SLOTS * config::EMBEDDING_DIM;
        let mut data = vec![0.0f32; batch_size * total_dim];

        // Get the shared embedding table (all slots share the same table in this example)
        let table = self
            .embedding_tables
            .values()
            .next()
            .expect("No embedding tables loaded");

        for (batch_idx, instance) in instances.iter().enumerate() {
            // Group FIDs by slot and sum/pool embeddings
            let mut slot_embeddings: HashMap<u32, Vec<f32>> = HashMap::new();

            for &fid in &instance.fids {
                let slot_id = extract_slot_from_fid(fid);
                let embedding = table.get(fid);

                slot_embeddings
                    .entry(slot_id)
                    .and_modify(|e| {
                        // Sum pooling for features in the same slot
                        for (i, &val) in embedding.iter().enumerate() {
                            e[i] += val;
                        }
                    })
                    .or_insert_with(|| embedding.to_vec());
            }

            // Concatenate slot embeddings
            for slot_id in 0..config::NUM_SLOTS {
                let slot_embedding = slot_embeddings
                    .get(&(slot_id as u32))
                    .cloned()
                    .unwrap_or_else(|| vec![0.0; config::EMBEDDING_DIM]);

                let start_idx = batch_idx * total_dim + slot_id * config::EMBEDDING_DIM;
                data[start_idx..start_idx + config::EMBEDDING_DIM].copy_from_slice(&slot_embedding);
            }
        }

        Tensor::from_data(&[batch_size, total_dim], data)
    }

    /// Runs prediction on a batch of instances.
    ///
    /// # Arguments
    ///
    /// * `instances` - Batch of instances to predict
    ///
    /// # Returns
    ///
    /// Tensor of prediction scores, shape [batch_size, 1]
    fn predict(&self, instances: &[Instance]) -> Tensor {
        // Step 1: Look up embeddings
        let embeddings = self.lookup_embeddings(instances);

        // Step 2: Forward through MLP
        let logits = self
            .mlp
            .forward(&embeddings)
            .expect("MLP forward pass failed");

        // Step 3: Apply sigmoid for probability output
        let predictions = self
            .sigmoid
            .forward(&logits)
            .expect("Sigmoid forward pass failed");

        predictions
    }

    /// Returns the model's global step.
    fn global_step(&self) -> u64 {
        self.global_step
    }
}

// ============================================================================
// Demo Checkpoint Generation
// ============================================================================

/// Generates a demo checkpoint for testing.
fn generate_demo_checkpoint(output_dir: &Path) -> Result<ModelState, String> {
    println!("Generating demo checkpoint at: {}", output_dir.display());

    // Create model state
    let mut state = ModelState::new(10000);

    // Create embedding table with random embeddings
    let mut table = HashTableState::new("embeddings", config::EMBEDDING_DIM);

    // Pre-populate some embeddings for common FIDs
    let mut seed: u64 = 42;
    for slot_id in 0..config::NUM_SLOTS {
        for feature_value in 0..100 {
            let fid = make_fid_v1(slot_id as u32, feature_value);
            let embedding: Vec<f32> = (0..config::EMBEDDING_DIM)
                .map(|_| {
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let u1 = ((seed >> 16) & 0x7fff) as f32 / 32768.0 + 1e-10;
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let u2 = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    z * 0.1 // Standard deviation 0.1
                })
                .collect();
            table.insert(fid as i64, embedding);
        }
    }

    println!("  Created embedding table with {} entries", table.len());
    state.add_hash_table(table);

    // Add some dense parameters
    let total_embedding_dim = config::NUM_SLOTS * config::EMBEDDING_DIM;
    state.add_dense_param(
        "mlp.layer0.weight",
        vec![0.1; total_embedding_dim * config::MLP_HIDDEN[0]],
    );
    state.add_dense_param("mlp.layer0.bias", vec![0.0; config::MLP_HIDDEN[0]]);
    state.add_dense_param(
        "mlp.layer1.weight",
        vec![0.1; config::MLP_HIDDEN[0] * config::MLP_HIDDEN[1]],
    );
    state.add_dense_param("mlp.layer1.bias", vec![0.0; config::MLP_HIDDEN[1]]);
    state.add_dense_param("mlp.layer2.weight", vec![0.1; config::MLP_HIDDEN[1] * 1]);
    state.add_dense_param("mlp.layer2.bias", vec![0.0; 1]);

    // Add metadata
    state.set_metadata("model_name", "demo_ffm_model");
    state.set_metadata("framework", "monolith-rs");

    // Export the model
    let config = monolith_checkpoint::ExportConfig::new(output_dir)
        .with_format(monolith_checkpoint::ExportFormat::SavedModel)
        .with_version("1.0.0");

    let exporter = ModelExporter::new(config);
    exporter
        .export(&state)
        .map_err(|e| format!("Failed to export model: {}", e))?;

    println!("  Exported model to: {}", output_dir.display());

    Ok(state)
}

// ============================================================================
// Benchmarking
// ============================================================================

/// Statistics collected during benchmarking.
#[derive(Debug)]
struct BenchmarkStats {
    /// Total time spent in inference.
    total_time: Duration,
    /// Individual batch latencies.
    latencies: Vec<Duration>,
    /// Number of samples processed.
    num_samples: usize,
}

impl BenchmarkStats {
    fn new() -> Self {
        Self {
            total_time: Duration::ZERO,
            latencies: Vec::new(),
            num_samples: 0,
        }
    }

    fn record(&mut self, latency: Duration, batch_size: usize) {
        self.latencies.push(latency);
        self.total_time += latency;
        self.num_samples += batch_size;
    }

    fn p50(&self) -> Duration {
        self.percentile(50)
    }

    fn p99(&self) -> Duration {
        self.percentile(99)
    }

    fn percentile(&self, p: usize) -> Duration {
        if self.latencies.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.latencies.clone();
        sorted.sort();
        let idx = (p * sorted.len() / 100).min(sorted.len() - 1);
        sorted[idx]
    }

    fn mean(&self) -> Duration {
        if self.latencies.is_empty() {
            return Duration::ZERO;
        }
        self.total_time / self.latencies.len() as u32
    }

    fn throughput(&self) -> f64 {
        if self.total_time.as_secs_f64() == 0.0 {
            return 0.0;
        }
        self.num_samples as f64 / self.total_time.as_secs_f64()
    }

    fn report(&self) {
        println!();
        println!("=== Benchmark Results ===");
        println!("Total samples:    {}", self.num_samples);
        println!("Total batches:    {}", self.latencies.len());
        println!("Total time:       {:.3} s", self.total_time.as_secs_f64());
        println!();
        println!("Latency Statistics:");
        println!(
            "  Mean:           {:.3} ms",
            self.mean().as_secs_f64() * 1000.0
        );
        println!(
            "  P50:            {:.3} ms",
            self.p50().as_secs_f64() * 1000.0
        );
        println!(
            "  P99:            {:.3} ms",
            self.p99().as_secs_f64() * 1000.0
        );
        println!();
        println!("Throughput:       {:.2} samples/sec", self.throughput());
        println!("========================");
    }
}

/// Runs the benchmark.
fn run_benchmark(predictor: &Predictor, args: &Args) -> BenchmarkStats {
    println!();
    println!("Running benchmark mode...");
    println!("  Batch size: {}", args.batch_size);
    println!("  Num batches: {}", args.num_batches);

    let mut stats = BenchmarkStats::new();

    // Warmup
    println!("  Warming up...");
    for i in 0..10 {
        let batch = generate_batch(args.batch_size, i);
        let _ = predictor.predict(&batch);
    }

    // Benchmark
    println!("  Running benchmark...");
    for batch_idx in 0..args.num_batches {
        let batch = generate_batch(args.batch_size, batch_idx + 10);

        let start = Instant::now();
        let _ = predictor.predict(&batch);
        let elapsed = start.elapsed();

        stats.record(elapsed, args.batch_size);

        if (batch_idx + 1) % 100 == 0 {
            println!("    Processed {} batches...", batch_idx + 1);
        }
    }

    stats
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    println!("=================================");
    println!("Monolith Local Predictor Example");
    println!("=================================");
    println!();

    let args = Args::parse();

    // Determine model path
    let model_path = args
        .model_path
        .clone()
        .unwrap_or_else(|| PathBuf::from("/tmp/monolith_demo_model"));

    // Create predictor
    let predictor = if model_path.exists() {
        println!("Found existing model at: {}", model_path.display());
        Predictor::from_checkpoint(&model_path).expect("Failed to load model from checkpoint")
    } else if args.generate_demo {
        println!("No model found, generating demo checkpoint...");
        let state =
            generate_demo_checkpoint(&model_path).expect("Failed to generate demo checkpoint");
        Predictor::from_model_state(&state).expect("Failed to create predictor from model state")
    } else {
        eprintln!("Error: Model path does not exist: {}", model_path.display());
        eprintln!("       Use --generate-demo to create a demo model, or provide a valid path.");
        std::process::exit(1);
    };

    println!();
    println!("Predictor ready (global_step={})", predictor.global_step());
    println!();

    if args.benchmark {
        // Run benchmark mode
        let stats = run_benchmark(&predictor, &args);
        stats.report();
    } else {
        // Run demo prediction
        println!("Running demo prediction...");
        println!();

        // Generate a batch of test instances
        let batch_size = args.batch_size;
        println!("Generating {} test instances...", batch_size);
        let instances = generate_batch(batch_size, 0);

        // Show sample instance
        println!();
        println!("Sample instance (first one):");
        println!("  Number of FIDs: {}", instances[0].fids.len());
        println!("  First 5 FIDs:");
        for (i, &fid) in instances[0].fids.iter().take(5).enumerate() {
            let slot = extract_slot_from_fid(fid);
            let feature = extract_feature_from_fid(fid);
            println!(
                "    [{}] FID={}, slot={}, feature={}",
                i, fid, slot, feature
            );
        }

        // Run prediction
        println!();
        println!("Running prediction...");
        let start = Instant::now();
        let predictions = predictor.predict(&instances);
        let elapsed = start.elapsed();

        // Show results
        println!();
        println!("Prediction Results:");
        println!("  Shape: {:?}", predictions.shape());
        println!("  Time: {:.3} ms", elapsed.as_secs_f64() * 1000.0);
        println!();
        println!("First 10 prediction scores:");
        for (i, &score) in predictions.data().iter().take(10).enumerate() {
            println!("  Instance {}: {:.6}", i, score);
        }

        // Show statistics
        let scores = predictions.data();
        let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
        let min = scores.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        println!();
        println!("Score Statistics:");
        println!("  Mean: {:.6}", mean);
        println!("  Min:  {:.6}", min);
        println!("  Max:  {:.6}", max);
    }

    println!();
    println!("Done!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_fid_v1() {
        let fid = make_fid_v1(10, 12345);
        assert_eq!(extract_slot_from_fid(fid), 10);
        assert_eq!(extract_feature_from_fid(fid), 12345);
    }

    #[test]
    fn test_instance_generation() {
        let instance = Instance::generate_random(42);
        assert_eq!(
            instance.fids.len(),
            config::NUM_SLOTS * config::FEATURES_PER_INSTANCE
        );

        // All FIDs should have valid slot IDs
        for &fid in &instance.fids {
            let slot = extract_slot_from_fid(fid);
            assert!(slot < config::NUM_SLOTS as u32);
        }
    }

    #[test]
    fn test_batch_generation() {
        let batch = generate_batch(32, 0);
        assert_eq!(batch.len(), 32);
    }

    #[test]
    fn test_demo_checkpoint_generation() {
        let temp_dir = std::env::temp_dir().join("monolith_test_demo");
        let _ = std::fs::remove_dir_all(&temp_dir);

        let state = generate_demo_checkpoint(&temp_dir).unwrap();
        assert_eq!(state.hash_tables.len(), 1);
        assert!(!state.dense_params.is_empty());

        // Clean up
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_predictor_from_model_state() {
        let mut state = ModelState::new(100);
        let mut table = HashTableState::new("embeddings", config::EMBEDDING_DIM);

        // Add some test embeddings
        for slot_id in 0..config::NUM_SLOTS {
            for i in 0..10 {
                let fid = make_fid_v1(slot_id as u32, i);
                table.insert(fid as i64, vec![0.1; config::EMBEDDING_DIM]);
            }
        }
        state.add_hash_table(table);

        let predictor = Predictor::from_model_state(&state).unwrap();
        assert_eq!(predictor.global_step(), 100);
    }

    #[test]
    fn test_predictor_inference() {
        let mut state = ModelState::new(100);
        let mut table = HashTableState::new("embeddings", config::EMBEDDING_DIM);

        // Add embeddings for test FIDs
        for slot_id in 0..config::NUM_SLOTS {
            for i in 0..config::VOCAB_SIZE.min(100) {
                let fid = make_fid_v1(slot_id as u32, i);
                table.insert(fid as i64, vec![0.1; config::EMBEDDING_DIM]);
            }
        }
        state.add_hash_table(table);

        let predictor = Predictor::from_model_state(&state).unwrap();

        // Generate and predict
        let batch = generate_batch(8, 0);
        let predictions = predictor.predict(&batch);

        assert_eq!(predictions.shape(), &[8, 1]);

        // All predictions should be valid probabilities (0-1 after sigmoid)
        for &score in predictions.data() {
            assert!(score >= 0.0 && score <= 1.0, "Score {} not in [0,1]", score);
        }
    }

    #[test]
    fn test_benchmark_stats() {
        let mut stats = BenchmarkStats::new();

        stats.record(Duration::from_millis(10), 32);
        stats.record(Duration::from_millis(15), 32);
        stats.record(Duration::from_millis(12), 32);

        assert_eq!(stats.num_samples, 96);
        assert_eq!(stats.latencies.len(), 3);
        assert!(stats.throughput() > 0.0);
    }
}
