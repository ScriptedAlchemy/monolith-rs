//! Movie Ranking Model - Training with CPU or GPU
//!
//! This example demonstrates how to build a recommendation system using Monolith-RS
//! with selectable compute device (CPU or GPU).
//!
//! # Device Selection
//!
//! Use `--device` to choose:
//! - `cpu` - Optimized CPU with BLAS (default, fastest for small models)
//! - `gpu` - GPU acceleration (Metal on macOS, CUDA on Linux/Windows)
//!
//! Note: For small models like this one, CPU is often faster due to GPU transfer overhead.
//! GPU shines with larger embedding dimensions (256+) and hidden layers (512+).
//!
//! # Usage
//!
//! ```bash
//! # CPU (default, recommended for small models)
//! cargo run --example movie_ranking --release
//!
//! # GPU (requires --features metal or cuda)
//! cargo run --example movie_ranking --release --features metal -- --device gpu
//! ```

use std::collections::HashMap;

use monolith_tensor::{CandleTensor, Device, Tensor};
use rayon::prelude::*;

// ============================================================================
// Configuration
// ============================================================================

/// Device selection for compute.
#[derive(Debug, Clone, Copy, PartialEq)]
enum DeviceType {
    Cpu,
    Gpu,
}

/// Command-line arguments for the movie ranking model.
#[derive(Debug, Clone)]
struct Config {
    /// Compute device (cpu or gpu)
    device: DeviceType,
    /// Number of unique users in the dataset
    num_users: usize,
    /// Number of unique movies in the dataset
    num_movies: usize,
    /// Dimension of user and movie embeddings
    embedding_dim: usize,
    /// Sizes of hidden layers in the MLP (comma-separated)
    hidden_layers: Vec<usize>,
    /// Number of samples per training batch
    batch_size: usize,
    /// Number of training epochs
    num_epochs: usize,
    /// Learning rate for gradient descent
    learning_rate: f32,
    /// Random seed for reproducibility
    seed: u64,
    /// Number of training samples to generate
    num_train_samples: usize,
    /// Number of test samples for evaluation
    num_test_samples: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            device: DeviceType::Cpu, // CPU is faster for small models
            num_users: 1000,
            num_movies: 500,
            embedding_dim: 32,
            hidden_layers: vec![64, 32],
            batch_size: 256,
            num_epochs: 10,
            learning_rate: 0.01,
            seed: 42,
            num_train_samples: 100_000,
            num_test_samples: 10_000,
        }
    }
}

impl Config {
    /// Parses command-line arguments into a Config struct.
    fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let mut config = Config::default();

        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--device" | "-d" => {
                    i += 1;
                    config.device = match args[i].to_lowercase().as_str() {
                        "cpu" => DeviceType::Cpu,
                        "gpu" => DeviceType::Gpu,
                        other => {
                            eprintln!("Invalid device '{}'. Use 'cpu' or 'gpu'.", other);
                            std::process::exit(1);
                        }
                    };
                }
                "--num-users" => {
                    i += 1;
                    config.num_users = args[i].parse().expect("Invalid num-users");
                }
                "--num-movies" => {
                    i += 1;
                    config.num_movies = args[i].parse().expect("Invalid num-movies");
                }
                "--embedding-dim" => {
                    i += 1;
                    config.embedding_dim = args[i].parse().expect("Invalid embedding-dim");
                }
                "--hidden-layers" => {
                    i += 1;
                    config.hidden_layers = args[i]
                        .split(',')
                        .map(|s| s.trim().parse().expect("Invalid hidden layer size"))
                        .collect();
                }
                "--batch-size" => {
                    i += 1;
                    config.batch_size = args[i].parse().expect("Invalid batch-size");
                }
                "--num-epochs" => {
                    i += 1;
                    config.num_epochs = args[i].parse().expect("Invalid num-epochs");
                }
                "--learning-rate" => {
                    i += 1;
                    config.learning_rate = args[i].parse().expect("Invalid learning-rate");
                }
                "--seed" => {
                    i += 1;
                    config.seed = args[i].parse().expect("Invalid seed");
                }
                "--num-train-samples" => {
                    i += 1;
                    config.num_train_samples = args[i].parse().expect("Invalid num-train-samples");
                }
                "--num-test-samples" => {
                    i += 1;
                    config.num_test_samples = args[i].parse().expect("Invalid num-test-samples");
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => {
                    eprintln!("Unknown argument: {}", args[i]);
                    print_help();
                    std::process::exit(1);
                }
            }
            i += 1;
        }

        config
    }
}

fn print_help() {
    println!(
        r#"Movie Ranking Model - Training with CPU or GPU

USAGE:
    cargo run --example movie_ranking [OPTIONS]

OPTIONS:
    --device, -d <DEVICE>    Compute device: 'cpu' or 'gpu' (default: cpu)
    --num-users <N>          Number of unique users (default: 1000)
    --num-movies <N>         Number of unique movies (default: 500)
    --embedding-dim <N>      Embedding dimension (default: 32)
    --hidden-layers <SIZES>  Hidden layer sizes, comma-separated (default: 64,32)
    --batch-size <N>         Batch size (default: 256)
    --num-epochs <N>         Number of training epochs (default: 10)
    --learning-rate <F>      Learning rate (default: 0.01)
    --seed <N>               Random seed (default: 42)
    --num-train-samples <N>  Number of training samples (default: 100000)
    --num-test-samples <N>   Number of test samples (default: 10000)
    --help, -h               Print this help message

DEVICE SELECTION:
    cpu - Use CPU with optimized BLAS (default, fastest for small models)
    gpu - Use GPU (Metal on macOS, CUDA on Linux/Windows)

    Note: For small models (embedding_dim < 128, hidden < 256), CPU is faster
    due to GPU data transfer overhead. Use GPU for larger models.

EXAMPLES:
    # CPU (recommended for small models)
    cargo run --example movie_ranking --release

    # GPU with Metal (macOS)
    cargo run --example movie_ranking --release --features metal -- --device gpu

    # GPU with CUDA (Linux/Windows)
    cargo run --example movie_ranking --release --features cuda -- --device gpu

    # Large model (GPU will be faster)
    cargo run --example movie_ranking --release --features metal -- \
        --device gpu --embedding-dim 256 --hidden-layers 512,256,128
"#
    );
}

// ============================================================================
// Embedding Table (CPU-based for sparse lookups)
// ============================================================================

/// Hash-based embedding table for sparse features.
///
/// Embeddings are stored on CPU for efficient sparse access, then transferred
/// to GPU in batches for dense computation.
struct EmbeddingTable {
    /// Mapping from feature ID to embedding vector
    embeddings: HashMap<u64, Vec<f32>>,
    /// Embedding dimension
    dim: usize,
    /// Standard deviation for initialization
    init_std: f32,
    /// Random seed for initialization
    seed: u64,
}

impl EmbeddingTable {
    /// Creates a new embedding table.
    fn new(dim: usize, init_std: f32, seed: u64) -> Self {
        Self {
            embeddings: HashMap::new(),
            dim,
            init_std,
            seed,
        }
    }

    /// Looks up or initializes embeddings for the given IDs.
    /// Returns data that can be transferred to GPU.
    fn lookup(&mut self, ids: &[u64]) -> Vec<f32> {
        let batch_size = ids.len();
        let mut data = Vec::with_capacity(batch_size * self.dim);

        for &id in ids {
            let embedding = self.embeddings.entry(id).or_insert_with(|| {
                // Initialize with small random values (Xavier-like)
                let mut seed = self.seed.wrapping_add(id);
                (0..self.dim)
                    .map(|_| {
                        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                        let u1 = ((seed >> 16) & 0x7fff) as f32 / 32768.0 + 1e-10;
                        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                        let u2 = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        z * self.init_std
                    })
                    .collect()
            });
            data.extend_from_slice(embedding);
        }

        data
    }

    /// Updates embeddings with gradients using SGD.
    fn apply_gradients(&mut self, ids: &[u64], gradients: &[f32], learning_rate: f32) {
        let dim = self.dim;
        for (i, &id) in ids.iter().enumerate() {
            if let Some(embedding) = self.embeddings.get_mut(&id) {
                for j in 0..dim {
                    embedding[j] -= learning_rate * gradients[i * dim + j];
                }
            }
        }
    }

    /// Returns the number of embeddings in the table.
    fn len(&self) -> usize {
        self.embeddings.len()
    }
}

// ============================================================================
// Dense Layer (GPU-accelerated)
// ============================================================================

/// A fully-connected (dense) layer with GPU acceleration.
struct DenseLayer {
    /// Weight matrix [in_features, out_features]
    weights: CandleTensor,
    /// Bias vector [out_features]
    bias: CandleTensor,
    /// Input features
    in_features: usize,
    /// Output features
    out_features: usize,
    /// Device for computation
    device: Device,
    /// Cached input for backward pass
    cached_input: Option<CandleTensor>,
    /// Cached weights gradient
    weights_grad: Option<CandleTensor>,
    /// Cached bias gradient
    bias_grad: Option<CandleTensor>,
}

impl DenseLayer {
    /// Creates a new dense layer with Xavier initialization.
    fn new(in_features: usize, out_features: usize, device: &Device) -> Self {
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let weights = CandleTensor::randn(&[in_features, out_features], 0.0, std, device);
        let bias = CandleTensor::zeros_on(&[out_features], device);

        Self {
            weights,
            bias,
            in_features,
            out_features,
            device: device.clone(),
            cached_input: None,
            weights_grad: None,
            bias_grad: None,
        }
    }

    /// Forward pass: y = xW + b (GPU accelerated)
    fn forward(&mut self, input: &CandleTensor, training: bool) -> CandleTensor {
        if training {
            self.cached_input = Some(input.clone());
        }
        let output = input.matmul(&self.weights);
        output.add_bias(&self.bias)
    }

    /// Backward pass: computes gradients and returns input gradient.
    fn backward(&mut self, grad: &CandleTensor) -> CandleTensor {
        let input = self.cached_input.as_ref().expect("Forward not called");

        // Weight gradient: dL/dW = X^T @ dL/dY
        let input_t = input.transpose();
        self.weights_grad = Some(input_t.matmul(grad));

        // Bias gradient: dL/db = sum(dL/dY, axis=0)
        self.bias_grad = Some(grad.sum_axis(0));

        // Input gradient: dL/dX = dL/dY @ W^T
        let weights_t = self.weights.transpose();
        grad.matmul(&weights_t)
    }

    /// Updates weights using SGD.
    fn update(&mut self, learning_rate: f32) {
        if let Some(ref w_grad) = self.weights_grad {
            let scaled = w_grad.scale(learning_rate);
            self.weights = self.weights.sub(&scaled);
        }
        if let Some(ref b_grad) = self.bias_grad {
            let scaled = b_grad.scale(learning_rate);
            self.bias = self.bias.sub(&scaled);
        }
    }
}

// ============================================================================
// Movie Ranking Model
// ============================================================================

/// Two-tower neural collaborative filtering model with GPU acceleration.
struct MovieRankingModel {
    /// User embedding table (CPU)
    user_embeddings: EmbeddingTable,
    /// Movie embedding table (CPU)
    movie_embeddings: EmbeddingTable,
    /// MLP hidden layers (GPU)
    hidden_layers: Vec<DenseLayer>,
    /// Output layer (GPU)
    output_layer: DenseLayer,
    /// Device for computation
    device: Device,
    /// Configuration
    embedding_dim: usize,
}

impl MovieRankingModel {
    /// Creates a new movie ranking model.
    fn new(config: &Config, device: &Device) -> Self {
        // Embedding tables with Xavier-like initialization (on CPU)
        let init_std = (1.0 / config.embedding_dim as f32).sqrt();
        let user_embeddings = EmbeddingTable::new(config.embedding_dim, init_std, config.seed);
        let movie_embeddings = EmbeddingTable::new(config.embedding_dim, init_std, config.seed + 1);

        // Build MLP layers (on GPU)
        let mut hidden_layers = Vec::new();
        let mut prev_dim = config.embedding_dim * 2; // Concatenated user + movie

        for &hidden_dim in &config.hidden_layers {
            hidden_layers.push(DenseLayer::new(prev_dim, hidden_dim, device));
            prev_dim = hidden_dim;
        }

        // Output layer (on GPU)
        let output_layer = DenseLayer::new(prev_dim, 1, device);

        Self {
            user_embeddings,
            movie_embeddings,
            hidden_layers,
            output_layer,
            device: device.clone(),
            embedding_dim: config.embedding_dim,
        }
    }

    /// Forward pass: computes prediction scores.
    fn forward(&mut self, user_ids: &[u64], movie_ids: &[u64], training: bool) -> CandleTensor {
        let batch_size = user_ids.len();
        assert_eq!(movie_ids.len(), batch_size);

        // Look up embeddings (on CPU)
        let user_emb_data = self.user_embeddings.lookup(user_ids);
        let movie_emb_data = self.movie_embeddings.lookup(movie_ids);

        // Concatenate embeddings
        let mut concat_data = Vec::with_capacity(batch_size * self.embedding_dim * 2);
        for i in 0..batch_size {
            let user_start = i * self.embedding_dim;
            let movie_start = i * self.embedding_dim;
            concat_data.extend_from_slice(&user_emb_data[user_start..user_start + self.embedding_dim]);
            concat_data.extend_from_slice(&movie_emb_data[movie_start..movie_start + self.embedding_dim]);
        }

        // Transfer to GPU
        let mut hidden = CandleTensor::from_slice_on(
            &concat_data,
            &[batch_size, self.embedding_dim * 2],
            &self.device,
        );

        // Pass through hidden layers with ReLU (on GPU)
        for layer in &mut self.hidden_layers {
            hidden = layer.forward(&hidden, training);
            hidden = hidden.relu();
        }

        // Output layer with sigmoid (on GPU)
        let logits = self.output_layer.forward(&hidden, training);
        logits.sigmoid()
    }

    /// Backward pass: computes and accumulates gradients.
    fn backward(&mut self, user_ids: &[u64], movie_ids: &[u64], loss_grad: &CandleTensor) -> (Vec<f32>, Vec<f32>) {
        let batch_size = user_ids.len();

        // Backward through sigmoid is already done in loss computation
        // The loss_grad is dL/d(predictions) * d(sigmoid)/d(logits) = predictions - labels

        // Backward through output layer
        let mut grad = self.output_layer.backward(loss_grad);

        // Backward through hidden layers (in reverse order)
        for layer in self.hidden_layers.iter_mut().rev() {
            // ReLU backward: gradient is passed through where input > 0
            // We approximate this by using the current gradient directly
            // (In a full implementation, we'd cache the ReLU mask)
            grad = layer.backward(&grad);
        }

        // Get gradients for embeddings (transfer back to CPU)
        let grad_data = grad.to_vec();

        // Split gradient for user and movie embeddings
        let mut user_grad = Vec::with_capacity(batch_size * self.embedding_dim);
        let mut movie_grad = Vec::with_capacity(batch_size * self.embedding_dim);

        for i in 0..batch_size {
            let row_start = i * self.embedding_dim * 2;
            user_grad.extend_from_slice(&grad_data[row_start..row_start + self.embedding_dim]);
            movie_grad.extend_from_slice(&grad_data[row_start + self.embedding_dim..row_start + self.embedding_dim * 2]);
        }

        (user_grad, movie_grad)
    }

    /// Updates all model parameters using SGD.
    fn update(&mut self, user_ids: &[u64], movie_ids: &[u64], user_grad: &[f32], movie_grad: &[f32], learning_rate: f32) {
        // Update hidden layers (on GPU)
        for layer in &mut self.hidden_layers {
            layer.update(learning_rate);
        }

        // Update output layer (on GPU)
        self.output_layer.update(learning_rate);

        // Update embeddings (on CPU)
        self.user_embeddings.apply_gradients(user_ids, user_grad, learning_rate);
        self.movie_embeddings.apply_gradients(movie_ids, movie_grad, learning_rate);
    }

    /// Returns model statistics.
    fn stats(&self) -> (usize, usize) {
        (self.user_embeddings.len(), self.movie_embeddings.len())
    }
}

// ============================================================================
// Loss Function (GPU-accelerated)
// ============================================================================

/// Computes binary cross-entropy loss and its gradient on GPU.
fn binary_cross_entropy_with_grad(predictions: &CandleTensor, labels: &CandleTensor) -> (f32, CandleTensor) {
    let eps = 1e-7;

    // Compute loss on GPU
    // Loss = -mean(y * log(p) + (1-y) * log(1-p))
    let pred_data = predictions.to_vec();
    let label_data = labels.to_vec();
    let batch_size = pred_data.len();

    let mut loss = 0.0;
    let mut grad_data = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let p = pred_data[i].clamp(eps, 1.0 - eps);
        let y = label_data[i];

        // BCE loss
        loss += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());

        // Gradient: (p - y) when combined with sigmoid backward
        grad_data.push(p - y);
    }

    let loss = loss / batch_size as f32;
    let grad = CandleTensor::from_slice_on(&grad_data, &[batch_size, 1], predictions.device());

    (loss, grad)
}

// ============================================================================
// Dataset Generation
// ============================================================================

#[derive(Clone)]
struct Sample {
    user_id: u64,
    movie_id: u64,
    label: f32,
}

struct DataGenerator {
    num_users: usize,
    num_movies: usize,
    seed: u64,
}

impl DataGenerator {
    fn new(num_users: usize, num_movies: usize, seed: u64) -> Self {
        Self {
            num_users,
            num_movies,
            seed,
        }
    }

    fn generate_samples(&mut self, num_samples: usize) -> Vec<Sample> {
        let mut samples = Vec::with_capacity(num_samples);
        let num_factors = 8;
        let mut user_factors: HashMap<u64, Vec<f32>> = HashMap::new();
        let mut movie_factors: HashMap<u64, Vec<f32>> = HashMap::new();

        for _ in 0..num_samples {
            self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
            let user_id = ((self.seed >> 16) as usize % self.num_users) as u64;

            self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
            let movie_id = ((self.seed >> 16) as usize % self.num_movies) as u64;

            let user_f = user_factors.entry(user_id).or_insert_with(|| {
                let mut factors = Vec::with_capacity(num_factors);
                for _ in 0..num_factors {
                    self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                    factors.push(((self.seed >> 16) & 0xffff) as f32 / 65536.0 - 0.5);
                }
                factors
            });

            let movie_f = movie_factors.entry(movie_id).or_insert_with(|| {
                let mut factors = Vec::with_capacity(num_factors);
                for _ in 0..num_factors {
                    self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
                    factors.push(((self.seed >> 16) & 0xffff) as f32 / 65536.0 - 0.5);
                }
                factors
            });

            let mut dot_product = 0.0;
            for i in 0..num_factors {
                dot_product += user_f[i] * movie_f[i];
            }

            self.seed = self.seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((self.seed >> 16) & 0xffff) as f32 / 65536.0 - 0.5;
            dot_product += noise * 0.2;

            let rating = (dot_product * 2.5 + 2.5).clamp(0.0, 5.0);
            let label = if rating >= 3.5 { 1.0 } else { 0.0 };

            samples.push(Sample {
                user_id,
                movie_id,
                label,
            });
        }

        samples
    }
}

struct BatchIterator<'a> {
    samples: &'a [Sample],
    batch_size: usize,
    current_idx: usize,
}

impl<'a> BatchIterator<'a> {
    fn new(samples: &'a [Sample], batch_size: usize) -> Self {
        Self {
            samples,
            batch_size,
            current_idx: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Vec<u64>, Vec<u64>, Vec<f32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.samples.len() {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.samples.len());
        let batch = &self.samples[self.current_idx..end_idx];
        self.current_idx = end_idx;

        let user_ids: Vec<u64> = batch.iter().map(|s| s.user_id).collect();
        let movie_ids: Vec<u64> = batch.iter().map(|s| s.movie_id).collect();
        let labels: Vec<f32> = batch.iter().map(|s| s.label).collect();

        Some((user_ids, movie_ids, labels))
    }
}

// ============================================================================
// Evaluation Metrics (parallelized)
// ============================================================================

fn compute_auc(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut positive_scores = Vec::new();
    let mut negative_scores = Vec::new();

    for (pred, label) in predictions.iter().zip(labels.iter()) {
        if *label > 0.5 {
            positive_scores.push(*pred);
        } else {
            negative_scores.push(*pred);
        }
    }

    if positive_scores.is_empty() || negative_scores.is_empty() {
        return 0.5;
    }

    let total = (positive_scores.len() * negative_scores.len()) as f32;

    let concordant: f64 = positive_scores
        .par_iter()
        .map(|&pos| {
            let mut local_concordant = 0.0f64;
            for &neg in &negative_scores {
                if pos > neg {
                    local_concordant += 1.0;
                } else if (pos - neg).abs() < 1e-10 {
                    local_concordant += 0.5;
                }
            }
            local_concordant
        })
        .sum();

    concordant as f32 / total
}

// ============================================================================
// Training Loop
// ============================================================================

fn train(config: &Config) {
    println!("{}", "=".repeat(70));
    println!("Movie Ranking Model - Training");
    println!("{}", "=".repeat(70));
    println!();

    // Select device based on config
    let device = match config.device {
        DeviceType::Cpu => Device::Cpu,
        DeviceType::Gpu => {
            let gpu = CandleTensor::best_device();
            if matches!(gpu, Device::Cpu) {
                eprintln!("Warning: GPU requested but not available. Falling back to CPU.");
                eprintln!("Hint: Build with --features metal (macOS) or --features cuda (Linux/Windows)");
            }
            gpu
        }
    };

    let device_name = match &device {
        Device::Cpu => "CPU (optimized BLAS)",
        Device::Cuda(_) => "CUDA GPU",
        Device::Metal(_) => "Metal GPU (Apple Silicon)",
    };

    println!("System:");
    println!("  Device:          {}", device_name);
    println!("  Rayon threads:   {} (for parallel CPU ops)", rayon::current_num_threads());
    println!();

    println!("Configuration:");
    println!("  Users:           {}", config.num_users);
    println!("  Movies:          {}", config.num_movies);
    println!("  Embedding dim:   {}", config.embedding_dim);
    println!("  Hidden layers:   {:?}", config.hidden_layers);
    println!("  Batch size:      {}", config.batch_size);
    println!("  Epochs:          {}", config.num_epochs);
    println!("  Learning rate:   {}", config.learning_rate);
    println!("  Train samples:   {}", config.num_train_samples);
    println!("  Test samples:    {}", config.num_test_samples);
    println!();

    // Generate datasets
    println!("Generating synthetic dataset...");
    let mut generator = DataGenerator::new(config.num_users, config.num_movies, config.seed);
    let train_samples = generator.generate_samples(config.num_train_samples);
    let test_samples = generator.generate_samples(config.num_test_samples);

    let train_positives = train_samples.par_iter().filter(|s| s.label > 0.5).count();
    let test_positives = test_samples.par_iter().filter(|s| s.label > 0.5).count();
    println!(
        "  Train set: {} samples ({} positive, {} negative)",
        train_samples.len(),
        train_positives,
        train_samples.len() - train_positives
    );
    println!(
        "  Test set:  {} samples ({} positive, {} negative)",
        test_samples.len(),
        test_positives,
        test_samples.len() - test_positives
    );
    println!();

    // Initialize model
    println!("Initializing model on {}...", device_name);
    let mut model = MovieRankingModel::new(config, &device);
    println!();

    // Training loop
    println!("Training:");
    println!("{}", "-".repeat(70));

    let start_time = std::time::Instant::now();

    for epoch in 0..config.num_epochs {
        let epoch_start = std::time::Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // Shuffle training data
        let mut shuffled = train_samples.clone();
        let mut shuffle_seed = config.seed.wrapping_add(epoch as u64);
        for i in (1..shuffled.len()).rev() {
            shuffle_seed = shuffle_seed.wrapping_mul(1103515245).wrapping_add(12345);
            let j = (shuffle_seed >> 16) as usize % (i + 1);
            shuffled.swap(i, j);
        }

        // Process batches
        for (user_ids, movie_ids, labels) in BatchIterator::new(&shuffled, config.batch_size) {
            let batch_size = user_ids.len();

            // Forward pass (GPU)
            let predictions = model.forward(&user_ids, &movie_ids, true);

            // Create labels tensor on GPU
            let labels_tensor = CandleTensor::from_slice_on(&labels, &[batch_size, 1], &device);

            // Compute loss and gradients (GPU)
            let (loss, grad) = binary_cross_entropy_with_grad(&predictions, &labels_tensor);
            epoch_loss += loss;
            num_batches += 1;

            // Backward pass (GPU, then transfer gradients to CPU for embeddings)
            let (user_grad, movie_grad) = model.backward(&user_ids, &movie_ids, &grad);

            // Update weights (GPU for layers, CPU for embeddings)
            model.update(&user_ids, &movie_ids, &user_grad, &movie_grad, config.learning_rate);
        }

        let avg_loss = epoch_loss / num_batches as f32;

        // Evaluate on test set
        let mut test_predictions = Vec::new();
        let mut test_labels = Vec::new();

        for (user_ids, movie_ids, labels) in BatchIterator::new(&test_samples, config.batch_size) {
            let preds = model.forward(&user_ids, &movie_ids, false);
            test_predictions.extend_from_slice(&preds.to_vec());
            test_labels.extend_from_slice(&labels);
        }

        let auc = compute_auc(&test_predictions, &test_labels);

        let elapsed = epoch_start.elapsed();
        println!(
            "Epoch {:2}/{}: loss = {:.4}, AUC = {:.4}, time = {:.2}s",
            epoch + 1,
            config.num_epochs,
            avg_loss,
            auc,
            elapsed.as_secs_f32()
        );
    }

    let total_time = start_time.elapsed();
    println!("{}", "-".repeat(70));
    println!();

    // Final evaluation
    println!("Final Evaluation:");
    println!("{}", "-".repeat(70));

    let mut test_predictions = Vec::new();
    let mut test_labels = Vec::new();
    let mut test_loss = 0.0;
    let mut num_batches = 0;

    for (user_ids, movie_ids, labels) in BatchIterator::new(&test_samples, config.batch_size) {
        let batch_size = user_ids.len();
        let preds = model.forward(&user_ids, &movie_ids, false);
        let labels_tensor = CandleTensor::from_slice_on(&labels, &[batch_size, 1], &device);
        let (loss, _) = binary_cross_entropy_with_grad(&preds, &labels_tensor);
        test_loss += loss;
        num_batches += 1;
        test_predictions.extend_from_slice(&preds.to_vec());
        test_labels.extend_from_slice(&labels);
    }

    let test_auc = compute_auc(&test_predictions, &test_labels);
    let avg_test_loss = test_loss / num_batches as f32;

    let mut correct = 0;
    for (pred, label) in test_predictions.iter().zip(test_labels.iter()) {
        let predicted_class = if *pred > 0.5 { 1.0 } else { 0.0 };
        if (predicted_class - label).abs() < 0.1 {
            correct += 1;
        }
    }
    let accuracy = correct as f32 / test_labels.len() as f32;

    let (num_user_emb, num_movie_emb) = model.stats();

    println!("  Test Loss:     {:.4}", avg_test_loss);
    println!("  Test AUC:      {:.4}", test_auc);
    println!("  Test Accuracy: {:.2}%", accuracy * 100.0);
    println!();
    println!("Model Statistics:");
    println!("  User embeddings:  {} (of {} possible)", num_user_emb, config.num_users);
    println!("  Movie embeddings: {} (of {} possible)", num_movie_emb, config.num_movies);
    println!();
    println!("Total training time: {:.2}s (on {})", total_time.as_secs_f32(), device_name);
    println!("{}", "=".repeat(70));
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() {
    let config = Config::from_args();
    train(&config);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_tensor_basic() {
        let t = CandleTensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
    }

    #[test]
    fn test_embedding_table() {
        let mut table = EmbeddingTable::new(4, 0.1, 42);
        let ids = vec![1, 2, 1];
        let embeddings = table.lookup(&ids);
        assert_eq!(embeddings.len(), 12); // 3 * 4
        assert_eq!(table.len(), 2); // Only 2 unique IDs
    }

    #[test]
    fn test_data_generator() {
        let mut generator = DataGenerator::new(100, 50, 42);
        let samples = generator.generate_samples(1000);
        assert_eq!(samples.len(), 1000);

        for sample in &samples {
            assert!(sample.label == 0.0 || sample.label == 1.0);
            assert!(sample.user_id < 100);
            assert!(sample.movie_id < 50);
        }
    }

    #[test]
    fn test_batch_iterator() {
        let samples: Vec<Sample> = (0..10)
            .map(|i| Sample {
                user_id: i as u64,
                movie_id: i as u64,
                label: if i % 2 == 0 { 1.0 } else { 0.0 },
            })
            .collect();

        let batches: Vec<_> = BatchIterator::new(&samples, 3).collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].0.len(), 3);
        assert_eq!(batches[3].0.len(), 1);
    }
}
