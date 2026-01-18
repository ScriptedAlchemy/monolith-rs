// Copyright 2024 Monolith-RS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Basic FFM (Field-aware Factorization Machine) Training Example
//!
//! This example demonstrates how to train an FFM model using monolith-rs.
//! FFM is a powerful model for CTR (Click-Through Rate) prediction that learns
//! field-aware feature interactions.
//!
//! ## Key Components
//!
//! 1. **FFMLayer**: The core model that computes field-aware feature interactions
//! 2. **Hash Table**: Stores sparse feature embeddings efficiently
//! 3. **Optimizer**: Updates embeddings based on computed gradients
//! 4. **Checkpointing**: Saves model state for later restoration
//!
//! ## Running the Example
//!
//! ```bash
//! cargo run --example basic_training -- --num-epochs 10 --batch-size 64 --learning-rate 0.01 --embedding-dim 8
//! ```

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

// Note: Using full paths since we're an example outside the workspace crates
use clap::Parser;

/// Command-line arguments for the training example.
#[derive(Parser, Debug)]
#[command(name = "basic_training")]
#[command(about = "Basic FFM training example for monolith-rs")]
#[command(version = "0.1.0")]
struct Args {
    /// Number of training epochs
    #[arg(long, default_value = "10")]
    num_epochs: usize,

    /// Batch size for training
    #[arg(long, default_value = "64")]
    batch_size: usize,

    /// Learning rate for the optimizer
    #[arg(long, default_value = "0.01")]
    learning_rate: f32,

    /// Embedding dimension for sparse features
    #[arg(long, default_value = "8")]
    embedding_dim: usize,

    /// Directory to save the model checkpoint
    #[arg(long, default_value = "/tmp/ffm_checkpoint")]
    checkpoint_dir: PathBuf,

    /// Number of steps between metric prints
    #[arg(long, default_value = "10")]
    print_every: usize,

    /// Random seed for reproducibility
    #[arg(long, default_value = "42")]
    seed: u64,
}

// ============================================================================
// Synthetic Data Generation
// ============================================================================

/// Configuration for synthetic data generation.
/// Based on the Python demo's FFM model configuration.
const NUM_FIELDS: usize = 6;
const VOCAB_SIZES: [usize; 6] = [5, 5, 5, 5, 5, 5];

/// A simple pseudo-random number generator for reproducible data.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max
    }
}

/// A single training example with sparse features and a label.
#[derive(Debug, Clone)]
struct TrainingExample {
    /// Feature IDs per field (field_index -> list of feature IDs)
    field_features: Vec<Vec<i64>>,
    /// Binary label (0 or 1)
    label: f32,
}

/// Generates synthetic training data similar to the Python demo.
///
/// Each example has:
/// - One or more feature IDs per field
/// - A binary label based on some synthetic pattern
fn generate_synthetic_data(num_examples: usize, rng: &mut SimpleRng) -> Vec<TrainingExample> {
    let mut examples = Vec::with_capacity(num_examples);

    for _ in 0..num_examples {
        let mut field_features = Vec::with_capacity(NUM_FIELDS);
        let mut feature_sum = 0i64;

        // Generate features for each field
        for (field_idx, &vocab_size) in VOCAB_SIZES.iter().enumerate() {
            // Each field has 1-3 features (multi-hot encoding)
            let num_features = 1 + rng.next_usize(3);
            let mut features = Vec::with_capacity(num_features);

            for _ in 0..num_features {
                // Generate feature ID: field_offset + local_id
                let local_id = rng.next_usize(vocab_size) as i64;
                let field_offset = (field_idx * 100) as i64; // Separate namespace per field
                let feature_id = field_offset + local_id;
                features.push(feature_id);
                feature_sum += local_id;
            }

            field_features.push(features);
        }

        // Generate label based on a synthetic pattern
        // Higher feature IDs tend to lead to positive labels
        let label = if feature_sum > (NUM_FIELDS * 2) as i64 {
            1.0
        } else if rng.next_f32() > 0.5 {
            1.0
        } else {
            0.0
        };

        examples.push(TrainingExample {
            field_features,
            label,
        });
    }

    examples
}

// Note: In production, you would use monolith_layers::tensor::Tensor
// This example uses native Rust types (Vec<f32>) for simplicity.

// ============================================================================
// Simple Embedding Hash Table
// ============================================================================

/// A simple embedding hash table for storing sparse feature embeddings.
/// In production, use monolith_hash_table::CuckooEmbeddingHashTable.
struct EmbeddingHashTable {
    /// Feature ID -> embedding vector
    entries: HashMap<i64, Vec<f32>>,
    /// Embedding dimension
    dim: usize,
    /// Learning rate for SGD updates
    learning_rate: f32,
}

impl EmbeddingHashTable {
    fn new(dim: usize, learning_rate: f32) -> Self {
        Self {
            entries: HashMap::new(),
            dim,
            learning_rate,
        }
    }

    /// Gets or initializes an embedding for the given feature ID.
    /// New embeddings are initialized with Xavier initialization.
    fn get_or_create(&mut self, id: i64, rng: &mut SimpleRng) -> Vec<f32> {
        if let Some(emb) = self.entries.get(&id) {
            return emb.clone();
        }

        // Xavier initialization
        let std = (2.0 / self.dim as f32).sqrt();
        let embedding: Vec<f32> = (0..self.dim)
            .map(|_| (rng.next_f32() - 0.5) * 2.0 * std)
            .collect();

        self.entries.insert(id, embedding.clone());
        embedding
    }

    /// Applies gradient updates to embeddings using SGD.
    fn apply_gradients(&mut self, gradients: &HashMap<i64, Vec<f32>>) {
        for (id, grad) in gradients {
            if let Some(emb) = self.entries.get_mut(id) {
                for (e, g) in emb.iter_mut().zip(grad.iter()) {
                    *e -= self.learning_rate * g;
                }
            }
        }
    }

    /// Returns the number of embeddings stored.
    fn size(&self) -> usize {
        self.entries.len()
    }

    /// Exports all embeddings for checkpointing.
    fn export(&self) -> HashMap<i64, Vec<f32>> {
        self.entries.clone()
    }
}

// ============================================================================
// FFM Layer Implementation
// ============================================================================

/// Field-aware Factorization Machine layer.
///
/// FFM computes pairwise feature interactions where each feature has
/// field-specific embeddings. The output is:
///
/// y = sum over all field pairs (i,j) where i < j of <v_{i,fj}, v_{j,fi}>
///
/// This implementation uses a hash table to store embeddings for sparse features.
struct FFMLayer {
    num_fields: usize,
    #[allow(dead_code)]
    embedding_dim: usize,
    /// Bias term for each field
    biases: Vec<f32>,
    /// Embedding tables per field pair (field_i, field_j) -> table
    /// We use a single hash table with encoded keys for simplicity
    embedding_table: EmbeddingHashTable,
    /// Whether to use bias
    use_bias: bool,
}

impl FFMLayer {
    fn new(num_fields: usize, embedding_dim: usize, learning_rate: f32, use_bias: bool) -> Self {
        Self {
            num_fields,
            embedding_dim,
            biases: vec![0.0; num_fields],
            embedding_table: EmbeddingHashTable::new(embedding_dim, learning_rate),
            use_bias,
        }
    }

    /// Encodes a (feature_id, target_field) pair into a unique key.
    fn encode_key(feature_id: i64, target_field: usize) -> i64 {
        // Use high bits for target_field to avoid collisions
        ((target_field as i64) << 32) | (feature_id & 0xFFFFFFFF)
    }

    /// Forward pass: computes FFM output for a batch of examples.
    ///
    /// # Arguments
    /// * `examples` - Batch of training examples
    /// * `rng` - Random number generator for initialization
    ///
    /// # Returns
    /// * Predictions for each example (before sigmoid)
    /// * Cached embeddings for backward pass
    fn forward(
        &mut self,
        examples: &[TrainingExample],
        rng: &mut SimpleRng,
    ) -> (Vec<f32>, Vec<HashMap<i64, Vec<f32>>>) {
        let batch_size = examples.len();
        let mut outputs = vec![0.0f32; batch_size];
        let mut cached_embeddings = Vec::with_capacity(batch_size);

        for (batch_idx, example) in examples.iter().enumerate() {
            let mut interaction_sum = 0.0f32;
            let mut example_cache = HashMap::new();

            // Compute pairwise field interactions
            for field_i in 0..self.num_fields {
                for field_j in (field_i + 1)..self.num_fields {
                    // Get features from both fields
                    let features_i = &example.field_features[field_i];
                    let features_j = &example.field_features[field_j];

                    // For each pair of features, compute interaction
                    for &feat_i in features_i {
                        for &feat_j in features_j {
                            // Get v_{i,fj}: embedding of feat_i for interacting with field_j
                            let key_i_fj = Self::encode_key(feat_i, field_j);
                            let emb_i_fj = self.embedding_table.get_or_create(key_i_fj, rng);
                            example_cache.insert(key_i_fj, emb_i_fj.clone());

                            // Get v_{j,fi}: embedding of feat_j for interacting with field_i
                            let key_j_fi = Self::encode_key(feat_j, field_i);
                            let emb_j_fi = self.embedding_table.get_or_create(key_j_fi, rng);
                            example_cache.insert(key_j_fi, emb_j_fi.clone());

                            // Inner product: <v_{i,fj}, v_{j,fi}>
                            let inner: f32 = emb_i_fj
                                .iter()
                                .zip(emb_j_fi.iter())
                                .map(|(a, b)| a * b)
                                .sum();

                            interaction_sum += inner;
                        }
                    }
                }
            }

            // Add bias if enabled
            if self.use_bias {
                for field_idx in 0..self.num_fields {
                    interaction_sum += self.biases[field_idx];
                }
            }

            outputs[batch_idx] = interaction_sum;
            cached_embeddings.push(example_cache);
        }

        (outputs, cached_embeddings)
    }

    /// Backward pass: computes gradients and applies updates.
    ///
    /// # Arguments
    /// * `examples` - Batch of training examples
    /// * `predictions` - Predictions from forward pass (after sigmoid)
    /// * `labels` - Ground truth labels
    /// * `cached_embeddings` - Cached embeddings from forward pass
    ///
    /// # Returns
    /// * Average loss for the batch
    fn backward(
        &mut self,
        examples: &[TrainingExample],
        predictions: &[f32],
        labels: &[f32],
        cached_embeddings: &[HashMap<i64, Vec<f32>>],
    ) -> f32 {
        let batch_size = examples.len();
        let mut total_loss = 0.0f32;
        let mut gradients: HashMap<i64, Vec<f32>> = HashMap::new();

        for (batch_idx, example) in examples.iter().enumerate() {
            let pred = predictions[batch_idx];
            let label = labels[batch_idx];

            // Binary cross entropy loss: -y*log(p) - (1-y)*log(1-p)
            // Clamp predictions to avoid log(0)
            let pred_clamped = pred.max(1e-7).min(1.0 - 1e-7);
            let loss = -label * pred_clamped.ln() - (1.0 - label) * (1.0 - pred_clamped).ln();
            total_loss += loss;

            // Gradient of loss w.r.t. logits: pred - label
            let grad_output = pred - label;

            let example_cache = &cached_embeddings[batch_idx];

            // Compute gradients for each field pair
            for field_i in 0..self.num_fields {
                for field_j in (field_i + 1)..self.num_fields {
                    let features_i = &example.field_features[field_i];
                    let features_j = &example.field_features[field_j];

                    for &feat_i in features_i {
                        for &feat_j in features_j {
                            let key_i_fj = Self::encode_key(feat_i, field_j);
                            let key_j_fi = Self::encode_key(feat_j, field_i);

                            let emb_i_fj = example_cache.get(&key_i_fj).unwrap();
                            let emb_j_fi = example_cache.get(&key_j_fi).unwrap();

                            // Gradient w.r.t. v_{i,fj}: grad_output * v_{j,fi}
                            let grad_i_fj: Vec<f32> =
                                emb_j_fi.iter().map(|v| grad_output * v).collect();

                            // Gradient w.r.t. v_{j,fi}: grad_output * v_{i,fj}
                            let grad_j_fi: Vec<f32> =
                                emb_i_fj.iter().map(|v| grad_output * v).collect();

                            // Accumulate gradients
                            gradients
                                .entry(key_i_fj)
                                .and_modify(|g| {
                                    for (gi, gi_new) in g.iter_mut().zip(grad_i_fj.iter()) {
                                        *gi += gi_new;
                                    }
                                })
                                .or_insert(grad_i_fj);

                            gradients
                                .entry(key_j_fi)
                                .and_modify(|g| {
                                    for (gi, gi_new) in g.iter_mut().zip(grad_j_fi.iter()) {
                                        *gi += gi_new;
                                    }
                                })
                                .or_insert(grad_j_fi);
                        }
                    }
                }
            }

            // Update bias gradients
            if self.use_bias {
                for field_idx in 0..self.num_fields {
                    self.biases[field_idx] -= self.embedding_table.learning_rate * grad_output;
                }
            }
        }

        // Average gradients over batch and apply
        for grad in gradients.values_mut() {
            for g in grad.iter_mut() {
                *g /= batch_size as f32;
            }
        }
        self.embedding_table.apply_gradients(&gradients);

        total_loss / batch_size as f32
    }

    /// Returns the number of embeddings in the model.
    fn num_embeddings(&self) -> usize {
        self.embedding_table.size()
    }

    /// Exports model state for checkpointing.
    fn export_state(&self) -> (HashMap<i64, Vec<f32>>, Vec<f32>) {
        (self.embedding_table.export(), self.biases.clone())
    }
}

// ============================================================================
// Loss and Metrics
// ============================================================================

/// Applies sigmoid activation.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Computes binary cross entropy loss.
fn binary_cross_entropy(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut total_loss = 0.0f32;
    for (pred, label) in predictions.iter().zip(labels.iter()) {
        let pred_clamped = pred.max(1e-7).min(1.0 - 1e-7);
        let loss = -label * pred_clamped.ln() - (1.0 - label) * (1.0 - pred_clamped).ln();
        total_loss += loss;
    }
    total_loss / predictions.len() as f32
}

/// Computes accuracy.
fn accuracy(predictions: &[f32], labels: &[f32]) -> f32 {
    let mut correct = 0;
    for (pred, label) in predictions.iter().zip(labels.iter()) {
        let pred_class = if *pred > 0.5 { 1.0 } else { 0.0 };
        if (pred_class - label).abs() < 0.5 {
            correct += 1;
        }
    }
    correct as f32 / predictions.len() as f32
}

/// Computes AUC (Area Under the ROC Curve).
fn auc(predictions: &[f32], labels: &[f32]) -> f32 {
    // Simple AUC calculation using pairwise comparisons
    let mut num_pos_neg_pairs = 0;
    let mut correct_rankings = 0.0f32;

    for (i, (pred_i, label_i)) in predictions.iter().zip(labels.iter()).enumerate() {
        for (pred_j, label_j) in predictions.iter().zip(labels.iter()).skip(i + 1) {
            if *label_i != *label_j {
                num_pos_neg_pairs += 1;
                let (pos_pred, neg_pred) = if *label_i > *label_j {
                    (*pred_i, *pred_j)
                } else {
                    (*pred_j, *pred_i)
                };

                if pos_pred > neg_pred {
                    correct_rankings += 1.0;
                } else if (pos_pred - neg_pred).abs() < 1e-7 {
                    correct_rankings += 0.5;
                }
            }
        }
    }

    if num_pos_neg_pairs == 0 {
        0.5
    } else {
        correct_rankings / num_pos_neg_pairs as f32
    }
}

// ============================================================================
// Checkpointing
// ============================================================================

/// Saves model checkpoint to disk.
fn save_checkpoint(
    checkpoint_dir: &std::path::Path,
    step: u64,
    embeddings: &HashMap<i64, Vec<f32>>,
    biases: &[f32],
) -> std::io::Result<()> {
    std::fs::create_dir_all(checkpoint_dir)?;

    // Create checkpoint data
    let checkpoint_path = checkpoint_dir.join(format!("checkpoint_{}.json", step));

    // Serialize embeddings (simplified - in production use monolith_checkpoint)
    let checkpoint_data = serde_json::json!({
        "version": 1,
        "global_step": step,
        "num_embeddings": embeddings.len(),
        "embedding_dim": embeddings.values().next().map(|v| v.len()).unwrap_or(0),
        "biases": biases,
        // Note: In production, embeddings would be stored in binary format
        // Here we just save metadata for demonstration
        "embeddings_count": embeddings.len(),
    });

    std::fs::write(
        &checkpoint_path,
        serde_json::to_string_pretty(&checkpoint_data)?,
    )?;

    println!("Saved checkpoint to {:?}", checkpoint_path);
    Ok(())
}

// ============================================================================
// Main Training Loop
// ============================================================================

fn main() {
    // Parse command-line arguments
    let args = Args::parse();

    println!("=== FFM Training Example ===");
    println!("Configuration:");
    println!("  - Epochs: {}", args.num_epochs);
    println!("  - Batch size: {}", args.batch_size);
    println!("  - Learning rate: {}", args.learning_rate);
    println!("  - Embedding dimension: {}", args.embedding_dim);
    println!("  - Checkpoint directory: {:?}", args.checkpoint_dir);
    println!("  - Print every: {} steps", args.print_every);
    println!("  - Random seed: {}", args.seed);
    println!();

    // Initialize random number generator
    let mut rng = SimpleRng::new(args.seed);

    // Generate synthetic training data
    println!("Generating synthetic data...");
    let num_examples = 1000;
    let data = generate_synthetic_data(num_examples, &mut rng);
    println!("Generated {} examples", data.len());

    // Split into train/validation
    let split_idx = (data.len() as f32 * 0.8) as usize;
    let (train_data, val_data) = data.split_at(split_idx);
    println!("Training set: {} examples", train_data.len());
    println!("Validation set: {} examples", val_data.len());
    println!();

    // Create FFM model
    println!("Initializing FFM model...");
    let mut ffm = FFMLayer::new(
        NUM_FIELDS,
        args.embedding_dim,
        args.learning_rate,
        true, // use bias
    );
    println!(
        "Model initialized with {} fields and {}-dim embeddings",
        NUM_FIELDS, args.embedding_dim
    );
    println!();

    // Training loop
    println!("Starting training...");
    println!("{:-<80}", "");

    let training_start = Instant::now();
    let mut global_step = 0u64;
    let num_batches = train_data.len() / args.batch_size;

    for epoch in 0..args.num_epochs {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0f32;
        let mut epoch_samples = 0;

        // Shuffle training data (simple implementation)
        let mut indices: Vec<usize> = (0..train_data.len()).collect();
        for i in (1..indices.len()).rev() {
            let j = rng.next_usize(i + 1);
            indices.swap(i, j);
        }

        // Process batches
        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * args.batch_size;
            let batch_end = (batch_idx + 1) * args.batch_size;
            let batch_indices = &indices[batch_start..batch_end];

            // Get batch data
            let batch: Vec<TrainingExample> = batch_indices
                .iter()
                .map(|&i| train_data[i].clone())
                .collect();
            let labels: Vec<f32> = batch.iter().map(|e| e.label).collect();

            // Forward pass
            let (logits, cached_embeddings) = ffm.forward(&batch, &mut rng);
            let predictions: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();

            // Backward pass
            let loss = ffm.backward(&batch, &predictions, &labels, &cached_embeddings);
            epoch_loss += loss * args.batch_size as f32;
            epoch_samples += args.batch_size;
            global_step += 1;

            // Print metrics periodically
            if global_step % args.print_every as u64 == 0 {
                let acc = accuracy(&predictions, &labels);
                let batch_auc = auc(&predictions, &labels);
                println!(
                    "Step {:5} | Loss: {:.4} | Acc: {:.4} | AUC: {:.4} | Embeddings: {}",
                    global_step,
                    loss,
                    acc,
                    batch_auc,
                    ffm.num_embeddings()
                );
            }
        }

        // End of epoch metrics
        let epoch_elapsed = epoch_start.elapsed();
        let avg_epoch_loss = epoch_loss / epoch_samples as f32;

        // Validation
        let (val_logits, _) = ffm.forward(val_data, &mut rng);
        let val_predictions: Vec<f32> = val_logits.iter().map(|&x| sigmoid(x)).collect();
        let val_labels: Vec<f32> = val_data.iter().map(|e| e.label).collect();
        let val_loss = binary_cross_entropy(&val_predictions, &val_labels);
        let val_acc = accuracy(&val_predictions, &val_labels);
        let val_auc = auc(&val_predictions, &val_labels);

        println!("{:-<80}", "");
        println!(
            "Epoch {}/{} completed in {:.2}s",
            epoch + 1,
            args.num_epochs,
            epoch_elapsed.as_secs_f32()
        );
        println!(
            "  Train Loss: {:.4} | Val Loss: {:.4} | Val Acc: {:.4} | Val AUC: {:.4}",
            avg_epoch_loss, val_loss, val_acc, val_auc
        );
        println!("{:-<80}", "");
    }

    let total_elapsed = training_start.elapsed();
    println!();
    println!("Training completed in {:.2}s", total_elapsed.as_secs_f32());
    println!("Final model has {} embeddings", ffm.num_embeddings());

    // Save checkpoint
    println!();
    println!("Saving checkpoint...");
    let (embeddings, biases) = ffm.export_state();
    if let Err(e) = save_checkpoint(&args.checkpoint_dir, global_step, &embeddings, &biases) {
        eprintln!("Warning: Failed to save checkpoint: {}", e);
    }

    println!();
    println!("=== Training Complete ===");
    println!();
    println!("Summary:");
    println!("  - Total steps: {}", global_step);
    println!("  - Total embeddings: {}", ffm.num_embeddings());
    println!("  - Training time: {:.2}s", total_elapsed.as_secs_f32());
    println!("  - Checkpoint saved to: {:?}", args.checkpoint_dir);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rng() {
        let mut rng = SimpleRng::new(42);
        let val1 = rng.next_f32();
        let val2 = rng.next_f32();
        assert!(val1 >= 0.0 && val1 < 1.0);
        assert!(val2 >= 0.0 && val2 < 1.0);
        assert_ne!(val1, val2);
    }

    #[test]
    fn test_generate_synthetic_data() {
        let mut rng = SimpleRng::new(42);
        let data = generate_synthetic_data(10, &mut rng);
        assert_eq!(data.len(), 10);

        for example in &data {
            assert_eq!(example.field_features.len(), NUM_FIELDS);
            assert!(example.label == 0.0 || example.label == 1.0);
        }
    }

    #[test]
    fn test_embedding_hash_table() {
        let mut table = EmbeddingHashTable::new(4, 0.01);
        let mut rng = SimpleRng::new(42);

        let emb1 = table.get_or_create(1, &mut rng);
        assert_eq!(emb1.len(), 4);
        assert_eq!(table.size(), 1);

        // Getting the same ID should return the same embedding
        let emb1_again = table.get_or_create(1, &mut rng);
        assert_eq!(emb1, emb1_again);
        assert_eq!(table.size(), 1);
    }

    #[test]
    fn test_ffm_forward() {
        let mut ffm = FFMLayer::new(3, 4, 0.01, false);
        let mut rng = SimpleRng::new(42);

        let example = TrainingExample {
            field_features: vec![vec![0], vec![1], vec![2]],
            label: 1.0,
        };

        let (outputs, cached) = ffm.forward(&[example], &mut rng);
        assert_eq!(outputs.len(), 1);
        assert_eq!(cached.len(), 1);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_accuracy() {
        let predictions = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        assert!((accuracy(&predictions, &labels) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_auc() {
        // Perfect predictions
        let predictions = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![1.0, 1.0, 0.0, 0.0];
        let auc_val = auc(&predictions, &labels);
        assert!((auc_val - 1.0).abs() < 1e-6);

        // Random predictions
        let predictions = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![1.0, 0.0, 1.0, 0.0];
        let auc_val = auc(&predictions, &labels);
        assert!((auc_val - 0.5).abs() < 0.1);
    }
}
