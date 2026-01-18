//! Model Export Example for Monolith-RS
//!
//! This example demonstrates how to:
//! 1. Train a simple movie ranking model
//! 2. Save checkpoints during training
//! 3. Export the model in multiple formats (Bincode, JSON, MessagePack)
//! 4. Support standalone and distributed export modes
//! 5. Save incremental deltas
//! 6. Verify exported models can be loaded correctly
//!
//! # Usage
//!
//! ```bash
//! cargo run --example model_export -- \
//!     --checkpoint-dir /tmp/monolith/checkpoints \
//!     --export-dir /tmp/monolith/exports \
//!     --format bincode \
//!     --compression gzip
//! ```
//!
//! This example mirrors the Python demo_export.py from the original Monolith.

use clap::{Parser, ValueEnum};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

// Import from monolith-checkpoint crate
use monolith_checkpoint::{
    BincodeSerializer, Checkpoint, CheckpointConfig, CheckpointDelta, CheckpointManager,
    CheckpointReader, CheckpointWriter, CompressionType, ExportConfig, ExportFormat,
    ExportManifest, HashTableState, JsonCheckpointer, JsonSerializer, MessagePackSerializer,
    ModelExporter, ModelState, OptimizerState,
};

// ============================================================================
// Configuration Types
// ============================================================================

/// Export format for model serialization.
#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq)]
pub enum SerializationFormat {
    /// Bincode format - fast binary serialization (recommended for production)
    #[default]
    Bincode,
    /// JSON format - human-readable, good for debugging
    Json,
    /// MessagePack format - compact binary, cross-platform compatible
    Msgpack,
}

impl std::fmt::Display for SerializationFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SerializationFormat::Bincode => write!(f, "bincode"),
            SerializationFormat::Json => write!(f, "json"),
            SerializationFormat::Msgpack => write!(f, "msgpack"),
        }
    }
}

/// Export mode for model deployment.
#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq)]
pub enum ExportMode {
    /// Standalone mode - all model parameters in a single export
    #[default]
    Standalone,
    /// Distributed mode - model parameters sharded for distributed serving
    Distributed,
}

impl std::fmt::Display for ExportMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportMode::Standalone => write!(f, "standalone"),
            ExportMode::Distributed => write!(f, "distributed"),
        }
    }
}

/// Compression type for exported files.
#[derive(Debug, Clone, Copy, Default, ValueEnum, PartialEq)]
pub enum Compression {
    /// No compression
    #[default]
    None,
    /// Gzip compression
    Gzip,
}

/// Export configuration structure.
///
/// This struct encapsulates all settings needed to export a trained model.
#[derive(Debug, Clone)]
pub struct ModelExportConfig {
    /// Path to the checkpoint directory or file to export from.
    pub checkpoint_path: PathBuf,
    /// Path where the exported model will be saved.
    pub export_path: PathBuf,
    /// Serialization format to use.
    pub format: SerializationFormat,
    /// Export mode (standalone or distributed).
    pub mode: ExportMode,
    /// Compression to apply to exported files.
    pub compression: Compression,
    /// Number of shards for distributed export.
    pub num_shards: usize,
    /// Model version string.
    pub version: String,
    /// Whether to include optimizer state in export.
    pub include_optimizer: bool,
}

impl ModelExportConfig {
    /// Create a new export configuration.
    pub fn new(checkpoint_path: PathBuf, export_path: PathBuf) -> Self {
        Self {
            checkpoint_path,
            export_path,
            format: SerializationFormat::default(),
            mode: ExportMode::default(),
            compression: Compression::default(),
            num_shards: 1,
            version: "1.0.0".to_string(),
            include_optimizer: false,
        }
    }

    /// Set the serialization format.
    pub fn with_format(mut self, format: SerializationFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the export mode.
    pub fn with_mode(mut self, mode: ExportMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the compression type.
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set the number of shards for distributed export.
    pub fn with_num_shards(mut self, num_shards: usize) -> Self {
        self.num_shards = num_shards;
        self
    }

    /// Set the model version.
    pub fn with_version(mut self, version: impl Into<String>) -> Self {
        self.version = version.into();
        self
    }

    /// Include optimizer state in export.
    pub fn with_optimizer(mut self, include: bool) -> Self {
        self.include_optimizer = include;
        self
    }

    /// Get the compression type for checkpoint serialization.
    fn compression_type(&self) -> CompressionType {
        match self.compression {
            Compression::None => CompressionType::None,
            Compression::Gzip => CompressionType::Gzip,
        }
    }

    /// Get the file extension for the serialization format.
    fn file_extension(&self) -> &str {
        match self.format {
            SerializationFormat::Bincode => "bin",
            SerializationFormat::Json => "json",
            SerializationFormat::Msgpack => "msgpack",
        }
    }
}

// ============================================================================
// Command Line Arguments
// ============================================================================

/// Model Export CLI for Monolith-RS
///
/// Export trained models for serving in various formats.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Directory containing training checkpoints
    #[arg(long, short = 'c', default_value = "/tmp/monolith/checkpoints")]
    checkpoint_dir: PathBuf,

    /// Directory for exported model
    #[arg(long, short = 'e', default_value = "/tmp/monolith/exports")]
    export_dir: PathBuf,

    /// Serialization format
    #[arg(long, short = 'f', default_value = "bincode")]
    format: SerializationFormat,

    /// Compression type
    #[arg(long, short = 'z', default_value = "none")]
    compression: Compression,

    /// Export mode
    #[arg(long, short = 'm', default_value = "standalone")]
    mode: ExportMode,

    /// Number of shards for distributed export
    #[arg(long, default_value = "1")]
    num_shards: usize,

    /// Model version string for export
    #[arg(long = "model-version", default_value = "1.0.0")]
    model_version: String,

    /// Run training before export (otherwise expects existing checkpoint)
    #[arg(long, default_value = "true")]
    train: bool,

    /// Number of training steps
    #[arg(long, default_value = "100")]
    train_steps: u64,

    /// Include optimizer state in export
    #[arg(long)]
    include_optimizer: bool,

    /// Verify export by loading it back
    #[arg(long, default_value = "true")]
    verify: bool,
}

// ============================================================================
// Movie Ranking Model Simulation
// ============================================================================

/// Simple movie ranking model for demonstration.
///
/// This simulates a recommendation model with:
/// - User embeddings (user features)
/// - Movie embeddings (item features)
/// - Dense layers for prediction
struct MovieRankingModel {
    /// User embedding table
    user_embeddings: HashTableState,
    /// Movie embedding table
    movie_embeddings: HashTableState,
    /// Dense layer weights
    dense_weights: HashMap<String, Vec<f32>>,
    /// Current training step
    global_step: u64,
    /// Learning rate
    learning_rate: f64,
}

impl MovieRankingModel {
    /// Create a new movie ranking model.
    fn new() -> Self {
        let embedding_dim = 32;

        // Initialize user embeddings
        let mut user_embeddings = HashTableState::new("user_embeddings", embedding_dim);
        for user_id in 0..100 {
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|i| ((user_id * 17 + i) as f32 % 100.0) / 100.0 - 0.5)
                .collect();
            user_embeddings.insert(user_id as i64, embedding);
        }

        // Initialize movie embeddings
        let mut movie_embeddings = HashTableState::new("movie_embeddings", embedding_dim);
        for movie_id in 0..500 {
            let embedding: Vec<f32> = (0..embedding_dim)
                .map(|i| ((movie_id * 23 + i) as f32 % 100.0) / 100.0 - 0.5)
                .collect();
            movie_embeddings.insert(movie_id as i64, embedding);
        }

        // Initialize dense layer weights
        let mut dense_weights = HashMap::new();
        let hidden_dim = 64;
        let output_dim = 1;

        // Layer 1: concat(user, movie) -> hidden
        dense_weights.insert(
            "fc1.weight".to_string(),
            vec![0.1; embedding_dim * 2 * hidden_dim],
        );
        dense_weights.insert("fc1.bias".to_string(), vec![0.0; hidden_dim]);

        // Layer 2: hidden -> output
        dense_weights.insert("fc2.weight".to_string(), vec![0.1; hidden_dim * output_dim]);
        dense_weights.insert("fc2.bias".to_string(), vec![0.0; output_dim]);

        Self {
            user_embeddings,
            movie_embeddings,
            dense_weights,
            global_step: 0,
            learning_rate: 0.001,
        }
    }

    /// Simulate a training step.
    fn train_step(&mut self) {
        self.global_step += 1;

        // Simulate weight updates (in real training, this would be gradient descent)
        // Here we just add some noise to simulate learning
        let decay = 1.0 - (self.global_step as f32 * 0.0001);

        for (_, weights) in self.dense_weights.iter_mut() {
            for w in weights.iter_mut() {
                *w *= decay;
                *w += (self.global_step as f32 % 7.0 - 3.0) * 0.0001;
            }
        }

        // Update some embeddings (simulating gradient updates on accessed embeddings)
        let user_id = (self.global_step % 100) as i64;
        if let Some(embedding) = self.user_embeddings.entries.get_mut(&user_id) {
            for e in embedding.iter_mut() {
                *e *= decay;
            }
        }

        let movie_id = (self.global_step % 500) as i64;
        if let Some(embedding) = self.movie_embeddings.entries.get_mut(&movie_id) {
            for e in embedding.iter_mut() {
                *e *= decay;
            }
        }
    }

    /// Run training for a specified number of steps.
    fn train(&mut self, num_steps: u64) {
        println!("Training movie ranking model for {} steps...", num_steps);
        let start = std::time::Instant::now();

        for step in 1..=num_steps {
            self.train_step();

            if step % 10 == 0 {
                println!(
                    "  Step {}/{}: loss = {:.4}",
                    step,
                    num_steps,
                    self.simulated_loss()
                );
            }
        }

        let elapsed = start.elapsed();
        println!(
            "Training complete. {} steps in {:.2}s ({:.0} steps/sec)",
            num_steps,
            elapsed.as_secs_f64(),
            num_steps as f64 / elapsed.as_secs_f64()
        );
    }

    /// Simulate a loss value (decreasing over training).
    fn simulated_loss(&self) -> f32 {
        1.0 / (1.0 + self.global_step as f32 * 0.01)
    }

    /// Convert model to ModelState for checkpointing.
    fn to_model_state(&self) -> ModelState {
        let mut state = ModelState::new(self.global_step);

        // Add embedding tables
        state.add_hash_table(self.user_embeddings.clone());
        state.add_hash_table(self.movie_embeddings.clone());

        // Add dense parameters
        for (name, weights) in &self.dense_weights {
            state.add_dense_param(name, weights.clone());
        }

        // Add optimizer state
        let optimizer =
            OptimizerState::new("adam", "all_params", self.learning_rate, self.global_step);
        state.add_optimizer(optimizer);

        // Add metadata
        state.set_metadata("model_type", "movie_ranking");
        state.set_metadata("embedding_dim", "32");
        state.set_metadata("hidden_dim", "64");

        state
    }

    /// Create a delta representing changes since the last checkpoint.
    fn create_delta(&self, base_step: u64) -> CheckpointDelta {
        let mut delta = CheckpointDelta::new(base_step, self.global_step);

        // In a real implementation, we would track which embeddings were updated.
        // For this demo, we include a sample of updated embeddings.
        let mut user_updates = HashMap::new();
        for user_id in (base_step..self.global_step).take(10) {
            let id = (user_id % 100) as i64;
            if let Some(embedding) = self.user_embeddings.entries.get(&id) {
                user_updates.insert(id, embedding.clone());
            }
        }
        if !user_updates.is_empty() {
            delta
                .updated_embeddings
                .insert("user_embeddings".to_string(), user_updates);
        }

        let mut movie_updates = HashMap::new();
        for movie_id in (base_step..self.global_step).take(10) {
            let id = (movie_id % 500) as i64;
            if let Some(embedding) = self.movie_embeddings.entries.get(&id) {
                movie_updates.insert(id, embedding.clone());
            }
        }
        if !movie_updates.is_empty() {
            delta
                .updated_embeddings
                .insert("movie_embeddings".to_string(), movie_updates);
        }

        // Add updated dense parameters
        for (name, weights) in &self.dense_weights {
            delta
                .updated_dense_params
                .insert(name.clone(), weights.clone());
        }

        delta
    }
}

// ============================================================================
// Model Exporter Implementation
// ============================================================================

/// Exports a model in the specified format and mode.
struct MultiFormatExporter {
    config: ModelExportConfig,
}

impl MultiFormatExporter {
    /// Create a new multi-format exporter.
    fn new(config: ModelExportConfig) -> Self {
        Self { config }
    }

    /// Export the model state.
    fn export(&self, state: &ModelState) -> anyhow::Result<PathBuf> {
        println!(
            "\nExporting model (format={}, mode={}, compression={:?})...",
            self.config.format, self.config.mode, self.config.compression
        );

        // Create output directory
        std::fs::create_dir_all(&self.config.export_path)?;

        match self.config.mode {
            ExportMode::Standalone => self.export_standalone(state),
            ExportMode::Distributed => self.export_distributed(state),
        }
    }

    /// Export in standalone mode (single file).
    fn export_standalone(&self, state: &ModelState) -> anyhow::Result<PathBuf> {
        let checkpoint = Checkpoint::new(state.clone());

        let filename = format!("model.{}", self.config.file_extension());
        let output_path = self.config.export_path.join(&filename);

        match self.config.format {
            SerializationFormat::Bincode => {
                let writer = CheckpointWriter::new(BincodeSerializer::new())
                    .with_compression(self.config.compression_type());
                writer.write_to_file(&output_path, &checkpoint)?;
            }
            SerializationFormat::Json => {
                let writer = CheckpointWriter::new(JsonSerializer::pretty())
                    .with_compression(self.config.compression_type());
                writer.write_to_file(&output_path, &checkpoint)?;
            }
            SerializationFormat::Msgpack => {
                let writer = CheckpointWriter::new(MessagePackSerializer::new())
                    .with_compression(self.config.compression_type());
                writer.write_to_file(&output_path, &checkpoint)?;
            }
        }

        // Write manifest
        self.write_manifest(state)?;

        println!("  Standalone export saved to: {:?}", output_path);
        Ok(self.config.export_path.clone())
    }

    /// Export in distributed mode (sharded files).
    fn export_distributed(&self, state: &ModelState) -> anyhow::Result<PathBuf> {
        let num_shards = self.config.num_shards.max(1);
        println!(
            "  Creating {} shards for distributed serving...",
            num_shards
        );

        // Create shards directory
        let shards_dir = self.config.export_path.join("shards");
        std::fs::create_dir_all(&shards_dir)?;

        // Shard embedding tables across multiple files
        for (table_idx, table) in state.hash_tables.iter().enumerate() {
            let entries: Vec<_> = table.entries.iter().collect();
            let entries_per_shard = (entries.len() + num_shards - 1) / num_shards;

            for shard_idx in 0..num_shards {
                let start = shard_idx * entries_per_shard;
                let end = (start + entries_per_shard).min(entries.len());

                if start >= entries.len() {
                    break;
                }

                let mut shard_table = HashTableState::new(&table.name, table.dim);
                for (key, value) in &entries[start..end] {
                    shard_table.insert(**key, (*value).clone());
                }

                let mut shard_state = ModelState::new(state.global_step);
                shard_state.add_hash_table(shard_table);

                let shard_checkpoint = Checkpoint::new(shard_state);
                let shard_filename = format!(
                    "table_{}_shard_{}.{}",
                    table_idx,
                    shard_idx,
                    self.config.file_extension()
                );
                let shard_path = shards_dir.join(&shard_filename);

                self.write_checkpoint(&shard_path, &shard_checkpoint)?;
            }
        }

        // Write dense parameters to a single file
        let mut dense_state = ModelState::new(state.global_step);
        for (name, values) in &state.dense_params {
            dense_state.add_dense_param(name, values.clone());
        }

        if self.config.include_optimizer {
            for opt in &state.optimizers {
                dense_state.add_optimizer(opt.clone());
            }
        }

        let dense_checkpoint = Checkpoint::new(dense_state);
        let dense_path = shards_dir.join(format!("dense.{}", self.config.file_extension()));
        self.write_checkpoint(&dense_path, &dense_checkpoint)?;

        // Write manifest
        self.write_distributed_manifest(state, num_shards)?;

        println!(
            "  Distributed export saved to: {:?}",
            self.config.export_path
        );
        Ok(self.config.export_path.clone())
    }

    /// Write a checkpoint file using the configured format.
    fn write_checkpoint(&self, path: &PathBuf, checkpoint: &Checkpoint) -> anyhow::Result<()> {
        match self.config.format {
            SerializationFormat::Bincode => {
                let writer = CheckpointWriter::new(BincodeSerializer::new())
                    .with_compression(self.config.compression_type());
                writer.write_to_file(path, checkpoint)?;
            }
            SerializationFormat::Json => {
                let writer = CheckpointWriter::new(JsonSerializer::pretty())
                    .with_compression(self.config.compression_type());
                writer.write_to_file(path, checkpoint)?;
            }
            SerializationFormat::Msgpack => {
                let writer = CheckpointWriter::new(MessagePackSerializer::new())
                    .with_compression(self.config.compression_type());
                writer.write_to_file(path, checkpoint)?;
            }
        }
        Ok(())
    }

    /// Write export manifest for standalone export.
    fn write_manifest(&self, state: &ModelState) -> anyhow::Result<()> {
        let manifest = ExportManifest {
            version: self.config.version.clone(),
            format: self.config.format.to_string(),
            global_step: state.global_step,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            embedding_tables: state.hash_tables.iter().map(|t| t.name.clone()).collect(),
            dense_params: state.dense_params.keys().cloned().collect(),
            includes_optimizer: self.config.include_optimizer,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("mode".to_string(), self.config.mode.to_string());
                meta.insert(
                    "compression".to_string(),
                    format!("{:?}", self.config.compression),
                );
                meta
            },
        };

        let manifest_path = self.config.export_path.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, manifest_json)?;

        Ok(())
    }

    /// Write export manifest for distributed export.
    fn write_distributed_manifest(
        &self,
        state: &ModelState,
        num_shards: usize,
    ) -> anyhow::Result<()> {
        let manifest = ExportManifest {
            version: self.config.version.clone(),
            format: self.config.format.to_string(),
            global_step: state.global_step,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            embedding_tables: state.hash_tables.iter().map(|t| t.name.clone()).collect(),
            dense_params: state.dense_params.keys().cloned().collect(),
            includes_optimizer: self.config.include_optimizer,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("mode".to_string(), "distributed".to_string());
                meta.insert("num_shards".to_string(), num_shards.to_string());
                meta.insert(
                    "compression".to_string(),
                    format!("{:?}", self.config.compression),
                );
                for (idx, table) in state.hash_tables.iter().enumerate() {
                    meta.insert(
                        format!("table_{}_entries", idx),
                        table.entries.len().to_string(),
                    );
                }
                meta
            },
        };

        let manifest_path = self.config.export_path.join("manifest.json");
        let manifest_json = serde_json::to_string_pretty(&manifest)?;
        std::fs::write(&manifest_path, manifest_json)?;

        Ok(())
    }

    /// Export incremental delta.
    fn export_delta(&self, delta: &CheckpointDelta) -> anyhow::Result<PathBuf> {
        let deltas_dir = self.config.export_path.join("deltas");
        std::fs::create_dir_all(&deltas_dir)?;

        let filename = format!(
            "delta_{}_to_{}.{}",
            delta.base_step,
            delta.new_step,
            self.config.file_extension()
        );
        let delta_path = deltas_dir.join(&filename);

        match self.config.format {
            SerializationFormat::Bincode => {
                let writer = CheckpointWriter::new(BincodeSerializer::new())
                    .with_compression(self.config.compression_type());
                writer.write_incremental(&delta_path, delta)?;
            }
            SerializationFormat::Json => {
                let writer = CheckpointWriter::new(JsonSerializer::pretty())
                    .with_compression(self.config.compression_type());
                writer.write_incremental(&delta_path, delta)?;
            }
            SerializationFormat::Msgpack => {
                let writer = CheckpointWriter::new(MessagePackSerializer::new())
                    .with_compression(self.config.compression_type());
                writer.write_incremental(&delta_path, delta)?;
            }
        }

        println!("  Delta exported to: {:?}", delta_path);
        Ok(delta_path)
    }
}

// ============================================================================
// Export Verification
// ============================================================================

/// Verify that an exported model can be loaded correctly.
fn verify_export(config: &ModelExportConfig, original_state: &ModelState) -> anyhow::Result<()> {
    println!("\nVerifying exported model...");

    // Load manifest
    let manifest_path = config.export_path.join("manifest.json");
    let manifest_content = std::fs::read_to_string(&manifest_path)?;
    let manifest: ExportManifest = serde_json::from_str(&manifest_content)?;

    println!("  Manifest loaded:");
    println!("    - Version: {}", manifest.version);
    println!("    - Global step: {}", manifest.global_step);
    println!("    - Embedding tables: {:?}", manifest.embedding_tables);
    println!("    - Dense params: {:?}", manifest.dense_params);

    // Load checkpoint based on mode
    match config.mode {
        ExportMode::Standalone => {
            let model_path = config
                .export_path
                .join(format!("model.{}", config.file_extension()));
            let loaded = load_checkpoint(&model_path, config)?;

            // Verify state matches
            assert_eq!(
                loaded.state.global_step, original_state.global_step,
                "Global step mismatch"
            );
            assert_eq!(
                loaded.state.hash_tables.len(),
                original_state.hash_tables.len(),
                "Hash tables count mismatch"
            );
            assert_eq!(
                loaded.state.dense_params.len(),
                original_state.dense_params.len(),
                "Dense params count mismatch"
            );

            // Verify embedding data
            for (idx, original_table) in original_state.hash_tables.iter().enumerate() {
                let loaded_table = &loaded.state.hash_tables[idx];
                assert_eq!(
                    loaded_table.name, original_table.name,
                    "Table name mismatch"
                );
                assert_eq!(
                    loaded_table.dim, original_table.dim,
                    "Table dimension mismatch"
                );
                assert_eq!(
                    loaded_table.entries.len(),
                    original_table.entries.len(),
                    "Table entries count mismatch"
                );

                // Spot check some entries
                for (key, original_value) in original_table.entries.iter().take(5) {
                    let loaded_value = loaded_table
                        .entries
                        .get(key)
                        .expect("Missing entry in loaded table");
                    assert_eq!(
                        loaded_value, original_value,
                        "Entry value mismatch for key {}",
                        key
                    );
                }
            }

            println!("  Standalone export verification PASSED");
        }
        ExportMode::Distributed => {
            let shards_dir = config.export_path.join("shards");

            // Load dense parameters
            let dense_path = shards_dir.join(format!("dense.{}", config.file_extension()));
            let dense_checkpoint = load_checkpoint(&dense_path, config)?;

            assert_eq!(
                dense_checkpoint.state.dense_params.len(),
                original_state.dense_params.len(),
                "Dense params count mismatch"
            );

            // Load and aggregate embedding shards
            let mut total_entries = 0;
            for (table_idx, _) in original_state.hash_tables.iter().enumerate() {
                for shard_idx in 0..config.num_shards {
                    let shard_filename = format!(
                        "table_{}_shard_{}.{}",
                        table_idx,
                        shard_idx,
                        config.file_extension()
                    );
                    let shard_path = shards_dir.join(&shard_filename);

                    if shard_path.exists() {
                        let shard_checkpoint = load_checkpoint(&shard_path, config)?;
                        for table in &shard_checkpoint.state.hash_tables {
                            total_entries += table.entries.len();
                        }
                    }
                }
            }

            let original_total: usize = original_state
                .hash_tables
                .iter()
                .map(|t| t.entries.len())
                .sum();

            assert_eq!(
                total_entries, original_total,
                "Total embedding entries mismatch"
            );

            println!("  Distributed export verification PASSED");
        }
    }

    Ok(())
}

/// Load a checkpoint file using the specified format.
fn load_checkpoint(path: &PathBuf, config: &ModelExportConfig) -> anyhow::Result<Checkpoint> {
    let compression = config.compression_type();

    let checkpoint = match config.format {
        SerializationFormat::Bincode => {
            let reader =
                CheckpointReader::new(BincodeSerializer::new()).with_compression(compression);
            reader.read_from_file(path)?
        }
        SerializationFormat::Json => {
            let reader = CheckpointReader::new(JsonSerializer::new()).with_compression(compression);
            reader.read_from_file(path)?
        }
        SerializationFormat::Msgpack => {
            let reader =
                CheckpointReader::new(MessagePackSerializer::new()).with_compression(compression);
            reader.read_from_file(path)?
        }
    };

    Ok(checkpoint)
}

// ============================================================================
// Main Entry Point
// ============================================================================

fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Parse command line arguments
    let args = Args::parse();

    println!("==============================================");
    println!("  Monolith-RS Model Export Example");
    println!("==============================================");
    println!();
    println!("Configuration:");
    println!("  Checkpoint dir: {:?}", args.checkpoint_dir);
    println!("  Export dir: {:?}", args.export_dir);
    println!("  Format: {}", args.format);
    println!("  Mode: {}", args.mode);
    println!("  Compression: {:?}", args.compression);
    println!("  Num shards: {}", args.num_shards);
    println!("  Version: {}", args.model_version);
    println!();

    // Create or load model
    let model_state = if args.train {
        // Train a new model
        let mut model = MovieRankingModel::new();
        model.train(args.train_steps);

        // Save checkpoint during training
        std::fs::create_dir_all(&args.checkpoint_dir)?;
        let checkpoint_config = CheckpointConfig::new(&args.checkpoint_dir)
            .with_max_to_keep(3)
            .with_checkpoint_interval(50);

        let mut manager = CheckpointManager::new(checkpoint_config, JsonCheckpointer::new());

        // Save intermediate checkpoints
        let state_50 = {
            let mut temp_model = MovieRankingModel::new();
            temp_model.train(50);
            temp_model.to_model_state()
        };
        let info = manager.save(&state_50)?;
        println!(
            "\nCheckpoint saved: {:?} (step {})",
            info.path, info.global_step
        );

        // Save final checkpoint
        let final_state = model.to_model_state();
        let info = manager.save(&final_state)?;
        println!(
            "Checkpoint saved: {:?} (step {})",
            info.path, info.global_step
        );

        // Demonstrate incremental delta
        println!(
            "\nCreating incremental delta (step 50 -> {})...",
            model.global_step
        );
        let delta = model.create_delta(50);
        println!(
            "  Delta contains {} embedding updates",
            delta
                .updated_embeddings
                .values()
                .map(|m| m.len())
                .sum::<usize>()
        );

        final_state
    } else {
        // Load existing checkpoint
        let checkpoint_config = CheckpointConfig::new(&args.checkpoint_dir);
        let manager = CheckpointManager::new(checkpoint_config, JsonCheckpointer::new());

        println!(
            "Loading latest checkpoint from {:?}...",
            args.checkpoint_dir
        );
        manager.restore_latest()?
    };

    println!(
        "\nModel state: step={}, {} tables, {} dense params",
        model_state.global_step,
        model_state.hash_tables.len(),
        model_state.dense_params.len()
    );

    // Create export configuration
    let export_config =
        ModelExportConfig::new(args.checkpoint_dir.clone(), args.export_dir.clone())
            .with_format(args.format)
            .with_mode(args.mode)
            .with_compression(args.compression)
            .with_num_shards(args.num_shards)
            .with_version(&args.model_version)
            .with_optimizer(args.include_optimizer);

    // Export the model
    let exporter = MultiFormatExporter::new(export_config.clone());
    let export_path = exporter.export(&model_state)?;

    // Export incremental delta if we have training history
    if args.train && model_state.global_step > 50 {
        let model = MovieRankingModel::new();
        let delta = model.create_delta(50);
        exporter.export_delta(&delta)?;
    }

    // Verify export if requested
    if args.verify {
        verify_export(&export_config, &model_state)?;
    }

    // Print summary
    println!("\n==============================================");
    println!("  Export Complete!");
    println!("==============================================");
    println!();
    println!("Exported model:");
    println!("  Path: {:?}", export_path);
    println!("  Format: {}", args.format);
    println!("  Mode: {}", args.mode);
    println!("  Step: {}", model_state.global_step);
    println!();

    // Demonstrate using the built-in ModelExporter for SavedModel format
    println!("Bonus: Exporting in SavedModel format using built-in exporter...");
    let saved_model_dir = args.export_dir.join("saved_model");
    let saved_model_config = ExportConfig::new(&saved_model_dir)
        .with_format(ExportFormat::SavedModel)
        .with_version(&args.model_version)
        .with_metadata("exported_by", "model_export_example");

    let saved_model_exporter = ModelExporter::new(saved_model_config);
    saved_model_exporter.export(&model_state)?;
    println!("  SavedModel exported to: {:?}", saved_model_dir);

    // Load and verify SavedModel manifest
    let manifest = ModelExporter::load_manifest(&saved_model_dir)?;
    println!(
        "  Verified manifest: version={}, step={}",
        manifest.version, manifest.global_step
    );

    println!("\nAll exports completed successfully!");

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_movie_ranking_model() {
        let mut model = MovieRankingModel::new();
        assert_eq!(model.global_step, 0);

        model.train(10);
        assert_eq!(model.global_step, 10);

        let state = model.to_model_state();
        assert_eq!(state.global_step, 10);
        assert_eq!(state.hash_tables.len(), 2);
        assert_eq!(state.dense_params.len(), 4);
    }

    #[test]
    fn test_export_config() {
        let config =
            ModelExportConfig::new(PathBuf::from("/tmp/ckpt"), PathBuf::from("/tmp/export"))
                .with_format(SerializationFormat::Json)
                .with_mode(ExportMode::Distributed)
                .with_compression(Compression::Gzip)
                .with_num_shards(4)
                .with_version("2.0.0");

        assert_eq!(config.format, SerializationFormat::Json);
        assert_eq!(config.mode, ExportMode::Distributed);
        assert_eq!(config.compression, Compression::Gzip);
        assert_eq!(config.num_shards, 4);
        assert_eq!(config.version, "2.0.0");
    }

    #[test]
    fn test_standalone_export_bincode() {
        let dir = tempdir().unwrap();
        let export_path = dir.path().join("export");

        let mut model = MovieRankingModel::new();
        model.train(5);
        let state = model.to_model_state();

        let config = ModelExportConfig::new(dir.path().to_path_buf(), export_path.clone())
            .with_format(SerializationFormat::Bincode)
            .with_mode(ExportMode::Standalone);

        let exporter = MultiFormatExporter::new(config.clone());
        exporter.export(&state).unwrap();

        // Verify export
        assert!(export_path.join("model.bin").exists());
        assert!(export_path.join("manifest.json").exists());

        verify_export(&config, &state).unwrap();
    }

    #[test]
    fn test_standalone_export_json() {
        let dir = tempdir().unwrap();
        let export_path = dir.path().join("export");

        let mut model = MovieRankingModel::new();
        model.train(5);
        let state = model.to_model_state();

        let config = ModelExportConfig::new(dir.path().to_path_buf(), export_path.clone())
            .with_format(SerializationFormat::Json)
            .with_mode(ExportMode::Standalone);

        let exporter = MultiFormatExporter::new(config.clone());
        exporter.export(&state).unwrap();

        assert!(export_path.join("model.json").exists());
        verify_export(&config, &state).unwrap();
    }

    #[test]
    fn test_standalone_export_msgpack() {
        let dir = tempdir().unwrap();
        let export_path = dir.path().join("export");

        let mut model = MovieRankingModel::new();
        model.train(5);
        let state = model.to_model_state();

        let config = ModelExportConfig::new(dir.path().to_path_buf(), export_path.clone())
            .with_format(SerializationFormat::Msgpack)
            .with_mode(ExportMode::Standalone);

        let exporter = MultiFormatExporter::new(config.clone());
        exporter.export(&state).unwrap();

        assert!(export_path.join("model.msgpack").exists());
        verify_export(&config, &state).unwrap();
    }

    #[test]
    fn test_distributed_export() {
        let dir = tempdir().unwrap();
        let export_path = dir.path().join("export");

        let mut model = MovieRankingModel::new();
        model.train(5);
        let state = model.to_model_state();

        let config = ModelExportConfig::new(dir.path().to_path_buf(), export_path.clone())
            .with_format(SerializationFormat::Bincode)
            .with_mode(ExportMode::Distributed)
            .with_num_shards(3);

        let exporter = MultiFormatExporter::new(config.clone());
        exporter.export(&state).unwrap();

        // Verify shards directory exists
        assert!(export_path.join("shards").is_dir());
        assert!(export_path.join("shards/dense.bin").exists());
        assert!(export_path.join("manifest.json").exists());

        verify_export(&config, &state).unwrap();
    }

    #[test]
    fn test_export_with_compression() {
        let dir = tempdir().unwrap();
        let export_path = dir.path().join("export");

        let mut model = MovieRankingModel::new();
        model.train(5);
        let state = model.to_model_state();

        let config = ModelExportConfig::new(dir.path().to_path_buf(), export_path.clone())
            .with_format(SerializationFormat::Bincode)
            .with_mode(ExportMode::Standalone)
            .with_compression(Compression::Gzip);

        let exporter = MultiFormatExporter::new(config.clone());
        exporter.export(&state).unwrap();

        assert!(export_path.join("model.bin").exists());
        verify_export(&config, &state).unwrap();
    }

    #[test]
    fn test_delta_export() {
        let dir = tempdir().unwrap();
        let export_path = dir.path().join("export");

        let mut model = MovieRankingModel::new();
        model.train(20);
        let delta = model.create_delta(10);

        let config = ModelExportConfig::new(dir.path().to_path_buf(), export_path.clone())
            .with_format(SerializationFormat::Json);

        let exporter = MultiFormatExporter::new(config);
        let delta_path = exporter.export_delta(&delta).unwrap();

        assert!(delta_path.exists());
    }

    #[test]
    fn test_checkpoint_manager_integration() {
        let dir = tempdir().unwrap();
        let checkpoint_dir = dir.path().join("checkpoints");

        // Train and save checkpoints
        let mut model = MovieRankingModel::new();
        model.train(30);

        let checkpoint_config = CheckpointConfig::new(&checkpoint_dir).with_max_to_keep(2);
        let mut manager = CheckpointManager::new(checkpoint_config, JsonCheckpointer::new());

        // Save multiple checkpoints
        for step in [10, 20, 30] {
            let mut temp_model = MovieRankingModel::new();
            temp_model.train(step);
            let state = temp_model.to_model_state();
            manager.save(&state).unwrap();
        }

        // Should only keep 2 checkpoints
        assert_eq!(manager.checkpoint_count(), 2);

        // Restore latest
        let restored = manager.restore_latest().unwrap();
        assert_eq!(restored.global_step, 30);
    }
}
