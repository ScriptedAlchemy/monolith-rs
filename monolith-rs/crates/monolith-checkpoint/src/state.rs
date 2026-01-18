//! Model state representation for checkpointing.
//!
//! This module defines the structures that represent all saveable state
//! in a Monolith model, including embedding tables and optimizer state.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// State of a single embedding hash table.
///
/// Contains the serialized embeddings and metadata needed to restore
/// the table to its previous state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HashTableState {
    /// Name/identifier of this hash table.
    pub name: String,

    /// Embedding dimension for this table.
    pub dim: usize,

    /// Serialized key-value pairs.
    /// Keys are feature IDs (i64), values are flattened embedding vectors.
    pub entries: HashMap<i64, Vec<f32>>,

    /// Optional metadata about the table configuration.
    pub metadata: HashMap<String, String>,
}

impl HashTableState {
    /// Create a new empty hash table state.
    ///
    /// # Arguments
    ///
    /// * `name` - Name/identifier for this hash table
    /// * `dim` - Embedding dimension
    pub fn new(name: impl Into<String>, dim: usize) -> Self {
        Self {
            name: name.into(),
            dim,
            entries: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add an entry to the hash table state.
    ///
    /// # Arguments
    ///
    /// * `key` - Feature ID
    /// * `embedding` - Embedding vector (must have length equal to `dim`)
    ///
    /// # Panics
    ///
    /// Panics if the embedding dimension doesn't match.
    pub fn insert(&mut self, key: i64, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.dim,
            "Embedding dimension mismatch: expected {}, got {}",
            self.dim,
            embedding.len()
        );
        self.entries.insert(key, embedding);
    }

    /// Get the number of entries in this table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// State of an optimizer for a specific parameter group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Name of the optimizer (e.g., "adam", "sgd", "adagrad").
    pub optimizer_type: String,

    /// Name of the parameter group this optimizer is associated with.
    pub param_group: String,

    /// Learning rate at checkpoint time.
    pub learning_rate: f64,

    /// Current training step/iteration.
    pub step: u64,

    /// First moment estimates (for Adam-like optimizers).
    /// Keyed by parameter name.
    pub first_moments: HashMap<String, Vec<f32>>,

    /// Second moment estimates (for Adam-like optimizers).
    /// Keyed by parameter name.
    pub second_moments: HashMap<String, Vec<f32>>,

    /// Accumulated gradients (for Adagrad-like optimizers).
    /// Keyed by parameter name.
    pub accumulators: HashMap<String, Vec<f32>>,

    /// Additional optimizer-specific state.
    pub extra_state: HashMap<String, Vec<u8>>,
}

impl OptimizerState {
    /// Create a new optimizer state.
    ///
    /// # Arguments
    ///
    /// * `optimizer_type` - Type of optimizer (e.g., "adam", "sgd")
    /// * `param_group` - Name of the parameter group
    /// * `learning_rate` - Current learning rate
    /// * `step` - Current training step
    pub fn new(
        optimizer_type: impl Into<String>,
        param_group: impl Into<String>,
        learning_rate: f64,
        step: u64,
    ) -> Self {
        Self {
            optimizer_type: optimizer_type.into(),
            param_group: param_group.into(),
            learning_rate,
            step,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            accumulators: HashMap::new(),
            extra_state: HashMap::new(),
        }
    }
}

/// Complete model state for checkpointing.
///
/// This struct contains all the state needed to save and restore
/// a Monolith model, including:
///
/// - Embedding hash tables
/// - Optimizer state
/// - Training metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Version of the checkpoint format.
    pub version: u32,

    /// Global training step at checkpoint time.
    pub global_step: u64,

    /// Timestamp when checkpoint was created (Unix epoch seconds).
    pub timestamp: u64,

    /// State of all embedding hash tables.
    pub hash_tables: Vec<HashTableState>,

    /// State of all optimizers.
    pub optimizers: Vec<OptimizerState>,

    /// Dense model parameters (non-embedding weights).
    /// Keyed by parameter name, values are flattened tensors.
    pub dense_params: HashMap<String, Vec<f32>>,

    /// Additional metadata about the model/training.
    pub metadata: HashMap<String, String>,
}

impl ModelState {
    /// Create a new empty model state.
    ///
    /// # Arguments
    ///
    /// * `global_step` - Current global training step
    pub fn new(global_step: u64) -> Self {
        Self {
            version: 1,
            global_step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            hash_tables: Vec::new(),
            optimizers: Vec::new(),
            dense_params: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add a hash table state.
    pub fn add_hash_table(&mut self, table: HashTableState) {
        self.hash_tables.push(table);
    }

    /// Add an optimizer state.
    pub fn add_optimizer(&mut self, optimizer: OptimizerState) {
        self.optimizers.push(optimizer);
    }

    /// Add a dense parameter.
    pub fn add_dense_param(&mut self, name: impl Into<String>, values: Vec<f32>) {
        self.dense_params.insert(name.into(), values);
    }

    /// Set metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Get the total number of embedding entries across all tables.
    pub fn total_embeddings(&self) -> usize {
        self.hash_tables.iter().map(|t| t.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_table_state_new() {
        let table = HashTableState::new("user_embeddings", 64);
        assert_eq!(table.name, "user_embeddings");
        assert_eq!(table.dim, 64);
        assert!(table.is_empty());
    }

    #[test]
    fn test_hash_table_state_insert() {
        let mut table = HashTableState::new("test", 4);
        table.insert(1, vec![1.0, 2.0, 3.0, 4.0]);
        table.insert(2, vec![5.0, 6.0, 7.0, 8.0]);

        assert_eq!(table.len(), 2);
        assert!(!table.is_empty());
        assert_eq!(table.entries.get(&1), Some(&vec![1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    #[should_panic(expected = "Embedding dimension mismatch")]
    fn test_hash_table_state_insert_wrong_dim() {
        let mut table = HashTableState::new("test", 4);
        table.insert(1, vec![1.0, 2.0, 3.0]); // Wrong dimension
    }

    #[test]
    fn test_optimizer_state_new() {
        let opt = OptimizerState::new("adam", "embeddings", 0.001, 1000);
        assert_eq!(opt.optimizer_type, "adam");
        assert_eq!(opt.param_group, "embeddings");
        assert_eq!(opt.learning_rate, 0.001);
        assert_eq!(opt.step, 1000);
    }

    #[test]
    fn test_model_state_new() {
        let state = ModelState::new(5000);
        assert_eq!(state.version, 1);
        assert_eq!(state.global_step, 5000);
        assert!(state.hash_tables.is_empty());
        assert!(state.optimizers.is_empty());
    }

    #[test]
    fn test_model_state_add_components() {
        let mut state = ModelState::new(1000);

        let mut table = HashTableState::new("embeddings", 32);
        table.insert(1, vec![0.0; 32]);
        table.insert(2, vec![1.0; 32]);
        state.add_hash_table(table);

        let opt = OptimizerState::new("sgd", "embeddings", 0.01, 1000);
        state.add_optimizer(opt);

        state.add_dense_param("fc1.weight", vec![0.5; 100]);
        state.set_metadata("model_name", "test_model");

        assert_eq!(state.hash_tables.len(), 1);
        assert_eq!(state.optimizers.len(), 1);
        assert_eq!(state.dense_params.len(), 1);
        assert_eq!(state.total_embeddings(), 2);
        assert_eq!(
            state.metadata.get("model_name"),
            Some(&"test_model".to_string())
        );
    }

    #[test]
    fn test_model_state_serialization() {
        let mut state = ModelState::new(100);
        state.set_metadata("test", "value");

        let json = serde_json::to_string(&state).expect("Failed to serialize");
        let restored: ModelState = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(restored.global_step, 100);
        assert_eq!(restored.metadata.get("test"), Some(&"value".to_string()));
    }
}
