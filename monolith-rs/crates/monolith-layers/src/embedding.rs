//! Embedding lookup layer.
//!
//! This module provides the [`EmbeddingLookup`] layer for looking up
//! embeddings from a hash table, commonly used for sparse feature representations.

use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A simple hash table for embedding storage.
///
/// This is a placeholder that would normally wrap the monolith-hash-table crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingHashTable {
    /// Embeddings stored by feature ID
    embeddings: HashMap<u64, Vec<f32>>,
    /// Embedding dimension
    dim: usize,
    /// Default embedding for missing keys
    default_embedding: Vec<f32>,
}

impl EmbeddingHashTable {
    /// Creates a new embedding hash table.
    ///
    /// # Arguments
    ///
    /// * `dim` - The embedding dimension
    pub fn new(dim: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dim,
            default_embedding: vec![0.0; dim],
        }
    }

    /// Creates a hash table with initial capacity.
    ///
    /// # Arguments
    ///
    /// * `dim` - The embedding dimension
    /// * `capacity` - Initial capacity for the hash table
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            embeddings: HashMap::with_capacity(capacity),
            dim,
            default_embedding: vec![0.0; dim],
        }
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of embeddings stored.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Returns true if the table is empty.
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Inserts or updates an embedding.
    ///
    /// # Arguments
    ///
    /// * `id` - The feature ID
    /// * `embedding` - The embedding vector
    ///
    /// # Panics
    ///
    /// Panics if the embedding dimension doesn't match
    pub fn insert(&mut self, id: u64, embedding: Vec<f32>) {
        assert_eq!(
            embedding.len(),
            self.dim,
            "Embedding dimension mismatch: expected {}, got {}",
            self.dim,
            embedding.len()
        );
        self.embeddings.insert(id, embedding);
    }

    /// Looks up an embedding by ID.
    ///
    /// Returns the default embedding if the ID is not found.
    pub fn get(&self, id: u64) -> &[f32] {
        self.embeddings
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&self.default_embedding)
    }

    /// Looks up an embedding by ID, returning None if not found.
    pub fn get_opt(&self, id: u64) -> Option<&[f32]> {
        self.embeddings.get(&id).map(|v| v.as_slice())
    }

    /// Checks if an ID exists in the table.
    pub fn contains(&self, id: u64) -> bool {
        self.embeddings.contains_key(&id)
    }

    /// Removes an embedding by ID.
    pub fn remove(&mut self, id: u64) -> Option<Vec<f32>> {
        self.embeddings.remove(&id)
    }

    /// Sets the default embedding for missing keys.
    pub fn set_default(&mut self, default: Vec<f32>) {
        assert_eq!(
            default.len(),
            self.dim,
            "Default embedding dimension mismatch"
        );
        self.default_embedding = default;
    }

    /// Initializes embeddings with random values.
    ///
    /// # Arguments
    ///
    /// * `ids` - The feature IDs to initialize
    /// * `std` - Standard deviation for normal initialization
    pub fn initialize_random(&mut self, ids: &[u64], std: f32) {
        let mut seed: u64 = 42;
        for &id in ids {
            let embedding: Vec<f32> = (0..self.dim)
                .map(|_| {
                    // Box-Muller transform
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let u1 = ((seed >> 16) & 0x7fff) as f32 / 32768.0 + 1e-10;
                    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                    let u2 = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                    z * std
                })
                .collect();
            self.embeddings.insert(id, embedding);
        }
    }
}

/// Embedding lookup layer.
///
/// Looks up embeddings from a hash table based on feature IDs.
/// This is commonly used for sparse categorical features in recommendation systems.
///
/// # Example
///
/// ```
/// use monolith_layers::embedding::{EmbeddingLookup, EmbeddingHashTable};
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let mut hash_table = EmbeddingHashTable::new(8);
/// hash_table.insert(1, vec![0.1; 8]);
/// hash_table.insert(2, vec![0.2; 8]);
///
/// let lookup = EmbeddingLookup::new(hash_table);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingLookup {
    /// The underlying hash table for embeddings
    hash_table: EmbeddingHashTable,
    /// Cached feature IDs for backward pass
    cached_ids: Option<Vec<u64>>,
    /// Gradient accumulator for embeddings
    grad_accumulator: HashMap<u64, Vec<f32>>,
}

impl EmbeddingLookup {
    /// Creates a new embedding lookup layer.
    ///
    /// # Arguments
    ///
    /// * `hash_table` - The hash table containing embeddings
    pub fn new(hash_table: EmbeddingHashTable) -> Self {
        Self {
            hash_table,
            cached_ids: None,
            grad_accumulator: HashMap::new(),
        }
    }

    /// Creates an embedding lookup with a new hash table.
    ///
    /// # Arguments
    ///
    /// * `dim` - The embedding dimension
    pub fn with_dim(dim: usize) -> Self {
        Self::new(EmbeddingHashTable::new(dim))
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.hash_table.dim()
    }

    /// Returns a reference to the underlying hash table.
    pub fn hash_table(&self) -> &EmbeddingHashTable {
        &self.hash_table
    }

    /// Returns a mutable reference to the underlying hash table.
    pub fn hash_table_mut(&mut self) -> &mut EmbeddingHashTable {
        &mut self.hash_table
    }

    /// Looks up embeddings for given feature IDs.
    ///
    /// # Arguments
    ///
    /// * `ids` - Feature IDs to look up
    ///
    /// # Returns
    ///
    /// A tensor of shape [len(ids), embedding_dim]
    pub fn lookup(&self, ids: &[u64]) -> Tensor {
        let dim = self.hash_table.dim();
        let mut data = Vec::with_capacity(ids.len() * dim);

        for &id in ids {
            data.extend_from_slice(self.hash_table.get(id));
        }

        Tensor::from_data(&[ids.len(), dim], data)
    }

    /// Looks up embeddings and caches IDs for backward pass.
    ///
    /// # Arguments
    ///
    /// * `ids` - Feature IDs to look up
    pub fn lookup_train(&mut self, ids: &[u64]) -> Tensor {
        self.cached_ids = Some(ids.to_vec());
        self.lookup(ids)
    }

    /// Accumulates gradients for the embeddings.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient tensor of shape [num_ids, embedding_dim]
    pub fn accumulate_grad(&mut self, grad: &Tensor) -> Result<(), LayerError> {
        let ids = self.cached_ids.as_ref().ok_or(LayerError::NotInitialized)?;

        if grad.shape()[0] != ids.len() {
            return Err(LayerError::ShapeMismatch {
                expected: vec![ids.len(), self.dim()],
                actual: grad.shape().to_vec(),
            });
        }

        let dim = self.dim();
        for (i, &id) in ids.iter().enumerate() {
            let grad_slice = &grad.data()[i * dim..(i + 1) * dim];

            self.grad_accumulator
                .entry(id)
                .and_modify(|acc| {
                    for (a, &g) in acc.iter_mut().zip(grad_slice) {
                        *a += g;
                    }
                })
                .or_insert_with(|| grad_slice.to_vec());
        }

        Ok(())
    }

    /// Returns and clears the accumulated gradients.
    pub fn take_gradients(&mut self) -> HashMap<u64, Vec<f32>> {
        std::mem::take(&mut self.grad_accumulator)
    }

    /// Applies gradients to the embeddings.
    ///
    /// # Arguments
    ///
    /// * `learning_rate` - The learning rate for the update
    pub fn apply_gradients(&mut self, learning_rate: f32) {
        for (id, grad) in self.grad_accumulator.drain() {
            if let Some(embedding) = self.hash_table.embeddings.get_mut(&id) {
                for (e, g) in embedding.iter_mut().zip(grad.iter()) {
                    *e -= learning_rate * g;
                }
            }
        }
    }

    /// Clears cached values and gradients.
    pub fn clear_cache(&mut self) {
        self.cached_ids = None;
        self.grad_accumulator.clear();
    }
}

impl Layer for EmbeddingLookup {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Input is expected to be a 1D tensor of feature IDs encoded as f32
        if input.ndim() != 1 && input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "EmbeddingLookup expects 1D or 2D input, got {}D",
                    input.ndim()
                ),
            });
        }

        let ids: Vec<u64> = if input.ndim() == 1 {
            input.data().iter().map(|&x| x as u64).collect()
        } else {
            // For 2D input, flatten
            input.data().iter().map(|&x| x as u64).collect()
        };

        Ok(self.lookup(&ids))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        // Accumulate gradients for the embeddings
        self.accumulate_grad(grad)?;

        // For embeddings, we don't propagate gradients to the input (IDs)
        // Return a zero tensor with the same shape as cached_ids
        let num_ids = self.cached_ids.as_ref().map(|ids| ids.len()).unwrap_or(0);
        Ok(Tensor::zeros(&[num_ids]))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // Embedding tables don't expose parameters as tensors directly
        // They use a different update mechanism through apply_gradients
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "EmbeddingLookup"
    }
}

/// Configuration for pooled embedding lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PoolingMode {
    /// Sum pooling
    Sum,
    /// Mean pooling
    Mean,
    /// Max pooling
    Max,
}

impl Default for PoolingMode {
    fn default() -> Self {
        Self::Sum
    }
}

/// Pooled embedding lookup for variable-length feature lists.
///
/// Looks up embeddings for multiple feature IDs and pools them together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PooledEmbeddingLookup {
    /// The underlying embedding lookup
    lookup: EmbeddingLookup,
    /// Pooling mode
    pooling: PoolingMode,
}

impl PooledEmbeddingLookup {
    /// Creates a new pooled embedding lookup.
    ///
    /// # Arguments
    ///
    /// * `hash_table` - The hash table containing embeddings
    /// * `pooling` - The pooling mode to use
    pub fn new(hash_table: EmbeddingHashTable, pooling: PoolingMode) -> Self {
        Self {
            lookup: EmbeddingLookup::new(hash_table),
            pooling,
        }
    }

    /// Looks up and pools embeddings for feature ID lists.
    ///
    /// # Arguments
    ///
    /// * `id_lists` - List of feature ID lists (one per sample in batch)
    ///
    /// # Returns
    ///
    /// A tensor of shape [batch_size, embedding_dim]
    pub fn lookup_pooled(&self, id_lists: &[Vec<u64>]) -> Tensor {
        let dim = self.lookup.dim();
        let batch_size = id_lists.len();
        let mut output = vec![0.0; batch_size * dim];

        for (i, ids) in id_lists.iter().enumerate() {
            if ids.is_empty() {
                continue;
            }

            let embeddings = self.lookup.lookup(ids);

            match self.pooling {
                PoolingMode::Sum => {
                    for (j, emb) in embeddings.data().chunks(dim).enumerate() {
                        if j < ids.len() {
                            for (k, &val) in emb.iter().enumerate() {
                                output[i * dim + k] += val;
                            }
                        }
                    }
                }
                PoolingMode::Mean => {
                    for emb in embeddings.data().chunks(dim) {
                        for (k, &val) in emb.iter().enumerate() {
                            output[i * dim + k] += val;
                        }
                    }
                    let count = ids.len() as f32;
                    for k in 0..dim {
                        output[i * dim + k] /= count;
                    }
                }
                PoolingMode::Max => {
                    // Initialize with first embedding
                    let first = self.lookup.hash_table().get(ids[0]);
                    for (k, &val) in first.iter().enumerate() {
                        output[i * dim + k] = val;
                    }
                    // Take element-wise max with remaining embeddings
                    for &id in &ids[1..] {
                        let emb = self.lookup.hash_table().get(id);
                        for (k, &val) in emb.iter().enumerate() {
                            output[i * dim + k] = output[i * dim + k].max(val);
                        }
                    }
                }
            }
        }

        Tensor::from_data(&[batch_size, dim], output)
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.lookup.dim()
    }

    /// Returns a reference to the underlying hash table.
    pub fn hash_table(&self) -> &EmbeddingHashTable {
        self.lookup.hash_table()
    }

    /// Returns a mutable reference to the underlying hash table.
    pub fn hash_table_mut(&mut self) -> &mut EmbeddingHashTable {
        self.lookup.hash_table_mut()
    }
}

/// Sequence embedding lookup that outputs first N embeddings with zero-padding.
///
/// This is equivalent to Python's `FirstN` combiner. It extracts the first
/// `max_seq_length` embeddings from each sample's feature list and zero-pads
/// shorter sequences.
///
/// # Example
///
/// ```
/// use monolith_layers::embedding::{SequenceEmbeddingLookup, EmbeddingHashTable};
///
/// let mut table = EmbeddingHashTable::new(4);
/// table.insert(1, vec![1.0; 4]);
/// table.insert(2, vec![2.0; 4]);
///
/// let seq_lookup = SequenceEmbeddingLookup::new(table, 3); // max 3 items
///
/// // Batch of 2 samples with variable-length sequences
/// let id_lists = vec![vec![1, 2], vec![1]];
/// let result = seq_lookup.lookup_sequence(&id_lists);
/// // Shape: [2, 3, 4] - second sample is zero-padded
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceEmbeddingLookup {
    /// The underlying embedding lookup
    lookup: EmbeddingLookup,
    /// Maximum sequence length (first N items)
    max_seq_length: usize,
}

impl SequenceEmbeddingLookup {
    /// Creates a new sequence embedding lookup (FirstN combiner).
    ///
    /// # Arguments
    ///
    /// * `hash_table` - The hash table containing embeddings
    /// * `max_seq_length` - Maximum sequence length (must be > 0)
    ///
    /// # Panics
    ///
    /// Panics if `max_seq_length` is 0.
    pub fn new(hash_table: EmbeddingHashTable, max_seq_length: usize) -> Self {
        assert!(max_seq_length > 0, "max_seq_length must be greater than 0");
        Self {
            lookup: EmbeddingLookup::new(hash_table),
            max_seq_length,
        }
    }

    /// Returns the maximum sequence length.
    pub fn max_seq_length(&self) -> usize {
        self.max_seq_length
    }

    /// Returns the embedding dimension.
    pub fn dim(&self) -> usize {
        self.lookup.dim()
    }

    /// Returns a reference to the underlying hash table.
    pub fn hash_table(&self) -> &EmbeddingHashTable {
        self.lookup.hash_table()
    }

    /// Returns a mutable reference to the underlying hash table.
    pub fn hash_table_mut(&mut self) -> &mut EmbeddingHashTable {
        self.lookup.hash_table_mut()
    }

    /// Looks up embeddings as sequences with zero-padding.
    ///
    /// Takes the first `max_seq_length` embeddings from each sample.
    /// Shorter sequences are zero-padded at the end.
    ///
    /// # Arguments
    ///
    /// * `id_lists` - List of feature ID lists (one per sample in batch)
    ///
    /// # Returns
    ///
    /// A tensor of shape [batch_size, max_seq_length, embedding_dim]
    pub fn lookup_sequence(&self, id_lists: &[Vec<u64>]) -> Tensor {
        let dim = self.lookup.dim();
        let batch_size = id_lists.len();
        let seq_len = self.max_seq_length;

        // Initialize with zeros (handles padding automatically)
        let mut output = vec![0.0; batch_size * seq_len * dim];

        for (batch_idx, ids) in id_lists.iter().enumerate() {
            // Take at most max_seq_length items
            let num_items = ids.len().min(seq_len);

            for (seq_idx, &id) in ids.iter().take(num_items).enumerate() {
                let emb = self.lookup.hash_table().get(id);
                let offset = batch_idx * seq_len * dim + seq_idx * dim;
                output[offset..offset + dim].copy_from_slice(emb);
            }
        }

        Tensor::from_data(&[batch_size, seq_len, dim], output)
    }

    /// Looks up embeddings as sequences for training (caches IDs).
    ///
    /// # Arguments
    ///
    /// * `id_lists` - List of feature ID lists (one per sample in batch)
    pub fn lookup_sequence_train(&mut self, id_lists: &[Vec<u64>]) -> Tensor {
        // Cache all IDs for backward pass
        let all_ids: Vec<u64> = id_lists
            .iter()
            .flat_map(|ids| ids.iter().take(self.max_seq_length).copied())
            .collect();
        self.lookup.cached_ids = Some(all_ids);

        self.lookup_sequence(id_lists)
    }

    /// Accumulates gradients for sequence embeddings.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient tensor of shape [batch_size, max_seq_length, embedding_dim]
    /// * `id_lists` - The original ID lists used for lookup
    pub fn accumulate_grad(
        &mut self,
        grad: &Tensor,
        id_lists: &[Vec<u64>],
    ) -> Result<(), LayerError> {
        let dim = self.dim();
        let seq_len = self.max_seq_length;

        for (batch_idx, ids) in id_lists.iter().enumerate() {
            let num_items = ids.len().min(seq_len);

            for (seq_idx, &id) in ids.iter().take(num_items).enumerate() {
                let offset = batch_idx * seq_len * dim + seq_idx * dim;
                let grad_slice = &grad.data()[offset..offset + dim];

                self.lookup
                    .grad_accumulator
                    .entry(id)
                    .and_modify(|acc| {
                        for (a, &g) in acc.iter_mut().zip(grad_slice) {
                            *a += g;
                        }
                    })
                    .or_insert_with(|| grad_slice.to_vec());
            }
        }

        Ok(())
    }

    /// Returns and clears the accumulated gradients.
    pub fn take_gradients(&mut self) -> HashMap<u64, Vec<f32>> {
        self.lookup.take_gradients()
    }

    /// Applies gradients to the embeddings.
    pub fn apply_gradients(&mut self, learning_rate: f32) {
        self.lookup.apply_gradients(learning_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_hash_table() {
        let mut table = EmbeddingHashTable::new(4);
        assert_eq!(table.dim(), 4);
        assert!(table.is_empty());

        table.insert(1, vec![1.0, 2.0, 3.0, 4.0]);
        table.insert(2, vec![5.0, 6.0, 7.0, 8.0]);

        assert_eq!(table.len(), 2);
        assert!(table.contains(1));
        assert!(!table.contains(3));

        assert_eq!(table.get(1), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(table.get(2), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(table.get(999), &[0.0, 0.0, 0.0, 0.0]); // Default
    }

    #[test]
    fn test_embedding_lookup() {
        let mut table = EmbeddingHashTable::new(4);
        table.insert(1, vec![1.0, 1.0, 1.0, 1.0]);
        table.insert(2, vec![2.0, 2.0, 2.0, 2.0]);
        table.insert(3, vec![3.0, 3.0, 3.0, 3.0]);

        let lookup = EmbeddingLookup::new(table);
        let result = lookup.lookup(&[1, 2, 3]);

        assert_eq!(result.shape(), &[3, 4]);
        assert_eq!(&result.data()[0..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&result.data()[4..8], &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(&result.data()[8..12], &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_embedding_lookup_layer() {
        let mut table = EmbeddingHashTable::new(4);
        table.insert(1, vec![1.0, 1.0, 1.0, 1.0]);
        table.insert(2, vec![2.0, 2.0, 2.0, 2.0]);

        let lookup = EmbeddingLookup::new(table);

        // Input as tensor of IDs
        let input = Tensor::from_data(&[3], vec![1.0, 2.0, 1.0]);
        let output = lookup.forward(&input).unwrap();

        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_embedding_gradient_accumulation() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 1.0]);
        table.insert(2, vec![2.0, 2.0]);

        let mut lookup = EmbeddingLookup::new(table);

        // Lookup and cache IDs
        let _ = lookup.lookup_train(&[1, 2, 1]);

        // Accumulate gradients
        let grad = Tensor::from_data(&[3, 2], vec![0.1, 0.1, 0.2, 0.2, 0.3, 0.3]);
        lookup.accumulate_grad(&grad).unwrap();

        let grads = lookup.take_gradients();

        // ID 1 appears twice, so gradients should be summed
        assert_eq!(grads.get(&1).unwrap(), &[0.4, 0.4]); // 0.1 + 0.3
        assert_eq!(grads.get(&2).unwrap(), &[0.2, 0.2]);
    }

    #[test]
    fn test_pooled_embedding_sum() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 1.0]);
        table.insert(2, vec![2.0, 2.0]);
        table.insert(3, vec![3.0, 3.0]);

        let pooled = PooledEmbeddingLookup::new(table, PoolingMode::Sum);

        let id_lists = vec![vec![1, 2], vec![3]];
        let result = pooled.lookup_pooled(&id_lists);

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(&result.data()[0..2], &[3.0, 3.0]); // 1+2
        assert_eq!(&result.data()[2..4], &[3.0, 3.0]); // just 3
    }

    #[test]
    fn test_pooled_embedding_mean() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 1.0]);
        table.insert(2, vec![3.0, 3.0]);

        let pooled = PooledEmbeddingLookup::new(table, PoolingMode::Mean);

        let id_lists = vec![vec![1, 2]];
        let result = pooled.lookup_pooled(&id_lists);

        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(&result.data()[0..2], &[2.0, 2.0]); // (1+3)/2
    }

    #[test]
    fn test_pooled_embedding_max() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 3.0]);
        table.insert(2, vec![2.0, 1.0]);

        let pooled = PooledEmbeddingLookup::new(table, PoolingMode::Max);

        let id_lists = vec![vec![1, 2]];
        let result = pooled.lookup_pooled(&id_lists);

        assert_eq!(result.shape(), &[1, 2]);
        assert_eq!(&result.data()[0..2], &[2.0, 3.0]); // max(1,2), max(3,1)
    }

    #[test]
    fn test_embedding_apply_gradients() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 1.0]);

        let mut lookup = EmbeddingLookup::new(table);
        let _ = lookup.lookup_train(&[1]);

        let grad = Tensor::from_data(&[1, 2], vec![1.0, 2.0]);
        lookup.accumulate_grad(&grad).unwrap();

        lookup.apply_gradients(0.1);

        // New embedding = old - lr * grad = [1, 1] - 0.1 * [1, 2] = [0.9, 0.8]
        let new_emb = lookup.hash_table().get(1);
        assert!((new_emb[0] - 0.9).abs() < 1e-6);
        assert!((new_emb[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_sequence_embedding_lookup() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 1.0]);
        table.insert(2, vec![2.0, 2.0]);
        table.insert(3, vec![3.0, 3.0]);

        let seq_lookup = SequenceEmbeddingLookup::new(table, 3);
        assert_eq!(seq_lookup.max_seq_length(), 3);
        assert_eq!(seq_lookup.dim(), 2);

        // Batch of 2: first has 2 items, second has 1 item
        let id_lists = vec![vec![1, 2], vec![3]];
        let result = seq_lookup.lookup_sequence(&id_lists);

        // Shape: [2, 3, 2]
        assert_eq!(result.shape(), &[2, 3, 2]);

        // First sample: [1,1], [2,2], [0,0] (zero-padded)
        assert_eq!(&result.data()[0..2], &[1.0, 1.0]);
        assert_eq!(&result.data()[2..4], &[2.0, 2.0]);
        assert_eq!(&result.data()[4..6], &[0.0, 0.0]);

        // Second sample: [3,3], [0,0], [0,0] (zero-padded)
        assert_eq!(&result.data()[6..8], &[3.0, 3.0]);
        assert_eq!(&result.data()[8..10], &[0.0, 0.0]);
        assert_eq!(&result.data()[10..12], &[0.0, 0.0]);
    }

    #[test]
    fn test_sequence_embedding_truncation() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 1.0]);
        table.insert(2, vec![2.0, 2.0]);
        table.insert(3, vec![3.0, 3.0]);
        table.insert(4, vec![4.0, 4.0]);

        // max_seq_length = 2, so should truncate longer sequences
        let seq_lookup = SequenceEmbeddingLookup::new(table, 2);

        let id_lists = vec![vec![1, 2, 3, 4]]; // 4 items, but max is 2
        let result = seq_lookup.lookup_sequence(&id_lists);

        // Shape: [1, 2, 2] - only first 2 items
        assert_eq!(result.shape(), &[1, 2, 2]);
        assert_eq!(&result.data()[0..2], &[1.0, 1.0]);
        assert_eq!(&result.data()[2..4], &[2.0, 2.0]);
    }

    #[test]
    fn test_sequence_embedding_gradient() {
        let mut table = EmbeddingHashTable::new(2);
        table.insert(1, vec![1.0, 1.0]);
        table.insert(2, vec![2.0, 2.0]);

        let mut seq_lookup = SequenceEmbeddingLookup::new(table, 2);

        let id_lists = vec![vec![1, 2], vec![1]];
        let _output = seq_lookup.lookup_sequence_train(&id_lists);

        // Gradient shape: [2, 2, 2] = [batch=2, seq=2, dim=2]
        let grad = Tensor::from_data(
            &[2, 2, 2],
            vec![
                0.1, 0.1, // batch 0, seq 0 (id=1)
                0.2, 0.2, // batch 0, seq 1 (id=2)
                0.3, 0.3, // batch 1, seq 0 (id=1)
                0.0, 0.0, // batch 1, seq 1 (padding, no grad)
            ],
        );
        seq_lookup.accumulate_grad(&grad, &id_lists).unwrap();

        let grads = seq_lookup.take_gradients();
        // id=1 appears at batch0/seq0 and batch1/seq0: 0.1 + 0.3 = 0.4
        assert_eq!(grads.get(&1).unwrap(), &[0.4, 0.4]);
        // id=2 appears at batch0/seq1 only: 0.2
        assert_eq!(grads.get(&2).unwrap(), &[0.2, 0.2]);
    }
}
