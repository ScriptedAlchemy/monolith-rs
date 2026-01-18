//! Multi hash table implementation for sharded embedding storage.

use crate::cuckoo::CuckooEmbeddingHashTable;
use crate::error::{HashTableError, Result};
use crate::traits::EmbeddingHashTable;

/// A sharded hash table for concurrent embedding access.
///
/// `MultiHashTable` distributes entries across multiple underlying hash tables
/// (shards) based on the feature ID. This enables better parallelism and
/// reduces contention in multi-threaded scenarios.
///
/// # Sharding Strategy
///
/// Entries are assigned to shards using `id % num_shards`. This provides
/// good distribution for sequential IDs and ensures consistent shard
/// assignment for the same ID.
///
/// # Example
///
/// ```
/// use monolith_hash_table::{MultiHashTable, EmbeddingHashTable};
///
/// // Create a table with 4 shards, each with capacity 256 and dimension 8
/// let mut table = MultiHashTable::new(4, 256, 8);
///
/// let ids = vec![1, 2, 100, 101];
/// let embeddings = vec![
///     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,  // id 1
///     0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,  // id 2
///     1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4,  // id 100
///     2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2,  // id 101
/// ];
///
/// table.assign(&ids, &embeddings).unwrap();
///
/// let mut output = vec![0.0; 32];
/// table.lookup(&ids, &mut output).unwrap();
/// ```
#[derive(Debug)]
pub struct MultiHashTable {
    /// The underlying sharded tables.
    shards: Vec<CuckooEmbeddingHashTable>,
    /// Number of shards.
    num_shards: usize,
    /// Dimension of each embedding.
    dim: usize,
}

impl MultiHashTable {
    /// Creates a new multi hash table with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `num_shards` - Number of shards to create
    /// * `capacity_per_shard` - Maximum entries per shard
    /// * `dim` - Embedding dimension
    ///
    /// # Panics
    ///
    /// Panics if `num_shards` is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_hash_table::{MultiHashTable, EmbeddingHashTable};
    ///
    /// let table = MultiHashTable::new(4, 1024, 8);
    /// assert_eq!(table.num_shards(), 4);
    /// assert_eq!(table.dim(), 8);
    /// ```
    pub fn new(num_shards: usize, capacity_per_shard: usize, dim: usize) -> Self {
        assert!(num_shards > 0, "num_shards must be greater than 0");

        let shards = (0..num_shards)
            .map(|_| CuckooEmbeddingHashTable::new(capacity_per_shard, dim))
            .collect();

        Self {
            shards,
            num_shards,
            dim,
        }
    }

    /// Creates a new multi hash table with a custom learning rate.
    ///
    /// # Arguments
    ///
    /// * `num_shards` - Number of shards to create
    /// * `capacity_per_shard` - Maximum entries per shard
    /// * `dim` - Embedding dimension
    /// * `learning_rate` - Learning rate for gradient updates
    pub fn with_learning_rate(
        num_shards: usize,
        capacity_per_shard: usize,
        dim: usize,
        learning_rate: f32,
    ) -> Self {
        assert!(num_shards > 0, "num_shards must be greater than 0");

        let shards = (0..num_shards)
            .map(|_| {
                CuckooEmbeddingHashTable::with_learning_rate(capacity_per_shard, dim, learning_rate)
            })
            .collect();

        Self {
            shards,
            num_shards,
            dim,
        }
    }

    /// Returns the number of shards.
    pub fn num_shards(&self) -> usize {
        self.num_shards
    }

    /// Returns the total capacity across all shards.
    pub fn total_capacity(&self) -> usize {
        self.shards.iter().map(|s| s.capacity()).sum()
    }

    /// Computes the shard index for a given ID.
    #[inline]
    fn shard_for_id(&self, id: i64) -> usize {
        // Handle negative IDs properly
        let positive_id = if id >= 0 {
            id as usize
        } else {
            (id.wrapping_abs() as usize).wrapping_add(1)
        };
        positive_id % self.num_shards
    }

    /// Returns a reference to a specific shard.
    ///
    /// # Arguments
    ///
    /// * `shard_idx` - The shard index
    ///
    /// # Returns
    ///
    /// A reference to the shard, or an error if the index is invalid.
    pub fn get_shard(&self, shard_idx: usize) -> Result<&CuckooEmbeddingHashTable> {
        self.shards
            .get(shard_idx)
            .ok_or(HashTableError::InvalidShardIndex {
                index: shard_idx,
                num_shards: self.num_shards,
            })
    }

    /// Returns a mutable reference to a specific shard.
    ///
    /// # Arguments
    ///
    /// * `shard_idx` - The shard index
    ///
    /// # Returns
    ///
    /// A mutable reference to the shard, or an error if the index is invalid.
    pub fn get_shard_mut(&mut self, shard_idx: usize) -> Result<&mut CuckooEmbeddingHashTable> {
        let num_shards = self.num_shards;
        self.shards
            .get_mut(shard_idx)
            .ok_or(HashTableError::InvalidShardIndex {
                index: shard_idx,
                num_shards,
            })
    }

    /// Sets the learning rate for all shards.
    pub fn set_learning_rate(&mut self, lr: f32) {
        for shard in &mut self.shards {
            shard.set_learning_rate(lr);
        }
    }

    /// Clears all entries from all shards.
    pub fn clear(&mut self) {
        for shard in &mut self.shards {
            shard.clear();
        }
    }

    /// Returns the total memory usage across all shards in bytes.
    pub fn memory_usage(&self) -> usize {
        self.shards.iter().map(|s| s.memory_usage()).sum::<usize>() + std::mem::size_of::<Self>()
    }

    /// Returns the size of each shard.
    pub fn shard_sizes(&self) -> Vec<usize> {
        self.shards.iter().map(|s| s.size()).collect()
    }

    /// Returns the load factor of each shard.
    pub fn shard_load_factors(&self) -> Vec<f64> {
        self.shards.iter().map(|s| s.load_factor()).collect()
    }

    /// Validates that the embedding buffer size matches the expected size.
    fn validate_buffer_size(&self, ids_len: usize, buffer_len: usize) -> Result<()> {
        let expected = ids_len * self.dim;
        if buffer_len != expected {
            return Err(HashTableError::DimensionMismatch {
                expected,
                actual: buffer_len,
            });
        }
        Ok(())
    }
}

impl EmbeddingHashTable for MultiHashTable {
    fn lookup(&self, ids: &[i64], embeddings: &mut [f32]) -> Result<()> {
        self.validate_buffer_size(ids.len(), embeddings.len())?;

        for (i, &id) in ids.iter().enumerate() {
            let shard_idx = self.shard_for_id(id);
            let shard = &self.shards[shard_idx];

            let start = i * self.dim;
            let end = start + self.dim;

            // Look up single ID in the appropriate shard
            let mut single_embedding = vec![0.0; self.dim];
            shard.lookup(&[id], &mut single_embedding)?;
            embeddings[start..end].copy_from_slice(&single_embedding);
        }

        Ok(())
    }

    fn assign(&mut self, ids: &[i64], embeddings: &[f32]) -> Result<()> {
        self.validate_buffer_size(ids.len(), embeddings.len())?;

        for (i, &id) in ids.iter().enumerate() {
            let shard_idx = self.shard_for_id(id);
            let shard = &mut self.shards[shard_idx];

            let start = i * self.dim;
            let end = start + self.dim;

            shard.assign(&[id], &embeddings[start..end])?;
        }

        Ok(())
    }

    fn apply_gradients(&mut self, ids: &[i64], gradients: &[f32]) -> Result<()> {
        self.validate_buffer_size(ids.len(), gradients.len())?;

        for (i, &id) in ids.iter().enumerate() {
            let shard_idx = self.shard_for_id(id);
            let shard = &mut self.shards[shard_idx];

            let start = i * self.dim;
            let end = start + self.dim;

            shard.apply_gradients(&[id], &gradients[start..end])?;
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.shards.iter().map(|s| s.size()).sum()
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn contains(&self, id: i64) -> bool {
        let shard_idx = self.shard_for_id(id);
        self.shards[shard_idx].contains(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_new() {
        let table = MultiHashTable::new(4, 256, 8);
        assert_eq!(table.num_shards(), 4);
        assert_eq!(table.dim(), 8);
        assert_eq!(table.size(), 0);
        assert_eq!(table.total_capacity(), 1024);
    }

    #[test]
    #[should_panic(expected = "num_shards must be greater than 0")]
    fn test_multi_zero_shards() {
        MultiHashTable::new(0, 256, 8);
    }

    #[test]
    fn test_multi_sharding() {
        let mut table = MultiHashTable::new(4, 256, 2);

        // Insert IDs that map to different shards
        let ids = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let embeddings = vec![
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0,
        ];

        table.assign(&ids, &embeddings).unwrap();

        // Check shard sizes - should be roughly even
        let shard_sizes = table.shard_sizes();
        assert_eq!(shard_sizes.iter().sum::<usize>(), 8);

        // Verify lookups work correctly
        let mut output = vec![0.0; 16];
        table.lookup(&ids, &mut output).unwrap();
        assert_eq!(output, embeddings);
    }

    #[test]
    fn test_multi_negative_ids() {
        let mut table = MultiHashTable::new(4, 256, 2);

        let ids = vec![-1, -2, -100, 1, 2, 100];
        let embeddings = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0];

        table.assign(&ids, &embeddings).unwrap();

        let mut output = vec![0.0; 12];
        table.lookup(&ids, &mut output).unwrap();
        assert_eq!(output, embeddings);
    }

    #[test]
    fn test_multi_apply_gradients() {
        let mut table = MultiHashTable::with_learning_rate(4, 256, 2, 0.1);

        let ids = vec![1, 2, 3, 4];
        let embeddings = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];

        table.assign(&ids, &embeddings).unwrap();

        let gradients = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        table.apply_gradients(&ids, &gradients).unwrap();

        let mut output = vec![0.0; 8];
        table.lookup(&ids, &mut output).unwrap();

        // After SGD: embedding - lr * gradient
        assert!((output[0] - 0.9).abs() < 1e-6);
        assert!((output[2] - 1.9).abs() < 1e-6);
        assert!((output[4] - 2.9).abs() < 1e-6);
        assert!((output[6] - 3.9).abs() < 1e-6);
    }

    #[test]
    fn test_multi_contains() {
        let mut table = MultiHashTable::new(4, 256, 2);

        table
            .assign(&[1, 5, 9], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();

        assert!(table.contains(1));
        assert!(table.contains(5));
        assert!(table.contains(9));
        assert!(!table.contains(2));
        assert!(!table.contains(100));
    }

    #[test]
    fn test_multi_clear() {
        let mut table = MultiHashTable::new(4, 256, 2);

        table
            .assign(&[1, 2, 3, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .unwrap();
        assert_eq!(table.size(), 4);

        table.clear();
        assert_eq!(table.size(), 0);

        for shard_size in table.shard_sizes() {
            assert_eq!(shard_size, 0);
        }
    }

    #[test]
    fn test_multi_get_shard() {
        let table = MultiHashTable::new(4, 256, 2);

        assert!(table.get_shard(0).is_ok());
        assert!(table.get_shard(3).is_ok());
        assert!(matches!(
            table.get_shard(4),
            Err(HashTableError::InvalidShardIndex {
                index: 4,
                num_shards: 4
            })
        ));
    }

    #[test]
    fn test_multi_set_learning_rate() {
        let mut table = MultiHashTable::new(4, 256, 2);

        table.set_learning_rate(0.5);

        for i in 0..4 {
            let shard = table.get_shard(i).unwrap();
            assert!((shard.learning_rate() - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multi_dimension_mismatch() {
        let mut table = MultiHashTable::new(4, 256, 4);

        let result = table.assign(&[1], &[1.0, 2.0]); // Should be 4 elements
        assert!(matches!(
            result,
            Err(HashTableError::DimensionMismatch { .. })
        ));
    }
}
