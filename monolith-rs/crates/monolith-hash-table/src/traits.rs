//! Core traits for embedding hash tables.

use crate::Result;

/// A trait defining the interface for embedding hash tables.
///
/// This trait provides the core operations needed for storing and retrieving
/// embeddings in a hash table structure. Implementations may vary in their
/// underlying storage mechanism, concurrency support, and performance
/// characteristics.
///
/// # Thread Safety
///
/// Implementations should document their thread safety guarantees. Some
/// implementations may require external synchronization for concurrent access.
///
/// # Example
///
/// ```
/// use monolith_hash_table::{EmbeddingHashTable, CuckooEmbeddingHashTable};
///
/// let mut table = CuckooEmbeddingHashTable::new(1024, 8);
///
/// // Store embeddings
/// let ids = vec![1, 2];
/// let embeddings = vec![0.1; 16]; // 2 embeddings of dim 8
/// table.assign(&ids, &embeddings).unwrap();
///
/// // Retrieve embeddings
/// let mut output = vec![0.0; 16];
/// table.lookup(&ids, &mut output).unwrap();
/// ```
pub trait EmbeddingHashTable {
    /// Looks up embeddings for the given IDs.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of feature IDs to look up
    /// * `embeddings` - Output buffer for the embeddings, must have length `ids.len() * dim()`
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all IDs were found, or an error if any ID is missing
    /// or the output buffer has incorrect size.
    ///
    /// # Errors
    ///
    /// * [`HashTableError::IdNotFound`] - If any ID is not in the table
    /// * [`HashTableError::DimensionMismatch`] - If the output buffer size is incorrect
    fn lookup(&self, ids: &[i64], embeddings: &mut [f32]) -> Result<()>;

    /// Assigns embeddings to the given IDs.
    ///
    /// If an ID already exists, its embedding is overwritten. If the ID is new,
    /// it is inserted into the table.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of feature IDs to assign
    /// * `embeddings` - Embeddings to store, must have length `ids.len() * dim()`
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// * [`HashTableError::TableFull`] - If the table cannot accommodate new entries
    /// * [`HashTableError::DimensionMismatch`] - If the embedding buffer size is incorrect
    fn assign(&mut self, ids: &[i64], embeddings: &[f32]) -> Result<()>;

    /// Applies gradients to update embeddings for the given IDs.
    ///
    /// This method is used during training to update embeddings based on
    /// computed gradients. The update rule depends on the optimizer
    /// configuration of the table.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of feature IDs to update
    /// * `gradients` - Gradients to apply, must have length `ids.len() * dim()`
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success.
    ///
    /// # Errors
    ///
    /// * [`HashTableError::IdNotFound`] - If any ID is not in the table
    /// * [`HashTableError::DimensionMismatch`] - If the gradient buffer size is incorrect
    fn apply_gradients(&mut self, ids: &[i64], gradients: &[f32]) -> Result<()>;

    /// Returns the number of entries in the table.
    fn size(&self) -> usize;

    /// Returns the embedding dimension.
    fn dim(&self) -> usize;

    /// Checks if the table contains an entry for the given ID.
    ///
    /// # Arguments
    ///
    /// * `id` - The feature ID to check
    ///
    /// # Returns
    ///
    /// `true` if the ID exists in the table, `false` otherwise.
    fn contains(&self, id: i64) -> bool;

    /// Evicts entries that match the table's eviction policy.
    ///
    /// This method removes entries from the table based on the configured
    /// eviction policy. The default implementation does nothing and returns 0.
    ///
    /// # Arguments
    ///
    /// * `current_time` - The current timestamp (Unix timestamp in seconds)
    ///
    /// # Returns
    ///
    /// The number of entries that were evicted.
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_hash_table::{EmbeddingHashTable, CuckooEmbeddingHashTable};
    /// use monolith_hash_table::eviction::TimeBasedEviction;
    ///
    /// let mut table = CuckooEmbeddingHashTable::with_eviction_policy(
    ///     1024, 4, Box::new(TimeBasedEviction::new(3600))
    /// );
    ///
    /// // Add some entries
    /// table.assign(&[1, 2], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    ///
    /// // Evict entries older than 1 hour
    /// let evicted_count = table.evict(7200);
    /// ```
    fn evict(&mut self, _current_time: u64) -> usize {
        // Default implementation: no eviction
        0
    }

    /// Applies gradients with an optional timestamp update.
    ///
    /// This is similar to `apply_gradients`, but also updates the entry's
    /// timestamp if provided. The default implementation ignores the timestamp
    /// and delegates to `apply_gradients`.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of feature IDs to update
    /// * `gradients` - Gradients to apply, must have length `ids.len() * dim()`
    /// * `current_time` - Optional timestamp to set on updated entries
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success.
    fn apply_gradients_with_timestamp(
        &mut self,
        ids: &[i64],
        gradients: &[f32],
        _current_time: Option<u64>,
    ) -> Result<()> {
        // Default implementation: ignore timestamp
        self.apply_gradients(ids, gradients)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A simple mock implementation for testing the trait
    struct MockHashTable {
        dim: usize,
        entries: hashbrown::HashMap<i64, Vec<f32>>,
    }

    impl MockHashTable {
        fn new(dim: usize) -> Self {
            Self {
                dim,
                entries: hashbrown::HashMap::new(),
            }
        }
    }

    impl EmbeddingHashTable for MockHashTable {
        fn lookup(&self, ids: &[i64], embeddings: &mut [f32]) -> Result<()> {
            if embeddings.len() != ids.len() * self.dim {
                return Err(crate::HashTableError::DimensionMismatch {
                    expected: ids.len() * self.dim,
                    actual: embeddings.len(),
                });
            }

            for (i, &id) in ids.iter().enumerate() {
                let entry = self
                    .entries
                    .get(&id)
                    .ok_or(crate::HashTableError::IdNotFound { id })?;
                let start = i * self.dim;
                embeddings[start..start + self.dim].copy_from_slice(entry);
            }
            Ok(())
        }

        fn assign(&mut self, ids: &[i64], embeddings: &[f32]) -> Result<()> {
            if embeddings.len() != ids.len() * self.dim {
                return Err(crate::HashTableError::DimensionMismatch {
                    expected: ids.len() * self.dim,
                    actual: embeddings.len(),
                });
            }

            for (i, &id) in ids.iter().enumerate() {
                let start = i * self.dim;
                let embedding = embeddings[start..start + self.dim].to_vec();
                self.entries.insert(id, embedding);
            }
            Ok(())
        }

        fn apply_gradients(&mut self, ids: &[i64], gradients: &[f32]) -> Result<()> {
            if gradients.len() != ids.len() * self.dim {
                return Err(crate::HashTableError::DimensionMismatch {
                    expected: ids.len() * self.dim,
                    actual: gradients.len(),
                });
            }

            for (i, &id) in ids.iter().enumerate() {
                let entry = self
                    .entries
                    .get_mut(&id)
                    .ok_or(crate::HashTableError::IdNotFound { id })?;
                let start = i * self.dim;
                for (j, grad) in gradients[start..start + self.dim].iter().enumerate() {
                    entry[j] -= grad * 0.01; // Simple SGD with lr=0.01
                }
            }
            Ok(())
        }

        fn size(&self) -> usize {
            self.entries.len()
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn contains(&self, id: i64) -> bool {
            self.entries.contains_key(&id)
        }
    }

    #[test]
    fn test_mock_hash_table() {
        let mut table = MockHashTable::new(4);

        // Assign
        let ids = vec![1, 2];
        let embeddings = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        table.assign(&ids, &embeddings).unwrap();

        assert_eq!(table.size(), 2);
        assert!(table.contains(1));
        assert!(table.contains(2));
        assert!(!table.contains(3));

        // Lookup
        let mut output = vec![0.0; 8];
        table.lookup(&ids, &mut output).unwrap();
        assert_eq!(output, embeddings);

        // Apply gradients
        let gradients = vec![0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1];
        table.apply_gradients(&ids, &gradients).unwrap();

        table.lookup(&ids, &mut output).unwrap();
        // After SGD: embedding - 0.01 * gradient
        assert!((output[0] - 0.999).abs() < 1e-6);
    }
}
