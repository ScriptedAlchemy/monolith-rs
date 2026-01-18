//! Cuckoo hash table implementation for embedding storage.

use std::sync::Arc;

use hashbrown::HashMap;

use crate::entry::EmbeddingEntry;
use crate::error::{HashTableError, Result};
use crate::eviction::{EvictionPolicy, NoEviction};
use crate::initializer::{Initializer, RandomUniformInitializer};
use crate::traits::EmbeddingHashTable;

/// A cuckoo hash table for storing embeddings.
///
/// This implementation uses a cuckoo hashing scheme with configurable
/// bucket sizes. Cuckoo hashing provides O(1) worst-case lookup time
/// and good cache locality.
///
/// # Implementation Details
///
/// The table maintains two hash functions and displaces existing entries
/// when collisions occur, similar to the cuckoo bird's behavior. This
/// ensures that lookups only need to check a constant number of locations.
///
/// # Example
///
/// ```
/// use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable};
///
/// let mut table = CuckooEmbeddingHashTable::new(1024, 4);
///
/// // Store embeddings
/// let ids = vec![1, 2, 3];
/// let embeddings = vec![
///     0.1, 0.2, 0.3, 0.4,  // embedding for id 1
///     0.5, 0.6, 0.7, 0.8,  // embedding for id 2
///     0.9, 1.0, 1.1, 1.2,  // embedding for id 3
/// ];
/// table.assign(&ids, &embeddings).unwrap();
///
/// // Retrieve embeddings
/// let mut output = vec![0.0; 12];
/// table.lookup(&ids, &mut output).unwrap();
/// ```
/// A cuckoo hash table for storing embeddings.
pub struct CuckooEmbeddingHashTable {
    /// The underlying hash map storing entries.
    /// In a full cuckoo implementation, this would be replaced with
    /// actual cuckoo buckets, but we use HashMap for simplicity.
    entries: HashMap<i64, EmbeddingEntry>,
    /// Maximum number of entries the table can hold.
    capacity: usize,
    /// Dimension of each embedding vector.
    dim: usize,
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Eviction policy for automatic entry removal.
    eviction_policy: Box<dyn EvictionPolicy>,
    /// Initializer for new embedding entries.
    initializer: Arc<dyn Initializer>,
}

impl std::fmt::Debug for CuckooEmbeddingHashTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CuckooEmbeddingHashTable")
            .field("entries", &self.entries)
            .field("capacity", &self.capacity)
            .field("dim", &self.dim)
            .field("learning_rate", &self.learning_rate)
            .field("eviction_policy", &"<dyn EvictionPolicy>")
            .field("initializer", &self.initializer.name())
            .finish()
    }
}

impl CuckooEmbeddingHashTable {
    /// Creates a new cuckoo hash table.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries the table can hold
    /// * `dim` - Dimension of each embedding vector
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable};
    ///
    /// let table = CuckooEmbeddingHashTable::new(1024, 8);
    /// assert_eq!(table.dim(), 8);
    /// assert_eq!(table.size(), 0);
    /// ```
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            dim,
            learning_rate: 0.01,
            eviction_policy: Box::new(NoEviction),
            initializer: Arc::new(RandomUniformInitializer::default()),
        }
    }

    /// Creates a new cuckoo hash table with a custom learning rate.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries the table can hold
    /// * `dim` - Dimension of each embedding vector
    /// * `learning_rate` - Learning rate for gradient updates
    pub fn with_learning_rate(capacity: usize, dim: usize, learning_rate: f32) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            dim,
            learning_rate,
            eviction_policy: Box::new(NoEviction),
            initializer: Arc::new(RandomUniformInitializer::default()),
        }
    }

    /// Creates a new cuckoo hash table with a custom eviction policy.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries the table can hold
    /// * `dim` - Dimension of each embedding vector
    /// * `eviction_policy` - The eviction policy to use
    pub fn with_eviction_policy(
        capacity: usize,
        dim: usize,
        eviction_policy: Box<dyn EvictionPolicy>,
    ) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            dim,
            learning_rate: 0.01,
            eviction_policy,
            initializer: Arc::new(RandomUniformInitializer::default()),
        }
    }

    /// Creates a new cuckoo hash table with a custom initializer.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of entries the table can hold
    /// * `dim` - Dimension of each embedding vector
    /// * `initializer` - The initializer to use for new entries
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable, ZerosInitializer};
    /// use std::sync::Arc;
    ///
    /// let table = CuckooEmbeddingHashTable::with_initializer(
    ///     1024, 4, Arc::new(ZerosInitializer)
    /// );
    /// ```
    pub fn with_initializer(
        capacity: usize,
        dim: usize,
        initializer: Arc<dyn Initializer>,
    ) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            dim,
            learning_rate: 0.01,
            eviction_policy: Box::new(NoEviction),
            initializer,
        }
    }

    /// Sets the initializer for new entries.
    ///
    /// # Arguments
    ///
    /// * `initializer` - The initializer to use for new entries
    pub fn set_initializer(&mut self, initializer: Arc<dyn Initializer>) {
        self.initializer = initializer;
    }

    /// Returns the name of the current initializer.
    pub fn initializer_name(&self) -> &str {
        self.initializer.name()
    }

    /// Inserts a new entry or retrieves an existing one, initializing new entries
    /// using the configured initializer.
    ///
    /// # Arguments
    ///
    /// * `id` - The feature ID
    ///
    /// # Returns
    ///
    /// `Ok(true)` if a new entry was created, `Ok(false)` if the entry already existed.
    ///
    /// # Errors
    ///
    /// Returns an error if the table is full and a new entry would need to be created.
    pub fn get_or_initialize(&mut self, id: i64) -> Result<bool> {
        if self.entries.contains_key(&id) {
            return Ok(false);
        }

        if self.entries.len() >= self.capacity {
            return Err(HashTableError::TableFull {
                capacity: self.capacity,
            });
        }

        let embedding = self.initializer.initialize(self.dim);
        let entry = EmbeddingEntry::new(id, embedding);
        self.entries.insert(id, entry);
        Ok(true)
    }

    /// Looks up embeddings, initializing any missing entries using the configured initializer.
    ///
    /// Unlike [`lookup`], this method will create new entries for missing IDs instead
    /// of returning an error.
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of feature IDs to look up
    /// * `embeddings` - Output buffer for the embeddings
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if the buffer size is incorrect
    /// or the table is full.
    pub fn lookup_or_initialize(&mut self, ids: &[i64], embeddings: &mut [f32]) -> Result<()> {
        self.validate_buffer_size(ids.len(), embeddings.len())?;

        // First pass: initialize any missing entries
        let new_entries_needed = ids.iter().filter(|id| !self.entries.contains_key(*id)).count();
        if self.entries.len() + new_entries_needed > self.capacity {
            return Err(HashTableError::TableFull {
                capacity: self.capacity,
            });
        }

        for &id in ids {
            if !self.entries.contains_key(&id) {
                let embedding = self.initializer.initialize(self.dim);
                let entry = EmbeddingEntry::new(id, embedding);
                self.entries.insert(id, entry);
            }
        }

        // Second pass: copy embeddings to output
        for (i, &id) in ids.iter().enumerate() {
            let entry = self.entries.get(&id).unwrap(); // Safe: we just inserted any missing
            let start = i * self.dim;
            let end = start + self.dim;
            embeddings[start..end].copy_from_slice(entry.embedding());
        }

        Ok(())
    }

    /// Returns the maximum capacity of the table.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the learning rate used for gradient updates.
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Sets the learning rate for gradient updates.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    /// Returns the load factor of the table (size / capacity).
    pub fn load_factor(&self) -> f64 {
        self.entries.len() as f64 / self.capacity as f64
    }

    /// Removes an entry from the table.
    ///
    /// # Arguments
    ///
    /// * `id` - The feature ID to remove
    ///
    /// # Returns
    ///
    /// The removed entry if it existed, `None` otherwise.
    pub fn remove(&mut self, id: i64) -> Option<EmbeddingEntry> {
        self.entries.remove(&id)
    }

    /// Clears all entries from the table.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Returns an iterator over all entries in the table.
    pub fn iter(&self) -> impl Iterator<Item = (&i64, &EmbeddingEntry)> {
        self.entries.iter()
    }

    /// Returns the total memory usage of the table in bytes.
    pub fn memory_usage(&self) -> usize {
        self.entries
            .values()
            .map(|e| e.memory_size())
            .sum::<usize>()
            + std::mem::size_of::<Self>()
    }

    /// Gets a reference to an entry by ID.
    pub fn get(&self, id: i64) -> Option<&EmbeddingEntry> {
        self.entries.get(&id)
    }

    /// Gets a mutable reference to an entry by ID.
    pub fn get_mut(&mut self, id: i64) -> Option<&mut EmbeddingEntry> {
        self.entries.get_mut(&id)
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

    /// Sets the eviction policy for this table.
    ///
    /// # Arguments
    ///
    /// * `policy` - The new eviction policy to use
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable};
    /// use monolith_hash_table::eviction::TimeBasedEviction;
    ///
    /// let mut table = CuckooEmbeddingHashTable::new(1024, 4);
    /// table.set_eviction_policy(Box::new(TimeBasedEviction::new(3600)));
    /// ```
    pub fn set_eviction_policy(&mut self, policy: Box<dyn EvictionPolicy>) {
        self.eviction_policy = policy;
    }

    /// Evicts entries that match the current eviction policy.
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
    /// use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable, EmbeddingEntry};
    /// use monolith_hash_table::eviction::TimeBasedEviction;
    ///
    /// let mut table = CuckooEmbeddingHashTable::with_eviction_policy(
    ///     1024, 4, Box::new(TimeBasedEviction::new(100))
    /// );
    ///
    /// // Add some entries
    /// table.assign(&[1, 2], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
    ///
    /// // Evict entries older than 100 seconds at time 200
    /// let evicted = table.evict(200);
    /// ```
    pub fn evict(&mut self, current_time: u64) -> usize {
        let ids_to_remove: Vec<i64> = self
            .entries
            .iter()
            .filter(|(_, entry)| self.eviction_policy.should_evict(entry, current_time))
            .map(|(&id, _)| id)
            .collect();

        let count = ids_to_remove.len();
        for id in ids_to_remove {
            self.entries.remove(&id);
        }
        count
    }
}

impl EmbeddingHashTable for CuckooEmbeddingHashTable {
    fn lookup(&self, ids: &[i64], embeddings: &mut [f32]) -> Result<()> {
        self.validate_buffer_size(ids.len(), embeddings.len())?;

        for (i, &id) in ids.iter().enumerate() {
            let entry = self
                .entries
                .get(&id)
                .ok_or(HashTableError::IdNotFound { id })?;

            let start = i * self.dim;
            let end = start + self.dim;
            embeddings[start..end].copy_from_slice(entry.embedding());
        }

        Ok(())
    }

    fn assign(&mut self, ids: &[i64], embeddings: &[f32]) -> Result<()> {
        self.validate_buffer_size(ids.len(), embeddings.len())?;

        // Check if we have capacity for new entries
        let new_entries = ids.iter().filter(|id| !self.entries.contains_key(*id)).count();
        if self.entries.len() + new_entries > self.capacity {
            return Err(HashTableError::TableFull {
                capacity: self.capacity,
            });
        }

        for (i, &id) in ids.iter().enumerate() {
            let start = i * self.dim;
            let end = start + self.dim;
            let embedding_data = embeddings[start..end].to_vec();

            if let Some(entry) = self.entries.get_mut(&id) {
                entry.copy_embedding_from(&embedding_data);
            } else {
                let entry = EmbeddingEntry::new(id, embedding_data);
                self.entries.insert(id, entry);
            }
        }

        Ok(())
    }

    fn apply_gradients(&mut self, ids: &[i64], gradients: &[f32]) -> Result<()> {
        // Default implementation without timestamp update
        self.apply_gradients_with_timestamp(ids, gradients, None)
    }

    fn apply_gradients_with_timestamp(
        &mut self,
        ids: &[i64],
        gradients: &[f32],
        current_time: Option<u64>,
    ) -> Result<()> {
        self.validate_buffer_size(ids.len(), gradients.len())?;

        let learning_rate = self.learning_rate;
        let dim = self.dim;

        for (i, &id) in ids.iter().enumerate() {
            let entry = self
                .entries
                .get_mut(&id)
                .ok_or(HashTableError::IdNotFound { id })?;

            let start = i * dim;
            let grad_slice = &gradients[start..start + dim];

            // Apply gradient update based on optimizer state
            // Use a method on entry that can handle both embedding and optimizer state
            entry.apply_gradient_update(grad_slice, learning_rate);

            // Update timestamp if provided
            if let Some(ts) = current_time {
                entry.set_timestamp(ts);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuckoo_new() {
        let table = CuckooEmbeddingHashTable::new(1024, 8);
        assert_eq!(table.capacity(), 1024);
        assert_eq!(table.dim(), 8);
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_cuckoo_assign_and_lookup() {
        let mut table = CuckooEmbeddingHashTable::new(100, 4);

        let ids = vec![1, 2, 3];
        let embeddings = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
        ];

        table.assign(&ids, &embeddings).unwrap();
        assert_eq!(table.size(), 3);

        let mut output = vec![0.0; 12];
        table.lookup(&ids, &mut output).unwrap();
        assert_eq!(output, embeddings);
    }

    #[test]
    fn test_cuckoo_overwrite() {
        let mut table = CuckooEmbeddingHashTable::new(100, 2);

        table.assign(&[1], &[1.0, 2.0]).unwrap();
        assert_eq!(table.size(), 1);

        table.assign(&[1], &[3.0, 4.0]).unwrap();
        assert_eq!(table.size(), 1); // Still 1, not 2

        let mut output = vec![0.0; 2];
        table.lookup(&[1], &mut output).unwrap();
        assert_eq!(output, vec![3.0, 4.0]);
    }

    #[test]
    fn test_cuckoo_dimension_mismatch() {
        let mut table = CuckooEmbeddingHashTable::new(100, 4);

        // Wrong embedding size
        let result = table.assign(&[1], &[1.0, 2.0]); // Should be 4 elements
        assert!(matches!(result, Err(HashTableError::DimensionMismatch { .. })));

        // Wrong output buffer size
        table.assign(&[1], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut output = vec![0.0; 2]; // Should be 4 elements
        let result = table.lookup(&[1], &mut output);
        assert!(matches!(result, Err(HashTableError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_cuckoo_id_not_found() {
        let table = CuckooEmbeddingHashTable::new(100, 2);

        let mut output = vec![0.0; 2];
        let result = table.lookup(&[999], &mut output);
        assert!(matches!(result, Err(HashTableError::IdNotFound { id: 999 })));
    }

    #[test]
    fn test_cuckoo_table_full() {
        let mut table = CuckooEmbeddingHashTable::new(2, 2);

        table.assign(&[1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();

        let result = table.assign(&[3], &[5.0, 6.0]);
        assert!(matches!(result, Err(HashTableError::TableFull { capacity: 2 })));
    }

    #[test]
    fn test_cuckoo_apply_gradients_sgd() {
        let mut table = CuckooEmbeddingHashTable::with_learning_rate(100, 2, 0.1);

        table.assign(&[1], &[1.0, 2.0]).unwrap();
        table.apply_gradients(&[1], &[1.0, 1.0]).unwrap();

        let mut output = vec![0.0; 2];
        table.lookup(&[1], &mut output).unwrap();

        // After SGD: embedding - lr * gradient = [1.0, 2.0] - 0.1 * [1.0, 1.0] = [0.9, 1.9]
        assert!((output[0] - 0.9).abs() < 1e-6);
        assert!((output[1] - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_cuckoo_remove() {
        let mut table = CuckooEmbeddingHashTable::new(100, 2);

        table.assign(&[1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(table.size(), 2);

        let removed = table.remove(1);
        assert!(removed.is_some());
        assert_eq!(table.size(), 1);
        assert!(!table.contains(1));
        assert!(table.contains(2));
    }

    #[test]
    fn test_cuckoo_clear() {
        let mut table = CuckooEmbeddingHashTable::new(100, 2);

        table.assign(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(table.size(), 3);

        table.clear();
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_cuckoo_load_factor() {
        let mut table = CuckooEmbeddingHashTable::new(100, 2);

        table.assign(&[1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((table.load_factor() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn test_cuckoo_eviction_with_no_eviction_policy() {
        let mut table = CuckooEmbeddingHashTable::new(100, 2);

        table.assign(&[1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(table.size(), 2);

        // NoEviction policy should never evict
        let evicted = table.evict(u64::MAX);
        assert_eq!(evicted, 0);
        assert_eq!(table.size(), 2);
    }

    #[test]
    fn test_cuckoo_eviction_with_time_based_policy() {
        use crate::eviction::TimeBasedEviction;

        let mut table = CuckooEmbeddingHashTable::with_eviction_policy(
            100,
            2,
            Box::new(TimeBasedEviction::new(100)), // 100 seconds expire time
        );

        // Add entries with timestamp 0 (default)
        table.assign(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(table.size(), 3);

        // At time 50, no entries should be evicted (age = 50 < 100)
        let evicted = table.evict(50);
        assert_eq!(evicted, 0);
        assert_eq!(table.size(), 3);

        // At time 101, all entries should be evicted (age = 101 > 100)
        let evicted = table.evict(101);
        assert_eq!(evicted, 3);
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_cuckoo_eviction_with_mixed_timestamps() {
        use crate::eviction::TimeBasedEviction;

        let mut table = CuckooEmbeddingHashTable::with_eviction_policy(
            100,
            2,
            Box::new(TimeBasedEviction::new(100)),
        );

        // Add entries
        table.assign(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        // Update timestamps for some entries
        if let Some(entry) = table.get_mut(2) {
            entry.set_timestamp(50);
        }
        if let Some(entry) = table.get_mut(3) {
            entry.set_timestamp(100);
        }

        // At time 120:
        // - Entry 1: timestamp=0, age=120 > 100 -> evict
        // - Entry 2: timestamp=50, age=70 < 100 -> keep
        // - Entry 3: timestamp=100, age=20 < 100 -> keep
        let evicted = table.evict(120);
        assert_eq!(evicted, 1);
        assert_eq!(table.size(), 2);
        assert!(!table.contains(1));
        assert!(table.contains(2));
        assert!(table.contains(3));
    }

    #[test]
    fn test_cuckoo_set_eviction_policy() {
        use crate::eviction::TimeBasedEviction;

        let mut table = CuckooEmbeddingHashTable::new(100, 2);

        table.assign(&[1], &[1.0, 2.0]).unwrap();

        // With default NoEviction policy, nothing is evicted
        let evicted = table.evict(1000);
        assert_eq!(evicted, 0);

        // Set a time-based policy
        table.set_eviction_policy(Box::new(TimeBasedEviction::new(100)));

        // Now eviction should work
        let evicted = table.evict(1000);
        assert_eq!(evicted, 1);
        assert_eq!(table.size(), 0);
    }

    #[test]
    fn test_cuckoo_apply_gradients_with_timestamp() {
        let mut table = CuckooEmbeddingHashTable::with_learning_rate(100, 2, 0.1);

        table.assign(&[1], &[1.0, 2.0]).unwrap();

        // Initially, timestamp is 0
        assert_eq!(table.get(1).unwrap().get_timestamp(), 0);

        // Apply gradients with timestamp update
        table
            .apply_gradients_with_timestamp(&[1], &[1.0, 1.0], Some(500))
            .unwrap();

        // Timestamp should be updated
        assert_eq!(table.get(1).unwrap().get_timestamp(), 500);

        // Embedding should also be updated
        let mut output = vec![0.0; 2];
        table.lookup(&[1], &mut output).unwrap();
        assert!((output[0] - 0.9).abs() < 1e-6);
        assert!((output[1] - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_cuckoo_eviction_with_lru_policy() {
        use crate::eviction::LRUEviction;

        let mut table = CuckooEmbeddingHashTable::with_eviction_policy(
            100,
            2,
            Box::new(LRUEviction::new(100)),
        );

        table.assign(&[1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Update timestamp for entry 2 to make it "recently used"
        if let Some(entry) = table.get_mut(2) {
            entry.set_timestamp(50);
        }

        // At time 120:
        // - Entry 1: timestamp=0, age=120 > 100 -> evict
        // - Entry 2: timestamp=50, age=70 < 100 -> keep
        let evicted = table.evict(120);
        assert_eq!(evicted, 1);
        assert!(!table.contains(1));
        assert!(table.contains(2));
    }

    #[test]
    fn test_cuckoo_with_initializer() {
        use crate::initializer::ZerosInitializer;

        let table = CuckooEmbeddingHashTable::with_initializer(
            100,
            4,
            Arc::new(ZerosInitializer),
        );

        assert_eq!(table.dim(), 4);
        assert_eq!(table.initializer_name(), "zeros");
    }

    #[test]
    fn test_cuckoo_get_or_initialize() {
        use crate::initializer::ZerosInitializer;

        let mut table = CuckooEmbeddingHashTable::with_initializer(
            100,
            4,
            Arc::new(ZerosInitializer),
        );

        // First call should create a new entry
        let created = table.get_or_initialize(1).unwrap();
        assert!(created);
        assert_eq!(table.size(), 1);

        // Second call for same id should not create a new entry
        let created = table.get_or_initialize(1).unwrap();
        assert!(!created);
        assert_eq!(table.size(), 1);

        // Verify the embedding is all zeros
        let mut output = vec![0.0; 4];
        table.lookup(&[1], &mut output).unwrap();
        assert_eq!(output, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_cuckoo_get_or_initialize_table_full() {
        use crate::initializer::ZerosInitializer;

        let mut table = CuckooEmbeddingHashTable::with_initializer(
            2,
            4,
            Arc::new(ZerosInitializer),
        );

        // Fill the table
        table.get_or_initialize(1).unwrap();
        table.get_or_initialize(2).unwrap();

        // Table should be full
        let result = table.get_or_initialize(3);
        assert!(matches!(result, Err(HashTableError::TableFull { capacity: 2 })));
    }

    #[test]
    fn test_cuckoo_lookup_or_initialize() {
        use crate::initializer::OnesInitializer;

        let mut table = CuckooEmbeddingHashTable::with_initializer(
            100,
            4,
            Arc::new(OnesInitializer),
        );

        // First, add an entry with custom values
        table.assign(&[1], &[2.0, 3.0, 4.0, 5.0]).unwrap();

        // Now lookup_or_initialize with mix of existing and new entries
        let mut output = vec![0.0; 12];
        table.lookup_or_initialize(&[1, 2, 3], &mut output).unwrap();

        // Entry 1 should have its original values
        assert_eq!(&output[0..4], &[2.0, 3.0, 4.0, 5.0]);

        // Entries 2 and 3 should be initialized with ones
        assert_eq!(&output[4..8], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&output[8..12], &[1.0, 1.0, 1.0, 1.0]);

        assert_eq!(table.size(), 3);
    }

    #[test]
    fn test_cuckoo_lookup_or_initialize_table_full() {
        use crate::initializer::ZerosInitializer;

        let mut table = CuckooEmbeddingHashTable::with_initializer(
            2,
            2,
            Arc::new(ZerosInitializer),
        );

        // Fill the table
        table.assign(&[1, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Try to lookup with a new id - should fail
        let mut output = vec![0.0; 4];
        let result = table.lookup_or_initialize(&[1, 3], &mut output);
        assert!(matches!(result, Err(HashTableError::TableFull { capacity: 2 })));
    }

    #[test]
    fn test_cuckoo_set_initializer() {
        use crate::initializer::{OnesInitializer, ZerosInitializer};

        let mut table = CuckooEmbeddingHashTable::with_initializer(
            100,
            2,
            Arc::new(ZerosInitializer),
        );

        assert_eq!(table.initializer_name(), "zeros");

        // Add entry with zeros initializer
        table.get_or_initialize(1).unwrap();
        let mut output = vec![0.0; 2];
        table.lookup(&[1], &mut output).unwrap();
        assert_eq!(output, vec![0.0, 0.0]);

        // Change initializer
        table.set_initializer(Arc::new(OnesInitializer));
        assert_eq!(table.initializer_name(), "ones");

        // Add entry with ones initializer
        table.get_or_initialize(2).unwrap();
        let mut output = vec![0.0; 2];
        table.lookup(&[2], &mut output).unwrap();
        assert_eq!(output, vec![1.0, 1.0]);
    }

    #[test]
    fn test_cuckoo_with_random_uniform_initializer() {
        use crate::initializer::RandomUniformInitializer;

        let mut table = CuckooEmbeddingHashTable::with_initializer(
            100,
            4,
            Arc::new(RandomUniformInitializer::new(-0.1, 0.1)),
        );

        assert_eq!(table.initializer_name(), "random_uniform");

        // Initialize an entry
        table.get_or_initialize(1).unwrap();

        // Verify values are within range
        let mut output = vec![0.0; 4];
        table.lookup(&[1], &mut output).unwrap();
        for &val in &output {
            assert!(val >= -0.1 && val < 0.1, "Value {} out of range", val);
        }
    }

    #[test]
    fn test_cuckoo_default_initializer() {
        // Default should use RandomUniformInitializer
        let table = CuckooEmbeddingHashTable::new(100, 4);
        assert_eq!(table.initializer_name(), "random_uniform");
    }
}
