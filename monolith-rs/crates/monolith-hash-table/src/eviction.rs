//! Eviction policies for embedding hash tables.
//!
//! This module provides various eviction policies that can be used to
//! automatically remove entries from hash tables based on different criteria.
//!
//! # Available Policies
//!
//! - [`NoEviction`] - Never evict entries (default)
//! - [`TimeBasedEviction`] - Evict entries older than a specified expire time
//! - [`LRUEviction`] - Evict least recently used entries when over capacity

use std::collections::HashMap;

use crate::entry::EmbeddingEntry;

/// A slot identifier for per-slot eviction configuration.
pub type SlotId = u32;

/// A trait defining the interface for eviction policies.
///
/// Implementations determine when entries should be removed from the hash table
/// based on various criteria such as age, access patterns, or capacity limits.
///
/// # Thread Safety
///
/// All implementations must be `Send + Sync` to support concurrent access.
///
/// # Example
///
/// ```
/// use monolith_hash_table::eviction::{EvictionPolicy, TimeBasedEviction};
/// use monolith_hash_table::EmbeddingEntry;
///
/// let policy = TimeBasedEviction::new(3600); // 1 hour expire time
/// let mut entry = EmbeddingEntry::with_timestamp(1, vec![0.1, 0.2], 1000);
///
/// // Entry is old, should be evicted
/// assert!(policy.should_evict(&entry, 5000));
/// ```
pub trait EvictionPolicy: Send + Sync {
    /// Determines whether an entry should be evicted.
    ///
    /// # Arguments
    ///
    /// * `entry` - The embedding entry to check
    /// * `current_time` - The current timestamp (Unix timestamp in seconds)
    ///
    /// # Returns
    ///
    /// `true` if the entry should be evicted, `false` otherwise.
    fn should_evict(&self, entry: &EmbeddingEntry, current_time: u64) -> bool;
}

/// A policy that never evicts entries.
///
/// This is the default policy and is useful when you want to manage
/// eviction manually or when entries should never be automatically removed.
///
/// # Example
///
/// ```
/// use monolith_hash_table::eviction::{EvictionPolicy, NoEviction};
/// use monolith_hash_table::EmbeddingEntry;
///
/// let policy = NoEviction;
/// let entry = EmbeddingEntry::new(1, vec![0.1, 0.2]);
///
/// // NoEviction never evicts
/// assert!(!policy.should_evict(&entry, 0));
/// assert!(!policy.should_evict(&entry, u64::MAX));
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct NoEviction;

impl EvictionPolicy for NoEviction {
    fn should_evict(&self, _entry: &EmbeddingEntry, _current_time: u64) -> bool {
        false
    }
}

/// Time-based eviction policy that removes entries older than a specified time.
///
/// This policy supports both a default expire time and per-slot expire times.
/// Per-slot configurations allow different expire times for different feature
/// categories.
///
/// # Example
///
/// ```
/// use monolith_hash_table::eviction::{EvictionPolicy, TimeBasedEviction};
/// use monolith_hash_table::EmbeddingEntry;
///
/// // Create policy with 1 hour default expire time
/// let mut policy = TimeBasedEviction::new(3600);
///
/// // Set 30 minutes expire time for slot 0
/// policy.set_slot_expire_time(0, 1800);
///
/// let mut entry = EmbeddingEntry::with_timestamp(1, vec![0.1, 0.2], 1000);
///
/// // At time 4000, entry (age = 3000) exceeds default expire (3600)? No
/// // At time 5000, entry (age = 4000) exceeds default expire (3600)? Yes
/// assert!(policy.should_evict(&entry, 5000));
/// ```
#[derive(Debug, Clone)]
pub struct TimeBasedEviction {
    /// Default expire time in seconds.
    default_expire_time: u64,
    /// Per-slot expire times.
    slot_expire_times: HashMap<SlotId, u64>,
}

impl TimeBasedEviction {
    /// Creates a new time-based eviction policy with the given default expire time.
    ///
    /// # Arguments
    ///
    /// * `default_expire_time` - The default time in seconds after which entries expire
    pub fn new(default_expire_time: u64) -> Self {
        Self {
            default_expire_time,
            slot_expire_times: HashMap::new(),
        }
    }

    /// Sets the expire time for a specific slot.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot identifier
    /// * `expire_time` - The expire time in seconds for this slot
    pub fn set_slot_expire_time(&mut self, slot_id: SlotId, expire_time: u64) {
        self.slot_expire_times.insert(slot_id, expire_time);
    }

    /// Gets the expire time for a specific slot, falling back to the default.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot identifier
    ///
    /// # Returns
    ///
    /// The expire time for the slot, or the default if not configured.
    pub fn get_expire_time(&self, slot_id: SlotId) -> u64 {
        self.slot_expire_times
            .get(&slot_id)
            .copied()
            .unwrap_or(self.default_expire_time)
    }

    /// Returns the default expire time.
    pub fn default_expire_time(&self) -> u64 {
        self.default_expire_time
    }

    /// Removes the per-slot expire time configuration.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot identifier to remove
    pub fn remove_slot_expire_time(&mut self, slot_id: SlotId) {
        self.slot_expire_times.remove(&slot_id);
    }

    /// Clears all per-slot expire time configurations.
    pub fn clear_slot_expire_times(&mut self) {
        self.slot_expire_times.clear();
    }
}

impl EvictionPolicy for TimeBasedEviction {
    fn should_evict(&self, entry: &EmbeddingEntry, current_time: u64) -> bool {
        let entry_time = entry.get_timestamp();

        // If current_time is less than entry_time, don't evict
        // (handles potential time skew or initialization with 0)
        if current_time < entry_time {
            return false;
        }

        let age = current_time - entry_time;
        age > self.default_expire_time
    }
}

/// LRU (Least Recently Used) eviction policy.
///
/// This policy evicts entries that haven't been accessed recently when
/// the table exceeds its capacity threshold. It uses the entry's timestamp
/// to determine how recently it was used.
///
/// # Example
///
/// ```
/// use monolith_hash_table::eviction::{EvictionPolicy, LRUEviction};
/// use monolith_hash_table::EmbeddingEntry;
///
/// // Create LRU policy with max age of 1 hour
/// let policy = LRUEviction::new(3600);
///
/// let old_entry = EmbeddingEntry::with_timestamp(1, vec![0.1], 1000);
/// let new_entry = EmbeddingEntry::with_timestamp(2, vec![0.2], 5000);
///
/// // At time 6000, old entry (age=5000) exceeds max age (3600)
/// assert!(policy.should_evict(&old_entry, 6000));
/// // New entry (age=1000) does not exceed max age
/// assert!(!policy.should_evict(&new_entry, 6000));
/// ```
#[derive(Debug, Clone)]
pub struct LRUEviction {
    /// Maximum age in seconds before an entry is considered for eviction.
    max_age: u64,
}

impl LRUEviction {
    /// Creates a new LRU eviction policy.
    ///
    /// # Arguments
    ///
    /// * `max_age` - Maximum age in seconds before an entry can be evicted
    pub fn new(max_age: u64) -> Self {
        Self { max_age }
    }

    /// Returns the maximum age configuration.
    pub fn max_age(&self) -> u64 {
        self.max_age
    }

    /// Sets the maximum age for eviction.
    pub fn set_max_age(&mut self, max_age: u64) {
        self.max_age = max_age;
    }
}

impl EvictionPolicy for LRUEviction {
    fn should_evict(&self, entry: &EmbeddingEntry, current_time: u64) -> bool {
        let entry_time = entry.get_timestamp();

        // If current_time is less than entry_time, don't evict
        if current_time < entry_time {
            return false;
        }

        let age = current_time - entry_time;
        age > self.max_age
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_eviction() {
        let policy = NoEviction;
        let entry = EmbeddingEntry::new(1, vec![0.1, 0.2]);

        assert!(!policy.should_evict(&entry, 0));
        assert!(!policy.should_evict(&entry, 1000));
        assert!(!policy.should_evict(&entry, u64::MAX));
    }

    #[test]
    fn test_time_based_eviction_basic() {
        let policy = TimeBasedEviction::new(3600); // 1 hour

        // Entry created at time 1000
        let entry = EmbeddingEntry::with_timestamp(1, vec![0.1, 0.2], 1000);

        // At time 2000, age is 1000 seconds - should NOT evict
        assert!(!policy.should_evict(&entry, 2000));

        // At time 4600, age is 3600 seconds - should NOT evict (not greater)
        assert!(!policy.should_evict(&entry, 4600));

        // At time 4601, age is 3601 seconds - should evict
        assert!(policy.should_evict(&entry, 4601));
    }

    #[test]
    fn test_time_based_eviction_handles_time_skew() {
        let policy = TimeBasedEviction::new(3600);

        // Entry created at time 5000
        let entry = EmbeddingEntry::with_timestamp(1, vec![0.1, 0.2], 5000);

        // Current time is before entry time (time skew) - should NOT evict
        assert!(!policy.should_evict(&entry, 1000));
    }

    #[test]
    fn test_time_based_eviction_slot_config() {
        let mut policy = TimeBasedEviction::new(3600);
        policy.set_slot_expire_time(0, 1800); // 30 minutes for slot 0
        policy.set_slot_expire_time(1, 7200); // 2 hours for slot 1

        assert_eq!(policy.get_expire_time(0), 1800);
        assert_eq!(policy.get_expire_time(1), 7200);
        assert_eq!(policy.get_expire_time(99), 3600); // Falls back to default

        policy.remove_slot_expire_time(0);
        assert_eq!(policy.get_expire_time(0), 3600); // Now uses default

        policy.clear_slot_expire_times();
        assert_eq!(policy.get_expire_time(1), 3600); // All cleared
    }

    #[test]
    fn test_lru_eviction_basic() {
        let policy = LRUEviction::new(3600);

        // Entry last accessed at time 1000
        let entry = EmbeddingEntry::with_timestamp(1, vec![0.1, 0.2], 1000);

        // At time 2000, age is 1000 - should NOT evict
        assert!(!policy.should_evict(&entry, 2000));

        // At time 4601, age is 3601 - should evict
        assert!(policy.should_evict(&entry, 4601));
    }

    #[test]
    fn test_lru_eviction_recently_accessed() {
        let policy = LRUEviction::new(3600);

        // Entry recently accessed at time 5000
        let entry = EmbeddingEntry::with_timestamp(1, vec![0.1, 0.2], 5000);

        // At time 5500, age is only 500 - should NOT evict
        assert!(!policy.should_evict(&entry, 5500));
    }

    #[test]
    fn test_lru_eviction_handles_time_skew() {
        let policy = LRUEviction::new(3600);

        // Entry accessed at time 5000
        let entry = EmbeddingEntry::with_timestamp(1, vec![0.1, 0.2], 5000);

        // Current time is before entry time - should NOT evict
        assert!(!policy.should_evict(&entry, 1000));
    }

    #[test]
    fn test_lru_eviction_config() {
        let mut policy = LRUEviction::new(3600);
        assert_eq!(policy.max_age(), 3600);

        policy.set_max_age(7200);
        assert_eq!(policy.max_age(), 7200);
    }

    #[test]
    fn test_eviction_with_zero_timestamp() {
        let time_policy = TimeBasedEviction::new(100);
        let lru_policy = LRUEviction::new(100);

        // Entry with default timestamp (0)
        let entry = EmbeddingEntry::new(1, vec![0.1, 0.2]);
        assert_eq!(entry.get_timestamp(), 0);

        // At time 50, should NOT evict
        assert!(!time_policy.should_evict(&entry, 50));
        assert!(!lru_policy.should_evict(&entry, 50));

        // At time 101, should evict
        assert!(time_policy.should_evict(&entry, 101));
        assert!(lru_policy.should_evict(&entry, 101));
    }
}
