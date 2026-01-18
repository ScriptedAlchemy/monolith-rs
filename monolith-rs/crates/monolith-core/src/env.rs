//! Environment configuration for feature management.
//!
//! This module provides the central environment configuration for managing
//! feature slots, vocabulary sizes, and default parameters in Monolith.
//!
//! # Overview
//!
//! The [`Env`] struct serves as the central configuration hub for feature management,
//! providing:
//! - Mapping from slot IDs to feature slot configurations
//! - Vocabulary size management per slot
//! - Default embedding dimensions and thresholds
//! - FID (Feature ID) operations for slot extraction and creation
//!
//! # Example
//!
//! ```
//! use monolith_core::env::{Env, EnvBuilder};
//! use monolith_core::feature::FeatureSlot;
//!
//! // Build an environment using the builder pattern
//! let env = EnvBuilder::new()
//!     .with_slot(1, FeatureSlot::new(1, "user_id", 64))
//!     .with_slot(2, FeatureSlot::new(2, "item_id", 32))
//!     .with_vocab_size(1, 1_000_000)
//!     .with_vocab_size(2, 500_000)
//!     .with_default_dim(64)
//!     .build();
//!
//! assert_eq!(env.num_slots(), 2);
//! assert_eq!(env.get_vocab_size(1), Some(1_000_000));
//! ```

use std::collections::HashMap;

use crate::feature::FeatureSlot;
use crate::fid::{extract_slot, make_fid, Fid, SlotId};

/// Central environment for feature management.
///
/// The `Env` struct provides a unified configuration for managing feature slots,
/// vocabulary sizes, and default parameters throughout the Monolith system.
#[derive(Debug, Clone)]
pub struct Env {
    /// Mapping from slot ID to FeatureSlot configuration.
    slots: HashMap<SlotId, FeatureSlot>,

    /// Mapping from slot ID to vocab size.
    vocab_sizes: HashMap<SlotId, usize>,

    /// Default embedding dimension.
    default_dim: usize,

    /// Default occurrence threshold.
    default_occurrence_threshold: i64,

    /// Default expire time in days.
    default_expire_time: i64,
}

impl Default for Env {
    fn default() -> Self {
        Self {
            slots: HashMap::new(),
            vocab_sizes: HashMap::new(),
            default_dim: 64,
            default_occurrence_threshold: 1,
            default_expire_time: 7,
        }
    }
}

impl Env {
    /// Creates a new empty environment with default settings.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::Env;
    ///
    /// let env = Env::new();
    /// assert_eq!(env.num_slots(), 0);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns a reference to the feature slot for the given slot ID.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID to look up.
    ///
    /// # Returns
    ///
    /// `Some(&FeatureSlot)` if the slot exists, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    /// use monolith_core::feature::FeatureSlot;
    ///
    /// let env = EnvBuilder::new()
    ///     .with_slot(1, FeatureSlot::new(1, "user_id", 64))
    ///     .build();
    ///
    /// let slot = env.get_slot(1).unwrap();
    /// assert_eq!(slot.name(), "user_id");
    /// assert!(env.get_slot(999).is_none());
    /// ```
    #[inline]
    pub fn get_slot(&self, slot_id: SlotId) -> Option<&FeatureSlot> {
        self.slots.get(&slot_id)
    }

    /// Returns the vocabulary size for the given slot ID.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID to look up.
    ///
    /// # Returns
    ///
    /// `Some(usize)` if a vocabulary size is configured, `None` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    ///
    /// let env = EnvBuilder::new()
    ///     .with_vocab_size(1, 1_000_000)
    ///     .build();
    ///
    /// assert_eq!(env.get_vocab_size(1), Some(1_000_000));
    /// assert_eq!(env.get_vocab_size(2), None);
    /// ```
    #[inline]
    pub fn get_vocab_size(&self, slot_id: SlotId) -> Option<usize> {
        self.vocab_sizes.get(&slot_id).copied()
    }

    /// Registers a feature slot in the environment.
    ///
    /// If a slot with the same ID already exists, it will be replaced.
    ///
    /// # Arguments
    ///
    /// * `slot` - The feature slot to register.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::Env;
    /// use monolith_core::feature::FeatureSlot;
    ///
    /// let mut env = Env::new();
    /// env.register_slot(FeatureSlot::new(1, "user_id", 64));
    ///
    /// assert_eq!(env.num_slots(), 1);
    /// assert!(env.get_slot(1).is_some());
    /// ```
    pub fn register_slot(&mut self, slot: FeatureSlot) {
        self.slots.insert(slot.slot_id(), slot);
    }

    /// Returns an iterator over all slot IDs in the environment.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    /// use monolith_core::feature::FeatureSlot;
    ///
    /// let env = EnvBuilder::new()
    ///     .with_slot(1, FeatureSlot::new(1, "user_id", 64))
    ///     .with_slot(2, FeatureSlot::new(2, "item_id", 32))
    ///     .build();
    ///
    /// let slot_ids: Vec<_> = env.slot_ids().collect();
    /// assert_eq!(slot_ids.len(), 2);
    /// ```
    pub fn slot_ids(&self) -> impl Iterator<Item = SlotId> + '_ {
        self.slots.keys().copied()
    }

    /// Returns the number of registered slots.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    /// use monolith_core::feature::FeatureSlot;
    ///
    /// let env = EnvBuilder::new()
    ///     .with_slot(1, FeatureSlot::new(1, "user_id", 64))
    ///     .build();
    ///
    /// assert_eq!(env.num_slots(), 1);
    /// ```
    #[inline]
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    /// Returns a mutable reference to the slot for the given ID, creating it if necessary.
    ///
    /// If the slot doesn't exist, a new slot is created with the default dimension
    /// and an auto-generated name.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID to get or create.
    ///
    /// # Returns
    ///
    /// A mutable reference to the feature slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::Env;
    ///
    /// let mut env = Env::new();
    ///
    /// // Creates a new slot since it doesn't exist
    /// let slot = env.get_or_create_slot(1);
    /// assert_eq!(slot.slot_id(), 1);
    ///
    /// // Returns the existing slot
    /// let slot = env.get_or_create_slot(1);
    /// assert_eq!(slot.slot_id(), 1);
    /// assert_eq!(env.num_slots(), 1);
    /// ```
    pub fn get_or_create_slot(&mut self, slot_id: SlotId) -> &mut FeatureSlot {
        let default_dim = self.default_dim;
        self.slots.entry(slot_id).or_insert_with(|| {
            FeatureSlot::new(slot_id, format!("slot_{}", slot_id), default_dim)
        })
    }

    /// Extracts the slot ID from a feature ID.
    ///
    /// This is a convenience method that delegates to the `fid::extract_slot` function.
    ///
    /// # Arguments
    ///
    /// * `fid` - The feature ID to extract the slot from.
    ///
    /// # Returns
    ///
    /// The slot ID encoded in the feature ID.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::Env;
    /// use monolith_core::fid::make_fid;
    ///
    /// let env = Env::new();
    /// let fid = make_fid(42, 12345).unwrap();
    ///
    /// assert_eq!(env.extract_slot_from_fid(fid), 42);
    /// ```
    #[inline]
    pub fn extract_slot_from_fid(&self, fid: Fid) -> SlotId {
        extract_slot(fid)
    }

    /// Creates a feature ID from a slot ID and feature value.
    ///
    /// This is a convenience method that delegates to the `fid::make_fid` function.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID.
    /// * `feature` - The feature hash value.
    ///
    /// # Returns
    ///
    /// The combined feature ID, or an error if the inputs are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::Env;
    ///
    /// let env = Env::new();
    /// let fid = env.make_fid(42, 12345).unwrap();
    ///
    /// assert_eq!(env.extract_slot_from_fid(fid), 42);
    /// ```
    #[inline]
    pub fn make_fid(&self, slot_id: SlotId, feature: i64) -> crate::error::Result<Fid> {
        make_fid(slot_id, feature)
    }

    /// Returns the default embedding dimension.
    #[inline]
    pub fn default_dim(&self) -> usize {
        self.default_dim
    }

    /// Returns the default occurrence threshold.
    #[inline]
    pub fn default_occurrence_threshold(&self) -> i64 {
        self.default_occurrence_threshold
    }

    /// Returns the default expire time in days.
    #[inline]
    pub fn default_expire_time(&self) -> i64 {
        self.default_expire_time
    }
}

/// Builder for constructing an [`Env`] instance.
///
/// The builder provides a fluent interface for configuring the environment
/// with slots, vocabulary sizes, and default parameters.
///
/// # Example
///
/// ```
/// use monolith_core::env::EnvBuilder;
/// use monolith_core::feature::FeatureSlot;
///
/// let env = EnvBuilder::new()
///     .with_slot(1, FeatureSlot::new(1, "user_id", 64))
///     .with_vocab_size(1, 1_000_000)
///     .with_default_dim(128)
///     .build();
/// ```
#[derive(Debug, Default)]
pub struct EnvBuilder {
    slots: HashMap<SlotId, FeatureSlot>,
    vocab_sizes: HashMap<SlotId, usize>,
    default_dim: Option<usize>,
    default_occurrence_threshold: Option<i64>,
    default_expire_time: Option<i64>,
}

impl EnvBuilder {
    /// Creates a new environment builder.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    ///
    /// let builder = EnvBuilder::new();
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a feature slot to the environment.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID for this configuration.
    /// * `config` - The feature slot configuration.
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    /// use monolith_core::feature::FeatureSlot;
    ///
    /// let builder = EnvBuilder::new()
    ///     .with_slot(1, FeatureSlot::new(1, "user_id", 64));
    /// ```
    pub fn with_slot(mut self, slot_id: SlotId, config: FeatureSlot) -> Self {
        self.slots.insert(slot_id, config);
        self
    }

    /// Sets the vocabulary size for a slot.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID.
    /// * `size` - The vocabulary size.
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    ///
    /// let builder = EnvBuilder::new()
    ///     .with_vocab_size(1, 1_000_000);
    /// ```
    pub fn with_vocab_size(mut self, slot_id: SlotId, size: usize) -> Self {
        self.vocab_sizes.insert(slot_id, size);
        self
    }

    /// Sets the default embedding dimension.
    ///
    /// This dimension is used when creating new slots via `get_or_create_slot`.
    ///
    /// # Arguments
    ///
    /// * `dim` - The default embedding dimension.
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    ///
    /// let builder = EnvBuilder::new()
    ///     .with_default_dim(128);
    /// ```
    pub fn with_default_dim(mut self, dim: usize) -> Self {
        self.default_dim = Some(dim);
        self
    }

    /// Sets the default occurrence threshold.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The default occurrence threshold.
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    pub fn with_default_occurrence_threshold(mut self, threshold: i64) -> Self {
        self.default_occurrence_threshold = Some(threshold);
        self
    }

    /// Sets the default expire time in days.
    ///
    /// # Arguments
    ///
    /// * `days` - The default expire time in days.
    ///
    /// # Returns
    ///
    /// The builder for method chaining.
    pub fn with_default_expire_time(mut self, days: i64) -> Self {
        self.default_expire_time = Some(days);
        self
    }

    /// Builds the environment.
    ///
    /// # Returns
    ///
    /// The constructed [`Env`] instance.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::env::EnvBuilder;
    /// use monolith_core::feature::FeatureSlot;
    ///
    /// let env = EnvBuilder::new()
    ///     .with_slot(1, FeatureSlot::new(1, "user_id", 64))
    ///     .with_default_dim(64)
    ///     .build();
    ///
    /// assert_eq!(env.num_slots(), 1);
    /// ```
    pub fn build(self) -> Env {
        Env {
            slots: self.slots,
            vocab_sizes: self.vocab_sizes,
            default_dim: self.default_dim.unwrap_or(64),
            default_occurrence_threshold: self.default_occurrence_threshold.unwrap_or(1),
            default_expire_time: self.default_expire_time.unwrap_or(7),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_new() {
        let env = Env::new();
        assert_eq!(env.num_slots(), 0);
        assert_eq!(env.default_dim(), 64);
        assert_eq!(env.default_occurrence_threshold(), 1);
        assert_eq!(env.default_expire_time(), 7);
    }

    #[test]
    fn test_env_register_slot() {
        let mut env = Env::new();
        let slot = FeatureSlot::new(1, "user_id", 64);

        env.register_slot(slot);

        assert_eq!(env.num_slots(), 1);
        let retrieved = env.get_slot(1).unwrap();
        assert_eq!(retrieved.name(), "user_id");
        assert_eq!(retrieved.dim(), 64);
    }

    #[test]
    fn test_env_register_slot_replaces_existing() {
        let mut env = Env::new();

        env.register_slot(FeatureSlot::new(1, "old_name", 32));
        env.register_slot(FeatureSlot::new(1, "new_name", 64));

        assert_eq!(env.num_slots(), 1);
        let slot = env.get_slot(1).unwrap();
        assert_eq!(slot.name(), "new_name");
        assert_eq!(slot.dim(), 64);
    }

    #[test]
    fn test_env_get_slot_not_found() {
        let env = Env::new();
        assert!(env.get_slot(999).is_none());
    }

    #[test]
    fn test_env_vocab_size() {
        let env = EnvBuilder::new()
            .with_vocab_size(1, 1_000_000)
            .with_vocab_size(2, 500_000)
            .build();

        assert_eq!(env.get_vocab_size(1), Some(1_000_000));
        assert_eq!(env.get_vocab_size(2), Some(500_000));
        assert_eq!(env.get_vocab_size(3), None);
    }

    #[test]
    fn test_env_slot_ids() {
        let env = EnvBuilder::new()
            .with_slot(1, FeatureSlot::new(1, "a", 32))
            .with_slot(5, FeatureSlot::new(5, "b", 32))
            .with_slot(3, FeatureSlot::new(3, "c", 32))
            .build();

        let mut slot_ids: Vec<_> = env.slot_ids().collect();
        slot_ids.sort();

        assert_eq!(slot_ids, vec![1, 3, 5]);
    }

    #[test]
    fn test_env_get_or_create_slot_new() {
        let mut env = Env::new();

        let slot = env.get_or_create_slot(1);
        assert_eq!(slot.slot_id(), 1);
        assert_eq!(slot.name(), "slot_1");
        assert_eq!(slot.dim(), 64); // default dim

        assert_eq!(env.num_slots(), 1);
    }

    #[test]
    fn test_env_get_or_create_slot_existing() {
        let mut env = Env::new();
        env.register_slot(FeatureSlot::new(1, "custom_name", 128));

        let slot = env.get_or_create_slot(1);
        assert_eq!(slot.slot_id(), 1);
        assert_eq!(slot.name(), "custom_name");
        assert_eq!(slot.dim(), 128);

        assert_eq!(env.num_slots(), 1);
    }

    #[test]
    fn test_env_get_or_create_slot_uses_default_dim() {
        let mut env = EnvBuilder::new().with_default_dim(256).build();

        let slot = env.get_or_create_slot(1);
        assert_eq!(slot.dim(), 256);
    }

    #[test]
    fn test_env_extract_slot_from_fid() {
        let env = Env::new();
        let fid = make_fid(42, 12345).unwrap();

        assert_eq!(env.extract_slot_from_fid(fid), 42);
    }

    #[test]
    fn test_env_make_fid() {
        let env = Env::new();
        let fid = env.make_fid(42, 12345).unwrap();

        assert_eq!(env.extract_slot_from_fid(fid), 42);
    }

    #[test]
    fn test_env_make_fid_invalid() {
        let env = Env::new();

        assert!(env.make_fid(-1, 0).is_err());
        assert!(env.make_fid(0, -1).is_err());
    }

    #[test]
    fn test_env_builder_new() {
        let builder = EnvBuilder::new();
        let env = builder.build();

        assert_eq!(env.num_slots(), 0);
        assert_eq!(env.default_dim(), 64);
    }

    #[test]
    fn test_env_builder_with_slot() {
        let env = EnvBuilder::new()
            .with_slot(1, FeatureSlot::new(1, "user_id", 64))
            .with_slot(2, FeatureSlot::new(2, "item_id", 32))
            .build();

        assert_eq!(env.num_slots(), 2);
        assert_eq!(env.get_slot(1).unwrap().name(), "user_id");
        assert_eq!(env.get_slot(2).unwrap().name(), "item_id");
    }

    #[test]
    fn test_env_builder_with_vocab_size() {
        let env = EnvBuilder::new()
            .with_vocab_size(1, 1_000_000)
            .build();

        assert_eq!(env.get_vocab_size(1), Some(1_000_000));
    }

    #[test]
    fn test_env_builder_with_default_dim() {
        let env = EnvBuilder::new()
            .with_default_dim(128)
            .build();

        assert_eq!(env.default_dim(), 128);
    }

    #[test]
    fn test_env_builder_with_default_occurrence_threshold() {
        let env = EnvBuilder::new()
            .with_default_occurrence_threshold(10)
            .build();

        assert_eq!(env.default_occurrence_threshold(), 10);
    }

    #[test]
    fn test_env_builder_with_default_expire_time() {
        let env = EnvBuilder::new()
            .with_default_expire_time(30)
            .build();

        assert_eq!(env.default_expire_time(), 30);
    }

    #[test]
    fn test_env_builder_full() {
        let env = EnvBuilder::new()
            .with_slot(1, FeatureSlot::new(1, "user_id", 64))
            .with_slot(2, FeatureSlot::new(2, "item_id", 32))
            .with_vocab_size(1, 1_000_000)
            .with_vocab_size(2, 500_000)
            .with_default_dim(64)
            .with_default_occurrence_threshold(5)
            .with_default_expire_time(14)
            .build();

        assert_eq!(env.num_slots(), 2);
        assert_eq!(env.get_vocab_size(1), Some(1_000_000));
        assert_eq!(env.get_vocab_size(2), Some(500_000));
        assert_eq!(env.default_dim(), 64);
        assert_eq!(env.default_occurrence_threshold(), 5);
        assert_eq!(env.default_expire_time(), 14);
    }

    #[test]
    fn test_env_clone() {
        let env = EnvBuilder::new()
            .with_slot(1, FeatureSlot::new(1, "user_id", 64))
            .with_vocab_size(1, 1_000_000)
            .build();

        let cloned = env.clone();

        assert_eq!(cloned.num_slots(), 1);
        assert_eq!(cloned.get_slot(1).unwrap().name(), "user_id");
        assert_eq!(cloned.get_vocab_size(1), Some(1_000_000));
    }

    #[test]
    fn test_env_default() {
        let env = Env::default();

        assert_eq!(env.num_slots(), 0);
        assert_eq!(env.default_dim(), 64);
        assert_eq!(env.default_occurrence_threshold(), 1);
        assert_eq!(env.default_expire_time(), 7);
    }
}
