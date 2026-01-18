//! Feature abstractions for Monolith.
//!
//! This module provides the core abstractions for working with features in Monolith,
//! including feature slots, slices, and columns. These abstractions are used throughout
//! the system for representing and manipulating feature data.
//!
//! # Overview
//!
//! - [`FeatureSlot`]: Represents a logical grouping of features with a shared embedding dimension.
//! - [`FeatureSlice`]: A view into a contiguous range of features within a slot.
//! - [`FeatureColumn`]: A trait for feature columns that can be sparse or dense.
//! - [`SparseFeatureColumn`]: A column of sparse features with indices and values.
//! - [`DenseFeatureColumn`]: A column of dense features with continuous values.

use serde::{Deserialize, Serialize};

use crate::error::{MonolithError, Result};
use crate::fid::{Fid, SlotId};

/// A feature slot represents a logical grouping of features.
///
/// Each slot has a unique ID and defines the embedding dimension for features
/// belonging to this slot. Slots are used to organize features into meaningful
/// categories (e.g., user features, item features, context features).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSlot {
    /// The unique identifier for this slot.
    slot_id: SlotId,

    /// The name of this slot (for debugging and logging).
    name: String,

    /// The embedding dimension for features in this slot.
    dim: usize,

    /// Whether this slot uses pooling for multiple features.
    pooling: bool,

    /// The maximum number of features allowed in this slot per example.
    max_features: Option<usize>,
}

impl FeatureSlot {
    /// Creates a new feature slot.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The unique identifier for this slot.
    /// * `name` - The name of this slot.
    /// * `dim` - The embedding dimension for features in this slot.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::feature::FeatureSlot;
    ///
    /// let slot = FeatureSlot::new(1, "user_id", 64);
    /// assert_eq!(slot.slot_id(), 1);
    /// assert_eq!(slot.dim(), 64);
    /// ```
    pub fn new(slot_id: SlotId, name: impl Into<String>, dim: usize) -> Self {
        Self {
            slot_id,
            name: name.into(),
            dim,
            pooling: false,
            max_features: None,
        }
    }

    /// Creates a new feature slot with pooling enabled.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The unique identifier for this slot.
    /// * `name` - The name of this slot.
    /// * `dim` - The embedding dimension for features in this slot.
    /// * `max_features` - The maximum number of features allowed per example.
    pub fn with_pooling(
        slot_id: SlotId,
        name: impl Into<String>,
        dim: usize,
        max_features: Option<usize>,
    ) -> Self {
        Self {
            slot_id,
            name: name.into(),
            dim,
            pooling: true,
            max_features,
        }
    }

    /// Returns the slot ID.
    #[inline]
    pub fn slot_id(&self) -> SlotId {
        self.slot_id
    }

    /// Returns the slot name.
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the embedding dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns whether pooling is enabled for this slot.
    #[inline]
    pub fn pooling(&self) -> bool {
        self.pooling
    }

    /// Returns the maximum number of features allowed per example.
    #[inline]
    pub fn max_features(&self) -> Option<usize> {
        self.max_features
    }

    /// Enables or disables pooling for this slot.
    pub fn set_pooling(&mut self, pooling: bool) {
        self.pooling = pooling;
    }

    /// Sets the maximum number of features allowed per example.
    pub fn set_max_features(&mut self, max_features: Option<usize>) {
        self.max_features = max_features;
    }
}

/// A slice of features within a feature column.
///
/// A feature slice represents a contiguous range of features, typically
/// used to represent the features for a single example within a batch.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureSlice {
    /// The starting offset in the feature column.
    offset: usize,

    /// The number of features in this slice.
    length: usize,

    /// The slot ID this slice belongs to.
    slot_id: SlotId,
}

impl FeatureSlice {
    /// Creates a new feature slice.
    ///
    /// # Arguments
    ///
    /// * `offset` - The starting offset in the feature column.
    /// * `length` - The number of features in this slice.
    /// * `slot_id` - The slot ID this slice belongs to.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::feature::FeatureSlice;
    ///
    /// let slice = FeatureSlice::new(0, 10, 1);
    /// assert_eq!(slice.offset(), 0);
    /// assert_eq!(slice.length(), 10);
    /// ```
    pub fn new(offset: usize, length: usize, slot_id: SlotId) -> Self {
        Self {
            offset,
            length,
            slot_id,
        }
    }

    /// Returns the starting offset.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the length of the slice.
    #[inline]
    pub fn length(&self) -> usize {
        self.length
    }

    /// Returns the slot ID.
    #[inline]
    pub fn slot_id(&self) -> SlotId {
        self.slot_id
    }

    /// Returns the end offset (exclusive).
    #[inline]
    pub fn end(&self) -> usize {
        self.offset + self.length
    }

    /// Returns whether this slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Validates that this slice is within bounds of the given total length.
    pub fn validate(&self, total: usize) -> Result<()> {
        if self.end() > total {
            return Err(MonolithError::SliceOutOfBounds {
                offset: self.offset,
                length: self.length,
                total,
            });
        }
        Ok(())
    }
}

/// A trait for feature columns.
///
/// Feature columns hold the feature data for a batch of examples. Different
/// column types (sparse, dense) implement this trait to provide a common
/// interface for feature access.
pub trait FeatureColumn {
    /// Returns the slot ID for this column.
    fn slot_id(&self) -> SlotId;

    /// Returns the number of examples in this column.
    fn batch_size(&self) -> usize;

    /// Returns the total number of features in this column.
    fn total_features(&self) -> usize;

    /// Returns whether this column is empty.
    fn is_empty(&self) -> bool {
        self.total_features() == 0
    }

    /// Clears all features from this column.
    fn clear(&mut self);
}

/// A sparse feature column.
///
/// Sparse feature columns store features as (fid, value) pairs, which is
/// efficient when most features are absent (zero). This is the most common
/// representation for categorical features.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseFeatureColumn {
    /// The slot ID for this column.
    slot_id: SlotId,

    /// The feature IDs.
    fids: Vec<Fid>,

    /// The feature values (same length as fids).
    values: Vec<f32>,

    /// Row offsets indicating where each example's features start.
    /// Length is batch_size + 1, where offsets[i+1] - offsets[i] gives
    /// the number of features for example i.
    offsets: Vec<usize>,
}

impl SparseFeatureColumn {
    /// Creates a new empty sparse feature column.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID for this column.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::feature::{SparseFeatureColumn, FeatureColumn};
    ///
    /// let column = SparseFeatureColumn::new(1);
    /// assert!(column.is_empty());
    /// ```
    pub fn new(slot_id: SlotId) -> Self {
        Self {
            slot_id,
            fids: Vec::new(),
            values: Vec::new(),
            offsets: vec![0],
        }
    }

    /// Creates a sparse feature column with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID for this column.
    /// * `batch_capacity` - The expected number of examples.
    /// * `feature_capacity` - The expected total number of features.
    pub fn with_capacity(slot_id: SlotId, batch_capacity: usize, feature_capacity: usize) -> Self {
        let mut offsets = Vec::with_capacity(batch_capacity + 1);
        offsets.push(0);

        Self {
            slot_id,
            fids: Vec::with_capacity(feature_capacity),
            values: Vec::with_capacity(feature_capacity),
            offsets,
        }
    }

    /// Adds features for a new example.
    ///
    /// # Arguments
    ///
    /// * `fids` - The feature IDs for this example.
    /// * `values` - The feature values for this example.
    ///
    /// # Panics
    ///
    /// Panics if `fids` and `values` have different lengths.
    pub fn push_example(&mut self, fids: &[Fid], values: &[f32]) {
        assert_eq!(
            fids.len(),
            values.len(),
            "fids and values must have the same length"
        );

        self.fids.extend_from_slice(fids);
        self.values.extend_from_slice(values);
        self.offsets.push(self.fids.len());
    }

    /// Returns all feature IDs.
    #[inline]
    pub fn fids(&self) -> &[Fid] {
        &self.fids
    }

    /// Returns all feature values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns the row offsets.
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Returns a slice for the given example index.
    pub fn slice(&self, example_idx: usize) -> Result<FeatureSlice> {
        if example_idx >= self.batch_size() {
            return Err(MonolithError::SliceOutOfBounds {
                offset: example_idx,
                length: 1,
                total: self.batch_size(),
            });
        }

        let offset = self.offsets[example_idx];
        let length = self.offsets[example_idx + 1] - offset;

        Ok(FeatureSlice::new(offset, length, self.slot_id))
    }

    /// Returns the features for the given example index.
    pub fn example_features(&self, example_idx: usize) -> Result<(&[Fid], &[f32])> {
        let slice = self.slice(example_idx)?;
        Ok((
            &self.fids[slice.offset()..slice.end()],
            &self.values[slice.offset()..slice.end()],
        ))
    }
}

impl FeatureColumn for SparseFeatureColumn {
    fn slot_id(&self) -> SlotId {
        self.slot_id
    }

    fn batch_size(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn total_features(&self) -> usize {
        self.fids.len()
    }

    fn clear(&mut self) {
        self.fids.clear();
        self.values.clear();
        self.offsets.clear();
        self.offsets.push(0);
    }
}

/// A dense feature column.
///
/// Dense feature columns store continuous feature values in a flat array.
/// Each example has the same number of features (the dimension).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseFeatureColumn {
    /// The slot ID for this column.
    slot_id: SlotId,

    /// The dimension (number of features per example).
    dim: usize,

    /// The feature values (length is batch_size * dim).
    values: Vec<f32>,
}

impl DenseFeatureColumn {
    /// Creates a new empty dense feature column.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID for this column.
    /// * `dim` - The dimension (number of features per example).
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::feature::{DenseFeatureColumn, FeatureColumn};
    ///
    /// let column = DenseFeatureColumn::new(1, 64);
    /// assert!(column.is_empty());
    /// assert_eq!(column.dim(), 64);
    /// ```
    pub fn new(slot_id: SlotId, dim: usize) -> Self {
        Self {
            slot_id,
            dim,
            values: Vec::new(),
        }
    }

    /// Creates a dense feature column with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID for this column.
    /// * `dim` - The dimension (number of features per example).
    /// * `batch_capacity` - The expected number of examples.
    pub fn with_capacity(slot_id: SlotId, dim: usize, batch_capacity: usize) -> Self {
        Self {
            slot_id,
            dim,
            values: Vec::with_capacity(batch_capacity * dim),
        }
    }

    /// Returns the dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns all feature values.
    #[inline]
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Returns mutable access to all feature values.
    #[inline]
    pub fn values_mut(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// Adds features for a new example.
    ///
    /// # Arguments
    ///
    /// * `values` - The feature values for this example (must have length == dim).
    ///
    /// # Returns
    ///
    /// Returns an error if the values length doesn't match the dimension.
    pub fn push_example(&mut self, values: &[f32]) -> Result<()> {
        if values.len() != self.dim {
            return Err(MonolithError::InvalidDimension {
                expected: self.dim,
                actual: values.len(),
            });
        }

        self.values.extend_from_slice(values);
        Ok(())
    }

    /// Returns the features for the given example index.
    pub fn example_features(&self, example_idx: usize) -> Result<&[f32]> {
        if example_idx >= self.batch_size() {
            return Err(MonolithError::SliceOutOfBounds {
                offset: example_idx,
                length: 1,
                total: self.batch_size(),
            });
        }

        let start = example_idx * self.dim;
        let end = start + self.dim;
        Ok(&self.values[start..end])
    }

    /// Returns mutable access to the features for the given example index.
    pub fn example_features_mut(&mut self, example_idx: usize) -> Result<&mut [f32]> {
        if example_idx >= self.batch_size() {
            return Err(MonolithError::SliceOutOfBounds {
                offset: example_idx,
                length: 1,
                total: self.batch_size(),
            });
        }

        let start = example_idx * self.dim;
        let end = start + self.dim;
        Ok(&mut self.values[start..end])
    }
}

impl FeatureColumn for DenseFeatureColumn {
    fn slot_id(&self) -> SlotId {
        self.slot_id
    }

    fn batch_size(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.values.len() / self.dim
        }
    }

    fn total_features(&self) -> usize {
        self.values.len()
    }

    fn clear(&mut self) {
        self.values.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_slot_new() {
        let slot = FeatureSlot::new(1, "user_id", 64);
        assert_eq!(slot.slot_id(), 1);
        assert_eq!(slot.name(), "user_id");
        assert_eq!(slot.dim(), 64);
        assert!(!slot.pooling());
        assert!(slot.max_features().is_none());
    }

    #[test]
    fn test_feature_slot_with_pooling() {
        let slot = FeatureSlot::with_pooling(2, "user_tags", 32, Some(100));
        assert_eq!(slot.slot_id(), 2);
        assert!(slot.pooling());
        assert_eq!(slot.max_features(), Some(100));
    }

    #[test]
    fn test_feature_slice() {
        let slice = FeatureSlice::new(10, 5, 1);
        assert_eq!(slice.offset(), 10);
        assert_eq!(slice.length(), 5);
        assert_eq!(slice.slot_id(), 1);
        assert_eq!(slice.end(), 15);
        assert!(!slice.is_empty());

        let empty_slice = FeatureSlice::new(0, 0, 1);
        assert!(empty_slice.is_empty());
    }

    #[test]
    fn test_feature_slice_validate() {
        let slice = FeatureSlice::new(5, 5, 1);
        assert!(slice.validate(10).is_ok());
        assert!(slice.validate(9).is_err());
    }

    #[test]
    fn test_sparse_feature_column_new() {
        let column = SparseFeatureColumn::new(1);
        assert_eq!(column.slot_id(), 1);
        assert!(column.is_empty());
        assert_eq!(column.batch_size(), 0);
        assert_eq!(column.total_features(), 0);
    }

    #[test]
    fn test_sparse_feature_column_push_example() {
        let mut column = SparseFeatureColumn::new(1);

        column.push_example(&[100, 200, 300], &[1.0, 2.0, 3.0]);
        column.push_example(&[400], &[4.0]);
        column.push_example(&[], &[]);

        assert_eq!(column.batch_size(), 3);
        assert_eq!(column.total_features(), 4);
        assert_eq!(column.fids(), &[100, 200, 300, 400]);
        assert_eq!(column.values(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(column.offsets(), &[0, 3, 4, 4]);
    }

    #[test]
    fn test_sparse_feature_column_slice() {
        let mut column = SparseFeatureColumn::new(1);
        column.push_example(&[100, 200], &[1.0, 2.0]);
        column.push_example(&[300], &[3.0]);

        let slice = column.slice(0).unwrap();
        assert_eq!(slice.offset(), 0);
        assert_eq!(slice.length(), 2);

        let slice = column.slice(1).unwrap();
        assert_eq!(slice.offset(), 2);
        assert_eq!(slice.length(), 1);

        assert!(column.slice(2).is_err());
    }

    #[test]
    fn test_sparse_feature_column_example_features() {
        let mut column = SparseFeatureColumn::new(1);
        column.push_example(&[100, 200], &[1.0, 2.0]);
        column.push_example(&[300], &[3.0]);

        let (fids, values) = column.example_features(0).unwrap();
        assert_eq!(fids, &[100, 200]);
        assert_eq!(values, &[1.0, 2.0]);

        let (fids, values) = column.example_features(1).unwrap();
        assert_eq!(fids, &[300]);
        assert_eq!(values, &[3.0]);
    }

    #[test]
    fn test_sparse_feature_column_clear() {
        let mut column = SparseFeatureColumn::new(1);
        column.push_example(&[100, 200], &[1.0, 2.0]);

        column.clear();
        assert!(column.is_empty());
        assert_eq!(column.batch_size(), 0);
        assert_eq!(column.offsets(), &[0]);
    }

    #[test]
    fn test_dense_feature_column_new() {
        let column = DenseFeatureColumn::new(1, 64);
        assert_eq!(column.slot_id(), 1);
        assert_eq!(column.dim(), 64);
        assert!(column.is_empty());
        assert_eq!(column.batch_size(), 0);
    }

    #[test]
    fn test_dense_feature_column_push_example() {
        let mut column = DenseFeatureColumn::new(1, 3);

        column.push_example(&[1.0, 2.0, 3.0]).unwrap();
        column.push_example(&[4.0, 5.0, 6.0]).unwrap();

        assert_eq!(column.batch_size(), 2);
        assert_eq!(column.total_features(), 6);
        assert_eq!(column.values(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_dense_feature_column_push_wrong_dim() {
        let mut column = DenseFeatureColumn::new(1, 3);
        let result = column.push_example(&[1.0, 2.0]);
        assert!(matches!(
            result,
            Err(MonolithError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn test_dense_feature_column_example_features() {
        let mut column = DenseFeatureColumn::new(1, 3);
        column.push_example(&[1.0, 2.0, 3.0]).unwrap();
        column.push_example(&[4.0, 5.0, 6.0]).unwrap();

        let features = column.example_features(0).unwrap();
        assert_eq!(features, &[1.0, 2.0, 3.0]);

        let features = column.example_features(1).unwrap();
        assert_eq!(features, &[4.0, 5.0, 6.0]);

        assert!(column.example_features(2).is_err());
    }

    #[test]
    fn test_dense_feature_column_clear() {
        let mut column = DenseFeatureColumn::new(1, 3);
        column.push_example(&[1.0, 2.0, 3.0]).unwrap();

        column.clear();
        assert!(column.is_empty());
        assert_eq!(column.batch_size(), 0);
    }
}
