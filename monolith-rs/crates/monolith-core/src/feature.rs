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

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{MonolithError, Result};
use crate::fid::{Fid, SlotId};

// -----------------------------------------------------------------------------
// Python-parity (monolith/core/feature.py)
//
// The Rust crate historically had a simpler feature API aimed at batch feature
// storage (Sparse/DenseFeatureColumn). Python's core uses "Sail-like" objects:
// FeatureSlot -> FeatureSlice definitions + FeatureColumnV1 + Env merge/split
// logic used by tests.
//
// We keep the existing column structs for downstream crates and add a distinct
// set of types (SailFeature*) to avoid a breaking rename.
// -----------------------------------------------------------------------------

/// Sail-like FeatureSlice implementation.
///
/// Mirrors Python `FeatureSlice` in `monolith/core/feature.py`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SailFeatureSlice {
    slot_id: SlotId,
    dim: usize,
    slice_index: usize,
}

impl SailFeatureSlice {
    pub fn new(slot_id: SlotId, dim: usize, slice_index: usize) -> Self {
        Self {
            slot_id,
            dim,
            slice_index,
        }
    }

    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    #[inline]
    pub fn slice_index(&self) -> usize {
        self.slice_index
    }

    #[inline]
    pub fn slot_id(&self) -> SlotId {
        self.slot_id
    }
}

impl std::fmt::Display for SailFeatureSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Python __repr__ returns: [FeatureSlice][slot_{slot_id}][{slice_index}]
        write!(
            f,
            "[FeatureSlice][slot_{}][{}]",
            self.slot_id, self.slice_index
        )
    }
}

/// A minimal embedding representation used for merge/split tests.
///
/// Python tests use TensorFlow tensors; in Rust we keep a simple 2D matrix
/// (row-major) with only the operations we need.
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingTensor {
    data: Vec<Vec<i64>>, // i64 to match Python test integer constants.
}

impl EmbeddingTensor {
    pub fn from_row(row: Vec<i64>) -> Self {
        Self { data: vec![row] }
    }

    pub fn data(&self) -> &[Vec<i64>] {
        &self.data
    }

    fn num_cols(&self) -> usize {
        self.data.get(0).map(|r| r.len()).unwrap_or(0)
    }

    fn split_cols(&self, splits: &[usize]) -> Vec<EmbeddingTensor> {
        let total: usize = splits.iter().sum();
        assert_eq!(
            total,
            self.num_cols(),
            "split sizes must sum to embedding width"
        );
        let mut out = Vec::with_capacity(splits.len());
        let mut start = 0;
        for &w in splits {
            let end = start + w;
            let mut rows = Vec::with_capacity(self.data.len());
            for r in &self.data {
                rows.push(r[start..end].to_vec());
            }
            out.push(EmbeddingTensor { data: rows });
            start = end;
        }
        out
    }
}

/// Sail-like FeatureSlot implementation.
///
/// In Python this object is registered into `Env` and owns FeatureSlices.
#[derive(Debug, Clone)]
pub struct SailFeatureSlot {
    slot_id: SlotId,
    has_bias: bool,
    feature_slices: Vec<SailFeatureSlice>,
    merged_feature_slices: Vec<SailFeatureSlice>,
}

impl SailFeatureSlot {
    pub fn new(slot_id: SlotId, has_bias: bool) -> Self {
        let mut slot = Self {
            slot_id,
            has_bias,
            feature_slices: Vec::new(),
            merged_feature_slices: Vec::new(),
        };
        if has_bias {
            // Bias slice is always index 0 with dim 1.
            slot.feature_slices
                .push(SailFeatureSlice::new(slot_id, 1, 0));
        }
        slot
    }

    #[inline]
    pub fn slot_id(&self) -> SlotId {
        self.slot_id
    }

    #[inline]
    pub fn has_bias(&self) -> bool {
        self.has_bias
    }

    #[inline]
    pub fn feature_slices(&self) -> &[SailFeatureSlice] {
        &self.feature_slices
    }

    #[inline]
    pub fn merged_feature_slices(&self) -> &[SailFeatureSlice] {
        &self.merged_feature_slices
    }

    pub fn add_feature_slice(&mut self, dim: usize) -> SailFeatureSlice {
        let idx = self.feature_slices.len();
        let slice = SailFeatureSlice::new(self.slot_id, dim, idx);
        self.feature_slices.push(slice.clone());
        slice
    }

    /// Clears merged slices.
    ///
    /// Python stores merged slices in a separate list and can rebuild it.
    fn clear_merged_feature_slices(&mut self) {
        self.merged_feature_slices.clear();
    }

    fn push_merged_feature_slice(&mut self, slice: SailFeatureSlice) {
        self.merged_feature_slices.push(slice);
    }

    fn add_merged_feature_slice(&mut self, dim: usize) -> SailFeatureSlice {
        let idx = self.merged_feature_slices.len();
        let slice = SailFeatureSlice::new(self.slot_id, dim, idx);
        self.merged_feature_slices.push(slice.clone());
        slice
    }
}

/// Sail-like FeatureColumnV1 implementation.
#[derive(Debug, Clone)]
pub struct FeatureColumnV1 {
    slot_id: SlotId,
    fc_name: String,
    /// Placeholder markers for slices that have been looked up.
    feature_slice_to_placeholder: BTreeMap<SailFeatureSlice, ()>,
    /// Placeholder markers for merged slices after calling `merge_vector_in_same_slot`.
    merged_feature_slice_to_placeholder: BTreeMap<SailFeatureSlice, ()>,
}

impl FeatureColumnV1 {
    pub fn new(slot_id: SlotId, fc_name: impl Into<String>) -> Self {
        Self {
            slot_id,
            fc_name: fc_name.into(),
            feature_slice_to_placeholder: BTreeMap::new(),
            merged_feature_slice_to_placeholder: BTreeMap::new(),
        }
    }

    #[inline]
    pub fn fc_name(&self) -> &str {
        &self.fc_name
    }

    #[inline]
    pub fn slot_id(&self) -> SlotId {
        self.slot_id
    }

    #[inline]
    pub fn feature_slice_placeholders(&self) -> &BTreeMap<SailFeatureSlice, ()> {
        &self.feature_slice_to_placeholder
    }

    #[inline]
    pub fn merged_feature_slice_placeholders(&self) -> &BTreeMap<SailFeatureSlice, ()> {
        &self.merged_feature_slice_to_placeholder
    }
}

/// Env that owns sail-like FeatureSlots and FeatureColumns and implements
/// merge/split logic.
#[derive(Debug, Default, Clone)]
pub struct SailEnv {
    merge_vector: bool,
    slots: BTreeMap<SlotId, SailFeatureSlot>,
    columns_by_slot: BTreeMap<SlotId, Vec<FeatureColumnV1>>,
    pub tpu_features: BTreeMap<String, EmbeddingTensor>,
}

impl SailEnv {
    pub fn new(merge_vector: bool) -> Self {
        Self {
            merge_vector,
            ..Default::default()
        }
    }

    pub fn create_feature_slot(&mut self, slot_id: SlotId, has_bias: bool) -> &mut SailFeatureSlot {
        self.slots
            .entry(slot_id)
            .or_insert_with(|| SailFeatureSlot::new(slot_id, has_bias))
    }

    pub fn get_feature_slot(&self, slot_id: SlotId) -> Option<&SailFeatureSlot> {
        self.slots.get(&slot_id)
    }

    pub fn get_feature_slot_mut(&mut self, slot_id: SlotId) -> Option<&mut SailFeatureSlot> {
        self.slots.get_mut(&slot_id)
    }

    pub fn create_feature_column_v1(
        &mut self,
        slot_id: SlotId,
        fc_name: impl Into<String>,
    ) -> &mut FeatureColumnV1 {
        let col = FeatureColumnV1::new(slot_id, fc_name);
        self.columns_by_slot.entry(slot_id).or_default().push(col);
        let cols = self.columns_by_slot.get_mut(&slot_id).expect("inserted");
        cols.last_mut().expect("just pushed")
    }

    pub fn feature_columns(&self, slot_id: SlotId) -> Option<&[FeatureColumnV1]> {
        self.columns_by_slot.get(&slot_id).map(|v| v.as_slice())
    }

    pub fn embedding_lookup(&mut self, col: &mut FeatureColumnV1, slice: &SailFeatureSlice) {
        assert_eq!(col.slot_id(), slice.slot_id());
        // Record placeholder existence.
        col.feature_slice_to_placeholder.insert(slice.clone(), ());
    }

    pub fn merge_vector_in_same_slot(&mut self) {
        if !self.merge_vector {
            return;
        }
        let slot_ids: Vec<SlotId> = self.slots.keys().copied().collect();
        for slot_id in slot_ids {
            let has_bias = self
                .slots
                .get(&slot_id)
                .map(|s| s.has_bias())
                .unwrap_or(false);
            let feature_slices = self
                .slots
                .get(&slot_id)
                .map(|s| s.feature_slices.clone())
                .unwrap_or_default();

            let mut merged_slices = Vec::new();
            let mut merged_vector_dim = 0usize;

            for fs in &feature_slices {
                if has_bias && fs.slice_index() == 0 {
                    assert_eq!(
                        fs.dim(),
                        1,
                        "Bias in {} must have dim equal to 1, but actual dim is {}.",
                        slot_id,
                        fs.dim()
                    );
                    merged_slices.push(fs.clone());
                } else {
                    merged_vector_dim += fs.dim();
                }
            }

            // Update slot merged_feature_slices.
            let slot = self
                .slots
                .get_mut(&slot_id)
                .expect("slot exists while iterating keys");
            slot.clear_merged_feature_slices();
            for s in &merged_slices {
                slot.push_merged_feature_slice(s.clone());
            }

            if merged_vector_dim > 0 {
                let merged = slot.add_merged_feature_slice(merged_vector_dim);
                // Add placeholder mapping for each column.
                if let Some(cols) = self.columns_by_slot.get_mut(&slot_id) {
                    for col in cols {
                        // Bias slice: keep existing placeholder.
                        for b in &merged_slices {
                            col.merged_feature_slice_to_placeholder
                                .insert(b.clone(), ());
                        }
                        col.merged_feature_slice_to_placeholder
                            .insert(merged.clone(), ());
                    }
                }
            } else {
                // Only bias slice exists.
                if let Some(cols) = self.columns_by_slot.get_mut(&slot_id) {
                    for col in cols {
                        for b in &merged_slices {
                            col.merged_feature_slice_to_placeholder
                                .insert(b.clone(), ());
                        }
                    }
                }
            }
        }
    }

    pub fn split_merged_embedding(&mut self, slot_id: SlotId) {
        let slot = match self.slots.get(&slot_id) {
            Some(s) => s.clone(),
            None => return,
        };
        let has_bias = slot.has_bias();
        let feature_slices = slot.feature_slices.clone();

        let Some(cols) = self.columns_by_slot.get(&slot_id) else {
            return;
        };

        // For each column, find merged embedding tensor (index 0 for no bias, index 1 when bias exists).
        for col in cols {
            let mut merged_embedding: Option<EmbeddingTensor> = None;
            for merged_slice in &slot.merged_feature_slices {
                if merged_slice.slice_index() == 0 && has_bias {
                    assert_eq!(
                        merged_slice.dim(),
                        1,
                        "Bias in {} must have dim equal to 1, but actual dim is {}.",
                        col.fc_name(),
                        merged_slice.dim()
                    );
                } else {
                    let key = format!("{}_{}", col.fc_name(), merged_slice.slice_index());
                    merged_embedding = self.tpu_features.get(&key).cloned();
                }
            }

            if let Some(merged) = merged_embedding {
                let mut dim_splits: Vec<usize> = feature_slices.iter().map(|s| s.dim()).collect();
                if has_bias {
                    dim_splits = dim_splits.into_iter().skip(1).collect();
                }
                let splits = merged.split_cols(&dim_splits);
                // Write back split tensors into tpu_features.
                for fs in &feature_slices {
                    if fs.slice_index() == 0 && has_bias {
                        assert_eq!(
                            fs.dim(),
                            1,
                            "Bias in {} must have dim equal to 1, but actual dim is {}.",
                            col.fc_name(),
                            fs.dim()
                        );
                        continue;
                    }
                    let split_index = if has_bias {
                        fs.slice_index().saturating_sub(1)
                    } else {
                        fs.slice_index()
                    };
                    let split = splits
                        .get(split_index)
                        .cloned()
                        .expect("split index in bounds");
                    let key = format!("{}_{}", col.fc_name(), fs.slice_index());
                    self.tpu_features.insert(key, split);
                }
            }
        }
    }
}

#[cfg(test)]
mod python_parity_tests {
    use super::*;

    // Mirrors monolith/core/feature_test.py::FeatureSlotTest::test_has_bias
    #[test]
    fn test_feature_slot_has_bias() {
        let mut env = SailEnv::new(false);
        let slot = env.create_feature_slot(1, true);
        assert_eq!(slot.feature_slices().len(), 1);
        assert_eq!(slot.feature_slices()[0].dim(), 1);
        assert_eq!(slot.feature_slices()[0].slice_index(), 0);
    }

    // Mirrors monolith/core/feature_test.py::FeatureSlotTest::test_add_feature_slice
    #[test]
    fn test_feature_slot_add_feature_slice() {
        let mut env = SailEnv::new(false);
        let slot = env.create_feature_slot(1, true);
        slot.add_feature_slice(10);

        assert_eq!(slot.feature_slices().len(), 2);
        assert_eq!(slot.feature_slices()[0].dim(), 1);
        assert_eq!(slot.feature_slices()[0].slice_index(), 0);
        assert_eq!(slot.feature_slices()[1].dim(), 10);
        assert_eq!(slot.feature_slices()[1].slice_index(), 1);
    }

    // Mirrors monolith/core/feature_test.py::FeatureColumnV1Test::test_add_feature_column
    #[test]
    fn test_feature_column_add_feature_column() {
        let mut env = SailEnv::new(false);
        let slot = env.create_feature_slot(1, true);
        slot.add_feature_slice(10);
        env.create_feature_column_v1(1, "fc_name_1");

        let cols = env.feature_columns(1).unwrap();
        assert_eq!(cols.len(), 1);
    }

    // Mirrors monolith/core/feature_test.py::FeatureColumnV1Test::test_merge_split_vector_in_same_slot
    #[test]
    fn test_merge_split_vector_in_same_slot() {
        let mut env = SailEnv::new(true);

        // Test merge logic.
        let slot_1 = env.create_feature_slot(1, true);
        let slice_1_1 = slot_1.add_feature_slice(2);
        env.create_feature_slot(2, true);
        let slot_3 = env.create_feature_slot(3, false);
        let slice_3_0 = slot_3.add_feature_slice(2);
        let slice_3_1 = slot_3.add_feature_slice(3);
        let slot_4 = env.create_feature_slot(4, true);
        let slice_4_1 = slot_4.add_feature_slice(2);
        let slice_4_2 = slot_4.add_feature_slice(3);
        let slice_4_3 = slot_4.add_feature_slice(4);

        // Feature columns and embedding lookups.
        let fc_1: *mut FeatureColumnV1 = env.create_feature_column_v1(1, "fc_name_1");
        unsafe { env.embedding_lookup(&mut *fc_1, &slice_1_1) };
        env.create_feature_column_v1(2, "fc_name_2");
        let fc_3: *mut FeatureColumnV1 = env.create_feature_column_v1(3, "fc_name_3");
        unsafe {
            env.embedding_lookup(&mut *fc_3, &slice_3_0);
            env.embedding_lookup(&mut *fc_3, &slice_3_1);
        }
        let fc_4: *mut FeatureColumnV1 = env.create_feature_column_v1(4, "fc_name_4");
        unsafe {
            env.embedding_lookup(&mut *fc_4, &slice_4_1);
            env.embedding_lookup(&mut *fc_4, &slice_4_2);
            env.embedding_lookup(&mut *fc_4, &slice_4_3);
        }
        let fc_5: *mut FeatureColumnV1 = env.create_feature_column_v1(4, "fc_name_5");
        unsafe {
            env.embedding_lookup(&mut *fc_5, &slice_4_1);
            env.embedding_lookup(&mut *fc_5, &slice_4_2);
            env.embedding_lookup(&mut *fc_5, &slice_4_3);
        }

        env.merge_vector_in_same_slot();

        // Check the length of merged feature slices in FeatureSlot
        assert_eq!(
            env.get_feature_slot(1)
                .unwrap()
                .merged_feature_slices()
                .len(),
            2
        );
        assert_eq!(
            env.get_feature_slot(2)
                .unwrap()
                .merged_feature_slices()
                .len(),
            1
        );
        assert_eq!(
            env.get_feature_slot(3)
                .unwrap()
                .merged_feature_slices()
                .len(),
            1
        );
        assert_eq!(
            env.get_feature_slot(4)
                .unwrap()
                .merged_feature_slices()
                .len(),
            2
        );

        // Check the dim of each merged feature slice in FeatureSlot
        let fs1 = env.get_feature_slot(1).unwrap();
        assert_eq!(fs1.merged_feature_slices()[0].dim(), 1);
        assert_eq!(fs1.merged_feature_slices()[1].dim(), 2);
        let fs2 = env.get_feature_slot(2).unwrap();
        assert_eq!(fs2.merged_feature_slices()[0].dim(), 1);
        let fs3 = env.get_feature_slot(3).unwrap();
        assert_eq!(fs3.merged_feature_slices()[0].dim(), 5);
        let fs4 = env.get_feature_slot(4).unwrap();
        assert_eq!(fs4.merged_feature_slices()[0].dim(), 1);
        assert_eq!(fs4.merged_feature_slices()[1].dim(), 9);

        // Check merged slice placeholders exist for each column.
        let fc1 = &env.feature_columns(1).unwrap()[0];
        assert!(fc1
            .merged_feature_slice_placeholders()
            .contains_key(&fs1.merged_feature_slices()[0]));
        assert!(fc1
            .merged_feature_slice_placeholders()
            .contains_key(&fs1.merged_feature_slices()[1]));
        let fc2 = &env.feature_columns(2).unwrap()[0];
        assert!(fc2
            .merged_feature_slice_placeholders()
            .contains_key(&fs2.merged_feature_slices()[0]));
        let fc3 = &env.feature_columns(3).unwrap()[0];
        assert!(fc3
            .merged_feature_slice_placeholders()
            .contains_key(&fs3.merged_feature_slices()[0]));
        let fc4 = &env.feature_columns(4).unwrap()[0];
        let fc5 = &env.feature_columns(4).unwrap()[1];
        assert!(fc4
            .merged_feature_slice_placeholders()
            .contains_key(&fs4.merged_feature_slices()[0]));
        assert!(fc4
            .merged_feature_slice_placeholders()
            .contains_key(&fs4.merged_feature_slices()[1]));
        assert!(fc5
            .merged_feature_slice_placeholders()
            .contains_key(&fs4.merged_feature_slices()[0]));
        assert!(fc5
            .merged_feature_slice_placeholders()
            .contains_key(&fs4.merged_feature_slices()[1]));

        // Test split logic
        env.tpu_features.insert(
            "fc_name_1_0".to_string(),
            EmbeddingTensor::from_row(vec![1]),
        );
        env.tpu_features.insert(
            "fc_name_1_1".to_string(),
            EmbeddingTensor::from_row(vec![2, 3]),
        );
        env.tpu_features.insert(
            "fc_name_2_0".to_string(),
            EmbeddingTensor::from_row(vec![4]),
        );
        env.tpu_features.insert(
            "fc_name_3_0".to_string(),
            EmbeddingTensor::from_row(vec![7, 8, 9, 10, 11]),
        );
        env.tpu_features.insert(
            "fc_name_4_0".to_string(),
            EmbeddingTensor::from_row(vec![12]),
        );
        env.tpu_features.insert(
            "fc_name_4_1".to_string(),
            EmbeddingTensor::from_row(vec![13, 14, 15, 16, 17, 18, 19, 20, 21]),
        );
        env.tpu_features.insert(
            "fc_name_5_0".to_string(),
            EmbeddingTensor::from_row(vec![12]),
        );
        env.tpu_features.insert(
            "fc_name_5_1".to_string(),
            EmbeddingTensor::from_row(vec![13, 14, 15, 16, 17, 18, 19, 20, 21]),
        );

        env.split_merged_embedding(1);
        env.split_merged_embedding(2);
        env.split_merged_embedding(3);
        env.split_merged_embedding(4);

        assert_eq!(env.tpu_features["fc_name_1_0"].data(), &[vec![1]]);
        assert_eq!(env.tpu_features["fc_name_1_1"].data(), &[vec![2, 3]]);
        assert_eq!(env.tpu_features["fc_name_2_0"].data(), &[vec![4]]);
        assert_eq!(env.tpu_features["fc_name_3_0"].data(), &[vec![7, 8]]);
        assert_eq!(env.tpu_features["fc_name_3_1"].data(), &[vec![9, 10, 11]]);
        assert_eq!(env.tpu_features["fc_name_4_0"].data(), &[vec![12]]);
        assert_eq!(env.tpu_features["fc_name_4_1"].data(), &[vec![13, 14]]);
        assert_eq!(env.tpu_features["fc_name_4_2"].data(), &[vec![15, 16, 17]]);
        assert_eq!(
            env.tpu_features["fc_name_4_3"].data(),
            &[vec![18, 19, 20, 21]]
        );
        assert_eq!(env.tpu_features["fc_name_5_0"].data(), &[vec![12]]);
        assert_eq!(env.tpu_features["fc_name_5_1"].data(), &[vec![13, 14]]);
        assert_eq!(env.tpu_features["fc_name_5_2"].data(), &[vec![15, 16, 17]]);
        assert_eq!(
            env.tpu_features["fc_name_5_3"].data(),
            &[vec![18, 19, 20, 21]]
        );
    }
}

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
        slice
            .validate(10)
            .expect("slice fully within bounds should pass validation");
        slice
            .validate(9)
            .expect_err("slice extending beyond total length should fail validation");
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

        column
            .slice(2)
            .expect_err("requesting slice for out-of-range example index should fail");
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

        column
            .example_features(2)
            .expect_err("requesting dense example features out-of-range should fail");
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
