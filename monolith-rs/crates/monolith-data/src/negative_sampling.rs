//! Negative sampling for training data.
//!
//! This module provides negative sampling strategies commonly used in recommendation
//! systems and embedding learning. Negative sampling helps models learn to distinguish
//! between positive (observed) and negative (unobserved) examples.
//!
//! # Sampling Strategies
//!
//! - [`UniformNegativeSampler`]: Random uniform sampling from an item pool
//! - [`FrequencyNegativeSampler`]: Frequency-based sampling with temperature smoothing
//! - [`InBatchNegativeSampler`]: Uses other batch items as negatives
//!
//! # Example
//!
//! ```
//! use monolith_data::negative_sampling::{
//!     NegativeSampler, UniformNegativeSampler, NegativeSamplingConfig, SamplingStrategy,
//! };
//! use monolith_data::{create_example, add_feature};
//!
//! // Create a sampler with a pool of item IDs
//! let item_pool: Vec<i64> = (1000..2000).collect();
//! let sampler = UniformNegativeSampler::new(item_pool, true);
//!
//! // Create a positive example
//! let mut positive = create_example();
//! add_feature(&mut positive, "user_id", vec![123], vec![1.0]);
//! add_feature(&mut positive, "item_id", vec![1001], vec![1.0]);
//!
//! // Generate negative samples
//! let negatives = sampler.sample(&positive, 5);
//! assert_eq!(negatives.len(), 5);
//! ```

use monolith_core::Fid;
use monolith_proto::Example;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use crate::batch::Batch;
use crate::dataset::Dataset;
use crate::example::{add_feature, clone_example, get_feature, remove_feature};

/// Trait for negative sampling strategies.
///
/// Negative samplers generate negative examples from a positive example by
/// replacing certain features (typically item IDs) with randomly sampled
/// alternatives.
pub trait NegativeSampler: Send + Sync {
    /// Generates negative samples from a positive example.
    ///
    /// # Arguments
    ///
    /// * `positive` - The positive example to generate negatives from
    /// * `num_negatives` - The number of negative samples to generate
    ///
    /// # Returns
    ///
    /// A vector of negative examples derived from the positive example.
    fn sample(&self, positive: &Example, num_negatives: usize) -> Vec<Example>;
}

/// Random uniform negative sampler.
///
/// Samples items uniformly at random from a pool of candidate item IDs.
/// This is the simplest and most common negative sampling strategy.
pub struct UniformNegativeSampler {
    /// Pool of item IDs to sample from.
    item_pool: Vec<Fid>,
    /// Whether to sample with replacement.
    sample_with_replacement: bool,
    /// Name of the feature to replace with negative items.
    item_feature_name: String,
    /// Random state for sampling.
    rng_state: AtomicU64,
}

impl UniformNegativeSampler {
    /// Creates a new uniform negative sampler.
    ///
    /// # Arguments
    ///
    /// * `item_pool` - The pool of item IDs to sample from
    /// * `sample_with_replacement` - Whether to allow sampling the same item multiple times
    pub fn new(item_pool: Vec<Fid>, sample_with_replacement: bool) -> Self {
        Self {
            item_pool,
            sample_with_replacement,
            item_feature_name: "item_id".to_string(),
            rng_state: AtomicU64::new(0x12345678_9abcdef0),
        }
    }

    /// Sets the name of the feature to replace with negative items.
    ///
    /// # Arguments
    ///
    /// * `name` - The feature name (default: "item_id")
    pub fn with_item_feature_name(mut self, name: impl Into<String>) -> Self {
        self.item_feature_name = name.into();
        self
    }

    /// Sets the random seed for reproducible sampling.
    ///
    /// # Arguments
    ///
    /// * `seed` - The random seed
    pub fn with_seed(self, seed: u64) -> Self {
        self.rng_state.store(seed, Ordering::SeqCst);
        self
    }

    /// Generates a random number using xorshift64 with atomic compare-exchange.
    fn next_random(&self) -> u64 {
        loop {
            let current = self.rng_state.load(Ordering::SeqCst);
            let mut x = current;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if self
                .rng_state
                .compare_exchange(current, x, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return x;
            }
        }
    }

    /// Samples a random item from the pool.
    fn sample_item(&self) -> Option<Fid> {
        if self.item_pool.is_empty() {
            return None;
        }
        let idx = (self.next_random() as usize) % self.item_pool.len();
        Some(self.item_pool[idx])
    }
}

impl NegativeSampler for UniformNegativeSampler {
    fn sample(&self, positive: &Example, num_negatives: usize) -> Vec<Example> {
        if self.item_pool.is_empty() {
            return Vec::new();
        }

        // Get the positive item ID to exclude
        let positive_item = get_feature(positive, &self.item_feature_name)
            .and_then(|f| f.fid.first().copied());

        let mut negatives = Vec::with_capacity(num_negatives);
        let mut sampled_items = Vec::new();

        for _ in 0..num_negatives {
            // Try to find a valid negative item
            let mut attempts = 0;
            let max_attempts = 100;

            loop {
                if let Some(item) = self.sample_item() {
                    // Skip if it's the positive item
                    if Some(item) == positive_item {
                        attempts += 1;
                        if attempts >= max_attempts {
                            break;
                        }
                        continue;
                    }

                    // Skip if already sampled (when not sampling with replacement)
                    if !self.sample_with_replacement && sampled_items.contains(&item) {
                        attempts += 1;
                        if attempts >= max_attempts {
                            break;
                        }
                        continue;
                    }

                    // Create negative example
                    let mut negative = clone_example(positive);
                    // Remove existing item feature and add new one
                    remove_feature(&mut negative, &self.item_feature_name);
                    add_feature(&mut negative, &self.item_feature_name, vec![item], vec![1.0]);
                    // Remove existing label and add new one
                    remove_feature(&mut negative, "label");
                    add_feature(&mut negative, "label", vec![0], vec![0.0]);

                    negatives.push(negative);
                    sampled_items.push(item);
                    break;
                } else {
                    break;
                }
            }
        }

        negatives
    }
}

/// Frequency-based negative sampler.
///
/// Samples items based on their frequency in the training data, with temperature
/// smoothing to control the sampling distribution. Higher temperature makes the
/// distribution more uniform, while lower temperature emphasizes popular items.
pub struct FrequencyNegativeSampler {
    /// Item frequencies for weighted sampling.
    item_frequencies: HashMap<Fid, f32>,
    /// Temperature for smoothing the distribution (higher = more uniform).
    temperature: f32,
    /// Name of the feature to replace with negative items.
    item_feature_name: String,
    /// Precomputed cumulative distribution for efficient sampling.
    cumulative_dist: RwLock<Option<Vec<(Fid, f64)>>>,
    /// Random state for sampling.
    rng_state: AtomicU64,
}

impl FrequencyNegativeSampler {
    /// Creates a new frequency-based negative sampler.
    ///
    /// # Arguments
    ///
    /// * `item_frequencies` - Map from item IDs to their frequencies
    /// * `temperature` - Temperature for smoothing (default: 1.0)
    pub fn new(item_frequencies: HashMap<Fid, f32>, temperature: f32) -> Self {
        let sampler = Self {
            item_frequencies,
            temperature,
            item_feature_name: "item_id".to_string(),
            cumulative_dist: RwLock::new(None),
            rng_state: AtomicU64::new(0xfedcba98_76543210),
        };
        sampler.precompute_distribution();
        sampler
    }

    /// Sets the name of the feature to replace with negative items.
    pub fn with_item_feature_name(mut self, name: impl Into<String>) -> Self {
        self.item_feature_name = name.into();
        self
    }

    /// Sets the random seed for reproducible sampling.
    pub fn with_seed(self, seed: u64) -> Self {
        self.rng_state.store(seed, Ordering::SeqCst);
        self
    }

    /// Precomputes the cumulative distribution for efficient sampling.
    fn precompute_distribution(&self) {
        if self.item_frequencies.is_empty() {
            return;
        }

        // Apply temperature smoothing: p_i = freq_i^(1/T) / sum(freq_j^(1/T))
        let inv_temp = 1.0 / self.temperature.max(0.01);
        let smoothed: Vec<(Fid, f64)> = self
            .item_frequencies
            .iter()
            .map(|(&fid, &freq)| (fid, (freq as f64).powf(inv_temp as f64)))
            .collect();

        let total: f64 = smoothed.iter().map(|(_, s)| s).sum();
        if total == 0.0 {
            return;
        }

        // Build cumulative distribution
        let mut cumulative = Vec::with_capacity(smoothed.len());
        let mut cum = 0.0;
        for (fid, prob) in smoothed {
            cum += prob / total;
            cumulative.push((fid, cum));
        }

        *self.cumulative_dist.write().unwrap() = Some(cumulative);
    }

    /// Generates a random number using xorshift64 with atomic compare-exchange.
    fn next_random(&self) -> u64 {
        loop {
            let current = self.rng_state.load(Ordering::SeqCst);
            let mut x = current;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if self
                .rng_state
                .compare_exchange(current, x, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return x;
            }
        }
    }

    /// Samples a random item based on frequency distribution.
    fn sample_item(&self) -> Option<Fid> {
        let dist = self.cumulative_dist.read().unwrap();
        let cumulative = dist.as_ref()?;

        if cumulative.is_empty() {
            return None;
        }

        // Generate random value in [0, 1)
        let r = (self.next_random() as f64) / (u64::MAX as f64);

        // Binary search for the sampled item
        let idx = cumulative
            .binary_search_by(|(_, cum)| cum.partial_cmp(&r).unwrap())
            .unwrap_or_else(|i| i.min(cumulative.len() - 1));

        Some(cumulative[idx].0)
    }
}

impl NegativeSampler for FrequencyNegativeSampler {
    fn sample(&self, positive: &Example, num_negatives: usize) -> Vec<Example> {
        let positive_item = get_feature(positive, &self.item_feature_name)
            .and_then(|f| f.fid.first().copied());

        let mut negatives = Vec::with_capacity(num_negatives);

        for _ in 0..num_negatives {
            let mut attempts = 0;
            let max_attempts = 100;

            while attempts < max_attempts {
                if let Some(item) = self.sample_item() {
                    if Some(item) != positive_item {
                        let mut negative = clone_example(positive);
                        // Remove existing item feature and add new one
                        remove_feature(&mut negative, &self.item_feature_name);
                        add_feature(&mut negative, &self.item_feature_name, vec![item], vec![1.0]);
                        // Remove existing label and add new one
                        remove_feature(&mut negative, "label");
                        add_feature(&mut negative, "label", vec![0], vec![0.0]);
                        negatives.push(negative);
                        break;
                    }
                }
                attempts += 1;
            }
        }

        negatives
    }
}

/// In-batch negative sampler.
///
/// Uses other examples within the same batch as negative samples. This is
/// efficient because it reuses computation and naturally creates hard negatives
/// from examples that appear together.
pub struct InBatchNegativeSampler {
    /// Name of the feature to use for negative sampling.
    item_feature_name: String,
    /// Random state for shuffling.
    rng_state: AtomicU64,
}

impl InBatchNegativeSampler {
    /// Creates a new in-batch negative sampler.
    pub fn new() -> Self {
        Self {
            item_feature_name: "item_id".to_string(),
            rng_state: AtomicU64::new(0xabcdef12_34567890),
        }
    }

    /// Sets the name of the feature to use for negative sampling.
    pub fn with_item_feature_name(mut self, name: impl Into<String>) -> Self {
        self.item_feature_name = name.into();
        self
    }

    /// Sets the random seed for reproducible sampling.
    pub fn with_seed(self, seed: u64) -> Self {
        self.rng_state.store(seed, Ordering::SeqCst);
        self
    }

    /// Generates a random number using xorshift64 with atomic compare-exchange.
    fn next_random(&self) -> u64 {
        loop {
            let current = self.rng_state.load(Ordering::SeqCst);
            let mut x = current;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if self
                .rng_state
                .compare_exchange(current, x, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return x;
            }
        }
    }

    /// Samples negative examples from a batch for a given positive example.
    ///
    /// # Arguments
    ///
    /// * `batch` - The batch of examples to sample from
    /// * `positive_idx` - The index of the positive example in the batch
    /// * `num_negatives` - The number of negative samples to generate
    ///
    /// # Returns
    ///
    /// A vector of negative examples from other batch items.
    pub fn sample_from_batch(
        &self,
        batch: &Batch,
        positive_idx: usize,
        num_negatives: usize,
    ) -> Vec<Example> {
        let examples = batch.examples();
        if examples.len() <= 1 {
            return Vec::new();
        }

        let positive = &examples[positive_idx];
        let positive_item = get_feature(positive, &self.item_feature_name)
            .and_then(|f| f.fid.first().copied());

        // Collect indices of other examples
        let mut candidate_indices: Vec<usize> = (0..examples.len())
            .filter(|&i| i != positive_idx)
            .collect();

        // Shuffle candidates
        for i in (1..candidate_indices.len()).rev() {
            let j = (self.next_random() as usize) % (i + 1);
            candidate_indices.swap(i, j);
        }

        let mut negatives = Vec::with_capacity(num_negatives.min(candidate_indices.len()));

        for &idx in candidate_indices.iter().take(num_negatives) {
            let other = &examples[idx];
            let other_item = get_feature(other, &self.item_feature_name)
                .and_then(|f| f.fid.first().copied());

            // Skip if same item
            if other_item == positive_item {
                continue;
            }

            // Create negative by using the other example's item with current user
            let mut negative = clone_example(positive);
            if let Some(item) = other_item {
                // Remove existing item feature and add new one
                remove_feature(&mut negative, &self.item_feature_name);
                add_feature(&mut negative, &self.item_feature_name, vec![item], vec![1.0]);
            }
            // Remove existing label and add new one
            remove_feature(&mut negative, "label");
            add_feature(&mut negative, "label", vec![0], vec![0.0]);

            negatives.push(negative);
        }

        negatives
    }

    /// Generates negative samples for all examples in a batch.
    ///
    /// # Arguments
    ///
    /// * `batch` - The batch of positive examples
    /// * `num_negatives` - The number of negative samples per positive example
    ///
    /// # Returns
    ///
    /// A new batch containing both positive and negative examples.
    pub fn sample_batch(&self, batch: &Batch, num_negatives: usize) -> Batch {
        let mut all_examples = Vec::new();

        for (idx, positive) in batch.examples().iter().enumerate() {
            // Add the positive example with label 1
            let mut pos = clone_example(positive);
            // Remove existing label and add new one
            remove_feature(&mut pos, "label");
            add_feature(&mut pos, "label", vec![1], vec![1.0]);
            all_examples.push(pos);

            // Add negative samples
            let negatives = self.sample_from_batch(batch, idx, num_negatives);
            all_examples.extend(negatives);
        }

        Batch::new(all_examples)
    }
}

impl Default for InBatchNegativeSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl NegativeSampler for InBatchNegativeSampler {
    fn sample(&self, _positive: &Example, _num_negatives: usize) -> Vec<Example> {
        // In-batch sampling requires the full batch context
        // For individual examples, return empty
        Vec::new()
    }
}

/// Sampling strategy enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Uniform random sampling.
    Uniform,
    /// Frequency-based sampling.
    Frequency,
    /// In-batch negative sampling.
    InBatch,
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        Self::Uniform
    }
}

/// Configuration for negative sampling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NegativeSamplingConfig {
    /// Number of negative samples per positive example.
    pub num_negatives: usize,
    /// The sampling strategy to use.
    pub sampling_strategy: SamplingStrategy,
    /// Temperature for frequency-based sampling (higher = more uniform).
    pub temperature: f32,
    /// Whether to sample with replacement (for uniform sampling).
    pub sample_with_replacement: bool,
    /// Name of the item feature to replace.
    pub item_feature_name: String,
}

impl Default for NegativeSamplingConfig {
    fn default() -> Self {
        Self {
            num_negatives: 5,
            sampling_strategy: SamplingStrategy::Uniform,
            temperature: 1.0,
            sample_with_replacement: true,
            item_feature_name: "item_id".to_string(),
        }
    }
}

impl NegativeSamplingConfig {
    /// Creates a new configuration with the specified number of negatives.
    pub fn new(num_negatives: usize) -> Self {
        Self {
            num_negatives,
            ..Default::default()
        }
    }

    /// Sets the sampling strategy.
    pub fn with_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.sampling_strategy = strategy;
        self
    }

    /// Sets the temperature for frequency-based sampling.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Sets whether to sample with replacement.
    pub fn with_replacement(mut self, with_replacement: bool) -> Self {
        self.sample_with_replacement = with_replacement;
        self
    }

    /// Sets the item feature name.
    pub fn with_item_feature_name(mut self, name: impl Into<String>) -> Self {
        self.item_feature_name = name.into();
        self
    }
}

/// A dataset wrapper that adds negative samples to each example.
///
/// This wraps another dataset and augments each positive example with
/// negative samples generated by the configured sampler.
pub struct NegativeSamplingDataset<D>
where
    D: Dataset,
{
    /// The underlying dataset.
    inner: D,
    /// Number of negative samples per positive example.
    num_negatives: usize,
    /// The negative sampler to use.
    sampler: Box<dyn NegativeSampler>,
}

impl<D> NegativeSamplingDataset<D>
where
    D: Dataset,
{
    /// Creates a new negative sampling dataset.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying dataset of positive examples
    /// * `num_negatives` - The number of negative samples per positive
    /// * `sampler` - The negative sampler to use
    pub fn new(inner: D, num_negatives: usize, sampler: Box<dyn NegativeSampler>) -> Self {
        Self {
            inner,
            num_negatives,
            sampler,
        }
    }

    /// Returns an iterator over examples with negative samples.
    pub fn iter(self) -> NegativeSamplingIterator<D::Iter> {
        NegativeSamplingIterator {
            inner: self.inner.iter(),
            num_negatives: self.num_negatives,
            sampler: self.sampler,
            pending: Vec::new(),
        }
    }
}

/// Iterator that yields positive and negative examples.
pub struct NegativeSamplingIterator<I> {
    inner: I,
    num_negatives: usize,
    sampler: Box<dyn NegativeSampler>,
    pending: Vec<Example>,
}

impl<I> Iterator for NegativeSamplingIterator<I>
where
    I: Iterator<Item = Example>,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        // Return pending negatives first
        if let Some(example) = self.pending.pop() {
            return Some(example);
        }

        // Get next positive example
        let positive = self.inner.next()?;

        // Generate negative samples
        let negatives = self.sampler.sample(&positive, self.num_negatives);
        self.pending = negatives;

        // Return the positive example with label 1
        let mut pos = positive;
        // Remove existing label and add new one
        remove_feature(&mut pos, "label");
        add_feature(&mut pos, "label", vec![1], vec![1.0]);
        Some(pos)
    }
}

impl<D> Dataset for NegativeSamplingDataset<D>
where
    D: Dataset,
{
    type Iter = NegativeSamplingIterator<D::Iter>;

    fn iter(self) -> Self::Iter {
        NegativeSamplingIterator {
            inner: self.inner.iter(),
            num_negatives: self.num_negatives,
            sampler: self.sampler,
            pending: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::VecDataset;
    use crate::example::create_example;

    fn make_test_example(user_id: i64, item_id: i64) -> Example {
        let mut ex = create_example();
        add_feature(&mut ex, "user_id", vec![user_id], vec![1.0]);
        add_feature(&mut ex, "item_id", vec![item_id], vec![1.0]);
        ex
    }

    #[test]
    fn test_uniform_sampler_basic() {
        let item_pool: Vec<Fid> = (1000..1100).collect();
        let sampler = UniformNegativeSampler::new(item_pool, true).with_seed(42);

        let positive = make_test_example(1, 1050);
        let negatives = sampler.sample(&positive, 5);

        assert_eq!(negatives.len(), 5);
        for neg in &negatives {
            // Check that negatives have item_id feature
            let item_feature = get_feature(neg, "item_id").unwrap();
            assert!(!item_feature.fid.is_empty());
            // Check that negatives have label 0
            let label_feature = get_feature(neg, "label").unwrap();
            assert_eq!(label_feature.value[0], 0.0);
        }
    }

    #[test]
    fn test_uniform_sampler_without_replacement() {
        let item_pool: Vec<Fid> = (1000..1010).collect();
        let sampler = UniformNegativeSampler::new(item_pool, false).with_seed(42);

        let positive = make_test_example(1, 999); // Item not in pool
        let negatives = sampler.sample(&positive, 5);

        // All negative items should be unique
        let items: Vec<_> = negatives
            .iter()
            .filter_map(|ex| get_feature(ex, "item_id"))
            .filter_map(|f| f.fid.first().copied())
            .collect();

        let unique_count = {
            let mut sorted = items.clone();
            sorted.sort();
            sorted.dedup();
            sorted.len()
        };

        assert_eq!(items.len(), unique_count);
    }

    #[test]
    fn test_uniform_sampler_excludes_positive() {
        let item_pool: Vec<Fid> = vec![1000, 1001, 1002];
        let sampler = UniformNegativeSampler::new(item_pool, true).with_seed(12345);

        let positive = make_test_example(1, 1001);
        let negatives = sampler.sample(&positive, 100);

        // None of the negatives should have the positive item
        for neg in &negatives {
            let item = get_feature(neg, "item_id")
                .and_then(|f| f.fid.first().copied())
                .unwrap();
            assert_ne!(item, 1001);
        }
    }

    #[test]
    fn test_frequency_sampler_basic() {
        let mut frequencies = HashMap::new();
        frequencies.insert(1000, 100.0);
        frequencies.insert(1001, 50.0);
        frequencies.insert(1002, 25.0);
        frequencies.insert(1003, 10.0);

        let sampler = FrequencyNegativeSampler::new(frequencies, 1.0).with_seed(42);

        let positive = make_test_example(1, 999);
        let negatives = sampler.sample(&positive, 10);

        assert_eq!(negatives.len(), 10);
        for neg in &negatives {
            let label = get_feature(neg, "label").unwrap();
            assert_eq!(label.value[0], 0.0);
        }
    }

    #[test]
    fn test_frequency_sampler_temperature() {
        let mut frequencies = HashMap::new();
        frequencies.insert(1000, 100.0);
        frequencies.insert(1001, 1.0);

        // Low temperature should heavily favor the frequent item
        let sampler = FrequencyNegativeSampler::new(frequencies.clone(), 0.1).with_seed(42);

        let positive = make_test_example(1, 999);
        let negatives = sampler.sample(&positive, 100);

        let count_1000 = negatives
            .iter()
            .filter(|ex| {
                get_feature(ex, "item_id")
                    .and_then(|f| f.fid.first().copied())
                    == Some(1000)
            })
            .count();

        // With low temperature, most samples should be the frequent item
        assert!(count_1000 > 80);
    }

    #[test]
    fn test_in_batch_sampler_basic() {
        let sampler = InBatchNegativeSampler::new().with_seed(42);

        let examples = vec![
            make_test_example(1, 1000),
            make_test_example(2, 1001),
            make_test_example(3, 1002),
            make_test_example(4, 1003),
        ];
        let batch = Batch::new(examples);

        let negatives = sampler.sample_from_batch(&batch, 0, 2);

        // Should get negatives from other batch items
        assert!(!negatives.is_empty());
        assert!(negatives.len() <= 2);

        for neg in &negatives {
            let item = get_feature(neg, "item_id")
                .and_then(|f| f.fid.first().copied())
                .unwrap();
            // Should not be the positive item
            assert_ne!(item, 1000);
        }
    }

    #[test]
    fn test_in_batch_sampler_batch() {
        let sampler = InBatchNegativeSampler::new().with_seed(42);

        let examples = vec![
            make_test_example(1, 1000),
            make_test_example(2, 1001),
            make_test_example(3, 1002),
        ];
        let batch = Batch::new(examples);

        let augmented = sampler.sample_batch(&batch, 2);

        // Each positive gets up to 2 negatives
        // Total should be 3 positives + up to 6 negatives
        assert!(augmented.len() >= 3);
        assert!(augmented.len() <= 9);
    }

    #[test]
    fn test_sampling_config_default() {
        let config = NegativeSamplingConfig::default();
        assert_eq!(config.num_negatives, 5);
        assert_eq!(config.sampling_strategy, SamplingStrategy::Uniform);
        assert_eq!(config.temperature, 1.0);
        assert!(config.sample_with_replacement);
    }

    #[test]
    fn test_sampling_config_builder() {
        let config = NegativeSamplingConfig::new(10)
            .with_strategy(SamplingStrategy::Frequency)
            .with_temperature(0.5)
            .with_replacement(false)
            .with_item_feature_name("product_id");

        assert_eq!(config.num_negatives, 10);
        assert_eq!(config.sampling_strategy, SamplingStrategy::Frequency);
        assert_eq!(config.temperature, 0.5);
        assert!(!config.sample_with_replacement);
        assert_eq!(config.item_feature_name, "product_id");
    }

    #[test]
    fn test_negative_sampling_dataset() {
        let examples = vec![
            make_test_example(1, 1000),
            make_test_example(2, 1001),
            make_test_example(3, 1002),
        ];
        let dataset = VecDataset::new(examples);

        let item_pool: Vec<Fid> = (2000..2100).collect();
        let sampler = Box::new(UniformNegativeSampler::new(item_pool, true).with_seed(42));

        let neg_dataset = NegativeSamplingDataset::new(dataset, 2, sampler);
        let all_examples: Vec<_> = neg_dataset.iter().collect();

        // 3 positives + 3*2 negatives = 9 examples
        assert_eq!(all_examples.len(), 9);

        // Check that we have both positive and negative labels
        let positive_count = all_examples
            .iter()
            .filter(|ex| {
                get_feature(ex, "label")
                    .map(|f| f.value[0] == 1.0)
                    .unwrap_or(false)
            })
            .count();

        let negative_count = all_examples
            .iter()
            .filter(|ex| {
                get_feature(ex, "label")
                    .map(|f| f.value[0] == 0.0)
                    .unwrap_or(false)
            })
            .count();

        assert_eq!(positive_count, 3);
        assert_eq!(negative_count, 6);
    }

    #[test]
    fn test_uniform_sampler_empty_pool() {
        let sampler = UniformNegativeSampler::new(Vec::new(), true);

        let positive = make_test_example(1, 1000);
        let negatives = sampler.sample(&positive, 5);

        assert!(negatives.is_empty());
    }

    #[test]
    fn test_in_batch_sampler_single_example() {
        let sampler = InBatchNegativeSampler::new();

        let examples = vec![make_test_example(1, 1000)];
        let batch = Batch::new(examples);

        let negatives = sampler.sample_from_batch(&batch, 0, 5);

        // Can't create negatives from a single-example batch
        assert!(negatives.is_empty());
    }

    #[test]
    fn test_sampling_strategy_default() {
        assert_eq!(SamplingStrategy::default(), SamplingStrategy::Uniform);
    }

    #[test]
    fn test_custom_item_feature_name() {
        let item_pool: Vec<Fid> = (5000..5100).collect();
        let sampler = UniformNegativeSampler::new(item_pool, true)
            .with_item_feature_name("product_id")
            .with_seed(42);

        let mut positive = create_example();
        add_feature(&mut positive, "user_id", vec![1], vec![1.0]);
        add_feature(&mut positive, "product_id", vec![5050], vec![1.0]);

        let negatives = sampler.sample(&positive, 3);

        assert_eq!(negatives.len(), 3);
        for neg in &negatives {
            // Should have product_id, not item_id
            let product_feature = get_feature(neg, "product_id").unwrap();
            assert!(!product_feature.fid.is_empty());
        }
    }
}
