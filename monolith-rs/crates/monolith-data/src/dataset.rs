//! Dataset trait and combinators for data pipelines.
//!
//! This module provides the core [`Dataset`] trait that all data sources implement,
//! along with various combinators for transforming, batching, and shuffling datasets.
//!
//! # Example
//!
//! ```
//! use monolith_data::{Dataset, VecDataset, create_example, add_feature};
//!
//! let examples: Vec<_> = (0..100).map(|_| create_example()).collect();
//! let dataset = VecDataset::new(examples)
//!     .shuffle(50)
//!     .map(|mut ex| {
//!         add_feature(&mut ex, "processed", vec![1], vec![1.0]);
//!         ex
//!     })
//!     .take(10);
//!
//! for example in dataset.iter() {
//!     // Process example
//! }
//! ```

use crate::batch::BatchedDataset;
use crate::transform::{Transform, TransformedDataset};
use monolith_proto::Example;
use std::collections::VecDeque;

/// A dataset that can produce examples.
///
/// The `Dataset` trait is the foundation of the data pipeline. It provides
/// methods for iterating over examples and chaining transformations.
///
/// # Type Parameters
///
/// Implementations should be able to produce [`Example`] protobuf messages
/// either directly or through transformations.
pub trait Dataset: Sized {
    /// The iterator type returned by this dataset.
    type Iter: Iterator<Item = Example>;

    /// Returns an iterator over examples in this dataset.
    fn iter(self) -> Self::Iter;

    /// Batches examples into groups of `batch_size`.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The number of examples per batch
    ///
    /// # Returns
    ///
    /// A [`BatchedDataset`] that yields batches of examples.
    fn batch(self, batch_size: usize) -> BatchedDataset<Self::Iter> {
        BatchedDataset::new(self.iter(), batch_size)
    }

    /// Shuffles examples using a buffer.
    ///
    /// This uses reservoir sampling to shuffle examples. A buffer of
    /// `buffer_size` examples is maintained, and examples are randomly
    /// selected from the buffer.
    ///
    /// # Arguments
    ///
    /// * `buffer_size` - The size of the shuffle buffer
    ///
    /// # Returns
    ///
    /// A [`ShuffledDataset`] that yields examples in random order.
    fn shuffle(self, buffer_size: usize) -> ShuffledDataset<Self::Iter> {
        ShuffledDataset::new(self.iter(), buffer_size)
    }

    /// Maps a function over examples.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply to each example
    ///
    /// # Type Parameters
    ///
    /// * `F` - The function type
    ///
    /// # Returns
    ///
    /// A [`MappedDataset`] that applies the function to each example.
    fn map<F>(self, f: F) -> MappedDataset<Self::Iter, F>
    where
        F: FnMut(Example) -> Example,
    {
        MappedDataset::new(self.iter(), f)
    }

    /// Filters examples based on a predicate.
    ///
    /// # Arguments
    ///
    /// * `predicate` - A function that returns `true` for examples to keep
    ///
    /// # Type Parameters
    ///
    /// * `F` - The predicate function type
    ///
    /// # Returns
    ///
    /// A [`FilteredDataset`] that only yields matching examples.
    fn filter<F>(self, predicate: F) -> FilteredDataset<Self::Iter, F>
    where
        F: FnMut(&Example) -> bool,
    {
        FilteredDataset::new(self.iter(), predicate)
    }

    /// Takes only the first `n` examples.
    ///
    /// # Arguments
    ///
    /// * `n` - The maximum number of examples to take
    ///
    /// # Returns
    ///
    /// A [`TakeDataset`] that yields at most `n` examples.
    fn take(self, n: usize) -> TakeDataset<Self::Iter> {
        TakeDataset::new(self.iter(), n)
    }

    /// Skips the first `n` examples.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of examples to skip
    ///
    /// # Returns
    ///
    /// A [`SkipDataset`] that skips the first `n` examples.
    fn skip(self, n: usize) -> SkipDataset<Self::Iter> {
        SkipDataset::new(self.iter(), n)
    }

    /// Applies a transform to examples.
    ///
    /// # Arguments
    ///
    /// * `transform` - The transform to apply
    ///
    /// # Type Parameters
    ///
    /// * `T` - The transform type
    ///
    /// # Returns
    ///
    /// A [`TransformedDataset`] that applies the transform.
    fn transform<T>(self, transform: T) -> TransformedDataset<Self::Iter, T>
    where
        T: Transform,
    {
        TransformedDataset::new(self.iter(), transform)
    }

    /// Repeats the dataset indefinitely.
    ///
    /// Note: This requires the iterator to be clonable.
    ///
    /// # Returns
    ///
    /// A dataset that repeats forever.
    fn repeat(self) -> RepeatDataset<Self>
    where
        Self: Clone,
    {
        RepeatDataset::new(self)
    }

    /// Collects all examples into a vector.
    ///
    /// # Returns
    ///
    /// A vector containing all examples from the dataset.
    fn collect_vec(self) -> Vec<Example> {
        self.iter().collect()
    }

    /// Counts the number of examples in the dataset.
    ///
    /// Note: This consumes the dataset.
    ///
    /// # Returns
    ///
    /// The total number of examples.
    fn count(self) -> usize {
        self.iter().count()
    }
}

/// A dataset backed by a vector of examples.
///
/// This is useful for testing and for datasets that fit in memory.
#[derive(Clone)]
pub struct VecDataset {
    examples: Vec<Example>,
}

impl VecDataset {
    /// Creates a new dataset from a vector of examples.
    pub fn new(examples: Vec<Example>) -> Self {
        Self { examples }
    }

    /// Creates an empty dataset.
    pub fn empty() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Returns the number of examples in the dataset.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns `true` if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

impl Dataset for VecDataset {
    type Iter = std::vec::IntoIter<Example>;

    fn iter(self) -> Self::Iter {
        self.examples.into_iter()
    }
}

impl FromIterator<Example> for VecDataset {
    fn from_iter<I: IntoIterator<Item = Example>>(iter: I) -> Self {
        Self {
            examples: iter.into_iter().collect(),
        }
    }
}

/// A dataset that shuffles examples using a buffer.
pub struct ShuffledDataset<I> {
    inner: I,
    buffer_size: usize,
}

impl<I> ShuffledDataset<I>
where
    I: Iterator<Item = Example>,
{
    /// Creates a new shuffled dataset.
    pub fn new(inner: I, buffer_size: usize) -> Self {
        Self { inner, buffer_size }
    }

    /// Returns an iterator over shuffled examples.
    pub fn iter(self) -> ShuffleIterator<I> {
        ShuffleIterator::new(self.inner, self.buffer_size)
    }
}

/// Iterator that shuffles examples using reservoir sampling.
pub struct ShuffleIterator<I> {
    inner: I,
    buffer: VecDeque<Example>,
    buffer_size: usize,
    rng_state: u64,
}

impl<I> ShuffleIterator<I>
where
    I: Iterator<Item = Example>,
{
    fn new(inner: I, buffer_size: usize) -> Self {
        Self {
            inner,
            buffer: VecDeque::with_capacity(buffer_size),
            buffer_size,
            // Simple seed based on memory address for basic randomness
            rng_state: 0x12345678_9abcdef0,
        }
    }

    // Simple xorshift64 PRNG
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }
}

impl<I> Iterator for ShuffleIterator<I>
where
    I: Iterator<Item = Example>,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        // Fill buffer if not full
        while self.buffer.len() < self.buffer_size {
            match self.inner.next() {
                Some(example) => self.buffer.push_back(example),
                None => break,
            }
        }

        if self.buffer.is_empty() {
            return None;
        }

        // Pick a random index and swap with front
        let idx = (self.next_random() as usize) % self.buffer.len();
        self.buffer.swap(0, idx);
        self.buffer.pop_front()
    }
}

impl<I> Dataset for ShuffledDataset<I>
where
    I: Iterator<Item = Example>,
{
    type Iter = ShuffleIterator<I>;

    fn iter(self) -> Self::Iter {
        ShuffleIterator::new(self.inner, self.buffer_size)
    }
}

/// A dataset that applies a mapping function to examples.
pub struct MappedDataset<I, F> {
    inner: I,
    f: F,
}

impl<I, F> MappedDataset<I, F>
where
    I: Iterator<Item = Example>,
    F: FnMut(Example) -> Example,
{
    /// Creates a new mapped dataset.
    pub fn new(inner: I, f: F) -> Self {
        Self { inner, f }
    }

    /// Returns an iterator over mapped examples.
    pub fn iter(self) -> MapIterator<I, F> {
        MapIterator {
            inner: self.inner,
            f: self.f,
        }
    }
}

/// Iterator that applies a mapping function.
pub struct MapIterator<I, F> {
    inner: I,
    f: F,
}

impl<I, F> Iterator for MapIterator<I, F>
where
    I: Iterator<Item = Example>,
    F: FnMut(Example) -> Example,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(&mut self.f)
    }
}

impl<I, F> Dataset for MappedDataset<I, F>
where
    I: Iterator<Item = Example>,
    F: FnMut(Example) -> Example,
{
    type Iter = MapIterator<I, F>;

    fn iter(self) -> Self::Iter {
        MapIterator {
            inner: self.inner,
            f: self.f,
        }
    }
}

/// A dataset that filters examples based on a predicate.
pub struct FilteredDataset<I, F> {
    inner: I,
    predicate: F,
}

impl<I, F> FilteredDataset<I, F>
where
    I: Iterator<Item = Example>,
    F: FnMut(&Example) -> bool,
{
    /// Creates a new filtered dataset.
    pub fn new(inner: I, predicate: F) -> Self {
        Self { inner, predicate }
    }

    /// Returns an iterator over filtered examples.
    pub fn iter(self) -> FilterIterator<I, F> {
        FilterIterator {
            inner: self.inner,
            predicate: self.predicate,
        }
    }
}

/// Iterator that filters examples.
pub struct FilterIterator<I, F> {
    inner: I,
    predicate: F,
}

impl<I, F> Iterator for FilterIterator<I, F>
where
    I: Iterator<Item = Example>,
    F: FnMut(&Example) -> bool,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let example = self.inner.next()?;
            if (self.predicate)(&example) {
                return Some(example);
            }
        }
    }
}

impl<I, F> Dataset for FilteredDataset<I, F>
where
    I: Iterator<Item = Example>,
    F: FnMut(&Example) -> bool,
{
    type Iter = FilterIterator<I, F>;

    fn iter(self) -> Self::Iter {
        FilterIterator {
            inner: self.inner,
            predicate: self.predicate,
        }
    }
}

/// A dataset that takes only the first N examples.
pub struct TakeDataset<I> {
    inner: I,
    remaining: usize,
}

impl<I> TakeDataset<I>
where
    I: Iterator<Item = Example>,
{
    /// Creates a new take dataset.
    pub fn new(inner: I, n: usize) -> Self {
        Self {
            inner,
            remaining: n,
        }
    }

    /// Returns an iterator over the first N examples.
    pub fn iter(self) -> TakeIterator<I> {
        TakeIterator {
            inner: self.inner,
            remaining: self.remaining,
        }
    }
}

/// Iterator that takes the first N examples.
pub struct TakeIterator<I> {
    inner: I,
    remaining: usize,
}

impl<I> Iterator for TakeIterator<I>
where
    I: Iterator<Item = Example>,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        self.inner.next()
    }
}

impl<I> Dataset for TakeDataset<I>
where
    I: Iterator<Item = Example>,
{
    type Iter = TakeIterator<I>;

    fn iter(self) -> Self::Iter {
        TakeIterator {
            inner: self.inner,
            remaining: self.remaining,
        }
    }
}

/// A dataset that skips the first N examples.
pub struct SkipDataset<I> {
    inner: I,
    to_skip: usize,
}

impl<I> SkipDataset<I>
where
    I: Iterator<Item = Example>,
{
    /// Creates a new skip dataset.
    pub fn new(inner: I, n: usize) -> Self {
        Self { inner, to_skip: n }
    }

    /// Returns an iterator that skips the first N examples.
    pub fn iter(self) -> SkipIterator<I> {
        SkipIterator {
            inner: self.inner,
            to_skip: self.to_skip,
        }
    }
}

/// Iterator that skips the first N examples.
pub struct SkipIterator<I> {
    inner: I,
    to_skip: usize,
}

impl<I> Iterator for SkipIterator<I>
where
    I: Iterator<Item = Example>,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        while self.to_skip > 0 {
            self.inner.next()?;
            self.to_skip -= 1;
        }
        self.inner.next()
    }
}

impl<I> Dataset for SkipDataset<I>
where
    I: Iterator<Item = Example>,
{
    type Iter = SkipIterator<I>;

    fn iter(self) -> Self::Iter {
        SkipIterator {
            inner: self.inner,
            to_skip: self.to_skip,
        }
    }
}

/// A dataset that repeats indefinitely.
pub struct RepeatDataset<D> {
    dataset: D,
}

impl<D> RepeatDataset<D>
where
    D: Dataset + Clone,
{
    /// Creates a new repeat dataset.
    pub fn new(dataset: D) -> Self {
        Self { dataset }
    }

    /// Returns an iterator that repeats forever.
    pub fn iter(self) -> RepeatIterator<D> {
        RepeatIterator {
            dataset: self.dataset.clone(),
            current: Some(self.dataset.iter()),
        }
    }
}

/// Iterator that repeats a dataset indefinitely.
pub struct RepeatIterator<D>
where
    D: Dataset + Clone,
{
    dataset: D,
    current: Option<D::Iter>,
}

impl<D> Iterator for RepeatIterator<D>
where
    D: Dataset + Clone,
{
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut iter) = self.current {
                if let Some(example) = iter.next() {
                    return Some(example);
                }
            }
            // Restart from the beginning
            self.current = Some(self.dataset.clone().iter());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::{add_feature, create_example, get_feature};
    use monolith_proto::monolith::io::proto::feature;

    fn make_examples(count: usize) -> Vec<Example> {
        (0..count)
            .map(|i| {
                let mut ex = create_example();
                add_feature(&mut ex, "id", vec![i as i64], vec![i as f32]);
                ex
            })
            .collect()
    }

    #[test]
    fn test_vec_dataset() {
        let examples = make_examples(5);
        let dataset = VecDataset::new(examples);
        assert_eq!(dataset.len(), 5);

        let collected: Vec<_> = dataset.iter().collect();
        assert_eq!(collected.len(), 5);
    }

    #[test]
    fn test_vec_dataset_empty() {
        let dataset = VecDataset::empty();
        assert!(dataset.is_empty());
        assert_eq!(dataset.count(), 0);
    }

    #[test]
    fn test_batch() {
        let examples = make_examples(10);
        let dataset = VecDataset::new(examples);
        let batches: Vec<_> = dataset.batch(3).iter().collect();

        assert_eq!(batches.len(), 4); // 3 + 3 + 3 + 1
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1);
    }

    #[test]
    fn test_shuffle() {
        let examples = make_examples(100);
        let dataset = VecDataset::new(examples.clone());
        let shuffled: Vec<_> = dataset.shuffle(50).iter().collect();

        assert_eq!(shuffled.len(), 100);
        // Check that we have all examples (by checking total is same)
        // Note: order may be different
    }

    #[test]
    fn test_map() {
        let examples = make_examples(5);
        let dataset = VecDataset::new(examples);
        let mapped: Vec<_> = dataset
            .map(|mut ex| {
                add_feature(&mut ex, "mapped", vec![1], vec![1.0]);
                ex
            })
            .iter()
            .collect();

        assert_eq!(mapped.len(), 5);
        for ex in &mapped {
            assert!(get_feature(ex, "mapped").is_some());
        }
    }

    #[test]
    fn test_filter() {
        let examples = make_examples(10);
        let dataset = VecDataset::new(examples);
        let filtered: Vec<_> = dataset
            .filter(|ex| {
                get_feature(ex, "id")
                    .and_then(|f| match &f.r#type {
                        Some(feature::Type::FidV2List(l)) => {
                            l.value.first().copied().map(|v| v as i64)
                        }
                        Some(feature::Type::FidV1List(l)) => {
                            l.value.first().copied().map(|v| v as i64)
                        }
                        _ => None,
                    })
                    .map(|v| v % 2 == 0)
                    .unwrap_or(false)
            })
            .iter()
            .collect();

        assert_eq!(filtered.len(), 5); // 0, 2, 4, 6, 8
    }

    #[test]
    fn test_take() {
        let examples = make_examples(10);
        let dataset = VecDataset::new(examples);
        let taken: Vec<_> = dataset.take(3).iter().collect();

        assert_eq!(taken.len(), 3);
    }

    #[test]
    fn test_skip() {
        let examples = make_examples(10);
        let dataset = VecDataset::new(examples);
        let skipped: Vec<_> = dataset.skip(7).iter().collect();

        assert_eq!(skipped.len(), 3);
    }

    #[test]
    fn test_repeat() {
        let examples = make_examples(3);
        let dataset = VecDataset::new(examples);
        let repeated: Vec<_> = dataset.repeat().iter().take(10).collect();

        assert_eq!(repeated.len(), 10);
    }

    #[test]
    fn test_chained_operations() {
        let examples = make_examples(100);
        let dataset = VecDataset::new(examples);

        let result: Vec<_> = dataset
            .filter(|ex| {
                get_feature(ex, "id")
                    .and_then(|f| match &f.r#type {
                        Some(feature::Type::FidV2List(l)) => {
                            l.value.first().copied().map(|v| v as i64)
                        }
                        Some(feature::Type::FidV1List(l)) => {
                            l.value.first().copied().map(|v| v as i64)
                        }
                        _ => None,
                    })
                    .map(|v| v < 50)
                    .unwrap_or(false)
            })
            .map(|mut ex| {
                add_feature(&mut ex, "processed", vec![1], vec![1.0]);
                ex
            })
            .take(10)
            .iter()
            .collect();

        assert_eq!(result.len(), 10);
        for ex in &result {
            assert!(get_feature(ex, "processed").is_some());
        }
    }

    #[test]
    fn test_collect_vec() {
        let examples = make_examples(5);
        let dataset = VecDataset::new(examples);
        let collected = dataset.collect_vec();
        assert_eq!(collected.len(), 5);
    }

    #[test]
    fn test_count() {
        let examples = make_examples(42);
        let dataset = VecDataset::new(examples);
        assert_eq!(dataset.count(), 42);
    }

    #[test]
    fn test_from_iterator() {
        let examples = make_examples(5);
        let dataset: VecDataset = examples.into_iter().collect();
        assert_eq!(dataset.len(), 5);
    }
}
