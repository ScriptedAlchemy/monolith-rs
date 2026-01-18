//! Batching support for datasets.
//!
//! This module provides the [`BatchedDataset`] wrapper that groups examples
//! into batches for efficient processing during training.
//!
//! # Example
//!
//! ```
//! use monolith_data::batch::{Batch, BatchedDataset};
//! use monolith_data::{Dataset, VecDataset, create_example};
//!
//! let examples: Vec<_> = (0..10).map(|_| create_example()).collect();
//! let dataset = VecDataset::new(examples);
//! let batched = dataset.batch(3);
//! for batch in batched.iter() {
//!     assert!(batch.len() <= 3);
//! }
//! ```

use monolith_proto::Example;

/// A batch of examples.
///
/// `Batch` is a simple container holding a vector of [`Example`]s that were
/// grouped together by a [`BatchedDataset`].
#[derive(Debug, Clone)]
pub struct Batch {
    /// The examples in this batch.
    examples: Vec<Example>,
}

impl Batch {
    /// Creates a new batch from a vector of examples.
    ///
    /// # Arguments
    ///
    /// * `examples` - The examples to include in the batch
    pub fn new(examples: Vec<Example>) -> Self {
        Self { examples }
    }

    /// Creates an empty batch.
    pub fn empty() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Returns the number of examples in this batch.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Returns `true` if this batch contains no examples.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Returns a reference to the examples in this batch.
    pub fn examples(&self) -> &[Example] {
        &self.examples
    }

    /// Returns a mutable reference to the examples in this batch.
    pub fn examples_mut(&mut self) -> &mut [Example] {
        &mut self.examples
    }

    /// Consumes the batch and returns the underlying vector of examples.
    pub fn into_examples(self) -> Vec<Example> {
        self.examples
    }

    /// Returns an iterator over the examples in this batch.
    pub fn iter(&self) -> impl Iterator<Item = &Example> {
        self.examples.iter()
    }

    /// Returns a mutable iterator over the examples in this batch.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Example> {
        self.examples.iter_mut()
    }
}

impl IntoIterator for Batch {
    type Item = Example;
    type IntoIter = std::vec::IntoIter<Example>;

    fn into_iter(self) -> Self::IntoIter {
        self.examples.into_iter()
    }
}

impl<'a> IntoIterator for &'a Batch {
    type Item = &'a Example;
    type IntoIter = std::slice::Iter<'a, Example>;

    fn into_iter(self) -> Self::IntoIter {
        self.examples.iter()
    }
}

impl FromIterator<Example> for Batch {
    fn from_iter<I: IntoIterator<Item = Example>>(iter: I) -> Self {
        Self {
            examples: iter.into_iter().collect(),
        }
    }
}

/// A dataset wrapper that groups examples into batches.
///
/// `BatchedDataset` wraps another dataset and yields batches of examples
/// instead of individual examples. This is useful for training where
/// processing multiple examples at once is more efficient.
///
/// # Type Parameters
///
/// * `I` - The underlying iterator type
pub struct BatchedDataset<I> {
    inner: I,
    batch_size: usize,
    drop_remainder: bool,
}

impl<I> BatchedDataset<I>
where
    I: Iterator<Item = Example>,
{
    /// Creates a new batched dataset.
    ///
    /// # Arguments
    ///
    /// * `inner` - The underlying iterator of examples
    /// * `batch_size` - The number of examples per batch
    pub fn new(inner: I, batch_size: usize) -> Self {
        Self {
            inner,
            batch_size,
            drop_remainder: false,
        }
    }

    /// Sets whether to drop the last batch if it's smaller than `batch_size`.
    ///
    /// By default, the last batch is included even if smaller.
    ///
    /// # Arguments
    ///
    /// * `drop` - If `true`, drop incomplete batches
    pub fn drop_remainder(mut self, drop: bool) -> Self {
        self.drop_remainder = drop;
        self
    }

    /// Returns an iterator over batches.
    pub fn iter(self) -> BatchIterator<I> {
        BatchIterator {
            inner: self.inner,
            batch_size: self.batch_size,
            drop_remainder: self.drop_remainder,
        }
    }
}

/// Iterator that yields batches of examples.
pub struct BatchIterator<I> {
    inner: I,
    batch_size: usize,
    drop_remainder: bool,
}

impl<I> Iterator for BatchIterator<I>
where
    I: Iterator<Item = Example>,
{
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut examples = Vec::with_capacity(self.batch_size);

        for _ in 0..self.batch_size {
            match self.inner.next() {
                Some(example) => examples.push(example),
                None => break,
            }
        }

        if examples.is_empty() {
            return None;
        }

        if self.drop_remainder && examples.len() < self.batch_size {
            return None;
        }

        Some(Batch::new(examples))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::create_example;

    fn make_examples(count: usize) -> Vec<Example> {
        (0..count).map(|_| create_example()).collect()
    }

    #[test]
    fn test_batch_new() {
        let examples = make_examples(5);
        let batch = Batch::new(examples.clone());
        assert_eq!(batch.len(), 5);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_batch_empty() {
        let batch = Batch::empty();
        assert_eq!(batch.len(), 0);
        assert!(batch.is_empty());
    }

    #[test]
    fn test_batch_into_examples() {
        let examples = make_examples(3);
        let batch = Batch::new(examples);
        let recovered = batch.into_examples();
        assert_eq!(recovered.len(), 3);
    }

    #[test]
    fn test_batch_iter() {
        let examples = make_examples(4);
        let batch = Batch::new(examples);
        let count = batch.iter().count();
        assert_eq!(count, 4);
    }

    #[test]
    fn test_batch_from_iterator() {
        let examples = make_examples(6);
        let batch: Batch = examples.into_iter().collect();
        assert_eq!(batch.len(), 6);
    }

    #[test]
    fn test_batched_dataset_exact_batches() {
        let examples = make_examples(10);
        let dataset = BatchedDataset::new(examples.into_iter(), 5);
        let batches: Vec<_> = dataset.iter().collect();

        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 5);
        assert_eq!(batches[1].len(), 5);
    }

    #[test]
    fn test_batched_dataset_incomplete_batch() {
        let examples = make_examples(7);
        let dataset = BatchedDataset::new(examples.into_iter(), 3);
        let batches: Vec<_> = dataset.iter().collect();

        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 1); // Incomplete batch
    }

    #[test]
    fn test_batched_dataset_drop_remainder() {
        let examples = make_examples(7);
        let dataset = BatchedDataset::new(examples.into_iter(), 3).drop_remainder(true);
        let batches: Vec<_> = dataset.iter().collect();

        assert_eq!(batches.len(), 2); // Last incomplete batch dropped
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
    }

    #[test]
    fn test_batched_dataset_empty() {
        let examples: Vec<Example> = vec![];
        let dataset = BatchedDataset::new(examples.into_iter(), 5);
        let batches: Vec<_> = dataset.iter().collect();

        assert!(batches.is_empty());
    }

    #[test]
    fn test_batched_dataset_single_example() {
        let examples = make_examples(1);
        let dataset = BatchedDataset::new(examples.into_iter(), 5);
        let batches: Vec<_> = dataset.iter().collect();

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_batched_dataset_single_example_drop_remainder() {
        let examples = make_examples(1);
        let dataset = BatchedDataset::new(examples.into_iter(), 5).drop_remainder(true);
        let batches: Vec<_> = dataset.iter().collect();

        assert!(batches.is_empty()); // Single example batch dropped
    }
}
