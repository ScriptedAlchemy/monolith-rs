//! Interleaved dataset utilities for reading multiple TFRecord files.

use std::collections::VecDeque;
use std::path::PathBuf;

use monolith_proto::Example;

use crate::compression::CompressionType;
use crate::dataset::Dataset;
use crate::tfrecord::{TFRecordDataset, TFRecordIterator};

/// Dataset that interleaves records from multiple TFRecord files.
#[derive(Clone)]
pub struct InterleavedDataset {
    paths: Vec<PathBuf>,
    cycle_length: usize,
    verify_crc: bool,
    compression: Option<CompressionType>,
}

impl InterleavedDataset {
    /// Creates a new interleaved dataset from TFRecord paths.
    pub fn new(
        paths: Vec<PathBuf>,
        cycle_length: usize,
        verify_crc: bool,
        compression: Option<CompressionType>,
    ) -> Self {
        Self {
            paths,
            cycle_length: cycle_length.max(1),
            verify_crc,
            compression,
        }
    }

    /// Creates an interleaved dataset from an existing TFRecordDataset.
    pub fn from_tfrecord(dataset: TFRecordDataset, cycle_length: usize) -> Self {
        Self::new(
            dataset.paths().to_vec(),
            cycle_length,
            true,
            dataset.compression(),
        )
    }
}

impl Dataset for InterleavedDataset {
    type Iter = InterleavedIterator;

    fn iter(self) -> Self::Iter {
        InterleavedIterator::new(
            self.paths,
            self.cycle_length,
            self.verify_crc,
            self.compression,
        )
    }
}

/// Iterator that interleaves records across multiple TFRecord files.
pub struct InterleavedIterator {
    pending: VecDeque<TFRecordIterator>,
    active: Vec<TFRecordIterator>,
    cycle_length: usize,
    index: usize,
}

impl InterleavedIterator {
    fn new(
        paths: Vec<PathBuf>,
        cycle_length: usize,
        verify_crc: bool,
        compression: Option<CompressionType>,
    ) -> Self {
        let mut pending = VecDeque::new();
        for path in paths {
            pending.push_back(TFRecordIterator::new(vec![path], verify_crc, compression));
        }

        let mut active = Vec::new();
        for _ in 0..cycle_length.min(pending.len()) {
            if let Some(iter) = pending.pop_front() {
                active.push(iter);
            }
        }

        Self {
            pending,
            active,
            cycle_length,
            index: 0,
        }
    }

    fn refill_if_needed(&mut self) {
        while self.active.len() < self.cycle_length {
            if let Some(next) = self.pending.pop_front() {
                self.active.push(next);
            } else {
                break;
            }
        }
    }
}

impl Iterator for InterleavedIterator {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.active.is_empty() {
                return None;
            }
            if self.index >= self.active.len() {
                self.index = 0;
            }
            let idx = self.index;
            if let Some(item) = self.active[idx].next() {
                self.index = (self.index + 1) % self.active.len();
                return Some(item);
            } else {
                self.active.remove(idx);
                self.refill_if_needed();
            }
        }
    }
}
