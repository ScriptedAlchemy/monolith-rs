//! Data pipeline, parsing, and datasets for Monolith.
//!
//! This crate provides the data loading and transformation infrastructure for
//! the Monolith recommendation system. It includes support for various data
//! formats (TFRecord, Parquet, Kafka) and a flexible pipeline API for
//! preprocessing training examples.
//!
//! # Overview
//!
//! The core abstraction is the [`Dataset`] trait, which provides a uniform
//! interface for data sources and transformations. Datasets can be chained
//! together to form pipelines:
//!
//! ```
//! use monolith_data::{Dataset, VecDataset, create_example, add_feature, has_feature};
//!
//! let examples: Vec<_> = (0..100).map(|_| create_example()).collect();
//! let pipeline = VecDataset::new(examples)
//!     .shuffle(50)
//!     .map(|mut ex| {
//!         add_feature(&mut ex, "processed", vec![1], vec![1.0]);
//!         ex
//!     })
//!     .filter(|ex| has_feature(ex, "processed"))
//!     .batch(32);
//!
//! for batch in pipeline.iter() {
//!     assert!(batch.len() <= 32);
//! }
//! ```
//!
//! # Modules
//!
//! - [`dataset`] - The core [`Dataset`] trait and combinators
//! - [`tfrecord`] - TFRecord file format reader/writer
//! - [`parquet`] - Parquet file format reader (requires `parquet` feature)
//! - [`compression`] - Compression support for TFRecord files
//! - [`batch`] - Batching utilities
//! - [`transform`] - Transform traits and implementations
//! - [`example`] - Utilities for working with Example protos
//! - [`instance`] - Instance format parsing and manipulation
//! - [`negative_sampling`] - Negative sampling strategies for training
//!
//! # Features
//!
//! - `parquet` - Enable Parquet file support
//! - `kafka` - Enable Kafka streaming support
//! - `snappy` - Enable Snappy compression for TFRecord files
//! - `gzip` - Enable Gzip/Zlib compression for TFRecord files
//! - `compression` - Enable all compression formats
//!
//! # Example: Reading TFRecords
//!
//! ```no_run
//! use monolith_data::tfrecord::TFRecordDataset;
//!
//! // This example requires a TFRecord file to exist
//! let dataset = TFRecordDataset::open("data/train.tfrecord").unwrap();
//! for example in dataset.iter().take(100) {
//!     println!("Example: {:?}", example);
//! }
//! ```
//!
//! # Example: Building a Transform Pipeline
//!
//! ```
//! use monolith_data::transform::{Transform, TransformChain, FilterTransform, MapTransform};
//! use monolith_data::{has_feature, add_feature, create_example};
//!
//! let pipeline = TransformChain::new()
//!     .add(FilterTransform::new(|ex| has_feature(ex, "user_id")))
//!     .add(MapTransform::new(|mut ex| {
//!         add_feature(&mut ex, "timestamp", vec![12345], vec![1.0]);
//!         ex
//!     }));
//!
//! // Example without user_id gets filtered out
//! let example = create_example();
//! assert!(pipeline.apply(example).is_none());
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod batch;
pub mod compression;
pub mod dataset;
pub mod example;
pub mod instance;
pub mod interleave;
pub mod kafka;
pub mod negative_sampling;
#[cfg(feature = "parquet")]
pub mod parquet;
pub mod tfrecord;
pub mod transform;
pub mod utils;
pub mod feature_list;

// Re-export main types for convenience
pub use batch::{Batch, BatchedDataset};
pub use compression::{compress, decompress, CompressionError, CompressionType};
pub use dataset::{
    Dataset, FilteredDataset, MappedDataset, RepeatDataset, ShuffledDataset, SkipDataset,
    TakeDataset, VecDataset,
};
pub use example::{
    add_feature, clone_example, create_example, create_example_with_line_id, feature_count,
    feature_names, get_feature, get_feature_data, get_feature_mut, has_feature, merge_examples,
    remove_feature, total_fid_count,
};
pub use instance::{
    extract_feature, extract_slot, make_fid, DenseFeature, Instance, InstanceBatch, InstanceError,
    InstanceParser, LineIdInfo, SparseFeature, Tensor,
};
pub use interleave::{InterleavedDataset, InterleavedIterator};
pub use kafka::{
    KafkaConfig, KafkaConsumer, KafkaDataSource, KafkaError, KafkaMessage, MockKafkaConsumer,
    OffsetReset,
};
pub use negative_sampling::{
    FrequencyNegativeSampler, InBatchNegativeSampler, NegativeSampler, NegativeSamplingConfig,
    NegativeSamplingDataset, SamplingStrategy, UniformNegativeSampler,
};
#[cfg(feature = "parquet")]
pub use parquet::{
    ParquetConfig, ParquetDataSource, ParquetError, ParquetIterator, ParquetSchema, Predicate,
};
pub use tfrecord::{TFRecordDataset, TFRecordError, TFRecordReader, TFRecordWriter};
pub use transform::{
    FilterMapTransform, FilterTransform, MapTransform, Transform, TransformChain,
    TransformedDataset,
};
pub use utils::{
    add_feature as add_valid_feature, enable_tob_env, get_feature_name_and_slot,
    get_slot_feature_name, get_slot_from_feature_name, register_slots,
};

/// Prelude module for convenient imports.
///
/// ```
/// use monolith_data::prelude::*;
///
/// let example = create_example();
/// assert!(!has_feature(&example, "test"));
/// ```
pub mod prelude {
    pub use crate::batch::{Batch, BatchedDataset};
    pub use crate::compression::CompressionType;
    pub use crate::dataset::{Dataset, VecDataset};
    pub use crate::example::{add_feature, create_example, get_feature, has_feature};
    pub use crate::instance::{
        extract_feature, extract_slot, make_fid, Instance, InstanceBatch, InstanceParser,
    };
    pub use crate::kafka::{KafkaConfig, KafkaDataSource, MockKafkaConsumer, OffsetReset};
    pub use crate::negative_sampling::{
        NegativeSampler, NegativeSamplingConfig, NegativeSamplingDataset, SamplingStrategy,
        UniformNegativeSampler,
    };
    #[cfg(feature = "parquet")]
    pub use crate::parquet::{ParquetConfig, ParquetDataSource, Predicate};
    pub use crate::tfrecord::TFRecordDataset;
    pub use crate::transform::{FilterTransform, MapTransform, Transform, TransformChain};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        use crate::prelude::*;

        let example = create_example();
        assert!(!has_feature(&example, "test"));
    }

    #[test]
    fn test_full_pipeline() {
        use crate::prelude::*;

        // Create some test examples
        let examples: Vec<_> = (0..100)
            .map(|i| {
                let mut ex = create_example();
                add_feature(&mut ex, "id", vec![i], vec![i as f32]);
                ex
            })
            .collect();

        // Build dataset
        let dataset = VecDataset::new(examples);

        // Apply transformations
        let results: Vec<_> = dataset
            .filter(|ex| {
                get_feature(ex, "id")
                    .map(|f| crate::example::extract_feature_data(f))
                    .map(|d| d.fid.first().copied().unwrap_or(0) % 2 == 0)
                    .unwrap_or(false)
            })
            .map(|mut ex| {
                add_feature(&mut ex, "even", vec![1], vec![1.0]);
                ex
            })
            .take(10)
            .iter()
            .collect();

        assert_eq!(results.len(), 10);
        for ex in &results {
            assert!(has_feature(ex, "even"));
        }
    }

    #[test]
    fn test_batch_pipeline() {
        use crate::prelude::*;

        let examples: Vec<_> = (0..25)
            .map(|i| {
                let mut ex = create_example();
                add_feature(&mut ex, "id", vec![i], vec![i as f32]);
                ex
            })
            .collect();

        let dataset = VecDataset::new(examples);
        let batches: Vec<_> = dataset.batch(10).iter().collect();

        assert_eq!(batches.len(), 3); // 10 + 10 + 5
        assert_eq!(batches[0].len(), 10);
        assert_eq!(batches[1].len(), 10);
        assert_eq!(batches[2].len(), 5);
    }

    #[test]
    fn test_transform_chain_integration() {
        let chain = TransformChain::new()
            .add(FilterTransform::new(|ex| has_feature(ex, "required")))
            .add(MapTransform::new(|mut ex| {
                add_feature(&mut ex, "processed", vec![1], vec![1.0]);
                ex
            }));

        // Example without required feature should be filtered
        let ex_without = create_example();
        assert!(chain.apply(ex_without).is_none());

        // Example with required feature should pass through and be modified
        let mut ex_with = create_example();
        add_feature(&mut ex_with, "required", vec![1], vec![1.0]);

        let result = chain.apply(ex_with).unwrap();
        assert!(has_feature(&result, "required"));
        assert!(has_feature(&result, "processed"));
    }
}
