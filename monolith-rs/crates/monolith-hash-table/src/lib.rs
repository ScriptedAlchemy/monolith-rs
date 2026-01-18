//! Hash table implementations for embedding storage.
//!
//! This crate provides various hash table implementations optimized for
//! storing and retrieving embeddings in machine learning systems.
//!
//! # Overview
//!
//! The main components are:
//!
//! - [`EmbeddingHashTable`] - The core trait defining the interface for embedding storage
//! - [`CuckooEmbeddingHashTable`] - A cuckoo hash table implementation for efficient lookups
//! - [`MultiHashTable`] - A sharded hash table for concurrent access
//!
//! # Example
//!
//! ```
//! use monolith_hash_table::{EmbeddingHashTable, CuckooEmbeddingHashTable};
//!
//! let mut table = CuckooEmbeddingHashTable::new(1024, 8);
//!
//! // Assign embeddings
//! let ids = vec![1, 2, 3];
//! let embeddings = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
//!                       0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
//!                       1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4];
//! table.assign(&ids, &embeddings).unwrap();
//!
//! // Lookup embeddings
//! let mut output = vec![0.0; 24];
//! table.lookup(&ids, &mut output).unwrap();
//! ```

mod cuckoo;
pub mod compressor;
mod entry;
mod error;
pub mod eviction;
pub mod initializer;
mod multi;
mod traits;

pub use compressor::{Compressor, FixedR8Compressor, Fp16Compressor, NoCompression, OneBitCompressor};
pub use cuckoo::CuckooEmbeddingHashTable;
pub use entry::{EmbeddingEntry, OptimizerState};
pub use error::{HashTableError, Result};
pub use eviction::{EvictionPolicy, LRUEviction, NoEviction, TimeBasedEviction};
pub use initializer::{
    ConstantInitializer, Initializer, InitializerFactory, OnesInitializer,
    RandomNormalInitializer, RandomUniformInitializer, TruncatedNormalInitializer,
    XavierNormalInitializer, XavierUniformInitializer, ZerosInitializer,
};
pub use multi::MultiHashTable;
pub use traits::EmbeddingHashTable;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuckoo_basic_operations() {
        let mut table = CuckooEmbeddingHashTable::new(1024, 4);

        // Test assign and lookup
        let ids = vec![1, 2, 3];
        let embeddings = vec![
            1.0, 2.0, 3.0, 4.0, // id 1
            5.0, 6.0, 7.0, 8.0, // id 2
            9.0, 10.0, 11.0, 12.0, // id 3
        ];

        table.assign(&ids, &embeddings).unwrap();

        assert!(table.contains(1));
        assert!(table.contains(2));
        assert!(table.contains(3));
        assert!(!table.contains(4));

        let mut output = vec![0.0; 12];
        table.lookup(&ids, &mut output).unwrap();

        assert_eq!(output, embeddings);
    }

    #[test]
    fn test_multi_hash_table() {
        let mut table = MultiHashTable::new(4, 256, 4);

        let ids = vec![1, 2, 3, 100, 101, 102];
        let embeddings = vec![
            1.0, 2.0, 3.0, 4.0, // id 1
            5.0, 6.0, 7.0, 8.0, // id 2
            9.0, 10.0, 11.0, 12.0, // id 3
            13.0, 14.0, 15.0, 16.0, // id 100
            17.0, 18.0, 19.0, 20.0, // id 101
            21.0, 22.0, 23.0, 24.0, // id 102
        ];

        table.assign(&ids, &embeddings).unwrap();

        for id in &ids {
            assert!(table.contains(*id));
        }

        let mut output = vec![0.0; 24];
        table.lookup(&ids, &mut output).unwrap();

        assert_eq!(output, embeddings);
    }
}
