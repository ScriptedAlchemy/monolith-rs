//! Embedding compressors for reducing memory footprint.
//!
//! This module provides various compression strategies for embeddings,
//! trading off between memory usage and precision.
//!
//! # Available Compressors
//!
//! - [`NoCompression`] - Pass-through, no compression applied
//! - [`Fp16Compressor`] - Half-precision floating point (16-bit)
//! - [`FixedR8Compressor`] - 8-bit fixed point quantization with scale
//! - [`OneBitCompressor`] - Binary quantization (sign only)
//!
//! # Example
//!
//! ```
//! use monolith_hash_table::compressor::{Compressor, Fp16Compressor};
//!
//! let compressor = Fp16Compressor;
//! let embedding = vec![1.0f32, -0.5, 0.25, 0.0];
//!
//! let compressed = compressor.compress(&embedding);
//! let decompressed = compressor.decompress(&compressed, 4);
//!
//! // Values are approximately equal (some precision loss with fp16)
//! for (orig, decomp) in embedding.iter().zip(decompressed.iter()) {
//!     assert!((orig - decomp).abs() < 0.001);
//! }
//! ```

use half::f16;

/// Trait for embedding compression strategies.
///
/// Compressors convert f32 embeddings to a compressed byte representation
/// and back. This allows trading off memory usage against precision.
pub trait Compressor: Send + Sync {
    /// Compress an embedding to bytes.
    fn compress(&self, embedding: &[f32]) -> Vec<u8>;

    /// Decompress bytes back to an embedding.
    ///
    /// # Arguments
    /// * `data` - The compressed byte data
    /// * `dim` - The expected dimension of the output embedding
    fn decompress(&self, data: &[u8], dim: usize) -> Vec<f32>;

    /// Returns the compressed size in bytes for an embedding of the given dimension.
    fn compressed_size(&self, dim: usize) -> usize;
}

/// No compression - pass-through implementation.
///
/// This compressor stores embeddings as raw f32 bytes without any compression.
/// Useful as a baseline or when maximum precision is required.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoCompression;

impl Compressor for NoCompression {
    fn compress(&self, embedding: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(embedding.len() * 4);
        for &value in embedding {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    fn decompress(&self, data: &[u8], dim: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(dim);
        for chunk in data.chunks_exact(4).take(dim) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            result.push(f32::from_le_bytes(bytes));
        }
        result
    }

    fn compressed_size(&self, dim: usize) -> usize {
        dim * 4
    }
}

/// Half-precision (f16) compressor.
///
/// Converts f32 values to f16 (IEEE 754 half-precision), reducing storage
/// by 50% with minimal precision loss for typical embedding values.
///
/// # Precision
///
/// f16 has:
/// - 1 sign bit
/// - 5 exponent bits (range roughly 6.0e-5 to 65504)
/// - 10 mantissa bits (~3 decimal digits precision)
#[derive(Debug, Clone, Copy, Default)]
pub struct Fp16Compressor;

impl Compressor for Fp16Compressor {
    fn compress(&self, embedding: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(embedding.len() * 2);
        for &value in embedding {
            let half = f16::from_f32(value);
            bytes.extend_from_slice(&half.to_le_bytes());
        }
        bytes
    }

    fn decompress(&self, data: &[u8], dim: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(dim);
        for chunk in data.chunks_exact(2).take(dim) {
            let bytes: [u8; 2] = chunk.try_into().unwrap();
            let half = f16::from_le_bytes(bytes);
            result.push(half.to_f32());
        }
        result
    }

    fn compressed_size(&self, dim: usize) -> usize {
        dim * 2
    }
}

/// 8-bit fixed point compressor with scale factor.
///
/// Quantizes f32 values to 8-bit integers using a uniform scale.
/// The scale is stored as part of the compressed data to enable reconstruction.
///
/// # Algorithm
///
/// 1. Find the maximum absolute value in the embedding
/// 2. Scale all values to [-127, 127] range
/// 3. Store scale factor (4 bytes) followed by quantized values (1 byte each)
///
/// # Compression Ratio
///
/// For an embedding of dimension D:
/// - Original: 4D bytes
/// - Compressed: 4 + D bytes (scale + quantized values)
/// - Ratio: ~4x for large D
#[derive(Debug, Clone, Copy, Default)]
pub struct FixedR8Compressor;

impl FixedR8Compressor {
    /// The maximum value for the quantized range.
    const QUANT_MAX: f32 = 127.0;
}

impl Compressor for FixedR8Compressor {
    fn compress(&self, embedding: &[f32]) -> Vec<u8> {
        if embedding.is_empty() {
            return vec![0u8; 4]; // Just store zero scale
        }

        // Find max absolute value for scaling
        let max_abs = embedding.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

        // Compute scale factor (avoid division by zero)
        let scale = if max_abs > 0.0 {
            max_abs / Self::QUANT_MAX
        } else {
            1.0
        };

        let mut bytes = Vec::with_capacity(4 + embedding.len());

        // Store scale factor first
        bytes.extend_from_slice(&scale.to_le_bytes());

        // Quantize and store each value
        for &value in embedding {
            let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
            bytes.push(quantized as u8);
        }

        bytes
    }

    fn decompress(&self, data: &[u8], dim: usize) -> Vec<f32> {
        if data.len() < 4 {
            return vec![0.0; dim];
        }

        // Read scale factor
        let scale_bytes: [u8; 4] = data[0..4].try_into().unwrap();
        let scale = f32::from_le_bytes(scale_bytes);

        // Dequantize values
        let mut result = Vec::with_capacity(dim);
        for &byte in data[4..].iter().take(dim) {
            let quantized = byte as i8;
            result.push(quantized as f32 * scale);
        }

        // Pad with zeros if not enough data
        while result.len() < dim {
            result.push(0.0);
        }

        result
    }

    fn compressed_size(&self, dim: usize) -> usize {
        4 + dim // 4 bytes for scale + 1 byte per dimension
    }
}

/// One-bit (binary) compressor.
///
/// Stores only the sign of each value, achieving maximum compression
/// at the cost of significant precision loss.
///
/// # Algorithm
///
/// - Positive values (including zero) are stored as 1
/// - Negative values are stored as 0
/// - 8 values are packed into each byte
/// - Reconstruction uses +1.0 for 1 and -1.0 for 0
///
/// # Compression Ratio
///
/// For an embedding of dimension D:
/// - Original: 4D bytes
/// - Compressed: ceil(D/8) bytes
/// - Ratio: 32x
///
/// # Use Cases
///
/// Binary embeddings are useful for:
/// - Approximate nearest neighbor search with Hamming distance
/// - Memory-constrained environments
/// - When only relative directions matter
#[derive(Debug, Clone, Copy, Default)]
pub struct OneBitCompressor;

impl Compressor for OneBitCompressor {
    fn compress(&self, embedding: &[f32]) -> Vec<u8> {
        let num_bytes = embedding.len().div_ceil(8);
        let mut bytes = vec![0u8; num_bytes];

        for (i, &value) in embedding.iter().enumerate() {
            if value >= 0.0 {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        bytes
    }

    fn decompress(&self, data: &[u8], dim: usize) -> Vec<f32> {
        let mut result = Vec::with_capacity(dim);

        for i in 0..dim {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            if byte_idx < data.len() {
                let bit = (data[byte_idx] >> bit_idx) & 1;
                result.push(if bit == 1 { 1.0 } else { -1.0 });
            } else {
                result.push(1.0); // Default to positive if not enough data
            }
        }

        result
    }

    fn compressed_size(&self, dim: usize) -> usize {
        dim.div_ceil(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_compression_roundtrip() {
        let compressor = NoCompression;
        let embedding = vec![1.0, -2.5, 0.0, 3.14159, -0.001, 100.0];

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        assert_eq!(embedding, decompressed);
    }

    #[test]
    fn test_no_compression_size() {
        let compressor = NoCompression;
        assert_eq!(compressor.compressed_size(10), 40);
        assert_eq!(compressor.compressed_size(128), 512);
    }

    #[test]
    fn test_fp16_roundtrip() {
        let compressor = Fp16Compressor;
        let embedding = vec![1.0, -0.5, 0.25, 0.0, 2.0, -3.0];

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        // fp16 should be exact for these simple values
        for (orig, decomp) in embedding.iter().zip(decompressed.iter()) {
            assert!(
                (orig - decomp).abs() < 0.001,
                "orig={}, decomp={}",
                orig,
                decomp
            );
        }
    }

    #[test]
    fn test_fp16_precision() {
        let compressor = Fp16Compressor;
        // Test with values that may have some precision loss
        let embedding = vec![0.123456, -0.654321, 1.111111];

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        // fp16 has ~3 decimal digits of precision
        for (orig, decomp) in embedding.iter().zip(decompressed.iter()) {
            assert!(
                (orig - decomp).abs() < 0.01,
                "orig={}, decomp={}",
                orig,
                decomp
            );
        }
    }

    #[test]
    fn test_fp16_size() {
        let compressor = Fp16Compressor;
        assert_eq!(compressor.compressed_size(10), 20);
        assert_eq!(compressor.compressed_size(128), 256);
    }

    #[test]
    fn test_fixed_r8_roundtrip() {
        let compressor = FixedR8Compressor;
        let embedding = vec![1.0, -1.0, 0.5, -0.5, 0.0];

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        // Should be reasonably close
        for (orig, decomp) in embedding.iter().zip(decompressed.iter()) {
            assert!(
                (orig - decomp).abs() < 0.02,
                "orig={}, decomp={}",
                orig,
                decomp
            );
        }
    }

    #[test]
    fn test_fixed_r8_scale() {
        let compressor = FixedR8Compressor;
        // Test that large values are handled correctly
        let embedding = vec![100.0, -100.0, 50.0, -50.0];

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        for (orig, decomp) in embedding.iter().zip(decompressed.iter()) {
            // Allow 1% error
            assert!(
                (orig - decomp).abs() < orig.abs() * 0.02 + 0.1,
                "orig={}, decomp={}",
                orig,
                decomp
            );
        }
    }

    #[test]
    fn test_fixed_r8_zeros() {
        let compressor = FixedR8Compressor;
        let embedding = vec![0.0, 0.0, 0.0, 0.0];

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        for value in decompressed {
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn test_fixed_r8_size() {
        let compressor = FixedR8Compressor;
        assert_eq!(compressor.compressed_size(10), 14); // 4 + 10
        assert_eq!(compressor.compressed_size(128), 132); // 4 + 128
    }

    #[test]
    fn test_one_bit_roundtrip() {
        let compressor = OneBitCompressor;
        let embedding = vec![1.0, -1.0, 0.5, -0.5, 0.0, -0.001];

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        // One-bit only preserves sign
        let expected = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        assert_eq!(decompressed, expected);
    }

    #[test]
    fn test_one_bit_packing() {
        let compressor = OneBitCompressor;
        // Test that bits are packed correctly
        let embedding = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]; // 8 values = 1 byte

        let compressed = compressor.compress(&embedding);
        assert_eq!(compressed.len(), 1);

        // Pattern: 1,0,1,0,1,0,1,0 = 0b01010101 = 85
        assert_eq!(compressed[0], 0b01010101);
    }

    #[test]
    fn test_one_bit_size() {
        let compressor = OneBitCompressor;
        assert_eq!(compressor.compressed_size(1), 1);
        assert_eq!(compressor.compressed_size(8), 1);
        assert_eq!(compressor.compressed_size(9), 2);
        assert_eq!(compressor.compressed_size(128), 16);
    }

    #[test]
    fn test_one_bit_partial_byte() {
        let compressor = OneBitCompressor;
        // Test with 10 values (needs 2 bytes, but only 2 bits used in second byte)
        let embedding = vec![1.0; 10];

        let compressed = compressor.compress(&embedding);
        assert_eq!(compressed.len(), 2);

        let decompressed = compressor.decompress(&compressed, 10);
        assert_eq!(decompressed.len(), 10);
        assert!(decompressed.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_empty_embedding() {
        let no_comp = NoCompression;
        let fp16 = Fp16Compressor;
        let fixed_r8 = FixedR8Compressor;
        let one_bit = OneBitCompressor;

        let empty: Vec<f32> = vec![];

        assert!(no_comp.compress(&empty).is_empty());
        assert!(fp16.compress(&empty).is_empty());
        assert_eq!(fixed_r8.compress(&empty).len(), 4); // Just scale
        assert!(one_bit.compress(&empty).is_empty());
    }

    #[test]
    fn test_compressor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<NoCompression>();
        assert_send_sync::<Fp16Compressor>();
        assert_send_sync::<FixedR8Compressor>();
        assert_send_sync::<OneBitCompressor>();
    }

    #[test]
    fn test_trait_object() {
        // Verify compressors work as trait objects
        let compressors: Vec<Box<dyn Compressor>> = vec![
            Box::new(NoCompression),
            Box::new(Fp16Compressor),
            Box::new(FixedR8Compressor),
            Box::new(OneBitCompressor),
        ];

        let embedding = vec![1.0, -1.0, 0.5, -0.5];

        for compressor in compressors {
            let compressed = compressor.compress(&embedding);
            let decompressed = compressor.decompress(&compressed, embedding.len());
            assert_eq!(decompressed.len(), embedding.len());
        }
    }

    #[test]
    fn test_large_embedding() {
        let compressor = Fp16Compressor;
        let embedding: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.001).collect();

        let compressed = compressor.compress(&embedding);
        let decompressed = compressor.decompress(&compressed, embedding.len());

        assert_eq!(decompressed.len(), embedding.len());
        for (orig, decomp) in embedding.iter().zip(decompressed.iter()) {
            assert!((orig - decomp).abs() < 0.01);
        }
    }
}
