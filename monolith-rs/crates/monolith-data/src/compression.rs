//! Compression support for TFRecord files.
//!
//! This module provides compression and decompression functionality for TFRecord files,
//! supporting Gzip, Zlib, and Snappy compression formats.
//!
//! # Example
//!
//! ```
//! use monolith_data::compression::{CompressionType, compress, decompress};
//!
//! let data = b"hello world";
//!
//! // Compress with gzip
//! #[cfg(feature = "gzip")]
//! {
//!     let compressed = compress(data, CompressionType::Gzip).unwrap();
//!     let decompressed = decompress(&compressed, CompressionType::Gzip).unwrap();
//!     assert_eq!(decompressed, data);
//! }
//! ```

use std::io;
use thiserror::Error;

/// Errors that can occur during compression/decompression operations.
#[derive(Error, Debug)]
pub enum CompressionError {
    /// An I/O error occurred during compression/decompression.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// The requested compression type is not available.
    #[error("Compression type {0:?} is not available. Enable the corresponding feature.")]
    NotAvailable(CompressionType),

    /// Failed to decompress data.
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
}

/// Result type for compression operations.
pub type Result<T> = std::result::Result<T, CompressionError>;

/// Supported compression types for TFRecord files.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CompressionType {
    /// No compression.
    #[default]
    None,
    /// Gzip compression.
    Gzip,
    /// Snappy compression.
    Snappy,
    /// Zlib compression.
    Zlib,
}

impl CompressionType {
    /// Detects the compression type from a file extension.
    ///
    /// # Arguments
    ///
    /// * `path` - The file path to analyze
    ///
    /// # Returns
    ///
    /// The detected compression type, or `None` if uncompressed.
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_data::compression::CompressionType;
    ///
    /// assert_eq!(CompressionType::from_extension("data.tfrecord.gz"), CompressionType::Gzip);
    /// assert_eq!(CompressionType::from_extension("data.tfrecord.snappy"), CompressionType::Snappy);
    /// assert_eq!(CompressionType::from_extension("data.tfrecord"), CompressionType::None);
    /// ```
    pub fn from_extension(path: &str) -> Self {
        let path_lower = path.to_lowercase();
        if path_lower.ends_with(".gz") || path_lower.ends_with(".gzip") {
            CompressionType::Gzip
        } else if path_lower.ends_with(".snappy") || path_lower.ends_with(".snp") {
            CompressionType::Snappy
        } else if path_lower.ends_with(".zlib") || path_lower.ends_with(".z") {
            CompressionType::Zlib
        } else {
            CompressionType::None
        }
    }

    /// Returns the file extension typically used for this compression type.
    pub fn extension(&self) -> &'static str {
        match self {
            CompressionType::None => "",
            CompressionType::Gzip => ".gz",
            CompressionType::Snappy => ".snappy",
            CompressionType::Zlib => ".zlib",
        }
    }
}

/// Compresses data using the specified compression type.
///
/// # Arguments
///
/// * `data` - The data to compress
/// * `compression` - The compression type to use
///
/// # Returns
///
/// The compressed data, or the original data if compression type is `None`.
///
/// # Errors
///
/// Returns an error if the compression type is not available (feature not enabled)
/// or if compression fails.
pub fn compress(data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),
        CompressionType::Gzip => compress_gzip(data),
        CompressionType::Zlib => compress_zlib(data),
        CompressionType::Snappy => compress_snappy(data),
    }
}

/// Decompresses data using the specified compression type.
///
/// # Arguments
///
/// * `data` - The compressed data
/// * `compression` - The compression type used
///
/// # Returns
///
/// The decompressed data, or the original data if compression type is `None`.
///
/// # Errors
///
/// Returns an error if the compression type is not available (feature not enabled)
/// or if decompression fails.
pub fn decompress(data: &[u8], compression: CompressionType) -> Result<Vec<u8>> {
    match compression {
        CompressionType::None => Ok(data.to_vec()),
        CompressionType::Gzip => decompress_gzip(data),
        CompressionType::Zlib => decompress_zlib(data),
        CompressionType::Snappy => decompress_snappy(data),
    }
}

// Gzip compression/decompression

#[cfg(feature = "gzip")]
fn compress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::write::GzEncoder;
    use flate2::Compression;

    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

#[cfg(not(feature = "gzip"))]
fn compress_gzip(_data: &[u8]) -> Result<Vec<u8>> {
    Err(CompressionError::NotAvailable(CompressionType::Gzip))
}

#[cfg(feature = "gzip")]
fn decompress_gzip(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::GzDecoder;

    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

#[cfg(not(feature = "gzip"))]
fn decompress_gzip(_data: &[u8]) -> Result<Vec<u8>> {
    Err(CompressionError::NotAvailable(CompressionType::Gzip))
}

// Zlib compression/decompression

#[cfg(feature = "gzip")]
fn compress_zlib(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::write::ZlibEncoder;
    use flate2::Compression;

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

#[cfg(not(feature = "gzip"))]
fn compress_zlib(_data: &[u8]) -> Result<Vec<u8>> {
    Err(CompressionError::NotAvailable(CompressionType::Zlib))
}

#[cfg(feature = "gzip")]
fn decompress_zlib(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::ZlibDecoder;

    let mut decoder = ZlibDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

#[cfg(not(feature = "gzip"))]
fn decompress_zlib(_data: &[u8]) -> Result<Vec<u8>> {
    Err(CompressionError::NotAvailable(CompressionType::Zlib))
}

// Snappy compression/decompression

#[cfg(feature = "snappy")]
fn compress_snappy(data: &[u8]) -> Result<Vec<u8>> {
    let mut encoder = snap::write::FrameEncoder::new(Vec::new());
    encoder.write_all(data)?;
    encoder.into_inner().map_err(|e| {
        CompressionError::DecompressionFailed(format!("Failed to finish snappy compression: {}", e))
    })
}

#[cfg(not(feature = "snappy"))]
fn compress_snappy(_data: &[u8]) -> Result<Vec<u8>> {
    Err(CompressionError::NotAvailable(CompressionType::Snappy))
}

#[cfg(feature = "snappy")]
fn decompress_snappy(data: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = snap::read::FrameDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

#[cfg(not(feature = "snappy"))]
fn decompress_snappy(_data: &[u8]) -> Result<Vec<u8>> {
    Err(CompressionError::NotAvailable(CompressionType::Snappy))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_type_from_extension() {
        assert_eq!(
            CompressionType::from_extension("data.tfrecord.gz"),
            CompressionType::Gzip
        );
        assert_eq!(
            CompressionType::from_extension("data.tfrecord.gzip"),
            CompressionType::Gzip
        );
        assert_eq!(
            CompressionType::from_extension("data.tfrecord.snappy"),
            CompressionType::Snappy
        );
        assert_eq!(
            CompressionType::from_extension("data.tfrecord.snp"),
            CompressionType::Snappy
        );
        assert_eq!(
            CompressionType::from_extension("data.tfrecord.zlib"),
            CompressionType::Zlib
        );
        assert_eq!(
            CompressionType::from_extension("data.tfrecord.z"),
            CompressionType::Zlib
        );
        assert_eq!(
            CompressionType::from_extension("data.tfrecord"),
            CompressionType::None
        );
        assert_eq!(
            CompressionType::from_extension("DATA.TFRECORD.GZ"),
            CompressionType::Gzip
        );
    }

    #[test]
    fn test_compression_type_extension() {
        assert_eq!(CompressionType::None.extension(), "");
        assert_eq!(CompressionType::Gzip.extension(), ".gz");
        assert_eq!(CompressionType::Snappy.extension(), ".snappy");
        assert_eq!(CompressionType::Zlib.extension(), ".zlib");
    }

    #[test]
    fn test_no_compression() {
        let data = b"hello world, this is test data";
        let compressed = compress(data, CompressionType::None).unwrap();
        assert_eq!(compressed, data);

        let decompressed = decompress(&compressed, CompressionType::None).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn test_gzip_compression() {
        let data = b"hello world, this is test data for gzip compression";
        let compressed = compress(data, CompressionType::Gzip).unwrap();

        // Compressed data should be different from original (unless very small)
        assert_ne!(compressed.as_slice(), data.as_slice());

        let decompressed = decompress(&compressed, CompressionType::Gzip).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn test_zlib_compression() {
        let data = b"hello world, this is test data for zlib compression";
        let compressed = compress(data, CompressionType::Zlib).unwrap();

        // Compressed data should be different from original
        assert_ne!(compressed.as_slice(), data.as_slice());

        let decompressed = decompress(&compressed, CompressionType::Zlib).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "snappy")]
    #[test]
    fn test_snappy_compression() {
        let data = b"hello world, this is test data for snappy compression";
        let compressed = compress(data, CompressionType::Snappy).unwrap();

        let decompressed = decompress(&compressed, CompressionType::Snappy).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_all_compression_types() {
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(100);

        for compression_type in [
            CompressionType::None,
            CompressionType::Gzip,
            CompressionType::Zlib,
            CompressionType::Snappy,
        ] {
            let compressed = compress(&data, compression_type).unwrap();
            let decompressed = decompress(&compressed, compression_type).unwrap();
            assert_eq!(decompressed, data, "Failed for {:?}", compression_type);
        }
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_compression_ratio() {
        // Highly compressible data
        let data = b"aaaaaaaaaa".repeat(1000);

        let gzip_compressed = compress(&data, CompressionType::Gzip).unwrap();
        let snappy_compressed = compress(&data, CompressionType::Snappy).unwrap();

        // Compressed should be significantly smaller for repetitive data
        assert!(gzip_compressed.len() < data.len() / 10);
        assert!(snappy_compressed.len() < data.len() / 10);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn test_empty_data_compression() {
        let data: &[u8] = b"";

        let compressed = compress(data, CompressionType::Gzip).unwrap();
        let decompressed = decompress(&compressed, CompressionType::Gzip).unwrap();
        assert_eq!(decompressed, data);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn test_large_data_compression() {
        // 1MB of random-ish data
        let data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();

        let compressed = compress(&data, CompressionType::Gzip).unwrap();
        let decompressed = decompress(&compressed, CompressionType::Gzip).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_compression_type_default() {
        let default: CompressionType = Default::default();
        assert_eq!(default, CompressionType::None);
    }
}
