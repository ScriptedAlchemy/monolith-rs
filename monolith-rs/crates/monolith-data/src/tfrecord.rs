//! TFRecord file format reader.
//!
//! This module provides [`TFRecordDataset`] for reading TFRecord files, which is
//! the standard file format used for storing training data in TensorFlow-based
//! systems including the original Monolith implementation.
//!
//! # TFRecord Format
//!
//! Each record in a TFRecord file has the following format:
//! - `uint64` length
//! - `uint32` masked CRC32 of length
//! - `byte[length]` data
//! - `uint32` masked CRC32 of data
//!
//! # Compression Support
//!
//! TFRecord files can be compressed using Gzip, Zlib, or Snappy. Compression is
//! applied at the record level (each record's data payload is compressed).
//!
//! ```no_run
//! use monolith_data::tfrecord::TFRecordDataset;
//! use monolith_data::compression::CompressionType;
//!
//! // Auto-detect compression from file extension
//! let dataset = TFRecordDataset::open("data/train.tfrecord.gz").unwrap();
//!
//! // Or specify explicitly
//! let dataset = TFRecordDataset::open("data/train.tfrecord")
//!     .unwrap()
//!     .with_compression(CompressionType::Gzip);
//! ```
//!
//! # Example
//!
//! ```no_run
//! use monolith_data::tfrecord::TFRecordDataset;
//!
//! // This example requires a TFRecord file to exist
//! let dataset = TFRecordDataset::open("data/train.tfrecord").unwrap();
//! for example in dataset.iter() {
//!     println!("{:?}", example);
//! }
//! ```

use crate::dataset::Dataset;
use bytes::{Buf, Bytes};
use glob::glob;
use monolith_proto::Example;
use prost::Message;
use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::compression::{self, CompressionType};

/// Errors that can occur when reading TFRecord files.
#[derive(Error, Debug)]
pub enum TFRecordError {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Failed to decode a protobuf message.
    #[error("Protobuf decode error: {0}")]
    Decode(#[from] prost::DecodeError),

    /// CRC checksum mismatch.
    #[error("CRC checksum mismatch: expected {expected:#x}, got {actual:#x}")]
    CrcMismatch {
        /// The expected CRC value.
        expected: u32,
        /// The actual CRC value.
        actual: u32,
    },

    /// Unexpected end of file.
    #[error("Unexpected end of file")]
    UnexpectedEof,

    /// Invalid record format.
    #[error("Invalid record format: {0}")]
    InvalidFormat(String),

    /// Compression error.
    #[error("Compression error: {0}")]
    Compression(#[from] compression::CompressionError),
}

/// Result type for TFRecord operations.
pub type Result<T> = std::result::Result<T, TFRecordError>;

/// A dataset that reads examples from TFRecord files.
///
/// `TFRecordDataset` provides an iterator over [`Example`] protobuf messages
/// stored in TFRecord format files.
///
/// # Compression
///
/// Compression can be automatically detected from file extensions (`.gz`, `.snappy`, `.zlib`)
/// or explicitly configured using [`with_compression`](Self::with_compression).
#[derive(Clone)]
pub struct TFRecordDataset {
    paths: Vec<PathBuf>,
    verify_crc: bool,
    compression: Option<CompressionType>,
}

impl TFRecordDataset {
    /// Opens a single TFRecord file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the TFRecord file
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        // Verify file exists
        if !path.exists() {
            return Err(TFRecordError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("File not found: {}", path.display()),
            )));
        }
        Ok(Self {
            paths: vec![path],
            verify_crc: true,
            compression: None,
        })
    }

    /// Opens multiple TFRecord files.
    ///
    /// # Arguments
    ///
    /// * `paths` - Paths to the TFRecord files
    ///
    /// # Errors
    ///
    /// Returns an error if any file cannot be found.
    pub fn open_multiple<P: AsRef<Path>>(paths: &[P]) -> Result<Self> {
        let paths: Vec<PathBuf> = paths.iter().map(|p| p.as_ref().to_path_buf()).collect();
        for path in &paths {
            if !path.exists() {
                return Err(TFRecordError::Io(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("File not found: {}", path.display()),
                )));
            }
        }
        Ok(Self {
            paths,
            verify_crc: true,
            compression: None,
        })
    }

    /// Creates a dataset from a glob pattern.
    ///
    /// # Arguments
    ///
    /// * `pattern` - A glob pattern to match files
    ///
    /// Supported inputs:
    /// - Single glob: `"data/train-*.tfrecord"`
    /// - Comma-separated globs/paths: `"a/*.tfrecord,b/*.tfrecord,/tmp/file.tfrecord"`
    /// - File list prefixed by `@`: `"@/tmp/tfrecord_files.txt"` where each line is a path/glob.
    pub fn from_pattern(pattern: &str) -> Result<Self> {
        fn is_glob_pattern(s: &str) -> bool {
            s.contains('*') || s.contains('?') || s.contains('[') || s.contains('{')
        }

        fn expand_specs(spec: &str) -> Result<Vec<String>> {
            if let Some(list_path) = spec.strip_prefix('@') {
                let raw = std::fs::read_to_string(list_path).map_err(TFRecordError::Io)?;
                let specs = raw
                    .lines()
                    .map(|l| l.trim())
                    .filter(|l| !l.is_empty() && !l.starts_with('#'))
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>();
                return Ok(specs);
            }
            Ok(spec
                .split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.to_string())
                .collect())
        }

        let specs = expand_specs(pattern)?;
        let mut unique = BTreeSet::new();

        for spec in specs {
            if is_glob_pattern(&spec) {
                for entry in glob(&spec).map_err(|e| TFRecordError::InvalidFormat(e.to_string()))? {
                    let path = entry.map_err(|e| TFRecordError::InvalidFormat(e.to_string()))?;
                    if path.exists() {
                        unique.insert(path);
                    }
                }
            } else {
                let path = PathBuf::from(&spec);
                if path.exists() {
                    unique.insert(path);
                }
            }
        }

        if unique.is_empty() {
            return Err(TFRecordError::Io(io::Error::new(
                io::ErrorKind::NotFound,
                format!("No files matched pattern: {}", pattern),
            )));
        }
        let paths = unique.into_iter().collect::<Vec<_>>();
        Ok(Self {
            paths,
            verify_crc: true,
            compression: None,
        })
    }

    /// Sets whether to verify CRC checksums when reading records.
    ///
    /// # Arguments
    ///
    /// * `verify` - If `true`, verify CRC checksums (default)
    pub fn verify_crc(mut self, verify: bool) -> Self {
        self.verify_crc = verify;
        self
    }

    /// Sets the compression type for reading records.
    ///
    /// If not set, compression will be auto-detected from file extensions.
    ///
    /// # Arguments
    ///
    /// * `compression` - The compression type to use
    ///
    /// # Example
    ///
    /// ```no_run
    /// use monolith_data::tfrecord::TFRecordDataset;
    /// use monolith_data::compression::CompressionType;
    ///
    /// let dataset = TFRecordDataset::open("data/train.tfrecord")
    ///     .unwrap()
    ///     .with_compression(CompressionType::Gzip);
    /// ```
    pub fn with_compression(mut self, compression: CompressionType) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Returns the configured compression type, if any.
    pub fn compression(&self) -> Option<CompressionType> {
        self.compression
    }

    /// Returns an iterator over examples in the dataset.
    pub fn iter(&self) -> TFRecordIterator {
        TFRecordIterator::new(self.paths.clone(), self.verify_crc, self.compression)
    }

    /// Returns the number of files in this dataset.
    pub fn file_count(&self) -> usize {
        self.paths.len()
    }

    /// Returns the file paths in this dataset.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }
}

impl Dataset for TFRecordDataset {
    type Iter = TFRecordIterator;

    fn iter(self) -> Self::Iter {
        TFRecordIterator::new(self.paths, self.verify_crc, self.compression)
    }
}

/// Iterator over examples in TFRecord files.
pub struct TFRecordIterator {
    paths: Vec<PathBuf>,
    current_file_index: usize,
    reader: Option<TFRecordReader<BufReader<File>>>,
    verify_crc: bool,
    compression: Option<CompressionType>,
}

impl TFRecordIterator {
    pub(crate) fn new(
        paths: Vec<PathBuf>,
        verify_crc: bool,
        compression: Option<CompressionType>,
    ) -> Self {
        Self {
            paths,
            current_file_index: 0,
            reader: None,
            verify_crc,
            compression,
        }
    }

    fn open_next_file(&mut self) -> Option<()> {
        while self.current_file_index < self.paths.len() {
            let path = &self.paths[self.current_file_index];
            self.current_file_index += 1;

            match File::open(path) {
                Ok(file) => {
                    let reader = BufReader::new(file);
                    // Determine compression: use explicit setting or auto-detect from extension
                    let compression = self.compression.unwrap_or_else(|| {
                        CompressionType::from_extension(&path.to_string_lossy())
                    });
                    self.reader = Some(
                        TFRecordReader::new(reader, self.verify_crc).with_compression(compression),
                    );
                    return Some(());
                }
                Err(_) => continue, // Skip files that can't be opened
            }
        }
        None
    }
}

impl Iterator for TFRecordIterator {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // If we have a reader, try to read the next record
            if let Some(ref mut reader) = self.reader {
                match reader.read_example() {
                    Ok(Some(example)) => return Some(example),
                    Ok(None) => {
                        // End of file, try next file
                        self.reader = None;
                    }
                    Err(_) => {
                        // Error reading, try next file
                        self.reader = None;
                    }
                }
            }

            // Try to open next file
            self.open_next_file()?;
        }
    }
}

/// Low-level TFRecord file reader.
///
/// This reader handles the raw TFRecord format, including reading
/// length-prefixed records and optionally verifying CRC checksums.
///
/// # Compression
///
/// The reader supports decompressing records on the fly. Compression is applied
/// at the record level (the data payload of each record is compressed).
pub struct TFRecordReader<R> {
    reader: R,
    verify_crc: bool,
    compression: CompressionType,
}

impl<R: Read> TFRecordReader<R> {
    /// Creates a new TFRecord reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - The underlying reader
    /// * `verify_crc` - Whether to verify CRC checksums
    pub fn new(reader: R, verify_crc: bool) -> Self {
        Self {
            reader,
            verify_crc,
            compression: CompressionType::None,
        }
    }

    /// Sets the compression type for this reader.
    ///
    /// # Arguments
    ///
    /// * `compression` - The compression type to use for decompressing records
    pub fn with_compression(mut self, compression: CompressionType) -> Self {
        self.compression = compression;
        self
    }

    /// Returns the current compression type.
    pub fn compression(&self) -> CompressionType {
        self.compression
    }

    /// Reads the next raw record from the file.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(bytes))` - The next record's data (decompressed if compression is set)
    /// - `Ok(None)` - End of file reached
    /// - `Err(e)` - An error occurred
    pub fn read_record(&mut self) -> Result<Option<Bytes>> {
        // Read length (8 bytes, little-endian)
        let mut length_buf = [0u8; 8];
        match self.reader.read_exact(&mut length_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(TFRecordError::Io(e)),
        }
        let length = u64::from_le_bytes(length_buf) as usize;

        // Read length CRC (4 bytes, little-endian)
        let mut length_crc_buf = [0u8; 4];
        self.reader.read_exact(&mut length_crc_buf)?;
        let length_crc = u32::from_le_bytes(length_crc_buf);

        // Verify length CRC if enabled
        if self.verify_crc {
            let expected_crc = masked_crc32c(&length_buf);
            if length_crc != expected_crc {
                return Err(TFRecordError::CrcMismatch {
                    expected: expected_crc,
                    actual: length_crc,
                });
            }
        }

        // Read data
        let mut data = vec![0u8; length];
        self.reader.read_exact(&mut data)?;

        // Read data CRC (4 bytes, little-endian)
        let mut data_crc_buf = [0u8; 4];
        self.reader.read_exact(&mut data_crc_buf)?;
        let data_crc = u32::from_le_bytes(data_crc_buf);

        // Verify data CRC if enabled
        if self.verify_crc {
            let expected_crc = masked_crc32c(&data);
            if data_crc != expected_crc {
                return Err(TFRecordError::CrcMismatch {
                    expected: expected_crc,
                    actual: data_crc,
                });
            }
        }

        // Decompress if needed
        let data = compression::decompress(&data, self.compression)?;

        Ok(Some(Bytes::from(data)))
    }

    /// Reads the next Example protobuf from the file.
    ///
    /// # Returns
    ///
    /// - `Ok(Some(example))` - The next example
    /// - `Ok(None)` - End of file reached
    /// - `Err(e)` - An error occurred
    pub fn read_example(&mut self) -> Result<Option<Example>> {
        match self.read_record()? {
            Some(data) => {
                let example = Example::decode(data.chunk())?;
                Ok(Some(example))
            }
            None => Ok(None),
        }
    }
}

/// Computes the masked CRC32C checksum used by TFRecord.
///
/// The mask is: `((crc >> 15) | (crc << 17)) + 0xa282ead8`
fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    crc.rotate_right(15).wrapping_add(0xa282ead8)
}

/// Computes CRC32C checksum.
///
/// This is a simple implementation. In production, you would use
/// a hardware-accelerated implementation from the `crc32c` crate.
fn crc32c(data: &[u8]) -> u32 {
    // CRC32C polynomial
    const POLY: u32 = 0x82f63b78;

    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ POLY
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Writes examples to TFRecord format.
///
/// # Compression
///
/// The writer supports compressing records on the fly. Compression is applied
/// at the record level (the data payload of each record is compressed).
pub struct TFRecordWriter<W> {
    writer: W,
    compression: CompressionType,
}

impl<W: io::Write> TFRecordWriter<W> {
    /// Creates a new TFRecord writer.
    ///
    /// # Arguments
    ///
    /// * `writer` - The underlying writer
    pub fn new(writer: W) -> Self {
        Self {
            writer,
            compression: CompressionType::None,
        }
    }

    /// Sets the compression type for this writer.
    ///
    /// # Arguments
    ///
    /// * `compression` - The compression type to use for compressing records
    pub fn with_compression(mut self, compression: CompressionType) -> Self {
        self.compression = compression;
        self
    }

    /// Returns the current compression type.
    pub fn compression(&self) -> CompressionType {
        self.compression
    }

    /// Writes a raw record to the file.
    ///
    /// # Arguments
    ///
    /// * `data` - The record data to write (will be compressed if compression is set)
    pub fn write_record(&mut self, data: &[u8]) -> Result<()> {
        // Compress if needed
        let data = compression::compress(data, self.compression)?;

        let length = data.len() as u64;
        let length_bytes = length.to_le_bytes();

        // Write length
        self.writer.write_all(&length_bytes)?;

        // Write length CRC
        let length_crc = masked_crc32c(&length_bytes);
        self.writer.write_all(&length_crc.to_le_bytes())?;

        // Write data
        self.writer.write_all(&data)?;

        // Write data CRC
        let data_crc = masked_crc32c(&data);
        self.writer.write_all(&data_crc.to_le_bytes())?;

        Ok(())
    }

    /// Writes an Example protobuf to the file.
    ///
    /// # Arguments
    ///
    /// * `example` - The example to write
    pub fn write_example(&mut self, example: &Example) -> Result<()> {
        let data = example.encode_to_vec();
        self.write_record(&data)
    }

    /// Flushes the underlying writer.
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::example::{add_feature, create_example};
    use std::io::Cursor;
    use tempfile::tempdir;

    #[test]
    fn test_crc32c() {
        // Test vector from the CRC32C spec
        let data = b"123456789";
        let crc = crc32c(data);
        assert_eq!(crc, 0xe3069283);
    }

    #[test]
    fn test_masked_crc() {
        let data = b"test";
        let crc = masked_crc32c(data);
        // Just verify it produces a value and is deterministic
        assert_eq!(crc, masked_crc32c(data));
    }

    #[test]
    fn test_write_and_read_record() {
        let data = b"hello world";
        let mut buffer = Vec::new();

        // Write
        {
            let mut writer = TFRecordWriter::new(&mut buffer);
            writer.write_record(data).unwrap();
        }

        // Read
        {
            let mut reader = TFRecordReader::new(Cursor::new(&buffer), true);
            let read_data = reader.read_record().unwrap().unwrap();
            assert_eq!(read_data.as_ref(), data);
        }
    }

    #[test]
    fn test_write_and_read_example() {
        let mut example = create_example();
        add_feature(&mut example, "feature1", vec![1, 2, 3], vec![1.0, 2.0, 3.0]);
        add_feature(&mut example, "feature2", vec![4, 5], vec![4.0, 5.0]);

        let mut buffer = Vec::new();

        // Write
        {
            let mut writer = TFRecordWriter::new(&mut buffer);
            writer.write_example(&example).unwrap();
        }

        // Read
        {
            let mut reader = TFRecordReader::new(Cursor::new(&buffer), true);
            let read_example = reader.read_example().unwrap().unwrap();
            assert_eq!(read_example, example);
        }
    }

    #[test]
    fn test_write_and_read_multiple_examples() {
        let examples: Vec<Example> = (0..5)
            .map(|i| {
                let mut ex = create_example();
                add_feature(&mut ex, "id", vec![i], vec![i as f32]);
                ex
            })
            .collect();

        let mut buffer = Vec::new();

        // Write all examples
        {
            let mut writer = TFRecordWriter::new(&mut buffer);
            for example in &examples {
                writer.write_example(example).unwrap();
            }
        }

        // Read all examples
        {
            let mut reader = TFRecordReader::new(Cursor::new(&buffer), true);
            for expected in &examples {
                let read_example = reader.read_example().unwrap().unwrap();
                assert_eq!(&read_example, expected);
            }
            // Should be end of file
            assert!(reader.read_example().unwrap().is_none());
        }
    }

    #[test]
    fn test_read_empty() {
        let buffer: Vec<u8> = Vec::new();
        let mut reader = TFRecordReader::new(Cursor::new(&buffer), true);
        assert!(reader.read_record().unwrap().is_none());
    }

    #[test]
    fn test_crc_verification_failure() {
        let data = b"test data";
        let mut buffer = Vec::new();

        // Write a valid record
        {
            let mut writer = TFRecordWriter::new(&mut buffer);
            writer.write_record(data).unwrap();
        }

        // Corrupt the data CRC (last 4 bytes)
        let len = buffer.len();
        buffer[len - 1] ^= 0xff;

        // Read should fail with CRC mismatch
        let mut reader = TFRecordReader::new(Cursor::new(&buffer), true);
        let result = reader.read_record();
        assert!(matches!(result, Err(TFRecordError::CrcMismatch { .. })));
    }

    #[test]
    fn test_crc_verification_disabled() {
        let data = b"test data";
        let mut buffer = Vec::new();

        // Write a valid record
        {
            let mut writer = TFRecordWriter::new(&mut buffer);
            writer.write_record(data).unwrap();
        }

        // Corrupt the data CRC (last 4 bytes)
        let len = buffer.len();
        buffer[len - 1] ^= 0xff;

        // Read with CRC verification disabled should succeed
        let mut reader = TFRecordReader::new(Cursor::new(&buffer), false);
        let result = reader.read_record().unwrap().unwrap();
        assert_eq!(result.as_ref(), data);
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn test_write_and_read_record_gzip() {
        let data = b"hello world, this is gzip compressed data";
        let mut buffer = Vec::new();

        // Write with gzip compression
        {
            let mut writer =
                TFRecordWriter::new(&mut buffer).with_compression(CompressionType::Gzip);
            writer.write_record(data).unwrap();
        }

        // Read with gzip decompression
        {
            let mut reader = TFRecordReader::new(Cursor::new(&buffer), true)
                .with_compression(CompressionType::Gzip);
            let read_data = reader.read_record().unwrap().unwrap();
            assert_eq!(read_data.as_ref(), data);
        }
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn test_write_and_read_example_gzip() {
        let mut example = create_example();
        add_feature(&mut example, "feature1", vec![1, 2, 3], vec![1.0, 2.0, 3.0]);
        add_feature(&mut example, "feature2", vec![4, 5], vec![4.0, 5.0]);

        let mut buffer = Vec::new();

        // Write with gzip compression
        {
            let mut writer =
                TFRecordWriter::new(&mut buffer).with_compression(CompressionType::Gzip);
            writer.write_example(&example).unwrap();
        }

        // Read with gzip decompression
        {
            let mut reader = TFRecordReader::new(Cursor::new(&buffer), true)
                .with_compression(CompressionType::Gzip);
            let read_example = reader.read_example().unwrap().unwrap();
            assert_eq!(read_example, example);
        }
    }

    #[cfg(feature = "gzip")]
    #[test]
    fn test_write_and_read_record_zlib() {
        let data = b"hello world, this is zlib compressed data";
        let mut buffer = Vec::new();

        // Write with zlib compression
        {
            let mut writer =
                TFRecordWriter::new(&mut buffer).with_compression(CompressionType::Zlib);
            writer.write_record(data).unwrap();
        }

        // Read with zlib decompression
        {
            let mut reader = TFRecordReader::new(Cursor::new(&buffer), true)
                .with_compression(CompressionType::Zlib);
            let read_data = reader.read_record().unwrap().unwrap();
            assert_eq!(read_data.as_ref(), data);
        }
    }

    #[cfg(feature = "snappy")]
    #[test]
    fn test_write_and_read_record_snappy() {
        let data = b"hello world, this is snappy compressed data";
        let mut buffer = Vec::new();

        // Write with snappy compression
        {
            let mut writer =
                TFRecordWriter::new(&mut buffer).with_compression(CompressionType::Snappy);
            writer.write_record(data).unwrap();
        }

        // Read with snappy decompression
        {
            let mut reader = TFRecordReader::new(Cursor::new(&buffer), true)
                .with_compression(CompressionType::Snappy);
            let read_data = reader.read_record().unwrap().unwrap();
            assert_eq!(read_data.as_ref(), data);
        }
    }

    #[cfg(feature = "compression")]
    #[test]
    fn test_write_and_read_multiple_examples_compressed() {
        let examples: Vec<Example> = (0..5)
            .map(|i| {
                let mut ex = create_example();
                add_feature(&mut ex, "id", vec![i], vec![i as f32]);
                ex
            })
            .collect();

        for compression in [
            CompressionType::None,
            CompressionType::Gzip,
            CompressionType::Zlib,
            CompressionType::Snappy,
        ] {
            let mut buffer = Vec::new();

            // Write all examples with compression
            {
                let mut writer = TFRecordWriter::new(&mut buffer).with_compression(compression);
                for example in &examples {
                    writer.write_example(example).unwrap();
                }
            }

            // Read all examples with decompression
            {
                let mut reader =
                    TFRecordReader::new(Cursor::new(&buffer), true).with_compression(compression);
                for expected in &examples {
                    let read_example = reader.read_example().unwrap().unwrap();
                    assert_eq!(&read_example, expected);
                }
                // Should be end of file
                assert!(reader.read_example().unwrap().is_none());
            }
        }
    }

    #[test]
    fn test_reader_writer_compression_getters() {
        let buffer: Vec<u8> = Vec::new();

        let reader: TFRecordReader<Cursor<&Vec<u8>>> =
            TFRecordReader::new(Cursor::new(&buffer), true);
        assert_eq!(reader.compression(), CompressionType::None);

        let reader = reader.with_compression(CompressionType::Gzip);
        assert_eq!(reader.compression(), CompressionType::Gzip);

        let mut out_buffer = Vec::new();
        let writer: TFRecordWriter<&mut Vec<u8>> = TFRecordWriter::new(&mut out_buffer);
        assert_eq!(writer.compression(), CompressionType::None);
    }

    #[test]
    fn test_dataset_from_pattern_glob_and_csv() {
        let tmp = tempdir().unwrap();
        let p1 = tmp.path().join("a.tfrecord");
        let p2 = tmp.path().join("b.tfrecord");
        std::fs::write(&p1, b"").unwrap();
        std::fs::write(&p2, b"").unwrap();

        let pattern = format!("{}/{}.tfrecord,{}", tmp.path().display(), "*", p1.display());
        let ds = TFRecordDataset::from_pattern(&pattern).unwrap();
        assert_eq!(ds.file_count(), 2);
        assert!(ds.paths().contains(&p1));
        assert!(ds.paths().contains(&p2));
    }

    #[test]
    fn test_dataset_from_pattern_at_filelist() {
        let tmp = tempdir().unwrap();
        let p1 = tmp.path().join("x.tfrecord");
        let p2 = tmp.path().join("y.tfrecord");
        std::fs::write(&p1, b"").unwrap();
        std::fs::write(&p2, b"").unwrap();

        let filelist = tmp.path().join("files.txt");
        let content = format!(
            "{}\n{}\n",
            p1.display(),
            tmp.path().join("*.tfrecord").display()
        );
        std::fs::write(&filelist, content).unwrap();

        let spec = format!("@{}", filelist.display());
        let ds = TFRecordDataset::from_pattern(&spec).unwrap();
        assert_eq!(ds.file_count(), 2);
        assert!(ds.paths().contains(&p1));
        assert!(ds.paths().contains(&p2));
    }

    #[test]
    fn test_dataset_from_pattern_not_found() {
        let tmp = tempdir().unwrap();
        let spec = format!("{}/no-match-*.tfrecord", tmp.path().display());
        let result = TFRecordDataset::from_pattern(&spec);
        assert!(matches!(result, Err(TFRecordError::Io(_))));
    }
}
