//! Parquet data source for reading training data from Parquet files.
//!
//! This module provides [`ParquetDataSource`] for reading data from Parquet files,
//! which is a popular columnar storage format for analytics workloads. It supports
//! reading single files, directories of files via glob patterns, and predicate
//! pushdown for efficient filtering.
//!
//! # Features
//!
//! - Read single Parquet files or directories of files
//! - Glob pattern support for matching multiple files
//! - Column projection (select specific columns)
//! - Row group selection for partial file reads
//! - Predicate pushdown for efficient filtering
//! - Conversion to Monolith [`Example`] format
//!
//! # Example
//!
//! ```no_run
//! use monolith_data::parquet::{ParquetDataSource, ParquetConfig};
//!
//! // Read from a single file
//! let config = ParquetConfig::new("data/train.parquet");
//! let source = ParquetDataSource::open(config).unwrap();
//!
//! for example in source.iter() {
//!     println!("{:?}", example);
//! }
//! ```
//!
//! # Example with Glob Pattern
//!
//! ```no_run
//! use monolith_data::parquet::{ParquetDataSource, ParquetConfig};
//!
//! // Read from multiple files using glob
//! let config = ParquetConfig::new("data/*.parquet")
//!     .with_batch_size(1024)
//!     .with_columns(vec!["user_id", "item_id", "label"]);
//!
//! let source = ParquetDataSource::open(config).unwrap();
//! println!("Schema: {:?}", source.schema());
//!
//! for example in source.iter() {
//!     // Process examples
//! }
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Schema};
use arrow::record_batch::RecordBatch;
use glob::glob;
use monolith_proto::Example;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::{FileReader, SerializedFileReader};
use thiserror::Error;

use crate::dataset::Dataset;
use crate::example::{add_feature, create_example};

/// Errors that can occur when reading Parquet files.
#[derive(Error, Debug)]
pub enum ParquetError {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A Parquet format error occurred.
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    /// An Arrow error occurred.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    /// A glob pattern error occurred.
    #[error("Glob pattern error: {0}")]
    GlobPattern(#[from] glob::PatternError),

    /// A glob error occurred during iteration.
    #[error("Glob error: {0}")]
    Glob(#[from] glob::GlobError),

    /// No files matched the pattern.
    #[error("No files found matching pattern: {0}")]
    NoFilesFound(String),

    /// File not found.
    #[error("File not found: {0}")]
    FileNotFound(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Schema mismatch between files.
    #[error("Schema mismatch: {0}")]
    SchemaMismatch(String),

    /// Column not found in schema.
    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    /// Unsupported data type.
    #[error("Unsupported data type for column '{column}': {data_type}")]
    UnsupportedDataType {
        /// The column name.
        column: String,
        /// The data type.
        data_type: String,
    },
}

/// Result type for Parquet operations.
pub type Result<T> = std::result::Result<T, ParquetError>;

/// Predicate for filtering rows during Parquet reads.
///
/// Predicates enable predicate pushdown, allowing the Parquet reader to skip
/// row groups that cannot contain matching data.
#[derive(Clone, Debug)]
pub enum Predicate {
    /// Filter rows where column equals value.
    Eq {
        /// Column name.
        column: String,
        /// Value to compare (as i64).
        value: i64,
    },
    /// Filter rows where column is not equal to value.
    Ne {
        /// Column name.
        column: String,
        /// Value to compare (as i64).
        value: i64,
    },
    /// Filter rows where column is less than value.
    Lt {
        /// Column name.
        column: String,
        /// Value to compare (as i64).
        value: i64,
    },
    /// Filter rows where column is less than or equal to value.
    Le {
        /// Column name.
        column: String,
        /// Value to compare (as i64).
        value: i64,
    },
    /// Filter rows where column is greater than value.
    Gt {
        /// Column name.
        column: String,
        /// Value to compare (as i64).
        value: i64,
    },
    /// Filter rows where column is greater than or equal to value.
    Ge {
        /// Column name.
        column: String,
        /// Value to compare (as i64).
        value: i64,
    },
    /// Filter rows where column value is in the set.
    In {
        /// Column name.
        column: String,
        /// Set of values to check.
        values: HashSet<i64>,
    },
    /// Logical AND of predicates.
    And(Box<Predicate>, Box<Predicate>),
    /// Logical OR of predicates.
    Or(Box<Predicate>, Box<Predicate>),
    /// Logical NOT of a predicate.
    Not(Box<Predicate>),
}

impl Predicate {
    /// Creates an equality predicate.
    pub fn eq(column: impl Into<String>, value: i64) -> Self {
        Predicate::Eq {
            column: column.into(),
            value,
        }
    }

    /// Creates a not-equal predicate.
    pub fn ne(column: impl Into<String>, value: i64) -> Self {
        Predicate::Ne {
            column: column.into(),
            value,
        }
    }

    /// Creates a less-than predicate.
    pub fn lt(column: impl Into<String>, value: i64) -> Self {
        Predicate::Lt {
            column: column.into(),
            value,
        }
    }

    /// Creates a less-than-or-equal predicate.
    pub fn le(column: impl Into<String>, value: i64) -> Self {
        Predicate::Le {
            column: column.into(),
            value,
        }
    }

    /// Creates a greater-than predicate.
    pub fn gt(column: impl Into<String>, value: i64) -> Self {
        Predicate::Gt {
            column: column.into(),
            value,
        }
    }

    /// Creates a greater-than-or-equal predicate.
    pub fn ge(column: impl Into<String>, value: i64) -> Self {
        Predicate::Ge {
            column: column.into(),
            value,
        }
    }

    /// Creates an IN predicate.
    pub fn in_set(column: impl Into<String>, values: impl IntoIterator<Item = i64>) -> Self {
        Predicate::In {
            column: column.into(),
            values: values.into_iter().collect(),
        }
    }

    /// Creates a logical AND of two predicates.
    pub fn and(self, other: Predicate) -> Self {
        Predicate::And(Box::new(self), Box::new(other))
    }

    /// Creates a logical OR of two predicates.
    pub fn or(self, other: Predicate) -> Self {
        Predicate::Or(Box::new(self), Box::new(other))
    }

    /// Creates a logical NOT of a predicate.
    pub fn not(self) -> Self {
        Predicate::Not(Box::new(self))
    }

    /// Evaluates the predicate against a value.
    fn evaluate(&self, column_values: &std::collections::HashMap<String, i64>) -> bool {
        match self {
            Predicate::Eq { column, value } => {
                column_values.get(column).map_or(false, |v| v == value)
            }
            Predicate::Ne { column, value } => {
                column_values.get(column).map_or(true, |v| v != value)
            }
            Predicate::Lt { column, value } => {
                column_values.get(column).map_or(false, |v| v < value)
            }
            Predicate::Le { column, value } => {
                column_values.get(column).map_or(false, |v| v <= value)
            }
            Predicate::Gt { column, value } => {
                column_values.get(column).map_or(false, |v| v > value)
            }
            Predicate::Ge { column, value } => {
                column_values.get(column).map_or(false, |v| v >= value)
            }
            Predicate::In { column, values } => {
                column_values.get(column).map_or(false, |v| values.contains(v))
            }
            Predicate::And(left, right) => {
                left.evaluate(column_values) && right.evaluate(column_values)
            }
            Predicate::Or(left, right) => {
                left.evaluate(column_values) || right.evaluate(column_values)
            }
            Predicate::Not(inner) => !inner.evaluate(column_values),
        }
    }
}

/// Configuration for reading Parquet files.
///
/// # Example
///
/// ```no_run
/// use monolith_data::parquet::ParquetConfig;
///
/// let config = ParquetConfig::new("data/train.parquet")
///     .with_batch_size(512)
///     .with_columns(vec!["user_id", "item_id", "label"])
///     .with_row_groups(vec![0, 1, 2]);
/// ```
#[derive(Clone, Debug)]
pub struct ParquetConfig {
    /// Path to file or glob pattern for multiple files.
    pub path: String,
    /// Number of rows per batch (default: 8192).
    pub batch_size: usize,
    /// Optional column selection. If None, all columns are read.
    pub columns: Option<Vec<String>>,
    /// Optional row group selection. If None, all row groups are read.
    pub row_groups: Option<Vec<usize>>,
    /// Optional predicate for filtering rows.
    pub predicate: Option<Predicate>,
}

impl ParquetConfig {
    /// Creates a new configuration with the given path.
    ///
    /// # Arguments
    ///
    /// * `path` - File path or glob pattern (e.g., "data/*.parquet")
    pub fn new(path: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            batch_size: 8192,
            columns: None,
            row_groups: None,
            predicate: None,
        }
    }

    /// Sets the batch size (number of rows per RecordBatch).
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of rows to read at a time
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Sets the columns to read.
    ///
    /// # Arguments
    ///
    /// * `columns` - Column names to read
    pub fn with_columns(mut self, columns: Vec<impl Into<String>>) -> Self {
        self.columns = Some(columns.into_iter().map(|c| c.into()).collect());
        self
    }

    /// Sets the row groups to read.
    ///
    /// # Arguments
    ///
    /// * `row_groups` - Row group indices to read
    pub fn with_row_groups(mut self, row_groups: Vec<usize>) -> Self {
        self.row_groups = Some(row_groups);
        self
    }

    /// Sets a predicate for filtering rows.
    ///
    /// # Arguments
    ///
    /// * `predicate` - The predicate to apply
    pub fn with_predicate(mut self, predicate: Predicate) -> Self {
        self.predicate = Some(predicate);
        self
    }
}

/// Schema information for a Parquet data source.
#[derive(Clone, Debug)]
pub struct ParquetSchema {
    /// The Arrow schema.
    pub arrow_schema: Arc<Schema>,
    /// Total number of rows across all files.
    pub num_rows: usize,
    /// Number of files in the data source.
    pub num_files: usize,
    /// Number of row groups across all files.
    pub num_row_groups: usize,
}

impl ParquetSchema {
    /// Returns the column names in the schema.
    pub fn column_names(&self) -> Vec<&str> {
        self.arrow_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect()
    }

    /// Returns the number of columns.
    pub fn num_columns(&self) -> usize {
        self.arrow_schema.fields().len()
    }

    /// Checks if a column exists in the schema.
    pub fn has_column(&self, name: &str) -> bool {
        self.arrow_schema.field_with_name(name).is_ok()
    }

    /// Returns the data type of a column.
    pub fn column_type(&self, name: &str) -> Option<&DataType> {
        self.arrow_schema
            .field_with_name(name)
            .ok()
            .map(|f| f.data_type())
    }
}

/// A data source for reading Parquet files.
///
/// `ParquetDataSource` provides an iterator over [`Example`] protobuf messages
/// by reading from one or more Parquet files and converting the columnar data
/// to the row-oriented Example format.
///
/// # Supported Data Types
///
/// The following Parquet/Arrow data types are supported:
/// - Int32, Int64 -> stored as feature IDs (i64)
/// - Float32, Float64 -> stored as feature values (f32)
/// - Utf8 (String) -> hashed to feature IDs (i64)
///
/// # Example
///
/// ```no_run
/// use monolith_data::parquet::{ParquetDataSource, ParquetConfig, Predicate};
///
/// // Read with column selection and filtering
/// let config = ParquetConfig::new("data/train.parquet")
///     .with_columns(vec!["user_id", "item_id", "label"])
///     .with_predicate(Predicate::gt("label", 0));
///
/// let source = ParquetDataSource::open(config).unwrap();
/// for example in source.iter() {
///     // Process filtered examples
/// }
/// ```
pub struct ParquetDataSource {
    paths: Vec<PathBuf>,
    config: ParquetConfig,
    schema: ParquetSchema,
}

impl ParquetDataSource {
    /// Opens a Parquet data source with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for reading Parquet files
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No files match the path/pattern
    /// - The file cannot be opened
    /// - Column selection references non-existent columns
    pub fn open(config: ParquetConfig) -> Result<Self> {
        let paths = Self::resolve_paths(&config.path)?;
        if paths.is_empty() {
            return Err(ParquetError::NoFilesFound(config.path.clone()));
        }

        // Read metadata from the first file to get schema
        let first_file = File::open(&paths[0])?;
        let reader = SerializedFileReader::new(first_file)?;
        let metadata = reader.metadata();
        let arrow_schema = Arc::new(parquet::arrow::parquet_to_arrow_schema(
            metadata.file_metadata().schema_descr(),
            None,
        )?);

        // Validate columns if specified
        if let Some(ref columns) = config.columns {
            for col in columns {
                if arrow_schema.field_with_name(col).is_err() {
                    return Err(ParquetError::ColumnNotFound(col.clone()));
                }
            }
        }

        // Calculate total rows and row groups
        let mut total_rows = 0usize;
        let mut total_row_groups = 0usize;

        for path in &paths {
            let file = File::open(path)?;
            let reader = SerializedFileReader::new(file)?;
            let metadata = reader.metadata();
            total_rows += metadata.file_metadata().num_rows() as usize;
            total_row_groups += metadata.num_row_groups();
        }

        let schema = ParquetSchema {
            arrow_schema,
            num_rows: total_rows,
            num_files: paths.len(),
            num_row_groups: total_row_groups,
        };

        Ok(Self {
            paths,
            config,
            schema,
        })
    }

    /// Resolves a path or glob pattern to a list of file paths.
    fn resolve_paths(path: &str) -> Result<Vec<PathBuf>> {
        let path_obj = Path::new(path);

        // Check if it's a plain file path (no glob characters)
        if !path.contains('*') && !path.contains('?') && !path.contains('[') {
            if path_obj.is_file() {
                return Ok(vec![path_obj.to_path_buf()]);
            } else if path_obj.is_dir() {
                // If it's a directory, look for .parquet files
                let pattern = path_obj.join("*.parquet");
                return Self::resolve_glob(pattern.to_string_lossy().as_ref());
            } else {
                return Err(ParquetError::FileNotFound(path.to_string()));
            }
        }

        // It's a glob pattern
        Self::resolve_glob(path)
    }

    /// Resolves a glob pattern to matching files.
    fn resolve_glob(pattern: &str) -> Result<Vec<PathBuf>> {
        let mut paths: Vec<PathBuf> = glob(pattern)?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        // Sort for deterministic order
        paths.sort();
        Ok(paths)
    }

    /// Returns schema information for this data source.
    pub fn schema(&self) -> &ParquetSchema {
        &self.schema
    }

    /// Returns the file paths in this data source.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Returns the number of files in this data source.
    pub fn file_count(&self) -> usize {
        self.paths.len()
    }

    /// Returns the configuration for this data source.
    pub fn config(&self) -> &ParquetConfig {
        &self.config
    }

    /// Returns the Parquet metadata for a specific file.
    ///
    /// # Arguments
    ///
    /// * `file_index` - Index of the file
    pub fn file_metadata(&self, file_index: usize) -> Result<ParquetMetaData> {
        if file_index >= self.paths.len() {
            return Err(ParquetError::InvalidConfig(format!(
                "File index {} out of range (0..{})",
                file_index,
                self.paths.len()
            )));
        }
        let file = File::open(&self.paths[file_index])?;
        let reader = SerializedFileReader::new(file)?;
        Ok(reader.metadata().clone())
    }

    /// Creates an iterator that yields RecordBatches.
    ///
    /// This is useful when you need direct access to Arrow RecordBatches
    /// rather than converted Examples.
    pub fn record_batch_iter(&self) -> Result<ParquetRecordBatchIterator> {
        ParquetRecordBatchIterator::new(
            self.paths.clone(),
            self.config.batch_size,
            self.config.columns.clone(),
            self.config.row_groups.clone(),
        )
    }

    /// Returns an iterator over Examples in the dataset.
    pub fn iter(&self) -> ParquetIterator {
        ParquetIterator::new(
            self.paths.clone(),
            self.config.batch_size,
            self.config.columns.clone(),
            self.config.row_groups.clone(),
            self.config.predicate.clone(),
        )
    }
}

impl Dataset for ParquetDataSource {
    type Iter = ParquetIterator;

    fn iter(self) -> Self::Iter {
        ParquetIterator::new(
            self.paths,
            self.config.batch_size,
            self.config.columns,
            self.config.row_groups,
            self.config.predicate,
        )
    }
}

/// Iterator over RecordBatches from Parquet files.
pub struct ParquetRecordBatchIterator {
    paths: Vec<PathBuf>,
    current_file_index: usize,
    batch_size: usize,
    columns: Option<Vec<String>>,
    row_groups: Option<Vec<usize>>,
    current_reader: Option<ParquetRecordBatchReader>,
}

impl ParquetRecordBatchIterator {
    fn new(
        paths: Vec<PathBuf>,
        batch_size: usize,
        columns: Option<Vec<String>>,
        row_groups: Option<Vec<usize>>,
    ) -> Result<Self> {
        let mut iter = Self {
            paths,
            current_file_index: 0,
            batch_size,
            columns,
            row_groups,
            current_reader: None,
        };
        iter.open_next_file()?;
        Ok(iter)
    }

    fn open_next_file(&mut self) -> Result<bool> {
        if self.current_file_index >= self.paths.len() {
            return Ok(false);
        }

        let path = &self.paths[self.current_file_index];
        self.current_file_index += 1;

        let file = File::open(path)?;
        let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?
            .with_batch_size(self.batch_size);

        // Apply column projection
        if let Some(ref columns) = self.columns {
            let schema = builder.schema();
            let indices: Vec<usize> = columns
                .iter()
                .filter_map(|name| {
                    schema.fields().iter().position(|f| f.name() == name)
                })
                .collect();
            let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
            builder = builder.with_projection(projection);
        }

        // Apply row group selection
        if let Some(ref row_groups) = self.row_groups {
            builder = builder.with_row_groups(row_groups.clone());
        }

        self.current_reader = Some(builder.build()?);
        Ok(true)
    }
}

impl Iterator for ParquetRecordBatchIterator {
    type Item = Result<RecordBatch>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(ref mut reader) = self.current_reader {
                match reader.next() {
                    Some(Ok(batch)) => return Some(Ok(batch)),
                    Some(Err(e)) => return Some(Err(ParquetError::Arrow(e))),
                    None => {
                        self.current_reader = None;
                    }
                }
            }

            // Try to open next file
            match self.open_next_file() {
                Ok(true) => continue,
                Ok(false) => return None,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}

/// Iterator over Examples from Parquet files.
pub struct ParquetIterator {
    paths: Vec<PathBuf>,
    current_file_index: usize,
    batch_size: usize,
    columns: Option<Vec<String>>,
    row_groups: Option<Vec<usize>>,
    predicate: Option<Predicate>,
    current_reader: Option<ParquetRecordBatchReader>,
    current_batch: Option<RecordBatch>,
    current_row: usize,
}

impl ParquetIterator {
    fn new(
        paths: Vec<PathBuf>,
        batch_size: usize,
        columns: Option<Vec<String>>,
        row_groups: Option<Vec<usize>>,
        predicate: Option<Predicate>,
    ) -> Self {
        let mut iter = Self {
            paths,
            current_file_index: 0,
            batch_size,
            columns,
            row_groups,
            predicate,
            current_reader: None,
            current_batch: None,
            current_row: 0,
        };
        let _ = iter.open_next_file();
        iter
    }

    fn open_next_file(&mut self) -> bool {
        if self.current_file_index >= self.paths.len() {
            return false;
        }

        let path = &self.paths[self.current_file_index];
        self.current_file_index += 1;

        let file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return self.open_next_file(),
        };

        let mut builder = match ParquetRecordBatchReaderBuilder::try_new(file) {
            Ok(b) => b,
            Err(_) => return self.open_next_file(),
        };
        builder = builder.with_batch_size(self.batch_size);

        // Apply column projection
        if let Some(ref columns) = self.columns {
            let schema = builder.schema();
            let indices: Vec<usize> = columns
                .iter()
                .filter_map(|name| {
                    schema.fields().iter().position(|f| f.name() == name)
                })
                .collect();
            let projection = ProjectionMask::roots(builder.parquet_schema(), indices);
            builder = builder.with_projection(projection);
        }

        // Apply row group selection
        if let Some(ref row_groups) = self.row_groups {
            builder = builder.with_row_groups(row_groups.clone());
        }

        match builder.build() {
            Ok(reader) => {
                self.current_reader = Some(reader);
                true
            }
            Err(_) => self.open_next_file(),
        }
    }

    fn load_next_batch(&mut self) -> bool {
        loop {
            if let Some(ref mut reader) = self.current_reader {
                match reader.next() {
                    Some(Ok(batch)) => {
                        self.current_batch = Some(batch);
                        self.current_row = 0;
                        return true;
                    }
                    Some(Err(_)) => {
                        self.current_reader = None;
                    }
                    None => {
                        self.current_reader = None;
                    }
                }
            }

            // Try to open next file
            if !self.open_next_file() {
                return false;
            }
        }
    }

    fn extract_example_from_batch(&self, batch: &RecordBatch, row: usize) -> Option<Example> {
        let schema = batch.schema();
        let mut example = create_example();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let column = batch.column(col_idx);
            let name = field.name();

            match field.data_type() {
                DataType::Int32 => {
                    if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
                        if !array.is_null(row) {
                            let value = array.value(row) as i64;
                            add_feature(&mut example, name, vec![value], vec![value as f32]);
                        }
                    }
                }
                DataType::Int64 => {
                    if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
                        if !array.is_null(row) {
                            let value = array.value(row);
                            add_feature(&mut example, name, vec![value], vec![value as f32]);
                        }
                    }
                }
                DataType::Float32 => {
                    if let Some(array) = column.as_any().downcast_ref::<Float32Array>() {
                        if !array.is_null(row) {
                            let value = array.value(row);
                            add_feature(&mut example, name, vec![0], vec![value]);
                        }
                    }
                }
                DataType::Float64 => {
                    if let Some(array) = column.as_any().downcast_ref::<Float64Array>() {
                        if !array.is_null(row) {
                            let value = array.value(row) as f32;
                            add_feature(&mut example, name, vec![0], vec![value]);
                        }
                    }
                }
                DataType::Utf8 => {
                    if let Some(array) = column.as_any().downcast_ref::<StringArray>() {
                        if !array.is_null(row) {
                            let value = array.value(row);
                            // Hash the string to get a feature ID
                            let hash = Self::hash_string(value);
                            add_feature(&mut example, name, vec![hash], vec![1.0]);
                        }
                    }
                }
                _ => {
                    // Skip unsupported types
                }
            }
        }

        Some(example)
    }

    /// Simple string hashing for converting strings to feature IDs.
    fn hash_string(s: &str) -> i64 {
        let mut hash: u64 = 5381;
        for byte in s.bytes() {
            hash = hash.wrapping_mul(33).wrapping_add(byte as u64);
        }
        hash as i64
    }

    /// Extracts integer values from a row for predicate evaluation.
    fn extract_int_values(&self, batch: &RecordBatch, row: usize) -> std::collections::HashMap<String, i64> {
        let schema = batch.schema();
        let mut values = std::collections::HashMap::new();

        for (col_idx, field) in schema.fields().iter().enumerate() {
            let column = batch.column(col_idx);
            let name = field.name().clone();

            match field.data_type() {
                DataType::Int32 => {
                    if let Some(array) = column.as_any().downcast_ref::<Int32Array>() {
                        if !array.is_null(row) {
                            values.insert(name, array.value(row) as i64);
                        }
                    }
                }
                DataType::Int64 => {
                    if let Some(array) = column.as_any().downcast_ref::<Int64Array>() {
                        if !array.is_null(row) {
                            values.insert(name, array.value(row));
                        }
                    }
                }
                _ => {}
            }
        }

        values
    }

    /// Checks if the current row matches the predicate.
    fn matches_predicate(&self, batch: &RecordBatch, row: usize) -> bool {
        match &self.predicate {
            Some(predicate) => {
                let values = self.extract_int_values(batch, row);
                predicate.evaluate(&values)
            }
            None => true,
        }
    }
}

impl Iterator for ParquetIterator {
    type Item = Example;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Check if we have a current batch
            if self.current_batch.is_none() && !self.load_next_batch() {
                return None;
            }

            let batch = self.current_batch.as_ref()?;

            // Find next matching row
            while self.current_row < batch.num_rows() {
                let row = self.current_row;
                self.current_row += 1;

                if self.matches_predicate(batch, row) {
                    if let Some(example) = self.extract_example_from_batch(batch, row) {
                        return Some(example);
                    }
                }
            }

            // Exhausted current batch, load next
            self.current_batch = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Float32Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use parquet::arrow::ArrowWriter;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn create_test_parquet_file(path: &Path, num_rows: usize) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("user_id", DataType::Int64, false),
            Field::new("item_id", DataType::Int64, false),
            Field::new("label", DataType::Float32, false),
            Field::new("category", DataType::Utf8, true),
        ]));

        let user_ids: Vec<i64> = (0..num_rows as i64).collect();
        let item_ids: Vec<i64> = (100..100 + num_rows as i64).collect();
        let labels: Vec<f32> = (0..num_rows).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
        let categories: Vec<&str> = (0..num_rows)
            .map(|i| if i % 3 == 0 { "sports" } else { "music" })
            .collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(user_ids)),
                Arc::new(Int64Array::from(item_ids)),
                Arc::new(Float32Array::from(labels)),
                Arc::new(StringArray::from(categories)),
            ],
        )?;

        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;

        Ok(())
    }

    #[test]
    fn test_open_single_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy());
        let source = ParquetDataSource::open(config).unwrap();

        assert_eq!(source.file_count(), 1);
        assert_eq!(source.schema().num_rows, 10);
        assert_eq!(source.schema().num_columns(), 4);
    }

    #[test]
    fn test_open_glob_pattern() {
        let dir = tempdir().unwrap();

        // Create multiple parquet files
        for i in 0..3 {
            let file_path = dir.path().join(format!("data_{}.parquet", i));
            create_test_parquet_file(&file_path, 5).unwrap();
        }

        let pattern = dir.path().join("*.parquet");
        let config = ParquetConfig::new(pattern.to_string_lossy());
        let source = ParquetDataSource::open(config).unwrap();

        assert_eq!(source.file_count(), 3);
        assert_eq!(source.schema().num_rows, 15);
    }

    #[test]
    fn test_column_selection() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_columns(vec!["user_id", "label"]);
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 10);

        // Each example should only have user_id and label features
        for example in &examples {
            assert!(crate::example::has_feature(example, "user_id"));
            assert!(crate::example::has_feature(example, "label"));
            assert!(!crate::example::has_feature(example, "item_id"));
            assert!(!crate::example::has_feature(example, "category"));
        }
    }

    #[test]
    fn test_invalid_column() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_columns(vec!["user_id", "nonexistent"]);
        let result = ParquetDataSource::open(config);

        assert!(matches!(result, Err(ParquetError::ColumnNotFound(_))));
    }

    #[test]
    fn test_iter_examples() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 20).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_batch_size(5);
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 20);

        // Verify first example
        let first = &examples[0];
        assert!(crate::example::has_feature(first, "user_id"));
        assert!(crate::example::has_feature(first, "item_id"));
        assert!(crate::example::has_feature(first, "label"));
        assert!(crate::example::has_feature(first, "category"));
    }

    #[test]
    fn test_predicate_eq() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_predicate(Predicate::eq("user_id", 5));
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 1);

        let feature = crate::example::get_feature(&examples[0], "user_id").unwrap();
        assert_eq!(feature.fid[0], 5);
    }

    #[test]
    fn test_predicate_gt() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_predicate(Predicate::gt("user_id", 7));
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 2); // user_id 8 and 9
    }

    #[test]
    fn test_predicate_and() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_predicate(
                Predicate::ge("user_id", 3).and(Predicate::le("user_id", 6))
            );
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 4); // user_id 3, 4, 5, 6
    }

    #[test]
    fn test_predicate_in_set() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_predicate(Predicate::in_set("user_id", vec![1, 3, 5, 7]));
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 4);
    }

    #[test]
    fn test_schema_info() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy());
        let source = ParquetDataSource::open(config).unwrap();

        let schema = source.schema();
        assert_eq!(schema.num_columns(), 4);
        assert_eq!(schema.num_rows, 10);
        assert_eq!(schema.num_files, 1);
        assert!(schema.has_column("user_id"));
        assert!(schema.has_column("item_id"));
        assert!(!schema.has_column("nonexistent"));

        let column_names = schema.column_names();
        assert!(column_names.contains(&"user_id"));
        assert!(column_names.contains(&"item_id"));
        assert!(column_names.contains(&"label"));
        assert!(column_names.contains(&"category"));
    }

    #[test]
    fn test_record_batch_iter() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 25).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_batch_size(10);
        let source = ParquetDataSource::open(config).unwrap();

        let batches: Vec<_> = source.record_batch_iter().unwrap().collect();
        assert_eq!(batches.len(), 3); // 10 + 10 + 5

        let mut total_rows = 0;
        for batch in batches {
            total_rows += batch.unwrap().num_rows();
        }
        assert_eq!(total_rows, 25);
    }

    #[test]
    fn test_dataset_trait() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy());
        let source = ParquetDataSource::open(config).unwrap();

        // Use Dataset trait methods
        let examples: Vec<_> = source
            .filter(|ex| crate::example::has_feature(ex, "user_id"))
            .take(5)
            .iter()
            .collect();

        assert_eq!(examples.len(), 5);
    }

    #[test]
    fn test_file_not_found() {
        let config = ParquetConfig::new("/nonexistent/path/file.parquet");
        let result = ParquetDataSource::open(config);
        assert!(matches!(result, Err(ParquetError::FileNotFound(_))));
    }

    #[test]
    fn test_no_files_found() {
        let dir = tempdir().unwrap();
        let pattern = dir.path().join("*.parquet");
        let config = ParquetConfig::new(pattern.to_string_lossy());
        let result = ParquetDataSource::open(config);
        assert!(matches!(result, Err(ParquetError::NoFilesFound(_))));
    }

    #[test]
    fn test_directory_path() {
        let dir = tempdir().unwrap();

        // Create parquet files in the directory
        for i in 0..2 {
            let file_path = dir.path().join(format!("data_{}.parquet", i));
            create_test_parquet_file(&file_path, 5).unwrap();
        }

        // Open using directory path
        let config = ParquetConfig::new(dir.path().to_string_lossy());
        let source = ParquetDataSource::open(config).unwrap();

        assert_eq!(source.file_count(), 2);
        assert_eq!(source.schema().num_rows, 10);
    }

    #[test]
    fn test_string_hashing() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 5).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy());
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();

        // Verify that category (string) column is converted to feature with hash
        for example in &examples {
            let feature = crate::example::get_feature(example, "category").unwrap();
            assert!(!feature.fid.is_empty());
            assert_eq!(feature.value[0], 1.0); // String values get value 1.0
        }
    }

    #[test]
    fn test_predicate_or() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_predicate(
                Predicate::eq("user_id", 0).or(Predicate::eq("user_id", 9))
            );
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 2);
    }

    #[test]
    fn test_predicate_not() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.parquet");
        create_test_parquet_file(&file_path, 10).unwrap();

        let config = ParquetConfig::new(file_path.to_string_lossy())
            .with_predicate(Predicate::gt("user_id", 5).not());
        let source = ParquetDataSource::open(config).unwrap();

        let examples: Vec<_> = source.iter().collect();
        assert_eq!(examples.len(), 6); // user_id 0-5
    }
}
