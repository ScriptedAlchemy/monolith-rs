//! Instance format parsing for Monolith.
//!
//! This module provides types and utilities for working with Instance data,
//! which represents a single training example with feature IDs (FIDs), feature
//! values, labels, and tracking information (line_id).
//!
//! # Instance Format
//!
//! An Instance contains:
//! - Sparse features: Feature IDs (FIDs) organized by slots
//! - Dense features: Float vectors for continuous features
//! - Labels: Training labels
//! - Line ID: Tracking information for the instance
//!
//! # FID Encoding
//!
//! Feature IDs (FIDs) encode both a slot ID and a feature hash:
//! - Upper 32 bits: Slot ID (feature group identifier)
//! - Lower 32 bits: Feature hash
//!
//! # Example
//!
//! ```
//! use monolith_data::instance::{Instance, InstanceParser, extract_slot, make_fid};
//!
//! // Create an instance manually
//! let mut instance = Instance::new();
//! instance.add_sparse_feature("user_id", vec![make_fid(1, 12345)], vec![1.0]);
//! instance.add_dense_feature("embedding", vec![0.1, 0.2, 0.3, 0.4]);
//! instance.set_label(vec![1.0]);
//!
//! // Parse from protobuf bytes
//! let parser = InstanceParser::new();
//! // let instance = parser.parse_from_bytes(&bytes).unwrap();
//!
//! // Extract slot from FID
//! let fid = make_fid(5, 999);
//! assert_eq!(extract_slot(fid), 5);
//! ```

use std::collections::HashMap;

use prost::Message;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use monolith_proto::idl::matrix::proto::LineId;
use monolith_proto::monolith::io::proto::{feature, FidList, FloatList};
use monolith_proto::{Example, Feature, NamedFeature};

/// Errors that can occur during instance parsing.
#[derive(Debug, Error)]
pub enum InstanceError {
    /// Error decoding protobuf data.
    #[error("protobuf decode error: {0}")]
    ProtobufDecode(#[from] prost::DecodeError),

    /// Error parsing JSON data.
    #[error("JSON parse error: {0}")]
    JsonParse(#[from] serde_json::Error),

    /// Error parsing CSV data.
    #[error("CSV parse error: {0}")]
    CsvParse(String),

    /// Invalid FID format.
    #[error("invalid FID format: {0}")]
    InvalidFid(String),

    /// Missing required field.
    #[error("missing required field: {0}")]
    MissingField(String),
}

/// Result type for instance operations.
pub type Result<T> = std::result::Result<T, InstanceError>;

// =============================================================================
// FID Manipulation Utilities
// =============================================================================

/// Number of bits for the feature portion of an FID.
const FEATURE_BITS: u32 = 54;

/// Mask for extracting the feature portion of an FID.
const FEATURE_MASK: i64 = (1i64 << FEATURE_BITS) - 1;

/// Extracts the slot ID from a feature ID (FID).
///
/// The slot is stored in the upper bits of the FID.
///
/// # Arguments
///
/// * `fid` - The feature ID to extract the slot from
///
/// # Returns
///
/// The slot ID as a 32-bit integer.
///
/// # Example
///
/// ```
/// use monolith_data::instance::{extract_slot, make_fid};
///
/// let fid = make_fid(10, 12345);
/// assert_eq!(extract_slot(fid), 10);
/// ```
#[inline]
pub fn extract_slot(fid: i64) -> i32 {
    (fid >> FEATURE_BITS) as i32
}

/// Extracts the feature hash from a feature ID (FID).
///
/// The feature hash is stored in the lower bits of the FID.
///
/// # Arguments
///
/// * `fid` - The feature ID to extract the feature from
///
/// # Returns
///
/// The feature hash as a 64-bit integer.
///
/// # Example
///
/// ```
/// use monolith_data::instance::{extract_feature, make_fid};
///
/// let fid = make_fid(10, 12345);
/// assert_eq!(extract_feature(fid), 12345);
/// ```
#[inline]
pub fn extract_feature(fid: i64) -> i64 {
    fid & FEATURE_MASK
}

/// Creates a feature ID (FID) from a slot ID and feature hash.
///
/// # Arguments
///
/// * `slot` - The slot ID (feature group identifier)
/// * `feature` - The feature hash
///
/// # Returns
///
/// A combined FID encoding both the slot and feature.
///
/// # Example
///
/// ```
/// use monolith_data::instance::{make_fid, extract_slot, extract_feature};
///
/// let fid = make_fid(5, 999);
/// assert_eq!(extract_slot(fid), 5);
/// assert_eq!(extract_feature(fid), 999);
/// ```
#[inline]
pub fn make_fid(slot: i32, feature: i64) -> i64 {
    ((slot as i64) << FEATURE_BITS) | (feature & FEATURE_MASK)
}

// =============================================================================
// Sparse Feature
// =============================================================================

/// A sparse feature with feature IDs and optional values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SparseFeature {
    /// Feature name/slot name.
    pub name: String,
    /// Feature IDs.
    pub fids: Vec<i64>,
    /// Feature values (weights). If empty, all FIDs have implicit weight 1.0.
    pub values: Vec<f32>,
}

impl SparseFeature {
    /// Creates a new sparse feature.
    pub fn new(name: impl Into<String>, fids: Vec<i64>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            fids,
            values,
        }
    }

    /// Returns the number of feature IDs.
    pub fn len(&self) -> usize {
        self.fids.len()
    }

    /// Returns true if this feature has no FIDs.
    pub fn is_empty(&self) -> bool {
        self.fids.is_empty()
    }

    /// Gets the value for a specific FID index, defaulting to 1.0 if values are empty.
    pub fn get_value(&self, index: usize) -> f32 {
        if self.values.is_empty() {
            1.0
        } else {
            self.values.get(index).copied().unwrap_or(1.0)
        }
    }

    /// Extracts all slots from the FIDs in this feature.
    pub fn slots(&self) -> Vec<i32> {
        self.fids.iter().map(|&fid| extract_slot(fid)).collect()
    }
}

// =============================================================================
// Dense Feature
// =============================================================================

/// A dense feature with continuous values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DenseFeature {
    /// Feature name.
    pub name: String,
    /// Dense values.
    pub values: Vec<f32>,
}

impl DenseFeature {
    /// Creates a new dense feature.
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        Self {
            name: name.into(),
            values,
        }
    }

    /// Returns the dimension of this dense feature.
    pub fn dim(&self) -> usize {
        self.values.len()
    }
}

// =============================================================================
// Line ID
// =============================================================================

/// Line ID for tracking instances through the pipeline.
///
/// This corresponds to the LineId proto message and contains metadata
/// about the training example.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LineIdInfo {
    /// User ID.
    pub uid: u64,
    /// Request timestamp.
    pub req_time: i64,
    /// Item ID.
    pub item_id: u64,
    /// User actions.
    pub actions: Vec<i32>,
    /// Channel ID.
    pub chnid: i64,
    /// Pre-actions.
    pub pre_actions: Vec<i32>,
    /// Sample rate.
    pub sample_rate: f32,
    /// Device type.
    pub device_type: String,
    /// Content ID.
    pub cid: String,
    /// User ID (string form).
    pub user_id: String,
    /// Whether this is a draw event.
    pub is_draw: bool,
    /// Rank position.
    pub rank: i32,
    /// Data source name.
    pub data_source_name: String,
    /// Raw bytes for serialization.
    pub raw: Vec<u8>,
}

impl LineIdInfo {
    /// Creates a new empty LineIdInfo.
    pub fn new() -> Self {
        Self {
            sample_rate: 1.0,
            ..Default::default()
        }
    }

    /// Creates a LineIdInfo from raw bytes.
    pub fn from_bytes(raw: Vec<u8>) -> Self {
        Self {
            raw,
            sample_rate: 1.0,
            ..Default::default()
        }
    }
}

// =============================================================================
// Instance
// =============================================================================

/// A single training instance with features, labels, and metadata.
///
/// Instance is the core data type for Monolith training data. It contains:
/// - Sparse features: Feature IDs organized by name/slot
/// - Dense features: Continuous-valued feature vectors
/// - Labels: Training targets
/// - Line ID: Tracking and metadata information
///
/// # Example
///
/// ```
/// use monolith_data::instance::{Instance, make_fid};
///
/// let mut instance = Instance::new();
///
/// // Add sparse features (e.g., categorical features)
/// instance.add_sparse_feature("user_id", vec![make_fid(1, 12345)], vec![1.0]);
/// instance.add_sparse_feature("item_id", vec![make_fid(2, 67890)], vec![1.0]);
///
/// // Add dense features (e.g., embeddings)
/// instance.add_dense_feature("user_embedding", vec![0.1, 0.2, 0.3]);
///
/// // Set labels
/// instance.set_label(vec![1.0, 0.0]);
///
/// assert!(instance.has_sparse_feature("user_id"));
/// assert_eq!(instance.label(), &[1.0, 0.0]);
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    /// Sparse features indexed by name.
    sparse_features: HashMap<String, SparseFeature>,
    /// Dense features indexed by name.
    dense_features: HashMap<String, DenseFeature>,
    /// Training labels.
    label: Vec<f32>,
    /// Instance weight.
    instance_weight: f32,
    /// Line ID tracking information.
    line_id: Option<LineIdInfo>,
}

impl Instance {
    /// Creates a new empty instance.
    pub fn new() -> Self {
        Self {
            sparse_features: HashMap::new(),
            dense_features: HashMap::new(),
            label: Vec::new(),
            instance_weight: 1.0,
            line_id: None,
        }
    }

    /// Adds a sparse feature to this instance.
    ///
    /// # Arguments
    ///
    /// * `name` - Feature name
    /// * `fids` - Feature IDs
    /// * `values` - Feature values (weights)
    pub fn add_sparse_feature(
        &mut self,
        name: impl Into<String>,
        fids: Vec<i64>,
        values: Vec<f32>,
    ) {
        let name = name.into();
        let feature = SparseFeature::new(name.clone(), fids, values);
        self.sparse_features.insert(name, feature);
    }

    /// Adds a dense feature to this instance.
    ///
    /// # Arguments
    ///
    /// * `name` - Feature name
    /// * `values` - Dense values
    pub fn add_dense_feature(&mut self, name: impl Into<String>, values: Vec<f32>) {
        let name = name.into();
        let feature = DenseFeature::new(name.clone(), values);
        self.dense_features.insert(name, feature);
    }

    /// Gets a sparse feature by name.
    pub fn get_sparse_feature(&self, name: &str) -> Option<&SparseFeature> {
        self.sparse_features.get(name)
    }

    /// Gets a dense feature by name.
    pub fn get_dense_feature(&self, name: &str) -> Option<&DenseFeature> {
        self.dense_features.get(name)
    }

    /// Checks if this instance has a sparse feature with the given name.
    pub fn has_sparse_feature(&self, name: &str) -> bool {
        self.sparse_features.contains_key(name)
    }

    /// Checks if this instance has a dense feature with the given name.
    pub fn has_dense_feature(&self, name: &str) -> bool {
        self.dense_features.contains_key(name)
    }

    /// Returns all sparse feature names.
    pub fn sparse_feature_names(&self) -> Vec<&str> {
        self.sparse_features.keys().map(|s| s.as_str()).collect()
    }

    /// Returns all dense feature names.
    pub fn dense_feature_names(&self) -> Vec<&str> {
        self.dense_features.keys().map(|s| s.as_str()).collect()
    }

    /// Returns an iterator over all sparse features.
    pub fn sparse_features(&self) -> impl Iterator<Item = &SparseFeature> {
        self.sparse_features.values()
    }

    /// Returns an iterator over all dense features.
    pub fn dense_features(&self) -> impl Iterator<Item = &DenseFeature> {
        self.dense_features.values()
    }

    /// Sets the label for this instance.
    pub fn set_label(&mut self, label: Vec<f32>) {
        self.label = label;
    }

    /// Returns the label.
    pub fn label(&self) -> &[f32] {
        &self.label
    }

    /// Sets the instance weight.
    pub fn set_instance_weight(&mut self, weight: f32) {
        self.instance_weight = weight;
    }

    /// Returns the instance weight.
    pub fn instance_weight(&self) -> f32 {
        self.instance_weight
    }

    /// Sets the line ID.
    pub fn set_line_id(&mut self, line_id: LineIdInfo) {
        self.line_id = Some(line_id);
    }

    /// Returns the line ID if present.
    pub fn line_id(&self) -> Option<&LineIdInfo> {
        self.line_id.as_ref()
    }

    /// Returns the total number of FIDs across all sparse features.
    pub fn total_fid_count(&self) -> usize {
        self.sparse_features.values().map(|f| f.len()).sum()
    }

    /// Collects all FIDs from all sparse features.
    pub fn all_fids(&self) -> Vec<i64> {
        self.sparse_features
            .values()
            .flat_map(|f| f.fids.iter().copied())
            .collect()
    }

    /// Collects all unique slots from all sparse features.
    pub fn all_slots(&self) -> Vec<i32> {
        let mut slots: Vec<i32> = self
            .sparse_features
            .values()
            .flat_map(|f| f.slots())
            .collect();
        slots.sort_unstable();
        slots.dedup();
        slots
    }

    /// Converts this instance to a tensor dictionary.
    ///
    /// The dictionary contains:
    /// - Sparse features as `{name}_fids` (i64 vec) and `{name}_values` (f32 vec)
    /// - Dense features as `{name}` (f32 vec)
    /// - Label as `label` (f32 vec)
    ///
    /// # Returns
    ///
    /// A HashMap with string keys and Tensor values.
    pub fn to_tensor_dict(&self) -> HashMap<String, Tensor> {
        let mut dict = HashMap::new();

        // Add sparse features
        for (name, feature) in &self.sparse_features {
            dict.insert(
                format!("{}_fids", name),
                Tensor::Int64(feature.fids.clone()),
            );
            let values = if feature.values.is_empty() {
                vec![1.0f32; feature.fids.len()]
            } else {
                feature.values.clone()
            };
            dict.insert(format!("{}_values", name), Tensor::Float(values));
        }

        // Add dense features
        for (name, feature) in &self.dense_features {
            dict.insert(name.clone(), Tensor::Float(feature.values.clone()));
        }

        // Add label
        if !self.label.is_empty() {
            dict.insert("label".to_string(), Tensor::Float(self.label.clone()));
        }

        // Add instance weight
        dict.insert(
            "instance_weight".to_string(),
            Tensor::Float(vec![self.instance_weight]),
        );

        dict
    }

    /// Converts this instance to an Example proto.
    pub fn to_example(&self) -> Example {
        let mut example = crate::example::create_example();
        example.label = self.label.clone();
        example.instance_weight = self.instance_weight;

        // Decode raw bytes into the real LineId proto if possible.
        example.line_id = self
            .line_id
            .as_ref()
            .and_then(|lid| LineId::decode(lid.raw.as_slice()).ok());

        // Add sparse features
        for (name, feature) in &self.sparse_features {
            // Store sparse features as fid_v2_list (values/weights are not represented here).
            let fid_list = FidList {
                value: feature.fids.iter().map(|&v| v as u64).collect(),
            };
            let proto_feature = Feature {
                r#type: Some(feature::Type::FidV2List(fid_list)),
            };
            example.named_feature.push(NamedFeature {
                id: 0,
                name: name.clone(),
                feature: Some(proto_feature),
                sorted_id: 0,
            });
        }

        // Add dense features as float_list
        for (name, feature) in &self.dense_features {
            let float_list = FloatList {
                value: feature.values.clone(),
            };
            let proto_feature = Feature {
                r#type: Some(feature::Type::FloatList(float_list)),
            };
            example.named_feature.push(NamedFeature {
                id: 0,
                name: name.clone(),
                feature: Some(proto_feature),
                sorted_id: 0,
            });
        }

        example
    }
}

// =============================================================================
// Tensor (simple representation for tensor_dict)
// =============================================================================

/// A simple tensor representation for the tensor dictionary.
#[derive(Debug, Clone, PartialEq)]
pub enum Tensor {
    /// 64-bit integer tensor.
    Int64(Vec<i64>),
    /// 32-bit float tensor.
    Float(Vec<f32>),
    /// 64-bit float tensor.
    Double(Vec<f64>),
    /// Bytes tensor.
    Bytes(Vec<Vec<u8>>),
}

impl Tensor {
    /// Returns the length of the tensor.
    pub fn len(&self) -> usize {
        match self {
            Tensor::Int64(v) => v.len(),
            Tensor::Float(v) => v.len(),
            Tensor::Double(v) => v.len(),
            Tensor::Bytes(v) => v.len(),
        }
    }

    /// Returns true if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tries to get this tensor as an i64 vector.
    pub fn as_int64(&self) -> Option<&Vec<i64>> {
        match self {
            Tensor::Int64(v) => Some(v),
            _ => None,
        }
    }

    /// Tries to get this tensor as an f32 vector.
    pub fn as_float(&self) -> Option<&Vec<f32>> {
        match self {
            Tensor::Float(v) => Some(v),
            _ => None,
        }
    }

    /// Tries to get this tensor as an f64 vector.
    pub fn as_double(&self) -> Option<&Vec<f64>> {
        match self {
            Tensor::Double(v) => Some(v),
            _ => None,
        }
    }

    /// Tries to get this tensor as a bytes vector.
    pub fn as_bytes(&self) -> Option<&Vec<Vec<u8>>> {
        match self {
            Tensor::Bytes(v) => Some(v),
            _ => None,
        }
    }
}

// =============================================================================
// Instance Batch
// =============================================================================

/// A batch of instances for efficient batch processing.
///
/// InstanceBatch provides methods for operating on multiple instances
/// at once, which is useful for batched training operations.
///
/// # Example
///
/// ```
/// use monolith_data::instance::{Instance, InstanceBatch, make_fid};
///
/// let mut batch = InstanceBatch::new();
///
/// for i in 0..3 {
///     let mut instance = Instance::new();
///     instance.add_sparse_feature("id", vec![make_fid(1, i as i64)], vec![1.0]);
///     instance.set_label(vec![i as f32]);
///     batch.push(instance);
/// }
///
/// assert_eq!(batch.len(), 3);
///
/// // Convert batch to tensor dict
/// let tensors = batch.to_tensor_dict();
/// ```
#[derive(Debug, Clone, Default)]
pub struct InstanceBatch {
    instances: Vec<Instance>,
}

impl InstanceBatch {
    /// Creates a new empty batch.
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
        }
    }

    /// Creates a batch with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            instances: Vec::with_capacity(capacity),
        }
    }

    /// Creates a batch from a vector of instances.
    pub fn from_vec(instances: Vec<Instance>) -> Self {
        Self { instances }
    }

    /// Adds an instance to the batch.
    pub fn push(&mut self, instance: Instance) {
        self.instances.push(instance);
    }

    /// Returns the number of instances in the batch.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Returns true if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Returns a reference to the instances.
    pub fn instances(&self) -> &[Instance] {
        &self.instances
    }

    /// Returns a mutable reference to the instances.
    pub fn instances_mut(&mut self) -> &mut [Instance] {
        &mut self.instances
    }

    /// Consumes the batch and returns the instances.
    pub fn into_instances(self) -> Vec<Instance> {
        self.instances
    }

    /// Returns an iterator over the instances.
    pub fn iter(&self) -> impl Iterator<Item = &Instance> {
        self.instances.iter()
    }

    /// Returns a mutable iterator over the instances.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Instance> {
        self.instances.iter_mut()
    }

    /// Converts the batch to a batched tensor dictionary.
    ///
    /// For each feature, the values from all instances are concatenated
    /// along with batch indices for sparse features.
    pub fn to_tensor_dict(&self) -> HashMap<String, Tensor> {
        if self.instances.is_empty() {
            return HashMap::new();
        }

        let mut dict: HashMap<String, Tensor> = HashMap::new();

        // Collect all sparse feature names
        let mut sparse_names: Vec<String> = self
            .instances
            .iter()
            .flat_map(|inst| {
                inst.sparse_feature_names()
                    .into_iter()
                    .map(|s| s.to_string())
            })
            .collect();
        sparse_names.sort();
        sparse_names.dedup();

        // Collect all dense feature names
        let mut dense_names: Vec<String> = self
            .instances
            .iter()
            .flat_map(|inst| {
                inst.dense_feature_names()
                    .into_iter()
                    .map(|s| s.to_string())
            })
            .collect();
        dense_names.sort();
        dense_names.dedup();

        // Batch sparse features
        for name in &sparse_names {
            let mut all_fids = Vec::new();
            let mut all_values = Vec::new();
            let mut batch_indices = Vec::new();

            for (batch_idx, instance) in self.instances.iter().enumerate() {
                if let Some(feature) = instance.get_sparse_feature(name) {
                    all_fids.extend_from_slice(&feature.fids);
                    if feature.values.is_empty() {
                        all_values.extend(std::iter::repeat(1.0f32).take(feature.fids.len()));
                    } else {
                        all_values.extend_from_slice(&feature.values);
                    }
                    batch_indices
                        .extend(std::iter::repeat(batch_idx as i64).take(feature.fids.len()));
                }
            }

            dict.insert(format!("{}_fids", name), Tensor::Int64(all_fids));
            dict.insert(format!("{}_values", name), Tensor::Float(all_values));
            dict.insert(format!("{}_batch_idx", name), Tensor::Int64(batch_indices));
        }

        // Batch dense features (concatenate all values)
        for name in &dense_names {
            let mut all_values = Vec::new();
            for instance in &self.instances {
                if let Some(feature) = instance.get_dense_feature(name) {
                    all_values.extend_from_slice(&feature.values);
                }
            }
            dict.insert(name.clone(), Tensor::Float(all_values));
        }

        // Batch labels
        let all_labels: Vec<f32> = self
            .instances
            .iter()
            .flat_map(|i| i.label().iter().copied())
            .collect();
        if !all_labels.is_empty() {
            dict.insert("label".to_string(), Tensor::Float(all_labels));
        }

        // Batch instance weights
        let weights: Vec<f32> = self.instances.iter().map(|i| i.instance_weight()).collect();
        dict.insert("instance_weight".to_string(), Tensor::Float(weights));

        dict
    }
}

impl IntoIterator for InstanceBatch {
    type Item = Instance;
    type IntoIter = std::vec::IntoIter<Instance>;

    fn into_iter(self) -> Self::IntoIter {
        self.instances.into_iter()
    }
}

impl FromIterator<Instance> for InstanceBatch {
    fn from_iter<T: IntoIterator<Item = Instance>>(iter: T) -> Self {
        Self {
            instances: iter.into_iter().collect(),
        }
    }
}

// =============================================================================
// Instance Parser
// =============================================================================

/// Parser for converting various formats to Instance.
///
/// InstanceParser supports parsing from:
/// - Protobuf bytes (Example proto)
/// - JSON strings
/// - CSV-like strings
///
/// # Example
///
/// ```
/// use monolith_data::instance::InstanceParser;
///
/// let parser = InstanceParser::new();
///
/// // Parse from JSON
/// let json = r#"{
///     "sparse_features": {
///         "user_id": {"fids": [12345], "values": [1.0]}
///     },
///     "label": [1.0]
/// }"#;
/// let instance = parser.parse_from_json(json).unwrap();
/// assert!(instance.has_sparse_feature("user_id"));
/// ```
#[derive(Debug, Clone, Default)]
pub struct InstanceParser {
    /// Default values for missing features.
    default_values: HashMap<String, f32>,
}

impl InstanceParser {
    /// Creates a new parser with default configuration.
    pub fn new() -> Self {
        Self {
            default_values: HashMap::new(),
        }
    }

    /// Sets a default value for a feature.
    pub fn with_default_value(mut self, feature_name: &str, value: f32) -> Self {
        self.default_values.insert(feature_name.to_string(), value);
        self
    }

    /// Parses an Instance from protobuf bytes.
    ///
    /// # Arguments
    ///
    /// * `data` - The protobuf-encoded bytes
    ///
    /// # Returns
    ///
    /// A parsed Instance or an error.
    pub fn parse_from_bytes(&self, data: &[u8]) -> Result<Instance> {
        let example = Example::decode(data)?;
        Ok(self.from_example(&example))
    }

    /// Parses an Instance from a JSON string.
    ///
    /// Expected JSON format:
    /// ```json
    /// {
    ///     "sparse_features": {
    ///         "feature_name": {"fids": [1, 2, 3], "values": [1.0, 1.0, 1.0]}
    ///     },
    ///     "dense_features": {
    ///         "feature_name": [0.1, 0.2, 0.3]
    ///     },
    ///     "label": [1.0],
    ///     "instance_weight": 1.0,
    ///     "line_id": {"uid": 12345, "item_id": 67890}
    /// }
    /// ```
    pub fn parse_from_json(&self, json: &str) -> Result<Instance> {
        let json_instance: JsonInstance = serde_json::from_str(json)?;
        Ok(json_instance.into())
    }

    /// Parses an Instance from a CSV-like format.
    ///
    /// Format: `label,fid1:value1,fid2:value2,...`
    ///
    /// # Arguments
    ///
    /// * `line` - The CSV line
    /// * `delimiter` - Field delimiter (default: ',')
    ///
    /// # Returns
    ///
    /// A parsed Instance or an error.
    pub fn parse_from_csv(&self, line: &str, delimiter: char) -> Result<Instance> {
        let parts: Vec<&str> = line.split(delimiter).collect();
        if parts.is_empty() {
            return Err(InstanceError::CsvParse("empty line".to_string()));
        }

        let mut instance = Instance::new();

        // First field is the label
        let label: f32 = parts[0]
            .trim()
            .parse()
            .map_err(|e| InstanceError::CsvParse(format!("invalid label: {}", e)))?;
        instance.set_label(vec![label]);

        // Remaining fields are fid:value pairs
        let mut fids = Vec::new();
        let mut values = Vec::new();

        for part in parts.iter().skip(1) {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }

            let (fid, value) = if let Some(colon_idx) = part.find(':') {
                let fid_str = &part[..colon_idx];
                let val_str = &part[colon_idx + 1..];
                let fid: i64 = fid_str.parse().map_err(|e| {
                    InstanceError::CsvParse(format!("invalid fid '{}': {}", fid_str, e))
                })?;
                let value: f32 = val_str.parse().map_err(|e| {
                    InstanceError::CsvParse(format!("invalid value '{}': {}", val_str, e))
                })?;
                (fid, value)
            } else {
                let fid: i64 = part.parse().map_err(|e| {
                    InstanceError::CsvParse(format!("invalid fid '{}': {}", part, e))
                })?;
                (fid, 1.0)
            };

            fids.push(fid);
            values.push(value);
        }

        if !fids.is_empty() {
            instance.add_sparse_feature("features", fids, values);
        }

        Ok(instance)
    }

    /// Converts an Example proto to an Instance.
    pub fn from_example(&self, example: &Example) -> Instance {
        let mut instance = Instance::new();

        for named_feature in &example.named_feature {
            if let Some(feature) = &named_feature.feature {
                match &feature.r#type {
                    Some(feature::Type::FidV2List(l)) => {
                        instance.add_sparse_feature(
                            &named_feature.name,
                            l.value.iter().map(|&v| v as i64).collect(),
                            vec![],
                        );
                    }
                    Some(feature::Type::FidV1List(l)) => {
                        instance.add_sparse_feature(
                            &named_feature.name,
                            l.value.iter().map(|&v| v as i64).collect(),
                            vec![],
                        );
                    }
                    Some(feature::Type::FloatList(l)) => {
                        instance.add_dense_feature(&named_feature.name, l.value.clone());
                    }
                    _ => {
                        // Ignore other encodings for now.
                    }
                }
            }
        }

        // Set line_id if present
        if let Some(line_id) = &example.line_id {
            instance.set_line_id(LineIdInfo::from_bytes(line_id.encode_to_vec()));
        }

        instance.set_label(example.label.clone());
        instance.set_instance_weight(example.instance_weight);

        instance
    }
}

// =============================================================================
// JSON Instance (for JSON parsing)
// =============================================================================

/// Helper struct for JSON deserialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonInstance {
    #[serde(default)]
    sparse_features: HashMap<String, JsonSparseFeature>,
    #[serde(default)]
    dense_features: HashMap<String, Vec<f32>>,
    #[serde(default)]
    label: Vec<f32>,
    #[serde(default = "default_weight")]
    instance_weight: f32,
    #[serde(default)]
    line_id: Option<JsonLineId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonSparseFeature {
    fids: Vec<i64>,
    #[serde(default)]
    values: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct JsonLineId {
    #[serde(default)]
    uid: u64,
    #[serde(default)]
    req_time: i64,
    #[serde(default)]
    item_id: u64,
    #[serde(default)]
    chnid: i64,
    #[serde(default)]
    user_id: String,
}

fn default_weight() -> f32 {
    1.0
}

impl From<JsonInstance> for Instance {
    fn from(json: JsonInstance) -> Self {
        let mut instance = Instance::new();

        for (name, feature) in json.sparse_features {
            instance.add_sparse_feature(&name, feature.fids, feature.values);
        }

        for (name, values) in json.dense_features {
            instance.add_dense_feature(&name, values);
        }

        instance.set_label(json.label);
        instance.set_instance_weight(json.instance_weight);

        if let Some(line_id) = json.line_id {
            instance.set_line_id(LineIdInfo {
                uid: line_id.uid,
                req_time: line_id.req_time,
                item_id: line_id.item_id,
                chnid: line_id.chnid,
                user_id: line_id.user_id,
                ..Default::default()
            });
        }

        instance
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use monolith_proto::monolith::io::proto::{feature, FidList, FloatList};

    // -------------------------------------------------------------------------
    // FID utilities tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_make_fid() {
        let fid = make_fid(10, 12345);
        assert_eq!(extract_slot(fid), 10);
        assert_eq!(extract_feature(fid), 12345);
    }

    #[test]
    fn test_fid_roundtrip() {
        for slot in [0, 1, 10, 100, 500] {
            for feature in [0i64, 1, 100, 10000, 1_000_000, 1_000_000_000] {
                let fid = make_fid(slot, feature);
                assert_eq!(
                    extract_slot(fid),
                    slot,
                    "slot mismatch for ({}, {})",
                    slot,
                    feature
                );
                assert_eq!(
                    extract_feature(fid),
                    feature,
                    "feature mismatch for ({}, {})",
                    slot,
                    feature
                );
            }
        }
    }

    #[test]
    fn test_fid_negative_slot() {
        // Negative slots should work (they're cast to i32)
        let fid = make_fid(-1, 100);
        assert_eq!(extract_slot(fid), -1);
        assert_eq!(extract_feature(fid), 100);
    }

    // -------------------------------------------------------------------------
    // SparseFeature tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_sparse_feature_new() {
        let feature = SparseFeature::new("test", vec![1, 2, 3], vec![1.0, 2.0, 3.0]);
        assert_eq!(feature.name, "test");
        assert_eq!(feature.len(), 3);
        assert!(!feature.is_empty());
    }

    #[test]
    fn test_sparse_feature_get_value() {
        let feature = SparseFeature::new("test", vec![1, 2, 3], vec![1.0, 2.0, 3.0]);
        assert_eq!(feature.get_value(0), 1.0);
        assert_eq!(feature.get_value(1), 2.0);
        assert_eq!(feature.get_value(2), 3.0);
        assert_eq!(feature.get_value(10), 1.0); // Out of bounds defaults to 1.0
    }

    #[test]
    fn test_sparse_feature_empty_values() {
        let feature = SparseFeature::new("test", vec![1, 2, 3], vec![]);
        // Empty values should default to 1.0
        assert_eq!(feature.get_value(0), 1.0);
        assert_eq!(feature.get_value(1), 1.0);
    }

    #[test]
    fn test_sparse_feature_slots() {
        let fids = vec![make_fid(1, 100), make_fid(2, 200), make_fid(1, 300)];
        let feature = SparseFeature::new("test", fids, vec![]);
        let slots = feature.slots();
        assert_eq!(slots, vec![1, 2, 1]);
    }

    // -------------------------------------------------------------------------
    // DenseFeature tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dense_feature_new() {
        let feature = DenseFeature::new("embedding", vec![0.1, 0.2, 0.3, 0.4]);
        assert_eq!(feature.name, "embedding");
        assert_eq!(feature.dim(), 4);
    }

    // -------------------------------------------------------------------------
    // Instance tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_instance_new() {
        let instance = Instance::new();
        assert!(instance.sparse_feature_names().is_empty());
        assert!(instance.dense_feature_names().is_empty());
        assert!(instance.label().is_empty());
        assert_eq!(instance.instance_weight(), 1.0);
        assert!(instance.line_id().is_none());
    }

    #[test]
    fn test_instance_add_sparse_feature() {
        let mut instance = Instance::new();
        instance.add_sparse_feature("user_id", vec![12345], vec![1.0]);

        assert!(instance.has_sparse_feature("user_id"));
        assert!(!instance.has_sparse_feature("item_id"));

        let feature = instance.get_sparse_feature("user_id").unwrap();
        assert_eq!(feature.fids, vec![12345]);
        assert_eq!(feature.values, vec![1.0]);
    }

    #[test]
    fn test_instance_add_dense_feature() {
        let mut instance = Instance::new();
        instance.add_dense_feature("embedding", vec![0.1, 0.2, 0.3]);

        assert!(instance.has_dense_feature("embedding"));

        let feature = instance.get_dense_feature("embedding").unwrap();
        assert_eq!(feature.values, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_instance_label() {
        let mut instance = Instance::new();
        instance.set_label(vec![1.0, 0.0]);

        assert_eq!(instance.label(), &[1.0, 0.0]);
    }

    #[test]
    fn test_instance_weight() {
        let mut instance = Instance::new();
        assert_eq!(instance.instance_weight(), 1.0);

        instance.set_instance_weight(0.5);
        assert_eq!(instance.instance_weight(), 0.5);
    }

    #[test]
    fn test_instance_line_id() {
        let mut instance = Instance::new();
        assert!(instance.line_id().is_none());

        let mut line_id = LineIdInfo::new();
        line_id.uid = 12345;
        line_id.item_id = 67890;
        instance.set_line_id(line_id);

        let lid = instance.line_id().unwrap();
        assert_eq!(lid.uid, 12345);
        assert_eq!(lid.item_id, 67890);
    }

    #[test]
    fn test_instance_total_fid_count() {
        let mut instance = Instance::new();
        instance.add_sparse_feature("a", vec![1, 2, 3], vec![]);
        instance.add_sparse_feature("b", vec![4, 5], vec![]);

        assert_eq!(instance.total_fid_count(), 5);
    }

    #[test]
    fn test_instance_all_fids() {
        let mut instance = Instance::new();
        instance.add_sparse_feature("a", vec![1, 2], vec![]);
        instance.add_sparse_feature("b", vec![3, 4], vec![]);

        let mut fids = instance.all_fids();
        fids.sort();
        assert_eq!(fids, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_instance_all_slots() {
        let mut instance = Instance::new();
        instance.add_sparse_feature("a", vec![make_fid(1, 100), make_fid(2, 200)], vec![]);
        instance.add_sparse_feature("b", vec![make_fid(1, 300), make_fid(3, 400)], vec![]);

        let slots = instance.all_slots();
        assert_eq!(slots, vec![1, 2, 3]);
    }

    #[test]
    fn test_instance_to_tensor_dict() {
        let mut instance = Instance::new();
        instance.add_sparse_feature("user_id", vec![12345], vec![1.0]);
        instance.add_dense_feature("embedding", vec![0.1, 0.2, 0.3]);
        instance.set_label(vec![1.0]);

        let dict = instance.to_tensor_dict();

        assert!(dict.contains_key("user_id_fids"));
        assert!(dict.contains_key("user_id_values"));
        assert!(dict.contains_key("embedding"));
        assert!(dict.contains_key("label"));

        assert_eq!(dict["user_id_fids"].as_int64().unwrap(), &vec![12345]);
        assert_eq!(dict["user_id_values"].as_float().unwrap(), &vec![1.0]);
        assert_eq!(dict["embedding"].as_float().unwrap(), &vec![0.1, 0.2, 0.3]);
        assert_eq!(dict["label"].as_float().unwrap(), &vec![1.0]);
    }

    #[test]
    fn test_instance_to_example() {
        let mut instance = Instance::new();
        instance.add_sparse_feature("user_id", vec![12345], vec![1.0]);
        instance.add_dense_feature("embedding", vec![0.1, 0.2, 0.3]);

        let example = instance.to_example();
        assert_eq!(example.named_feature.len(), 2);
    }

    // -------------------------------------------------------------------------
    // InstanceBatch tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_instance_batch_new() {
        let batch = InstanceBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.len(), 0);
    }

    #[test]
    fn test_instance_batch_push() {
        let mut batch = InstanceBatch::new();
        batch.push(Instance::new());
        batch.push(Instance::new());

        assert_eq!(batch.len(), 2);
    }

    #[test]
    fn test_instance_batch_from_vec() {
        let instances = vec![Instance::new(), Instance::new(), Instance::new()];
        let batch = InstanceBatch::from_vec(instances);

        assert_eq!(batch.len(), 3);
    }

    #[test]
    fn test_instance_batch_into_iter() {
        let mut batch = InstanceBatch::new();
        for i in 0..3 {
            let mut instance = Instance::new();
            instance.set_label(vec![i as f32]);
            batch.push(instance);
        }

        let instances: Vec<Instance> = batch.into_iter().collect();
        assert_eq!(instances.len(), 3);
    }

    #[test]
    fn test_instance_batch_collect() {
        let instances: InstanceBatch = (0..5)
            .map(|i| {
                let mut inst = Instance::new();
                inst.set_label(vec![i as f32]);
                inst
            })
            .collect();

        assert_eq!(instances.len(), 5);
    }

    #[test]
    fn test_instance_batch_to_tensor_dict() {
        let mut batch = InstanceBatch::new();

        for i in 0..3 {
            let mut instance = Instance::new();
            instance.add_sparse_feature("user_id", vec![i as i64], vec![1.0]);
            instance.set_label(vec![i as f32]);
            batch.push(instance);
        }

        let dict = batch.to_tensor_dict();

        // Check batched FIDs
        let fids = dict["user_id_fids"].as_int64().unwrap();
        assert_eq!(fids, &vec![0, 1, 2]);

        // Check batch indices
        let batch_idx = dict["user_id_batch_idx"].as_int64().unwrap();
        assert_eq!(batch_idx, &vec![0, 1, 2]);

        // Check batched labels
        let labels = dict["label"].as_float().unwrap();
        assert_eq!(labels, &vec![0.0, 1.0, 2.0]);
    }

    // -------------------------------------------------------------------------
    // InstanceParser tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_parser_parse_from_json() {
        let json = r#"{
            "sparse_features": {
                "user_id": {"fids": [12345], "values": [1.0]},
                "item_id": {"fids": [67890], "values": [1.0]}
            },
            "dense_features": {
                "embedding": [0.1, 0.2, 0.3]
            },
            "label": [1.0, 0.0]
        }"#;

        let parser = InstanceParser::new();
        let instance = parser.parse_from_json(json).unwrap();

        assert!(instance.has_sparse_feature("user_id"));
        assert!(instance.has_sparse_feature("item_id"));
        assert!(instance.has_dense_feature("embedding"));
        assert_eq!(instance.label(), &[1.0, 0.0]);
    }

    #[test]
    fn test_parser_parse_from_json_minimal() {
        let json = r#"{"sparse_features": {"test": {"fids": [1]}}}"#;

        let parser = InstanceParser::new();
        let instance = parser.parse_from_json(json).unwrap();

        assert!(instance.has_sparse_feature("test"));
        assert!(instance.label().is_empty());
    }

    #[test]
    fn test_parser_parse_from_csv() {
        let parser = InstanceParser::new();
        let instance = parser
            .parse_from_csv("1.0,100:0.5,200:0.3,300:0.2", ',')
            .unwrap();

        assert_eq!(instance.label(), &[1.0]);
        let feature = instance.get_sparse_feature("features").unwrap();
        assert_eq!(feature.fids, vec![100, 200, 300]);
        assert_eq!(feature.values, vec![0.5, 0.3, 0.2]);
    }

    #[test]
    fn test_parser_parse_from_csv_no_values() {
        let parser = InstanceParser::new();
        let instance = parser.parse_from_csv("0.0,100,200,300", ',').unwrap();

        assert_eq!(instance.label(), &[0.0]);
        let feature = instance.get_sparse_feature("features").unwrap();
        assert_eq!(feature.fids, vec![100, 200, 300]);
        assert_eq!(feature.values, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_parser_parse_from_bytes() {
        // Create an Example proto and encode it
        let example = Example {
            named_feature: vec![NamedFeature {
                id: 0,
                name: "test".to_string(),
                feature: Some(Feature {
                    r#type: Some(feature::Type::FidV2List(FidList {
                        value: vec![1, 2, 3],
                    })),
                }),
                sorted_id: 0,
            }],
            named_raw_feature: vec![],
            line_id: Some(monolith_proto::LineId::default()),
            label: vec![],
            instance_weight: 1.0,
            data_source_key: 0,
        };

        let bytes = example.encode_to_vec();

        let parser = InstanceParser::new();
        let instance = parser.parse_from_bytes(&bytes).unwrap();

        assert!(instance.has_sparse_feature("test"));
        let feature = instance.get_sparse_feature("test").unwrap();
        assert_eq!(feature.fids, vec![1, 2, 3]);
        assert_eq!(feature.values, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_parser_from_example() {
        let example = Example {
            named_feature: vec![
                NamedFeature {
                    id: 0,
                    name: "sparse".to_string(),
                    feature: Some(Feature {
                        r#type: Some(feature::Type::FidV2List(FidList { value: vec![1, 2] })),
                    }),
                    sorted_id: 0,
                },
                NamedFeature {
                    id: 1,
                    name: "dense".to_string(),
                    feature: Some(Feature {
                        r#type: Some(feature::Type::FloatList(FloatList {
                            value: vec![0.1, 0.2, 0.3],
                        })),
                    }),
                    sorted_id: 0,
                },
            ],
            named_raw_feature: vec![],
            line_id: Some(monolith_proto::LineId {
                uid: Some(12345),
                ..Default::default()
            }),
            label: vec![],
            instance_weight: 1.0,
            data_source_key: 0,
        };

        let parser = InstanceParser::new();
        let instance = parser.from_example(&example);

        assert!(instance.has_sparse_feature("sparse"));
        assert!(instance.has_dense_feature("dense"));
        assert!(instance.line_id().is_some());
    }

    // -------------------------------------------------------------------------
    // Tensor tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_tensor_int64() {
        let tensor = Tensor::Int64(vec![1, 2, 3]);
        assert_eq!(tensor.len(), 3);
        assert!(!tensor.is_empty());
        assert_eq!(tensor.as_int64().unwrap(), &vec![1, 2, 3]);
        assert!(tensor.as_float().is_none());
    }

    #[test]
    fn test_tensor_float() {
        let tensor = Tensor::Float(vec![1.0, 2.0, 3.0]);
        assert_eq!(tensor.len(), 3);
        assert_eq!(tensor.as_float().unwrap(), &vec![1.0, 2.0, 3.0]);
        assert!(tensor.as_int64().is_none());
    }

    #[test]
    fn test_tensor_empty() {
        let tensor = Tensor::Int64(vec![]);
        assert!(tensor.is_empty());
        assert_eq!(tensor.len(), 0);
    }

    // -------------------------------------------------------------------------
    // LineIdInfo tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_line_id_info_new() {
        let line_id = LineIdInfo::new();
        assert_eq!(line_id.uid, 0);
        assert_eq!(line_id.sample_rate, 1.0);
    }

    #[test]
    fn test_line_id_info_from_bytes() {
        let raw = vec![1, 2, 3, 4];
        let line_id = LineIdInfo::from_bytes(raw.clone());
        assert_eq!(line_id.raw, raw);
        assert_eq!(line_id.sample_rate, 1.0);
    }
}
