//! Group Interaction Layer implementation.
//!
//! This module provides layers for computing feature interactions within and across
//! groups of features. This is useful for recommendation systems where features
//! can be naturally grouped (e.g., user features, item features, context features).
//!
//! # Overview
//!
//! The Group Interaction Layer allows you to:
//! - Group features into semantic categories
//! - Compute interactions within groups (intra-group)
//! - Compute interactions across groups (inter-group)
//! - Use different interaction types (inner product, Hadamard product)
//!
//! # Example
//!
//! ```
//! use monolith_layers::group_interaction::{GroupInteractionConfig, GroupInteractionLayer, InteractionType};
//! use monolith_layers::tensor::Tensor;
//! use monolith_layers::layer::Layer;
//!
//! // Create groups: 3 groups with 4 features each (embedding dim = 8)
//! let config = GroupInteractionConfig::new(3, 4, 8)
//!     .with_inter_group(true)
//!     .with_intra_group(true);
//!
//! let layer = GroupInteractionLayer::from_config(&config);
//!
//! // Input: [batch_size, num_groups * features_per_group * embedding_dim]
//! let input = Tensor::rand(&[2, 96]);  // 2 samples, 3*4*8 = 96 features
//! let output = layer.forward(&input).unwrap();
//! ```

use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Type of interaction to compute between feature vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum InteractionType {
    /// Inner product: computes the dot product, resulting in a scalar
    #[default]
    InnerProduct,

    /// Hadamard product: element-wise multiplication, preserving dimensionality
    Hadamard,

    /// Concatenation: concatenates the two vectors
    Concat,
}

/// Configuration for the Group Interaction Layer.
///
/// # Example
///
/// ```
/// use monolith_layers::group_interaction::{GroupInteractionConfig, InteractionType};
///
/// let config = GroupInteractionConfig::new(3, 4, 16)
///     .with_interaction_type(InteractionType::Hadamard)
///     .with_inter_group(true)
///     .with_intra_group(false);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupInteractionConfig {
    /// Number of feature groups
    pub num_groups: usize,

    /// Number of features per group
    pub features_per_group: usize,

    /// Embedding dimension for each feature
    pub embedding_dim: usize,

    /// Type of interaction to compute
    pub interaction_type: InteractionType,

    /// Whether to compute inter-group interactions
    pub inter_group: bool,

    /// Whether to compute intra-group interactions
    pub intra_group: bool,

    /// Whether to include the original embeddings in the output
    pub include_original: bool,
}

impl GroupInteractionConfig {
    /// Creates a new Group Interaction configuration.
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of feature groups
    /// * `features_per_group` - Number of features in each group
    /// * `embedding_dim` - Dimension of each feature embedding
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::group_interaction::GroupInteractionConfig;
    ///
    /// let config = GroupInteractionConfig::new(3, 5, 8);
    /// assert_eq!(config.num_groups, 3);
    /// assert_eq!(config.features_per_group, 5);
    /// assert_eq!(config.embedding_dim, 8);
    /// ```
    pub fn new(num_groups: usize, features_per_group: usize, embedding_dim: usize) -> Self {
        Self {
            num_groups,
            features_per_group,
            embedding_dim,
            interaction_type: InteractionType::default(),
            inter_group: true,
            intra_group: false,
            include_original: true,
        }
    }

    /// Sets the interaction type.
    ///
    /// # Arguments
    ///
    /// * `interaction_type` - The type of interaction to compute
    pub fn with_interaction_type(mut self, interaction_type: InteractionType) -> Self {
        self.interaction_type = interaction_type;
        self
    }

    /// Sets whether to compute inter-group interactions.
    ///
    /// # Arguments
    ///
    /// * `inter_group` - Whether to enable inter-group interactions
    pub fn with_inter_group(mut self, inter_group: bool) -> Self {
        self.inter_group = inter_group;
        self
    }

    /// Sets whether to compute intra-group interactions.
    ///
    /// # Arguments
    ///
    /// * `intra_group` - Whether to enable intra-group interactions
    pub fn with_intra_group(mut self, intra_group: bool) -> Self {
        self.intra_group = intra_group;
        self
    }

    /// Sets whether to include original embeddings in output.
    ///
    /// # Arguments
    ///
    /// * `include_original` - Whether to include original embeddings
    pub fn with_include_original(mut self, include_original: bool) -> Self {
        self.include_original = include_original;
        self
    }

    /// Computes the output dimension based on configuration.
    pub fn output_dim(&self) -> usize {
        let total_features = self.num_groups * self.features_per_group;
        let mut output_dim = 0;

        // Original embeddings
        if self.include_original {
            output_dim += total_features * self.embedding_dim;
        }

        // Interaction dimensions depend on type
        let interaction_dim = match self.interaction_type {
            InteractionType::InnerProduct => 1,
            InteractionType::Hadamard => self.embedding_dim,
            InteractionType::Concat => self.embedding_dim * 2,
        };

        // Inter-group interactions: between features from different groups
        if self.inter_group {
            // Number of pairs = (total_features choose 2) - intra-group pairs
            // But for inter-group, we count pairs (i,j) where group(i) != group(j)
            let inter_pairs = self.count_inter_group_pairs();
            output_dim += inter_pairs * interaction_dim;
        }

        // Intra-group interactions: between features within the same group
        if self.intra_group {
            let intra_pairs = self.count_intra_group_pairs();
            output_dim += intra_pairs * interaction_dim;
        }

        output_dim
    }

    /// Counts the number of inter-group feature pairs.
    fn count_inter_group_pairs(&self) -> usize {
        let total_features = self.num_groups * self.features_per_group;
        let total_pairs = total_features * (total_features - 1) / 2;
        let intra_pairs = self.count_intra_group_pairs();
        total_pairs - intra_pairs
    }

    /// Counts the number of intra-group feature pairs.
    fn count_intra_group_pairs(&self) -> usize {
        let pairs_per_group = self.features_per_group * (self.features_per_group - 1) / 2;
        self.num_groups * pairs_per_group
    }
}

/// Group Interaction Layer for computing feature interactions within and across groups.
///
/// This layer takes grouped feature embeddings and computes pairwise interactions
/// according to the configuration. It supports both intra-group (within the same group)
/// and inter-group (across different groups) interactions.
///
/// # Example
///
/// ```
/// use monolith_layers::group_interaction::{GroupInteractionConfig, GroupInteractionLayer};
/// use monolith_layers::tensor::Tensor;
/// use monolith_layers::layer::Layer;
///
/// let config = GroupInteractionConfig::new(2, 3, 4);
/// let layer = GroupInteractionLayer::from_config(&config);
///
/// // Input shape: [batch, num_groups * features_per_group * embedding_dim]
/// let input = Tensor::rand(&[2, 24]);  // 2*3*4 = 24
/// let output = layer.forward(&input).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupInteractionLayer {
    /// Layer configuration
    config: GroupInteractionConfig,

    /// Cached input for backward pass
    cached_input: Option<Tensor>,

    /// Whether in training mode
    training: bool,
}

impl GroupInteractionLayer {
    /// Creates a new Group Interaction Layer from configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The layer configuration
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::group_interaction::{GroupInteractionConfig, GroupInteractionLayer};
    ///
    /// let config = GroupInteractionConfig::new(3, 4, 8);
    /// let layer = GroupInteractionLayer::from_config(&config);
    /// ```
    pub fn from_config(config: &GroupInteractionConfig) -> Self {
        Self {
            config: config.clone(),
            cached_input: None,
            training: true,
        }
    }

    /// Creates a new Group Interaction Layer with default configuration.
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of feature groups
    /// * `features_per_group` - Number of features per group
    /// * `embedding_dim` - Dimension of each feature embedding
    pub fn new(num_groups: usize, features_per_group: usize, embedding_dim: usize) -> Self {
        let config = GroupInteractionConfig::new(num_groups, features_per_group, embedding_dim);
        Self::from_config(&config)
    }

    /// Returns the layer configuration.
    pub fn config(&self) -> &GroupInteractionConfig {
        &self.config
    }

    /// Returns the expected input dimension.
    pub fn input_dim(&self) -> usize {
        self.config.num_groups * self.config.features_per_group * self.config.embedding_dim
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.config.output_dim()
    }

    /// Clears cached values.
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
    }

    /// Extracts a feature embedding from the flattened input.
    ///
    /// # Arguments
    ///
    /// * `input` - The input data slice
    /// * `group_idx` - The group index
    /// * `feature_idx` - The feature index within the group
    fn get_feature_embedding(
        &self,
        input: &[f32],
        group_idx: usize,
        feature_idx: usize,
    ) -> Vec<f32> {
        let flat_idx = group_idx * self.config.features_per_group + feature_idx;
        let offset = flat_idx * self.config.embedding_dim;
        input[offset..offset + self.config.embedding_dim].to_vec()
    }

    /// Computes the interaction between two feature embeddings.
    fn compute_interaction(&self, v1: &[f32], v2: &[f32]) -> Vec<f32> {
        match self.config.interaction_type {
            InteractionType::InnerProduct => {
                let inner: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
                vec![inner]
            }
            InteractionType::Hadamard => v1.iter().zip(v2.iter()).map(|(a, b)| a * b).collect(),
            InteractionType::Concat => {
                let mut result = v1.to_vec();
                result.extend_from_slice(v2);
                result
            }
        }
    }

    /// Gets the group index for a feature.
    fn get_group_idx(&self, feature_idx: usize) -> usize {
        feature_idx / self.config.features_per_group
    }

    /// Performs forward pass on grouped embeddings.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, num_groups * features_per_group * embedding_dim]
    fn forward_impl(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let batch_size = input.shape()[0];
        let input_dim = input.shape()[1];
        let expected_dim = self.input_dim();

        if input_dim != expected_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: expected_dim,
                actual: input_dim,
            });
        }

        let total_features = self.config.num_groups * self.config.features_per_group;
        let output_dim = self.output_dim();
        let mut output_data = vec![0.0f32; batch_size * output_dim];

        for b in 0..batch_size {
            let input_start = b * input_dim;
            let input_slice = &input.data()[input_start..input_start + input_dim];
            let output_start = b * output_dim;
            let mut output_offset = 0;

            // Copy original embeddings if configured
            if self.config.include_original {
                for i in 0..total_features * self.config.embedding_dim {
                    output_data[output_start + output_offset + i] = input_slice[i];
                }
                output_offset += total_features * self.config.embedding_dim;
            }

            // Compute interactions
            for i in 0..total_features {
                let group_i = self.get_group_idx(i);
                let feature_i = i % self.config.features_per_group;
                let v_i = self.get_feature_embedding(input_slice, group_i, feature_i);

                for j in (i + 1)..total_features {
                    let group_j = self.get_group_idx(j);
                    let feature_j = j % self.config.features_per_group;
                    let v_j = self.get_feature_embedding(input_slice, group_j, feature_j);

                    let same_group = group_i == group_j;

                    // Check if this interaction should be computed
                    let should_compute = (same_group && self.config.intra_group)
                        || (!same_group && self.config.inter_group);

                    if should_compute {
                        let interaction = self.compute_interaction(&v_i, &v_j);
                        for (k, &val) in interaction.iter().enumerate() {
                            output_data[output_start + output_offset + k] = val;
                        }
                        output_offset += interaction.len();
                    }
                }
            }
        }

        Ok(Tensor::from_data(&[batch_size, output_dim], output_data))
    }

    /// Performs forward pass with caching for training.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward_impl(input)
    }
}

impl Layer for GroupInteractionLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Expected 2D input, got {}D", input.ndim()),
            });
        }
        self.forward_impl(input)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let batch_size = input.shape()[0];
        let input_dim = self.input_dim();
        let output_dim = self.output_dim();
        let total_features = self.config.num_groups * self.config.features_per_group;
        let embedding_dim = self.config.embedding_dim;

        // Initialize input gradient
        let mut input_grad = vec![0.0f32; batch_size * input_dim];

        for b in 0..batch_size {
            let input_start = b * input_dim;
            let input_slice = &input.data()[input_start..input_start + input_dim];
            let grad_start = b * output_dim;
            let mut grad_offset = 0;

            // Gradient for original embeddings
            if self.config.include_original {
                for i in 0..total_features * embedding_dim {
                    input_grad[input_start + i] += grad.data()[grad_start + grad_offset + i];
                }
                grad_offset += total_features * embedding_dim;
            }

            // Gradient for interactions
            for i in 0..total_features {
                let group_i = self.get_group_idx(i);
                let feature_i = i % self.config.features_per_group;

                for j in (i + 1)..total_features {
                    let group_j = self.get_group_idx(j);
                    let feature_j = j % self.config.features_per_group;

                    let same_group = group_i == group_j;
                    let should_compute = (same_group && self.config.intra_group)
                        || (!same_group && self.config.inter_group);

                    if should_compute {
                        let v_i = self.get_feature_embedding(input_slice, group_i, feature_i);
                        let v_j = self.get_feature_embedding(input_slice, group_j, feature_j);

                        let offset_i = i * embedding_dim;
                        let offset_j = j * embedding_dim;

                        match self.config.interaction_type {
                            InteractionType::InnerProduct => {
                                // d(v_i . v_j)/d(v_i) = v_j
                                // d(v_i . v_j)/d(v_j) = v_i
                                let g = grad.data()[grad_start + grad_offset];
                                for k in 0..embedding_dim {
                                    input_grad[input_start + offset_i + k] += g * v_j[k];
                                    input_grad[input_start + offset_j + k] += g * v_i[k];
                                }
                                grad_offset += 1;
                            }
                            InteractionType::Hadamard => {
                                // d(v_i * v_j)/d(v_i) = v_j (element-wise)
                                // d(v_i * v_j)/d(v_j) = v_i (element-wise)
                                for k in 0..embedding_dim {
                                    let g = grad.data()[grad_start + grad_offset + k];
                                    input_grad[input_start + offset_i + k] += g * v_j[k];
                                    input_grad[input_start + offset_j + k] += g * v_i[k];
                                }
                                grad_offset += embedding_dim;
                            }
                            InteractionType::Concat => {
                                // d(concat)/d(v_i) = first half of grad
                                // d(concat)/d(v_j) = second half of grad
                                for k in 0..embedding_dim {
                                    input_grad[input_start + offset_i + k] +=
                                        grad.data()[grad_start + grad_offset + k];
                                    input_grad[input_start + offset_j + k] +=
                                        grad.data()[grad_start + grad_offset + embedding_dim + k];
                                }
                                grad_offset += embedding_dim * 2;
                            }
                        }
                    }
                }
            }
        }

        Ok(Tensor::from_data(&[batch_size, input_dim], input_grad))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // This layer has no learnable parameters
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "GroupInteractionLayer"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// A wrapper that combines group interaction with learned transformations.
///
/// This layer applies a dense transformation after computing group interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupInteractionWithProjection {
    /// The group interaction layer
    interaction: GroupInteractionLayer,

    /// Projection weights
    weights: Tensor,

    /// Projection bias
    bias: Tensor,

    /// Output dimension after projection
    output_dim: usize,

    /// Gradient for weights
    weights_grad: Option<Tensor>,

    /// Gradient for bias
    bias_grad: Option<Tensor>,

    /// Cached interaction output for backward
    cached_interaction: Option<Tensor>,

    /// Whether in training mode
    training: bool,
}

impl GroupInteractionWithProjection {
    /// Creates a new Group Interaction layer with projection.
    ///
    /// # Arguments
    ///
    /// * `config` - The group interaction configuration
    /// * `output_dim` - The dimension after projection
    pub fn new(config: &GroupInteractionConfig, output_dim: usize) -> Self {
        let interaction = GroupInteractionLayer::from_config(config);
        let interaction_dim = config.output_dim();

        // Xavier initialization
        let std = (2.0 / (interaction_dim + output_dim) as f32).sqrt();
        let weights = Tensor::randn(&[interaction_dim, output_dim], 0.0, std);
        let bias = Tensor::zeros(&[output_dim]);

        Self {
            interaction,
            weights,
            bias,
            output_dim,
            weights_grad: None,
            bias_grad: None,
            cached_interaction: None,
            training: true,
        }
    }

    /// Returns a reference to the projection weights.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Returns a reference to the projection bias.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Clears cached values.
    pub fn clear_cache(&mut self) {
        self.interaction.clear_cache();
        self.cached_interaction = None;
        self.weights_grad = None;
        self.bias_grad = None;
    }
}

impl Layer for GroupInteractionWithProjection {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let interaction_output = self.interaction.forward(input)?;
        let projected = interaction_output.matmul(&self.weights);
        Ok(projected.add(&self.bias))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let interaction_output = self
            .cached_interaction
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Gradient for weights: interaction^T @ grad
        self.weights_grad = Some(interaction_output.transpose().matmul(grad));

        // Gradient for bias: sum over batch
        self.bias_grad = Some(grad.sum_axis(0));

        // Gradient for interaction: grad @ weights^T
        let interaction_grad = grad.matmul(&self.weights.transpose());

        // Backward through interaction layer
        self.interaction.backward(&interaction_grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weights, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn name(&self) -> &str {
        "GroupInteractionWithProjection"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.interaction.set_training(training);
    }
}

impl GroupInteractionWithProjection {
    /// Performs forward pass with caching for training.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let interaction_output = self.interaction.forward_train(input)?;
        self.cached_interaction = Some(interaction_output.clone());
        let projected = interaction_output.matmul(&self.weights);
        Ok(projected.add(&self.bias))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interaction_type_default() {
        let t = InteractionType::default();
        assert_eq!(t, InteractionType::InnerProduct);
    }

    #[test]
    fn test_config_creation() {
        let config = GroupInteractionConfig::new(3, 4, 8);
        assert_eq!(config.num_groups, 3);
        assert_eq!(config.features_per_group, 4);
        assert_eq!(config.embedding_dim, 8);
        assert!(config.inter_group);
        assert!(!config.intra_group);
        assert!(config.include_original);
    }

    #[test]
    fn test_config_builder() {
        let config = GroupInteractionConfig::new(2, 3, 4)
            .with_interaction_type(InteractionType::Hadamard)
            .with_inter_group(false)
            .with_intra_group(true)
            .with_include_original(false);

        assert_eq!(config.interaction_type, InteractionType::Hadamard);
        assert!(!config.inter_group);
        assert!(config.intra_group);
        assert!(!config.include_original);
    }

    #[test]
    fn test_config_output_dim_inner_product() {
        // 2 groups, 3 features each, embedding dim 4
        let config = GroupInteractionConfig::new(2, 3, 4)
            .with_inter_group(true)
            .with_intra_group(false)
            .with_include_original(true);

        // Original: 2 * 3 * 4 = 24
        // Inter-group pairs: group0 vs group1 = 3 * 3 = 9 pairs, each produces 1 value
        // Total: 24 + 9 = 33
        let output_dim = config.output_dim();
        assert_eq!(output_dim, 33);
    }

    #[test]
    fn test_config_output_dim_hadamard() {
        let config = GroupInteractionConfig::new(2, 3, 4)
            .with_interaction_type(InteractionType::Hadamard)
            .with_inter_group(true)
            .with_intra_group(false)
            .with_include_original(true);

        // Original: 24
        // Inter-group pairs: 9 pairs, each produces 4 values
        // Total: 24 + 36 = 60
        let output_dim = config.output_dim();
        assert_eq!(output_dim, 60);
    }

    #[test]
    fn test_layer_creation() {
        let config = GroupInteractionConfig::new(2, 3, 4);
        let layer = GroupInteractionLayer::from_config(&config);

        assert_eq!(layer.input_dim(), 24); // 2 * 3 * 4
    }

    #[test]
    fn test_layer_new() {
        let layer = GroupInteractionLayer::new(2, 3, 4);
        assert_eq!(layer.input_dim(), 24);
    }

    #[test]
    fn test_forward_basic() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_inter_group(true)
            .with_intra_group(false);

        let layer = GroupInteractionLayer::from_config(&config);

        // Input: [2, 16] = 2 batches, 2 groups * 2 features * 4 dim
        let input = Tensor::rand(&[2, 16]);
        let output = layer.forward(&input).unwrap();

        let expected_dim = config.output_dim();
        assert_eq!(output.shape(), &[2, expected_dim]);
    }

    #[test]
    fn test_forward_with_intra_group() {
        let config = GroupInteractionConfig::new(2, 3, 4)
            .with_inter_group(true)
            .with_intra_group(true);

        let layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 24]);
        let output = layer.forward(&input).unwrap();

        let expected_dim = config.output_dim();
        assert_eq!(output.shape(), &[2, expected_dim]);
    }

    #[test]
    fn test_forward_no_original() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_inter_group(true)
            .with_include_original(false);

        let layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 16]);
        let output = layer.forward(&input).unwrap();

        let expected_dim = config.output_dim();
        assert_eq!(output.shape(), &[2, expected_dim]);
    }

    #[test]
    fn test_forward_hadamard() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_interaction_type(InteractionType::Hadamard)
            .with_inter_group(true);

        let layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 16]);
        let output = layer.forward(&input).unwrap();

        let expected_dim = config.output_dim();
        assert_eq!(output.shape(), &[2, expected_dim]);
    }

    #[test]
    fn test_forward_concat() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_interaction_type(InteractionType::Concat)
            .with_inter_group(true);

        let layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 16]);
        let output = layer.forward(&input).unwrap();

        let expected_dim = config.output_dim();
        assert_eq!(output.shape(), &[2, expected_dim]);
    }

    #[test]
    fn test_forward_invalid_dim() {
        let config = GroupInteractionConfig::new(2, 2, 4);
        let layer = GroupInteractionLayer::from_config(&config);

        let input = Tensor::rand(&[2, 20]); // Wrong dimension
        let result = layer.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_inter_group(true)
            .with_intra_group(false);

        let mut layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 16]);

        // Forward with caching
        let output = layer.forward_train(&input).unwrap();

        // Backward
        let grad = Tensor::ones(&output.shape().to_vec());
        let input_grad = layer.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[2, 16]);
    }

    #[test]
    fn test_backward_hadamard() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_interaction_type(InteractionType::Hadamard)
            .with_inter_group(true);

        let mut layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 16]);

        let output = layer.forward_train(&input).unwrap();
        let grad = Tensor::ones(&output.shape().to_vec());
        let input_grad = layer.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[2, 16]);
    }

    #[test]
    fn test_backward_concat() {
        let config = GroupInteractionConfig::new(2, 2, 4)
            .with_interaction_type(InteractionType::Concat)
            .with_inter_group(true);

        let mut layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 16]);

        let output = layer.forward_train(&input).unwrap();
        let grad = Tensor::ones(&output.shape().to_vec());
        let input_grad = layer.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[2, 16]);
    }

    #[test]
    fn test_no_parameters() {
        let config = GroupInteractionConfig::new(2, 2, 4);
        let layer = GroupInteractionLayer::from_config(&config);

        assert_eq!(layer.parameters().len(), 0);
    }

    #[test]
    fn test_training_mode() {
        let mut layer = GroupInteractionLayer::new(2, 2, 4);
        assert!(layer.is_training());

        layer.set_training(false);
        assert!(!layer.is_training());
    }

    #[test]
    fn test_layer_name() {
        let layer = GroupInteractionLayer::new(2, 2, 4);
        assert_eq!(layer.name(), "GroupInteractionLayer");
    }

    #[test]
    fn test_clear_cache() {
        let mut layer = GroupInteractionLayer::new(2, 2, 4);
        let input = Tensor::rand(&[2, 16]);

        let _output = layer.forward_train(&input).unwrap();
        layer.clear_cache();

        // After clearing, backward should fail
        let grad = Tensor::ones(&[2, layer.output_dim()]);
        let result = layer.backward(&grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_accessors() {
        let config = GroupInteractionConfig::new(3, 4, 8);
        let layer = GroupInteractionLayer::from_config(&config);

        assert_eq!(layer.config().num_groups, 3);
        assert_eq!(layer.config().features_per_group, 4);
        assert_eq!(layer.config().embedding_dim, 8);
    }

    #[test]
    fn test_projection_layer_creation() {
        let config = GroupInteractionConfig::new(2, 2, 4);
        let layer = GroupInteractionWithProjection::new(&config, 16);

        assert_eq!(layer.weights().shape()[1], 16);
        assert_eq!(layer.bias().shape(), &[16]);
    }

    #[test]
    fn test_projection_layer_forward() {
        let config = GroupInteractionConfig::new(2, 2, 4).with_inter_group(true);
        let layer = GroupInteractionWithProjection::new(&config, 16);

        let input = Tensor::rand(&[2, 16]);
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_projection_layer_backward() {
        let config = GroupInteractionConfig::new(2, 2, 4).with_inter_group(true);
        let mut layer = GroupInteractionWithProjection::new(&config, 16);

        let input = Tensor::rand(&[2, 16]);
        let _output = layer.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 16]);
        let input_grad = layer.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[2, 16]);
    }

    #[test]
    fn test_projection_layer_parameters() {
        let config = GroupInteractionConfig::new(2, 2, 4);
        let layer = GroupInteractionWithProjection::new(&config, 16);

        assert_eq!(layer.parameters().len(), 2); // weights + bias
    }

    #[test]
    fn test_projection_layer_name() {
        let config = GroupInteractionConfig::new(2, 2, 4);
        let layer = GroupInteractionWithProjection::new(&config, 16);

        assert_eq!(layer.name(), "GroupInteractionWithProjection");
    }

    #[test]
    fn test_only_intra_group() {
        let config = GroupInteractionConfig::new(2, 3, 4)
            .with_inter_group(false)
            .with_intra_group(true)
            .with_include_original(false);

        let layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 24]);
        let output = layer.forward(&input).unwrap();

        // Only intra-group pairs: 2 groups * C(3,2) = 2 * 3 = 6 pairs
        // Inner product: 6 * 1 = 6
        assert_eq!(output.shape(), &[2, 6]);
    }

    #[test]
    fn test_single_feature_per_group() {
        let config = GroupInteractionConfig::new(3, 1, 4)
            .with_inter_group(true)
            .with_intra_group(true)
            .with_include_original(false);

        let layer = GroupInteractionLayer::from_config(&config);
        let input = Tensor::rand(&[2, 12]);
        let output = layer.forward(&input).unwrap();

        // No intra-group pairs (need at least 2 features)
        // Inter-group pairs: C(3,2) = 3 pairs
        assert_eq!(output.shape(), &[2, 3]);
    }
}
