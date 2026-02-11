//! Field-aware Factorization Machine (FFM) layer implementation.
//!
//! This module provides the FFM layer for modeling field-aware feature interactions
//! in recommendation systems. FFM extends traditional Factorization Machines by
//! learning field-specific embeddings for each feature.
//!
//! # Overview
//!
//! In FFM, each feature has multiple embedding vectors, one for each field it may
//! interact with. This allows the model to capture different interaction patterns
//! between features from different fields.
//!
//! # Mathematical Formulation
//!
//! The FFM output is computed as:
//! ```text
//! y = sum over all field pairs (i,j) where i < j of <v_{i,fj}, v_{j,fi}>
//! ```
//! where:
//! - `v_{i,fj}` is the embedding of feature i for interacting with field j
//! - `v_{j,fi}` is the embedding of feature j for interacting with field i
#![allow(clippy::needless_range_loop, clippy::manual_is_multiple_of)]
//! - `<., .>` denotes the inner product
//!
//! # Example
//!
//! ```
//! use monolith_layers::ffm::{FFMConfig, FFMLayer};
//! use monolith_layers::tensor::Tensor;
//! use monolith_layers::layer::Layer;
//!
//! // Create an FFM layer with 5 fields and 8-dimensional embeddings
//! let config = FFMConfig::new(5, 8);
//! let ffm = FFMLayer::from_config(&config);
//!
//! // Forward pass with field indices and values
//! let field_indices = Tensor::from_data(&[2, 5], vec![0.0, 1.0, 2.0, 3.0, 4.0,
//!                                                      0.0, 1.0, 2.0, 3.0, 4.0]);
//! let field_values = Tensor::ones(&[2, 5]);
//! let output = ffm.forward_with_fields(&field_indices, &field_values).unwrap();
//! assert_eq!(output.shape(), &[2, 1]);  // batch_size x 1
//! ```
//!
//! # References
//!
//! - [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf)

use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration for the Field-aware Factorization Machine layer.
///
/// # Example
///
/// ```
/// use monolith_layers::ffm::FFMConfig;
///
/// // Basic config with 10 fields and 16-dimensional embeddings
/// let config = FFMConfig::new(10, 16);
///
/// // Config with bias term
/// let config_with_bias = FFMConfig::new(10, 16).with_bias(true);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFMConfig {
    /// Number of fields in the dataset
    pub num_fields: usize,

    /// Dimension of the latent embedding vectors
    pub embedding_dim: usize,

    /// Whether to include a bias term in the output
    pub use_bias: bool,
}

impl FFMConfig {
    /// Creates a new FFM configuration with default settings (no bias).
    ///
    /// # Arguments
    ///
    /// * `num_fields` - The number of feature fields
    /// * `embedding_dim` - The dimension of the embedding vectors
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::ffm::FFMConfig;
    ///
    /// let config = FFMConfig::new(5, 8);
    /// assert_eq!(config.num_fields, 5);
    /// assert_eq!(config.embedding_dim, 8);
    /// assert!(!config.use_bias);
    /// ```
    pub fn new(num_fields: usize, embedding_dim: usize) -> Self {
        Self {
            num_fields,
            embedding_dim,
            use_bias: false,
        }
    }

    /// Sets whether to use a bias term.
    ///
    /// # Arguments
    ///
    /// * `use_bias` - Whether to include a bias term
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::ffm::FFMConfig;
    ///
    /// let config = FFMConfig::new(5, 8).with_bias(true);
    /// assert!(config.use_bias);
    /// ```
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Sets the embedding dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The embedding dimension
    pub fn with_embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }
}

/// Field-aware Factorization Machine layer.
///
/// This layer computes pairwise feature interactions where each feature
/// has field-specific embeddings. The output is the sum of inner products
/// between feature embeddings for each pair of fields.
///
/// # Example
///
/// ```
/// use monolith_layers::ffm::{FFMConfig, FFMLayer};
/// use monolith_layers::tensor::Tensor;
///
/// let config = FFMConfig::new(3, 4);
/// let ffm = FFMLayer::from_config(&config);
///
/// // Forward with field indices and values
/// let indices = Tensor::from_data(&[1, 3], vec![0.0, 1.0, 2.0]);
/// let values = Tensor::from_data(&[1, 3], vec![1.0, 1.0, 1.0]);
/// let output = ffm.forward_with_fields(&indices, &values).unwrap();
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FFMLayer {
    /// Field embeddings: shape [num_fields, num_fields, embedding_dim]
    /// embeddings[i][j] is the embedding of field i for interacting with field j
    embeddings: Tensor,

    /// Bias term (optional)
    bias: Tensor,

    /// Number of fields
    num_fields: usize,

    /// Embedding dimension
    embedding_dim: usize,

    /// Whether to use bias
    use_bias: bool,

    /// Gradient for embeddings
    embeddings_grad: Option<Tensor>,

    /// Gradient for bias
    bias_grad: Option<Tensor>,

    /// Cached field indices for backward pass
    cached_field_indices: Option<Tensor>,

    /// Cached field values for backward pass
    cached_field_values: Option<Tensor>,

    /// Whether in training mode
    training: bool,
}

impl FFMLayer {
    /// Creates a new FFM layer from configuration.
    ///
    /// Embeddings are initialized using Xavier/Glorot initialization.
    ///
    /// # Arguments
    ///
    /// * `config` - The FFM configuration
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::ffm::{FFMConfig, FFMLayer};
    ///
    /// let config = FFMConfig::new(5, 8);
    /// let ffm = FFMLayer::from_config(&config);
    /// ```
    pub fn from_config(config: &FFMConfig) -> Self {
        // Xavier initialization for embeddings
        let std = (2.0 / (config.num_fields + config.embedding_dim) as f32).sqrt();
        let total_size = config.num_fields * config.num_fields * config.embedding_dim;
        let embeddings = Tensor::randn(
            &[config.num_fields, config.num_fields, config.embedding_dim],
            0.0,
            std,
        );

        // Handle case where randn might not support 3D properly
        let embeddings = if embeddings.numel() != total_size {
            // Fallback: create and reshape
            let data: Vec<f32> = (0..total_size)
                .map(|i| {
                    // Simple pseudo-random initialization
                    let seed = (i as u64).wrapping_mul(1103515245).wrapping_add(12345);
                    let val = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
                    (val - 0.5) * 2.0 * std
                })
                .collect();
            Tensor::from_data(
                &[config.num_fields, config.num_fields, config.embedding_dim],
                data,
            )
        } else {
            embeddings
        };

        let bias = Tensor::zeros(&[1]);

        Self {
            embeddings,
            bias,
            num_fields: config.num_fields,
            embedding_dim: config.embedding_dim,
            use_bias: config.use_bias,
            embeddings_grad: None,
            bias_grad: None,
            cached_field_indices: None,
            cached_field_values: None,
            training: true,
        }
    }

    /// Creates a new FFM layer with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `num_fields` - Number of feature fields
    /// * `embedding_dim` - Dimension of embedding vectors
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::ffm::FFMLayer;
    ///
    /// let ffm = FFMLayer::new(5, 8);
    /// ```
    pub fn new(num_fields: usize, embedding_dim: usize) -> Self {
        let config = FFMConfig::new(num_fields, embedding_dim);
        Self::from_config(&config)
    }

    /// Creates a new FFM layer with bias.
    ///
    /// # Arguments
    ///
    /// * `num_fields` - Number of feature fields
    /// * `embedding_dim` - Dimension of embedding vectors
    pub fn new_with_bias(num_fields: usize, embedding_dim: usize) -> Self {
        let config = FFMConfig::new(num_fields, embedding_dim).with_bias(true);
        Self::from_config(&config)
    }

    /// Returns the number of fields.
    pub fn num_fields(&self) -> usize {
        self.num_fields
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Returns whether bias is used.
    pub fn use_bias(&self) -> bool {
        self.use_bias
    }

    /// Returns a reference to the embeddings tensor.
    pub fn embeddings(&self) -> &Tensor {
        &self.embeddings
    }

    /// Returns a reference to the bias tensor.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Returns the embedding gradient if available.
    pub fn embeddings_grad(&self) -> Option<&Tensor> {
        self.embeddings_grad.as_ref()
    }

    /// Returns the bias gradient if available.
    pub fn bias_grad(&self) -> Option<&Tensor> {
        self.bias_grad.as_ref()
    }

    /// Clears cached values and gradients.
    pub fn clear_cache(&mut self) {
        self.cached_field_indices = None;
        self.cached_field_values = None;
        self.embeddings_grad = None;
        self.bias_grad = None;
    }

    /// Gets the embedding for a specific field pair.
    ///
    /// Returns the embedding of field `field_i` for interacting with field `field_j`.
    ///
    /// # Arguments
    ///
    /// * `field_i` - The source field index
    /// * `field_j` - The target field index
    fn get_embedding(&self, field_i: usize, field_j: usize) -> Vec<f32> {
        let offset = (field_i * self.num_fields + field_j) * self.embedding_dim;
        let embeddings_data = self.embeddings.data_ref();
        embeddings_data[offset..offset + self.embedding_dim].to_vec()
    }

    /// Computes the inner product of two embedding vectors.
    #[allow(dead_code)]
    fn inner_product(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
    }

    /// Performs forward pass with explicit field indices and values.
    ///
    /// Computes the FFM output as the sum of pairwise field interactions:
    /// `sum_{i<j} <v_{i,fj} * x_i, v_{j,fi} * x_j>`
    ///
    /// # Arguments
    ///
    /// * `field_indices` - Tensor of field indices, shape [batch_size, num_features]
    /// * `field_values` - Tensor of feature values, shape [batch_size, num_features]
    ///
    /// # Returns
    ///
    /// Output tensor of shape [batch_size, 1]
    ///
    /// # Errors
    ///
    /// Returns error if input shapes are incompatible
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::ffm::{FFMConfig, FFMLayer};
    /// use monolith_layers::tensor::Tensor;
    ///
    /// let ffm = FFMLayer::new(3, 4);
    /// let indices = Tensor::from_data(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
    /// let values = Tensor::ones(&[2, 3]);
    /// let output = ffm.forward_with_fields(&indices, &values).unwrap();
    /// assert_eq!(output.shape(), &[2, 1]);
    /// ```
    pub fn forward_with_fields(
        &self,
        field_indices: &Tensor,
        field_values: &Tensor,
    ) -> Result<Tensor, LayerError> {
        // Validate shapes
        if field_indices.ndim() != 2 || field_values.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "Expected 2D inputs, got indices: {}D, values: {}D",
                    field_indices.ndim(),
                    field_values.ndim()
                ),
            });
        }

        if field_indices.shape() != field_values.shape() {
            return Err(LayerError::ShapeMismatch {
                expected: field_indices.shape().to_vec(),
                actual: field_values.shape().to_vec(),
            });
        }

        let batch_size = field_indices.shape()[0];
        let num_features = field_indices.shape()[1];

        let mut output_data = vec![0.0f32; batch_size];

        let indices_data = field_indices.data_ref();
        let values_data = field_values.data_ref();

        // For each sample in the batch
        for b in 0..batch_size {
            let mut interaction_sum = 0.0f32;

            // Compute pairwise interactions between all field pairs
            for i in 0..num_features {
                let field_i = indices_data[b * num_features + i] as usize;
                let value_i = values_data[b * num_features + i];

                if field_i >= self.num_fields {
                    return Err(LayerError::ForwardError {
                        message: format!(
                            "Field index {} out of bounds (num_fields={})",
                            field_i, self.num_fields
                        ),
                    });
                }

                for j in (i + 1)..num_features {
                    let field_j = indices_data[b * num_features + j] as usize;
                    let value_j = values_data[b * num_features + j];

                    if field_j >= self.num_fields {
                        return Err(LayerError::ForwardError {
                            message: format!(
                                "Field index {} out of bounds (num_fields={})",
                                field_j, self.num_fields
                            ),
                        });
                    }

                    // Get embeddings: v_{i, fj} and v_{j, fi}
                    let v_i_fj = self.get_embedding(field_i, field_j);
                    let v_j_fi = self.get_embedding(field_j, field_i);

                    // Compute <v_{i,fj} * x_i, v_{j,fi} * x_j>
                    let inner: f32 = v_i_fj.iter().zip(v_j_fi.iter()).map(|(a, b)| a * b).sum();

                    interaction_sum += inner * value_i * value_j;
                }
            }

            output_data[b] = interaction_sum;
        }

        // Add bias if enabled
        if self.use_bias {
            let bias_val = self.bias.data_ref()[0];
            for val in &mut output_data {
                *val += bias_val;
            }
        }

        Ok(Tensor::from_data(&[batch_size, 1], output_data))
    }

    /// Performs forward pass with caching for training.
    ///
    /// # Arguments
    ///
    /// * `field_indices` - Tensor of field indices
    /// * `field_values` - Tensor of feature values
    pub fn forward_train_with_fields(
        &mut self,
        field_indices: &Tensor,
        field_values: &Tensor,
    ) -> Result<Tensor, LayerError> {
        self.cached_field_indices = Some(field_indices.clone());
        self.cached_field_values = Some(field_values.clone());
        self.forward_with_fields(field_indices, field_values)
    }

    /// Backward pass for FFM layer.
    ///
    /// Computes gradients for embeddings and bias.
    ///
    /// # Arguments
    ///
    /// * `grad` - Gradient from the next layer, shape [batch_size, 1]
    pub fn backward_ffm(&mut self, grad: &Tensor) -> Result<(), LayerError> {
        let field_indices = self
            .cached_field_indices
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let field_values = self
            .cached_field_values
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let batch_size = field_indices.shape()[0];
        let num_features = field_indices.shape()[1];

        // Initialize gradient accumulator
        let mut embeddings_grad_data =
            vec![0.0f32; self.num_fields * self.num_fields * self.embedding_dim];

        let grad_data = grad.data_ref();
        let indices_data = field_indices.data_ref();
        let values_data = field_values.data_ref();

        // Compute gradients
        for b in 0..batch_size {
            let grad_val = grad_data[b];

            for i in 0..num_features {
                let field_i = indices_data[b * num_features + i] as usize;
                let value_i = values_data[b * num_features + i];

                for j in (i + 1)..num_features {
                    let field_j = indices_data[b * num_features + j] as usize;
                    let value_j = values_data[b * num_features + j];

                    // Gradient w.r.t. v_{i, fj}: grad * x_i * x_j * v_{j, fi}
                    let v_j_fi = self.get_embedding(field_j, field_i);
                    let offset_i_fj = (field_i * self.num_fields + field_j) * self.embedding_dim;
                    for k in 0..self.embedding_dim {
                        embeddings_grad_data[offset_i_fj + k] +=
                            grad_val * value_i * value_j * v_j_fi[k];
                    }

                    // Gradient w.r.t. v_{j, fi}: grad * x_i * x_j * v_{i, fj}
                    let v_i_fj = self.get_embedding(field_i, field_j);
                    let offset_j_fi = (field_j * self.num_fields + field_i) * self.embedding_dim;
                    for k in 0..self.embedding_dim {
                        embeddings_grad_data[offset_j_fi + k] +=
                            grad_val * value_i * value_j * v_i_fj[k];
                    }
                }
            }
        }

        self.embeddings_grad = Some(Tensor::from_data(
            &[self.num_fields, self.num_fields, self.embedding_dim],
            embeddings_grad_data,
        ));

        // Bias gradient
        if self.use_bias {
            let bias_grad: f32 = grad_data.iter().sum();
            self.bias_grad = Some(Tensor::from_data(&[1], vec![bias_grad]));
        }

        Ok(())
    }
}

impl Layer for FFMLayer {
    /// Forward pass using a combined input tensor.
    ///
    /// Expects input of shape [batch_size, num_fields * 2] where the first half
    /// contains field indices and the second half contains values.
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Expected 2D input, got {}D", input.ndim()),
            });
        }

        let batch_size = input.shape()[0];
        let features = input.shape()[1];

        if features % 2 != 0 {
            return Err(LayerError::ForwardError {
                message: "Input features must be even (indices + values)".to_string(),
            });
        }

        let num_features = features / 2;

        // Split input into indices and values
        let mut indices_data = vec![0.0f32; batch_size * num_features];
        let mut values_data = vec![0.0f32; batch_size * num_features];

        let input_data = input.data_ref();
        for b in 0..batch_size {
            for f in 0..num_features {
                indices_data[b * num_features + f] = input_data[b * features + f];
                values_data[b * num_features + f] = input_data[b * features + num_features + f];
            }
        }

        let field_indices = Tensor::from_data(&[batch_size, num_features], indices_data);
        let field_values = Tensor::from_data(&[batch_size, num_features], values_data);

        self.forward_with_fields(&field_indices, &field_values)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        self.backward_ffm(grad)?;
        // FFM doesn't have a meaningful input gradient (indices are discrete)
        // Return zero gradient of the same shape as the cached input
        let field_indices = self
            .cached_field_indices
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let batch_size = field_indices.shape()[0];
        let num_features = field_indices.shape()[1];
        Ok(Tensor::zeros(&[batch_size, num_features * 2]))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        if self.use_bias {
            vec![&self.embeddings, &self.bias]
        } else {
            vec![&self.embeddings]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        if self.use_bias {
            vec![&mut self.embeddings, &mut self.bias]
        } else {
            vec![&mut self.embeddings]
        }
    }

    fn name(&self) -> &str {
        "FFMLayer"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffm_config_creation() {
        let config = FFMConfig::new(5, 8);
        assert_eq!(config.num_fields, 5);
        assert_eq!(config.embedding_dim, 8);
        assert!(!config.use_bias);

        let config_with_bias = FFMConfig::new(5, 8).with_bias(true);
        assert!(config_with_bias.use_bias);
    }

    #[test]
    fn test_ffm_layer_creation() {
        let config = FFMConfig::new(4, 6);
        let ffm = FFMLayer::from_config(&config);

        assert_eq!(ffm.num_fields(), 4);
        assert_eq!(ffm.embedding_dim(), 6);
        assert!(!ffm.use_bias());
        assert_eq!(ffm.embeddings().shape(), &[4, 4, 6]);
    }

    #[test]
    fn test_ffm_layer_new() {
        let ffm = FFMLayer::new(3, 4);
        assert_eq!(ffm.num_fields(), 3);
        assert_eq!(ffm.embedding_dim(), 4);
    }

    #[test]
    fn test_ffm_layer_with_bias() {
        let ffm = FFMLayer::new_with_bias(3, 4);
        assert!(ffm.use_bias());
    }

    #[test]
    fn test_ffm_forward_with_fields() {
        let ffm = FFMLayer::new(3, 4);

        // Batch of 2 samples, 3 features each
        let field_indices = Tensor::from_data(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let field_values = Tensor::ones(&[2, 3]);

        let output = ffm
            .forward_with_fields(&field_indices, &field_values)
            .unwrap();
        assert_eq!(output.shape(), &[2, 1]);
    }

    #[test]
    fn test_ffm_forward_single_sample() {
        let ffm = FFMLayer::new(2, 4);

        let field_indices = Tensor::from_data(&[1, 2], vec![0.0, 1.0]);
        let field_values = Tensor::from_data(&[1, 2], vec![1.0, 1.0]);

        let output = ffm
            .forward_with_fields(&field_indices, &field_values)
            .unwrap();
        assert_eq!(output.shape(), &[1, 1]);
    }

    #[test]
    fn test_ffm_forward_with_bias() {
        let ffm = FFMLayer::new_with_bias(2, 4);

        let field_indices = Tensor::from_data(&[1, 2], vec![0.0, 1.0]);
        let field_values = Tensor::ones(&[1, 2]);

        let output = ffm
            .forward_with_fields(&field_indices, &field_values)
            .unwrap();
        assert_eq!(output.shape(), &[1, 1]);
    }

    #[test]
    fn test_ffm_forward_invalid_field_index() {
        let ffm = FFMLayer::new(3, 4);

        // Field index 5 is out of bounds for num_fields=3
        let field_indices = Tensor::from_data(&[1, 2], vec![0.0, 5.0]);
        let field_values = Tensor::ones(&[1, 2]);

        let result = ffm.forward_with_fields(&field_indices, &field_values);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffm_forward_shape_mismatch() {
        let ffm = FFMLayer::new(3, 4);

        let field_indices = Tensor::from_data(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let field_values = Tensor::ones(&[2, 2]); // Wrong shape

        let result = ffm.forward_with_fields(&field_indices, &field_values);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffm_layer_trait_forward() {
        let ffm = FFMLayer::new(3, 4);

        // Input: [batch_size, num_features * 2] = [2, 6]
        // First 3 values are indices, last 3 are values
        let input = Tensor::from_data(
            &[2, 6],
            vec![
                0.0, 1.0, 2.0, 1.0, 1.0, 1.0, // sample 1: indices + values
                0.0, 1.0, 2.0, 1.0, 1.0, 1.0, // sample 2: indices + values
            ],
        );

        let output = ffm.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 1]);
    }

    #[test]
    fn test_ffm_backward() {
        let mut ffm = FFMLayer::new(3, 4);

        let field_indices = Tensor::from_data(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let field_values = Tensor::ones(&[2, 3]);

        // Forward with caching
        let _output = ffm
            .forward_train_with_fields(&field_indices, &field_values)
            .unwrap();

        // Backward
        let grad = Tensor::ones(&[2, 1]);
        ffm.backward_ffm(&grad).unwrap();

        assert!(ffm.embeddings_grad().is_some());
    }

    #[test]
    fn test_ffm_backward_with_bias() {
        let mut ffm = FFMLayer::new_with_bias(3, 4);

        let field_indices = Tensor::from_data(&[2, 3], vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        let field_values = Tensor::ones(&[2, 3]);

        // Forward with caching
        let _output = ffm
            .forward_train_with_fields(&field_indices, &field_values)
            .unwrap();

        // Backward
        let grad = Tensor::ones(&[2, 1]);
        ffm.backward_ffm(&grad).unwrap();

        assert!(ffm.embeddings_grad().is_some());
        assert!(ffm.bias_grad().is_some());
    }

    #[test]
    fn test_ffm_parameters() {
        let ffm = FFMLayer::new(3, 4);
        let params = ffm.parameters();
        assert_eq!(params.len(), 1); // only embeddings

        let ffm_with_bias = FFMLayer::new_with_bias(3, 4);
        let params = ffm_with_bias.parameters();
        assert_eq!(params.len(), 2); // embeddings + bias
    }

    #[test]
    fn test_ffm_clear_cache() {
        let mut ffm = FFMLayer::new(3, 4);

        let field_indices = Tensor::from_data(&[1, 3], vec![0.0, 1.0, 2.0]);
        let field_values = Tensor::ones(&[1, 3]);

        let _output = ffm
            .forward_train_with_fields(&field_indices, &field_values)
            .unwrap();

        ffm.clear_cache();

        // After clearing, backward should fail
        let grad = Tensor::ones(&[1, 1]);
        let result = ffm.backward_ffm(&grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_ffm_training_mode() {
        let mut ffm = FFMLayer::new(3, 4);
        assert!(ffm.is_training());

        ffm.set_training(false);
        assert!(!ffm.is_training());

        ffm.set_training(true);
        assert!(ffm.is_training());
    }

    #[test]
    fn test_ffm_name() {
        let ffm = FFMLayer::new(3, 4);
        assert_eq!(ffm.name(), "FFMLayer");
    }

    #[test]
    fn test_ffm_with_different_values() {
        let ffm = FFMLayer::new(2, 4);

        let field_indices = Tensor::from_data(&[1, 2], vec![0.0, 1.0]);
        let field_values = Tensor::from_data(&[1, 2], vec![2.0, 3.0]);

        let output = ffm
            .forward_with_fields(&field_indices, &field_values)
            .unwrap();
        assert_eq!(output.shape(), &[1, 1]);

        // The output should be scaled by value_i * value_j = 2.0 * 3.0 = 6.0
        // compared to values of 1.0
    }

    #[test]
    fn test_ffm_single_feature() {
        let ffm = FFMLayer::new(1, 4);

        // With only one feature, there are no pairwise interactions
        let field_indices = Tensor::from_data(&[1, 1], vec![0.0]);
        let field_values = Tensor::from_data(&[1, 1], vec![1.0]);

        let output = ffm
            .forward_with_fields(&field_indices, &field_values)
            .unwrap();
        assert_eq!(output.shape(), &[1, 1]);
        // Output should be 0 since there are no pairs
        assert_eq!(output.data()[0], 0.0);
    }

    #[test]
    fn test_ffm_config_with_embedding_dim() {
        let config = FFMConfig::new(5, 8).with_embedding_dim(16);
        assert_eq!(config.embedding_dim, 16);
    }

    #[test]
    fn test_ffm_large_batch() {
        let ffm = FFMLayer::new(4, 8);

        let batch_size = 32;
        let num_features = 4;

        // Create field indices cycling through fields
        let indices_data: Vec<f32> = (0..batch_size * num_features)
            .map(|i| (i % num_features) as f32)
            .collect();
        let field_indices = Tensor::from_data(&[batch_size, num_features], indices_data);
        let field_values = Tensor::ones(&[batch_size, num_features]);

        let output = ffm
            .forward_with_fields(&field_indices, &field_values)
            .unwrap();
        assert_eq!(output.shape(), &[batch_size, 1]);
    }
}
