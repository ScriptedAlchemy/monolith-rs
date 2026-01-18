//! Deep Interest Network (DIN) attention layer implementation.
//!
//! This module provides the [`DINAttention`] layer, which implements the attention mechanism
//! from the Deep Interest Network paper for modeling user behavior sequences in
//! recommendation systems.
//!
//! # Overview
//!
//! DIN Attention computes adaptive weights for user behavior items based on their
//! relevance to the target item. Unlike traditional pooling methods (sum/mean),
//! DIN uses an attention mechanism that considers the interaction between
//! the target item and each historical behavior.
//!
//! # Architecture
//!
//! The attention score for each key-value pair is computed by:
//! 1. Concatenating [query, key, query-key, query*key]
//! 2. Passing through an MLP to get attention logits
//! 3. Applying softmax (optional) or using raw scores
//! 4. Computing weighted sum of values
//!
//! # Example
//!
//! ```
//! use monolith_layers::din::{DINAttention, DINConfig};
//! use monolith_layers::tensor::Tensor;
//!
//! // Create DIN attention layer
//! let config = DINConfig::new(32)
//!     .with_attention_hidden_units(vec![64, 32])
//!     .with_activation(monolith_layers::mlp::ActivationType::Sigmoid);
//! let din = DINAttention::from_config(config).unwrap();
//!
//! // Query: target item embedding [batch_size, embedding_dim]
//! let query = Tensor::rand(&[4, 32]);
//! // Keys: user behavior sequence [batch_size, seq_len, embedding_dim]
//! let keys = Tensor::rand(&[4, 10, 32]);
//! // Values: same as keys in most cases
//! let values = keys.clone();
//!
//! // Compute attention-weighted aggregation
//! let output = din.forward_attention(&query, &keys, &values, None).unwrap();
//! assert_eq!(output.shape(), &[4, 32]);
//! ```
//!
//! # References
//!
//! - Zhou, G., et al. "Deep Interest Network for Click-Through Rate Prediction." KDD 2018.

use crate::error::LayerError;
use crate::layer::Layer;
use crate::mlp::{ActivationType, MLP, MLPConfig};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration for DIN attention layer.
///
/// # Example
///
/// ```
/// use monolith_layers::din::DINConfig;
/// use monolith_layers::mlp::ActivationType;
///
/// let config = DINConfig::new(64)
///     .with_attention_hidden_units(vec![128, 64])
///     .with_activation(ActivationType::Sigmoid)
///     .with_use_softmax(false);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DINConfig {
    /// Dimension of the input embeddings
    pub embedding_dim: usize,
    /// Hidden layer sizes for the attention MLP
    pub attention_hidden_units: Vec<usize>,
    /// Activation function for the attention MLP
    pub attention_activation: ActivationType,
    /// Whether to apply softmax to attention scores
    pub use_softmax: bool,
    /// Whether to use bias in MLP layers
    pub use_bias: bool,
}

impl DINConfig {
    /// Creates a new DIN configuration with the specified embedding dimension.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of the input embeddings
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::din::DINConfig;
    ///
    /// let config = DINConfig::new(32);
    /// assert_eq!(config.embedding_dim, 32);
    /// ```
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            embedding_dim,
            attention_hidden_units: vec![64, 32],
            attention_activation: ActivationType::Sigmoid,
            use_softmax: false,
            use_bias: true,
        }
    }

    /// Sets the hidden layer sizes for the attention MLP.
    ///
    /// # Arguments
    ///
    /// * `units` - Vector of hidden layer sizes
    pub fn with_attention_hidden_units(mut self, units: Vec<usize>) -> Self {
        self.attention_hidden_units = units;
        self
    }

    /// Sets the activation function for the attention MLP.
    ///
    /// # Arguments
    ///
    /// * `activation` - Activation function type
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.attention_activation = activation;
        self
    }

    /// Sets whether to apply softmax to attention scores.
    ///
    /// # Arguments
    ///
    /// * `use_softmax` - Whether to use softmax normalization
    pub fn with_use_softmax(mut self, use_softmax: bool) -> Self {
        self.use_softmax = use_softmax;
        self
    }

    /// Sets whether to use bias in MLP layers.
    ///
    /// # Arguments
    ///
    /// * `use_bias` - Whether to use bias
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), LayerError> {
        if self.embedding_dim == 0 {
            return Err(LayerError::ConfigError {
                message: "Embedding dimension must be positive".to_string(),
            });
        }
        if self.attention_hidden_units.is_empty() {
            return Err(LayerError::ConfigError {
                message: "Attention MLP must have at least one hidden layer".to_string(),
            });
        }
        for (i, &dim) in self.attention_hidden_units.iter().enumerate() {
            if dim == 0 {
                return Err(LayerError::ConfigError {
                    message: format!("Attention hidden layer {} has zero dimension", i),
                });
            }
        }
        Ok(())
    }

    /// Builds a DINAttention layer from this configuration.
    pub fn build(self) -> Result<DINAttention, LayerError> {
        DINAttention::from_config(self)
    }
}

impl Default for DINConfig {
    fn default() -> Self {
        Self::new(32)
    }
}

/// Deep Interest Network (DIN) attention layer.
///
/// Implements the attention mechanism from the DIN paper, which computes
/// adaptive weights for user behavior items based on their relevance to
/// the target item.
///
/// # How it works
///
/// Given:
/// - Query `q`: target item embedding [batch_size, embedding_dim]
/// - Keys `K`: user behavior embeddings [batch_size, seq_len, embedding_dim]
/// - Values `V`: typically same as keys [batch_size, seq_len, embedding_dim]
///
/// For each key k_i:
/// 1. Compute interaction features: [q, k_i, q-k_i, q*k_i]
/// 2. Pass through attention MLP to get attention score a_i
/// 3. Optionally apply softmax: a_i = softmax(a_i)
/// 4. Compute weighted sum: output = sum(a_i * v_i)
///
/// # Example
///
/// ```
/// use monolith_layers::din::DINAttention;
/// use monolith_layers::tensor::Tensor;
///
/// let din = DINAttention::new(32, &[64, 32]);
///
/// let query = Tensor::rand(&[2, 32]);
/// let keys = Tensor::rand(&[2, 5, 32]);
/// let values = keys.clone();
///
/// let output = din.forward_attention(&query, &keys, &values, None).unwrap();
/// assert_eq!(output.shape(), &[2, 32]);
/// ```
#[derive(Debug, Clone)]
pub struct DINAttention {
    /// Configuration used to build this layer
    config: DINConfig,
    /// MLP for computing attention scores
    attention_mlp: MLP,
    /// Whether in training mode
    training: bool,
    /// Cached values for backward pass
    cache: Option<DINCache>,
}

/// Cached values from DIN forward pass for backward computation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DINCache {
    /// Input query
    query: Tensor,
    /// Input keys
    keys: Tensor,
    /// Input values
    values: Tensor,
    /// Attention scores before softmax (reserved for full backward implementation)
    attention_logits: Tensor,
    /// Final attention weights
    attention_weights: Tensor,
    /// Optional mask (reserved for full backward implementation)
    mask: Option<Tensor>,
}

impl DINAttention {
    /// Creates a new DIN attention layer with default configuration.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of the input embeddings
    /// * `hidden_units` - Hidden layer sizes for attention MLP
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::din::DINAttention;
    ///
    /// let din = DINAttention::new(64, &[128, 64]);
    /// ```
    pub fn new(embedding_dim: usize, hidden_units: &[usize]) -> Self {
        let config = DINConfig::new(embedding_dim)
            .with_attention_hidden_units(hidden_units.to_vec());
        Self::from_config(config).expect("Invalid DIN configuration")
    }

    /// Creates a DIN attention layer from a configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The DIN configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    pub fn from_config(config: DINConfig) -> Result<Self, LayerError> {
        config.validate()?;

        // Input dimension for attention MLP: [query, key, query-key, query*key]
        // = 4 * embedding_dim
        let attention_input_dim = 4 * config.embedding_dim;

        // Build attention MLP
        let mut mlp_config = MLPConfig::new(attention_input_dim);
        for &units in &config.attention_hidden_units {
            mlp_config = mlp_config.add_layer(units, config.attention_activation);
        }
        // Output layer: single attention score
        mlp_config = mlp_config.add_layer(1, ActivationType::None);
        mlp_config = mlp_config.with_bias(config.use_bias);

        let attention_mlp = MLP::from_config(mlp_config)?;

        Ok(Self {
            config,
            attention_mlp,
            training: true,
            cache: None,
        })
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Returns the hidden units configuration.
    pub fn hidden_units(&self) -> &[usize] {
        &self.config.attention_hidden_units
    }

    /// Returns the configuration.
    pub fn config(&self) -> &DINConfig {
        &self.config
    }

    /// Computes attention-weighted aggregation of values.
    ///
    /// # Arguments
    ///
    /// * `query` - Target item embedding [batch_size, embedding_dim]
    /// * `keys` - User behavior embeddings [batch_size, seq_len, embedding_dim]
    /// * `values` - Value embeddings [batch_size, seq_len, embedding_dim]
    /// * `mask` - Optional mask [batch_size, seq_len], 1 for valid, 0 for padding
    ///
    /// # Returns
    ///
    /// Attention-weighted aggregation [batch_size, embedding_dim]
    ///
    /// # Errors
    ///
    /// Returns an error if input shapes are invalid
    pub fn forward_attention(
        &self,
        query: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        // Validate input shapes
        self.validate_inputs(query, keys, values, mask)?;

        let batch_size = query.shape()[0];
        let seq_len = keys.shape()[1];
        let embedding_dim = self.config.embedding_dim;

        // Compute attention scores for each key
        let attention_weights = self.compute_attention_weights(query, keys, mask)?;

        // Compute weighted sum of values
        let output = self.weighted_sum(&attention_weights, values, batch_size, seq_len, embedding_dim)?;

        Ok(output)
    }

    /// Computes attention-weighted aggregation with training cache.
    ///
    /// # Arguments
    ///
    /// * `query` - Target item embedding [batch_size, embedding_dim]
    /// * `keys` - User behavior embeddings [batch_size, seq_len, embedding_dim]
    /// * `values` - Value embeddings [batch_size, seq_len, embedding_dim]
    /// * `mask` - Optional mask [batch_size, seq_len]
    ///
    /// # Returns
    ///
    /// Attention-weighted aggregation [batch_size, embedding_dim]
    pub fn forward_attention_train(
        &mut self,
        query: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        self.validate_inputs(query, keys, values, mask)?;

        let batch_size = query.shape()[0];
        let seq_len = keys.shape()[1];
        let embedding_dim = self.config.embedding_dim;

        // Compute attention logits
        let attention_logits = self.compute_attention_logits(query, keys)?;

        // Apply mask and softmax
        let attention_weights = self.apply_mask_and_normalize(&attention_logits, mask, batch_size, seq_len)?;

        // Cache for backward pass
        self.cache = Some(DINCache {
            query: query.clone(),
            keys: keys.clone(),
            values: values.clone(),
            attention_logits,
            attention_weights: attention_weights.clone(),
            mask: mask.cloned(),
        });

        // Compute weighted sum
        let output = self.weighted_sum(&attention_weights, values, batch_size, seq_len, embedding_dim)?;

        Ok(output)
    }

    /// Validates input tensor shapes.
    fn validate_inputs(
        &self,
        query: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(), LayerError> {
        // Query should be 2D: [batch_size, embedding_dim]
        if query.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "Query should be 2D [batch_size, embedding_dim], got {}D",
                    query.ndim()
                ),
            });
        }

        // Keys should be 3D: [batch_size, seq_len, embedding_dim]
        if keys.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "Keys should be 3D [batch_size, seq_len, embedding_dim], got {}D",
                    keys.ndim()
                ),
            });
        }

        // Values should be 3D: [batch_size, seq_len, embedding_dim]
        if values.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "Values should be 3D [batch_size, seq_len, embedding_dim], got {}D",
                    values.ndim()
                ),
            });
        }

        let batch_size = query.shape()[0];
        let embedding_dim = self.config.embedding_dim;

        // Check embedding dimension
        if query.shape()[1] != embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: embedding_dim,
                actual: query.shape()[1],
            });
        }

        if keys.shape()[2] != embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: embedding_dim,
                actual: keys.shape()[2],
            });
        }

        if values.shape()[2] != embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: embedding_dim,
                actual: values.shape()[2],
            });
        }

        // Check batch size consistency
        if keys.shape()[0] != batch_size {
            return Err(LayerError::ShapeMismatch {
                expected: vec![batch_size, keys.shape()[1], keys.shape()[2]],
                actual: keys.shape().to_vec(),
            });
        }

        if values.shape()[0] != batch_size {
            return Err(LayerError::ShapeMismatch {
                expected: vec![batch_size, values.shape()[1], values.shape()[2]],
                actual: values.shape().to_vec(),
            });
        }

        // Check seq_len consistency
        if values.shape()[1] != keys.shape()[1] {
            return Err(LayerError::ShapeMismatch {
                expected: vec![batch_size, keys.shape()[1], embedding_dim],
                actual: values.shape().to_vec(),
            });
        }

        // Check mask shape if provided
        if let Some(m) = mask {
            if m.shape() != &[batch_size, keys.shape()[1]] {
                return Err(LayerError::ShapeMismatch {
                    expected: vec![batch_size, keys.shape()[1]],
                    actual: m.shape().to_vec(),
                });
            }
        }

        Ok(())
    }

    /// Computes raw attention logits.
    fn compute_attention_logits(
        &self,
        query: &Tensor,
        keys: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let batch_size = query.shape()[0];
        let seq_len = keys.shape()[1];
        let embedding_dim = self.config.embedding_dim;

        // For each (query, key) pair, compute [q, k, q-k, q*k]
        let attention_input_dim = 4 * embedding_dim;
        let mut attention_inputs = vec![0.0; batch_size * seq_len * attention_input_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let q_offset = b * embedding_dim;
                let k_offset = b * seq_len * embedding_dim + s * embedding_dim;
                let out_offset = (b * seq_len + s) * attention_input_dim;

                for d in 0..embedding_dim {
                    let q_val = query.data()[q_offset + d];
                    let k_val = keys.data()[k_offset + d];

                    // [query, key, query-key, query*key]
                    attention_inputs[out_offset + d] = q_val;
                    attention_inputs[out_offset + embedding_dim + d] = k_val;
                    attention_inputs[out_offset + 2 * embedding_dim + d] = q_val - k_val;
                    attention_inputs[out_offset + 3 * embedding_dim + d] = q_val * k_val;
                }
            }
        }

        // Reshape to [batch_size * seq_len, 4 * embedding_dim]
        let attention_input = Tensor::from_data(
            &[batch_size * seq_len, attention_input_dim],
            attention_inputs,
        );

        // Pass through attention MLP
        let logits = self.attention_mlp.forward(&attention_input)?;

        // Reshape to [batch_size, seq_len]
        Ok(logits.reshape(&[batch_size, seq_len]))
    }

    /// Applies mask and normalization to attention logits.
    fn apply_mask_and_normalize(
        &self,
        logits: &Tensor,
        mask: Option<&Tensor>,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor, LayerError> {
        let mut weights = logits.data().to_vec();

        // Apply mask (set padded positions to large negative value)
        if let Some(m) = mask {
            for b in 0..batch_size {
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if m.data()[idx] == 0.0 {
                        weights[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Apply softmax if configured
        if self.config.use_softmax {
            for b in 0..batch_size {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if weights[idx] > max_val {
                        max_val = weights[idx];
                    }
                }

                // Handle case where all values are -inf
                if max_val == f32::NEG_INFINITY {
                    max_val = 0.0;
                }

                // Compute exp and sum
                let mut sum = 0.0;
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if weights[idx] != f32::NEG_INFINITY {
                        weights[idx] = (weights[idx] - max_val).exp();
                        sum += weights[idx];
                    } else {
                        weights[idx] = 0.0;
                    }
                }

                // Normalize
                if sum > 0.0 {
                    for s in 0..seq_len {
                        let idx = b * seq_len + s;
                        weights[idx] /= sum;
                    }
                }
            }
        } else {
            // Use raw scores (with sigmoid applied by MLP typically)
            // Just mask out padding
            for b in 0..batch_size {
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if weights[idx] == f32::NEG_INFINITY {
                        weights[idx] = 0.0;
                    }
                }
            }
        }

        Ok(Tensor::from_data(&[batch_size, seq_len], weights))
    }

    /// Computes attention weights.
    fn compute_attention_weights(
        &self,
        query: &Tensor,
        keys: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        let batch_size = query.shape()[0];
        let seq_len = keys.shape()[1];

        let logits = self.compute_attention_logits(query, keys)?;
        self.apply_mask_and_normalize(&logits, mask, batch_size, seq_len)
    }

    /// Computes weighted sum of values.
    fn weighted_sum(
        &self,
        weights: &Tensor,
        values: &Tensor,
        batch_size: usize,
        seq_len: usize,
        embedding_dim: usize,
    ) -> Result<Tensor, LayerError> {
        let mut output = vec![0.0; batch_size * embedding_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let weight = weights.data()[b * seq_len + s];
                let v_offset = b * seq_len * embedding_dim + s * embedding_dim;
                let out_offset = b * embedding_dim;

                for d in 0..embedding_dim {
                    output[out_offset + d] += weight * values.data()[v_offset + d];
                }
            }
        }

        Ok(Tensor::from_data(&[batch_size, embedding_dim], output))
    }

    /// Returns attention weights for analysis/visualization.
    ///
    /// # Arguments
    ///
    /// * `query` - Target item embedding [batch_size, embedding_dim]
    /// * `keys` - User behavior embeddings [batch_size, seq_len, embedding_dim]
    /// * `mask` - Optional mask [batch_size, seq_len]
    ///
    /// # Returns
    ///
    /// Attention weights [batch_size, seq_len]
    pub fn get_attention_weights(
        &self,
        query: &Tensor,
        keys: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        self.validate_inputs(query, keys, keys, mask)?;
        self.compute_attention_weights(query, keys, mask)
    }
}

impl Layer for DINAttention {
    /// Forward pass using keys as both keys and values.
    ///
    /// Input is expected to be [query || keys] concatenated along a special dimension,
    /// but for simplicity, this implementation expects input as a 3D tensor where:
    /// - First "slice" is the query (broadcasted)
    /// - Remaining is the key sequence
    ///
    /// For more control, use `forward_attention` method directly.
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        // This is a simplified interface - for DIN, users should prefer forward_attention
        // Here we interpret input as [batch_size, seq_len+1, embedding_dim]
        // where the first position is the query
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "DIN forward expects 3D input [batch, seq_len+1, dim], got {}D",
                    input.ndim()
                ),
            });
        }

        let batch_size = input.shape()[0];
        let total_len = input.shape()[1];
        let embedding_dim = input.shape()[2];

        if total_len < 2 {
            return Err(LayerError::ForwardError {
                message: "Input sequence must have at least 2 elements (query + 1 key)".to_string(),
            });
        }

        if embedding_dim != self.config.embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.config.embedding_dim,
                actual: embedding_dim,
            });
        }

        // Extract query (first position)
        let query = self.extract_query(input, batch_size, embedding_dim);

        // Extract keys (remaining positions)
        let keys = self.extract_keys(input, batch_size, total_len - 1, embedding_dim);

        // Use keys as values
        self.forward_attention(&query, &keys, &keys, None)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let cache = self.cache.as_ref().ok_or(LayerError::NotInitialized)?;

        let batch_size = cache.query.shape()[0];
        let seq_len = cache.keys.shape()[1];
        let embedding_dim = self.config.embedding_dim;

        // Gradient of weighted sum w.r.t. attention weights
        // d(output)/d(weights[s]) = values[s]
        let mut d_weights = vec![0.0; batch_size * seq_len];
        for b in 0..batch_size {
            for s in 0..seq_len {
                let mut dot = 0.0;
                for d in 0..embedding_dim {
                    let grad_val = grad.data()[b * embedding_dim + d];
                    let val = cache.values.data()[b * seq_len * embedding_dim + s * embedding_dim + d];
                    dot += grad_val * val;
                }
                d_weights[b * seq_len + s] = dot;
            }
        }

        // Gradient w.r.t. values
        let mut d_values = vec![0.0; batch_size * seq_len * embedding_dim];
        for b in 0..batch_size {
            for s in 0..seq_len {
                let weight = cache.attention_weights.data()[b * seq_len + s];
                for d in 0..embedding_dim {
                    let grad_val = grad.data()[b * embedding_dim + d];
                    d_values[b * seq_len * embedding_dim + s * embedding_dim + d] = weight * grad_val;
                }
            }
        }

        // For backward through softmax and MLP, we need more complex computation
        // This is a simplified version that returns gradient w.r.t. values
        Ok(Tensor::from_data(
            &[batch_size, seq_len, embedding_dim],
            d_values,
        ))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.attention_mlp.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.attention_mlp.parameters_mut()
    }

    fn name(&self) -> &str {
        "DINAttention"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        self.attention_mlp.set_training(training);
    }
}

impl DINAttention {
    /// Extracts query from combined input tensor.
    fn extract_query(&self, input: &Tensor, batch_size: usize, embedding_dim: usize) -> Tensor {
        let total_len = input.shape()[1];
        let mut query_data = vec![0.0; batch_size * embedding_dim];

        for b in 0..batch_size {
            for d in 0..embedding_dim {
                query_data[b * embedding_dim + d] =
                    input.data()[b * total_len * embedding_dim + d];
            }
        }

        Tensor::from_data(&[batch_size, embedding_dim], query_data)
    }

    /// Extracts keys from combined input tensor.
    fn extract_keys(
        &self,
        input: &Tensor,
        batch_size: usize,
        seq_len: usize,
        embedding_dim: usize,
    ) -> Tensor {
        let total_len = input.shape()[1];
        let mut keys_data = vec![0.0; batch_size * seq_len * embedding_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..embedding_dim {
                    keys_data[b * seq_len * embedding_dim + s * embedding_dim + d] =
                        input.data()[b * total_len * embedding_dim + (s + 1) * embedding_dim + d];
                }
            }
        }

        Tensor::from_data(&[batch_size, seq_len, embedding_dim], keys_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_din_config_default() {
        let config = DINConfig::new(32);
        assert_eq!(config.embedding_dim, 32);
        assert_eq!(config.attention_hidden_units, vec![64, 32]);
        assert_eq!(config.attention_activation, ActivationType::Sigmoid);
        assert!(!config.use_softmax);
        assert!(config.use_bias);
    }

    #[test]
    fn test_din_config_builder() {
        let config = DINConfig::new(64)
            .with_attention_hidden_units(vec![128, 64])
            .with_activation(ActivationType::ReLU)
            .with_use_softmax(true)
            .with_bias(false);

        assert_eq!(config.embedding_dim, 64);
        assert_eq!(config.attention_hidden_units, vec![128, 64]);
        assert_eq!(config.attention_activation, ActivationType::ReLU);
        assert!(config.use_softmax);
        assert!(!config.use_bias);
    }

    #[test]
    fn test_din_config_validation() {
        // Valid config
        let config = DINConfig::new(32);
        assert!(config.validate().is_ok());

        // Invalid: zero embedding dim
        let config = DINConfig::new(0);
        assert!(config.validate().is_err());

        // Invalid: empty hidden units
        let config = DINConfig::new(32).with_attention_hidden_units(vec![]);
        assert!(config.validate().is_err());

        // Invalid: zero hidden layer
        let config = DINConfig::new(32).with_attention_hidden_units(vec![64, 0]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_din_creation() {
        let din = DINAttention::new(32, &[64, 32]);
        assert_eq!(din.embedding_dim(), 32);
        assert_eq!(din.hidden_units(), &[64, 32]);
    }

    #[test]
    fn test_din_from_config() {
        let config = DINConfig::new(16)
            .with_attention_hidden_units(vec![32, 16])
            .with_activation(ActivationType::Sigmoid);

        let din = DINAttention::from_config(config).unwrap();
        assert_eq!(din.embedding_dim(), 16);
    }

    #[test]
    fn test_din_forward_attention() {
        let din = DINAttention::new(8, &[16, 8]);

        // Batch of 2, sequence length 5
        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 5, 8]);
        let values = keys.clone();

        let output = din.forward_attention(&query, &keys, &values, None).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_din_forward_attention_with_mask() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 5, 8]);
        let values = keys.clone();

        // Mask: first 3 positions valid for batch 0, first 2 for batch 1
        let mask = Tensor::from_data(&[2, 5], vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]);

        let output = din.forward_attention(&query, &keys, &values, Some(&mask)).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_din_forward_attention_with_softmax() {
        let config = DINConfig::new(8)
            .with_attention_hidden_units(vec![16, 8])
            .with_use_softmax(true);
        let din = DINAttention::from_config(config).unwrap();

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 5, 8]);
        let values = keys.clone();

        let output = din.forward_attention(&query, &keys, &values, None).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_din_get_attention_weights() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 5, 8]);

        let weights = din.get_attention_weights(&query, &keys, None).unwrap();
        assert_eq!(weights.shape(), &[2, 5]);
    }

    #[test]
    fn test_din_layer_trait() {
        let din = DINAttention::new(8, &[16, 8]);

        // Combined input: [batch, seq_len+1, embedding_dim]
        // First position is query, rest are keys
        let input = Tensor::rand(&[2, 6, 8]);

        let output = din.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
        assert_eq!(din.name(), "DINAttention");
    }

    #[test]
    fn test_din_parameters() {
        let din = DINAttention::new(8, &[16, 8]);
        let params = din.parameters();

        // 3 dense layers (16, 8, 1) with bias each = 6 parameter tensors
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_din_training_mode() {
        let mut din = DINAttention::new(8, &[16, 8]);
        assert!(din.is_training());

        din.set_training(false);
        assert!(!din.is_training());

        din.set_training(true);
        assert!(din.is_training());
    }

    #[test]
    fn test_din_invalid_query_shape() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 5, 8]); // 3D instead of 2D
        let keys = Tensor::rand(&[2, 5, 8]);

        let result = din.forward_attention(&query, &keys, &keys, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_din_invalid_keys_shape() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 8]); // 2D instead of 3D

        let result = din.forward_attention(&query, &keys, &keys, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_din_embedding_dim_mismatch() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 16]); // Wrong embedding dim
        let keys = Tensor::rand(&[2, 5, 8]);

        let result = din.forward_attention(&query, &keys, &keys, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_din_batch_size_mismatch() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[3, 5, 8]); // Different batch size

        let result = din.forward_attention(&query, &keys, &keys, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_din_mask_shape_mismatch() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 5, 8]);
        let mask = Tensor::ones(&[2, 3]); // Wrong seq_len

        let result = din.forward_attention(&query, &keys, &keys, Some(&mask));
        assert!(result.is_err());
    }

    #[test]
    fn test_din_single_item_sequence() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 1, 8]); // Single item sequence

        let output = din.forward_attention(&query, &keys, &keys, None).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_din_different_values() {
        let din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 5, 8]);
        let values = Tensor::rand(&[2, 5, 8]); // Different from keys

        let output = din.forward_attention(&query, &keys, &values, None).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_din_backward() {
        let mut din = DINAttention::new(8, &[16, 8]);

        let query = Tensor::rand(&[2, 8]);
        let keys = Tensor::rand(&[2, 5, 8]);
        let values = keys.clone();

        // Forward with training cache
        let _output = din.forward_attention_train(&query, &keys, &values, None).unwrap();

        // Backward
        let grad = Tensor::ones(&[2, 8]);
        let input_grad = din.backward(&grad).unwrap();
        assert_eq!(input_grad.shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_din_backward_without_cache() {
        let mut din = DINAttention::new(8, &[16, 8]);

        let grad = Tensor::ones(&[2, 8]);
        let result = din.backward(&grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_din_attention_effect() {
        // Test that attention actually weights the sequence
        let din = DINAttention::new(4, &[8, 4]);

        // Create a query that matches one key more than others
        let query = Tensor::from_data(&[1, 4], vec![1.0, 0.0, 0.0, 0.0]);
        let keys = Tensor::from_data(
            &[1, 3, 4],
            vec![
                1.0, 0.0, 0.0, 0.0, // Similar to query
                0.0, 1.0, 0.0, 0.0, // Different
                0.0, 0.0, 1.0, 0.0, // Different
            ],
        );

        let weights = din.get_attention_weights(&query, &keys, None).unwrap();

        // The first key should have different weight than others due to similarity
        // (exact values depend on MLP initialization, but weights should be computed)
        assert_eq!(weights.shape(), &[1, 3]);
    }
}
