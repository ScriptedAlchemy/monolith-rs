//! SENet (Squeeze-and-Excitation Network) layer for feature importance.
//!
//! This module provides the [`SENetLayer`] which implements the squeeze-and-excitation
//! mechanism for learning feature importance weights. The SE block consists of:
//!
//! 1. **Squeeze**: Global average pooling on the feature dimension
//! 2. **Excitation**: Two FC layers (reduction -> expansion) with ReLU and Sigmoid
//! 3. **Reweight**: Element-wise multiplication of input features by attention weights
//!
//! # Reference
//!
//! Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
//! In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
//!
//! # Example
//!
//! ```
//! use monolith_layers::senet::{SENetLayer, SENetConfig};
//! use monolith_layers::tensor::Tensor;
//! use monolith_layers::layer::Layer;
//!
//! let config = SENetConfig::new(64)
//!     .with_reduction_ratio(4)
//!     .with_bias(true);
//!
//! let senet = SENetLayer::from_config(config).unwrap();
//! let input = Tensor::rand(&[32, 64]);  // batch of 32, 64 features
//! let output = senet.forward(&input).unwrap();
//! assert_eq!(output.shape(), &[32, 64]);
//! ```

use crate::dense::Dense;
use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration for the SENet layer.
///
/// # Example
///
/// ```
/// use monolith_layers::senet::SENetConfig;
///
/// let config = SENetConfig::new(128)
///     .with_reduction_ratio(8)
///     .with_bias(true);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SENetConfig {
    /// Number of input features.
    pub input_dim: usize,
    /// Reduction ratio for the squeeze operation (default: 4).
    /// The bottleneck dimension will be input_dim / reduction_ratio.
    pub reduction_ratio: usize,
    /// Whether to use bias in FC layers (default: true).
    pub use_bias: bool,
}

impl SENetConfig {
    /// Creates a new SENet configuration with the specified input dimension.
    ///
    /// Default values:
    /// - reduction_ratio: 4
    /// - use_bias: true
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            reduction_ratio: 4,
            use_bias: true,
        }
    }

    /// Sets the reduction ratio for the squeeze operation.
    ///
    /// The bottleneck dimension will be input_dim / reduction_ratio.
    /// A higher ratio means more compression in the bottleneck.
    ///
    /// # Arguments
    ///
    /// * `ratio` - The reduction ratio (typically 4, 8, or 16)
    pub fn with_reduction_ratio(mut self, ratio: usize) -> Self {
        self.reduction_ratio = ratio;
        self
    }

    /// Sets whether to use bias in the FC layers.
    ///
    /// # Arguments
    ///
    /// * `use_bias` - Whether to include bias terms
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Builds a SENetLayer from this configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid (e.g., reduction_ratio is 0
    /// or larger than input_dim).
    pub fn build(self) -> Result<SENetLayer, LayerError> {
        SENetLayer::from_config(self)
    }
}

/// Squeeze-and-Excitation Network layer for learning feature importance.
///
/// The SE block adaptively recalibrates channel-wise (feature-wise) responses
/// by explicitly modeling interdependencies between features. It consists of:
///
/// 1. **Squeeze**: Compute global statistics via average pooling
/// 2. **Excitation**: Learn feature importance via two FC layers with non-linearities
/// 3. **Scale**: Reweight the original features by the learned importance weights
///
/// For 2D inputs of shape `[batch_size, features]`:
/// - Squeeze: Compute mean across batch (conceptually, we treat each feature as a channel)
/// - Actually, for feature recalibration, we keep the batch dimension and compute
///   attention weights per sample
///
/// # Architecture
///
/// ```text
/// Input [B, C] --> Squeeze (identity for 2D) --> FC1 [C -> C/r] --> ReLU
///                                                     |
///                                                     v
///                                           FC2 [C/r -> C] --> Sigmoid
///                                                     |
///                                                     v
///                                           Attention weights [B, C]
///                                                     |
///                                                     v
/// Output = Input * Attention weights [B, C]
/// ```
///
/// # Example
///
/// ```
/// use monolith_layers::senet::SENetLayer;
/// use monolith_layers::tensor::Tensor;
/// use monolith_layers::layer::Layer;
///
/// let senet = SENetLayer::new(64, 4, true);
/// let input = Tensor::rand(&[8, 64]);
/// let output = senet.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[8, 64]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SENetLayer {
    /// First FC layer for reduction (squeeze)
    fc1: Dense,
    /// Second FC layer for expansion (excitation)
    fc2: Dense,
    /// Input dimension (number of features)
    input_dim: usize,
    /// Bottleneck dimension (input_dim / reduction_ratio)
    bottleneck_dim: usize,
    /// Reduction ratio
    reduction_ratio: usize,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Cached attention weights for backward pass
    cached_attention: Option<Tensor>,
    /// Cached intermediate activations for backward pass
    cached_fc1_output: Option<Tensor>,
    /// Training mode flag
    training: bool,
}

impl SENetLayer {
    /// Creates a new SENet layer with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features
    /// * `reduction_ratio` - Reduction ratio for bottleneck (typically 4, 8, or 16)
    /// * `use_bias` - Whether to use bias in FC layers
    ///
    /// # Panics
    ///
    /// Panics if reduction_ratio is 0 or if input_dim / reduction_ratio is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::senet::SENetLayer;
    ///
    /// let senet = SENetLayer::new(128, 4, true);
    /// ```
    pub fn new(input_dim: usize, reduction_ratio: usize, use_bias: bool) -> Self {
        assert!(reduction_ratio > 0, "reduction_ratio must be positive");
        let bottleneck_dim = (input_dim / reduction_ratio).max(1);

        let fc1 = if use_bias {
            Dense::new(input_dim, bottleneck_dim)
        } else {
            Dense::new_no_bias(input_dim, bottleneck_dim)
        };

        let fc2 = if use_bias {
            Dense::new(bottleneck_dim, input_dim)
        } else {
            Dense::new_no_bias(bottleneck_dim, input_dim)
        };

        Self {
            fc1,
            fc2,
            input_dim,
            bottleneck_dim,
            reduction_ratio,
            cached_input: None,
            cached_attention: None,
            cached_fc1_output: None,
            training: true,
        }
    }

    /// Creates a SENet layer from a configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The SENet configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn from_config(config: SENetConfig) -> Result<Self, LayerError> {
        if config.reduction_ratio == 0 {
            return Err(LayerError::ConfigError {
                message: "reduction_ratio must be positive".to_string(),
            });
        }
        if config.input_dim == 0 {
            return Err(LayerError::ConfigError {
                message: "input_dim must be positive".to_string(),
            });
        }
        let bottleneck_dim = config.input_dim / config.reduction_ratio;
        if bottleneck_dim == 0 {
            return Err(LayerError::ConfigError {
                message: format!(
                    "Bottleneck dimension is 0: input_dim={} / reduction_ratio={} = 0. \
                     Use a smaller reduction_ratio.",
                    config.input_dim, config.reduction_ratio
                ),
            });
        }

        Ok(Self::new(
            config.input_dim,
            config.reduction_ratio,
            config.use_bias,
        ))
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the bottleneck dimension.
    pub fn bottleneck_dim(&self) -> usize {
        self.bottleneck_dim
    }

    /// Returns the reduction ratio.
    pub fn reduction_ratio(&self) -> usize {
        self.reduction_ratio
    }

    /// Returns the learned attention weights from the last forward pass.
    ///
    /// This can be useful for analyzing which features the model considers important.
    pub fn last_attention_weights(&self) -> Option<&Tensor> {
        self.cached_attention.as_ref()
    }

    /// Clears cached values.
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.cached_attention = None;
        self.cached_fc1_output = None;
    }

    /// Applies ReLU activation element-wise.
    fn relu(x: &Tensor) -> Tensor {
        x.map(|v| v.max(0.0))
    }

    /// Applies Sigmoid activation element-wise.
    fn sigmoid(x: &Tensor) -> Tensor {
        x.map(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Computes the derivative of ReLU.
    fn relu_grad(x: &Tensor) -> Tensor {
        x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    /// Computes the derivative of Sigmoid given the sigmoid output.
    fn sigmoid_grad(sigmoid_output: &Tensor) -> Tensor {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let one_minus = sigmoid_output.map(|v| 1.0 - v);
        sigmoid_output.mul(&one_minus)
    }

    /// Performs forward pass and caches intermediate values for backward pass.
    ///
    /// Use this method during training to enable gradient computation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// Output tensor of the same shape as input, with features reweighted
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Validate input
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Expected 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[1],
            });
        }

        // Cache input
        self.cached_input = Some(input.clone());

        // Squeeze and Excitation
        // For 2D feature input, we directly apply the excitation network
        // FC1: input_dim -> bottleneck_dim, then ReLU
        let fc1_out = self.fc1.forward(input)?;
        let fc1_relu = Self::relu(&fc1_out);
        self.cached_fc1_output = Some(fc1_out);

        // FC2: bottleneck_dim -> input_dim, then Sigmoid
        let fc2_out = self.fc2.forward(&fc1_relu)?;
        let attention = Self::sigmoid(&fc2_out);
        self.cached_attention = Some(attention.clone());

        // Scale: element-wise multiplication
        let output = self.scale_features(input, &attention);

        Ok(output)
    }

    /// Scales input features by attention weights with broadcasting.
    fn scale_features(&self, input: &Tensor, attention: &Tensor) -> Tensor {
        input.mul(attention)
    }
}

impl Layer for SENetLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Validate input
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Expected 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[1],
            });
        }

        // Squeeze and Excitation
        // FC1: input_dim -> bottleneck_dim, then ReLU
        let fc1_out = self.fc1.forward(input)?;
        let fc1_relu = Self::relu(&fc1_out);

        // FC2: bottleneck_dim -> input_dim, then Sigmoid
        let fc2_out = self.fc2.forward(&fc1_relu)?;
        let attention = Self::sigmoid(&fc2_out);

        // Scale: element-wise multiplication
        let output = self.scale_features(input, &attention);

        Ok(output)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let attention = self
            .cached_attention
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let fc1_output = self
            .cached_fc1_output
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Validate gradient shape
        if grad.shape() != input.shape() {
            return Err(LayerError::ShapeMismatch {
                expected: input.shape().to_vec(),
                actual: grad.shape().to_vec(),
            });
        }

        // Backward through scale: output = input * attention
        // d_loss/d_input = d_loss/d_output * attention
        // d_loss/d_attention = d_loss/d_output * input
        let d_attention = grad.mul(input);
        let d_input_from_scale = grad.mul(attention);

        // Backward through sigmoid
        let sigmoid_grad = Self::sigmoid_grad(attention);
        let d_fc2_out = d_attention.mul(&sigmoid_grad);

        // Backward through FC2
        // We need to compute gradients for fc2's weights
        let _fc1_relu = Self::relu(fc1_output);

        // For proper backward, we would need to call fc2.backward()
        // But since fc2 didn't cache inputs, we compute the input gradient manually
        // d_loss/d_fc1_relu = d_fc2_out @ fc2.weights^T
        let d_fc1_relu = d_fc2_out.matmul(&self.fc2.weights().transpose());

        // Backward through ReLU
        let relu_grad = Self::relu_grad(fc1_output);
        let d_fc1_out = d_fc1_relu.mul(&relu_grad);

        // Backward through FC1
        // d_loss/d_input_from_excitation = d_fc1_out @ fc1.weights^T
        let d_input_from_excitation = d_fc1_out.matmul(&self.fc1.weights().transpose());

        // Total gradient is sum of gradients from both paths
        let d_input = d_input_from_scale.add(&d_input_from_excitation);

        Ok(d_input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.fc1.parameters_mut();
        params.extend(self.fc2.parameters_mut());
        params
    }

    fn name(&self) -> &str {
        "SENetLayer"
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
    fn test_senet_config_defaults() {
        let config = SENetConfig::new(64);
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.reduction_ratio, 4);
        assert!(config.use_bias);
    }

    #[test]
    fn test_senet_config_builder() {
        let config = SENetConfig::new(128)
            .with_reduction_ratio(8)
            .with_bias(false);

        assert_eq!(config.input_dim, 128);
        assert_eq!(config.reduction_ratio, 8);
        assert!(!config.use_bias);
    }

    #[test]
    fn test_senet_creation() {
        let senet = SENetLayer::new(64, 4, true);
        assert_eq!(senet.input_dim(), 64);
        assert_eq!(senet.bottleneck_dim(), 16);
        assert_eq!(senet.reduction_ratio(), 4);
    }

    #[test]
    fn test_senet_from_config() {
        let config = SENetConfig::new(128).with_reduction_ratio(8);
        let senet = SENetLayer::from_config(config).unwrap();

        assert_eq!(senet.input_dim(), 128);
        assert_eq!(senet.bottleneck_dim(), 16);
    }

    #[test]
    fn test_senet_from_config_error() {
        // Test invalid reduction ratio
        let config = SENetConfig {
            input_dim: 64,
            reduction_ratio: 0,
            use_bias: true,
        };
        assert!(SENetLayer::from_config(config).is_err());

        // Test bottleneck dimension becoming 0
        let config = SENetConfig {
            input_dim: 4,
            reduction_ratio: 8,
            use_bias: true,
        };
        assert!(SENetLayer::from_config(config).is_err());
    }

    #[test]
    fn test_senet_forward_shape() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 64]);

        let output = senet.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8, 64]);
    }

    #[test]
    fn test_senet_forward_preserves_batch_size() {
        let senet = SENetLayer::new(32, 4, true);

        // Test with different batch sizes
        for batch_size in [1, 4, 16, 32] {
            let input = Tensor::rand(&[batch_size, 32]);
            let output = senet.forward(&input).unwrap();
            assert_eq!(output.shape(), &[batch_size, 32]);
        }
    }

    #[test]
    fn test_senet_forward_invalid_input_dim() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 32]); // Wrong input dimension

        let result = senet.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_senet_forward_invalid_input_ndim() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 4, 64]); // 3D instead of 2D

        let result = senet.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_senet_attention_range() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 64]);

        let _output = senet.forward(&input).unwrap();

        // After forward, we can't access cached_attention since forward doesn't cache
        // Use forward_train instead
        let mut senet_train = SENetLayer::new(64, 4, true);
        let _output = senet_train.forward_train(&input).unwrap();

        if let Some(attention) = senet_train.last_attention_weights() {
            // All attention weights should be between 0 and 1 (sigmoid output)
            for &val in attention.data() {
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Attention weight {} out of range",
                    val
                );
            }
        }
    }

    #[test]
    fn test_senet_parameters() {
        let senet = SENetLayer::new(64, 4, true);
        let params = senet.parameters();

        // FC1 has weights and bias, FC2 has weights and bias
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_senet_parameters_no_bias() {
        let senet = SENetLayer::new(64, 4, false);
        let params = senet.parameters();

        // FC1 has only weights, FC2 has only weights
        assert_eq!(params.len(), 2);
    }

    #[test]
    fn test_senet_backward() {
        let mut senet = SENetLayer::new(32, 4, true);
        let input = Tensor::rand(&[4, 32]);

        // Forward pass with caching
        let _output = senet.forward_train(&input).unwrap();

        // Backward pass
        let grad = Tensor::ones(&[4, 32]);
        let input_grad = senet.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[4, 32]);
    }

    #[test]
    fn test_senet_backward_without_forward() {
        let mut senet = SENetLayer::new(32, 4, true);
        let grad = Tensor::ones(&[4, 32]);

        let result = senet.backward(&grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_senet_training_mode() {
        let mut senet = SENetLayer::new(64, 4, true);
        assert!(senet.is_training());

        senet.set_training(false);
        assert!(!senet.is_training());

        senet.set_training(true);
        assert!(senet.is_training());
    }

    #[test]
    fn test_senet_name() {
        let senet = SENetLayer::new(64, 4, true);
        assert_eq!(senet.name(), "SENetLayer");
    }

    #[test]
    fn test_senet_clear_cache() {
        let mut senet = SENetLayer::new(32, 4, true);
        let input = Tensor::rand(&[4, 32]);

        let _output = senet.forward_train(&input).unwrap();
        assert!(senet.last_attention_weights().is_some());

        senet.clear_cache();
        assert!(senet.last_attention_weights().is_none());
    }

    #[test]
    fn test_senet_different_reduction_ratios() {
        for ratio in [2, 4, 8, 16] {
            let senet = SENetLayer::new(64, ratio, true);
            let expected_bottleneck = 64 / ratio;
            assert_eq!(senet.bottleneck_dim(), expected_bottleneck);

            let input = Tensor::rand(&[4, 64]);
            let output = senet.forward(&input).unwrap();
            assert_eq!(output.shape(), &[4, 64]);
        }
    }

    #[test]
    fn test_senet_with_ones_input() {
        let senet = SENetLayer::new(16, 4, true);
        let input = Tensor::ones(&[2, 16]);

        let output = senet.forward(&input).unwrap();

        // Output should be input * attention
        // Since input is all ones, output should equal attention weights
        // (scaled by the learned attention)
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_senet_config_build() {
        let config = SENetConfig::new(64).with_reduction_ratio(4);
        let senet = config.build().unwrap();

        assert_eq!(senet.input_dim(), 64);
        assert_eq!(senet.bottleneck_dim(), 16);
    }

    #[test]
    fn test_senet_minimum_bottleneck() {
        // When input_dim / reduction_ratio < 1, bottleneck should be at least 1
        let senet = SENetLayer::new(3, 4, true);
        assert_eq!(senet.bottleneck_dim(), 1); // max(3/4, 1) = max(0, 1) = 1
    }
}
