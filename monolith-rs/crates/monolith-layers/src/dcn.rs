//! Deep & Cross Network (DCN) layer implementations.
//!
//! This module provides the DCN components for modeling explicit feature interactions
//! in recommendation systems. DCN efficiently captures feature interactions of bounded
//! degrees through cross layers.
//!
//! # Overview
//!
//! The Deep & Cross Network architecture consists of:
//! - **Cross Network**: Explicitly models feature interactions through cross layers
//! - **Deep Network**: Traditional MLP for implicit feature learning
//!
//! This module implements the cross network component with support for both:
//! - **DCN (Vector)**: Original DCN with vector weights
//! - **DCN-V2 (Matrix)**: Improved DCN with matrix weights for richer interactions
//!
//! # Mathematical Formulation
//!
//! Each cross layer computes:
//! ```text
//! x_{l+1} = x_0 * (w^T * x_l + b) + x_l
//! ```
//! where:
//! - `x_0` is the original input (base layer)
//! - `x_l` is the output from the previous cross layer
//! - `w` is the weight vector (or matrix in DCN-V2)
//! - `b` is the bias
//!
//! # Example
//!
//! ```
//! use monolith_layers::dcn::{CrossNetwork, DCNConfig, DCNMode};
//! use monolith_layers::tensor::Tensor;
//! use monolith_layers::layer::Layer;
//!
//! // Create a cross network with 3 layers
//! let config = DCNConfig::new(64, 3);
//! let cross_network = CrossNetwork::from_config(&config);
//!
//! // Forward pass
//! let input = Tensor::rand(&[32, 64]);  // batch of 32, 64 features
//! let output = cross_network.forward(&input).unwrap();
//! assert_eq!(output.shape(), &[32, 64]);
//! ```
//!
//! # References
//!
//! - [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
//! - [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)

use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Mode for the DCN cross layer computation.
///
/// Determines whether weights are vectors (original DCN) or matrices (DCN-V2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DCNMode {
    /// Original DCN: weight is a vector of shape [d, 1]
    ///
    /// This is more memory efficient but has limited expressiveness.
    #[default]
    Vector,

    /// DCN-V2: weight is a matrix of shape [d, d]
    ///
    /// This provides richer feature interaction modeling at the cost of more parameters.
    Matrix,
}

/// Configuration for the Deep & Cross Network.
///
/// # Example
///
/// ```
/// use monolith_layers::dcn::{DCNConfig, DCNMode};
///
/// // Create config for 3-layer cross network with vector weights
/// let config = DCNConfig::new(128, 3);
///
/// // Create config for DCN-V2 with matrix weights
/// let config_v2 = DCNConfig::new(128, 3).with_mode(DCNMode::Matrix);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DCNConfig {
    /// Input dimension (feature dimension)
    pub input_dim: usize,

    /// Number of cross layers
    pub num_cross_layers: usize,

    /// DCN mode (Vector or Matrix)
    pub mode: DCNMode,
}

impl DCNConfig {
    /// Creates a new DCN configuration with default Vector mode.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - The input feature dimension
    /// * `num_cross_layers` - Number of cross layers to stack
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::dcn::DCNConfig;
    ///
    /// let config = DCNConfig::new(64, 3);
    /// assert_eq!(config.input_dim, 64);
    /// assert_eq!(config.num_cross_layers, 3);
    /// ```
    pub fn new(input_dim: usize, num_cross_layers: usize) -> Self {
        Self {
            input_dim,
            num_cross_layers,
            mode: DCNMode::default(),
        }
    }

    /// Sets the DCN mode (Vector or Matrix).
    ///
    /// # Arguments
    ///
    /// * `mode` - The DCN mode to use
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::dcn::{DCNConfig, DCNMode};
    ///
    /// let config = DCNConfig::new(64, 3).with_mode(DCNMode::Matrix);
    /// assert_eq!(config.mode, DCNMode::Matrix);
    /// ```
    pub fn with_mode(mut self, mode: DCNMode) -> Self {
        self.mode = mode;
        self
    }
}

/// A single cross layer in the DCN architecture.
///
/// Computes: `x_{l+1} = x_0 * (w^T * x_l + b) + x_l`
///
/// where `x_0` is the original input, `x_l` is the current input,
/// `w` is the weight (vector or matrix), and `b` is the bias.
///
/// # Example
///
/// ```
/// use monolith_layers::dcn::{CrossLayer, DCNMode};
/// use monolith_layers::tensor::Tensor;
/// use monolith_layers::layer::Layer;
///
/// let layer = CrossLayer::new(32, DCNMode::Vector);
/// let x0 = Tensor::rand(&[8, 32]);  // original input
/// let xl = Tensor::rand(&[8, 32]);  // current layer input
///
/// let output = layer.forward_with_x0(&x0, &xl).unwrap();
/// assert_eq!(output.shape(), &[8, 32]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLayer {
    /// Weight tensor: [d, 1] for Vector mode, [d, d] for Matrix mode
    weight: Tensor,

    /// Bias tensor of shape [d]
    bias: Tensor,

    /// Input dimension
    input_dim: usize,

    /// DCN mode (Vector or Matrix)
    mode: DCNMode,

    /// Gradient for weight
    weight_grad: Option<Tensor>,

    /// Gradient for bias
    bias_grad: Option<Tensor>,

    /// Cached x0 for backward pass
    cached_x0: Option<Tensor>,

    /// Cached xl for backward pass
    cached_xl: Option<Tensor>,

    /// Whether in training mode
    training: bool,
}

impl CrossLayer {
    /// Creates a new cross layer with the specified dimension and mode.
    ///
    /// Weights are initialized using Xavier/Glorot initialization.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - The input/output dimension
    /// * `mode` - The DCN mode (Vector or Matrix)
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::dcn::{CrossLayer, DCNMode};
    ///
    /// let layer = CrossLayer::new(64, DCNMode::Vector);
    /// ```
    pub fn new(input_dim: usize, mode: DCNMode) -> Self {
        let weight = match mode {
            DCNMode::Vector => {
                // Xavier initialization for vector: [d, 1]
                let std = (2.0 / (input_dim + 1) as f32).sqrt();
                Tensor::randn(&[input_dim, 1], 0.0, std)
            }
            DCNMode::Matrix => {
                // Xavier initialization for matrix: [d, d]
                let std = (2.0 / (input_dim + input_dim) as f32).sqrt();
                Tensor::randn(&[input_dim, input_dim], 0.0, std)
            }
        };

        let bias = Tensor::zeros(&[input_dim]);

        Self {
            weight,
            bias,
            input_dim,
            mode,
            weight_grad: None,
            bias_grad: None,
            cached_x0: None,
            cached_xl: None,
            training: true,
        }
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the DCN mode.
    pub fn mode(&self) -> DCNMode {
        self.mode
    }

    /// Returns a reference to the weight tensor.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Returns a reference to the bias tensor.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Returns the weight gradient if available.
    pub fn weight_grad(&self) -> Option<&Tensor> {
        self.weight_grad.as_ref()
    }

    /// Returns the bias gradient if available.
    pub fn bias_grad(&self) -> Option<&Tensor> {
        self.bias_grad.as_ref()
    }

    /// Clears cached values and gradients.
    pub fn clear_cache(&mut self) {
        self.cached_x0 = None;
        self.cached_xl = None;
        self.weight_grad = None;
        self.bias_grad = None;
    }

    /// Performs forward pass with explicit x0 (base input).
    ///
    /// Computes: `x_{l+1} = x_0 * (w^T * x_l + b) + x_l`
    ///
    /// # Arguments
    ///
    /// * `x0` - The original input tensor (base layer)
    /// * `xl` - The current layer input tensor
    ///
    /// # Returns
    ///
    /// The output tensor of the same shape as input
    ///
    /// # Errors
    ///
    /// Returns error if input shapes are incompatible
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::dcn::{CrossLayer, DCNMode};
    /// use monolith_layers::tensor::Tensor;
    ///
    /// let layer = CrossLayer::new(32, DCNMode::Vector);
    /// let x0 = Tensor::rand(&[4, 32]);
    /// let xl = Tensor::rand(&[4, 32]);
    ///
    /// let output = layer.forward_with_x0(&x0, &xl).unwrap();
    /// ```
    pub fn forward_with_x0(&self, x0: &Tensor, xl: &Tensor) -> Result<Tensor, LayerError> {
        // Validate shapes
        if x0.ndim() != 2 || xl.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "Expected 2D inputs, got x0: {}D, xl: {}D",
                    x0.ndim(),
                    xl.ndim()
                ),
            });
        }

        if x0.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: x0.shape()[1],
            });
        }

        if xl.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: xl.shape()[1],
            });
        }

        if x0.shape()[0] != xl.shape()[0] {
            return Err(LayerError::ShapeMismatch {
                expected: vec![x0.shape()[0], self.input_dim],
                actual: vec![xl.shape()[0], xl.shape()[1]],
            });
        }

        let batch_size = x0.shape()[0];

        // Compute based on mode
        let output = match self.mode {
            DCNMode::Vector => {
                // x_{l+1} = x_0 * (w^T * x_l + b) + x_l
                // w is [d, 1], so w^T is [1, d]
                // x_l @ w gives [batch, 1]
                // Then we need to broadcast multiply with x_0

                // Step 1: xl @ w -> [batch, 1]
                let xl_w = xl.matmul(&self.weight);

                // Step 2: Add bias (broadcast) -> [batch, d]
                // xl_w is [batch, 1], need to broadcast to [batch, d] and add bias
                let mut interaction = vec![0.0f32; batch_size * self.input_dim];
                for i in 0..batch_size {
                    let scalar = xl_w.data()[i];
                    for j in 0..self.input_dim {
                        interaction[i * self.input_dim + j] = scalar + self.bias.data()[j];
                    }
                }
                let interaction = Tensor::from_data(&[batch_size, self.input_dim], interaction);

                // Step 3: x_0 * interaction (element-wise)
                let cross = x0.mul(&interaction);

                // Step 4: Add residual xl
                cross.add(xl)
            }
            DCNMode::Matrix => {
                // x_{l+1} = x_0 * (x_l @ W + b) + x_l
                // W is [d, d]

                // Step 1: xl @ W -> [batch, d]
                let xl_w = xl.matmul(&self.weight);

                // Step 2: Add bias -> [batch, d]
                let interaction = xl_w.add(&self.bias);

                // Step 3: x_0 * interaction (element-wise)
                let cross = x0.mul(&interaction);

                // Step 4: Add residual xl
                cross.add(xl)
            }
        };

        Ok(output)
    }

    /// Performs forward pass with caching for training.
    ///
    /// # Arguments
    ///
    /// * `x0` - The original input tensor (base layer)
    /// * `xl` - The current layer input tensor
    pub fn forward_train_with_x0(
        &mut self,
        x0: &Tensor,
        xl: &Tensor,
    ) -> Result<Tensor, LayerError> {
        self.cached_x0 = Some(x0.clone());
        self.cached_xl = Some(xl.clone());
        self.forward_with_x0(x0, xl)
    }
}

impl Layer for CrossLayer {
    /// Forward pass using input as both x0 and xl.
    ///
    /// This is useful for the first cross layer where x0 = xl = input.
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.forward_with_x0(input, input)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let x0 = self
            .cached_x0
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let xl = self
            .cached_xl
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let batch_size = grad.shape()[0];

        // Compute gradients based on mode
        match self.mode {
            DCNMode::Vector => {
                // Forward: interaction = x_l @ w + b (broadcast)
                //          cross = x_0 * interaction
                //          output = cross + x_l
                //
                // Backward:
                // d_cross = grad (since output = cross + xl)
                // d_xl_residual = grad
                //
                // d_interaction = d_cross * x_0 (element-wise)
                // d_x0 = d_cross * interaction (element-wise)
                //
                // For vector mode:
                // interaction_scalar = x_l @ w -> [batch, 1]
                // interaction = broadcast(interaction_scalar) + b
                //
                // d_b = sum(d_interaction, axis=0)
                // d_interaction_scalar = sum(d_interaction, axis=1) -> [batch, 1]
                // d_w = x_l^T @ d_interaction_scalar -> [d, 1]
                // d_xl_from_w = d_interaction_scalar @ w^T -> [batch, d]

                // Recompute interaction for gradient
                let xl_w = xl.matmul(&self.weight);
                let mut interaction_data = vec![0.0f32; batch_size * self.input_dim];
                for i in 0..batch_size {
                    let scalar = xl_w.data()[i];
                    for j in 0..self.input_dim {
                        interaction_data[i * self.input_dim + j] = scalar + self.bias.data()[j];
                    }
                }
                let interaction =
                    Tensor::from_data(&[batch_size, self.input_dim], interaction_data);

                // d_interaction = grad * x0
                let d_interaction = grad.mul(x0);

                // d_bias = sum(d_interaction, axis=0)
                self.bias_grad = Some(d_interaction.sum_axis(0));

                // d_interaction_scalar = sum(d_interaction, axis=1) -> [batch]
                let d_interaction_scalar_vec = d_interaction.sum_axis(1);
                let d_interaction_scalar =
                    d_interaction_scalar_vec.reshape(&[batch_size, 1]);

                // d_weight = xl^T @ d_interaction_scalar -> [d, 1]
                self.weight_grad = Some(xl.transpose().matmul(&d_interaction_scalar));

                // d_xl from weight path = d_interaction_scalar @ w^T
                let d_xl_w = d_interaction_scalar.matmul(&self.weight.transpose());

                // d_x0 = grad * interaction
                let _d_x0 = grad.mul(&interaction);

                // Total d_xl = grad (residual) + d_xl_w
                let d_xl = grad.add(&d_xl_w);

                Ok(d_xl)
            }
            DCNMode::Matrix => {
                // Forward: interaction = x_l @ W + b
                //          cross = x_0 * interaction
                //          output = cross + x_l
                //
                // Backward:
                // d_cross = grad
                // d_interaction = d_cross * x_0
                //
                // d_b = sum(d_interaction, axis=0)
                // d_W = x_l^T @ d_interaction
                // d_xl_from_w = d_interaction @ W^T
                //
                // d_xl = grad (residual) + d_xl_from_w

                // Recompute interaction
                let xl_w = xl.matmul(&self.weight);
                let interaction = xl_w.add(&self.bias);

                // d_interaction = grad * x0
                let d_interaction = grad.mul(x0);

                // d_bias = sum(d_interaction, axis=0)
                self.bias_grad = Some(d_interaction.sum_axis(0));

                // d_weight = xl^T @ d_interaction
                self.weight_grad = Some(xl.transpose().matmul(&d_interaction));

                // d_xl from weight path
                let d_xl_w = d_interaction.matmul(&self.weight.transpose());

                // d_x0 = grad * interaction
                let _d_x0 = grad.mul(&interaction);

                // Total d_xl = grad (residual) + d_xl_from_w
                let d_xl = grad.add(&d_xl_w);

                Ok(d_xl)
            }
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }

    fn name(&self) -> &str {
        "CrossLayer"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// A stack of cross layers forming the cross network in DCN.
///
/// The cross network applies multiple cross layers sequentially,
/// with each layer receiving x0 (original input) and the output
/// from the previous layer.
///
/// # Example
///
/// ```
/// use monolith_layers::dcn::{CrossNetwork, DCNConfig, DCNMode};
/// use monolith_layers::tensor::Tensor;
/// use monolith_layers::layer::Layer;
///
/// // Create a 3-layer cross network
/// let config = DCNConfig::new(64, 3);
/// let cross_net = CrossNetwork::from_config(&config);
///
/// let input = Tensor::rand(&[8, 64]);
/// let output = cross_net.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[8, 64]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossNetwork {
    /// Stack of cross layers
    layers: Vec<CrossLayer>,

    /// Number of cross layers
    num_layers: usize,

    /// Input dimension
    input_dim: usize,

    /// DCN mode
    mode: DCNMode,

    /// Cached x0 for backward pass
    cached_x0: Option<Tensor>,

    /// Whether in training mode
    training: bool,
}

impl CrossNetwork {
    /// Creates a new cross network with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The DCN configuration
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::dcn::{CrossNetwork, DCNConfig};
    ///
    /// let config = DCNConfig::new(64, 3);
    /// let cross_net = CrossNetwork::from_config(&config);
    /// ```
    pub fn from_config(config: &DCNConfig) -> Self {
        let layers: Vec<CrossLayer> = (0..config.num_cross_layers)
            .map(|_| CrossLayer::new(config.input_dim, config.mode))
            .collect();

        Self {
            layers,
            num_layers: config.num_cross_layers,
            input_dim: config.input_dim,
            mode: config.mode,
            cached_x0: None,
            training: true,
        }
    }

    /// Creates a new cross network with the given parameters.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - The input feature dimension
    /// * `num_layers` - Number of cross layers
    /// * `mode` - The DCN mode (Vector or Matrix)
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::dcn::{CrossNetwork, DCNMode};
    ///
    /// let cross_net = CrossNetwork::new(64, 3, DCNMode::Vector);
    /// ```
    pub fn new(input_dim: usize, num_layers: usize, mode: DCNMode) -> Self {
        let config = DCNConfig {
            input_dim,
            num_cross_layers: num_layers,
            mode,
        };
        Self::from_config(&config)
    }

    /// Returns the number of cross layers.
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the DCN mode.
    pub fn mode(&self) -> DCNMode {
        self.mode
    }

    /// Returns a reference to the cross layers.
    pub fn layers(&self) -> &[CrossLayer] {
        &self.layers
    }

    /// Returns a mutable reference to the cross layers.
    pub fn layers_mut(&mut self) -> &mut [CrossLayer] {
        &mut self.layers
    }

    /// Clears all cached values and gradients.
    pub fn clear_cache(&mut self) {
        self.cached_x0 = None;
        for layer in &mut self.layers {
            layer.clear_cache();
        }
    }

    /// Performs forward pass with caching for training.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_x0 = Some(input.clone());

        let x0 = input;
        let mut xl = input.clone();

        for layer in &mut self.layers {
            xl = layer.forward_train_with_x0(x0, &xl)?;
        }

        Ok(xl)
    }
}

impl Layer for CrossNetwork {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let x0 = input;
        let mut xl = input.clone();

        for layer in &self.layers {
            xl = layer.forward_with_x0(x0, &xl)?;
        }

        Ok(xl)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let _x0 = self
            .cached_x0
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Backward through layers in reverse order
        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward(&current_grad)?;
        }

        Ok(current_grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }

    fn name(&self) -> &str {
        "CrossNetwork"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dcn_config_creation() {
        let config = DCNConfig::new(64, 3);
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.num_cross_layers, 3);
        assert_eq!(config.mode, DCNMode::Vector);

        let config_v2 = DCNConfig::new(128, 2).with_mode(DCNMode::Matrix);
        assert_eq!(config_v2.input_dim, 128);
        assert_eq!(config_v2.num_cross_layers, 2);
        assert_eq!(config_v2.mode, DCNMode::Matrix);
    }

    #[test]
    fn test_cross_layer_vector_mode() {
        let layer = CrossLayer::new(32, DCNMode::Vector);
        assert_eq!(layer.input_dim(), 32);
        assert_eq!(layer.mode(), DCNMode::Vector);
        assert_eq!(layer.weight().shape(), &[32, 1]);
        assert_eq!(layer.bias().shape(), &[32]);
    }

    #[test]
    fn test_cross_layer_matrix_mode() {
        let layer = CrossLayer::new(32, DCNMode::Matrix);
        assert_eq!(layer.input_dim(), 32);
        assert_eq!(layer.mode(), DCNMode::Matrix);
        assert_eq!(layer.weight().shape(), &[32, 32]);
        assert_eq!(layer.bias().shape(), &[32]);
    }

    #[test]
    fn test_cross_layer_forward_vector() {
        let layer = CrossLayer::new(16, DCNMode::Vector);
        let x0 = Tensor::rand(&[4, 16]);
        let xl = Tensor::rand(&[4, 16]);

        let output = layer.forward_with_x0(&x0, &xl).unwrap();
        assert_eq!(output.shape(), &[4, 16]);
    }

    #[test]
    fn test_cross_layer_forward_matrix() {
        let layer = CrossLayer::new(16, DCNMode::Matrix);
        let x0 = Tensor::rand(&[4, 16]);
        let xl = Tensor::rand(&[4, 16]);

        let output = layer.forward_with_x0(&x0, &xl).unwrap();
        assert_eq!(output.shape(), &[4, 16]);
    }

    #[test]
    fn test_cross_layer_layer_trait() {
        let layer = CrossLayer::new(16, DCNMode::Vector);
        let input = Tensor::rand(&[4, 16]);

        // When using Layer::forward, x0 = xl = input
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 16]);
        assert_eq!(layer.name(), "CrossLayer");
    }

    #[test]
    fn test_cross_layer_backward_vector() {
        let mut layer = CrossLayer::new(8, DCNMode::Vector);
        let x0 = Tensor::rand(&[2, 8]);
        let xl = Tensor::rand(&[2, 8]);

        // Forward with caching
        let _output = layer.forward_train_with_x0(&x0, &xl).unwrap();

        // Backward
        let grad = Tensor::ones(&[2, 8]);
        let input_grad = layer.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[2, 8]);
        assert!(layer.weight_grad().is_some());
        assert!(layer.bias_grad().is_some());
    }

    #[test]
    fn test_cross_layer_backward_matrix() {
        let mut layer = CrossLayer::new(8, DCNMode::Matrix);
        let x0 = Tensor::rand(&[2, 8]);
        let xl = Tensor::rand(&[2, 8]);

        // Forward with caching
        let _output = layer.forward_train_with_x0(&x0, &xl).unwrap();

        // Backward
        let grad = Tensor::ones(&[2, 8]);
        let input_grad = layer.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[2, 8]);
        assert!(layer.weight_grad().is_some());
        assert!(layer.bias_grad().is_some());
    }

    #[test]
    fn test_cross_network_creation() {
        let config = DCNConfig::new(32, 3);
        let network = CrossNetwork::from_config(&config);

        assert_eq!(network.num_layers(), 3);
        assert_eq!(network.input_dim(), 32);
        assert_eq!(network.mode(), DCNMode::Vector);
        assert_eq!(network.layers().len(), 3);
    }

    #[test]
    fn test_cross_network_new() {
        let network = CrossNetwork::new(64, 4, DCNMode::Matrix);

        assert_eq!(network.num_layers(), 4);
        assert_eq!(network.input_dim(), 64);
        assert_eq!(network.mode(), DCNMode::Matrix);
    }

    #[test]
    fn test_cross_network_forward() {
        let config = DCNConfig::new(16, 2);
        let network = CrossNetwork::from_config(&config);

        let input = Tensor::rand(&[8, 16]);
        let output = network.forward(&input).unwrap();

        assert_eq!(output.shape(), &[8, 16]);
    }

    #[test]
    fn test_cross_network_forward_matrix_mode() {
        let config = DCNConfig::new(16, 2).with_mode(DCNMode::Matrix);
        let network = CrossNetwork::from_config(&config);

        let input = Tensor::rand(&[8, 16]);
        let output = network.forward(&input).unwrap();

        assert_eq!(output.shape(), &[8, 16]);
    }

    #[test]
    fn test_cross_network_backward() {
        let config = DCNConfig::new(8, 2);
        let mut network = CrossNetwork::from_config(&config);

        let input = Tensor::rand(&[4, 8]);

        // Forward with caching
        let _output = network.forward_train(&input).unwrap();

        // Backward
        let grad = Tensor::ones(&[4, 8]);
        let input_grad = network.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[4, 8]);
    }

    #[test]
    fn test_cross_network_parameters() {
        let config = DCNConfig::new(16, 3);
        let network = CrossNetwork::from_config(&config);

        // Each layer has weight and bias, so 3 layers * 2 = 6 parameters
        let params = network.parameters();
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_cross_network_training_mode() {
        let config = DCNConfig::new(16, 2);
        let mut network = CrossNetwork::from_config(&config);

        assert!(network.is_training());

        network.set_training(false);
        assert!(!network.is_training());
        for layer in network.layers() {
            assert!(!layer.is_training());
        }

        network.set_training(true);
        assert!(network.is_training());
    }

    #[test]
    fn test_cross_layer_invalid_input_dim() {
        let layer = CrossLayer::new(32, DCNMode::Vector);
        let x0 = Tensor::rand(&[4, 16]); // wrong dimension
        let xl = Tensor::rand(&[4, 32]);

        let result = layer.forward_with_x0(&x0, &xl);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_layer_batch_mismatch() {
        let layer = CrossLayer::new(32, DCNMode::Vector);
        let x0 = Tensor::rand(&[4, 32]);
        let xl = Tensor::rand(&[8, 32]); // different batch size

        let result = layer.forward_with_x0(&x0, &xl);
        assert!(result.is_err());
    }

    #[test]
    fn test_cross_network_clear_cache() {
        let config = DCNConfig::new(8, 2);
        let mut network = CrossNetwork::from_config(&config);

        let input = Tensor::rand(&[2, 8]);
        let _output = network.forward_train(&input).unwrap();

        network.clear_cache();
        // After clearing, backward should fail
        let grad = Tensor::ones(&[2, 8]);
        let result = network.backward(&grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_dcn_mode_default() {
        let mode = DCNMode::default();
        assert_eq!(mode, DCNMode::Vector);
    }

    #[test]
    fn test_cross_network_layer_trait() {
        let network = CrossNetwork::new(16, 2, DCNMode::Vector);
        assert_eq!(network.name(), "CrossNetwork");

        let input = Tensor::rand(&[4, 16]);
        let output = network.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 16]);
    }

    #[test]
    fn test_single_cross_layer_network() {
        // Test with just 1 cross layer
        let config = DCNConfig::new(8, 1);
        let network = CrossNetwork::from_config(&config);

        let input = Tensor::rand(&[2, 8]);
        let output = network.forward(&input).unwrap();

        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_deep_cross_network() {
        // Test with many cross layers
        let config = DCNConfig::new(16, 5);
        let network = CrossNetwork::from_config(&config);

        let input = Tensor::rand(&[4, 16]);
        let output = network.forward(&input).unwrap();

        assert_eq!(output.shape(), &[4, 16]);
    }
}
