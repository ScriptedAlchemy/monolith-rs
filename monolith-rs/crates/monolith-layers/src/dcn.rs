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
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::regularizer::Regularizer;
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

    /// DCN-Mixed: mixture of low-rank experts with gating.
    Mixed,
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
    /// Initializer for cross layer weights
    pub initializer: Initializer,
    /// Kernel regularizer for cross layer weights
    pub regularizer: Regularizer,
    /// Whether to enable kernel norm (weight normalization)
    pub allow_kernel_norm: bool,
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
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
            allow_kernel_norm: false,
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

    /// Sets initializer for cross layer weights.
    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    /// Sets kernel regularizer for cross layer weights.
    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    /// Enables kernel norm (weight normalization).
    pub fn with_kernel_norm(mut self, allow: bool) -> Self {
        self.allow_kernel_norm = allow;
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
    /// Whether to apply kernel norm (weight normalization)
    allow_kernel_norm: bool,
    /// Trainable kernel norm scale
    kernel_norm: Option<Tensor>,
    /// Gradient of kernel norm scale
    kernel_norm_grad: Option<Tensor>,
    /// Kernel regularizer
    kernel_regularizer: Regularizer,

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
        Self::new_with_options(
            input_dim,
            mode,
            Initializer::GlorotUniform,
            Regularizer::None,
            false,
        )
    }

    /// Creates a new cross layer with custom options.
    pub fn new_with_options(
        input_dim: usize,
        mode: DCNMode,
        initializer: Initializer,
        kernel_regularizer: Regularizer,
        allow_kernel_norm: bool,
    ) -> Self {
        let weight = match mode {
            DCNMode::Vector => initializer.initialize(&[input_dim, 1]),
            DCNMode::Matrix => initializer.initialize(&[input_dim, input_dim]),
            DCNMode::Mixed => Tensor::zeros(&[input_dim, 1]),
        };

        let bias = Tensor::zeros(&[input_dim]);
        let kernel_norm = if allow_kernel_norm {
            let out_dim = weight.shape()[1];
            let norm = weight
                .sqr()
                .sum_axis(0)
                .add(&Tensor::from_data(&[out_dim], vec![1e-6; out_dim]))
                .sqrt();
            Some(norm)
        } else {
            None
        };

        Self {
            weight,
            bias,
            allow_kernel_norm,
            kernel_norm,
            kernel_norm_grad: None,
            kernel_regularizer,
            input_dim,
            mode,
            weight_grad: None,
            bias_grad: None,
            cached_x0: None,
            cached_xl: None,
            training: true,
        }
    }

    fn normalized_weight(&self) -> (Tensor, Tensor) {
        let out_dim = self.weight.shape()[1];
        let eps = Tensor::from_data(&[out_dim], vec![1e-6; out_dim]);
        let norm = self.weight.sqr().sum_axis(0).add(&eps).sqrt();
        let norm_b = norm
            .reshape(&[1, out_dim])
            .broadcast_as(self.weight.shape());
        let w_norm = self.weight.div(&norm_b);
        (norm, w_norm)
    }

    fn effective_weight(&self) -> Tensor {
        if !self.allow_kernel_norm {
            return self.weight.clone();
        }
        let (_norm, mut w_norm) = self.normalized_weight();
        if let Some(scale) = &self.kernel_norm {
            let out_dim = self.weight.shape()[1];
            let scale_b = scale
                .reshape(&[1, out_dim])
                .broadcast_as(self.weight.shape());
            w_norm = w_norm.mul(&scale_b);
        }
        w_norm
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

    /// Returns a mutable reference to the weight tensor.
    pub fn weight_mut(&mut self) -> &mut Tensor {
        &mut self.weight
    }

    /// Returns a reference to the bias tensor.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Returns a mutable reference to the bias tensor.
    pub fn bias_mut(&mut self) -> &mut Tensor {
        &mut self.bias
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
        self.kernel_norm_grad = None;
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

        if self.mode == DCNMode::Mixed {
            return Err(LayerError::ForwardError {
                message: "CrossLayer does not support Mixed mode; use DCNLayer".to_string(),
            });
        }

        // Compute based on mode
        let output = match self.mode {
            DCNMode::Vector => {
                // x_{l+1} = x_0 * (w^T * x_l + b) + x_l
                // w is [d, 1], so w^T is [1, d]
                // x_l @ w gives [batch, 1]
                // Then we need to broadcast multiply with x_0

                // Step 1: xl @ w -> [batch, 1]
                let weights = self.effective_weight();
                let xl_w = xl.matmul(&weights);

                // Step 2: Broadcast and add bias -> [batch, d]
                let interaction = xl_w
                    .broadcast_as(&[batch_size, self.input_dim])
                    .add(&self.bias);

                // Step 3: x_0 * interaction (element-wise)
                let cross = x0.mul(&interaction);

                // Step 4: Add residual xl
                cross.add(xl)
            }
            DCNMode::Matrix => {
                // x_{l+1} = x_0 * (x_l @ W + b) + x_l
                // W is [d, d]

                // Step 1: xl @ W -> [batch, d]
                let weights = self.effective_weight();
                let xl_w = xl.matmul(&weights);

                // Step 2: Add bias -> [batch, d]
                let interaction = xl_w.add(&self.bias);

                // Step 3: x_0 * interaction (element-wise)
                let cross = x0.mul(&interaction);

                // Step 4: Add residual xl
                cross.add(xl)
            }
            DCNMode::Mixed => unreachable!("Mixed mode handled before match"),
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
        let x0 = self.cached_x0.as_ref().ok_or(LayerError::NotInitialized)?;
        let xl = self.cached_xl.as_ref().ok_or(LayerError::NotInitialized)?;

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
                let weights_eff = self.effective_weight();
                let xl_w = xl.matmul(&weights_eff);
                let interaction = xl_w
                    .broadcast_as(&[batch_size, self.input_dim])
                    .add(&self.bias);

                // d_interaction = grad * x0
                let d_interaction = grad.mul(x0);

                // d_bias = sum(d_interaction, axis=0)
                self.bias_grad = Some(d_interaction.sum_axis(0));

                // d_interaction_scalar = sum(d_interaction, axis=1) -> [batch]
                let d_interaction_scalar_vec = d_interaction.sum_axis(1);
                let d_interaction_scalar = d_interaction_scalar_vec.reshape(&[batch_size, 1]);

                // d_weight for effective weight
                let weights_grad_eff = xl.transpose().matmul(&d_interaction_scalar);

                if !self.allow_kernel_norm {
                    let mut weight_grad = weights_grad_eff.clone();
                    if let Some(reg_grad) = self.kernel_regularizer.grad(&self.weight) {
                        weight_grad = weight_grad.add(&reg_grad);
                    }
                    self.weight_grad = Some(weight_grad);
                } else {
                    let (norm, w_norm) = self.normalized_weight();
                    let out_dim = self.weight.shape()[1];
                    let norm_b = norm
                        .reshape(&[1, out_dim])
                        .broadcast_as(self.weight.shape());

                    let weights_grad_norm = if let Some(scale) = &self.kernel_norm {
                        let scale_b = scale
                            .reshape(&[1, out_dim])
                            .broadcast_as(self.weight.shape());
                        weights_grad_eff.mul(&scale_b)
                    } else {
                        weights_grad_eff.clone()
                    };

                    let dot = weights_grad_norm.mul(&w_norm).sum_axis(0);
                    let dot_b = dot.reshape(&[1, out_dim]).broadcast_as(self.weight.shape());
                    let mut weight_grad = weights_grad_norm.sub(&w_norm.mul(&dot_b)).div(&norm_b);
                    if let Some(reg_grad) = self.kernel_regularizer.grad(&self.weight) {
                        weight_grad = weight_grad.add(&reg_grad);
                    }
                    self.weight_grad = Some(weight_grad);
                    if self.kernel_norm.is_some() {
                        let kernel_norm_grad = weights_grad_eff.mul(&w_norm).sum_axis(0);
                        self.kernel_norm_grad = Some(kernel_norm_grad);
                    }
                }

                // d_xl from weight path = d_interaction_scalar @ w_eff^T
                let d_xl_w = d_interaction_scalar.matmul(&weights_eff.transpose());

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
                let weights_eff = self.effective_weight();
                let xl_w = xl.matmul(&weights_eff);
                let interaction = xl_w.add(&self.bias);

                // d_interaction = grad * x0
                let d_interaction = grad.mul(x0);

                // d_bias = sum(d_interaction, axis=0)
                self.bias_grad = Some(d_interaction.sum_axis(0));

                // d_weight for effective weight
                let weights_grad_eff = xl.transpose().matmul(&d_interaction);

                if !self.allow_kernel_norm {
                    let mut weight_grad = weights_grad_eff.clone();
                    if let Some(reg_grad) = self.kernel_regularizer.grad(&self.weight) {
                        weight_grad = weight_grad.add(&reg_grad);
                    }
                    self.weight_grad = Some(weight_grad);
                } else {
                    let (norm, w_norm) = self.normalized_weight();
                    let out_dim = self.weight.shape()[1];
                    let norm_b = norm
                        .reshape(&[1, out_dim])
                        .broadcast_as(self.weight.shape());

                    let weights_grad_norm = if let Some(scale) = &self.kernel_norm {
                        let scale_b = scale
                            .reshape(&[1, out_dim])
                            .broadcast_as(self.weight.shape());
                        weights_grad_eff.mul(&scale_b)
                    } else {
                        weights_grad_eff.clone()
                    };

                    let dot = weights_grad_norm.mul(&w_norm).sum_axis(0);
                    let dot_b = dot.reshape(&[1, out_dim]).broadcast_as(self.weight.shape());
                    let mut weight_grad = weights_grad_norm.sub(&w_norm.mul(&dot_b)).div(&norm_b);
                    if let Some(reg_grad) = self.kernel_regularizer.grad(&self.weight) {
                        weight_grad = weight_grad.add(&reg_grad);
                    }
                    self.weight_grad = Some(weight_grad);
                    if self.kernel_norm.is_some() {
                        let kernel_norm_grad = weights_grad_eff.mul(&w_norm).sum_axis(0);
                        self.kernel_norm_grad = Some(kernel_norm_grad);
                    }
                }

                // d_xl from weight path
                let d_xl_w = d_interaction.matmul(&weights_eff.transpose());

                // d_x0 = grad * interaction
                let _d_x0 = grad.mul(&interaction);

                // Total d_xl = grad (residual) + d_xl_from_w
                let d_xl = grad.add(&d_xl_w);

                Ok(d_xl)
            }
            DCNMode::Mixed => Err(LayerError::BackwardError {
                message: "CrossLayer does not support Mixed mode; use DCNLayer".to_string(),
            }),
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight, &self.bias];
        if let Some(kernel_norm) = &self.kernel_norm {
            params.push(kernel_norm);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weight, &mut self.bias];
        if let Some(kernel_norm) = &mut self.kernel_norm {
            params.push(kernel_norm);
        }
        params
    }

    fn name(&self) -> &str {
        "CrossLayer"
    }

    fn regularization_loss(&self) -> f32 {
        self.kernel_regularizer.loss(&self.weight)
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
            .map(|_| {
                CrossLayer::new_with_options(
                    config.input_dim,
                    config.mode,
                    config.initializer,
                    config.regularizer.clone(),
                    config.allow_kernel_norm,
                )
            })
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
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
            allow_kernel_norm: false,
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
        let _x0 = self.cached_x0.as_ref().ok_or(LayerError::NotInitialized)?;

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

/// Mixed cross layer (DCN-Mixed) with low-rank experts and gating.
#[derive(Debug, Clone)]
pub struct MixedCrossLayer {
    input_dim: usize,
    num_experts: usize,
    low_rank: usize,
    bias: Tensor,
    allow_kernel_norm: bool,
    kernel_regularizer: Regularizer,
    u: Vec<Tensor>,
    v: Vec<Tensor>,
    c: Vec<Tensor>,
    g: Vec<Tensor>,
    u_norm: Vec<Option<Tensor>>,
    v_norm: Vec<Option<Tensor>>,
    c_norm: Vec<Option<Tensor>>,
    g_norm: Vec<Option<Tensor>>,
    u_norm_grad: Vec<Option<Tensor>>,
    v_norm_grad: Vec<Option<Tensor>>,
    c_norm_grad: Vec<Option<Tensor>>,
    g_norm_grad: Vec<Option<Tensor>>,
    u_grad: Vec<Option<Tensor>>,
    v_grad: Vec<Option<Tensor>>,
    c_grad: Vec<Option<Tensor>>,
    g_grad: Vec<Option<Tensor>>,
    bias_grad: Option<Tensor>,
    cached_x0: Option<Tensor>,
    cached_xl: Option<Tensor>,
    cached_v: Vec<Tensor>,
    cached_cv: Vec<Tensor>,
    cached_ucv: Vec<Tensor>,
    cached_g: Vec<Tensor>,
    cached_out_stack: Option<Tensor>,
    cached_weights: Option<Tensor>,
}

impl MixedCrossLayer {
    pub fn new(
        input_dim: usize,
        num_experts: usize,
        low_rank: usize,
        initializer: Initializer,
        kernel_regularizer: Regularizer,
        allow_kernel_norm: bool,
    ) -> Self {
        let mut u = Vec::new();
        let mut v = Vec::new();
        let mut c = Vec::new();
        let mut g = Vec::new();
        let mut u_norm = Vec::new();
        let mut v_norm = Vec::new();
        let mut c_norm = Vec::new();
        let mut g_norm = Vec::new();
        for _ in 0..num_experts {
            let u_w = initializer.initialize(&[input_dim, low_rank]);
            let v_w = initializer.initialize(&[input_dim, low_rank]);
            let c_w = initializer.initialize(&[low_rank, low_rank]);
            let g_w = initializer.initialize(&[input_dim, 1]);

            let u_scale = if allow_kernel_norm {
                let norm = u_w
                    .sqr()
                    .sum_axis(0)
                    .add(&Tensor::from_data(&[low_rank], vec![1e-6; low_rank]))
                    .sqrt();
                Some(norm)
            } else {
                None
            };
            let v_scale = if allow_kernel_norm {
                let norm = v_w
                    .sqr()
                    .sum_axis(0)
                    .add(&Tensor::from_data(&[low_rank], vec![1e-6; low_rank]))
                    .sqrt();
                Some(norm)
            } else {
                None
            };
            let c_scale = if allow_kernel_norm {
                let norm = c_w
                    .sqr()
                    .sum_axis(0)
                    .add(&Tensor::from_data(&[low_rank], vec![1e-6; low_rank]))
                    .sqrt();
                Some(norm)
            } else {
                None
            };
            let g_scale = if allow_kernel_norm {
                let norm = g_w
                    .sqr()
                    .sum_axis(0)
                    .add(&Tensor::from_data(&[1], vec![1e-6]))
                    .sqrt();
                Some(norm)
            } else {
                None
            };

            u.push(u_w);
            v.push(v_w);
            c.push(c_w);
            g.push(g_w);
            u_norm.push(u_scale);
            v_norm.push(v_scale);
            c_norm.push(c_scale);
            g_norm.push(g_scale);
        }
        Self {
            input_dim,
            num_experts,
            low_rank,
            bias: Tensor::zeros(&[input_dim]),
            allow_kernel_norm,
            kernel_regularizer,
            u,
            v,
            c,
            g,
            u_norm,
            v_norm,
            c_norm,
            g_norm,
            u_norm_grad: vec![None; num_experts],
            v_norm_grad: vec![None; num_experts],
            c_norm_grad: vec![None; num_experts],
            g_norm_grad: vec![None; num_experts],
            u_grad: vec![None; num_experts],
            v_grad: vec![None; num_experts],
            c_grad: vec![None; num_experts],
            g_grad: vec![None; num_experts],
            bias_grad: None,
            cached_x0: None,
            cached_xl: None,
            cached_v: Vec::new(),
            cached_cv: Vec::new(),
            cached_ucv: Vec::new(),
            cached_g: Vec::new(),
            cached_out_stack: None,
            cached_weights: None,
        }
    }

    pub fn clear_cache(&mut self) {
        self.cached_x0 = None;
        self.cached_xl = None;
        self.cached_v.clear();
        self.cached_cv.clear();
        self.cached_ucv.clear();
        self.cached_g.clear();
        self.cached_out_stack = None;
        self.cached_weights = None;
        self.bias_grad = None;
        for g in &mut self.u_grad {
            *g = None;
        }
        for g in &mut self.v_grad {
            *g = None;
        }
        for g in &mut self.c_grad {
            *g = None;
        }
        for g in &mut self.g_grad {
            *g = None;
        }
        for g in &mut self.u_norm_grad {
            *g = None;
        }
        for g in &mut self.v_norm_grad {
            *g = None;
        }
        for g in &mut self.c_norm_grad {
            *g = None;
        }
        for g in &mut self.g_norm_grad {
            *g = None;
        }
    }

    fn effective_weight(weight: &Tensor, scale: Option<&Tensor>) -> Tensor {
        if let Some(scale) = scale {
            let out_dim = weight.shape()[1];
            let norm = weight
                .sqr()
                .sum_axis(0)
                .add(&Tensor::from_data(&[out_dim], vec![1e-6; out_dim]))
                .sqrt();
            let norm_b = norm.reshape(&[1, out_dim]).broadcast_as(weight.shape());
            let w_norm = weight.div(&norm_b);
            let scale_b = scale.reshape(&[1, out_dim]).broadcast_as(weight.shape());
            w_norm.mul(&scale_b)
        } else {
            weight.clone()
        }
    }

    fn weight_grad_with_norm(
        weight: &Tensor,
        grad_eff: &Tensor,
        scale: Option<&Tensor>,
    ) -> (Tensor, Option<Tensor>) {
        if let Some(scale) = scale {
            let out_dim = weight.shape()[1];
            let norm = weight
                .sqr()
                .sum_axis(0)
                .add(&Tensor::from_data(&[out_dim], vec![1e-6; out_dim]))
                .sqrt();
            let norm_b = norm.reshape(&[1, out_dim]).broadcast_as(weight.shape());
            let w_norm = weight.div(&norm_b);
            let scale_b = scale.reshape(&[1, out_dim]).broadcast_as(weight.shape());
            let grad_norm = grad_eff.mul(&scale_b);
            let dot = grad_norm.mul(&w_norm).sum_axis(0);
            let dot_b = dot.reshape(&[1, out_dim]).broadcast_as(weight.shape());
            let weight_grad = grad_norm.sub(&w_norm.mul(&dot_b)).div(&norm_b);
            let scale_grad = grad_eff.mul(&w_norm).sum_axis(0);
            (weight_grad, Some(scale_grad))
        } else {
            (grad_eff.clone(), None)
        }
    }

    fn regularization_loss(&self) -> f32 {
        let mut loss = 0.0;
        for u in &self.u {
            loss += self.kernel_regularizer.loss(u);
        }
        for v in &self.v {
            loss += self.kernel_regularizer.loss(v);
        }
        for c in &self.c {
            loss += self.kernel_regularizer.loss(c);
        }
        for g in &self.g {
            loss += self.kernel_regularizer.loss(g);
        }
        loss
    }

    fn forward_internal(
        &mut self,
        x0: &Tensor,
        xl: &Tensor,
        cache: bool,
    ) -> Result<Tensor, LayerError> {
        let batch = x0.shape()[0];
        let bias_b = self
            .bias
            .reshape(&[1, self.input_dim])
            .broadcast_as(&[batch, self.input_dim]);

        let mut g_list = Vec::with_capacity(self.num_experts);
        let mut v_list = Vec::with_capacity(self.num_experts);
        let mut cv_list = Vec::with_capacity(self.num_experts);
        let mut ucv_list = Vec::with_capacity(self.num_experts);
        let mut out_list = Vec::with_capacity(self.num_experts);

        for i in 0..self.num_experts {
            let g_w = Self::effective_weight(&self.g[i], self.g_norm[i].as_ref());
            let v_w = Self::effective_weight(&self.v[i], self.v_norm[i].as_ref());
            let c_w = Self::effective_weight(&self.c[i], self.c_norm[i].as_ref());
            let u_w = Self::effective_weight(&self.u[i], self.u_norm[i].as_ref());

            let g = xl.matmul(&g_w);
            let v = xl.matmul(&v_w).tanh();
            let cv = v.matmul(&c_w).tanh();
            let ucv = cv.matmul(&u_w.transpose());
            let out = x0.mul(&ucv.add(&bias_b));

            g_list.push(g);
            v_list.push(v);
            cv_list.push(cv);
            ucv_list.push(ucv);
            out_list.push(out);
        }

        let g_stack = Tensor::stack(&g_list, 1); // [B, E, 1]
        let weights = g_stack.softmax(1);
        let out_stack = Tensor::stack(&out_list, 2); // [B, D, E]
        let mixed = out_stack.matmul(&weights).reshape(&[batch, self.input_dim]);
        let output = mixed.add(xl);

        if cache {
            self.cached_x0 = Some(x0.clone());
            self.cached_xl = Some(xl.clone());
            self.cached_g = g_list;
            self.cached_v = v_list;
            self.cached_cv = cv_list;
            self.cached_ucv = ucv_list;
            self.cached_out_stack = Some(out_stack);
            self.cached_weights = Some(weights);
        }

        Ok(output)
    }

    pub fn forward_with_x0(&mut self, x0: &Tensor, xl: &Tensor) -> Result<Tensor, LayerError> {
        self.forward_internal(x0, xl, false)
    }

    pub fn forward_train_with_x0(
        &mut self,
        x0: &Tensor,
        xl: &Tensor,
    ) -> Result<Tensor, LayerError> {
        self.forward_internal(x0, xl, true)
    }

    pub fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let x0 = self.cached_x0.as_ref().ok_or(LayerError::NotInitialized)?;
        let xl = self.cached_xl.as_ref().ok_or(LayerError::NotInitialized)?;
        let out_stack = self
            .cached_out_stack
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let weights = self
            .cached_weights
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let batch = grad.shape()[0];
        let grad_y = grad.reshape(&[batch, self.input_dim, 1]);

        let weights_t = weights.transpose_dims(1, 2); // [B, 1, E]
        let grad_out_stack = grad_y.matmul(&weights_t); // [B, D, E]

        let out_stack_t = out_stack.transpose_dims(1, 2); // [B, E, D]
        let grad_weights = out_stack_t.matmul(&grad_y); // [B, E, 1]
        let dot = grad_weights.mul(weights).sum_axis(1); // [B, 1]
        let dot_b = dot.reshape(&[batch, 1, 1]).broadcast_as(weights.shape());
        let grad_g_stack = weights.mul(&grad_weights.sub(&dot_b)); // [B, E, 1]

        let grad_out_list = grad_out_stack.unstack(2);
        let grad_g_list = grad_g_stack.unstack(1);

        let mut grad_xl = grad.clone();
        let mut bias_grad = Tensor::zeros(&[self.input_dim]);

        for i in 0..self.num_experts {
            let grad_out = &grad_out_list[i];
            let grad_g = &grad_g_list[i];
            let v = &self.cached_v[i];
            let cv = &self.cached_cv[i];
            let u_eff = Self::effective_weight(&self.u[i], self.u_norm[i].as_ref());
            let v_eff = Self::effective_weight(&self.v[i], self.v_norm[i].as_ref());
            let c_eff = Self::effective_weight(&self.c[i], self.c_norm[i].as_ref());
            let g_eff = Self::effective_weight(&self.g[i], self.g_norm[i].as_ref());

            // out = x0 * (ucv + bias)
            let grad_ucv = grad_out.mul(x0);
            bias_grad = bias_grad.add(&grad_out.mul(x0).sum_axis(0));

            // ucv = cv @ U^T
            let grad_u_eff = grad_ucv.transpose().matmul(cv);
            let grad_cv = grad_ucv.matmul(&u_eff);

            // cv = tanh(v @ C)
            let grad_t = grad_cv.mul(&Tensor::ones(cv.shape()).sub(&cv.sqr()));
            let grad_c_eff = v.transpose().matmul(&grad_t);
            let grad_v = grad_t.matmul(&c_eff.transpose());

            // v = tanh(xl @ V)
            let grad_t2 = grad_v.mul(&Tensor::ones(v.shape()).sub(&v.sqr()));
            let grad_v_eff = xl.transpose().matmul(&grad_t2);
            let grad_xl_v = grad_t2.matmul(&v_eff.transpose());

            // g = xl @ G
            let grad_g_eff = xl.transpose().matmul(grad_g);
            let grad_xl_g = grad_g.matmul(&g_eff.transpose());

            grad_xl = grad_xl.add(&grad_xl_v).add(&grad_xl_g);

            let (mut grad_u, u_norm_grad) =
                Self::weight_grad_with_norm(&self.u[i], &grad_u_eff, self.u_norm[i].as_ref());
            if let Some(reg_grad) = self.kernel_regularizer.grad(&self.u[i]) {
                grad_u = grad_u.add(&reg_grad);
            }
            self.u_grad[i] = Some(grad_u);
            self.u_norm_grad[i] = u_norm_grad;

            let (mut grad_v_w, v_norm_grad) =
                Self::weight_grad_with_norm(&self.v[i], &grad_v_eff, self.v_norm[i].as_ref());
            if let Some(reg_grad) = self.kernel_regularizer.grad(&self.v[i]) {
                grad_v_w = grad_v_w.add(&reg_grad);
            }
            self.v_grad[i] = Some(grad_v_w);
            self.v_norm_grad[i] = v_norm_grad;

            let (mut grad_c_w, c_norm_grad) =
                Self::weight_grad_with_norm(&self.c[i], &grad_c_eff, self.c_norm[i].as_ref());
            if let Some(reg_grad) = self.kernel_regularizer.grad(&self.c[i]) {
                grad_c_w = grad_c_w.add(&reg_grad);
            }
            self.c_grad[i] = Some(grad_c_w);
            self.c_norm_grad[i] = c_norm_grad;

            let (mut grad_g_w, g_norm_grad) =
                Self::weight_grad_with_norm(&self.g[i], &grad_g_eff, self.g_norm[i].as_ref());
            if let Some(reg_grad) = self.kernel_regularizer.grad(&self.g[i]) {
                grad_g_w = grad_g_w.add(&reg_grad);
            }
            self.g_grad[i] = Some(grad_g_w);
            self.g_norm_grad[i] = g_norm_grad;
        }

        self.bias_grad = Some(bias_grad);

        Ok(grad_xl)
    }

    pub fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.push(&self.bias);
        for i in 0..self.num_experts {
            params.push(&self.u[i]);
            params.push(&self.v[i]);
            params.push(&self.c[i]);
            params.push(&self.g[i]);
            if let Some(norm) = &self.u_norm[i] {
                params.push(norm);
            }
            if let Some(norm) = &self.v_norm[i] {
                params.push(norm);
            }
            if let Some(norm) = &self.c_norm[i] {
                params.push(norm);
            }
            if let Some(norm) = &self.g_norm[i] {
                params.push(norm);
            }
        }
        params
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.push(&mut self.bias);
        for u in &mut self.u {
            params.push(u);
        }
        for v in &mut self.v {
            params.push(v);
        }
        for c in &mut self.c {
            params.push(c);
        }
        for g in &mut self.g {
            params.push(g);
        }
        for norm in &mut self.u_norm {
            if let Some(norm) = norm {
                params.push(norm);
            }
        }
        for norm in &mut self.v_norm {
            if let Some(norm) = norm {
                params.push(norm);
            }
        }
        for norm in &mut self.c_norm {
            if let Some(norm) = norm {
                params.push(norm);
            }
        }
        for norm in &mut self.g_norm {
            if let Some(norm) = norm {
                params.push(norm);
            }
        }
        params
    }
}

/// DCN layer matching Python DCN (vector/matrix/mixed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DCNLayerConfig {
    pub input_dim: usize,
    pub layer_num: usize,
    pub dcn_type: DCNMode,
    pub num_experts: usize,
    pub low_rank: usize,
    pub initializer: Initializer,
    pub regularizer: Regularizer,
    pub allow_kernel_norm: bool,
    pub use_dropout: bool,
    pub keep_prob: f32,
    pub training: bool,
}

impl DCNLayerConfig {
    pub fn new(input_dim: usize, layer_num: usize) -> Self {
        Self {
            input_dim,
            layer_num,
            dcn_type: DCNMode::Matrix,
            num_experts: 1,
            low_rank: 0,
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
            allow_kernel_norm: false,
            use_dropout: false,
            keep_prob: 0.95,
            training: true,
        }
    }

    pub fn with_mode(mut self, mode: DCNMode) -> Self {
        self.dcn_type = mode;
        self
    }

    pub fn with_mixed(mut self, num_experts: usize, low_rank: usize) -> Self {
        self.dcn_type = DCNMode::Mixed;
        self.num_experts = num_experts;
        self.low_rank = low_rank;
        self
    }

    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    pub fn with_kernel_norm(mut self, allow: bool) -> Self {
        self.allow_kernel_norm = allow;
        self
    }

    pub fn with_dropout(mut self, keep_prob: f32) -> Self {
        self.use_dropout = true;
        self.keep_prob = keep_prob;
        self
    }
}

#[derive(Debug, Clone)]
enum DCNLayerKind {
    Basic(CrossLayer),
    Mixed(MixedCrossLayer),
}

/// DCN layer that supports vector/matrix/mixed.
#[derive(Debug, Clone)]
pub struct DCNLayer {
    layers: Vec<DCNLayerKind>,
    config: DCNLayerConfig,
    cached_x0: Option<Tensor>,
}

impl DCNLayer {
    pub fn from_config(config: DCNLayerConfig) -> Self {
        let mut layers = Vec::new();
        for _ in 0..config.layer_num {
            let layer = match config.dcn_type {
                DCNMode::Vector | DCNMode::Matrix => {
                    DCNLayerKind::Basic(CrossLayer::new_with_options(
                        config.input_dim,
                        config.dcn_type,
                        config.initializer,
                        config.regularizer.clone(),
                        config.allow_kernel_norm,
                    ))
                }
                DCNMode::Mixed => DCNLayerKind::Mixed(MixedCrossLayer::new(
                    config.input_dim,
                    config.num_experts,
                    config.low_rank,
                    config.initializer,
                    config.regularizer.clone(),
                    config.allow_kernel_norm,
                )),
            };
            layers.push(layer);
        }
        Self {
            layers,
            config,
            cached_x0: None,
        }
    }

    fn apply_dropout(&self, input: &Tensor) -> Tensor {
        if !self.config.use_dropout || !self.config.training {
            return input.clone();
        }
        let mask = Tensor::rand(input.shape()).ge_scalar(1.0 - self.config.keep_prob);
        input.mul(&mask).scale(1.0 / self.config.keep_prob)
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_x0 = Some(input.clone());
        let x0 = input;
        let mut xl = input.clone();
        let use_dropout = self.config.use_dropout && self.config.training;
        let keep_prob = self.config.keep_prob;
        for layer in &mut self.layers {
            xl = match layer {
                DCNLayerKind::Basic(l) => l.forward_train_with_x0(x0, &xl)?,
                DCNLayerKind::Mixed(l) => l.forward_train_with_x0(x0, &xl)?,
            };
            if use_dropout {
                let mask = Tensor::rand(xl.shape()).ge_scalar(1.0 - keep_prob);
                xl = xl.mul(&mask).scale(1.0 / keep_prob);
            }
        }
        Ok(xl)
    }
}

impl Layer for DCNLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let x0 = input;
        let mut xl = input.clone();
        for layer in &self.layers {
            xl = match layer {
                DCNLayerKind::Basic(l) => l.forward_with_x0(x0, &xl)?,
                DCNLayerKind::Mixed(l) => {
                    let mut layer = l.clone();
                    layer.forward_with_x0(x0, &xl)?
                }
            };
        }
        Ok(xl)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let _x0 = self.cached_x0.as_ref().ok_or(LayerError::NotInitialized)?;
        let mut current_grad = grad.clone();
        for layer in self.layers.iter_mut().rev() {
            current_grad = match layer {
                DCNLayerKind::Basic(l) => l.backward(&current_grad)?,
                DCNLayerKind::Mixed(l) => l.backward(&current_grad)?,
            };
        }
        Ok(current_grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            match layer {
                DCNLayerKind::Basic(l) => params.extend(l.parameters()),
                DCNLayerKind::Mixed(l) => params.extend(l.parameters()),
            }
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            match layer {
                DCNLayerKind::Basic(l) => params.extend(l.parameters_mut()),
                DCNLayerKind::Mixed(l) => params.extend(l.parameters_mut()),
            }
        }
        params
    }

    fn name(&self) -> &str {
        "DCNLayer"
    }

    fn regularization_loss(&self) -> f32 {
        self.layers
            .iter()
            .map(|layer| match layer {
                DCNLayerKind::Basic(l) => l.regularization_loss(),
                DCNLayerKind::Mixed(l) => l.regularization_loss(),
            })
            .sum()
    }

    fn is_training(&self) -> bool {
        self.config.training
    }

    fn set_training(&mut self, training: bool) {
        self.config.training = training;
    }
}
