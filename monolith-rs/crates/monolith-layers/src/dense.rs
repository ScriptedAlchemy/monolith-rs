//! Dense (fully connected) layer implementation.
//!
//! This module provides the [`Dense`] layer, which performs a linear transformation
//! `y = xW + b` where W is the weight matrix and b is the bias vector.

use crate::constraint::Constraint;
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// A dense (fully connected) neural network layer.
///
/// Performs the transformation `y = xW + b` where:
/// - `x` is the input tensor of shape `[batch_size, in_features]`
/// - `W` is the weight matrix of shape `[in_features, out_features]`
/// - `b` is the bias vector of shape `[out_features]`
/// - `y` is the output tensor of shape `[batch_size, out_features]`
///
/// # Example
///
/// ```
/// use monolith_layers::dense::Dense;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let layer = Dense::new(128, 64);
/// let input = Tensor::zeros(&[32, 128]); // batch of 32
/// let output = layer.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[32, 64]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dense {
    /// Weight matrix of shape [in_features, out_features]
    weights: Tensor,
    /// Bias vector of shape [out_features]
    bias: Tensor,
    /// Kernel regularizer
    kernel_regularizer: Regularizer,
    /// Bias regularizer
    bias_regularizer: Regularizer,
    /// Kernel constraint
    kernel_constraint: Constraint,
    /// Bias constraint
    bias_constraint: Constraint,
    /// Whether to apply kernel norm (weight normalization)
    allow_kernel_norm: bool,
    /// Whether the kernel norm scale is trainable
    kernel_norm_trainable: bool,
    /// Trainable kernel norm scale (shape [out_features])
    kernel_norm: Option<Tensor>,
    /// Gradient of kernel norm scale
    kernel_norm_grad: Option<Tensor>,
    /// Gradient of weights
    weights_grad: Option<Tensor>,
    /// Gradient of bias
    bias_grad: Option<Tensor>,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Input feature dimension
    in_features: usize,
    /// Output feature dimension
    out_features: usize,
    /// Whether to use bias
    use_bias: bool,
}

impl Dense {
    /// Creates a new dense layer with the specified input and output dimensions.
    ///
    /// Weights are initialized using Glorot uniform initialization and
    /// biases are initialized to zeros.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::dense::Dense;
    ///
    /// let layer = Dense::new(64, 32);
    /// ```
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::new_with_initializer(
            in_features,
            out_features,
            // Python Dense defaults to VarianceScaling(mode="fan_avg", distribution="uniform").
            Initializer::VarianceScaling {
                scale: 1.0,
                mode: crate::initializer::VarianceScalingMode::FanAvg,
                distribution: crate::initializer::VarianceScalingDistribution::Uniform,
                seed: None,
            },
            Initializer::Zeros,
            true,
        )
    }

    /// Creates a new dense layer without bias.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        Self::new_with_initializer(
            in_features,
            out_features,
            Initializer::VarianceScaling {
                scale: 1.0,
                mode: crate::initializer::VarianceScalingMode::FanAvg,
                distribution: crate::initializer::VarianceScalingDistribution::Uniform,
                seed: None,
            },
            Initializer::Zeros,
            false,
        )
    }

    /// Creates a new dense layer with custom initializers.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    /// * `weight_init` - Initializer for the weight matrix
    /// * `bias_init` - Initializer for the bias vector
    /// * `use_bias` - Whether to use bias
    pub fn new_with_initializer(
        in_features: usize,
        out_features: usize,
        weight_init: Initializer,
        bias_init: Initializer,
        use_bias: bool,
    ) -> Self {
        Self::new_with_options(
            in_features,
            out_features,
            weight_init,
            bias_init,
            use_bias,
            Regularizer::None,
            Regularizer::None,
            Constraint::None,
            Constraint::None,
        )
    }

    /// Creates a new dense layer with full options.
    pub fn new_with_options(
        in_features: usize,
        out_features: usize,
        weight_init: Initializer,
        bias_init: Initializer,
        use_bias: bool,
        kernel_regularizer: Regularizer,
        bias_regularizer: Regularizer,
        kernel_constraint: Constraint,
        bias_constraint: Constraint,
    ) -> Self {
        let weights = weight_init.initialize(&[in_features, out_features]);
        let bias = if use_bias {
            bias_init.initialize(&[out_features])
        } else {
            Tensor::zeros(&[out_features])
        };
        // Python `Dense` initializes a trainable per-output-column scale to `||W||_2`
        // and multiplies it with an L2-normalized kernel. At init this reconstructs `W`.
        let kernel_norm = weights.sqr().sum_axis(0).sqrt();

        Self {
            weights,
            bias,
            kernel_regularizer,
            bias_regularizer,
            kernel_constraint,
            bias_constraint,
            // Match Python `monolith/core/dense.py` defaults.
            allow_kernel_norm: true,
            kernel_norm_trainable: true,
            kernel_norm: Some(kernel_norm),
            kernel_norm_grad: None,
            weights_grad: None,
            bias_grad: None,
            cached_input: None,
            in_features,
            out_features,
            use_bias,
        }
    }

    /// Enables kernel norm (weight normalization).
    ///
    /// If `trainable` is true, a trainable scale vector is created (shape [out_features])
    /// initialized to the L2 norm of the current weights along axis 0.
    pub fn with_kernel_norm(mut self, trainable: bool) -> Self {
        self.enable_kernel_norm(trainable);
        self
    }

    /// Enables kernel norm on an existing layer.
    pub fn enable_kernel_norm(&mut self, trainable: bool) {
        self.allow_kernel_norm = true;
        self.kernel_norm_trainable = trainable;
        if trainable {
            self.kernel_norm = Some(self.weights.sqr().sum_axis(0).sqrt());
        } else {
            self.kernel_norm = None;
        }
    }

    /// Enables or disables kernel norm (weight normalization).
    ///
    /// Disabling restores plain `y = xW + b` behavior (no normalization).
    pub fn with_allow_kernel_norm(mut self, allow: bool) -> Self {
        self.set_allow_kernel_norm(allow);
        self
    }

    /// Enables or disables kernel norm (weight normalization) in-place.
    pub fn set_allow_kernel_norm(&mut self, allow: bool) {
        self.allow_kernel_norm = allow;
        if !allow {
            self.kernel_norm_trainable = false;
            self.kernel_norm = None;
            self.kernel_norm_grad = None;
        } else if self.kernel_norm_trainable && self.kernel_norm.is_none() {
            self.kernel_norm = Some(self.weights.sqr().sum_axis(0).sqrt());
        }
    }

    /// Creates a dense layer with custom weights and bias.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight tensor of shape [in_features, out_features]
    /// * `bias` - Bias tensor of shape [out_features]
    ///
    /// # Errors
    ///
    /// Returns an error if the shapes are incompatible
    pub fn from_weights(weights: Tensor, bias: Tensor) -> Result<Self, LayerError> {
        if weights.ndim() != 2 {
            return Err(LayerError::ConfigError {
                message: format!("Weights must be 2D, got {}D", weights.ndim()),
            });
        }
        if bias.ndim() != 1 {
            return Err(LayerError::ConfigError {
                message: format!("Bias must be 1D, got {}D", bias.ndim()),
            });
        }
        if weights.shape()[1] != bias.shape()[0] {
            return Err(LayerError::ShapeMismatch {
                expected: vec![weights.shape()[1]],
                actual: vec![bias.shape()[0]],
            });
        }

        let in_features = weights.shape()[0];
        let out_features = weights.shape()[1];
        // Compute initial trainable kernel norm scale from the provided weights
        // (matches Python: np.linalg.norm(init_kernel, axis=0)).
        let kernel_norm = weights.sqr().sum_axis(0).sqrt();

        Ok(Self {
            weights,
            bias,
            kernel_regularizer: Regularizer::None,
            bias_regularizer: Regularizer::None,
            kernel_constraint: Constraint::None,
            bias_constraint: Constraint::None,
            // Match Python defaults for newly constructed Dense.
            allow_kernel_norm: true,
            kernel_norm_trainable: true,
            kernel_norm: Some(kernel_norm),
            kernel_norm_grad: None,
            weights_grad: None,
            bias_grad: None,
            cached_input: None,
            in_features,
            out_features,
            use_bias: true,
        })
    }

    /// Sets the kernel regularizer.
    pub fn with_kernel_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.kernel_regularizer = regularizer;
        self
    }

    /// Sets the bias regularizer.
    pub fn with_bias_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.bias_regularizer = regularizer;
        self
    }

    /// Sets the kernel constraint.
    pub fn with_kernel_constraint(mut self, constraint: Constraint) -> Self {
        self.kernel_constraint = constraint;
        self
    }

    /// Sets the bias constraint.
    pub fn with_bias_constraint(mut self, constraint: Constraint) -> Self {
        self.bias_constraint = constraint;
        self
    }

    /// Returns the input feature dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Returns the output feature dimension.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Returns a reference to the weights tensor.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Returns a mutable reference to the weights tensor.
    pub fn weights_mut(&mut self) -> &mut Tensor {
        &mut self.weights
    }

    /// Returns a reference to the bias tensor.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Returns a mutable reference to the bias tensor.
    pub fn bias_mut(&mut self) -> &mut Tensor {
        &mut self.bias
    }

    /// Returns whether this layer uses bias.
    pub fn has_bias(&self) -> bool {
        self.use_bias
    }

    /// Returns the weight gradients if available.
    pub fn weights_grad(&self) -> Option<&Tensor> {
        self.weights_grad.as_ref()
    }

    /// Returns the bias gradients if available.
    pub fn bias_grad(&self) -> Option<&Tensor> {
        self.bias_grad.as_ref()
    }

    /// Returns the kernel norm gradients if available.
    pub fn kernel_norm_grad(&self) -> Option<&Tensor> {
        self.kernel_norm_grad.as_ref()
    }

    /// Clears the cached input and gradients.
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.weights_grad = None;
        self.bias_grad = None;
        self.kernel_norm_grad = None;
    }

    fn normalized_weights(&self) -> (Tensor, Tensor) {
        let eps = Tensor::from_data(&[self.out_features], vec![1e-6; self.out_features]);
        // Match TF `tf.nn.l2_normalize(..., epsilon=1e-6)` semantics:
        // denom = sqrt(max(sum(x^2), epsilon)).
        let norm = self.weights.sqr().sum_axis(0).maximum(&eps).sqrt();
        let norm_broadcast = norm
            .reshape(&[1, self.out_features])
            .broadcast_as(&[self.in_features, self.out_features]);
        let w_norm = self.weights.div(&norm_broadcast);
        (norm, w_norm)
    }

    fn effective_weights(&self) -> Tensor {
        if !self.allow_kernel_norm {
            return self.weights.clone();
        }

        let (_norm, mut w_norm) = self.normalized_weights();
        if self.kernel_norm_trainable {
            if let Some(scale) = &self.kernel_norm {
                let scale_broadcast = scale
                    .reshape(&[1, self.out_features])
                    .broadcast_as(&[self.in_features, self.out_features]);
                w_norm = w_norm.mul(&scale_broadcast);
            }
        }
        w_norm
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Validate input shape
        if input.ndim() < 2 {
            return Err(LayerError::ForwardError {
                message: format!("Expected >=2D input, got {}D", input.ndim()),
            });
        }
        let in_dim = *input.shape().last().unwrap();
        if in_dim != self.in_features {
            return Err(LayerError::InvalidInputDimension {
                expected: self.in_features,
                actual: in_dim,
            });
        }

        let weights = self.effective_weights();
        if input.ndim() == 2 {
            let output = input.matmul(&weights);
            let output = if self.use_bias {
                output.add(&self.bias)
            } else {
                output
            };
            return Ok(output);
        }

        let batch = input.numel() / in_dim;
        let input_2d = input.reshape(&[batch, in_dim]);
        let mut output = input_2d.matmul(&weights);
        if self.use_bias {
            output = output.add(&self.bias);
        }
        let mut out_shape = input.shape().to_vec();
        *out_shape.last_mut().unwrap() = self.out_features;
        Ok(output.reshape(&out_shape))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Validate gradient shape
        if grad.ndim() < 2 {
            return Err(LayerError::InvalidOutputDimension {
                expected: self.out_features,
                actual: 0,
            });
        }
        let out_dim = *grad.shape().last().unwrap();
        if out_dim != self.out_features {
            return Err(LayerError::InvalidOutputDimension {
                expected: self.out_features,
                actual: out_dim,
            });
        }

        let in_dim = *input.shape().last().unwrap();
        let batch = input.numel() / in_dim;
        let input_2d = input.reshape(&[batch, in_dim]);
        let grad_2d = grad.reshape(&[batch, out_dim]);

        // Compute gradients
        // dL/dW_eff = x^T @ dL/dy
        let weights_grad_eff = input_2d.transpose().matmul(&grad_2d);

        if !self.allow_kernel_norm {
            let mut weights_grad = weights_grad_eff;
            if let Some(reg_grad) = self.kernel_regularizer.grad(&self.weights) {
                weights_grad = weights_grad.add(&reg_grad);
            }
            self.weights_grad = Some(weights_grad);
        } else {
            let (norm, w_norm) = self.normalized_weights();
            let norm_broadcast = norm
                .reshape(&[1, self.out_features])
                .broadcast_as(&[self.in_features, self.out_features]);

            let weights_grad_norm = if self.kernel_norm_trainable {
                if let Some(scale) = &self.kernel_norm {
                    let scale_broadcast = scale
                        .reshape(&[1, self.out_features])
                        .broadcast_as(&[self.in_features, self.out_features]);
                    weights_grad_eff.mul(&scale_broadcast)
                } else {
                    weights_grad_eff.clone()
                }
            } else {
                weights_grad_eff.clone()
            };

            // dL/dw = (dL/dw_norm - w_norm * sum(dL/dw_norm * w_norm, axis=0)) / norm
            let dot = weights_grad_norm.mul(&w_norm).sum_axis(0);
            let dot_broadcast = dot
                .reshape(&[1, self.out_features])
                .broadcast_as(&[self.in_features, self.out_features]);
            let mut weights_grad = weights_grad_norm
                .sub(&w_norm.mul(&dot_broadcast))
                .div(&norm_broadcast);
            if let Some(reg_grad) = self.kernel_regularizer.grad(&self.weights) {
                weights_grad = weights_grad.add(&reg_grad);
            }
            self.weights_grad = Some(weights_grad);

            if self.kernel_norm_trainable {
                // dL/dg = sum(dL/dW_eff * w_norm, axis=0)
                let kernel_norm_grad = weights_grad_eff.mul(&w_norm).sum_axis(0);
                self.kernel_norm_grad = Some(kernel_norm_grad);
            }
        }

        // dL/db = sum(dL/dy, axis=0)
        if self.use_bias {
            let mut bias_grad = grad_2d.sum_axis(0);
            if let Some(reg_grad) = self.bias_regularizer.grad(&self.bias) {
                bias_grad = bias_grad.add(&reg_grad);
            }
            self.bias_grad = Some(bias_grad);
        }

        // dL/dx = dL/dy @ W_eff^T
        let weights = self.effective_weights();
        let input_grad = grad_2d.matmul(&weights.transpose());
        Ok(input_grad.reshape(input.shape()))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        params.push(&self.weights);
        if self.use_bias {
            params.push(&self.bias);
        }
        if self.kernel_norm_trainable {
            if let Some(kernel_norm) = &self.kernel_norm {
                params.push(kernel_norm);
            }
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.push(&mut self.weights);
        if self.use_bias {
            params.push(&mut self.bias);
        }
        if self.kernel_norm_trainable {
            if let Some(kernel_norm) = &mut self.kernel_norm {
                params.push(kernel_norm);
            }
        }
        params
    }

    fn name(&self) -> &str {
        "Dense"
    }

    fn regularization_loss(&self) -> f32 {
        let mut loss = self.kernel_regularizer.loss(&self.weights);
        if self.use_bias {
            loss += self.bias_regularizer.loss(&self.bias);
        }
        loss
    }

    fn apply_constraints(&mut self) {
        self.weights = self.kernel_constraint.apply(&self.weights);
        if self.use_bias {
            self.bias = self.bias_constraint.apply(&self.bias);
        }
    }
}

/// Caches the input for backward pass (must be called during forward in training).
impl Dense {
    /// Performs forward pass and caches input for backward pass.
    ///
    /// Use this method during training to enable gradient computation.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_creation() {
        let layer = Dense::new(64, 32);
        assert_eq!(layer.in_features(), 64);
        assert_eq!(layer.out_features(), 32);
        assert_eq!(layer.weights().shape(), &[64, 32]);
        assert_eq!(layer.bias().shape(), &[32]);
    }

    #[test]
    fn test_dense_forward() {
        let layer = Dense::new(10, 5);
        let input = Tensor::ones(&[3, 10]); // batch of 3

        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 5]);
    }

    #[test]
    fn test_dense_forward_higher_rank() {
        // Mirrors Python DenseTest using (batch, ..., input_dim).
        let layer = Dense::new(2, 3);
        let input = Tensor::ones(&[3, 4, 2]);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 4, 3]);
    }

    #[test]
    fn test_dense_default_kernel_norm_enabled() {
        // Python Dense defaults allow_kernel_norm=True and kernel_norm_trainable=True.
        let layer = Dense::new(10, 5);
        assert!(layer.allow_kernel_norm);
        assert!(layer.kernel_norm_trainable);
        assert!(layer.kernel_norm.is_some());
        assert_eq!(layer.kernel_norm.as_ref().unwrap().shape(), &[5]);
    }

    #[test]
    fn test_dense_allow_kernel_norm_can_be_disabled() {
        let layer = Dense::new(10, 5).with_allow_kernel_norm(false);
        assert!(!layer.allow_kernel_norm);
    }

    #[test]
    fn test_dense_forward_invalid_input() {
        let layer = Dense::new(10, 5);
        let input = Tensor::ones(&[3, 20]); // wrong input dimension

        let result = layer.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_dense_backward() {
        let mut layer = Dense::new(10, 5);
        let input = Tensor::ones(&[3, 10]);

        // Forward pass with caching
        let _output = layer.forward_train(&input).unwrap();

        // Backward pass
        let grad = Tensor::ones(&[3, 5]);
        let input_grad = layer.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[3, 10]);
        assert!(layer.weights_grad().is_some());
        assert!(layer.bias_grad().is_some());
    }

    #[test]
    fn test_dense_parameters() {
        let layer = Dense::new(10, 5);
        let params = layer.parameters();
        // Python Dense also has a trainable kernel-norm scale by default.
        assert_eq!(params.len(), 3); // weights, bias, kernel_norm
    }

    #[test]
    fn test_dense_no_bias() {
        let layer = Dense::new_no_bias(10, 5);
        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weights + kernel_norm
    }

    #[test]
    fn test_dense_from_weights() {
        let weights = Tensor::ones(&[10, 5]);
        let bias = Tensor::zeros(&[5]);

        let layer = Dense::from_weights(weights, bias).unwrap();
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
    }

    #[test]
    fn test_dense_from_weights_invalid() {
        let weights = Tensor::ones(&[10, 5]);
        let bias = Tensor::zeros(&[10]); // wrong size

        let result = Dense::from_weights(weights, bias);
        assert!(result.is_err());
    }
}
