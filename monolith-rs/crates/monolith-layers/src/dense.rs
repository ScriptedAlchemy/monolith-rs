//! Dense (fully connected) layer implementation.
//!
//! This module provides the [`Dense`] layer, which performs a linear transformation
//! `y = xW + b` where W is the weight matrix and b is the bias vector.

use crate::error::LayerError;
use crate::layer::Layer;
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
    /// Weights are initialized using Xavier/Glorot initialization and
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
        // Xavier/Glorot initialization
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let weights = Tensor::randn(&[in_features, out_features], 0.0, std);
        let bias = Tensor::zeros(&[out_features]);

        Self {
            weights,
            bias,
            weights_grad: None,
            bias_grad: None,
            cached_input: None,
            in_features,
            out_features,
            use_bias: true,
        }
    }

    /// Creates a new dense layer without bias.
    ///
    /// # Arguments
    ///
    /// * `in_features` - Number of input features
    /// * `out_features` - Number of output features
    pub fn new_no_bias(in_features: usize, out_features: usize) -> Self {
        let mut layer = Self::new(in_features, out_features);
        layer.use_bias = false;
        layer.bias = Tensor::zeros(&[out_features]);
        layer
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

        Ok(Self {
            weights,
            bias,
            weights_grad: None,
            bias_grad: None,
            cached_input: None,
            in_features,
            out_features,
            use_bias: true,
        })
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

    /// Returns a reference to the bias tensor.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }

    /// Returns the weight gradients if available.
    pub fn weights_grad(&self) -> Option<&Tensor> {
        self.weights_grad.as_ref()
    }

    /// Returns the bias gradients if available.
    pub fn bias_grad(&self) -> Option<&Tensor> {
        self.bias_grad.as_ref()
    }

    /// Clears the cached input and gradients.
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.weights_grad = None;
        self.bias_grad = None;
    }
}

impl Layer for Dense {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Validate input shape
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Expected 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.in_features {
            return Err(LayerError::InvalidInputDimension {
                expected: self.in_features,
                actual: input.shape()[1],
            });
        }

        // Compute y = xW + b
        let output = input.matmul(&self.weights);
        let output = if self.use_bias {
            output.add(&self.bias)
        } else {
            output
        };

        Ok(output)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Validate gradient shape
        if grad.shape()[1] != self.out_features {
            return Err(LayerError::InvalidOutputDimension {
                expected: self.out_features,
                actual: grad.shape()[1],
            });
        }

        // Compute gradients
        // dL/dW = x^T @ dL/dy
        let weights_grad = input.transpose().matmul(grad);
        self.weights_grad = Some(weights_grad);

        // dL/db = sum(dL/dy, axis=0)
        if self.use_bias {
            let bias_grad = grad.sum_axis(0);
            self.bias_grad = Some(bias_grad);
        }

        // dL/dx = dL/dy @ W^T
        let input_grad = grad.matmul(&self.weights.transpose());

        Ok(input_grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        if self.use_bias {
            vec![&self.weights, &self.bias]
        } else {
            vec![&self.weights]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        if self.use_bias {
            vec![&mut self.weights, &mut self.bias]
        } else {
            vec![&mut self.weights]
        }
    }

    fn name(&self) -> &str {
        "Dense"
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
        assert_eq!(params.len(), 2); // weights and bias
    }

    #[test]
    fn test_dense_no_bias() {
        let layer = Dense::new_no_bias(10, 5);
        let params = layer.parameters();
        assert_eq!(params.len(), 1); // only weights
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
