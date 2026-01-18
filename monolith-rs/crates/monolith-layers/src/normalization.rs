//! Normalization layers.
//!
//! This module provides normalization layers including Layer Normalization
//! and Batch Normalization.

use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Layer Normalization layer.
///
/// Normalizes the input across the feature dimension (last dimension),
/// then applies a learnable affine transformation.
///
/// The normalization is computed as:
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`
///
/// # Example
///
/// ```
/// use monolith_layers::normalization::LayerNorm;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let layer_norm = LayerNorm::new(64);
/// let input = Tensor::rand(&[32, 64]);
/// let output = layer_norm.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[32, 64]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    /// Learnable scale parameter (gamma)
    gamma: Tensor,
    /// Learnable shift parameter (beta)
    beta: Tensor,
    /// Small constant for numerical stability
    eps: f32,
    /// Normalized dimension
    normalized_shape: usize,
    /// Cached values for backward pass
    cached_input: Option<Tensor>,
    cached_mean: Option<Tensor>,
    cached_var: Option<Tensor>,
    /// Gradient accumulators
    gamma_grad: Option<Tensor>,
    beta_grad: Option<Tensor>,
}

impl LayerNorm {
    /// Creates a new Layer Normalization layer.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - The size of the dimension to normalize
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: Tensor::ones(&[normalized_shape]),
            beta: Tensor::zeros(&[normalized_shape]),
            eps: 1e-5,
            normalized_shape,
            cached_input: None,
            cached_mean: None,
            cached_var: None,
            gamma_grad: None,
            beta_grad: None,
        }
    }

    /// Creates a Layer Normalization layer with custom epsilon.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - The size of the dimension to normalize
    /// * `eps` - Small constant for numerical stability
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        let mut layer = Self::new(normalized_shape);
        layer.eps = eps;
        layer
    }

    /// Returns the normalized shape.
    pub fn normalized_shape(&self) -> usize {
        self.normalized_shape
    }

    /// Returns a reference to gamma (scale parameter).
    pub fn gamma(&self) -> &Tensor {
        &self.gamma
    }

    /// Returns a reference to beta (shift parameter).
    pub fn beta(&self) -> &Tensor {
        &self.beta
    }

    /// Performs forward pass and caches values for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());

        // Compute mean and variance along last dimension
        let mean = input.mean_axis(1);
        let var = input.var_axis(1);

        self.cached_mean = Some(mean.clone());
        self.cached_var = Some(var.clone());

        self.forward(input)
    }

    /// Clears cached values.
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.cached_mean = None;
        self.cached_var = None;
        self.gamma_grad = None;
        self.beta_grad = None;
    }
}

impl Layer for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("LayerNorm expects 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.normalized_shape {
            return Err(LayerError::InvalidInputDimension {
                expected: self.normalized_shape,
                actual: input.shape()[1],
            });
        }

        let batch_size = input.shape()[0];
        let dim = input.shape()[1];

        // Compute mean and variance along last dimension
        let mean = input.mean_axis(1);
        let var = input.var_axis(1);

        // Normalize: (x - mean) / sqrt(var + eps)
        let mut normalized = vec![0.0; input.numel()];
        for i in 0..batch_size {
            let mu = mean.data()[i];
            let std = (var.data()[i] + self.eps).sqrt();
            for j in 0..dim {
                let idx = i * dim + j;
                normalized[idx] = (input.data()[idx] - mu) / std;
            }
        }

        // Apply affine transformation: gamma * normalized + beta
        let mut output = vec![0.0; input.numel()];
        for i in 0..batch_size {
            for j in 0..dim {
                let idx = i * dim + j;
                output[idx] = self.gamma.data()[j] * normalized[idx] + self.beta.data()[j];
            }
        }

        Ok(Tensor::from_data(input.shape(), output))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let mean = self
            .cached_mean
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let var = self.cached_var.as_ref().ok_or(LayerError::NotInitialized)?;

        let batch_size = input.shape()[0];
        let dim = input.shape()[1];
        let n = dim as f32;

        // Compute normalized values
        let mut x_norm = vec![0.0; input.numel()];
        let mut std_inv = vec![0.0; batch_size];
        for i in 0..batch_size {
            std_inv[i] = 1.0 / (var.data()[i] + self.eps).sqrt();
            for j in 0..dim {
                let idx = i * dim + j;
                x_norm[idx] = (input.data()[idx] - mean.data()[i]) * std_inv[i];
            }
        }

        // Gradient w.r.t. gamma: sum over batch of grad * x_norm
        let mut gamma_grad = vec![0.0; dim];
        for j in 0..dim {
            for i in 0..batch_size {
                let idx = i * dim + j;
                gamma_grad[j] += grad.data()[idx] * x_norm[idx];
            }
        }
        self.gamma_grad = Some(Tensor::from_data(&[dim], gamma_grad));

        // Gradient w.r.t. beta: sum over batch of grad
        let beta_grad = grad.sum_axis(0);
        self.beta_grad = Some(beta_grad);

        // Gradient w.r.t. input
        let mut input_grad = vec![0.0; input.numel()];
        for i in 0..batch_size {
            // Compute intermediate values for this sample
            let mut dx_norm_sum = 0.0;
            let mut dx_norm_x_norm_sum = 0.0;

            for j in 0..dim {
                let idx = i * dim + j;
                let dx_norm = grad.data()[idx] * self.gamma.data()[j];
                dx_norm_sum += dx_norm;
                dx_norm_x_norm_sum += dx_norm * x_norm[idx];
            }

            // Compute gradient for each element
            for j in 0..dim {
                let idx = i * dim + j;
                let dx_norm = grad.data()[idx] * self.gamma.data()[j];
                input_grad[idx] =
                    std_inv[i] / n * (n * dx_norm - dx_norm_sum - x_norm[idx] * dx_norm_x_norm_sum);
            }
        }

        Ok(Tensor::from_data(input.shape(), input_grad))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }
}

/// Batch Normalization layer (stub implementation).
///
/// Normalizes the input across the batch dimension, then applies
/// a learnable affine transformation.
///
/// Note: This is a simplified implementation that does not include
/// running statistics for inference mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNorm {
    /// Learnable scale parameter (gamma)
    gamma: Tensor,
    /// Learnable shift parameter (beta)
    beta: Tensor,
    /// Running mean for inference
    running_mean: Tensor,
    /// Running variance for inference
    running_var: Tensor,
    /// Momentum for running statistics
    momentum: f32,
    /// Small constant for numerical stability
    eps: f32,
    /// Number of features
    num_features: usize,
    /// Whether in training mode
    training: bool,
    /// Cached values for backward pass
    cached_input: Option<Tensor>,
    cached_mean: Option<Tensor>,
    cached_var: Option<Tensor>,
    /// Gradient accumulators
    #[serde(skip)]
    gamma_grad: Option<Tensor>,
    #[serde(skip)]
    beta_grad: Option<Tensor>,
}

impl BatchNorm {
    /// Creates a new Batch Normalization layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features (C in [N, C] input)
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            momentum: 0.1,
            eps: 1e-5,
            num_features,
            training: true,
            cached_input: None,
            cached_mean: None,
            cached_var: None,
            gamma_grad: None,
            beta_grad: None,
        }
    }

    /// Creates a Batch Normalization layer with custom parameters.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of features
    /// * `momentum` - Momentum for running statistics
    /// * `eps` - Small constant for numerical stability
    pub fn with_params(num_features: usize, momentum: f32, eps: f32) -> Self {
        let mut layer = Self::new(num_features);
        layer.momentum = momentum;
        layer.eps = eps;
        layer
    }

    /// Returns the number of features.
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Performs forward pass and caches values for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("BatchNorm expects 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.num_features {
            return Err(LayerError::InvalidInputDimension {
                expected: self.num_features,
                actual: input.shape()[1],
            });
        }

        let batch_size = input.shape()[0];
        let dim = input.shape()[1];

        // Batch statistics: [C]
        let mean = input.mean_axis(0);
        let mut var = input.var_axis(0);
        // Match TF semantics: variance is non-negative (numerical safety).
        let var_data: Vec<f32> = var.data().iter().map(|&v| v.max(0.0)).collect();
        var = Tensor::from_data(&[dim], var_data);

        // Update running stats (momentum matches TF BN: moving = momentum * batch + (1-momentum) * moving).
        let m = self.momentum;
        let om = 1.0 - m;
        let mut new_rm = vec![0.0; dim];
        let mut new_rv = vec![0.0; dim];
        for j in 0..dim {
            new_rm[j] = m * mean.data()[j] + om * self.running_mean.data()[j];
            new_rv[j] = m * var.data()[j] + om * self.running_var.data()[j];
        }
        self.running_mean = Tensor::from_data(&[dim], new_rm);
        self.running_var = Tensor::from_data(&[dim], new_rv);

        // Cache for backward.
        self.cached_input = Some(input.clone());
        self.cached_mean = Some(mean.clone());
        self.cached_var = Some(var.clone());

        // Normalize and affine.
        let mut output = vec![0.0; input.numel()];
        for i in 0..batch_size {
            for j in 0..dim {
                let idx = i * dim + j;
                let x_norm =
                    (input.data()[idx] - mean.data()[j]) / (var.data()[j] + self.eps).sqrt();
                output[idx] = self.gamma.data()[j] * x_norm + self.beta.data()[j];
            }
        }

        Ok(Tensor::from_data(input.shape(), output))
    }

    /// Clears cached values and gradients.
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.cached_mean = None;
        self.cached_var = None;
        self.gamma_grad = None;
        self.beta_grad = None;
    }

    /// Returns gamma gradients if available.
    pub fn gamma_grad(&self) -> Option<&Tensor> {
        self.gamma_grad.as_ref()
    }

    /// Returns beta gradients if available.
    pub fn beta_grad(&self) -> Option<&Tensor> {
        self.beta_grad.as_ref()
    }

    /// Returns references to running mean/var.
    pub fn running_stats(&self) -> (&Tensor, &Tensor) {
        (&self.running_mean, &self.running_var)
    }
}

impl Layer for BatchNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("BatchNorm expects 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.num_features {
            return Err(LayerError::InvalidInputDimension {
                expected: self.num_features,
                actual: input.shape()[1],
            });
        }

        let batch_size = input.shape()[0];
        let dim = input.shape()[1];

        let (mean, var) = if self.training {
            // Use batch statistics during training
            (input.mean_axis(0), input.var_axis(0))
        } else {
            // Use running statistics during inference
            (self.running_mean.clone(), self.running_var.clone())
        };
        // Match TF semantics: variance is non-negative (numerical safety).
        let var = if var.numel() == dim {
            let var_data: Vec<f32> = var.data().iter().map(|&v| v.max(0.0)).collect();
            Tensor::from_data(&[dim], var_data)
        } else {
            var
        };

        // Normalize and apply affine transformation
        let mut output = vec![0.0; input.numel()];
        for i in 0..batch_size {
            for j in 0..dim {
                let idx = i * dim + j;
                let x_norm =
                    (input.data()[idx] - mean.data()[j]) / (var.data()[j] + self.eps).sqrt();
                output[idx] = self.gamma.data()[j] * x_norm + self.beta.data()[j];
            }
        }

        Ok(Tensor::from_data(input.shape(), output))
    }

    fn backward(&mut self, _grad: &Tensor) -> Result<Tensor, LayerError> {
        // Compute gradients for gamma, beta, and input.
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let mean = self
            .cached_mean
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let var = self.cached_var.as_ref().ok_or(LayerError::NotInitialized)?;

        let grad = _grad;
        if grad.shape() != input.shape() {
            return Err(LayerError::ShapeMismatch {
                expected: input.shape().to_vec(),
                actual: grad.shape().to_vec(),
            });
        }

        let batch_size = input.shape()[0];
        let dim = input.shape()[1];
        let n = batch_size as f32;

        // Precompute inv std and x_hat for each element.
        let mut inv_std = vec![0.0f32; dim];
        for j in 0..dim {
            inv_std[j] = 1.0 / (var.data()[j].max(0.0) + self.eps).sqrt();
        }
        let mut x_hat = vec![0.0f32; input.numel()];
        for i in 0..batch_size {
            for j in 0..dim {
                let idx = i * dim + j;
                x_hat[idx] = (input.data()[idx] - mean.data()[j]) * inv_std[j];
            }
        }

        // dBeta, dGamma over batch.
        let mut d_beta = vec![0.0f32; dim];
        let mut d_gamma = vec![0.0f32; dim];
        for i in 0..batch_size {
            for j in 0..dim {
                let idx = i * dim + j;
                d_beta[j] += grad.data()[idx];
                d_gamma[j] += grad.data()[idx] * x_hat[idx];
            }
        }
        self.beta_grad = Some(Tensor::from_data(&[dim], d_beta));
        self.gamma_grad = Some(Tensor::from_data(&[dim], d_gamma));

        // dx_hat = dY * gamma
        let mut dx_hat = vec![0.0f32; input.numel()];
        for i in 0..batch_size {
            for j in 0..dim {
                let idx = i * dim + j;
                dx_hat[idx] = grad.data()[idx] * self.gamma.data()[j];
            }
        }

        // Compute dvar and dmu per channel.
        let mut dvar = vec![0.0f32; dim];
        let mut dmu = vec![0.0f32; dim];
        for j in 0..dim {
            let mut sum_dxhat_xmu = 0.0f32;
            let mut sum_xmu = 0.0f32;
            let mut sum_dxhat = 0.0f32;
            for i in 0..batch_size {
                let idx = i * dim + j;
                let xmu = input.data()[idx] - mean.data()[j];
                sum_dxhat_xmu += dx_hat[idx] * xmu;
                sum_xmu += xmu;
                sum_dxhat += dx_hat[idx];
            }

            let inv_std_j = inv_std[j];
            let inv_std3 = inv_std_j * inv_std_j * inv_std_j;
            dvar[j] = -0.5 * sum_dxhat_xmu * inv_std3;
            dmu[j] = -inv_std_j * sum_dxhat + dvar[j] * (-2.0 * sum_xmu / n);
        }

        // dx
        let mut dx = vec![0.0f32; input.numel()];
        for i in 0..batch_size {
            for j in 0..dim {
                let idx = i * dim + j;
                let xmu = input.data()[idx] - mean.data()[j];
                dx[idx] = dx_hat[idx] * inv_std[j] + dvar[j] * 2.0 * xmu / n + dmu[j] / n;
            }
        }

        Ok(Tensor::from_data(input.shape(), dx))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn name(&self) -> &str {
        "BatchNorm"
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
    fn test_layer_norm_creation() {
        let ln = LayerNorm::new(64);
        assert_eq!(ln.normalized_shape(), 64);
        assert_eq!(ln.gamma().shape(), &[64]);
        assert_eq!(ln.beta().shape(), &[64]);
    }

    #[test]
    fn test_layer_norm_forward() {
        let ln = LayerNorm::new(10);
        let input = Tensor::rand(&[3, 10]);

        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 10]);
    }

    #[test]
    fn test_layer_norm_normalization() {
        let ln = LayerNorm::new(4);
        // Input where each row has different mean/variance
        let input = Tensor::from_data(
            &[2, 4],
            vec![
                1.0, 2.0, 3.0, 4.0, // mean=2.5, var=1.25
                10.0, 20.0, 30.0, 40.0, // mean=25, var=125
            ],
        );

        let output = ln.forward(&input).unwrap();

        // After normalization, each row should have approximately zero mean
        let row1_mean: f32 = output.data()[0..4].iter().sum::<f32>() / 4.0;
        let row2_mean: f32 = output.data()[4..8].iter().sum::<f32>() / 4.0;

        assert!(row1_mean.abs() < 0.1);
        assert!(row2_mean.abs() < 0.1);
    }

    #[test]
    fn test_layer_norm_invalid_input() {
        let ln = LayerNorm::new(64);
        let input = Tensor::rand(&[3, 32]); // Wrong dimension

        let result = ln.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_norm_parameters() {
        let ln = LayerNorm::new(64);
        let params = ln.parameters();
        assert_eq!(params.len(), 2); // gamma and beta
    }

    #[test]
    fn test_batch_norm_creation() {
        let bn = BatchNorm::new(64);
        assert_eq!(bn.num_features(), 64);
    }

    #[test]
    fn test_batch_norm_forward() {
        let bn = BatchNorm::new(10);
        let input = Tensor::rand(&[8, 10]); // Batch of 8

        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8, 10]);
    }

    #[test]
    fn test_batch_norm_training_mode() {
        let mut bn = BatchNorm::new(10);
        assert!(bn.is_training());

        bn.set_training(false);
        assert!(!bn.is_training());
    }

    #[test]
    fn test_batch_norm_parameters() {
        let bn = BatchNorm::new(64);
        let params = bn.parameters();
        assert_eq!(params.len(), 2); // gamma and beta
    }
}
