#![allow(clippy::needless_range_loop)]
//! Normalization layers.
//!
//! This module provides normalization layers including Layer Normalization
//! and Batch Normalization.

use crate::error::LayerError;
use crate::layer::Layer;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Layer Normalization layer.
///
/// Normalizes the input across the feature dimension (last dimension),
/// then applies a learnable affine transformation.
///
/// The normalization is computed as:
/// `y = (x - mean) / sqrt(var + eps) * gamma + beta`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    regularizer: Regularizer,
    eps: f32,
    normalized_shape: usize,
    cached_input: Option<Tensor>,
    cached_mean: Option<Tensor>,
    cached_var: Option<Tensor>,
    gamma_grad: Option<Tensor>,
    beta_grad: Option<Tensor>,
}

impl LayerNorm {
    /// Creates a new Layer Normalization layer.
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            gamma: Tensor::ones(&[normalized_shape]),
            beta: Tensor::zeros(&[normalized_shape]),
            regularizer: Regularizer::None,
            eps: 1e-6,
            normalized_shape,
            cached_input: None,
            cached_mean: None,
            cached_var: None,
            gamma_grad: None,
            beta_grad: None,
        }
    }

    /// Creates a Layer Normalization layer with custom epsilon.
    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        let mut layer = Self::new(normalized_shape);
        layer.eps = eps;
        layer
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
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

        let axis = input.ndim().saturating_sub(1);
        let mean = input.mean_axis(axis);
        let var = input.var_axis(axis);

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
        if input.ndim() < 2 {
            return Err(LayerError::ForwardError {
                message: format!("LayerNorm expects >=2D input, got {}D", input.ndim()),
            });
        }
        let dim = *input.shape().last().unwrap();
        if dim != self.normalized_shape {
            return Err(LayerError::InvalidInputDimension {
                expected: self.normalized_shape,
                actual: dim,
            });
        }

        let axis = input.ndim() - 1;
        let mean = input.mean_axis(axis);
        let var = input.var_axis(axis);

        let mut mean_shape = mean.shape().to_vec();
        mean_shape.push(1);
        let mut var_shape = var.shape().to_vec();
        var_shape.push(1);

        let mean_b = mean.reshape(&mean_shape).broadcast_as(input.shape());
        let var_b = var.reshape(&var_shape).broadcast_as(input.shape());
        let std = var_b
            .add(&Tensor::from_data(&[1], vec![self.eps]))
            .sqrt();
        let normalized = input.sub(&mean_b).div(&std);

        let mut gamma_shape = vec![1; input.ndim() - 1];
        gamma_shape.push(dim);
        let gamma_b = self.gamma.reshape(&gamma_shape).broadcast_as(input.shape());
        let beta_b = self.beta.reshape(&gamma_shape).broadcast_as(input.shape());

        Ok(normalized.mul(&gamma_b).add(&beta_b))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let dim = *input.shape().last().unwrap();
        let outer = input.numel() / dim;

        let input_2d = input.reshape(&[outer, dim]);
        let grad_2d = grad.reshape(&[outer, dim]);

        let mean = input_2d.mean_axis(1);
        let var = input_2d.var_axis(1);
        let std = var
            .add(&Tensor::from_data(&[1], vec![self.eps]))
            .sqrt();

        let mean_b = mean.reshape(&[outer, 1]).broadcast_as(&[outer, dim]);
        let std_b = std.reshape(&[outer, 1]).broadcast_as(&[outer, dim]);
        let x_norm = input_2d.sub(&mean_b).div(&std_b);

        let gamma_b = self.gamma.reshape(&[1, dim]).broadcast_as(&[outer, dim]);

        let mut gamma_grad = grad_2d.mul(&x_norm).sum_axis(0);
        let mut beta_grad = grad_2d.sum_axis(0);
        if let Some(reg_grad) = self.regularizer.grad(&self.gamma) {
            gamma_grad = gamma_grad.add(&reg_grad);
        }
        if let Some(reg_grad) = self.regularizer.grad(&self.beta) {
            beta_grad = beta_grad.add(&reg_grad);
        }
        self.gamma_grad = Some(gamma_grad);
        self.beta_grad = Some(beta_grad);

        let n = dim as f32;
        let grad_sum = grad_2d.sum_axis(1);
        let grad_dot = grad_2d.mul(&x_norm).sum_axis(1);
        let grad_sum_b = grad_sum.reshape(&[outer, 1]).broadcast_as(&[outer, dim]);
        let grad_dot_b = grad_dot.reshape(&[outer, 1]).broadcast_as(&[outer, dim]);

        let dx = grad_2d
            .sub(&grad_sum_b.scale(1.0 / n))
            .sub(&x_norm.mul(&grad_dot_b).scale(1.0 / n))
            .div(&std_b)
            .mul(&gamma_b);

        Ok(dx.reshape(input.shape()))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn regularization_loss(&self) -> f32 {
        self.regularizer.loss(&self.gamma) + self.regularizer.loss(&self.beta)
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }
}

/// Batch Normalization layer for N-D inputs.
///
/// Normalizes across all dimensions except the last feature dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNorm {
    gamma: Tensor,
    beta: Tensor,
    regularizer: Regularizer,
    gamma_grad: Option<Tensor>,
    beta_grad: Option<Tensor>,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: f32,
    eps: f32,
    renorm: bool,
    renorm_clipping: Option<(f32, f32, f32)>,
    renorm_momentum: f32,
    cached_input: Option<Tensor>,
    cached_mean: Option<Tensor>,
    cached_var: Option<Tensor>,
    cached_std: Option<Tensor>,
    cached_x_norm: Option<Tensor>,
    cached_r: Option<Tensor>,
}

impl BatchNorm {
    /// Creates a new BatchNorm layer.
    pub fn new(num_features: usize) -> Self {
        Self {
            gamma: Tensor::ones(&[num_features]),
            beta: Tensor::zeros(&[num_features]),
            regularizer: Regularizer::None,
            gamma_grad: None,
            beta_grad: None,
            running_mean: Tensor::zeros(&[num_features]),
            running_var: Tensor::ones(&[num_features]),
            momentum: 0.99,
            eps: 1e-6,
            renorm: false,
            renorm_clipping: None,
            renorm_momentum: 0.99,
            cached_input: None,
            cached_mean: None,
            cached_var: None,
            cached_std: None,
            cached_x_norm: None,
            cached_r: None,
        }
    }

    /// Creates BatchNorm with custom momentum and epsilon.
    pub fn with_momentum(num_features: usize, momentum: f32, eps: f32) -> Self {
        let mut bn = Self::new(num_features);
        bn.momentum = momentum;
        bn.eps = eps;
        bn
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    /// Enables batch renorm with optional clipping (rmin, rmax, dmax).
    pub fn with_renorm(
        mut self,
        renorm: bool,
        clipping: Option<(f32, f32, f32)>,
        renorm_momentum: f32,
    ) -> Self {
        self.renorm = renorm;
        self.renorm_clipping = clipping;
        self.renorm_momentum = renorm_momentum;
        self
    }

    /// Performs forward pass and caches values for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() < 2 {
            return Err(LayerError::ForwardError {
                message: format!("BatchNorm expects >=2D input, got {}D", input.ndim()),
            });
        }

        let dim = *input.shape().last().unwrap();
        if dim != self.gamma.shape()[0] {
            return Err(LayerError::InvalidInputDimension {
                expected: self.gamma.shape()[0],
                actual: dim,
            });
        }

        let outer = input.numel() / dim;
        let input_2d = input.reshape(&[outer, dim]);

        let mean = input_2d.mean_axis(0);
        let var = input_2d.var_axis(0);
        let eps_vec = Tensor::from_data(&[dim], vec![self.eps; dim]);
        let std = var.add(&eps_vec).sqrt();

        let mean_b = mean.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let std_b = std.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let x_norm = input_2d.sub(&mean_b).div(&std_b);

        let (x_hat, r_opt) = if self.renorm {
            let running_std = self.running_var.add(&eps_vec).sqrt();
            let mut r = std.div(&running_std);
            let mut d = mean.sub(&self.running_mean).div(&running_std);

            if let Some((rmin, rmax, dmax)) = self.renorm_clipping {
                r = r.clamp(rmin, rmax);
                d = d.clamp(-dmax, dmax);
            }

            let r_b = r.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
            let d_b = d.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
            (x_norm.mul(&r_b).add(&d_b), Some(r))
        } else {
            (x_norm.clone(), None)
        };

        let gamma_b = self.gamma.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let beta_b = self.beta.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let output = x_hat.mul(&gamma_b).add(&beta_b);

        let momentum = self.momentum;
        let mean_update = self.running_mean.scale(momentum).add(&mean.scale(1.0 - momentum));
        let var_update = self.running_var.scale(momentum).add(&var.scale(1.0 - momentum));
        self.running_mean = mean_update;
        self.running_var = var_update;

        self.cached_input = Some(input.clone());
        self.cached_mean = Some(mean);
        self.cached_var = Some(var);
        self.cached_std = Some(std);
        self.cached_x_norm = Some(x_norm);
        self.cached_r = r_opt;

        Ok(output.reshape(input.shape()))
    }
}

impl Layer for BatchNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() < 2 {
            return Err(LayerError::ForwardError {
                message: format!("BatchNorm expects >=2D input, got {}D", input.ndim()),
            });
        }

        let dim = *input.shape().last().unwrap();
        if dim != self.gamma.shape()[0] {
            return Err(LayerError::InvalidInputDimension {
                expected: self.gamma.shape()[0],
                actual: dim,
            });
        }

        let outer = input.numel() / dim;
        let input_2d = input.reshape(&[outer, dim]);

        let eps_vec = Tensor::from_data(&[dim], vec![self.eps; dim]);
        let std = self.running_var.add(&eps_vec).sqrt();

        let mean_b = self
            .running_mean
            .reshape(&[1, dim])
            .broadcast_as(&[outer, dim]);
        let std_b = std.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let x_norm = input_2d.sub(&mean_b).div(&std_b);

        let gamma_b = self.gamma.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let beta_b = self.beta.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        Ok(x_norm.mul(&gamma_b).add(&beta_b).reshape(input.shape()))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self.cached_input.as_ref().ok_or(LayerError::NotInitialized)?;
        let x_norm = self.cached_x_norm.as_ref().ok_or(LayerError::NotInitialized)?;
        let std = self.cached_std.as_ref().ok_or(LayerError::NotInitialized)?;

        let dim = *input.shape().last().unwrap();
        let outer = input.numel() / dim;
        let grad_2d = grad.reshape(&[outer, dim]);

        let gamma_b = self.gamma.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let mut dxhat = grad_2d.mul(&gamma_b);

        if self.renorm {
            if let Some(r) = &self.cached_r {
                let r_b = r.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
                dxhat = dxhat.mul(&r_b);
            }
        }

        let mut dgamma = grad_2d.mul(x_norm).sum_axis(0);
        let mut dbeta = grad_2d.sum_axis(0);
        if let Some(reg_grad) = self.regularizer.grad(&self.gamma) {
            dgamma = dgamma.add(&reg_grad);
        }
        if let Some(reg_grad) = self.regularizer.grad(&self.beta) {
            dbeta = dbeta.add(&reg_grad);
        }
        self.gamma_grad = Some(dgamma);
        self.beta_grad = Some(dbeta);

        let n = outer as f32;
        let sum_dxhat = dxhat.sum_axis(0);
        let sum_dxhat_xnorm = dxhat.mul(x_norm).sum_axis(0);
        let sum_dxhat_b = sum_dxhat.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let sum_dxhat_xnorm_b = sum_dxhat_xnorm.reshape(&[1, dim]).broadcast_as(&[outer, dim]);

        let std_b = std.reshape(&[1, dim]).broadcast_as(&[outer, dim]);
        let dx = dxhat
            .scale(n)
            .sub(&sum_dxhat_b)
            .sub(&x_norm.mul(&sum_dxhat_xnorm_b))
            .div(&std_b)
            .scale(1.0 / n);

        Ok(dx.reshape(input.shape()))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.gamma, &self.beta]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.gamma, &mut self.beta]
    }

    fn regularization_loss(&self) -> f32 {
        self.regularizer.loss(&self.gamma) + self.regularizer.loss(&self.beta)
    }

    fn name(&self) -> &str {
        "BatchNorm"
    }
}

/// GradNorm helper for multi-task gradient balancing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradNorm {
    loss_names: Vec<String>,
    scale: f32,
    loss_pow: f32,
    relative_diff: bool,
    epsilon: f32,
    weights: Tensor,
}

impl GradNorm {
    /// Creates a new GradNorm helper.
    pub fn new(loss_names: Vec<String>) -> Self {
        let n = loss_names.len().max(1);
        Self {
            loss_names,
            scale: 1.0,
            loss_pow: 2.0,
            relative_diff: false,
            epsilon: 1e-6,
            weights: Tensor::zeros(&[n]),
        }
    }

    /// Sets GradNorm hyperparameters.
    pub fn with_params(mut self, scale: f32, loss_pow: f32, relative_diff: bool, epsilon: f32) -> Self {
        self.scale = scale;
        self.loss_pow = loss_pow;
        self.relative_diff = relative_diff;
        self.epsilon = epsilon;
        self
    }

    /// Returns current softmax weights.
    pub fn weights(&self) -> Tensor {
        self.weights.softmax(0)
    }

    /// Computes GradNorm loss and weighted loss.
    pub fn compute(
        &self,
        losses: &[f32],
        grads: &[Tensor],
    ) -> Result<(f32, f32, Tensor), LayerError> {
        if losses.len() != grads.len() {
            return Err(LayerError::ForwardError {
                message: "GradNorm losses and grads length mismatch".to_string(),
            });
        }
        let n = losses.len().max(1) as f32;
        let weights = self.weights();
        let w = weights.data();

        let mut gnorms = Vec::with_capacity(grads.len());
        for g in grads {
            let norm = (g.sqr().sum()).sqrt();
            gnorms.push(norm);
        }

        let mut avgnorm = 0.0f32;
        for (i, &gn) in gnorms.iter().enumerate() {
            avgnorm += gn * w[i];
        }
        avgnorm /= n;

        let mut gnorm_loss = 0.0f32;
        for (i, &gn) in gnorms.iter().enumerate() {
            let mut diff = (gn * w[i] - avgnorm).abs();
            if self.relative_diff {
                diff /= avgnorm + self.epsilon;
            }
            gnorm_loss += diff.powf(self.loss_pow);
        }
        gnorm_loss *= self.scale;

        let mut weighted_loss = 0.0f32;
        for (i, &loss) in losses.iter().enumerate() {
            weighted_loss += loss * w[i];
        }

        Ok((gnorm_loss, weighted_loss, weights))
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
    fn test_batch_norm_forward() {
        let bn = BatchNorm::new(4);
        let input = Tensor::rand(&[2, 3, 4]);
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }
}
