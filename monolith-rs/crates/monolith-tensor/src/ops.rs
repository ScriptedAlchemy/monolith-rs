//! Common tensor operations.
//!
//! This module provides common activation functions and operations
//! that can be applied to tensors, including softmax, relu, sigmoid, and tanh.

use crate::ndarray_backend::NdArrayTensor;
use crate::tensor::Tensor;

/// Applies the softmax function along the last axis.
///
/// The softmax function converts a vector of values into a probability distribution.
/// Each element is transformed to exp(x) / sum(exp(x)).
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A new tensor with softmax applied along the last axis.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::softmax};
///
/// let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
/// let result = softmax(&t);
/// let sum: f32 = result.to_vec().iter().sum();
/// assert!((sum - 1.0).abs() < 1e-6);
/// ```
pub fn softmax(tensor: &NdArrayTensor) -> NdArrayTensor {
    let shape = tensor.shape();
    let data = tensor.to_vec();

    if shape.is_empty() {
        return NdArrayTensor::ones(&[]);
    }

    let last_dim = *shape.last().unwrap();
    let num_rows = data.len() / last_dim;

    let mut result = vec![0.0f32; data.len()];

    for row in 0..num_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let row_data = &data[start..end];

        // Find max for numerical stability
        let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute exp(x - max) and sum
        let mut exp_sum = 0.0f32;
        for (i, &x) in row_data.iter().enumerate() {
            let exp_val = (x - max_val).exp();
            result[start + i] = exp_val;
            exp_sum += exp_val;
        }

        // Normalize
        for i in 0..last_dim {
            result[start + i] /= exp_sum;
        }
    }

    NdArrayTensor::from_slice(&result, shape)
}

/// Applies the ReLU (Rectified Linear Unit) activation function element-wise.
///
/// ReLU(x) = max(0, x)
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A new tensor with ReLU applied element-wise.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::relu};
///
/// let t = NdArrayTensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);
/// let result = relu(&t);
/// assert_eq!(result.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
/// ```
pub fn relu(tensor: &NdArrayTensor) -> NdArrayTensor {
    tensor.map(|x| x.max(0.0))
}

/// Applies the Leaky ReLU activation function element-wise.
///
/// Leaky ReLU(x) = x if x >= 0, else alpha * x
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `alpha` - The slope for negative values (typically 0.01).
///
/// # Returns
///
/// A new tensor with Leaky ReLU applied element-wise.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::leaky_relu};
///
/// let t = NdArrayTensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);
/// let result = leaky_relu(&t, 0.1);
/// assert_eq!(result.to_vec(), vec![-0.1, 0.0, 1.0, 2.0]);
/// ```
pub fn leaky_relu(tensor: &NdArrayTensor, alpha: f32) -> NdArrayTensor {
    tensor.map(|x| if x >= 0.0 { x } else { alpha * x })
}

/// Applies the sigmoid activation function element-wise.
///
/// sigmoid(x) = 1 / (1 + exp(-x))
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A new tensor with sigmoid applied element-wise.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::sigmoid};
///
/// let t = NdArrayTensor::from_slice(&[0.0], &[1]);
/// let result = sigmoid(&t);
/// assert!((result.to_vec()[0] - 0.5).abs() < 1e-6);
/// ```
pub fn sigmoid(tensor: &NdArrayTensor) -> NdArrayTensor {
    tensor.map(|x| 1.0 / (1.0 + (-x).exp()))
}

/// Applies the tanh (hyperbolic tangent) activation function element-wise.
///
/// tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A new tensor with tanh applied element-wise.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::tanh};
///
/// let t = NdArrayTensor::from_slice(&[0.0], &[1]);
/// let result = tanh(&t);
/// assert!(result.to_vec()[0].abs() < 1e-6);
/// ```
pub fn tanh(tensor: &NdArrayTensor) -> NdArrayTensor {
    tensor.map(|x| x.tanh())
}

/// Applies the GELU (Gaussian Error Linear Unit) activation function element-wise.
///
/// GELU(x) = x * Φ(x), where Φ is the cumulative distribution function of the
/// standard normal distribution. This implementation uses the tanh approximation.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A new tensor with GELU applied element-wise.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::gelu};
///
/// let t = NdArrayTensor::from_slice(&[0.0, 1.0, -1.0], &[3]);
/// let result = gelu(&t);
/// assert!(result.to_vec()[0].abs() < 1e-6); // GELU(0) = 0
/// ```
pub fn gelu(tensor: &NdArrayTensor) -> NdArrayTensor {
    // Using the tanh approximation:
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/π)
    const COEFF: f32 = 0.044715;

    tensor.map(|x| {
        let x3 = x * x * x;
        let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
        0.5 * x * (1.0 + inner.tanh())
    })
}

/// Applies the Swish activation function element-wise.
///
/// Swish(x) = x * sigmoid(x)
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A new tensor with Swish applied element-wise.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::swish};
///
/// let t = NdArrayTensor::from_slice(&[0.0, 1.0, -1.0], &[3]);
/// let result = swish(&t);
/// assert!(result.to_vec()[0].abs() < 1e-6); // Swish(0) = 0
/// ```
pub fn swish(tensor: &NdArrayTensor) -> NdArrayTensor {
    tensor.map(|x| x / (1.0 + (-x).exp()))
}

/// Clips tensor values to be within a specified range.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
/// * `min` - The minimum value.
/// * `max` - The maximum value.
///
/// # Returns
///
/// A new tensor with values clipped to [min, max].
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::clip};
///
/// let t = NdArrayTensor::from_slice(&[-2.0, 0.0, 2.0, 4.0], &[4]);
/// let result = clip(&t, -1.0, 3.0);
/// assert_eq!(result.to_vec(), vec![-1.0, 0.0, 2.0, 3.0]);
/// ```
pub fn clip(tensor: &NdArrayTensor, min: f32, max: f32) -> NdArrayTensor {
    tensor.map(|x| x.clamp(min, max))
}

/// Computes the log-softmax function along the last axis.
///
/// log_softmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
///
/// This is more numerically stable than computing log(softmax(x)) directly.
///
/// # Arguments
///
/// * `tensor` - The input tensor.
///
/// # Returns
///
/// A new tensor with log-softmax applied along the last axis.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor, ops::log_softmax};
///
/// let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
/// let result = log_softmax(&t);
/// // exp of log_softmax should sum to 1
/// let exp_sum: f32 = result.to_vec().iter().map(|x| x.exp()).sum();
/// assert!((exp_sum - 1.0).abs() < 1e-5);
/// ```
pub fn log_softmax(tensor: &NdArrayTensor) -> NdArrayTensor {
    let shape = tensor.shape();
    let data = tensor.to_vec();

    if shape.is_empty() {
        return NdArrayTensor::zeros(&[]);
    }

    let last_dim = *shape.last().unwrap();
    let num_rows = data.len() / last_dim;

    let mut result = vec![0.0f32; data.len()];

    for row in 0..num_rows {
        let start = row * last_dim;
        let end = start + last_dim;
        let row_data = &data[start..end];

        // Find max for numerical stability
        let max_val = row_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Compute log(sum(exp(x - max)))
        let log_sum_exp: f32 = row_data
            .iter()
            .map(|&x| (x - max_val).exp())
            .sum::<f32>()
            .ln();

        // Compute x - max - log_sum_exp
        for (i, &x) in row_data.iter().enumerate() {
            result[start + i] = x - max_val - log_sum_exp;
        }
    }

    NdArrayTensor::from_slice(&result, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = softmax(&t);
        let sum: f32 = result.to_vec().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Test that output is positive
        assert!(result.to_vec().iter().all(|&x| x > 0.0));

        // Test that larger input gives larger output
        let vec = result.to_vec();
        assert!(vec[2] > vec[1]);
        assert!(vec[1] > vec[0]);
    }

    #[test]
    fn test_softmax_2d() {
        let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = softmax(&t);
        let vec = result.to_vec();

        // Each row should sum to 1
        let row1_sum: f32 = vec[0..3].iter().sum();
        let row2_sum: f32 = vec[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let t = NdArrayTensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let result = relu(&t);
        assert_eq!(result.to_vec(), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_leaky_relu() {
        let t = NdArrayTensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
        let result = leaky_relu(&t, 0.1);
        assert_eq!(result.to_vec(), vec![-0.2, -0.1, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid() {
        let t = NdArrayTensor::from_slice(&[0.0], &[1]);
        let result = sigmoid(&t);
        assert!((result.to_vec()[0] - 0.5).abs() < 1e-6);

        // Test bounds
        let t2 = NdArrayTensor::from_slice(&[-100.0, 100.0], &[2]);
        let result2 = sigmoid(&t2);
        let vec = result2.to_vec();
        assert!(vec[0] < 1e-6); // sigmoid(-100) ≈ 0
        assert!((vec[1] - 1.0).abs() < 1e-6); // sigmoid(100) ≈ 1
    }

    #[test]
    fn test_tanh() {
        let t = NdArrayTensor::from_slice(&[0.0], &[1]);
        let result = tanh(&t);
        assert!(result.to_vec()[0].abs() < 1e-6);

        // Test bounds
        let t2 = NdArrayTensor::from_slice(&[-100.0, 100.0], &[2]);
        let result2 = tanh(&t2);
        let vec = result2.to_vec();
        assert!((vec[0] + 1.0).abs() < 1e-6); // tanh(-100) ≈ -1
        assert!((vec[1] - 1.0).abs() < 1e-6); // tanh(100) ≈ 1
    }

    #[test]
    fn test_gelu() {
        let t = NdArrayTensor::from_slice(&[0.0, 1.0, -1.0], &[3]);
        let result = gelu(&t);
        let vec = result.to_vec();

        // GELU(0) = 0
        assert!(vec[0].abs() < 1e-6);
        // GELU(1) ≈ 0.841
        assert!((vec[1] - 0.841).abs() < 0.01);
        // GELU(-1) ≈ -0.159
        assert!((vec[2] + 0.159).abs() < 0.01);
    }

    #[test]
    fn test_swish() {
        let t = NdArrayTensor::from_slice(&[0.0, 1.0, -1.0], &[3]);
        let result = swish(&t);
        let vec = result.to_vec();

        // Swish(0) = 0
        assert!(vec[0].abs() < 1e-6);
        // Swish(1) = 1 * sigmoid(1) ≈ 0.731
        assert!((vec[1] - 0.731).abs() < 0.01);
    }

    #[test]
    fn test_clip() {
        let t = NdArrayTensor::from_slice(&[-2.0, 0.0, 2.0, 4.0], &[4]);
        let result = clip(&t, -1.0, 3.0);
        assert_eq!(result.to_vec(), vec![-1.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn test_log_softmax() {
        let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = log_softmax(&t);

        // exp of log_softmax should sum to 1
        let exp_sum: f32 = result.to_vec().iter().map(|x| x.exp()).sum();
        assert!((exp_sum - 1.0).abs() < 1e-5);

        // All values should be negative (log of probabilities)
        assert!(result.to_vec().iter().all(|&x| x < 0.0));
    }
}
