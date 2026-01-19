#![allow(clippy::needless_range_loop)]
//! Tensor type for neural network computations.
//!
//! This module provides a placeholder Tensor type that will be replaced
//! by the actual implementation from monolith-tensor once available.

use serde::{Deserialize, Serialize};

/// A multi-dimensional array for neural network computations.
///
/// This is a placeholder implementation that will be replaced by
/// the actual Tensor type from the monolith-tensor crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    /// The shape of the tensor (dimensions)
    shape: Vec<usize>,
    /// The underlying data in row-major order
    data: Vec<f32>,
}

impl Tensor {
    /// Creates a new tensor with the given shape, filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The dimensions of the tensor
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::tensor::Tensor;
    ///
    /// let t = Tensor::zeros(&[2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.numel(), 6);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: vec![0.0; numel],
        }
    }

    /// Creates a new tensor with the given shape, filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The dimensions of the tensor
    pub fn ones(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Self {
            shape: shape.to_vec(),
            data: vec![1.0; numel],
        }
    }

    /// Creates a new tensor with the given shape and data.
    ///
    /// # Arguments
    ///
    /// * `shape` - The dimensions of the tensor
    /// * `data` - The data to fill the tensor with
    ///
    /// # Panics
    ///
    /// Panics if the data length doesn't match the shape
    pub fn from_data(shape: &[usize], data: Vec<f32>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "Data length {} doesn't match shape {:?} (expected {})",
            data.len(),
            shape,
            numel
        );
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    /// Creates a tensor with random values from a uniform distribution [0, 1).
    ///
    /// # Arguments
    ///
    /// * `shape` - The dimensions of the tensor
    pub fn rand(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        // Simple LCG random number generator for reproducibility
        let mut seed: u64 = 42;
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed >> 16) & 0x7fff) as f32 / 32768.0
            })
            .collect();
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    /// Creates a tensor with random values from a normal distribution.
    ///
    /// # Arguments
    ///
    /// * `shape` - The dimensions of the tensor
    /// * `mean` - The mean of the distribution
    /// * `std` - The standard deviation of the distribution
    pub fn randn(shape: &[usize], mean: f32, std: f32) -> Self {
        let numel: usize = shape.iter().product();
        let mut seed: u64 = 42;
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                // Box-Muller transform for normal distribution
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let u1 = ((seed >> 16) & 0x7fff) as f32 / 32768.0 + 1e-10;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let u2 = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                z * std + mean
            })
            .collect();
        Self {
            shape: shape.to_vec(),
            data,
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Returns a reference to the underlying data.
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Matrix multiplication between two tensors.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply with
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions don't match
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(other.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(
            self.shape[1], other.shape[0],
            "Inner dimensions must match for matmul"
        );

        let m = self.shape[0];
        let k = self.shape[1];
        let n = other.shape[1];

        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += self.data[i * k + l] * other.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        Tensor::from_data(&[m, n], result)
    }

    /// Transposes a 2D tensor.
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
        let m = self.shape[0];
        let n = self.shape[1];

        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = self.data[i * n + j];
            }
        }

        Tensor::from_data(&[n, m], result)
    }

    /// Element-wise addition with broadcasting.
    pub fn add(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect();
            Tensor::from_data(&self.shape, data)
        } else if other.numel() == 1 {
            // Scalar broadcast
            let scalar = other.data[0];
            let data: Vec<f32> = self.data.iter().map(|a| a + scalar).collect();
            Tensor::from_data(&self.shape, data)
        } else if self.ndim() == 2 && other.ndim() == 1 && self.shape[1] == other.shape[0] {
            // Broadcast along rows (bias addition)
            let mut data = self.data.clone();
            let n = self.shape[1];
            for i in 0..self.shape[0] {
                for j in 0..n {
                    data[i * n + j] += other.data[j];
                }
            }
            Tensor::from_data(&self.shape, data)
        } else {
            panic!(
                "Cannot broadcast shapes {:?} and {:?}",
                self.shape, other.shape
            );
        }
    }

    /// Element-wise multiplication.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data: Vec<f32> = self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect();
            Tensor::from_data(&self.shape, data)
        } else if other.numel() == 1 {
            let scalar = other.data[0];
            let data: Vec<f32> = self.data.iter().map(|a| a * scalar).collect();
            Tensor::from_data(&self.shape, data)
        } else {
            panic!(
                "Cannot multiply shapes {:?} and {:?}",
                self.shape, other.shape
            );
        }
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|a| a * scalar).collect();
        Tensor::from_data(&self.shape, data)
    }

    /// Sum all elements in the tensor.
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Sum along an axis.
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.ndim(), "Axis out of bounds");

        if self.ndim() == 2 {
            if axis == 0 {
                // Sum along rows, result is [1, n]
                let n = self.shape[1];
                let mut result = vec![0.0; n];
                for i in 0..self.shape[0] {
                    for j in 0..n {
                        result[j] += self.data[i * n + j];
                    }
                }
                Tensor::from_data(&[n], result)
            } else {
                // Sum along columns, result is [m, 1]
                let n = self.shape[1];
                let result: Vec<f32> = (0..self.shape[0])
                    .map(|i| (0..n).map(|j| self.data[i * n + j]).sum())
                    .collect();
                Tensor::from_data(&[self.shape[0]], result)
            }
        } else {
            panic!("sum_axis only implemented for 2D tensors");
        }
    }

    /// Mean along an axis.
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        let sum = self.sum_axis(axis);
        let count = self.shape[axis] as f32;
        sum.scale(1.0 / count)
    }

    /// Variance along an axis.
    pub fn var_axis(&self, axis: usize) -> Tensor {
        let mean = self.mean_axis(axis);
        // Compute (x - mean)^2 and then take mean
        if self.ndim() == 2 && axis == 1 {
            let m = self.shape[0];
            let n = self.shape[1];
            let mut result = vec![0.0; m];
            for i in 0..m {
                let mu = mean.data[i];
                for j in 0..n {
                    let diff = self.data[i * n + j] - mu;
                    result[i] += diff * diff;
                }
                result[i] /= n as f32;
            }
            Tensor::from_data(&[m], result)
        } else if self.ndim() == 2 && axis == 0 {
            let m = self.shape[0];
            let n = self.shape[1];
            let mut result = vec![0.0; n];
            for j in 0..n {
                let mu = mean.data[j];
                for i in 0..m {
                    let diff = self.data[i * n + j] - mu;
                    result[j] += diff * diff;
                }
                result[j] /= m as f32;
            }
            Tensor::from_data(&[n], result)
        } else {
            panic!("var_axis only implemented for 2D tensors");
        }
    }

    /// Apply a function element-wise.
    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        let data: Vec<f32> = self.data.iter().map(|&x| f(x)).collect();
        Tensor::from_data(&self.shape, data)
    }

    /// Reshape the tensor to a new shape.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different number of elements
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of {} elements to shape {:?}",
            self.numel(),
            new_shape
        );
        Tensor::from_data(new_shape, self.data.clone())
    }
}

impl std::ops::Add for &Tensor {
    type Output = Tensor;

    fn add(self, other: &Tensor) -> Tensor {
        self.add(other)
    }
}

impl std::ops::Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &Tensor) -> Tensor {
        self.mul(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let t = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.data().iter().all(|&x| x == 0.0));

        let t = Tensor::ones(&[3, 2]);
        assert!(t.data().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_tensor_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_data(&[2, 3], data);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data()[0], 1.0);
        assert_eq!(t.data()[5], 6.0);
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::from_data(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_data(&[3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data()[0], 22.0); // 1*1 + 2*3 + 3*5
        assert_eq!(c.data()[1], 28.0); // 1*2 + 2*4 + 3*6
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::from_data(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.transpose();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data()[0], 1.0);
        assert_eq!(b.data()[1], 4.0);
        assert_eq!(b.data()[2], 2.0);
    }

    #[test]
    fn test_add_broadcast() {
        let a = Tensor::from_data(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::from_data(&[3], vec![10.0, 20.0, 30.0]);
        let c = a.add(&b);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data()[0], 11.0);
        assert_eq!(c.data()[1], 22.0);
        assert_eq!(c.data()[3], 14.0);
    }

    #[test]
    fn test_sum_axis() {
        let a = Tensor::from_data(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let sum0 = a.sum_axis(0);
        assert_eq!(sum0.shape(), &[3]);
        assert_eq!(sum0.data(), &[5.0, 7.0, 9.0]);

        let sum1 = a.sum_axis(1);
        assert_eq!(sum1.shape(), &[2]);
        assert_eq!(sum1.data(), &[6.0, 15.0]);
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::from_data(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.reshape(&[3, 2]);
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data(), a.data());
    }

    #[test]
    fn test_map() {
        let a = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = a.map(|x| x * 2.0);
        assert_eq!(b.data(), &[2.0, 4.0, 6.0, 8.0]);
    }
}
