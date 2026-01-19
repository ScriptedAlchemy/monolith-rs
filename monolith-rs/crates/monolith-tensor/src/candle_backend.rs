//! Candle-based tensor backend implementation with GPU acceleration.
//!
//! This module provides [`CandleTensor`], a tensor implementation backed by
//! the `candle-core` crate from Hugging Face. This backend supports:
//! - CPU operations (default)
//! - Metal acceleration on macOS (with `metal` feature)
//! - CUDA acceleration on Linux/Windows (with `cuda` feature)
//!
//! # Example
//!
//! ```ignore
//! use monolith_tensor::{Tensor, CandleTensor, Device};
//!
//! // Create tensors on the best available device
//! let device = CandleTensor::best_device();
//! let a = CandleTensor::zeros_on(&[2, 3], &device);
//! let b = CandleTensor::ones_on(&[2, 3], &device);
//! let c = a.add(&b);
//! ```

use candle_core::{DType as CandleDType, Device, Tensor as CandleTensorInner};

use crate::tensor::{DType, Tensor};

/// A tensor backed by candle-core's Tensor.
///
/// This struct wraps a `candle_core::Tensor` and implements the [`Tensor`] trait,
/// providing a GPU-accelerated tensor backend via Metal or CUDA.
#[derive(Clone, Debug)]
pub struct CandleTensor {
    inner: CandleTensorInner,
    /// Cached shape for the Tensor trait
    shape_cache: Vec<usize>,
}

impl CandleTensor {
    /// Returns the best available device (GPU if available, otherwise CPU).
    ///
    /// On macOS with Metal support, this returns a Metal device.
    /// On systems with CUDA, this returns a CUDA device.
    /// Otherwise, it falls back to CPU.
    pub fn best_device() -> Device {
        #[cfg(feature = "metal")]
        {
            Device::new_metal(0).unwrap_or(Device::Cpu)
        }
        #[cfg(all(feature = "cuda", not(feature = "metal")))]
        {
            Device::new_cuda(0).unwrap_or(Device::Cpu)
        }
        #[cfg(not(any(feature = "metal", feature = "cuda")))]
        {
            Device::Cpu
        }
    }

    /// Returns the CPU device.
    pub fn cpu_device() -> Device {
        Device::Cpu
    }

    /// Returns the device this tensor is on.
    pub fn device(&self) -> &Device {
        self.inner.device()
    }

    /// Creates a new CandleTensor from a candle-core Tensor.
    pub fn from_candle(inner: CandleTensorInner) -> Self {
        let shape_cache = inner.dims().to_vec();
        Self { inner, shape_cache }
    }

    /// Returns a reference to the underlying candle-core tensor.
    pub fn as_candle(&self) -> &CandleTensorInner {
        &self.inner
    }

    /// Consumes the tensor and returns the underlying candle-core tensor.
    pub fn into_candle(self) -> CandleTensorInner {
        self.inner
    }

    /// Creates a tensor filled with zeros on the specified device.
    pub fn zeros_on(shape: &[usize], device: &Device) -> Self {
        let inner = CandleTensorInner::zeros(shape, CandleDType::F32, device)
            .expect("Failed to create zeros tensor");
        Self::from_candle(inner)
    }

    /// Creates a tensor filled with ones on the specified device.
    pub fn ones_on(shape: &[usize], device: &Device) -> Self {
        let inner = CandleTensorInner::ones(shape, CandleDType::F32, device)
            .expect("Failed to create ones tensor");
        Self::from_candle(inner)
    }

    /// Creates a tensor from a slice on the specified device.
    pub fn from_slice_on(data: &[f32], shape: &[usize], device: &Device) -> Self {
        let inner =
            CandleTensorInner::from_slice(data, shape, device).expect("Failed to create tensor");
        Self::from_candle(inner)
    }

    /// Moves the tensor to the specified device.
    pub fn to_device(&self, device: &Device) -> Self {
        let inner = self.inner.to_device(device).expect("Failed to move tensor");
        Self::from_candle(inner)
    }

    /// Moves the tensor to CPU.
    pub fn to_cpu(&self) -> Self {
        self.to_device(&Device::Cpu)
    }

    /// Reshapes the tensor to a new shape.
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let inner = self.inner.reshape(new_shape).expect("Failed to reshape");
        Self::from_candle(inner)
    }

    /// Transposes a 2D tensor.
    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose only works on 2D tensors");
        let inner = self.inner.t().expect("Failed to transpose");
        Self::from_candle(inner)
    }

    /// Returns the sum of all elements.
    pub fn sum(&self) -> f32 {
        self.inner
            .sum_all()
            .expect("Failed to compute sum")
            .to_scalar::<f32>()
            .expect("Failed to get scalar")
    }

    /// Returns the mean of all elements.
    pub fn mean(&self) -> f32 {
        self.inner
            .mean_all()
            .expect("Failed to compute mean")
            .to_scalar::<f32>()
            .expect("Failed to get scalar")
    }

    /// Returns the maximum value.
    pub fn max(&self) -> f32 {
        self.inner
            .max(0)
            .expect("Failed to compute max")
            .flatten_all()
            .expect("Failed to flatten")
            .max(0)
            .expect("Failed to compute max")
            .to_scalar::<f32>()
            .expect("Failed to get scalar")
    }

    /// Returns the minimum value.
    pub fn min(&self) -> f32 {
        self.inner
            .min(0)
            .expect("Failed to compute min")
            .flatten_all()
            .expect("Failed to flatten")
            .min(0)
            .expect("Failed to compute min")
            .to_scalar::<f32>()
            .expect("Failed to get scalar")
    }

    /// Subtracts another tensor element-wise.
    pub fn sub(&self, other: &Self) -> Self {
        let inner = self
            .inner
            .sub(&other.inner)
            .expect("Failed to subtract tensors");
        Self::from_candle(inner)
    }

    /// Divides by another tensor element-wise.
    pub fn div(&self, other: &Self) -> Self {
        let inner = self
            .inner
            .div(&other.inner)
            .expect("Failed to divide tensors");
        Self::from_candle(inner)
    }

    /// Scales the tensor by a scalar.
    pub fn scale(&self, scalar: f32) -> Self {
        let inner = (self.inner.clone() * scalar as f64).expect("Failed to scale tensor");
        Self::from_candle(inner)
    }

    /// Creates a tensor with random normal values.
    pub fn randn(shape: &[usize], mean: f32, std: f32, device: &Device) -> Self {
        let inner = CandleTensorInner::randn(mean, std, shape, device)
            .expect("Failed to create random tensor");
        Self::from_candle(inner)
    }

    /// Creates a tensor with random uniform values in [0, 1).
    pub fn rand(shape: &[usize], device: &Device) -> Self {
        let inner = CandleTensorInner::rand(0.0f32, 1.0f32, shape, device)
            .expect("Failed to create random tensor");
        Self::from_candle(inner)
    }

    /// Applies ReLU activation.
    pub fn relu(&self) -> Self {
        let inner = self.inner.relu().expect("Failed to apply ReLU");
        Self::from_candle(inner)
    }

    /// Applies sigmoid activation.
    pub fn sigmoid(&self) -> Self {
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg = self.inner.neg().expect("Failed to negate");
        let exp = neg.exp().expect("Failed to exp");
        let one = CandleTensorInner::ones(self.inner.dims(), CandleDType::F32, self.device())
            .expect("Failed to create ones");
        let denom = one.add(&exp).expect("Failed to add");
        let ones = CandleTensorInner::ones(self.inner.dims(), CandleDType::F32, self.device())
            .expect("Failed to create ones");
        let inner = ones.div(&denom).expect("Failed to divide");
        Self::from_candle(inner)
    }

    /// Applies tanh activation.
    pub fn tanh(&self) -> Self {
        let inner = self.inner.tanh().expect("Failed to apply tanh");
        Self::from_candle(inner)
    }

    /// Applies softmax along the specified dimension.
    pub fn softmax(&self, dim: usize) -> Self {
        let inner = candle_nn::ops::softmax(&self.inner, dim).expect("Failed to apply softmax");
        Self::from_candle(inner)
    }

    /// Returns element-wise square root.
    pub fn sqrt(&self) -> Self {
        let inner = self.inner.sqrt().expect("Failed to compute sqrt");
        Self::from_candle(inner)
    }

    /// Returns element-wise square.
    pub fn sqr(&self) -> Self {
        let inner = self.inner.sqr().expect("Failed to compute sqr");
        Self::from_candle(inner)
    }

    /// Returns element-wise exponential.
    pub fn exp(&self) -> Self {
        let inner = self.inner.exp().expect("Failed to compute exp");
        Self::from_candle(inner)
    }

    /// Returns element-wise natural logarithm.
    pub fn log(&self) -> Self {
        let inner = self.inner.log().expect("Failed to compute log");
        Self::from_candle(inner)
    }

    /// Sums along the specified axis.
    pub fn sum_axis(&self, dim: usize) -> Self {
        let inner = self.inner.sum(dim).expect("Failed to sum");
        Self::from_candle(inner)
    }

    /// Computes mean along the specified axis.
    pub fn mean_axis(&self, dim: usize) -> Self {
        let inner = self.inner.mean(dim).expect("Failed to mean");
        Self::from_candle(inner)
    }

    /// Broadcasts the tensor to the given shape.
    pub fn broadcast(&self, shape: &[usize]) -> Self {
        let inner = self.inner.broadcast_as(shape).expect("Failed to broadcast");
        Self::from_candle(inner)
    }

    /// Returns the contiguous version of the tensor.
    pub fn contiguous(&self) -> Self {
        let inner = self.inner.contiguous().expect("Failed to make contiguous");
        Self::from_candle(inner)
    }

    /// Adds a bias vector to each row (broadcasting).
    pub fn add_bias(&self, bias: &Self) -> Self {
        let inner = self
            .inner
            .broadcast_add(&bias.inner)
            .expect("Failed to add bias");
        Self::from_candle(inner)
    }
}

impl Tensor for CandleTensor {
    fn shape(&self) -> &[usize] {
        &self.shape_cache
    }

    fn dtype(&self) -> DType {
        match self.inner.dtype() {
            CandleDType::F32 => DType::F32,
            CandleDType::F64 => DType::F64,
            CandleDType::I64 => DType::I64,
            _ => DType::F32, // Default to F32 for unsupported types
        }
    }

    fn zeros(shape: &[usize]) -> Self {
        Self::zeros_on(shape, &Self::best_device())
    }

    fn ones(shape: &[usize]) -> Self {
        Self::ones_on(shape, &Self::best_device())
    }

    fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        Self::from_slice_on(data, shape, &Self::best_device())
    }

    fn to_vec(&self) -> Vec<f32> {
        // Ensure tensor is on CPU for data access
        let cpu_tensor = if matches!(self.device(), Device::Cpu) {
            self.inner.clone()
        } else {
            self.inner
                .to_device(&Device::Cpu)
                .expect("Failed to move to CPU")
        };

        cpu_tensor
            .flatten_all()
            .expect("Failed to flatten")
            .to_vec1::<f32>()
            .expect("Failed to convert to vec")
    }

    fn add(&self, other: &Self) -> Self {
        let inner = self.inner.add(&other.inner).expect("Failed to add tensors");
        Self::from_candle(inner)
    }

    fn mul(&self, other: &Self) -> Self {
        let inner = self
            .inner
            .mul(&other.inner)
            .expect("Failed to multiply tensors");
        Self::from_candle(inner)
    }

    fn matmul(&self, other: &Self) -> Self {
        let inner = self.inner.matmul(&other.inner).expect("Failed to matmul");
        Self::from_candle(inner)
    }
}

impl From<CandleTensorInner> for CandleTensor {
    fn from(inner: CandleTensorInner) -> Self {
        Self::from_candle(inner)
    }
}

impl From<CandleTensor> for CandleTensorInner {
    fn from(tensor: CandleTensor) -> Self {
        tensor.into_candle()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = CandleTensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.to_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = CandleTensor::ones(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = CandleTensor::from_slice(&data, &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec(), data);
    }

    #[test]
    fn test_add() {
        let a = CandleTensor::ones(&[2, 3]);
        let b = CandleTensor::ones(&[2, 3]);
        let c = a.add(&b);
        assert!(c.to_vec().iter().all(|&x| (x - 2.0).abs() < 1e-6));
    }

    #[test]
    fn test_mul() {
        let a = CandleTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = CandleTensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
        let c = a.mul(&b);
        let result = c.to_vec();
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
        assert!((result[3] - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_2d() {
        // 2x3 matrix multiplied by 3x2 matrix
        let a = CandleTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = CandleTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = a.matmul(&b);

        assert_eq!(c.shape(), &[2, 2]);
        let result = c.to_vec();
        // Result should be:
        // [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6]   = [22, 28]
        // [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6]   = [49, 64]
        assert!((result[0] - 22.0).abs() < 1e-6);
        assert!((result[1] - 28.0).abs() < 1e-6);
        assert!((result[2] - 49.0).abs() < 1e-6);
        assert!((result[3] - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_reshape() {
        let t = CandleTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let reshaped = t.reshape(&[3, 2]);
        assert_eq!(reshaped.shape(), &[3, 2]);
    }

    #[test]
    fn test_transpose() {
        let t = CandleTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let transposed = t.transpose();
        assert_eq!(transposed.shape(), &[3, 2]);
    }

    #[test]
    fn test_relu() {
        let t = CandleTensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);
        let relu = t.relu();
        let result = relu.to_vec();
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
        assert!((result[2] - 1.0).abs() < 1e-6);
        assert!((result[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid() {
        let t = CandleTensor::from_slice(&[0.0, 0.0, 0.0, 0.0], &[4]);
        let sigmoid = t.sigmoid();
        // sigmoid(0) = 0.5
        for val in sigmoid.to_vec() {
            assert!((val - 0.5).abs() < 1e-5);
        }
    }

    #[test]
    fn test_device_detection() {
        let device = CandleTensor::best_device();
        println!("Best device: {:?}", device);

        let t = CandleTensor::zeros(&[2, 3]);
        println!("Tensor device: {:?}", t.device());
    }

    #[test]
    fn test_sum_mean() {
        let t = CandleTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!((t.sum() - 10.0).abs() < 1e-6);
        assert!((t.mean() - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_sub_div() {
        let a = CandleTensor::from_slice(&[4.0, 6.0, 8.0, 10.0], &[2, 2]);
        let b = CandleTensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);

        let sub = a.sub(&b);
        let sub_result = sub.to_vec();
        assert!((sub_result[0] - 2.0).abs() < 1e-6);
        assert!((sub_result[1] - 4.0).abs() < 1e-6);

        let div = a.div(&b);
        let div_result = div.to_vec();
        assert!((div_result[0] - 2.0).abs() < 1e-6);
        assert!((div_result[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale() {
        let t = CandleTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let scaled = t.scale(2.0);
        let result = scaled.to_vec();
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_dtype() {
        let t = CandleTensor::zeros(&[2, 2]);
        assert_eq!(t.dtype(), DType::F32);
    }
}
