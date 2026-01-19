#![allow(clippy::needless_range_loop)]
//! GPU-resident tensor type for neural network computations.
//!
//! When GPU is enabled, tensors keep their data on GPU and only sync
//! to CPU when explicitly needed (e.g., reading data for metrics).

use rayon::prelude::*;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(any(feature = "metal", feature = "cuda"))]
use monolith_tensor::{CandleTensor, Device, Tensor as TensorTrait};

static GPU_ENABLED: AtomicBool = AtomicBool::new(false);

/// Enable or disable GPU acceleration.
pub fn set_gpu_enabled(enabled: bool) {
    GPU_ENABLED.store(enabled, Ordering::Relaxed);
}

#[allow(dead_code)]
fn gpu_enabled() -> bool {
    GPU_ENABLED.load(Ordering::Relaxed)
}

/// Returns whether GPU usage is enabled.
pub fn is_gpu_enabled() -> bool {
    gpu_enabled()
}

#[cfg(any(feature = "metal", feature = "cuda"))]
fn get_device() -> Device {
    CandleTensor::best_device()
}

#[cfg(any(feature = "metal", feature = "cuda"))]
#[allow(dead_code)]
fn is_gpu_available() -> bool {
    !matches!(get_device(), Device::Cpu)
}

/// Returns whether a GPU device is available.
#[cfg(any(feature = "metal", feature = "cuda"))]
pub fn gpu_available() -> bool {
    is_gpu_available()
}

/// Returns false on non-GPU builds.
#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub fn gpu_available() -> bool {
    false
}

#[cfg(not(any(feature = "metal", feature = "cuda")))]
#[allow(dead_code)]
fn is_gpu_available() -> bool {
    false
}

/// Internal GPU data storage
#[cfg(any(feature = "metal", feature = "cuda"))]
#[derive(Debug)]
struct GpuData {
    tensor: CandleTensor,
}

/// CPU data with sync flag
#[derive(Debug, Clone)]
struct CpuData {
    data: Vec<f32>,
    current: bool,
}

/// A GPU-resident tensor for neural network computations.
///
/// When GPU is enabled:
/// - Data lives on GPU
/// - Operations chain on GPU without CPU transfers
/// - CPU data is only populated when `data()` is called
#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    /// CPU data - protected by RwLock for interior mutability (lazy sync from GPU)
    cpu_data: Arc<RwLock<CpuData>>,
    /// GPU data storage (not serialized)
    #[cfg(any(feature = "metal", feature = "cuda"))]
    gpu_data: Option<Arc<RwLock<GpuData>>>,
}

#[derive(Serialize, Deserialize)]
struct TensorSerde {
    shape: Vec<usize>,
    cpu_data: Vec<f32>,
}

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.sync_cpu_from_gpu();
        let guard = self.cpu_data.read().unwrap();
        let helper = TensorSerde {
            shape: self.shape.clone(),
            cpu_data: guard.data.clone(),
        };
        helper.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helper = TensorSerde::deserialize(deserializer)?;
        Ok(Self {
            shape: helper.shape,
            cpu_data: Arc::new(RwLock::new(CpuData {
                data: helper.cpu_data,
                current: true,
            })),
            #[cfg(any(feature = "metal", feature = "cuda"))]
            gpu_data: None,
        })
    }
}

/// Read guard for CPU tensor data.
pub struct TensorData<'a> {
    guard: std::sync::RwLockReadGuard<'a, CpuData>,
}

impl<'a> std::ops::Deref for TensorData<'a> {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.guard.data
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data() == other.data()
    }
}

impl Tensor {
    /// Creates a new tensor with the given shape, filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let device = get_device();
            let gpu_tensor = CandleTensor::zeros_on(shape, &device);
            return Self {
                shape: shape.to_vec(),
                cpu_data: Arc::new(RwLock::new(CpuData {
                    data: vec![0.0; numel],
                    current: true, // zeros is same on both
                })),
                gpu_data: Some(Arc::new(RwLock::new(GpuData { tensor: gpu_tensor }))),
            };
        }

        Self {
            shape: shape.to_vec(),
            cpu_data: Arc::new(RwLock::new(CpuData {
                data: vec![0.0; numel],
                current: true,
            })),
            #[cfg(any(feature = "metal", feature = "cuda"))]
            gpu_data: None,
        }
    }

    /// Creates a new tensor with the given shape, filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let device = get_device();
            let gpu_tensor = CandleTensor::ones_on(shape, &device);
            return Self {
                shape: shape.to_vec(),
                cpu_data: Arc::new(RwLock::new(CpuData {
                    data: vec![1.0; numel],
                    current: true,
                })),
                gpu_data: Some(Arc::new(RwLock::new(GpuData { tensor: gpu_tensor }))),
            };
        }

        Self {
            shape: shape.to_vec(),
            cpu_data: Arc::new(RwLock::new(CpuData {
                data: vec![1.0; numel],
                current: true,
            })),
            #[cfg(any(feature = "metal", feature = "cuda"))]
            gpu_data: None,
        }
    }

    /// Creates a new tensor with the given shape and data.
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

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let device = get_device();
            let gpu_tensor = CandleTensor::from_slice_on(&data, shape, &device);
            return Self {
                shape: shape.to_vec(),
                cpu_data: Arc::new(RwLock::new(CpuData {
                    data,
                    current: true,
                })),
                gpu_data: Some(Arc::new(RwLock::new(GpuData { tensor: gpu_tensor }))),
            };
        }

        Self {
            shape: shape.to_vec(),
            cpu_data: Arc::new(RwLock::new(CpuData {
                data,
                current: true,
            })),
            #[cfg(any(feature = "metal", feature = "cuda"))]
            gpu_data: None,
        }
    }

    /// Creates a tensor directly from GPU data (internal use).
    /// CPU data is NOT current - will be synced lazily when needed.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn from_gpu_tensor(gpu_tensor: CandleTensor) -> Self {
        let shape = gpu_tensor.shape().to_vec();
        let numel: usize = shape.iter().product();
        Self {
            shape,
            cpu_data: Arc::new(RwLock::new(CpuData {
                data: vec![0.0; numel], // Placeholder, will be filled on demand
                current: false,          // CPU data is NOT current
            })),
            gpu_data: Some(Arc::new(RwLock::new(GpuData { tensor: gpu_tensor }))),
        }
    }

    /// Returns whether CPU data is current (for testing)
    #[cfg(any(feature = "metal", feature = "cuda"))]
    pub fn is_cpu_current(&self) -> bool {
        self.cpu_data.read().unwrap().current
    }

    /// Creates a tensor with random values from a uniform distribution [0, 1).
    pub fn rand(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        let mut seed: u64 = 42;
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                ((seed >> 16) & 0x7fff) as f32 / 32768.0
            })
            .collect();
        Self::from_data(shape, data)
    }

    /// Creates a tensor with random values from a normal distribution.
    pub fn randn(shape: &[usize], mean: f32, std: f32) -> Self {
        let numel: usize = shape.iter().product();
        let mut seed: u64 = 42;
        let data: Vec<f32> = (0..numel)
            .map(|_| {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let u1 = ((seed >> 16) & 0x7fff) as f32 / 32768.0 + 1e-10;
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                let u2 = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                z * std + mean
            })
            .collect();
        Self::from_data(shape, data)
    }

    /// Ensures CPU data is current by syncing from GPU if needed.
    /// Uses interior mutability via RwLock.
    fn sync_cpu_from_gpu(&self) {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            let needs_sync = !self.cpu_data.read().unwrap().current;
            if needs_sync {
                if let Some(ref gpu_data) = self.gpu_data {
                    let gpu_guard = gpu_data.read().unwrap();
                    let new_data = gpu_guard.tensor.to_vec();
                    let mut cpu_guard = self.cpu_data.write().unwrap();
                    cpu_guard.data = new_data;
                    cpu_guard.current = true;
                }
            }
        }
    }

    /// Get GPU tensor, creating it from CPU data if needed
    #[cfg(any(feature = "metal", feature = "cuda"))]
    fn ensure_gpu_tensor(&self) -> CandleTensor {
        if let Some(ref gpu_data) = self.gpu_data {
            let guard = gpu_data.read().unwrap();
            guard.tensor.clone()
        } else {
            // Create GPU tensor from CPU data
            let device = get_device();
            let cpu_guard = self.cpu_data.read().unwrap();
            CandleTensor::from_slice_on(&cpu_guard.data, &self.shape, &device)
        }
    }

    /// Returns a Candle tensor on the best available device.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    pub fn to_candle(&self) -> CandleTensor {
        self.ensure_gpu_tensor()
    }

    /// Wraps a Candle tensor in a Tensor.
    #[cfg(any(feature = "metal", feature = "cuda"))]
    pub fn from_candle(gpu_tensor: CandleTensor) -> Self {
        Self::from_gpu_tensor(gpu_tensor)
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
        self.shape.iter().product()
    }

    /// Returns a reference to the underlying data (syncs from GPU if needed).
    /// Note: Returns Vec<f32> because we can't return a reference through RwLock.
    pub fn data(&self) -> Vec<f32> {
        self.data_ref().to_vec()
    }

    /// Returns a read guard for underlying data (syncs from GPU if needed).
    pub fn data_ref(&self) -> TensorData<'_> {
        self.sync_cpu_from_gpu();
        TensorData {
            guard: self.cpu_data.read().unwrap(),
        }
    }

    /// Returns a mutable copy of the underlying data as Vec.
    /// Caller must use set_data() to write back if modifications were made.
    /// This invalidates GPU data.
    pub fn data_mut(&mut self) -> Vec<f32> {
        // Ensure CPU data is current first
        self.sync_cpu_from_gpu();
        // Invalidate GPU data since we're modifying CPU
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            self.gpu_data = None;
        }
        // Return a copy of the data
        self.cpu_data.read().unwrap().data.clone()
    }

    /// Sets the CPU data directly. Used after modifying data from data_mut().
    pub fn set_data(&mut self, data: Vec<f32>) {
        assert_eq!(data.len(), self.numel(), "Data length mismatch");
        let mut guard = self.cpu_data.write().unwrap();
        guard.data = data;
        guard.current = true;
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            self.gpu_data = None;
        }
    }

    /// In-place modification helper - modifies data via closure.
    pub fn modify_data<F>(&mut self, f: F)
    where
        F: FnOnce(&mut [f32]),
    {
        self.sync_cpu_from_gpu();
        #[cfg(any(feature = "metal", feature = "cuda"))]
        {
            self.gpu_data = None;
        }
        let mut guard = self.cpu_data.write().unwrap();
        f(&mut guard.data);
        guard.current = true;
    }

    /// Matrix multiplication (supports batched matmul) - stays on GPU when possible
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(self.ndim() >= 2, "matmul requires at least 2D tensors");
        if self.ndim() != other.ndim() {
            if self.ndim() == 2 && other.ndim() > 2 {
                let mut shape = vec![1; other.ndim() - 2];
                shape.extend_from_slice(self.shape());
                let mut target = other.shape[..other.ndim() - 2].to_vec();
                target.push(self.shape[0]);
                target.push(self.shape[1]);
                let a_b = self.reshape(&shape).broadcast_as(&target);
                return a_b.matmul(other);
            } else if other.ndim() == 2 && self.ndim() > 2 {
                let mut shape = vec![1; self.ndim() - 2];
                shape.extend_from_slice(other.shape());
                let mut target = self.shape[..self.ndim() - 2].to_vec();
                target.push(other.shape[0]);
                target.push(other.shape[1]);
                let b_b = other.reshape(&shape).broadcast_as(&target);
                return self.matmul(&b_b);
            } else {
                panic!("matmul requires same rank or one operand to be 2D");
            }
        }

        let dim = self.ndim();
        let m = self.shape[dim - 2];
        let k = self.shape[dim - 1];
        let k2 = other.shape[dim - 2];
        let n = other.shape[dim - 1];
        assert_eq!(k, k2, "Inner dimensions must match for matmul");

        let mut batch_shape = Vec::with_capacity(dim - 2);
        for i in 0..dim - 2 {
            let a = self.shape[i];
            let b = other.shape[i];
            let out = if a == b {
                a
            } else if a == 1 {
                b
            } else if b == 1 {
                a
            } else {
                panic!("Batch dimensions must be broadcastable for matmul");
            };
            batch_shape.push(out);
        }

        let mut a_shape = self.shape.clone();
        a_shape[dim - 2] = m;
        a_shape[dim - 1] = k;
        let mut b_shape = other.shape.clone();
        b_shape[dim - 2] = k;
        b_shape[dim - 1] = n;

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let mut target_a = batch_shape.clone();
            target_a.push(m);
            target_a.push(k);
            let mut target_b = batch_shape.clone();
            target_b.push(k);
            target_b.push(n);
            let a = self.ensure_gpu_tensor();
            let b = other.ensure_gpu_tensor();
            let a = if a.shape() == target_a.as_slice() {
                a
            } else {
                a.broadcast_as(&target_a)
            };
            let b = if b.shape() == target_b.as_slice() {
                b
            } else {
                b.broadcast_as(&target_b)
            };
            let c = a.matmul(&b);
            return Self::from_gpu_tensor(c);
        }

        // CPU fallback (supports batched matmul with broadcasting)
        let batch: usize = batch_shape.iter().product();
        let c_batch_stride = m * n;
        let a_data = self.data();
        let b_data = other.data();
        let mut result = vec![0.0; batch * c_batch_stride];

        if batch == 1 {
            let work = m.saturating_mul(n).saturating_mul(k);
            if work >= 32_768 && m > 1 {
                result
                    .par_chunks_mut(n)
                    .enumerate()
                    .for_each(|(i, row)| {
                        let row_base = i * k;
                        for j in 0..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a_data[row_base + l] * b_data[l * n + j];
                            }
                            row[j] = sum;
                        }
                    });
            } else {
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }
            }
        } else {
            let mut a_strides = vec![0; dim];
            let mut b_strides = vec![0; dim];
            a_strides[dim - 1] = 1;
            b_strides[dim - 1] = 1;
            for i in (0..dim - 1).rev() {
                if i + 1 < dim {
                    a_strides[i] = a_strides[i + 1] * self.shape[i + 1];
                    b_strides[i] = b_strides[i + 1] * other.shape[i + 1];
                }
            }

            let mut batch_strides = vec![0; dim - 2];
            if !batch_shape.is_empty() {
                batch_strides[dim - 3] = 1;
                for i in (0..dim - 3).rev() {
                    batch_strides[i] = batch_strides[i + 1] * batch_shape[i + 1];
                }
            }

            for batch_idx in 0..batch {
                let mut rem = batch_idx;
                let mut a_base = 0;
                let mut b_base = 0;
                for i in 0..dim - 2 {
                    let stride = if batch_strides.is_empty() { 1 } else { batch_strides[i] };
                    let idx = if stride == 0 { 0 } else { rem / stride };
                    if stride != 0 {
                        rem %= stride;
                    }
                    let a_idx = if self.shape[i] == 1 { 0 } else { idx };
                    let b_idx = if other.shape[i] == 1 { 0 } else { idx };
                    a_base += a_idx * a_strides[i];
                    b_base += b_idx * b_strides[i];
                }

                let c_base = batch_idx * c_batch_stride;
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_data[a_base + i * k + l] * b_data[b_base + l * n + j];
                        }
                        result[c_base + i * n + j] = sum;
                    }
                }
            }
        }

        let mut out_shape = batch_shape;
        out_shape.push(m);
        out_shape.push(n);
        Tensor::from_data(&out_shape, result)
    }

    /// Transposes a 2D tensor - stays on GPU when possible
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose requires 2D tensor");
        let m = self.shape[0];
        let n = self.shape[1];

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let b = a.transpose();
            return Self::from_gpu_tensor(b);
        }

        let data = self.data_ref();
        let mut result = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                result[j * m + i] = data[i * n + j];
            }
        }

        Tensor::from_data(&[n, m], result)
    }

    /// Transposes two dimensions of the tensor.
    pub fn transpose_dims(&self, dim1: usize, dim2: usize) -> Tensor {
        assert!(dim1 < self.ndim(), "transpose dim1 out of bounds");
        assert!(dim2 < self.ndim(), "transpose dim2 out of bounds");
        if dim1 == dim2 {
            return self.clone();
        }

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let b = a.transpose_dims(dim1, dim2);
            return Self::from_gpu_tensor(b);
        }

        let mut perm: Vec<usize> = (0..self.ndim()).collect();
        perm.swap(dim1, dim2);
        self.permute(&perm)
    }

    /// Permutes tensor dimensions.
    pub fn permute(&self, dims: &[usize]) -> Tensor {
        assert_eq!(dims.len(), self.ndim(), "permute dims rank mismatch");
        let mut seen = vec![false; dims.len()];
        for &d in dims {
            assert!(d < dims.len(), "permute dim out of bounds");
            assert!(!seen[d], "permute dims must be unique");
            seen[d] = true;
        }

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let b = a.permute(dims);
            return Self::from_gpu_tensor(b);
        }

        let old_shape = &self.shape;
        let mut new_shape = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            new_shape[i] = old_shape[d];
        }

        let data = self.data_ref();
        let total = self.numel();
        let mut old_strides = vec![0; old_shape.len()];
        if !old_shape.is_empty() {
            old_strides[old_shape.len() - 1] = 1;
            for i in (0..old_shape.len().saturating_sub(1)).rev() {
                old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
            }
        }

        let mut new_strides = vec![0; new_shape.len()];
        if !new_shape.is_empty() {
            new_strides[new_shape.len() - 1] = 1;
            for i in (0..new_shape.len().saturating_sub(1)).rev() {
                new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
            }
        }

        let mut out = vec![0.0; total];
        for idx in 0..total {
            let mut rem = idx;
            let mut new_index = vec![0; new_shape.len()];
            for i in 0..new_shape.len() {
                let stride = new_strides[i];
                new_index[i] = if stride == 0 { 0 } else { rem / stride };
                if stride != 0 {
                    rem %= stride;
                }
            }

            let mut old_offset = 0;
            for (i, &d) in dims.iter().enumerate() {
                old_offset += new_index[i] * old_strides[d];
            }
            out[idx] = data[old_offset];
        }

        Tensor::from_data(&new_shape, out)
    }

    /// Element-wise addition - stays on GPU when possible
    pub fn add(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let c = a.add(&b);
                return Self::from_gpu_tensor(c);
            }

            let a_data = self.data_ref();
            let b_data = other.data_ref();
            let data: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a + b).collect();
            Tensor::from_data(&self.shape, data)
        } else if other.numel() == 1 {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let b_broadcast = b.broadcast_as(&self.shape);
                let c = a.add(&b_broadcast);
                return Self::from_gpu_tensor(c);
            }

            let scalar = other.data_ref()[0];
            let data: Vec<f32> = self.data_ref().iter().map(|a| a + scalar).collect();
            Tensor::from_data(&self.shape, data)
        } else if self.ndim() == 2 && other.ndim() == 1 && self.shape[1] == other.shape[0] {
            // Broadcast along rows (bias addition)
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let c = a.add_bias(&b);
                return Self::from_gpu_tensor(c);
            }

            let mut data = self.data();
            let bias = other.data_ref();
            let n = self.shape[1];
            for i in 0..self.shape[0] {
                for j in 0..n {
                    data[i * n + j] += bias[j];
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

    /// Element-wise multiplication - stays on GPU when possible
    pub fn mul(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let c = a.mul(&b);
                return Self::from_gpu_tensor(c);
            }

            let a_data = self.data_ref();
            let b_data = other.data_ref();
            let data: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).collect();
            Tensor::from_data(&self.shape, data)
        } else if other.numel() == 1 {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let b_broadcast = b.broadcast_as(&self.shape);
                let c = a.mul(&b_broadcast);
                return Self::from_gpu_tensor(c);
            }

            let scalar = other.data_ref()[0];
            let data: Vec<f32> = self.data_ref().iter().map(|a| a * scalar).collect();
            Tensor::from_data(&self.shape, data)
        } else {
            panic!(
                "Cannot multiply shapes {:?} and {:?}",
                self.shape, other.shape
            );
        }
    }

    /// Element-wise subtraction - stays on GPU when possible
    pub fn sub(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let c = a.sub(&b);
                return Self::from_gpu_tensor(c);
            }

            let a_data = self.data_ref();
            let b_data = other.data_ref();
            let data: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a - b).collect();
            Tensor::from_data(&self.shape, data)
        } else if other.numel() == 1 {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let b_broadcast = b.broadcast_as(&self.shape);
                let c = a.sub(&b_broadcast);
                return Self::from_gpu_tensor(c);
            }

            let scalar = other.data_ref()[0];
            let data: Vec<f32> = self.data_ref().iter().map(|a| a - scalar).collect();
            Tensor::from_data(&self.shape, data)
        } else {
            panic!(
                "Cannot subtract shapes {:?} and {:?}",
                self.shape, other.shape
            );
        }
    }

    /// Element-wise division - stays on GPU when possible
    pub fn div(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let c = a.div(&b);
                return Self::from_gpu_tensor(c);
            }

            let a_data = self.data_ref();
            let b_data = other.data_ref();
            let data: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(a, b)| a / b).collect();
            Tensor::from_data(&self.shape, data)
        } else if other.numel() == 1 {
            #[cfg(any(feature = "metal", feature = "cuda"))]
            if gpu_enabled() && is_gpu_available() {
                let a = self.ensure_gpu_tensor();
                let b = other.ensure_gpu_tensor();
                let b_broadcast = b.broadcast_as(&self.shape);
                let c = a.div(&b_broadcast);
                return Self::from_gpu_tensor(c);
            }

            let scalar = other.data_ref()[0];
            let data: Vec<f32> = self.data_ref().iter().map(|a| a / scalar).collect();
            Tensor::from_data(&self.shape, data)
        } else {
            panic!(
                "Cannot divide shapes {:?} and {:?}",
                self.shape, other.shape
            );
        }
    }

    /// Element-wise maximum (broadcasting supported).
    pub fn maximum(&self, other: &Tensor) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let b = other.ensure_gpu_tensor();
            let c = a.maximum(&b);
            return Self::from_gpu_tensor(c);
        }

        let (a, b) = if self.shape == other.shape {
            (self.clone(), other.clone())
        } else if self.ndim() >= other.ndim() {
            let b_broadcast = other.broadcast_as(&self.shape);
            (self.clone(), b_broadcast)
        } else {
            let a_broadcast = self.broadcast_as(&other.shape);
            (a_broadcast, other.clone())
        };

        let data_a = a.data_ref();
        let data_b = b.data_ref();
        let mut out = vec![0.0; data_a.len()];
        for i in 0..data_a.len() {
            out[i] = if data_a[i] > data_b[i] {
                data_a[i]
            } else {
                data_b[i]
            };
        }
        Tensor::from_data(&a.shape, out)
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.scale(scalar);
            return Self::from_gpu_tensor(c);
        }

        let data: Vec<f32> = self.data_ref().iter().map(|a| a * scalar).collect();
        Tensor::from_data(&self.shape, data)
    }

    /// Element-wise exponential.
    pub fn exp(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.exp();
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| v.exp())
    }

    /// Element-wise natural logarithm.
    pub fn log(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.log();
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| v.ln())
    }

    /// Element-wise clamp to [min, max].
    pub fn clamp(&self, min: f32, max: f32) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.clamp(min, max);
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| v.clamp(min, max))
    }

    /// Element-wise absolute value.
    pub fn abs(&self) -> Tensor {
        self.sqr().sqrt()
    }

    /// Element-wise reciprocal (1 / x).
    pub fn recip(&self) -> Tensor {
        let ones = Tensor::ones(&self.shape);
        ones.div(self)
    }

    /// Element-wise greater-than comparison with a scalar.
    pub fn gt_scalar(&self, value: f32) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.gt_scalar(value);
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| if v > value { 1.0 } else { 0.0 })
    }

    /// Element-wise greater-than-or-equal comparison with a scalar.
    pub fn ge_scalar(&self, value: f32) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.ge_scalar(value);
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| if v >= value { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than comparison with a scalar.
    pub fn lt_scalar(&self, value: f32) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.lt_scalar(value);
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| if v < value { 1.0 } else { 0.0 })
    }

    /// Element-wise less-than-or-equal comparison with a scalar.
    pub fn le_scalar(&self, value: f32) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.le_scalar(value);
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| if v <= value { 1.0 } else { 0.0 })
    }

    /// Element-wise square.
    pub fn sqr(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.sqr();
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| v * v)
    }

    /// Element-wise square root.
    pub fn sqrt(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let a = self.ensure_gpu_tensor();
            let c = a.sqrt();
            return Self::from_gpu_tensor(c);
        }

        self.map(|v| v.sqrt())
    }

    /// Broadcasts this tensor to a target shape.
    pub fn broadcast_as(&self, shape: &[usize]) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let b = gpu.broadcast_as(shape);
            return Self::from_gpu_tensor(b);
        }

        if self.shape == shape {
            return self.clone();
        }

        if self.ndim() == 2 && shape.len() == 2 {
            let (m, n) = (self.shape[0], self.shape[1]);
            let (out_m, out_n) = (shape[0], shape[1]);
            let data = self.data_ref();
            let mut out = vec![0.0; out_m * out_n];
            for i in 0..out_m {
                for j in 0..out_n {
                    let src_i = if m == 1 { 0 } else { i };
                    let src_j = if n == 1 { 0 } else { j };
                    out[i * out_n + j] = data[src_i * n + src_j];
                }
            }
            return Tensor::from_data(shape, out);
        }

        if self.ndim() == 3 && shape.len() == 3 {
            let (a, b, c) = (self.shape[0], self.shape[1], self.shape[2]);
            let (out_a, out_b, out_c) = (shape[0], shape[1], shape[2]);
            let data = self.data_ref();
            let mut out = vec![0.0; out_a * out_b * out_c];
            for i in 0..out_a {
                let src_i = if a == 1 { 0 } else { i };
                for j in 0..out_b {
                    let src_j = if b == 1 { 0 } else { j };
                    for k in 0..out_c {
                        let src_k = if c == 1 { 0 } else { k };
                        let src_idx = src_i * b * c + src_j * c + src_k;
                        let dst_idx = i * out_b * out_c + j * out_c + k;
                        out[dst_idx] = data[src_idx];
                    }
                }
            }
            return Tensor::from_data(shape, out);
        }

        if self.ndim() == 2 && shape.len() == 3 {
            let (b, d) = (self.shape[0], self.shape[1]);
            if shape[1] == b && shape[2] == d {
                let mut out = vec![0.0; shape[0] * b * d];
                let data = self.data_ref();
                for bi in 0..shape[0] {
                    let dst_base = bi * b * d;
                    out[dst_base..dst_base + b * d].copy_from_slice(&data);
                }
                return Tensor::from_data(shape, out);
            }
        }

        if self.ndim() == 1 && shape.len() == 3 {
            if self.shape[0] == shape[2] {
                let mut out = vec![0.0; shape[0] * shape[1] * shape[2]];
                let data = self.data_ref();
                for bi in 0..shape[0] {
                    for si in 0..shape[1] {
                        let dst_base = bi * shape[1] * shape[2] + si * shape[2];
                        out[dst_base..dst_base + shape[2]].copy_from_slice(&data);
                    }
                }
                return Tensor::from_data(shape, out);
            }
        }

        if self.ndim() == 1 && shape.len() == 2 {
            let (out_m, out_n) = (shape[0], shape[1]);
            let data = self.data_ref();
            if self.shape[0] == out_n {
                let mut out = vec![0.0; out_m * out_n];
                for i in 0..out_m {
                    let dst = i * out_n;
                    out[dst..dst + out_n].copy_from_slice(&data);
                }
                return Tensor::from_data(shape, out);
            } else if self.shape[0] == out_m {
                let mut out = vec![0.0; out_m * out_n];
                for i in 0..out_m {
                    for j in 0..out_n {
                        out[i * out_n + j] = data[i];
                    }
                }
                return Tensor::from_data(shape, out);
            }
        }

        if self.ndim() <= shape.len() {
            let mut in_shape = vec![1; shape.len() - self.ndim()];
            in_shape.extend_from_slice(&self.shape);
            for i in 0..shape.len() {
                if in_shape[i] != shape[i] && in_shape[i] != 1 {
                    panic!(
                        "broadcast_as not compatible for shapes {:?} -> {:?}",
                        self.shape, shape
                    );
                }
            }

            let data = self.data_ref();
            let total: usize = shape.iter().product();
            let mut in_strides = vec![0; shape.len()];
            if !shape.is_empty() {
                in_strides[shape.len() - 1] = 1;
                for i in (0..shape.len().saturating_sub(1)).rev() {
                    in_strides[i] = in_strides[i + 1] * in_shape[i + 1];
                }
            }
            let mut out_strides = vec![0; shape.len()];
            if !shape.is_empty() {
                out_strides[shape.len() - 1] = 1;
                for i in (0..shape.len().saturating_sub(1)).rev() {
                    out_strides[i] = out_strides[i + 1] * shape[i + 1];
                }
            }
            let mut out = vec![0.0; total];
            for idx in 0..total {
                let mut rem = idx;
                let mut in_offset = 0;
                for i in 0..shape.len() {
                    let stride = out_strides[i];
                    let out_idx = if stride == 0 { 0 } else { rem / stride };
                    if stride != 0 {
                        rem %= stride;
                    }
                    let in_idx = if in_shape[i] == 1 { 0 } else { out_idx };
                    in_offset += in_idx * in_strides[i];
                }
                out[idx] = data[in_offset];
            }
            return Tensor::from_data(shape, out);
        }

        panic!("broadcast_as not implemented for shapes {:?} -> {:?}", self.shape, shape);
    }

    /// Sigmoid activation.
    pub fn sigmoid(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let s = gpu.sigmoid();
            return Self::from_gpu_tensor(s);
        }

        self.map(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Tanh activation.
    pub fn tanh(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let t = gpu.tanh();
            return Self::from_gpu_tensor(t);
        }

        self.map(|v| v.tanh())
    }

    /// GELU activation (erf-based).
    pub fn gelu(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let g = gpu.gelu_erf();
            return Self::from_gpu_tensor(g);
        }

        // Erf-based GELU on CPU to match TensorFlow default
        let inv_sqrt2 = 0.707_106_77_f32; // 1 / sqrt(2)
        self.map(|x| 0.5 * x * (1.0 + libm::erff(x * inv_sqrt2)))
    }

    /// ReLU activation.
    pub fn relu(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let r = gpu.relu();
            return Self::from_gpu_tensor(r);
        }

        self.map(|v| v.max(0.0))
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() && self.gpu_data.is_some() {
            let gpu = self.ensure_gpu_tensor();
            return gpu.sum();
        }

        self.data_ref().iter().sum()
    }

    /// Sum along an axis
    pub fn sum_axis(&self, axis: usize) -> Tensor {
        assert!(axis < self.ndim(), "Axis out of bounds");

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let summed = gpu.sum_axis(axis);
            return Self::from_gpu_tensor(summed);
        }

        let data = self.data_ref();

        if self.ndim() == 2 {
            if axis == 0 {
                let n = self.shape[1];
                let mut result = vec![0.0; n];
                for i in 0..self.shape[0] {
                    for j in 0..n {
                        result[j] += data[i * n + j];
                    }
                }
                Tensor::from_data(&[n], result)
            } else {
                let n = self.shape[1];
                let result: Vec<f32> = (0..self.shape[0])
                    .map(|i| (0..n).map(|j| data[i * n + j]).sum())
                    .collect();
                Tensor::from_data(&[self.shape[0]], result)
            }
        } else {
            if self.ndim() == 3 {
                let (b, s, d) = (self.shape[0], self.shape[1], self.shape[2]);
                if axis == 2 {
                    let mut result = vec![0.0; b * s];
                    for bi in 0..b {
                        for si in 0..s {
                            let mut sum = 0.0;
                            let base = bi * s * d + si * d;
                            for di in 0..d {
                                sum += data[base + di];
                            }
                            result[bi * s + si] = sum;
                        }
                    }
                    Tensor::from_data(&[b, s], result)
                } else if axis == 1 {
                    let mut result = vec![0.0; b * d];
                    for bi in 0..b {
                        for di in 0..d {
                            let mut sum = 0.0;
                            for si in 0..s {
                                sum += data[bi * s * d + si * d + di];
                            }
                            result[bi * d + di] = sum;
                        }
                    }
                    Tensor::from_data(&[b, d], result)
                } else if axis == 0 {
                    let mut result = vec![0.0; s * d];
                    for si in 0..s {
                        for di in 0..d {
                            let mut sum = 0.0;
                            for bi in 0..b {
                                sum += data[bi * s * d + si * d + di];
                            }
                            result[si * d + di] = sum;
                        }
                    }
                    Tensor::from_data(&[s, d], result)
                } else {
                    panic!("sum_axis axis {} not implemented for 3D", axis);
                }
            } else {
                panic!("sum_axis not implemented for ndim {}", self.ndim());
            }
        }
    }

    /// Mean along an axis
    pub fn mean_axis(&self, axis: usize) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let mean = gpu.mean_axis(axis);
            return Self::from_gpu_tensor(mean);
        }

        let sum = self.sum_axis(axis);
        let count = self.shape[axis] as f32;
        sum.scale(1.0 / count)
    }

    /// Sum along multiple axes (axes will be removed).
    pub fn sum_axes(&self, axes: &[usize]) -> Tensor {
        if axes.is_empty() {
            return self.clone();
        }

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let mut axes_sorted = axes.to_vec();
            axes_sorted.sort_by(|a, b| b.cmp(a));
            let mut tensor = self.ensure_gpu_tensor();
            for axis in axes_sorted {
                tensor = tensor.sum_axis(axis);
            }
            return Self::from_gpu_tensor(tensor);
        }

        let mut axes_sorted = axes.to_vec();
        axes_sorted.sort();
        let mut keep = vec![true; self.ndim()];
        for &axis in &axes_sorted {
            keep[axis] = false;
        }
        let mut out_shape = Vec::new();
        for (i, &dim) in self.shape.iter().enumerate() {
            if keep[i] {
                out_shape.push(dim);
            }
        }
        if out_shape.is_empty() {
            let s = self.data_ref().iter().sum();
            return Tensor::from_data(&[1], vec![s]);
        }

        let total_out: usize = out_shape.iter().product();
        let mut out = vec![0.0f32; total_out];

        let mut in_strides = vec![0; self.ndim()];
        in_strides[self.ndim() - 1] = 1;
        for i in (0..self.ndim() - 1).rev() {
            in_strides[i] = in_strides[i + 1] * self.shape[i + 1];
        }

        let mut out_strides = vec![0; out_shape.len()];
        if !out_shape.is_empty() {
            out_strides[out_shape.len() - 1] = 1;
            for i in (0..out_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }

        let data = self.data_ref();
        for idx in 0..self.numel() {
            let mut rem = idx;
            let mut out_idx = 0;
            let mut out_axis = 0;
            for axis in 0..self.ndim() {
                let stride = in_strides[axis];
                let coord = rem / stride;
                rem %= stride;
                if keep[axis] {
                    out_idx += coord * out_strides[out_axis];
                    out_axis += 1;
                }
            }
            out[out_idx] += data[idx];
        }

        Tensor::from_data(&out_shape, out)
    }

    /// Softmax along an axis.
    pub fn softmax(&self, axis: usize) -> Tensor {
        assert!(axis < self.ndim(), "softmax axis out of bounds");

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let s = gpu.softmax(axis);
            return Self::from_gpu_tensor(s);
        }

        let data = self.data_ref();
        if self.ndim() == 1 {
            let mut out = vec![0.0; self.shape[0]];
            let mut max_val = f32::NEG_INFINITY;
            for &v in data.iter() {
                if v > max_val {
                    max_val = v;
                }
            }
            let mut sum = 0.0;
            for (i, &v) in data.iter().enumerate() {
                let e = (v - max_val).exp();
                out[i] = e;
                sum += e;
            }
            if sum > 0.0 {
                for v in out.iter_mut() {
                    *v /= sum;
                }
            }
            return Tensor::from_data(&[self.shape[0]], out);
        }

        if self.ndim() == 2 {
            let m = self.shape[0];
            let n = self.shape[1];
            let mut out = vec![0.0; m * n];
            if axis == 1 {
                for i in 0..m {
                    let row = &data[i * n..(i + 1) * n];
                    let mut max_val = f32::NEG_INFINITY;
                    for &v in row.iter() {
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    let mut sum = 0.0;
                    for j in 0..n {
                        let e = (row[j] - max_val).exp();
                        out[i * n + j] = e;
                        sum += e;
                    }
                    if sum > 0.0 {
                        for j in 0..n {
                            out[i * n + j] /= sum;
                        }
                    }
                }
            } else if axis == 0 {
                for j in 0..n {
                    let mut max_val = f32::NEG_INFINITY;
                    for i in 0..m {
                        let v = data[i * n + j];
                        if v > max_val {
                            max_val = v;
                        }
                    }
                    let mut sum = 0.0;
                    for i in 0..m {
                        let e = (data[i * n + j] - max_val).exp();
                        out[i * n + j] = e;
                        sum += e;
                    }
                    if sum > 0.0 {
                        for i in 0..m {
                            out[i * n + j] /= sum;
                        }
                    }
                }
            } else {
                panic!("softmax axis {} not implemented for 2D", axis);
            }
            return Tensor::from_data(&[m, n], out);
        }

        if self.ndim() == 3 {
            let (b, s, h) = (self.shape[0], self.shape[1], self.shape[2]);
            let mut out = vec![0.0; b * s * h];
            if axis == 2 {
                for bi in 0..b {
                    for si in 0..s {
                        let base = bi * s * h + si * h;
                        let mut max_val = f32::NEG_INFINITY;
                        for d in 0..h {
                            let v = data[base + d];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut sum = 0.0;
                        for d in 0..h {
                            let e = (data[base + d] - max_val).exp();
                            out[base + d] = e;
                            sum += e;
                        }
                        if sum > 0.0 {
                            for d in 0..h {
                                out[base + d] /= sum;
                            }
                        }
                    }
                }
            } else if axis == 1 {
                for bi in 0..b {
                    for d in 0..h {
                        let mut max_val = f32::NEG_INFINITY;
                        for si in 0..s {
                            let v = data[bi * s * h + si * h + d];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut sum = 0.0;
                        for si in 0..s {
                            let idx = bi * s * h + si * h + d;
                            let e = (data[idx] - max_val).exp();
                            out[idx] = e;
                            sum += e;
                        }
                        if sum > 0.0 {
                            for si in 0..s {
                                let idx = bi * s * h + si * h + d;
                                out[idx] /= sum;
                            }
                        }
                    }
                }
            } else if axis == 0 {
                for si in 0..s {
                    for d in 0..h {
                        let mut max_val = f32::NEG_INFINITY;
                        for bi in 0..b {
                            let v = data[bi * s * h + si * h + d];
                            if v > max_val {
                                max_val = v;
                            }
                        }
                        let mut sum = 0.0;
                        for bi in 0..b {
                            let idx = bi * s * h + si * h + d;
                            let e = (data[idx] - max_val).exp();
                            out[idx] = e;
                            sum += e;
                        }
                        if sum > 0.0 {
                            for bi in 0..b {
                                let idx = bi * s * h + si * h + d;
                                out[idx] /= sum;
                            }
                        }
                    }
                }
            } else {
                panic!("softmax axis {} not implemented for 3D", axis);
            }
            return Tensor::from_data(&[b, s, h], out);
        }

        panic!("softmax not implemented for ndim {}", self.ndim());
    }

    /// Max along an axis.
    pub fn max_axis(&self, axis: usize) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let m = gpu.max_axis(axis);
            return Self::from_gpu_tensor(m);
        }

        if self.ndim() == 1 {
            let mut max_val = f32::NEG_INFINITY;
            for &v in self.data_ref().iter() {
                if v > max_val {
                    max_val = v;
                }
            }
            return Tensor::from_data(&[1], vec![max_val]);
        }

        let mut out_shape = Vec::new();
        for (i, &dim) in self.shape.iter().enumerate() {
            if i != axis {
                out_shape.push(dim);
            }
        }
        if out_shape.is_empty() {
            let mut max_val = f32::NEG_INFINITY;
            for &v in self.data_ref().iter() {
                if v > max_val {
                    max_val = v;
                }
            }
            return Tensor::from_data(&[1], vec![max_val]);
        }

        let total_out: usize = out_shape.iter().product();
        let mut out = vec![f32::NEG_INFINITY; total_out];

        let mut in_strides = vec![0; self.ndim()];
        in_strides[self.ndim() - 1] = 1;
        for i in (0..self.ndim() - 1).rev() {
            in_strides[i] = in_strides[i + 1] * self.shape[i + 1];
        }

        let mut out_strides = vec![0; out_shape.len()];
        if !out_shape.is_empty() {
            out_strides[out_shape.len() - 1] = 1;
            for i in (0..out_shape.len() - 1).rev() {
                out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
            }
        }

        let data = self.data_ref();
        for idx in 0..self.numel() {
            let mut rem = idx;
            let mut out_idx = 0;
            let mut out_axis = 0;
            for ax in 0..self.ndim() {
                let stride = in_strides[ax];
                let coord = rem / stride;
                rem %= stride;
                if ax != axis {
                    out_idx += coord * out_strides[out_axis];
                    out_axis += 1;
                }
            }
            let v = data[idx];
            if v > out[out_idx] {
                out[out_idx] = v;
            }
        }

        Tensor::from_data(&out_shape, out)
    }

    /// Variance along an axis
    pub fn var_axis(&self, axis: usize) -> Tensor {
        if self.ndim() == 2 && (axis == 0 || axis == 1) {
            let mean = self.mean_axis(axis);
            let (m, n) = (self.shape[0], self.shape[1]);
            let mean_broadcast = if axis == 0 {
                mean.reshape(&[1, n]).broadcast_as(&[m, n])
            } else {
                mean.reshape(&[m, 1]).broadcast_as(&[m, n])
            };
            let diff = self.sub(&mean_broadcast);
            diff.sqr().mean_axis(axis)
        } else if self.ndim() == 3 && axis < 3 {
            let mean = self.mean_axis(axis);
            let (b, s, d) = (self.shape[0], self.shape[1], self.shape[2]);
            let mean_broadcast = if axis == 0 {
                mean.reshape(&[1, s, d]).broadcast_as(&[b, s, d])
            } else if axis == 1 {
                mean.reshape(&[b, 1, d]).broadcast_as(&[b, s, d])
            } else {
                mean.reshape(&[b, s, 1]).broadcast_as(&[b, s, d])
            };
            let diff = self.sub(&mean_broadcast);
            diff.sqr().mean_axis(axis)
        } else {
            panic!("var_axis only implemented for 2D/3D tensors");
        }
    }

    /// Apply a function element-wise
    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        let data: Vec<f32> = self.data_ref().iter().map(|&x| f(x)).collect();
        Tensor::from_data(&self.shape, data)
    }

    /// Reshape the tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of {} elements to shape {:?}",
            self.numel(),
            new_shape
        );

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() && self.gpu_data.is_some() {
            let gpu = self.ensure_gpu_tensor();
            let reshaped = gpu.reshape(new_shape);
            return Self::from_gpu_tensor(reshaped);
        }

        Tensor::from_data(new_shape, self.data())
    }

    /// Returns a contiguous copy of the tensor.
    pub fn contiguous(&self) -> Tensor {
        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let c = gpu.contiguous();
            return Self::from_gpu_tensor(c);
        }

        self.clone()
    }

    /// Narrows the tensor along a dimension.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Tensor {
        assert!(dim < self.ndim(), "narrow dim out of bounds");

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let narrowed = gpu.narrow(dim, start, len);
            return Self::from_gpu_tensor(narrowed);
        }

        if self.ndim() == 3 {
            let (b, s, d) = (self.shape[0], self.shape[1], self.shape[2]);
            if dim == 0 {
                let mut out = vec![0.0; len * s * d];
                let data = self.data_ref();
                let src_stride = s * d;
                let dst_stride = s * d;
                for i in 0..len {
                    let src_off = (start + i) * src_stride;
                    let dst_off = i * dst_stride;
                    out[dst_off..dst_off + dst_stride]
                        .copy_from_slice(&data[src_off..src_off + dst_stride]);
                }
                return Tensor::from_data(&[len, s, d], out);
            } else if dim == 1 {
                let mut out = vec![0.0; b * len * d];
                let data = self.data_ref();
                for bi in 0..b {
                    let src_base = bi * s * d + start * d;
                    let dst_base = bi * len * d;
                    out[dst_base..dst_base + len * d]
                        .copy_from_slice(&data[src_base..src_base + len * d]);
                }
                return Tensor::from_data(&[b, len, d], out);
            } else if dim == 2 {
                let mut out = vec![0.0; b * s * len];
                let data = self.data_ref();
                for bi in 0..b {
                    for si in 0..s {
                        let src_base = bi * s * d + si * d + start;
                        let dst_base = bi * s * len + si * len;
                        out[dst_base..dst_base + len]
                            .copy_from_slice(&data[src_base..src_base + len]);
                    }
                }
                return Tensor::from_data(&[b, s, len], out);
            }
        } else if self.ndim() == 2 {
            let (m, n) = (self.shape[0], self.shape[1]);
            if dim == 0 {
                let mut out = vec![0.0; len * n];
                let data = self.data_ref();
                let src_stride = n;
                for i in 0..len {
                    let src_off = (start + i) * src_stride;
                    let dst_off = i * src_stride;
                    out[dst_off..dst_off + src_stride]
                        .copy_from_slice(&data[src_off..src_off + src_stride]);
                }
                return Tensor::from_data(&[len, n], out);
            } else if dim == 1 {
                let mut out = vec![0.0; m * len];
                let data = self.data_ref();
                for i in 0..m {
                    let src_off = i * n + start;
                    let dst_off = i * len;
                    out[dst_off..dst_off + len]
                        .copy_from_slice(&data[src_off..src_off + len]);
                }
                return Tensor::from_data(&[m, len], out);
            }
        } else if self.ndim() == 1 {
            let data = self.data_ref();
            let out = data[start..start + len].to_vec();
            return Tensor::from_data(&[len], out);
        }

        panic!("narrow not implemented for ndim {}", self.ndim());
    }

    /// Squeezes a dimension of size 1.
    pub fn squeeze(&self, dim: usize) -> Tensor {
        assert!(dim < self.ndim(), "squeeze dim out of bounds");
        assert_eq!(self.shape[dim], 1, "squeeze requires dim size 1");

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu = self.ensure_gpu_tensor();
            let squeezed = gpu.squeeze(dim);
            return Self::from_gpu_tensor(squeezed);
        }

        let mut new_shape = self.shape.clone();
        new_shape.remove(dim);
        Tensor::from_data(&new_shape, self.data())
    }

    /// Concatenates tensors along a dimension.
    pub fn cat(tensors: &[Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "cat requires at least one tensor");
        assert!(dim < tensors[0].ndim(), "cat dim out of bounds");

        #[cfg(any(feature = "metal", feature = "cuda"))]
        if gpu_enabled() && is_gpu_available() {
            let gpu_tensors: Vec<CandleTensor> =
                tensors.iter().map(|t| t.ensure_gpu_tensor()).collect();
            let cat = CandleTensor::cat(&gpu_tensors, dim);
            return Self::from_gpu_tensor(cat);
        }

        let rank = tensors[0].ndim();
        for t in tensors.iter() {
            assert_eq!(t.ndim(), rank, "cat requires same rank");
        }

        if rank == 1 {
            let total: usize = tensors.iter().map(|t| t.shape()[0]).sum();
            let mut out = vec![0.0; total];
            let mut offset = 0;
            for t in tensors.iter() {
                let data = t.data_ref();
                out[offset..offset + data.len()].copy_from_slice(&data);
                offset += data.len();
            }
            return Tensor::from_data(&[total], out);
        }

        if rank == 2 {
            let rows: usize = tensors.iter().map(|t| t.shape()[0]).sum();
            let cols = tensors[0].shape()[1];
            for t in tensors.iter() {
                assert_eq!(t.shape()[1], cols, "cat requires matching dims");
            }
            if dim == 0 {
                let mut out = Vec::with_capacity(rows * cols);
                for t in tensors.iter() {
                    out.extend_from_slice(&t.data_ref());
                }
                return Tensor::from_data(&[rows, cols], out);
            } else if dim == 1 {
                let total_cols: usize = tensors.iter().map(|t| t.shape()[1]).sum();
                let mut out = vec![0.0; rows * total_cols];
                let mut col_offset = 0;
                for t in tensors.iter() {
                    let t_cols = t.shape()[1];
                    let data = t.data_ref();
                    for r in 0..rows {
                        let dst = r * total_cols + col_offset;
                        let src = r * t_cols;
                        out[dst..dst + t_cols].copy_from_slice(&data[src..src + t_cols]);
                    }
                    col_offset += t_cols;
                }
                return Tensor::from_data(&[rows, col_offset], out);
            }
        }

        if rank == 3 {
            let (b0, s0, d0) = (tensors[0].shape()[0], tensors[0].shape()[1], tensors[0].shape()[2]);
            match dim {
                0 => {
                    let total_b: usize = tensors.iter().map(|t| t.shape()[0]).sum();
                    for t in tensors.iter() {
                        assert_eq!(t.shape()[1], s0, "cat requires matching dims");
                        assert_eq!(t.shape()[2], d0, "cat requires matching dims");
                    }
                    let mut out = vec![0.0; total_b * s0 * d0];
                    let mut b_offset = 0;
                    for t in tensors.iter() {
                        let tb = t.shape()[0];
                        let data = t.data_ref();
                        let dst_base = b_offset * s0 * d0;
                        out[dst_base..dst_base + tb * s0 * d0].copy_from_slice(&data);
                        b_offset += tb;
                    }
                    return Tensor::from_data(&[total_b, s0, d0], out);
                }
                1 => {
                    for t in tensors.iter() {
                        assert_eq!(t.shape()[0], b0, "cat requires matching dims");
                        assert_eq!(t.shape()[2], d0, "cat requires matching dims");
                    }
                    let total_s: usize = tensors.iter().map(|t| t.shape()[1]).sum();
                    let mut out = vec![0.0; b0 * total_s * d0];
                    let mut s_offset = 0;
                    for t in tensors.iter() {
                        let ts = t.shape()[1];
                        let data = t.data_ref();
                        for b in 0..b0 {
                            let dst_base = b * total_s * d0 + s_offset * d0;
                            let src_base = b * ts * d0;
                            out[dst_base..dst_base + ts * d0]
                                .copy_from_slice(&data[src_base..src_base + ts * d0]);
                        }
                        s_offset += ts;
                    }
                    return Tensor::from_data(&[b0, total_s, d0], out);
                }
                2 => {
                    for t in tensors.iter() {
                        assert_eq!(t.shape()[0], b0, "cat requires matching dims");
                        assert_eq!(t.shape()[1], s0, "cat requires matching dims");
                    }
                    let total_d: usize = tensors.iter().map(|t| t.shape()[2]).sum();
                    let mut out = vec![0.0; b0 * s0 * total_d];
                    let mut d_offset = 0;
                    for t in tensors.iter() {
                        let td = t.shape()[2];
                        let data = t.data_ref();
                        for b in 0..b0 {
                            for s in 0..s0 {
                                let dst_base = b * s0 * total_d + s * total_d + d_offset;
                                let src_base = b * s0 * td + s * td;
                                out[dst_base..dst_base + td]
                                    .copy_from_slice(&data[src_base..src_base + td]);
                            }
                        }
                        d_offset += td;
                    }
                    return Tensor::from_data(&[b0, s0, total_d], out);
                }
                _ => {}
            }
        }

        panic!("cat not implemented for rank {} dim {}", rank, dim);
    }

    /// Stacks tensors along a new dimension.
    pub fn stack(tensors: &[Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "stack requires at least one tensor");
        let rank = tensors[0].ndim();
        assert!(dim <= rank, "stack dim out of bounds");
        for t in tensors.iter() {
            assert_eq!(t.ndim(), rank, "stack requires same rank");
            assert_eq!(t.shape(), tensors[0].shape(), "stack requires same shape");
        }

        let mut reshaped = Vec::with_capacity(tensors.len());
        for t in tensors.iter() {
            let mut shape = t.shape().to_vec();
            shape.insert(dim, 1);
            reshaped.push(t.reshape(&shape));
        }
        Tensor::cat(&reshaped, dim)
    }

    /// Unstacks a tensor along a dimension into a list of tensors.
    pub fn unstack(&self, dim: usize) -> Vec<Tensor> {
        assert!(dim < self.ndim(), "unstack dim out of bounds");
        let count = self.shape[dim];
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let slice = self.narrow(dim, i, 1);
            out.push(slice.squeeze(dim));
        }
        out
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
        assert_eq!(c.data()[0], 22.0);
        assert_eq!(c.data()[1], 28.0);
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
        assert_eq!(&sum0.data()[..], &[5.0, 7.0, 9.0]);

        let sum1 = a.sum_axis(1);
        assert_eq!(sum1.shape(), &[2]);
        assert_eq!(&sum1.data()[..], &[6.0, 15.0]);
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
        assert_eq!(&b.data()[..], &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::from_data(&[2, 2], vec![2.0, 3.0, 4.0, 5.0]);
        let c = a.mul(&b);
        assert_eq!(&c.data()[..], &[2.0, 6.0, 12.0, 20.0]);
    }
}

/// GPU feature tests - only run when metal or cuda is enabled
#[cfg(test)]
#[cfg(any(feature = "metal", feature = "cuda"))]
mod gpu_tests {
    use super::*;

    fn approx_eq(a: &[f32], b: &[f32], epsilon: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < epsilon)
    }

    #[test]
    fn test_gpu_matmul() {
        set_gpu_enabled(false);
        let a = Tensor::from_data(&[4, 8], (0..32).map(|i| i as f32).collect());
        let b = Tensor::from_data(&[8, 4], (0..32).map(|i| i as f32 * 0.5).collect());
        let cpu_result = a.matmul(&b);

        set_gpu_enabled(true);
        let a = Tensor::from_data(&[4, 8], (0..32).map(|i| i as f32).collect());
        let b = Tensor::from_data(&[8, 4], (0..32).map(|i| i as f32 * 0.5).collect());
        let gpu_result = a.matmul(&b);

        assert_eq!(cpu_result.shape(), gpu_result.shape());
        assert!(
            approx_eq(&cpu_result.data(), &gpu_result.data(), 1e-4),
            "GPU matmul result differs from CPU"
        );
    }

    #[test]
    fn test_gpu_transpose() {
        set_gpu_enabled(false);
        let a = Tensor::from_data(&[3, 4], (0..12).map(|i| i as f32).collect());
        let cpu_result = a.transpose();

        set_gpu_enabled(true);
        let a = Tensor::from_data(&[3, 4], (0..12).map(|i| i as f32).collect());
        let gpu_result = a.transpose();

        assert_eq!(cpu_result.shape(), gpu_result.shape());
        assert!(
            approx_eq(&cpu_result.data(), &gpu_result.data(), 1e-6),
            "GPU transpose result differs from CPU"
        );
    }

    #[test]
    fn test_gpu_add() {
        set_gpu_enabled(false);
        let a = Tensor::from_data(&[4, 4], (0..16).map(|i| i as f32).collect());
        let b = Tensor::from_data(&[4, 4], (0..16).map(|i| (i as f32) * 2.0).collect());
        let cpu_result = a.add(&b);

        set_gpu_enabled(true);
        let a = Tensor::from_data(&[4, 4], (0..16).map(|i| i as f32).collect());
        let b = Tensor::from_data(&[4, 4], (0..16).map(|i| (i as f32) * 2.0).collect());
        let gpu_result = a.add(&b);

        assert_eq!(cpu_result.shape(), gpu_result.shape());
        assert!(
            approx_eq(&cpu_result.data(), &gpu_result.data(), 1e-6),
            "GPU add result differs from CPU"
        );
    }

    #[test]
    fn test_gpu_mul() {
        set_gpu_enabled(false);
        let a = Tensor::from_data(&[4, 4], (0..16).map(|i| i as f32).collect());
        let b = Tensor::from_data(&[4, 4], (0..16).map(|i| (i as f32) * 0.5).collect());
        let cpu_result = a.mul(&b);

        set_gpu_enabled(true);
        let a = Tensor::from_data(&[4, 4], (0..16).map(|i| i as f32).collect());
        let b = Tensor::from_data(&[4, 4], (0..16).map(|i| (i as f32) * 0.5).collect());
        let gpu_result = a.mul(&b);

        assert_eq!(cpu_result.shape(), gpu_result.shape());
        assert!(
            approx_eq(&cpu_result.data(), &gpu_result.data(), 1e-6),
            "GPU mul result differs from CPU"
        );
    }

    #[test]
    fn test_gpu_large_matmul() {
        set_gpu_enabled(false);
        let a = Tensor::from_data(&[64, 128], (0..64 * 128).map(|i| (i as f32) * 0.001).collect());
        let b = Tensor::from_data(&[128, 64], (0..128 * 64).map(|i| (i as f32) * 0.001).collect());
        let cpu_result = a.matmul(&b);

        set_gpu_enabled(true);
        let a = Tensor::from_data(&[64, 128], (0..64 * 128).map(|i| (i as f32) * 0.001).collect());
        let b = Tensor::from_data(&[128, 64], (0..128 * 64).map(|i| (i as f32) * 0.001).collect());
        let gpu_result = a.matmul(&b);

        assert_eq!(cpu_result.shape(), gpu_result.shape());
        assert!(
            approx_eq(&cpu_result.data(), &gpu_result.data(), 0.5),
            "GPU large matmul result differs from CPU"
        );
    }

    #[test]
    fn test_gpu_chained_operations() {
        set_gpu_enabled(false);
        let x = Tensor::from_data(&[8, 16], (0..128).map(|i| (i as f32) * 0.1).collect());
        let w1 = Tensor::from_data(&[16, 8], (0..128).map(|i| (i as f32) * 0.01).collect());
        let w2 = Tensor::from_data(&[8, 4], (0..32).map(|i| (i as f32) * 0.02).collect());
        let h1 = x.matmul(&w1);
        let h2 = h1.matmul(&w2);
        let cpu_result = h2;

        set_gpu_enabled(true);
        let x = Tensor::from_data(&[8, 16], (0..128).map(|i| (i as f32) * 0.1).collect());
        let w1 = Tensor::from_data(&[16, 8], (0..128).map(|i| (i as f32) * 0.01).collect());
        let w2 = Tensor::from_data(&[8, 4], (0..32).map(|i| (i as f32) * 0.02).collect());
        let h1 = x.matmul(&w1);
        let h2 = h1.matmul(&w2);
        let gpu_result = h2;

        assert_eq!(cpu_result.shape(), gpu_result.shape());
        assert!(
            approx_eq(&cpu_result.data(), &gpu_result.data(), 1e-2),
            "GPU chained matmul result differs from CPU"
        );
    }

    #[test]
    fn test_gpu_resident_no_cpu_sync() {
        // Test that chained GPU operations don't sync to CPU until data() is called
        set_gpu_enabled(true);

        let x = Tensor::from_data(&[32, 64], (0..32*64).map(|i| (i as f32) * 0.01).collect());
        let w1 = Tensor::from_data(&[64, 32], (0..64*32).map(|i| (i as f32) * 0.01).collect());
        let w2 = Tensor::from_data(&[32, 16], (0..32*16).map(|i| (i as f32) * 0.01).collect());
        let w3 = Tensor::from_data(&[16, 8], (0..16*8).map(|i| (i as f32) * 0.01).collect());

        // Chain of operations - should all stay on GPU
        let h1 = x.matmul(&w1);
        assert!(!h1.is_cpu_current(), "h1 should not have synced to CPU");

        let h2 = h1.matmul(&w2);
        assert!(!h2.is_cpu_current(), "h2 should not have synced to CPU");

        let h3 = h2.matmul(&w3);
        assert!(!h3.is_cpu_current(), "h3 should not have synced to CPU");

        // Only now should it sync to CPU
        let _ = h3.data();
        assert!(h3.is_cpu_current(), "h3 should have synced to CPU after data() call");
    }
}
