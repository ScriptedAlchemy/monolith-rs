//! ndarray-based tensor backend implementation.
//!
//! This module provides [`NdArrayTensor`], a tensor implementation backed by
//! the `ndarray` crate. This is the default CPU backend for tensor operations.

use ndarray::{ArrayD, IxDyn};

use crate::tensor::{DType, Tensor};

/// A tensor backed by ndarray's ArrayD.
///
/// This struct wraps an `ndarray::ArrayD<f32>` and implements the [`Tensor`] trait,
/// providing a CPU-based tensor backend.
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, NdArrayTensor};
///
/// let t = NdArrayTensor::zeros(&[2, 3, 4]);
/// assert_eq!(t.shape(), &[2, 3, 4]);
/// assert_eq!(t.numel(), 24);
/// ```
#[derive(Clone, Debug)]
pub struct NdArrayTensor {
    data: ArrayD<f32>,
}

impl NdArrayTensor {
    /// Creates a new NdArrayTensor from an ndarray ArrayD.
    ///
    /// # Arguments
    ///
    /// * `data` - The underlying ndarray.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndarray::ArrayD;
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let arr = ArrayD::<f32>::zeros(vec![2, 3]);
    /// let t = NdArrayTensor::from_ndarray(arr);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// ```
    pub fn from_ndarray(data: ArrayD<f32>) -> Self {
        Self { data }
    }

    /// Returns a reference to the underlying ndarray.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::ones(&[2, 3]);
    /// let arr = t.as_ndarray();
    /// assert_eq!(arr.shape(), &[2, 3]);
    /// ```
    pub fn as_ndarray(&self) -> &ArrayD<f32> {
        &self.data
    }

    /// Returns a mutable reference to the underlying ndarray.
    pub fn as_ndarray_mut(&mut self) -> &mut ArrayD<f32> {
        &mut self.data
    }

    /// Consumes the tensor and returns the underlying ndarray.
    pub fn into_ndarray(self) -> ArrayD<f32> {
        self.data
    }

    /// Reshapes the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The new shape.
    ///
    /// # Panics
    ///
    /// Panics if the new shape has a different number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let reshaped = t.reshape(&[3, 2]);
    /// assert_eq!(reshaped.shape(), &[3, 2]);
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_numel,
            "Cannot reshape tensor of {} elements to shape {:?} with {} elements",
            self.numel(),
            new_shape,
            new_numel
        );

        let data = self.data.clone().into_shape(IxDyn(new_shape)).unwrap();
        Self { data }
    }

    /// Transposes a 2D tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not 2D.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let transposed = t.transpose();
    /// assert_eq!(transposed.shape(), &[3, 2]);
    /// ```
    pub fn transpose(&self) -> Self {
        assert_eq!(self.ndim(), 2, "transpose only works on 2D tensors");
        let transposed = self.data.clone().reversed_axes();
        Self { data: transposed }
    }

    /// Returns the sum of all elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// assert_eq!(t.sum(), 10.0);
    /// ```
    pub fn sum(&self) -> f32 {
        self.data.sum()
    }

    /// Returns the mean of all elements in the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// assert_eq!(t.mean(), 2.5);
    /// ```
    pub fn mean(&self) -> f32 {
        self.sum() / self.numel() as f32
    }

    /// Returns the maximum value in the tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 4.0, 2.0, 3.0], &[2, 2]);
    /// assert_eq!(t.max(), 4.0);
    /// ```
    pub fn max(&self) -> f32 {
        self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Returns the minimum value in the tensor.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 4.0, 2.0, 3.0], &[2, 2]);
    /// assert_eq!(t.min(), 1.0);
    /// ```
    pub fn min(&self) -> f32 {
        self.data.iter().cloned().fold(f32::INFINITY, f32::min)
    }

    /// Subtracts two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to subtract.
    ///
    /// # Panics
    ///
    /// Panics if the shapes don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let a = NdArrayTensor::from_slice(&[3.0, 4.0, 5.0, 6.0], &[2, 2]);
    /// let b = NdArrayTensor::ones(&[2, 2]);
    /// let c = a.sub(&b);
    /// assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    /// ```
    pub fn sub(&self, other: &Self) -> Self {
        let data = &self.data - &other.data;
        Self { data }
    }

    /// Divides two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to divide by.
    ///
    /// # Panics
    ///
    /// Panics if the shapes don't match.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let a = NdArrayTensor::from_slice(&[4.0, 6.0, 8.0, 10.0], &[2, 2]);
    /// let b = NdArrayTensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    /// let c = a.div(&b);
    /// assert_eq!(c.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    /// ```
    pub fn div(&self, other: &Self) -> Self {
        let data = &self.data / &other.data;
        Self { data }
    }

    /// Scales the tensor by a scalar value.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar to multiply by.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let scaled = t.scale(2.0);
    /// assert_eq!(scaled.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    /// ```
    pub fn scale(&self, scalar: f32) -> Self {
        let data = &self.data * scalar;
        Self { data }
    }

    /// Applies a function element-wise to the tensor.
    ///
    /// # Arguments
    ///
    /// * `f` - The function to apply.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::from_slice(&[1.0, 4.0, 9.0, 16.0], &[2, 2]);
    /// let sqrt = t.map(|x| x.sqrt());
    /// assert_eq!(sqrt.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        let data = self.data.mapv(f);
        Self { data }
    }
}

impl Tensor for NdArrayTensor {
    fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    fn dtype(&self) -> DType {
        DType::F32
    }

    fn zeros(shape: &[usize]) -> Self {
        let data = ArrayD::zeros(IxDyn(shape));
        Self { data }
    }

    fn ones(shape: &[usize]) -> Self {
        let data = ArrayD::ones(IxDyn(shape));
        Self { data }
    }

    fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "Data length {} does not match shape {:?} (expected {} elements)",
            data.len(),
            shape,
            expected_len
        );

        let arr = ArrayD::from_shape_vec(IxDyn(shape), data.to_vec()).unwrap();
        Self { data: arr }
    }

    fn to_vec(&self) -> Vec<f32> {
        self.data.iter().cloned().collect()
    }

    fn add(&self, other: &Self) -> Self {
        let data = &self.data + &other.data;
        Self { data }
    }

    fn mul(&self, other: &Self) -> Self {
        let data = &self.data * &other.data;
        Self { data }
    }

    fn matmul(&self, other: &Self) -> Self {
        assert!(
            self.ndim() >= 1 && other.ndim() >= 1,
            "matmul requires at least 1D tensors"
        );

        // Handle 2D case
        if self.ndim() == 2 && other.ndim() == 2 {
            let a = self
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let b = other
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            assert_eq!(
                a.ncols(),
                b.nrows(),
                "matmul shape mismatch: ({}, {}) x ({}, {})",
                a.nrows(),
                a.ncols(),
                b.nrows(),
                b.ncols()
            );

            let result = a.dot(&b);
            Self {
                data: result.into_dyn(),
            }
        } else if self.ndim() == 1 && other.ndim() == 1 {
            // Vector dot product
            let a = self
                .data
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap();
            let b = other
                .data
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap();

            assert_eq!(
                a.len(),
                b.len(),
                "dot product requires vectors of same length"
            );

            let result = a.dot(&b);
            Self {
                data: ArrayD::from_elem(IxDyn(&[]), result),
            }
        } else if self.ndim() == 1 && other.ndim() == 2 {
            // Vector-matrix multiplication
            let a = self
                .data
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap();
            let b = other
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();

            assert_eq!(
                a.len(),
                b.nrows(),
                "matmul shape mismatch: ({},) x ({}, {})",
                a.len(),
                b.nrows(),
                b.ncols()
            );

            let result = a.dot(&b);
            Self {
                data: result.into_dyn(),
            }
        } else if self.ndim() == 2 && other.ndim() == 1 {
            // Matrix-vector multiplication
            let a = self
                .data
                .view()
                .into_dimensionality::<ndarray::Ix2>()
                .unwrap();
            let b = other
                .data
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .unwrap();

            assert_eq!(
                a.ncols(),
                b.len(),
                "matmul shape mismatch: ({}, {}) x ({},)",
                a.nrows(),
                a.ncols(),
                b.len()
            );

            let result = a.dot(&b);
            Self {
                data: result.into_dyn(),
            }
        } else {
            // For higher dimensions, we'd need batched matmul
            // This is a simplified implementation that doesn't support batched matmul
            panic!(
                "Batched matmul not yet implemented for shapes {:?} and {:?}",
                self.shape(),
                other.shape()
            );
        }
    }
}

impl From<ArrayD<f32>> for NdArrayTensor {
    fn from(data: ArrayD<f32>) -> Self {
        Self::from_ndarray(data)
    }
}

impl From<NdArrayTensor> for ArrayD<f32> {
    fn from(tensor: NdArrayTensor) -> Self {
        tensor.into_ndarray()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros() {
        let t = NdArrayTensor::zeros(&[2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert!(t.to_vec().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let t = NdArrayTensor::ones(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert!(t.to_vec().iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_from_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = NdArrayTensor::from_slice(&data, &[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.to_vec(), data);
    }

    #[test]
    fn test_add() {
        let a = NdArrayTensor::ones(&[2, 3]);
        let b = NdArrayTensor::ones(&[2, 3]);
        let c = a.add(&b);
        assert!(c.to_vec().iter().all(|&x| x == 2.0));
    }

    #[test]
    fn test_mul() {
        let a = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = NdArrayTensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
        let c = a.mul(&b);
        assert_eq!(c.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matmul_2d() {
        // 2x3 matrix multiplied by 3x2 matrix
        let a = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let c = a.matmul(&b);

        assert_eq!(c.shape(), &[2, 2]);
        // Result should be:
        // [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6]   = [22, 28]
        // [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6]   = [49, 64]
        assert_eq!(c.to_vec(), vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_matmul_vector() {
        // Dot product of two vectors
        let a = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0], &[3]);
        let b = NdArrayTensor::from_slice(&[4.0, 5.0, 6.0], &[3]);
        let c = a.matmul(&b);

        assert_eq!(c.shape(), &[] as &[usize]);
        assert_eq!(c.to_vec(), vec![32.0]); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_reshape() {
        let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let reshaped = t.reshape(&[3, 2]);
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_transpose() {
        let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let transposed = t.transpose();
        assert_eq!(transposed.shape(), &[3, 2]);
        assert_eq!(transposed.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_sum_mean_max_min() {
        let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(t.sum(), 10.0);
        assert_eq!(t.mean(), 2.5);
        assert_eq!(t.max(), 4.0);
        assert_eq!(t.min(), 1.0);
    }

    #[test]
    fn test_sub_div() {
        let a = NdArrayTensor::from_slice(&[4.0, 6.0, 8.0, 10.0], &[2, 2]);
        let b = NdArrayTensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);

        let sub = a.sub(&b);
        assert_eq!(sub.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);

        let div = a.div(&b);
        assert_eq!(div.to_vec(), vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_scale() {
        let t = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let scaled = t.scale(2.0);
        assert_eq!(scaled.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_map() {
        let t = NdArrayTensor::from_slice(&[1.0, 4.0, 9.0, 16.0], &[2, 2]);
        let sqrt = t.map(|x| x.sqrt());
        assert_eq!(sqrt.to_vec(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dtype() {
        let t = NdArrayTensor::zeros(&[2, 2]);
        assert_eq!(t.dtype(), DType::F32);
    }
}
