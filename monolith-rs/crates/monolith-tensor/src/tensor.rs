//! Tensor trait and data types for backend-agnostic tensor operations.
//!
//! This module defines the core [`Tensor`] trait that all tensor backends must implement,
//! as well as the [`DType`] enum for representing tensor data types.

use std::fmt;

/// Data types supported by tensors.
///
/// This enum represents the numeric types that tensors can hold.
///
/// # Examples
///
/// ```
/// use monolith_tensor::DType;
///
/// let dtype = DType::F32;
/// assert_eq!(dtype.size_in_bytes(), 4);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// 32-bit floating point.
    F32,
    /// 64-bit floating point.
    F64,
    /// 32-bit signed integer.
    I32,
    /// 64-bit signed integer.
    I64,
}

impl DType {
    /// Returns the size of this data type in bytes.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::DType;
    ///
    /// assert_eq!(DType::F32.size_in_bytes(), 4);
    /// assert_eq!(DType::F64.size_in_bytes(), 8);
    /// assert_eq!(DType::I32.size_in_bytes(), 4);
    /// assert_eq!(DType::I64.size_in_bytes(), 8);
    /// ```
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
        }
    }

    /// Returns a human-readable name for this data type.
    pub fn name(&self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::I32 => "int32",
            DType::I64 => "int64",
        }
    }

    /// Returns whether this is a floating-point type.
    pub fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64)
    }

    /// Returns whether this is an integer type.
    pub fn is_integer(&self) -> bool {
        matches!(self, DType::I32 | DType::I64)
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Trait for backend-agnostic tensor operations.
///
/// This trait defines the core operations that all tensor backends must implement.
/// It allows writing code that is generic over the tensor backend, enabling
/// easy switching between different implementations (ndarray, tch, candle, etc.).
///
/// # Examples
///
/// ```
/// use monolith_tensor::{Tensor, DType, NdArrayTensor};
///
/// fn sum_tensors<T: Tensor>(a: &T, b: &T) -> T {
///     a.add(b)
/// }
///
/// let a = NdArrayTensor::ones(&[2, 3]);
/// let b = NdArrayTensor::ones(&[2, 3]);
/// let c = sum_tensors(&a, &b);
/// assert_eq!(c.to_vec(), vec![2.0; 6]);
/// ```
pub trait Tensor: Clone + fmt::Debug {
    /// Returns the shape of the tensor as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::zeros(&[2, 3, 4]);
    /// assert_eq!(t.shape(), &[2, 3, 4]);
    /// ```
    fn shape(&self) -> &[usize];

    /// Returns the data type of the tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, DType, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::zeros(&[2, 3]);
    /// assert_eq!(t.dtype(), DType::F32);
    /// ```
    fn dtype(&self) -> DType;

    /// Creates a tensor filled with zeros.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to create.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::zeros(&[2, 3]);
    /// assert_eq!(t.to_vec(), vec![0.0; 6]);
    /// ```
    fn zeros(shape: &[usize]) -> Self;

    /// Creates a tensor filled with ones.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the tensor to create.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::ones(&[2, 3]);
    /// assert_eq!(t.to_vec(), vec![1.0; 6]);
    /// ```
    fn ones(shape: &[usize]) -> Self;

    /// Creates a tensor from a slice of f32 values.
    ///
    /// # Arguments
    ///
    /// * `data` - The data to initialize the tensor with.
    /// * `shape` - The shape of the tensor.
    ///
    /// # Panics
    ///
    /// Panics if the length of `data` does not match the product of `shape`.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    /// let t = NdArrayTensor::from_slice(&data, &[2, 3]);
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.to_vec(), data);
    /// ```
    fn from_slice(data: &[f32], shape: &[usize]) -> Self;

    /// Converts the tensor to a flat Vec<f32>.
    ///
    /// The elements are returned in row-major (C) order.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let t = NdArrayTensor::ones(&[2, 3]);
    /// let vec = t.to_vec();
    /// assert_eq!(vec.len(), 6);
    /// ```
    fn to_vec(&self) -> Vec<f32>;

    /// Adds two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to add.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are not compatible for broadcasting.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let a = NdArrayTensor::ones(&[2, 3]);
    /// let b = NdArrayTensor::ones(&[2, 3]);
    /// let c = a.add(&b);
    /// assert_eq!(c.to_vec(), vec![2.0; 6]);
    /// ```
    fn add(&self, other: &Self) -> Self;

    /// Multiplies two tensors element-wise.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are not compatible for broadcasting.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let a = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    /// let b = NdArrayTensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]);
    /// let c = a.mul(&b);
    /// assert_eq!(c.to_vec(), vec![2.0, 4.0, 6.0, 8.0]);
    /// ```
    fn mul(&self, other: &Self) -> Self;

    /// Performs matrix multiplication.
    ///
    /// For 2D tensors, this is standard matrix multiplication.
    /// For higher-dimensional tensors, the last two dimensions are treated as matrices
    /// and batch multiplication is performed.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to multiply.
    ///
    /// # Panics
    ///
    /// Panics if the shapes are not compatible for matrix multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::{Tensor, NdArrayTensor};
    ///
    /// let a = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    /// let b = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
    /// let c = a.matmul(&b);
    /// assert_eq!(c.shape(), &[2, 2]);
    /// ```
    fn matmul(&self, other: &Self) -> Self;

    /// Returns the total number of elements in the tensor.
    fn numel(&self) -> usize {
        self.shape().iter().product()
    }

    /// Returns the number of dimensions (rank) of the tensor.
    fn ndim(&self) -> usize {
        self.shape().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_size() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::F64.size_in_bytes(), 8);
        assert_eq!(DType::I32.size_in_bytes(), 4);
        assert_eq!(DType::I64.size_in_bytes(), 8);
    }

    #[test]
    fn test_dtype_name() {
        assert_eq!(DType::F32.name(), "float32");
        assert_eq!(DType::F64.name(), "float64");
        assert_eq!(DType::I32.name(), "int32");
        assert_eq!(DType::I64.name(), "int64");
    }

    #[test]
    fn test_dtype_is_float() {
        assert!(DType::F32.is_float());
        assert!(DType::F64.is_float());
        assert!(!DType::I32.is_float());
        assert!(!DType::I64.is_float());
    }

    #[test]
    fn test_dtype_is_integer() {
        assert!(!DType::F32.is_integer());
        assert!(!DType::F64.is_integer());
        assert!(DType::I32.is_integer());
        assert!(DType::I64.is_integer());
    }

    #[test]
    fn test_dtype_display() {
        assert_eq!(format!("{}", DType::F32), "float32");
        assert_eq!(format!("{}", DType::I64), "int64");
    }
}
