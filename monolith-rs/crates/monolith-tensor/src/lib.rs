//! Backend-agnostic tensor abstractions for Monolith.
//!
//! This crate provides a pluggable tensor backend system that allows users to
//! switch between different tensor implementations (ndarray, tch, candle) without
//! changing their application code.
//!
//! # Overview
//!
//! The core of this crate is the [`Tensor`] trait, which defines the interface
//! that all tensor backends must implement. This includes basic operations like
//! creation (zeros, ones), element-wise operations (add, mul), and matrix operations
//! (matmul).
//!
//! # Features
//!
//! - **`cpu`** (default): Enables CPU-based tensor operations using ndarray.
//!
//! # Backends
//!
//! Currently, the following backend is implemented:
//!
//! - [`NdArrayTensor`]: A CPU-based backend using the `ndarray` crate.
//!
//! # Example
//!
//! ```rust
//! use monolith_tensor::{Tensor, NdArrayTensor, DType, Shape};
//!
//! // Create tensors
//! let a = NdArrayTensor::ones(&[2, 3]);
//! let b = NdArrayTensor::ones(&[2, 3]);
//!
//! // Perform operations
//! let c = a.add(&b);
//! assert_eq!(c.to_vec(), vec![2.0; 6]);
//!
//! // Check properties
//! assert_eq!(c.shape(), &[2, 3]);
//! assert_eq!(c.dtype(), DType::F32);
//! ```
//!
//! # Activation Functions
//!
//! The [`ops`] module provides common activation functions:
//!
//! ```rust
//! use monolith_tensor::{Tensor, NdArrayTensor, ops};
//!
//! let x = NdArrayTensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);
//!
//! let relu_result = ops::relu(&x);
//! assert_eq!(relu_result.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);
//!
//! let sigmoid_result = ops::sigmoid(&x);
//! // Values are between 0 and 1
//! ```

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod candle_backend;
pub mod ndarray_backend;
pub mod ops;
pub mod shape;
pub mod tensor;

// Re-exports for convenience
pub use candle_backend::CandleTensor;
pub use candle_core::Device;
pub use ndarray_backend::NdArrayTensor;
pub use shape::{is_matmul_compatible, matmul_output_shape, Shape};
pub use tensor::{DType, Tensor};

/// Error types for tensor operations.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// Shape mismatch error.
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// The expected shape.
        expected: Vec<usize>,
        /// The actual shape.
        got: Vec<usize>,
    },

    /// Invalid shape error.
    #[error("Invalid shape: {0}")]
    InvalidShape(String),

    /// Data type mismatch error.
    #[error("DType mismatch: expected {expected}, got {got}")]
    DTypeMismatch {
        /// The expected data type.
        expected: DType,
        /// The actual data type.
        got: DType,
    },

    /// Operation not supported error.
    #[error("Operation not supported: {0}")]
    NotSupported(String),

    /// Generic error with message.
    #[error("{0}")]
    Other(String),
}

/// Result type for tensor operations.
pub type TensorResult<T> = Result<T, TensorError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // Create tensors
        let a = NdArrayTensor::ones(&[2, 3]);
        let b = NdArrayTensor::ones(&[2, 3]);

        // Add them
        let c = a.add(&b);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec(), vec![2.0; 6]);

        // Check dtype
        assert_eq!(c.dtype(), DType::F32);
    }

    #[test]
    fn test_matmul_workflow() {
        // Create matrices
        let a = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = NdArrayTensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        // Matrix multiply
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_shape_utilities() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);

        let strides = shape.strides();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_ops_workflow() {
        let x = NdArrayTensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]);

        let relu_result = ops::relu(&x);
        assert_eq!(relu_result.to_vec(), vec![0.0, 0.0, 1.0, 2.0]);

        let sigmoid_result = ops::sigmoid(&x);
        // Check that sigmoid is between 0 and 1
        for val in sigmoid_result.to_vec() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_tensor_error() {
        let err = TensorError::ShapeMismatch {
            expected: vec![2, 3],
            got: vec![3, 2],
        };
        assert!(format!("{}", err).contains("Shape mismatch"));

        let err = TensorError::DTypeMismatch {
            expected: DType::F32,
            got: DType::F64,
        };
        assert!(format!("{}", err).contains("DType mismatch"));
    }
}
