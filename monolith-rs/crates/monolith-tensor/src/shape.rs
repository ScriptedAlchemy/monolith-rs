//! Shape utilities for tensor dimensions.
//!
//! This module provides the [`Shape`] type for representing and manipulating
//! tensor dimensions, along with helper functions for common shape operations.

use std::fmt;
use std::ops::{Deref, Index};

/// Represents the shape (dimensions) of a tensor.
///
/// A shape is an ordered sequence of dimension sizes. For example, a 2D matrix
/// with 3 rows and 4 columns has shape `[3, 4]`.
///
/// # Examples
///
/// ```
/// use monolith_tensor::Shape;
///
/// let shape = Shape::new(vec![2, 3, 4]);
/// assert_eq!(shape.ndim(), 3);
/// assert_eq!(shape.numel(), 24);
/// assert_eq!(shape[0], 2);
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new shape from the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `dims` - A vector of dimension sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.as_slice(), &[2, 3, 4]);
    /// ```
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Creates a scalar shape (zero dimensions).
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::scalar();
    /// assert_eq!(shape.ndim(), 0);
    /// assert_eq!(shape.numel(), 1);
    /// ```
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Creates a 1D shape (vector).
    ///
    /// # Arguments
    ///
    /// * `len` - The length of the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::vector(5);
    /// assert_eq!(shape.ndim(), 1);
    /// assert_eq!(shape.numel(), 5);
    /// ```
    pub fn vector(len: usize) -> Self {
        Self { dims: vec![len] }
    }

    /// Creates a 2D shape (matrix).
    ///
    /// # Arguments
    ///
    /// * `rows` - The number of rows.
    /// * `cols` - The number of columns.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::matrix(3, 4);
    /// assert_eq!(shape.ndim(), 2);
    /// assert_eq!(shape.numel(), 12);
    /// ```
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self {
            dims: vec![rows, cols],
        }
    }

    /// Returns the number of dimensions (rank) of the shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.ndim(), 3);
    /// ```
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements in a tensor of this shape.
    ///
    /// This is the product of all dimension sizes.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.numel(), 24);
    /// ```
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    /// Returns the dimensions as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// assert_eq!(shape.as_slice(), &[2, 3, 4]);
    /// ```
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the dimensions as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [usize] {
        &mut self.dims
    }

    /// Consumes the shape and returns the underlying dimensions vector.
    pub fn into_vec(self) -> Vec<usize> {
        self.dims
    }

    /// Returns whether this shape is compatible for broadcasting with another shape.
    ///
    /// Two shapes are broadcast-compatible if, starting from the trailing dimensions,
    /// the dimension sizes are either equal or one of them is 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let a = Shape::new(vec![3, 4]);
    /// let b = Shape::new(vec![4]);
    /// assert!(a.is_broadcast_compatible(&b));
    ///
    /// let c = Shape::new(vec![3, 1]);
    /// let d = Shape::new(vec![1, 4]);
    /// assert!(c.is_broadcast_compatible(&d));
    /// ```
    pub fn is_broadcast_compatible(&self, other: &Shape) -> bool {
        let self_iter = self.dims.iter().rev();
        let other_iter = other.dims.iter().rev();

        for (a, b) in self_iter.zip(other_iter) {
            if *a != *b && *a != 1 && *b != 1 {
                return false;
            }
        }
        true
    }

    /// Computes the broadcast shape of two shapes.
    ///
    /// Returns `None` if the shapes are not broadcast-compatible.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let a = Shape::new(vec![3, 1]);
    /// let b = Shape::new(vec![1, 4]);
    /// let result = a.broadcast_with(&b).unwrap();
    /// assert_eq!(result.as_slice(), &[3, 4]);
    /// ```
    pub fn broadcast_with(&self, other: &Shape) -> Option<Shape> {
        if !self.is_broadcast_compatible(other) {
            return None;
        }

        let max_ndim = self.ndim().max(other.ndim());
        let mut result = vec![0; max_ndim];

        for i in 0..max_ndim {
            let a = if i < self.ndim() {
                self.dims[self.ndim() - 1 - i]
            } else {
                1
            };
            let b = if i < other.ndim() {
                other.dims[other.ndim() - 1 - i]
            } else {
                1
            };
            result[max_ndim - 1 - i] = a.max(b);
        }

        Some(Shape::new(result))
    }

    /// Returns the strides for a contiguous tensor with this shape.
    ///
    /// Strides indicate how many elements to skip to move one position
    /// along each dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_tensor::Shape;
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// let strides = shape.strides();
    /// assert_eq!(strides, vec![12, 4, 1]);
    /// ```
    pub fn strides(&self) -> Vec<usize> {
        if self.dims.is_empty() {
            return vec![];
        }

        let mut strides = vec![1; self.ndim()];
        for i in (0..self.ndim() - 1).rev() {
            strides[i] = strides[i + 1] * self.dims[i + 1];
        }
        strides
    }
}

impl fmt::Debug for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Shape({:?})", self.dims)
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, ")")
    }
}

impl Deref for Shape {
    type Target = [usize];

    fn deref(&self) -> &Self::Target {
        &self.dims
    }
}

impl Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(dims: [usize; N]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl IntoIterator for Shape {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.into_iter()
    }
}

impl<'a> IntoIterator for &'a Shape {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.iter()
    }
}

/// Validates that the given shape is valid for a matrix multiplication.
///
/// For two matrices A (m x k) and B (k x n), the inner dimensions must match.
///
/// # Arguments
///
/// * `a` - The shape of the first matrix.
/// * `b` - The shape of the second matrix.
///
/// # Returns
///
/// Returns `true` if the shapes are compatible for matrix multiplication.
pub fn is_matmul_compatible(a: &Shape, b: &Shape) -> bool {
    if a.ndim() < 1 || b.ndim() < 1 {
        return false;
    }

    // For 1D tensors, treat as column/row vectors
    let a_cols = if a.ndim() == 1 { a[0] } else { a[a.ndim() - 1] };
    let b_rows = if b.ndim() == 1 { b[0] } else { b[b.ndim() - 2] };

    a_cols == b_rows
}

/// Computes the output shape of a matrix multiplication.
///
/// # Arguments
///
/// * `a` - The shape of the first matrix.
/// * `b` - The shape of the second matrix.
///
/// # Returns
///
/// Returns `Some(shape)` if the shapes are compatible, `None` otherwise.
pub fn matmul_output_shape(a: &Shape, b: &Shape) -> Option<Shape> {
    if !is_matmul_compatible(a, b) {
        return None;
    }

    match (a.ndim(), b.ndim()) {
        (1, 1) => Some(Shape::scalar()),
        (1, 2) => Some(Shape::vector(b[1])),
        (2, 1) => Some(Shape::vector(a[0])),
        (2, 2) => Some(Shape::matrix(a[0], b[1])),
        _ => {
            // Batched matmul
            let mut batch_dims = a.as_slice()[..a.ndim() - 2].to_vec();
            let m = a[a.ndim() - 2];
            let n = b[b.ndim() - 1];
            batch_dims.push(m);
            batch_dims.push(n);
            Some(Shape::new(batch_dims))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_new() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_scalar() {
        let shape = Shape::scalar();
        assert_eq!(shape.ndim(), 0);
        assert_eq!(shape.numel(), 1);
    }

    #[test]
    fn test_shape_vector() {
        let shape = Shape::vector(5);
        assert_eq!(shape.ndim(), 1);
        assert_eq!(shape.numel(), 5);
        assert_eq!(shape[0], 5);
    }

    #[test]
    fn test_shape_matrix() {
        let shape = Shape::matrix(3, 4);
        assert_eq!(shape.ndim(), 2);
        assert_eq!(shape.numel(), 12);
        assert_eq!(shape[0], 3);
        assert_eq!(shape[1], 4);
    }

    #[test]
    fn test_shape_numel() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.numel(), 24);
    }

    #[test]
    fn test_shape_broadcast_compatible() {
        let a = Shape::new(vec![3, 4]);
        let b = Shape::new(vec![4]);
        assert!(a.is_broadcast_compatible(&b));

        let c = Shape::new(vec![3, 1]);
        let d = Shape::new(vec![1, 4]);
        assert!(c.is_broadcast_compatible(&d));

        let e = Shape::new(vec![3, 4]);
        let f = Shape::new(vec![5]);
        assert!(!e.is_broadcast_compatible(&f));
    }

    #[test]
    fn test_shape_broadcast_with() {
        let a = Shape::new(vec![3, 1]);
        let b = Shape::new(vec![1, 4]);
        let result = a.broadcast_with(&b).unwrap();
        assert_eq!(result.as_slice(), &[3, 4]);

        let c = Shape::new(vec![2, 3, 4]);
        let d = Shape::new(vec![4]);
        let result = c.broadcast_with(&d).unwrap();
        assert_eq!(result.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_strides() {
        let shape = Shape::new(vec![2, 3, 4]);
        let strides = shape.strides();
        assert_eq!(strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_matmul_compatible() {
        let a = Shape::matrix(3, 4);
        let b = Shape::matrix(4, 5);
        assert!(is_matmul_compatible(&a, &b));

        let c = Shape::matrix(3, 4);
        let d = Shape::matrix(5, 6);
        assert!(!is_matmul_compatible(&c, &d));
    }

    #[test]
    fn test_matmul_output_shape() {
        let a = Shape::matrix(3, 4);
        let b = Shape::matrix(4, 5);
        let result = matmul_output_shape(&a, &b).unwrap();
        assert_eq!(result.as_slice(), &[3, 5]);

        let c = Shape::vector(4);
        let d = Shape::matrix(4, 5);
        let result = matmul_output_shape(&c, &d).unwrap();
        assert_eq!(result.as_slice(), &[5]);
    }

    #[test]
    fn test_shape_from_conversions() {
        let shape1: Shape = vec![2, 3, 4].into();
        assert_eq!(shape1.as_slice(), &[2, 3, 4]);

        let shape2: Shape = [2, 3, 4].into();
        assert_eq!(shape2.as_slice(), &[2, 3, 4]);

        let dims = &[2, 3, 4][..];
        let shape3: Shape = dims.into();
        assert_eq!(shape3.as_slice(), &[2, 3, 4]);
    }

    #[test]
    fn test_shape_display() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(format!("{}", shape), "(2, 3, 4)");
    }
}
