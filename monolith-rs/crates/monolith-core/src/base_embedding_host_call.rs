//! Minimal port of Python `monolith/core/base_embedding_host_call.py`.
//!
//! The Python version is TPU/TF specific. For Rust parity we port the core
//! tensor-shape logic that is unit-tested in Python: `_compute_new_value`.

use crate::error::{MonolithError, Result};

/// Computes the updated 1D buffer value by adding `delta` into `base` starting at `offset`.
///
/// Python semantics (TF):
/// - `delta` is padded at the end with zeros to match `base_len`
/// - `delta` is then "rolled" (circular shifted) by `offset` (mod `base_len`)
/// - elementwise added into `base`
///
/// This is equivalent to:
/// `out[i] = base[i] + delta[(i - offset) mod base_len]` for `i < delta_len`, else unchanged.
pub fn compute_new_value_i32(base: &[i32], delta: &[i32], offset: usize) -> Result<Vec<i32>> {
    if base.is_empty() {
        return Err(MonolithError::ConfigError {
            message: "base must be non-empty".to_string(),
        });
    }
    if delta.len() > base.len() {
        return Err(MonolithError::ConfigError {
            message: format!(
                "delta length ({}) must be <= base length ({})",
                delta.len(),
                base.len()
            ),
        });
    }

    let n = base.len();
    let shift = offset % n;
    let mut rolled = vec![0_i32; n];
    for (i, &v) in delta.iter().enumerate() {
        rolled[(i + shift) % n] = v;
    }

    Ok(base.iter().zip(rolled.iter()).map(|(a, b)| a + b).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_new_value_matches_python_test() {
        let mut base = vec![0_i32; 10];
        let delta = vec![1_i32, 1_i32];

        base = compute_new_value_i32(&base, &delta, 1).unwrap();
        assert_eq!(base, vec![0, 1, 1, 0, 0, 0, 0, 0, 0, 0]);

        base = compute_new_value_i32(&base, &delta, 5).unwrap();
        assert_eq!(base, vec![0, 1, 1, 0, 0, 1, 1, 0, 0, 0]);

        base = compute_new_value_i32(&base, &delta, 6).unwrap();
        assert_eq!(base, vec![0, 1, 1, 0, 0, 1, 2, 1, 0, 0]);
    }

    #[test]
    fn test_compute_new_value_validates_lengths() {
        let base = vec![0_i32; 2];
        let delta = vec![1_i32, 2_i32, 3_i32];
        let err = compute_new_value_i32(&base, &delta, 0).unwrap_err();
        assert!(err.to_string().contains("delta length"));
    }
}
