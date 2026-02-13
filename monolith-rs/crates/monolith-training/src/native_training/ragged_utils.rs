//! Ragged tensor helpers (TF-free parity for `monolith.native_training.ragged_utils`).
//!
//! The Python implementation caches `value_rowids` on the ragged object. In Rust we expose a
//! pure function that computes the row-ids from `row_splits`.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaggedUtilsError {
    EmptyRowSplits,
    NonMonotonicRowSplits {
        index: usize,
        start: usize,
        end: usize,
    },
}

impl std::fmt::Display for RaggedUtilsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRowSplits => write!(f, "row_splits must be non-empty (batch+1)"),
            Self::NonMonotonicRowSplits { index, start, end } => write!(
                f,
                "row_splits must be non-decreasing; row {index} has start={start} > end={end}"
            ),
        }
    }
}

impl std::error::Error for RaggedUtilsError {}

/// Compute fused `value_rowids` for a ragged tensor with `ragged_rank=1`.
///
/// `row_splits` has length `batch + 1`. For each row `i`, the values at indices
/// `[row_splits[i], row_splits[i+1])` belong to row `i`.
///
/// # Examples
///
/// ```
/// use monolith_training::native_training::ragged_utils::{
///     fused_value_rowids, RaggedUtilsError,
/// };
///
/// assert_eq!(fused_value_rowids(&[0, 0, 1, 3]), Ok(vec![1, 2, 2]));
/// assert!(matches!(
///     fused_value_rowids(&[]),
///     Err(RaggedUtilsError::EmptyRowSplits)
/// ));
/// assert!(matches!(
///     fused_value_rowids(&[0, 3, 2]),
///     Err(RaggedUtilsError::NonMonotonicRowSplits {
///         index: 1,
///         start: 3,
///         end: 2
///     })
/// ));
/// ```
pub fn fused_value_rowids(row_splits: &[usize]) -> Result<Vec<usize>, RaggedUtilsError> {
    if row_splits.is_empty() {
        return Err(RaggedUtilsError::EmptyRowSplits);
    }
    let batch = row_splits.len() - 1;
    let total = row_splits[batch];
    let mut out = Vec::with_capacity(total);
    for i in 0..batch {
        let start = row_splits[i];
        let end = row_splits[i + 1];
        if start > end {
            return Err(RaggedUtilsError::NonMonotonicRowSplits {
                index: i,
                start,
                end,
            });
        }
        let len = end - start;
        out.extend(std::iter::repeat(i).take(len));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Mirrors `ragged_utils_test.py`.
        // ragged: [[], [1], [2, 3]] => row_splits = [0, 0, 1, 3]
        let row_splits = [0usize, 0, 1, 3];
        let valueids =
            fused_value_rowids(&row_splits).expect("non-empty monotonic row_splits should succeed");
        assert_eq!(valueids, vec![1usize, 2, 2]);
    }

    #[test]
    fn test_empty_row_splits_returns_explicit_error() {
        let err =
            fused_value_rowids(&[]).expect_err("empty row_splits should return explicit error");
        assert!(
            matches!(err, RaggedUtilsError::EmptyRowSplits),
            "expected EmptyRowSplits for empty row_splits, got {err:?}"
        );
    }

    #[test]
    fn test_non_monotonic_row_splits_returns_explicit_error() {
        let row_splits = [0usize, 3, 2];
        let err = fused_value_rowids(&row_splits)
            .expect_err("decreasing row_splits should return explicit error");
        assert!(
            matches!(
                err,
                RaggedUtilsError::NonMonotonicRowSplits {
                    index: 1,
                    start: 3,
                    end: 2
                }
            ),
            "expected NonMonotonicRowSplits with detailed indices, got {err:?}"
        );
    }
}

