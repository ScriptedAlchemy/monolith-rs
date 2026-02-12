//! Ragged tensor helpers (TF-free parity for `monolith.native_training.ragged_utils`).
//!
//! The Python implementation caches `value_rowids` on the ragged object. In Rust we expose a
//! pure function that computes the row-ids from `row_splits`.

/// Compute fused `value_rowids` for a ragged tensor with `ragged_rank=1`.
///
/// `row_splits` has length `batch + 1`. For each row `i`, the values at indices
/// `[row_splits[i], row_splits[i+1])` belong to row `i`.
pub fn fused_value_rowids(row_splits: &[usize]) -> Vec<usize> {
    assert!(
        row_splits.len() >= 1,
        "row_splits must be non-empty (batch+1)"
    );
    let batch = row_splits.len() - 1;
    let total = *row_splits
        .last()
        .expect("row_splits should have at least one element");
    let mut out = Vec::with_capacity(total);
    for i in 0..batch {
        let start = row_splits[i];
        let end = row_splits[i + 1];
        assert!(start <= end, "row_splits must be non-decreasing");
        let len = end - start;
        out.extend(std::iter::repeat(i).take(len));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        // Mirrors `ragged_utils_test.py`.
        // ragged: [[], [1], [2, 3]] => row_splits = [0, 0, 1, 3]
        let row_splits = [0usize, 0, 1, 3];
        let valueids = fused_value_rowids(&row_splits);
        assert_eq!(valueids, vec![1usize, 2, 2]);
    }
}

