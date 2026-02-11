//! Sequence mask generation (Python parity for `monolith.native_training.gen_seq_mask`).

/// Generate a 2D mask of shape `[batch, max_len]` from a `row_splits` vector.
///
/// Python parity:
/// - `split` is a 1-D tensor of length `batch + 1`, starting with 0.
/// - Each row `i` has length `split[i + 1] - split[i]`.
/// - Output contains 1s for `j < row_len` and 0s otherwise.
pub fn gen_seq_mask_i64(row_splits: &[i64], max_len: usize) -> Vec<Vec<i64>> {
    assert!(
        row_splits.len() >= 1,
        "row_splits must be non-empty (batch+1)"
    );
    let batch = row_splits.len() - 1;
    let mut out = vec![vec![0i64; max_len]; batch];
    for i in 0..batch {
        let len = (row_splits[i + 1] - row_splits[i]).max(0) as usize;
        let upto = len.min(max_len);
        for j in 0..upto {
            out[i][j] = 1;
        }
    }
    out
}

/// Same as [`gen_seq_mask_i64`] but returns `i32` values (for Python's int32 path).
pub fn gen_seq_mask_i32(row_splits: &[i32], max_len: usize) -> Vec<Vec<i32>> {
    assert!(
        row_splits.len() >= 1,
        "row_splits must be non-empty (batch+1)"
    );
    let batch = row_splits.len() - 1;
    let mut out = vec![vec![0i32; max_len]; batch];
    for i in 0..batch {
        let len = (row_splits[i + 1] - row_splits[i]).max(0) as usize;
        let upto = len.min(max_len);
        for j in 0..upto {
            out[i][j] = 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_seq_mask_int32() {
        let split = [0i32, 5, 7, 9, 13];
        let mask = gen_seq_mask_i32(&split, 6);
        assert_eq!(
            mask,
            vec![
                vec![1, 1, 1, 1, 1, 0],
                vec![1, 1, 0, 0, 0, 0],
                vec![1, 1, 0, 0, 0, 0],
                vec![1, 1, 1, 1, 0, 0],
            ]
        );
    }

    #[test]
    fn test_gen_seq_mask_int64() {
        let split = [0i64, 5, 7, 9, 13];
        let mask = gen_seq_mask_i64(&split, 6);
        assert_eq!(
            mask,
            vec![
                vec![1, 1, 1, 1, 1, 0],
                vec![1, 1, 0, 0, 0, 0],
                vec![1, 1, 0, 0, 0, 0],
                vec![1, 1, 1, 1, 0, 0],
            ]
        );
    }
}

