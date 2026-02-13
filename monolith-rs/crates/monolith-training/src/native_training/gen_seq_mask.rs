//! Sequence mask generation (Python parity for `monolith.native_training.gen_seq_mask`).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenSeqMaskError {
    EmptyRowSplits,
}

impl std::fmt::Display for GenSeqMaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyRowSplits => write!(f, "row_splits must be non-empty (batch+1)"),
        }
    }
}

impl std::error::Error for GenSeqMaskError {}

/// Generate a 2D mask of shape `[batch, max_len]` from a `row_splits` vector.
///
/// Python parity:
/// - `split` is a 1-D tensor of length `batch + 1`, starting with 0.
/// - Each row `i` has length `split[i + 1] - split[i]`.
/// - Output contains 1s for `j < row_len` and 0s otherwise.
///
/// # Examples
///
/// ```
/// use monolith_training::native_training::gen_seq_mask::{gen_seq_mask_i64, GenSeqMaskError};
///
/// let mask = gen_seq_mask_i64(&[0, 2, 3], 3)
///     .expect("non-empty row_splits should generate sequence mask");
/// assert_eq!(mask, vec![vec![1, 1, 0], vec![1, 0, 0]]);
///
/// assert!(matches!(
///     gen_seq_mask_i64(&[], 3),
///     Err(GenSeqMaskError::EmptyRowSplits)
/// ));
/// ```
pub fn gen_seq_mask_i64(
    row_splits: &[i64],
    max_len: usize,
) -> Result<Vec<Vec<i64>>, GenSeqMaskError> {
    if row_splits.is_empty() {
        return Err(GenSeqMaskError::EmptyRowSplits);
    }
    let batch = row_splits.len() - 1;
    let mut out = vec![vec![0i64; max_len]; batch];
    for i in 0..batch {
        let len = (row_splits[i + 1] - row_splits[i]).max(0) as usize;
        let upto = len.min(max_len);
        for j in 0..upto {
            out[i][j] = 1;
        }
    }
    Ok(out)
}

/// Same as [`gen_seq_mask_i64`] but returns `i32` values (for Python's int32 path).
///
/// # Examples
///
/// ```
/// use monolith_training::native_training::gen_seq_mask::{gen_seq_mask_i32, GenSeqMaskError};
///
/// let mask = gen_seq_mask_i32(&[0, 1, 3], 3)
///     .expect("non-empty row_splits should generate sequence mask");
/// assert_eq!(mask, vec![vec![1, 0, 0], vec![1, 1, 0]]);
///
/// assert!(matches!(
///     gen_seq_mask_i32(&[], 3),
///     Err(GenSeqMaskError::EmptyRowSplits)
/// ));
/// ```
pub fn gen_seq_mask_i32(
    row_splits: &[i32],
    max_len: usize,
) -> Result<Vec<Vec<i32>>, GenSeqMaskError> {
    if row_splits.is_empty() {
        return Err(GenSeqMaskError::EmptyRowSplits);
    }
    let batch = row_splits.len() - 1;
    let mut out = vec![vec![0i32; max_len]; batch];
    for i in 0..batch {
        let len = (row_splits[i + 1] - row_splits[i]).max(0) as usize;
        let upto = len.min(max_len);
        for j in 0..upto {
            out[i][j] = 1;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_seq_mask_int32() {
        let split = [0i32, 5, 7, 9, 13];
        let mask = gen_seq_mask_i32(&split, 6).expect("non-empty row_splits should succeed");
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
        let mask = gen_seq_mask_i64(&split, 6).expect("non-empty row_splits should succeed");
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
    fn test_gen_seq_mask_rejects_empty_row_splits() {
        let err_i32 =
            gen_seq_mask_i32(&[], 6).expect_err("empty row_splits should return explicit error");
        assert!(
            matches!(err_i32, GenSeqMaskError::EmptyRowSplits),
            "expected EmptyRowSplits for i32 path, got {err_i32:?}"
        );

        let err_i64 =
            gen_seq_mask_i64(&[], 6).expect_err("empty row_splits should return explicit error");
        assert!(
            matches!(err_i64, GenSeqMaskError::EmptyRowSplits),
            "expected EmptyRowSplits for i64 path, got {err_i64:?}"
        );
    }

    #[test]
    fn test_gen_seq_mask_negative_deltas_are_clamped_to_zero() {
        let split_i32 = [0i32, -2, 1];
        let mask_i32 =
            gen_seq_mask_i32(&split_i32, 4).expect("non-empty row_splits should succeed");
        assert_eq!(
            mask_i32,
            vec![vec![0, 0, 0, 0], vec![1, 1, 1, 0]],
            "negative row-length deltas should clamp to zero for i32 path"
        );

        let split_i64 = [0i64, -2, 1];
        let mask_i64 =
            gen_seq_mask_i64(&split_i64, 4).expect("non-empty row_splits should succeed");
        assert_eq!(
            mask_i64,
            vec![vec![0, 0, 0, 0], vec![1, 1, 1, 0]],
            "negative row-length deltas should clamp to zero for i64 path"
        );
    }
}

