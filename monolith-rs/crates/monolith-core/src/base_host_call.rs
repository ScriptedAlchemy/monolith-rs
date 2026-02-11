//! Base host-call tensor packing utilities.
//!
//! Mirrors the TF-shape manipulation behavior in Python `monolith/core/base_host_call.py`:
//! - `record_summary_tensor`: registers 0D/1D tensors by name
//! - `compress_tensors`: groups tensors by dtype, concatenates, and adds batch dim
//! - `decompress_tensors`: splits compressed tensors back into per-metric tensors
//!
//! In Python, these utilities are used to reduce the number of host-call
//! arguments passed from TPU to host. Here we port the pure data logic so it
//! can be reused by Rust-side training loops.

use std::collections::BTreeMap;

use crate::error::{MonolithError, Result};

/// A simple "dtype-tagged" 1D tensor used for host-call packing.
#[derive(Clone, Debug, PartialEq)]
pub enum HostTensor1D {
    I64(Vec<i64>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl HostTensor1D {
    fn len(&self) -> usize {
        match self {
            HostTensor1D::I64(v) => v.len(),
            HostTensor1D::I32(v) => v.len(),
            HostTensor1D::F32(v) => v.len(),
            HostTensor1D::F64(v) => v.len(),
        }
    }

    fn dtype_key(&self) -> DTypeKey {
        match self {
            HostTensor1D::I64(_) => DTypeKey::I64,
            HostTensor1D::I32(_) => DTypeKey::I32,
            HostTensor1D::F32(_) => DTypeKey::F32,
            HostTensor1D::F64(_) => DTypeKey::F64,
        }
    }
}

/// A packed 2D tensor. Data is stored in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub enum HostTensor2D {
    I64 {
        rows: usize,
        cols: usize,
        data: Vec<i64>,
    },
    I32 {
        rows: usize,
        cols: usize,
        data: Vec<i32>,
    },
    F32 {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    },
    F64 {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
}

impl HostTensor2D {
    fn rows_cols(&self) -> (usize, usize) {
        match self {
            HostTensor2D::I64 { rows, cols, .. }
            | HostTensor2D::I32 { rows, cols, .. }
            | HostTensor2D::F32 { rows, cols, .. }
            | HostTensor2D::F64 { rows, cols, .. } => (*rows, *cols),
        }
    }
}

/// A decompressed tensor (after splitting).
#[derive(Clone, Debug, PartialEq)]
pub enum HostTensor {
    /// Rank-1 (after "squeeze" when `cols == 1`).
    I64(Vec<i64>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    /// Rank-2 (when `cols > 1`).
    I64Matrix {
        rows: usize,
        cols: usize,
        data: Vec<i64>,
    },
    I32Matrix {
        rows: usize,
        cols: usize,
        data: Vec<i32>,
    },
    F32Matrix {
        rows: usize,
        cols: usize,
        data: Vec<f32>,
    },
    F64Matrix {
        rows: usize,
        cols: usize,
        data: Vec<f64>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum DTypeKey {
    I64,
    I32,
    F32,
    F64,
}

/// Base host-call helper, holding names + tensors and compression metadata.
#[derive(Clone, Debug)]
pub struct BaseHostCall {
    enable_host_call: bool,
    tensor_names: Vec<String>,
    tensors: Vec<HostTensor1D>,
    lists_tensor_sizes: Vec<Vec<usize>>,
}

impl BaseHostCall {
    /// Creates a new BaseHostCall.
    pub fn new(enable_host_call: bool, global_step: i64) -> Self {
        Self {
            enable_host_call,
            tensor_names: vec!["global_step".to_string()],
            tensors: vec![HostTensor1D::I64(vec![global_step])],
            lists_tensor_sizes: Vec::new(),
        }
    }

    pub fn tensor_names(&self) -> &[String] {
        &self.tensor_names
    }

    pub fn tensors(&self) -> &[HostTensor1D] {
        &self.tensors
    }

    /// Records a named tensor. Only 0D/1D tensors are supported.
    pub fn record_summary_tensor(&mut self, name: &str, tensor: HostTensor1D) -> Result<()> {
        if !self.enable_host_call {
            return Ok(());
        }
        if self.tensor_names.iter().any(|n| n == name) {
            return Err(MonolithError::PyAssertionError {
                message: format!("{} already recorded", name),
            });
        }
        if tensor.len() == 0 {
            return Err(MonolithError::PyAssertionError {
                message: "Now we only support tensor with shape (k, ) or ()but we met tensor with shape: []"
                    .to_string(),
            });
        }
        self.tensor_names.push(name.to_string());
        self.tensors.push(tensor);
        Ok(())
    }

    /// Compresses current `tensors` grouped by dtype.
    ///
    /// After this call:
    /// - `tensor_names` becomes flattened metric names ordered by dtype group
    /// - `tensors` becomes a list of compressed tensors (one per dtype)
    pub fn compress_tensors(&mut self) -> Result<Vec<HostTensor2D>> {
        if self.tensor_names.len() != self.tensors.len() {
            return Err(MonolithError::PyAssertionError {
                message: format!(
                    "tensor_names and tensors must have same length,tensor_names length: {}, tensors length: {}",
                    self.tensor_names.len(),
                    self.tensors.len()
                ),
            });
        }

        let mut dtype_to_names: BTreeMap<DTypeKey, Vec<String>> = BTreeMap::new();
        let mut dtype_to_tensors: BTreeMap<DTypeKey, Vec<HostTensor1D>> = BTreeMap::new();

        for (name, t) in self.tensor_names.iter().zip(self.tensors.iter()) {
            dtype_to_names
                .entry(t.dtype_key())
                .or_default()
                .push(name.clone());
            dtype_to_tensors
                .entry(t.dtype_key())
                .or_default()
                .push(t.clone());
        }

        self.lists_tensor_sizes.clear();
        let mut compressed_names = Vec::new();
        let mut compressed = Vec::new();

        for (dtype, list) in dtype_to_tensors {
            compressed_names.extend(dtype_to_names.remove(&dtype).unwrap_or_default());

            let sizes = list.iter().map(|t| t.len()).collect::<Vec<_>>();
            self.lists_tensor_sizes.push(sizes);

            compressed.push(concat_expand(list)?);
        }

        self.tensor_names = compressed_names;
        // store as 1D in state (matches Python), but return 2D packed tensors for passing to host call.
        self.tensors = Vec::new();
        Ok(compressed)
    }

    /// Decompresses packed tensors back into per-metric tensors.
    ///
    /// Returns `(global_step, tensors)` where tensors include `global_step` at index 0.
    pub fn decompress_tensors(&self, tensors: &[HostTensor2D]) -> Result<(i64, Vec<HostTensor>)> {
        let mut out = Vec::new();
        for (index, packed) in tensors.iter().enumerate() {
            let (rows, cols) = packed.rows_cols();
            if rows == 0 || cols == 0 {
                return Err(MonolithError::PyAssertionError {
                    message: format!(
                        "Compressed tensors shape must be (n, m), met shape: ({rows}, {cols})"
                    ),
                });
            }
            let sizes =
                self.lists_tensor_sizes
                    .get(index)
                    .ok_or_else(|| MonolithError::InternalError {
                        message: "Missing tensor sizes metadata".to_string(),
                    })?;
            out.extend(split_squeeze(packed, sizes)?);
        }

        if self
            .tensor_names
            .first()
            .map(|s| s.as_str())
            .unwrap_or_default()
            != "global_step"
        {
            return Err(MonolithError::PyAssertionError {
                message: format!(
                    "The first tensor name must be global_step, met value: {}",
                    self.tensor_names.first().cloned().unwrap_or_default()
                ),
            });
        }

        let global_step = match out.first() {
            Some(HostTensor::I64(v)) => *v.get(0).unwrap_or(&0),
            Some(HostTensor::I64Matrix { data, .. }) => *data.get(0).unwrap_or(&0),
            _ => {
                return Err(MonolithError::InternalError {
                    message: "global_step missing or wrong dtype".to_string(),
                })
            }
        };
        Ok((global_step, out))
    }
}

fn concat_expand(list: Vec<HostTensor1D>) -> Result<HostTensor2D> {
    let dtype = list.first().ok_or_else(|| MonolithError::InternalError {
        message: "Empty tensor list".to_string(),
    })?;
    match dtype {
        HostTensor1D::I64(_) => {
            let mut data = Vec::new();
            for t in list {
                if let HostTensor1D::I64(v) = t {
                    data.extend(v);
                } else {
                    return Err(MonolithError::InternalError {
                        message: "dtype mismatch".to_string(),
                    });
                }
            }
            Ok(HostTensor2D::I64 {
                rows: 1,
                cols: data.len(),
                data,
            })
        }
        HostTensor1D::I32(_) => {
            let mut data = Vec::new();
            for t in list {
                if let HostTensor1D::I32(v) = t {
                    data.extend(v);
                } else {
                    return Err(MonolithError::InternalError {
                        message: "dtype mismatch".to_string(),
                    });
                }
            }
            Ok(HostTensor2D::I32 {
                rows: 1,
                cols: data.len(),
                data,
            })
        }
        HostTensor1D::F32(_) => {
            let mut data = Vec::new();
            for t in list {
                if let HostTensor1D::F32(v) = t {
                    data.extend(v);
                } else {
                    return Err(MonolithError::InternalError {
                        message: "dtype mismatch".to_string(),
                    });
                }
            }
            Ok(HostTensor2D::F32 {
                rows: 1,
                cols: data.len(),
                data,
            })
        }
        HostTensor1D::F64(_) => {
            let mut data = Vec::new();
            for t in list {
                if let HostTensor1D::F64(v) = t {
                    data.extend(v);
                } else {
                    return Err(MonolithError::InternalError {
                        message: "dtype mismatch".to_string(),
                    });
                }
            }
            Ok(HostTensor2D::F64 {
                rows: 1,
                cols: data.len(),
                data,
            })
        }
    }
}

fn split_squeeze(packed: &HostTensor2D, sizes: &[usize]) -> Result<Vec<HostTensor>> {
    let (rows, cols) = packed.rows_cols();
    let sum: usize = sizes.iter().sum();
    if sum != cols {
        return Err(MonolithError::InternalError {
            message: format!("Split sizes sum ({sum}) != packed cols ({cols})"),
        });
    }

    match packed {
        HostTensor2D::I64 { data, .. } => Ok(split_squeeze_typed(rows, cols, data, sizes)
            .into_iter()
            .map(|t| match t {
                SplitTyped::Vec(v) => HostTensor::I64(v),
                SplitTyped::Mat { rows, cols, data } => HostTensor::I64Matrix { rows, cols, data },
            })
            .collect()),
        HostTensor2D::I32 { data, .. } => Ok(split_squeeze_typed(rows, cols, data, sizes)
            .into_iter()
            .map(|t| match t {
                SplitTyped::Vec(v) => HostTensor::I32(v),
                SplitTyped::Mat { rows, cols, data } => HostTensor::I32Matrix { rows, cols, data },
            })
            .collect()),
        HostTensor2D::F32 { data, .. } => Ok(split_squeeze_typed(rows, cols, data, sizes)
            .into_iter()
            .map(|t| match t {
                SplitTyped::Vec(v) => HostTensor::F32(v),
                SplitTyped::Mat { rows, cols, data } => HostTensor::F32Matrix { rows, cols, data },
            })
            .collect()),
        HostTensor2D::F64 { data, .. } => Ok(split_squeeze_typed(rows, cols, data, sizes)
            .into_iter()
            .map(|t| match t {
                SplitTyped::Vec(v) => HostTensor::F64(v),
                SplitTyped::Mat { rows, cols, data } => HostTensor::F64Matrix { rows, cols, data },
            })
            .collect()),
    }
}

enum SplitTyped<T> {
    Vec(Vec<T>),
    Mat {
        rows: usize,
        cols: usize,
        data: Vec<T>,
    },
}

fn split_squeeze_typed<T: Copy>(
    rows: usize,
    cols: usize,
    data: &[T],
    sizes: &[usize],
) -> Vec<SplitTyped<T>> {
    let mut out = Vec::with_capacity(sizes.len());
    let mut offset = 0;
    for &k in sizes {
        // Extract a (rows, k) slice from the (rows, cols) packed data.
        let mut chunk = Vec::with_capacity(rows * k);
        for r in 0..rows {
            let base = r * cols + offset;
            chunk.extend_from_slice(&data[base..base + k]);
        }
        if k == 1 {
            // Squeeze: (rows, 1) -> (rows,)
            out.push(SplitTyped::Vec(
                (0..rows).map(|r| chunk[r]).collect::<Vec<_>>(),
            ));
        } else {
            out.push(SplitTyped::Mat {
                rows,
                cols: k,
                data: chunk,
            });
        }
        offset += k;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack_rows_i64(rows: Vec<Vec<i64>>) -> HostTensor2D {
        let r = rows.len();
        let c = rows.first().map(|v| v.len()).unwrap_or(0);
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            assert_eq!(row.len(), c);
            data.extend(row);
        }
        HostTensor2D::I64 {
            rows: r,
            cols: c,
            data,
        }
    }

    #[test]
    fn test_compress_decompress_roundtrip() {
        let mut hc = BaseHostCall::new(true, 7);
        hc.record_summary_tensor("a", HostTensor1D::I32(vec![1, 2, 3]))
            .unwrap();
        hc.record_summary_tensor("b", HostTensor1D::I32(vec![4]))
            .unwrap();
        hc.record_summary_tensor("c", HostTensor1D::F32(vec![0.5]))
            .unwrap();

        // Compress on a single core.
        let packed = hc.compress_tensors().unwrap();
        assert_eq!(
            hc.tensor_names(),
            &[
                "global_step".to_string(),
                "a".to_string(),
                "b".to_string(),
                "c".to_string()
            ]
        );

        // Simulate concatenation across 2 TPU cores for the first packed tensor (I64)
        // and the next ones. For this test we only have one I64 packed tensor (global_step),
        // and one I32 packed tensor, and one F32 packed tensor.
        assert_eq!(packed.len(), 3);

        let packed_i64 = match &packed[0] {
            HostTensor2D::I64 { data, cols, .. } => (data.clone(), *cols),
            _ => panic!("expected i64 packed"),
        };
        let packed_i32 = match &packed[1] {
            HostTensor2D::I32 { data, cols, .. } => (data.clone(), *cols),
            _ => panic!("expected i32 packed"),
        };
        let packed_f32 = match &packed[2] {
            HostTensor2D::F32 { data, cols, .. } => (data.clone(), *cols),
            _ => panic!("expected f32 packed"),
        };

        let args = vec![
            stack_rows_i64(vec![packed_i64.0.clone(), packed_i64.0]),
            HostTensor2D::I32 {
                rows: 2,
                cols: packed_i32.1,
                data: [packed_i32.0.clone(), packed_i32.0].concat(),
            },
            HostTensor2D::F32 {
                rows: 2,
                cols: packed_f32.1,
                data: [packed_f32.0.clone(), packed_f32.0].concat(),
            },
        ];

        let (gs, tensors) = hc.decompress_tensors(&args).unwrap();
        assert_eq!(gs, 7);

        // global_step: squeezed (rows,1) -> (rows,)
        assert_eq!(tensors[0], HostTensor::I64(vec![7, 7]));

        // a: (rows,3)
        assert_eq!(
            tensors[1],
            HostTensor::I32Matrix {
                rows: 2,
                cols: 3,
                data: vec![1, 2, 3, 1, 2, 3]
            }
        );
        // b: squeezed (rows,1)
        assert_eq!(tensors[2], HostTensor::I32(vec![4, 4]));
        // c: squeezed (rows,1)
        assert_eq!(tensors[3], HostTensor::F32(vec![0.5, 0.5]));
    }
}
