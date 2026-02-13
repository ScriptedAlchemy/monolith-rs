//! Helpers for merging lists of tensors (stack/concat/unstack) to mirror Python utils.

use crate::LayerError;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Merge behavior, mirroring the Python MergeType.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeType {
    Concat,
    Stack,
    None,
}

/// Output of merge utilities.
#[derive(Debug, Clone)]
pub enum MergeOutput {
    Tensor(Tensor),
    List(Vec<Tensor>),
}

/// Merge a list of tensors using the same semantics as Python `merge_tensor_list`.
pub fn merge_tensor_list(
    mut tensor_list: Vec<Tensor>,
    merge_type: MergeType,
    num_feature: Option<usize>,
    axis: usize,
    keep_list: bool,
) -> Result<MergeOutput, LayerError> {
    if tensor_list.is_empty() {
        return Ok(MergeOutput::List(Vec::new()));
    }

    if tensor_list.len() == 1 {
        let t = &tensor_list[0];
        let shape = t.shape();
        if shape.len() == 3 {
            let (batch_size, num_feat, emb_size) = (shape[0], shape[1], shape[2]);
            match merge_type {
                MergeType::Stack => {
                    let output = if keep_list {
                        MergeOutput::List(tensor_list)
                    } else {
                        MergeOutput::Tensor(t.clone())
                    };
                    return Ok(output);
                }
                MergeType::Concat => {
                    tensor_list[0] = t.reshape(&[batch_size, num_feat * emb_size]);
                    let output = if keep_list {
                        MergeOutput::List(tensor_list)
                    } else {
                        MergeOutput::Tensor(tensor_list[0].clone())
                    };
                    return Ok(output);
                }
                MergeType::None => {
                    let unstacked = t.unstack(axis);
                    return Ok(MergeOutput::List(unstacked));
                }
            }
        } else if shape.len() == 2 {
            if let Some(num_feature) = num_feature {
                if num_feature > 1 {
                    let (batch_size, emb_size) = (shape[0], shape[1]);
                    let per = emb_size / num_feature;
                    match merge_type {
                        MergeType::Stack => {
                            tensor_list[0] = t.reshape(&[batch_size, num_feature, per]);
                            let output = if keep_list {
                                MergeOutput::List(tensor_list)
                            } else {
                                MergeOutput::Tensor(tensor_list[0].clone())
                            };
                            return Ok(output);
                        }
                        MergeType::Concat => {
                            let output = if keep_list {
                                MergeOutput::List(tensor_list)
                            } else {
                                MergeOutput::Tensor(t.clone())
                            };
                            return Ok(output);
                        }
                        MergeType::None => {
                            let reshaped = t.reshape(&[batch_size, num_feature, per]);
                            let unstacked = reshaped.unstack(axis);
                            return Ok(MergeOutput::List(unstacked));
                        }
                    }
                }
            }

            let output = if keep_list {
                MergeOutput::List(tensor_list)
            } else {
                MergeOutput::Tensor(t.clone())
            };
            return Ok(output);
        } else {
            return Err(LayerError::ShapeMismatch {
                expected: vec![2, 3],
                actual: shape.to_vec(),
            });
        }
    }

    Ok(match merge_type {
        MergeType::Stack => {
            let stacked = Tensor::stack(&tensor_list, axis);
            if keep_list {
                MergeOutput::List(vec![stacked])
            } else {
                MergeOutput::Tensor(stacked)
            }
        }
        MergeType::Concat => {
            let concated = Tensor::cat(&tensor_list, axis);
            if keep_list {
                MergeOutput::List(vec![concated])
            } else {
                MergeOutput::Tensor(concated)
            }
        }
        MergeType::None => MergeOutput::List(tensor_list),
    })
}

/// Convenience helper that always returns a tensor (keep_list must be false).
pub fn merge_tensor_list_tensor(
    tensor_list: Vec<Tensor>,
    merge_type: MergeType,
    num_feature: Option<usize>,
    axis: usize,
) -> Result<Tensor, LayerError> {
    match merge_tensor_list(tensor_list, merge_type, num_feature, axis, false)? {
        MergeOutput::Tensor(t) => Ok(t),
        MergeOutput::List(mut list) => {
            if list.len() == 1 {
                Ok(list.remove(0))
            } else {
                Err(LayerError::ForwardError {
                    message: format!(
                        "merge_tensor_list_tensor expected a single tensor, got list of len {}",
                        list.len()
                    ),
                })
            }
        }
    }
}
