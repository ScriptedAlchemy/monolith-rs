//! Constraints for learnable parameters.

use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;

/// Constraint types supported for layer parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum Constraint {
    /// No constraint.
    #[default]
    None,
    /// Constrain weights to be non-negative.
    NonNeg,
    /// Constrain weights to a min/max range.
    MinMax { min: f32, max: f32 },
    /// Constrain the norm along an axis to be at most `max_value`.
    MaxNorm { max_value: f32, axis: Option<usize> },
    /// Constrain the norm along an axis to be 1.0.
    UnitNorm { axis: Option<usize> },
}

impl Constraint {
    /// Applies the constraint and returns the constrained tensor.
    pub fn apply(&self, param: &Tensor) -> Tensor {
        match *self {
            Constraint::None => param.clone(),
            Constraint::NonNeg => {
                let zeros = Tensor::zeros(&[1]);
                param.maximum(&zeros)
            }
            Constraint::MinMax { min, max } => param.clamp(min, max),
            Constraint::MaxNorm { max_value, axis } => {
                let axis = axis.unwrap_or(0);
                Self::apply_norm_constraint(param, axis, Some(max_value))
            }
            Constraint::UnitNorm { axis } => {
                let axis = axis.unwrap_or(0);
                Self::apply_norm_constraint(param, axis, None)
            }
        }
    }

    fn apply_norm_constraint(param: &Tensor, axis: usize, max_value: Option<f32>) -> Tensor {
        let ndim = param.ndim();
        if ndim == 0 {
            return param.clone();
        }
        let axis = axis.min(ndim.saturating_sub(1));
        let eps = Tensor::from_data(&[1], vec![1e-7]);
        let norm = param.sqr().sum_axis(axis).sqrt().add(&eps);

        let mut shape = norm.shape().to_vec();
        shape.insert(axis, 1);
        let norm_b = norm.reshape(&shape).broadcast_as(param.shape());

        let scale = if let Some(max_value) = max_value {
            let max = Tensor::from_data(&[1], vec![max_value]);
            max.broadcast_as(norm_b.shape())
                .div(&norm_b)
                .clamp(0.0, 1.0)
        } else {
            let ones = Tensor::ones(&[1]);
            ones.broadcast_as(norm_b.shape()).div(&norm_b)
        };

        param.mul(&scale)
    }
}
