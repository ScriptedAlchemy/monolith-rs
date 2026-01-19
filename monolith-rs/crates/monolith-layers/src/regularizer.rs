//! Regularization utilities for learnable parameters.

use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;

/// Regularizer types supported for layer parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub enum Regularizer {
    /// No regularization.
    #[default]
    None,
    /// L1 regularization with coefficient.
    L1(f32),
    /// L2 regularization with coefficient.
    L2(f32),
    /// Combined L1 + L2 regularization.
    L1L2 { l1: f32, l2: f32 },
}

impl Regularizer {
    /// Returns the regularization loss for the given parameter tensor.
    pub fn loss(&self, param: &Tensor) -> f32 {
        match *self {
            Regularizer::None => 0.0,
            Regularizer::L1(lambda) => param.abs().sum() * lambda,
            Regularizer::L2(lambda) => param.sqr().sum() * lambda,
            Regularizer::L1L2 { l1, l2 } => param.abs().sum() * l1 + param.sqr().sum() * l2,
        }
    }

    /// Returns the gradient contribution of this regularizer for the given parameter.
    pub fn grad(&self, param: &Tensor) -> Option<Tensor> {
        match *self {
            Regularizer::None => None,
            Regularizer::L1(lambda) => {
                let pos = param.gt_scalar(0.0);
                let neg = param.lt_scalar(0.0);
                let sign = pos.sub(&neg);
                Some(sign.scale(lambda))
            }
            Regularizer::L2(lambda) => Some(param.scale(2.0 * lambda)),
            Regularizer::L1L2 { l1, l2 } => {
                let pos = param.gt_scalar(0.0);
                let neg = param.lt_scalar(0.0);
                let sign = pos.sub(&neg);
                let l1_grad = sign.scale(l1);
                let l2_grad = param.scale(2.0 * l2);
                Some(l1_grad.add(&l2_grad))
            }
        }
    }
}
