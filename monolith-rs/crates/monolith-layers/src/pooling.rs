//! Pooling helpers for lists of tensors.

use crate::error::LayerError;
use crate::tensor::Tensor;

/// Trait for pooling a list of tensors.
pub trait Pooling {
    fn pool(&self, inputs: &[Tensor]) -> Result<Tensor, LayerError>;
}

/// Sum pooling.
#[derive(Debug, Clone, Copy, Default)]
pub struct SumPooling;

impl Pooling for SumPooling {
    fn pool(&self, inputs: &[Tensor]) -> Result<Tensor, LayerError> {
        if inputs.is_empty() {
            return Err(LayerError::ForwardError {
                message: "SumPooling expects non-empty input list".to_string(),
            });
        }
        let mut out = inputs[0].clone();
        for t in &inputs[1..] {
            out = out.add(t);
        }
        Ok(out)
    }
}

/// Average pooling.
#[derive(Debug, Clone, Copy, Default)]
pub struct AvgPooling;

impl Pooling for AvgPooling {
    fn pool(&self, inputs: &[Tensor]) -> Result<Tensor, LayerError> {
        if inputs.is_empty() {
            return Err(LayerError::ForwardError {
                message: "AvgPooling expects non-empty input list".to_string(),
            });
        }
        let mut out = inputs[0].clone();
        for t in &inputs[1..] {
            out = out.add(t);
        }
        Ok(out.scale(1.0 / inputs.len() as f32))
    }
}

/// Max pooling.
#[derive(Debug, Clone, Copy, Default)]
pub struct MaxPooling;

impl Pooling for MaxPooling {
    fn pool(&self, inputs: &[Tensor]) -> Result<Tensor, LayerError> {
        if inputs.is_empty() {
            return Err(LayerError::ForwardError {
                message: "MaxPooling expects non-empty input list".to_string(),
            });
        }
        let mut out = inputs[0].clone();
        for t in &inputs[1..] {
            out = out.maximum(t);
        }
        Ok(out)
    }
}
