//! Shared activation layer wrapper for reuse outside MLP.

use crate::activation::{
    ELU, Exponential, GELU, HardSigmoid, LeakyReLU, Linear, Mish, PReLU, ReLU, SELU, Sigmoid,
    Sigmoid2, Softmax, Softplus, Softsign, Swish, Tanh, ThresholdedReLU,
};
use crate::error::LayerError;
use crate::layer::Layer;
use crate::mlp::ActivationType;
use crate::tensor::Tensor;

/// Public activation layer wrapper to share forward/backward implementations.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub enum ActivationLayer {
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Sigmoid2(Sigmoid2),
    Tanh(Tanh),
    GELU(GELU),
    SELU(SELU),
    Softplus(Softplus),
    Softsign(Softsign),
    Swish(Swish),
    Mish(Mish),
    HardSigmoid(HardSigmoid),
    LeakyReLU(LeakyReLU),
    ELU(ELU),
    PReLU(PReLU),
    ThresholdedReLU(ThresholdedReLU),
    Softmax(Softmax),
    Linear(Linear),
    Exponential(Exponential),
    None,
}

impl ActivationLayer {
    pub fn from_activation_type(activation: ActivationType) -> Self {
        match activation {
            ActivationType::ReLU {
                max_value,
                negative_slope,
                threshold,
            } => ActivationLayer::ReLU(ReLU::with_params(max_value, negative_slope, threshold)),
            ActivationType::Sigmoid => ActivationLayer::Sigmoid(Sigmoid::new()),
            ActivationType::Sigmoid2 => ActivationLayer::Sigmoid2(Sigmoid2::new()),
            ActivationType::Tanh => ActivationLayer::Tanh(Tanh::new()),
            ActivationType::GELU => ActivationLayer::GELU(GELU::new()),
            ActivationType::SELU => ActivationLayer::SELU(SELU::new()),
            ActivationType::Softplus => ActivationLayer::Softplus(Softplus::new()),
            ActivationType::Softsign => ActivationLayer::Softsign(Softsign::new()),
            ActivationType::Swish => ActivationLayer::Swish(Swish::new()),
            ActivationType::Mish => ActivationLayer::Mish(Mish::new()),
            ActivationType::HardSigmoid => ActivationLayer::HardSigmoid(HardSigmoid::new()),
            ActivationType::LeakyReLU { alpha } => ActivationLayer::LeakyReLU(LeakyReLU::new(alpha)),
            ActivationType::ELU { alpha } => ActivationLayer::ELU(ELU::new(alpha)),
            ActivationType::PReLU {
                alpha,
                initializer,
                shared_axes,
                regularizer,
                constraint,
            } => ActivationLayer::PReLU(PReLU::with_params(
                alpha,
                initializer,
                shared_axes,
                regularizer,
                constraint,
            )),
            ActivationType::ThresholdedReLU { theta } => {
                ActivationLayer::ThresholdedReLU(ThresholdedReLU::new(theta))
            }
            ActivationType::Softmax { axis } => ActivationLayer::Softmax(Softmax::with_axis(axis)),
            ActivationType::Linear => ActivationLayer::Linear(Linear::new()),
            ActivationType::Exponential => ActivationLayer::Exponential(Exponential::new()),
            ActivationType::None => ActivationLayer::None,
        }
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            Self::ReLU(a) => a.forward(input),
            Self::Sigmoid(a) => a.forward(input),
            Self::Sigmoid2(a) => a.forward(input),
            Self::Tanh(a) => a.forward(input),
            Self::GELU(a) => a.forward(input),
            Self::SELU(a) => a.forward(input),
            Self::Softplus(a) => a.forward(input),
            Self::Softsign(a) => a.forward(input),
            Self::Swish(a) => a.forward(input),
            Self::Mish(a) => a.forward(input),
            Self::HardSigmoid(a) => a.forward(input),
            Self::LeakyReLU(a) => a.forward(input),
            Self::ELU(a) => a.forward(input),
            Self::PReLU(a) => a.forward(input),
            Self::ThresholdedReLU(a) => a.forward(input),
            Self::Softmax(a) => a.forward(input),
            Self::Linear(a) => a.forward(input),
            Self::Exponential(a) => a.forward(input),
            Self::None => Ok(input.clone()),
        }
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            Self::ReLU(a) => a.forward_train(input),
            Self::Sigmoid(a) => a.forward_train(input),
            Self::Sigmoid2(a) => a.forward_train(input),
            Self::Tanh(a) => a.forward_train(input),
            Self::GELU(a) => a.forward_train(input),
            Self::SELU(a) => a.forward_train(input),
            Self::Softplus(a) => a.forward_train(input),
            Self::Softsign(a) => a.forward_train(input),
            Self::Swish(a) => a.forward_train(input),
            Self::Mish(a) => a.forward_train(input),
            Self::HardSigmoid(a) => a.forward_train(input),
            Self::LeakyReLU(a) => a.forward_train(input),
            Self::ELU(a) => a.forward_train(input),
            Self::PReLU(a) => a.forward_train(input),
            Self::ThresholdedReLU(a) => a.forward_train(input),
            Self::Softmax(a) => a.forward_train(input),
            Self::Linear(a) => a.forward_train(input),
            Self::Exponential(a) => a.forward_train(input),
            Self::None => Ok(input.clone()),
        }
    }

    pub fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            Self::ReLU(a) => a.backward(grad),
            Self::Sigmoid(a) => a.backward(grad),
            Self::Sigmoid2(a) => a.backward(grad),
            Self::Tanh(a) => a.backward(grad),
            Self::GELU(a) => a.backward(grad),
            Self::SELU(a) => a.backward(grad),
            Self::Softplus(a) => a.backward(grad),
            Self::Softsign(a) => a.backward(grad),
            Self::Swish(a) => a.backward(grad),
            Self::Mish(a) => a.backward(grad),
            Self::HardSigmoid(a) => a.backward(grad),
            Self::LeakyReLU(a) => a.backward(grad),
            Self::ELU(a) => a.backward(grad),
            Self::PReLU(a) => a.backward(grad),
            Self::ThresholdedReLU(a) => a.backward(grad),
            Self::Softmax(a) => a.backward(grad),
            Self::Linear(a) => a.backward(grad),
            Self::Exponential(a) => a.backward(grad),
            Self::None => Ok(grad.clone()),
        }
    }
}
