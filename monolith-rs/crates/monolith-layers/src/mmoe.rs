//! Multi-gate Mixture of Experts (MMoE) layer implementation.
//!
//! This module provides the [`MMoE`] layer, which implements the Multi-gate Mixture
//! of Experts architecture for multi-task learning. MMoE uses a shared set of expert
//! networks and task-specific gating networks to learn task-specific combinations
//! of expert outputs.
//!
//! # Architecture
//!
//! The MMoE layer consists of:
//! - **Experts**: A set of shared MLP networks that process the input
//! - **Gates**: Per-task gating networks that compute softmax weights over experts
//! - **Output**: For each task, a weighted sum of expert outputs based on gate weights
//!
//! # Example
//!
//! ```
//! use monolith_layers::mmoe::{MMoE, MMoEConfig};
//! use monolith_layers::mlp::ActivationType;
//! use monolith_layers::tensor::Tensor;
//!
//! // Create an MMoE layer with 3 experts and 2 tasks
//! let config = MMoEConfig::new(64, 3, 2)
//!     .with_expert_hidden_units(vec![32, 32])
//!     .with_expert_activation(ActivationType::relu());
//!
//! let mmoe = MMoE::from_config(config).unwrap();
//!
//! // Forward pass
//! let input = Tensor::rand(&[8, 64]);  // batch of 8
//! let outputs = mmoe.forward_multi(&input).unwrap();
//! assert_eq!(outputs.len(), 2);  // one output per task
//! ```
//!
//! # References
//!
//! Ma, J., Zhao, Z., Yi, X., Chen, J., Hong, L., & Chi, E. H. (2018).
//! Modeling task relationships in multi-task learning with multi-gate mixture-of-experts.
//! In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery
//! & Data Mining (pp. 1930-1939).

use crate::activation::{
    ELU, Exponential, GELU, HardSigmoid, LeakyReLU, Linear, Mish, PReLU, ReLU, SELU, Sigmoid,
    Sigmoid2, Softmax, Softplus, Softsign, Swish, Tanh, ThresholdedReLU,
};
use crate::dense::Dense;
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::mlp::ActivationType;
use crate::normalization::BatchNorm;
use crate::constraint::Constraint;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// A single expert network in the MMoE layer.
///
/// An expert is a multi-layer perceptron (MLP) that processes the input and
/// produces a transformed output. All experts in an MMoE layer share the same
/// architecture but have different learned parameters.
///
/// # Example
///
/// ```
/// use monolith_layers::mmoe::Expert;
/// use monolith_layers::mlp::ActivationType;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let expert = Expert::new(64, &[32], 32, ActivationType::relu());
/// let input = Tensor::rand(&[8, 64]);
/// let output = expert.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[8, 32]);
/// ```
#[derive(Debug, Clone)]
pub struct Expert {
    /// Dense layers of the expert network
    dense_layers: Vec<Dense>,
    /// Optional batch norm on input
    input_batch_norm: Option<BatchNorm>,
    /// Optional batch norm after each dense layer (except last when disabled)
    batch_norms: Vec<Option<BatchNorm>>,
    /// Activation functions for each layer
    activations: Vec<ActivationWrapper>,
    /// Input dimension
    input_dim: usize,
    /// Output dimension
    output_dim: usize,
    /// Cached inputs for backward pass
    cached_inputs: Vec<Tensor>,
    /// Whether in training mode
    training: bool,
}

/// Internal wrapper for activation functions.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
enum ActivationWrapper {
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

impl ActivationWrapper {
    fn from_type(activation_type: ActivationType) -> Self {
        match activation_type {
            ActivationType::ReLU {
                max_value,
                negative_slope,
                threshold,
            } => ActivationWrapper::ReLU(ReLU::with_params(
                max_value,
                negative_slope,
                threshold,
            )),
            ActivationType::Sigmoid => ActivationWrapper::Sigmoid(Sigmoid::new()),
            ActivationType::Sigmoid2 => ActivationWrapper::Sigmoid2(Sigmoid2::new()),
            ActivationType::Tanh => ActivationWrapper::Tanh(Tanh::new()),
            ActivationType::GELU => ActivationWrapper::GELU(GELU::new()),
            ActivationType::SELU => ActivationWrapper::SELU(SELU::new()),
            ActivationType::Softplus => ActivationWrapper::Softplus(Softplus::new()),
            ActivationType::Softsign => ActivationWrapper::Softsign(Softsign::new()),
            ActivationType::Swish => ActivationWrapper::Swish(Swish::new()),
            ActivationType::Mish => ActivationWrapper::Mish(Mish::new()),
            ActivationType::HardSigmoid => ActivationWrapper::HardSigmoid(HardSigmoid::new()),
            ActivationType::LeakyReLU { alpha } => {
                ActivationWrapper::LeakyReLU(LeakyReLU::new(alpha))
            }
            ActivationType::ELU { alpha } => ActivationWrapper::ELU(ELU::new(alpha)),
            ActivationType::PReLU {
                alpha,
                initializer,
                shared_axes,
                regularizer,
                constraint,
            } => ActivationWrapper::PReLU(PReLU::with_params(
                alpha,
                initializer,
                shared_axes,
                regularizer,
                constraint,
            )),
            ActivationType::ThresholdedReLU { theta } => {
                ActivationWrapper::ThresholdedReLU(ThresholdedReLU::new(theta))
            }
            ActivationType::Softmax { axis } => {
                ActivationWrapper::Softmax(Softmax::with_axis(axis))
            }
            ActivationType::Linear => ActivationWrapper::Linear(Linear::new()),
            ActivationType::Exponential => ActivationWrapper::Exponential(Exponential::new()),
            ActivationType::None => ActivationWrapper::None,
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            ActivationWrapper::ReLU(a) => a.forward(input),
            ActivationWrapper::Sigmoid(a) => a.forward(input),
            ActivationWrapper::Sigmoid2(a) => a.forward(input),
            ActivationWrapper::Tanh(a) => a.forward(input),
            ActivationWrapper::GELU(a) => a.forward(input),
            ActivationWrapper::SELU(a) => a.forward(input),
            ActivationWrapper::Softplus(a) => a.forward(input),
            ActivationWrapper::Softsign(a) => a.forward(input),
            ActivationWrapper::Swish(a) => a.forward(input),
            ActivationWrapper::Mish(a) => a.forward(input),
            ActivationWrapper::HardSigmoid(a) => a.forward(input),
            ActivationWrapper::LeakyReLU(a) => a.forward(input),
            ActivationWrapper::ELU(a) => a.forward(input),
            ActivationWrapper::PReLU(a) => a.forward(input),
            ActivationWrapper::ThresholdedReLU(a) => a.forward(input),
            ActivationWrapper::Softmax(a) => a.forward(input),
            ActivationWrapper::Linear(a) => a.forward(input),
            ActivationWrapper::Exponential(a) => a.forward(input),
            ActivationWrapper::None => Ok(input.clone()),
        }
    }

    fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            ActivationWrapper::ReLU(a) => a.forward_train(input),
            ActivationWrapper::Sigmoid(a) => a.forward_train(input),
            ActivationWrapper::Sigmoid2(a) => a.forward_train(input),
            ActivationWrapper::Tanh(a) => a.forward_train(input),
            ActivationWrapper::GELU(a) => a.forward_train(input),
            ActivationWrapper::SELU(a) => a.forward_train(input),
            ActivationWrapper::Softplus(a) => a.forward_train(input),
            ActivationWrapper::Softsign(a) => a.forward_train(input),
            ActivationWrapper::Swish(a) => a.forward_train(input),
            ActivationWrapper::Mish(a) => a.forward_train(input),
            ActivationWrapper::HardSigmoid(a) => a.forward_train(input),
            ActivationWrapper::LeakyReLU(a) => a.forward_train(input),
            ActivationWrapper::ELU(a) => a.forward_train(input),
            ActivationWrapper::PReLU(a) => a.forward_train(input),
            ActivationWrapper::ThresholdedReLU(a) => a.forward_train(input),
            ActivationWrapper::Softmax(a) => a.forward_train(input),
            ActivationWrapper::Linear(a) => a.forward_train(input),
            ActivationWrapper::Exponential(a) => a.forward_train(input),
            ActivationWrapper::None => Ok(input.clone()),
        }
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            ActivationWrapper::ReLU(a) => a.backward(grad),
            ActivationWrapper::Sigmoid(a) => a.backward(grad),
            ActivationWrapper::Sigmoid2(a) => a.backward(grad),
            ActivationWrapper::Tanh(a) => a.backward(grad),
            ActivationWrapper::GELU(a) => a.backward(grad),
            ActivationWrapper::SELU(a) => a.backward(grad),
            ActivationWrapper::Softplus(a) => a.backward(grad),
            ActivationWrapper::Softsign(a) => a.backward(grad),
            ActivationWrapper::Swish(a) => a.backward(grad),
            ActivationWrapper::Mish(a) => a.backward(grad),
            ActivationWrapper::HardSigmoid(a) => a.backward(grad),
            ActivationWrapper::LeakyReLU(a) => a.backward(grad),
            ActivationWrapper::ELU(a) => a.backward(grad),
            ActivationWrapper::PReLU(a) => a.backward(grad),
            ActivationWrapper::ThresholdedReLU(a) => a.backward(grad),
            ActivationWrapper::Softmax(a) => a.backward(grad),
            ActivationWrapper::Linear(a) => a.backward(grad),
            ActivationWrapper::Exponential(a) => a.backward(grad),
            ActivationWrapper::None => Ok(grad.clone()),
        }
    }
}

/// Gate types supported by MMoE.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    Softmax,
    TopK,
    NoiseTopK,
}

impl Expert {
    /// Creates a new expert network.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `hidden_units` - Sizes of hidden layers
    /// * `output_dim` - Output dimension
    /// * `activation` - Activation function for hidden layers
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::mmoe::Expert;
    /// use monolith_layers::mlp::ActivationType;
    ///
    /// // Create an expert with input dim 64, hidden layers [32, 32], and output dim 16
    /// let expert = Expert::new(64, &[32, 32], 16, ActivationType::relu());
    /// ```
    pub fn new(
        input_dim: usize,
        hidden_units: &[usize],
        output_dim: usize,
        activation: ActivationType,
    ) -> Self {
        Self::new_with_options(
            input_dim,
            hidden_units,
            output_dim,
            activation,
            Initializer::GlorotUniform,
            true,
            Regularizer::None,
            Regularizer::None,
            Constraint::None,
            Constraint::None,
            false,
            false,
            false,
            0.99,
            false,
            None,
            0.99,
        )
    }

    /// Creates a new expert network with extended options.
    pub fn new_with_options(
        input_dim: usize,
        hidden_units: &[usize],
        output_dim: usize,
        activation: ActivationType,
        initializer: Initializer,
        use_bias: bool,
        kernel_regularizer: Regularizer,
        bias_regularizer: Regularizer,
        kernel_constraint: Constraint,
        bias_constraint: Constraint,
        use_weight_norm: bool,
        use_learnable_weight_norm: bool,
        enable_batch_norm: bool,
        batch_norm_momentum: f32,
        batch_norm_renorm: bool,
        batch_norm_renorm_clipping: Option<(f32, f32, f32)>,
        batch_norm_renorm_momentum: f32,
    ) -> Self {
        let mut dense_layers = Vec::new();
        let mut activations = Vec::new();
        let mut batch_norms = Vec::new();
        let mut prev_dim = input_dim;

        // Build hidden layers
        for &units in hidden_units {
            let mut dense = Dense::new_with_options(
                prev_dim,
                units,
                initializer,
                Initializer::Zeros,
                use_bias,
                kernel_regularizer.clone(),
                bias_regularizer.clone(),
                kernel_constraint.clone(),
                bias_constraint.clone(),
            );
            if use_weight_norm {
                dense = dense.with_kernel_norm(use_learnable_weight_norm);
            }
            dense_layers.push(dense);
            activations.push(ActivationWrapper::from_type(activation.clone()));

            let bn = if enable_batch_norm {
                let mut bn = BatchNorm::with_momentum(units, batch_norm_momentum, 1e-5);
                bn = bn.with_renorm(
                    batch_norm_renorm,
                    batch_norm_renorm_clipping,
                    batch_norm_renorm_momentum,
                );
                Some(bn)
            } else {
                None
            };
            batch_norms.push(bn);

            prev_dim = units;
        }

        // Output layer (no activation)
        let mut dense = Dense::new_with_options(
            prev_dim,
            output_dim,
            initializer,
            Initializer::Zeros,
            use_bias,
            kernel_regularizer.clone(),
            bias_regularizer.clone(),
            kernel_constraint.clone(),
            bias_constraint.clone(),
        );
        if use_weight_norm {
            dense = dense.with_kernel_norm(use_learnable_weight_norm);
        }
        dense_layers.push(dense);
        activations.push(ActivationWrapper::None);
        batch_norms.push(None);

        let input_batch_norm = if enable_batch_norm {
            let mut bn = BatchNorm::with_momentum(input_dim, batch_norm_momentum, 1e-5);
            bn = bn.with_renorm(
                batch_norm_renorm,
                batch_norm_renorm_clipping,
                batch_norm_renorm_momentum,
            );
            Some(bn)
        } else {
            None
        };

        Self {
            dense_layers,
            input_batch_norm,
            batch_norms,
            activations,
            input_dim,
            output_dim,
            cached_inputs: Vec::new(),
            training: true,
        }
    }

    /// Returns the input dimension of the expert.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the output dimension of the expert.
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    /// Performs forward pass and caches inputs for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_inputs.clear();
        let mut x = input.clone();

        if let Some(bn) = self.input_batch_norm.as_mut() {
            x = bn.forward_train(&x)?;
        }

        for idx in 0..self.dense_layers.len() {
            self.cached_inputs.push(x.clone());
            x = self.dense_layers[idx].forward_train(&x)?;
            if let Some(bn) = self.batch_norms[idx].as_mut() {
                x = bn.forward_train(&x)?;
            }
            x = self.activations[idx].forward_train(&x)?;
        }

        Ok(x)
    }
}

impl Layer for Expert {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Expert expects 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[1],
            });
        }

        let mut x = input.clone();
        if let Some(bn) = &self.input_batch_norm {
            x = bn.forward(&x)?;
        }
        for idx in 0..self.dense_layers.len() {
            x = self.dense_layers[idx].forward(&x)?;
            if let Some(bn) = &self.batch_norms[idx] {
                x = bn.forward(&x)?;
            }
            x = self.activations[idx].forward(&x)?;
        }
        Ok(x)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let mut g = grad.clone();

        // Backward through layers in reverse order
        for idx in (0..self.dense_layers.len()).rev() {
            g = self.activations[idx].backward(&g)?;
            if let Some(bn) = self.batch_norms[idx].as_mut() {
                g = bn.backward(&g)?;
            }
            g = self.dense_layers[idx].backward(&g)?;
        }

        if let Some(bn) = self.input_batch_norm.as_mut() {
            g = bn.backward(&g)?;
        }

        Ok(g)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params: Vec<&Tensor> = self
            .dense_layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect();
        if let Some(bn) = &self.input_batch_norm {
            params.extend(bn.parameters());
        }
        for bn in &self.batch_norms {
            if let Some(bn) = bn {
                params.extend(bn.parameters());
            }
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = self
            .dense_layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect();
        if let Some(bn) = self.input_batch_norm.as_mut() {
            params.extend(bn.parameters_mut());
        }
        for bn in &mut self.batch_norms {
            if let Some(bn) = bn {
                params.extend(bn.parameters_mut());
            }
        }
        params
    }

    fn name(&self) -> &str {
        "Expert"
    }

    fn regularization_loss(&self) -> f32 {
        self.dense_layers
            .iter()
            .map(|layer| layer.regularization_loss())
            .sum()
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn apply_constraints(&mut self) {
        for layer in &mut self.dense_layers {
            layer.apply_constraints();
        }
    }
}

/// Gating network for a single task in MMoE.
///
/// The gate computes a softmax distribution over experts for each input sample,
/// determining how much each expert should contribute to the task's output.
///
/// # Architecture
///
/// The gate is a simple linear layer followed by softmax:
/// - Input: [batch_size, input_dim]
/// - Output: [batch_size, num_experts] (softmax probabilities)
///
/// # Example
///
/// ```
/// use monolith_layers::mmoe::Gate;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let gate = Gate::new(64, 4);  // 64-dim input, 4 experts
/// let input = Tensor::rand(&[8, 64]);
/// let weights = gate.forward(&input).unwrap();
/// assert_eq!(weights.shape(), &[8, 4]);  // weights over 4 experts
/// ```
#[derive(Debug, Clone)]
pub struct Gate {
    /// Linear projection layer
    linear: Dense,
    /// Softmax activation
    softmax: Softmax,
    /// Gate type
    gate_type: GateType,
    /// Top-k for sparse gates
    top_k: usize,
    /// Optional noise weight for noise_topk
    noise_weight: Option<Tensor>,
    /// Gradient for noise weight
    noise_weight_grad: Option<Tensor>,
    /// Cached noise pre-activation
    cached_noise_pre: Option<Tensor>,
    /// Cached noise epsilon
    cached_noise_eps: Option<Tensor>,
    /// Input dimension
    input_dim: usize,
    /// Number of experts
    num_experts: usize,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Whether in training mode
    training: bool,
}

impl Gate {
    /// Creates a new gating network.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `num_experts` - Number of experts to weight
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::mmoe::Gate;
    ///
    /// let gate = Gate::new(128, 8);  // 128-dim input, 8 experts
    /// ```
    pub fn new(
        input_dim: usize,
        num_experts: usize,
    ) -> Self {
        Self::new_with_gate(input_dim, num_experts, GateType::Softmax, 1)
    }

    pub fn new_with_gate(
        input_dim: usize,
        num_experts: usize,
        gate_type: GateType,
        top_k: usize,
    ) -> Self {
        let mut linear = Dense::new(input_dim, num_experts);
        linear
            .weights_mut()
            .set_data(vec![0.0; input_dim * num_experts]);
        if linear.has_bias() {
            linear.bias_mut().set_data(vec![0.0; num_experts]);
        }
        let noise_weight = if matches!(gate_type, GateType::NoiseTopK) {
            Some(Initializer::GlorotNormal.initialize(&[input_dim, num_experts]))
        } else {
            None
        };
        Self {
            linear,
            softmax: Softmax::new(),
            gate_type,
            top_k: top_k.max(1),
            noise_weight,
            noise_weight_grad: None,
            cached_noise_pre: None,
            cached_noise_eps: None,
            input_dim,
            num_experts,
            cached_input: None,
            training: true,
        }
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        let mut logits = self.linear.forward_train(input)?;
        self.cached_noise_pre = None;
        self.cached_noise_eps = None;
        if matches!(self.gate_type, GateType::NoiseTopK) {
            if let Some(noise_weight) = &self.noise_weight {
                let noise_pre = input.matmul(noise_weight);
                let noise_scale = Softplus::new().forward(&noise_pre)?;
                let noise_eps = Tensor::randn(logits.shape(), 0.0, 1.0);
                let noise = noise_eps.mul(&noise_scale);
                logits = logits.add(&noise);
                self.cached_noise_pre = Some(noise_pre);
                self.cached_noise_eps = Some(noise_eps);
            }
        }
        let weights = self.softmax.forward_train(&logits)?;
        Ok(self.apply_topk(&weights))
    }

    fn apply_topk(&self, weights: &Tensor) -> Tensor {
        if matches!(self.gate_type, GateType::Softmax) {
            return weights.clone();
        }

        let batch = weights.shape()[0];
        let num_experts = weights.shape()[1];
        let data = weights.data();
        let mut out = vec![0.0; data.len()];

        for b in 0..batch {
            let row = &data[b * num_experts..(b + 1) * num_experts];
            let mut sorted = row.to_vec();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            let k = self.top_k.min(num_experts);
            let threshold = sorted[k - 1];
            let mut sum = 0.0;
            for i in 0..num_experts {
                let val = row[i];
                if val >= threshold {
                    out[b * num_experts + i] = val;
                    sum += val;
                }
            }
            if sum > 0.0 {
                for i in 0..num_experts {
                    out[b * num_experts + i] /= sum;
                }
            }
        }

        Tensor::from_data(&[batch, num_experts], out)
    }
}

impl Layer for Gate {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Gate expects 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[1],
            });
        }

        let mut logits = self.linear.forward(input)?;
        if matches!(self.gate_type, GateType::NoiseTopK) {
            if let Some(noise_weight) = &self.noise_weight {
                let noise_scale = Softplus::new().forward(&input.matmul(noise_weight))?;
                let noise = Tensor::randn(logits.shape(), 0.0, 1.0).mul(&noise_scale);
                logits = logits.add(&noise);
            }
        }
        let weights = self.softmax.forward(&logits)?;
        Ok(self.apply_topk(&weights))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let g_logits = self.softmax.backward(grad)?;
        let mut input_grad = self.linear.backward(&g_logits)?;

        if matches!(self.gate_type, GateType::NoiseTopK) {
            if let (Some(noise_pre), Some(noise_eps), Some(noise_weight)) = (
                self.cached_noise_pre.as_ref(),
                self.cached_noise_eps.as_ref(),
                self.noise_weight.as_ref(),
            ) {
                let input = self
                    .cached_input
                    .as_ref()
                    .ok_or(LayerError::NotInitialized)?;
                let noise_scale_grad = g_logits.mul(noise_eps);
                let noise_pre_grad = noise_scale_grad.mul(&noise_pre.sigmoid());
                let noise_weight_grad = input.transpose().matmul(&noise_pre_grad);
                self.noise_weight_grad = Some(noise_weight_grad);
                let input_grad_noise = noise_pre_grad.matmul(&noise_weight.transpose());
                input_grad = input_grad.add(&input_grad_noise);
            }
        }

        Ok(input_grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = self.linear.parameters();
        if let Some(noise_weight) = &self.noise_weight {
            params.push(noise_weight);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = self.linear.parameters_mut();
        if let Some(noise_weight) = &mut self.noise_weight {
            params.push(noise_weight);
        }
        params
    }

    fn name(&self) -> &str {
        "Gate"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

/// Configuration for building an MMoE layer.
///
/// # Example
///
/// ```
/// use monolith_layers::mmoe::MMoEConfig;
/// use monolith_layers::mlp::ActivationType;
///
/// let config = MMoEConfig::new(128, 4, 3)
///     .with_expert_hidden_units(vec![64, 64])
///     .with_expert_activation(ActivationType::relu());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMoEConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Optional gate input dimension (defaults to input_dim)
    pub gate_input_dim: Option<usize>,
    /// Number of expert networks
    pub num_experts: usize,
    /// Number of tasks (and gates)
    pub num_tasks: usize,
    /// Hidden layer units for each expert
    pub expert_hidden_units: Vec<usize>,
    /// Activation function for expert hidden layers
    pub expert_activation: ActivationType,
    /// Optional per-expert activations (length = num_experts)
    pub expert_activations: Option<Vec<ActivationType>>,
    /// Optional per-expert initializers (length = num_experts)
    pub expert_initializers: Option<Vec<Initializer>>,
    /// Output dimension of each expert (defaults to last hidden unit size or input_dim)
    pub expert_output_dim: Option<usize>,
    /// Gate type (softmax, topk, noise_topk)
    pub gate_type: GateType,
    /// Top-k value for topk/noise_topk gate
    pub top_k: usize,
    /// Whether to use bias in expert dense layers
    pub use_bias: bool,
    /// Kernel regularizer for expert dense layers
    pub kernel_regularizer: Regularizer,
    /// Bias regularizer for expert dense layers
    pub bias_regularizer: Regularizer,
    /// Kernel constraint for expert dense layers
    pub kernel_constraint: Constraint,
    /// Bias constraint for expert dense layers
    pub bias_constraint: Constraint,
    /// Whether to enable weight normalization in experts
    pub use_weight_norm: bool,
    /// Whether expert weight norm scale is trainable
    pub use_learnable_weight_norm: bool,
    /// Whether to enable batch normalization in experts
    pub enable_batch_normalization: bool,
    /// Batch normalization momentum
    pub batch_normalization_momentum: f32,
    /// Whether to use batch renorm
    pub batch_normalization_renorm: bool,
    /// Batch renorm clipping (rmin, rmax, dmax)
    pub batch_normalization_renorm_clipping: Option<(f32, f32, f32)>,
    /// Batch renorm momentum
    pub batch_normalization_renorm_momentum: f32,
}

impl MMoEConfig {
    /// Creates a new MMoE configuration.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `num_experts` - Number of expert networks
    /// * `num_tasks` - Number of tasks
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::mmoe::MMoEConfig;
    ///
    /// let config = MMoEConfig::new(64, 4, 2);
    /// ```
    pub fn new(input_dim: usize, num_experts: usize, num_tasks: usize) -> Self {
        Self {
            input_dim,
            gate_input_dim: None,
            num_experts,
            num_tasks,
            expert_hidden_units: Vec::new(),
            expert_activation: ActivationType::relu(),
            expert_activations: None,
            expert_initializers: None,
            expert_output_dim: None,
            gate_type: GateType::Softmax,
            top_k: 1,
            use_bias: true,
            kernel_regularizer: Regularizer::None,
            bias_regularizer: Regularizer::None,
            kernel_constraint: Constraint::None,
            bias_constraint: Constraint::None,
            use_weight_norm: false,
            use_learnable_weight_norm: false,
            enable_batch_normalization: false,
            batch_normalization_momentum: 0.99,
            batch_normalization_renorm: false,
            batch_normalization_renorm_clipping: None,
            batch_normalization_renorm_momentum: 0.99,
        }
    }

    /// Sets gate input dimension (when gate input differs from expert input).
    pub fn with_gate_input_dim(mut self, gate_input_dim: usize) -> Self {
        self.gate_input_dim = Some(gate_input_dim);
        self
    }

    /// Sets the hidden layer units for each expert.
    ///
    /// # Arguments
    ///
    /// * `units` - Vector of hidden layer sizes
    pub fn with_expert_hidden_units(mut self, units: Vec<usize>) -> Self {
        self.expert_hidden_units = units;
        self
    }

    /// Sets the activation function for expert hidden layers.
    ///
    /// # Arguments
    ///
    /// * `activation` - The activation function type
    pub fn with_expert_activation(mut self, activation: ActivationType) -> Self {
        self.expert_activation = activation;
        self
    }

    /// Sets whether to use bias in expert dense layers.
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Sets kernel and bias regularizers for expert dense layers.
    pub fn with_regularizers(
        mut self,
        kernel_regularizer: Regularizer,
        bias_regularizer: Regularizer,
    ) -> Self {
        self.kernel_regularizer = kernel_regularizer;
        self.bias_regularizer = bias_regularizer;
        self
    }

    /// Sets kernel and bias constraints for expert dense layers.
    pub fn with_constraints(
        mut self,
        kernel_constraint: Constraint,
        bias_constraint: Constraint,
    ) -> Self {
        self.kernel_constraint = kernel_constraint;
        self.bias_constraint = bias_constraint;
        self
    }

    /// Sets per-expert activations.
    pub fn with_expert_activations(mut self, activations: Vec<ActivationType>) -> Self {
        self.expert_activations = Some(activations);
        self
    }

    /// Sets per-expert initializers.
    pub fn with_expert_initializers(mut self, initializers: Vec<Initializer>) -> Self {
        self.expert_initializers = Some(initializers);
        self
    }

    /// Sets the output dimension of each expert.
    ///
    /// If not specified, defaults to the last hidden unit size, or input_dim if no hidden layers.
    ///
    /// # Arguments
    ///
    /// * `dim` - The expert output dimension
    pub fn with_expert_output_dim(mut self, dim: usize) -> Self {
        self.expert_output_dim = Some(dim);
        self
    }

    /// Sets the gate type.
    pub fn with_gate_type(mut self, gate_type: GateType) -> Self {
        self.gate_type = gate_type;
        self
    }

    /// Sets the top-k value for topk/noise_topk gates.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k.max(1);
        self
    }

    /// Enables weight normalization for experts.
    pub fn with_weight_norm(mut self, use_weight_norm: bool) -> Self {
        self.use_weight_norm = use_weight_norm;
        self
    }

    /// Sets whether expert weight norm scale is trainable.
    pub fn with_learnable_weight_norm(mut self, use_learnable: bool) -> Self {
        self.use_learnable_weight_norm = use_learnable;
        self
    }

    /// Enables batch normalization for experts.
    pub fn with_batch_normalization(
        mut self,
        enable: bool,
        momentum: f32,
        renorm: bool,
        renorm_clipping: Option<(f32, f32, f32)>,
        renorm_momentum: f32,
    ) -> Self {
        self.enable_batch_normalization = enable;
        self.batch_normalization_momentum = momentum;
        self.batch_normalization_renorm = renorm;
        self.batch_normalization_renorm_clipping = renorm_clipping;
        self.batch_normalization_renorm_momentum = renorm_momentum;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), LayerError> {
        if self.input_dim == 0 {
            return Err(LayerError::ConfigError {
                message: "Input dimension must be positive".to_string(),
            });
        }
        if self.num_experts == 0 {
            return Err(LayerError::ConfigError {
                message: "Number of experts must be positive".to_string(),
            });
        }
        if self.num_tasks == 0 {
            return Err(LayerError::ConfigError {
                message: "Number of tasks must be positive".to_string(),
            });
        }
        for (i, &units) in self.expert_hidden_units.iter().enumerate() {
            if units == 0 {
                return Err(LayerError::ConfigError {
                    message: format!("Expert hidden layer {} has zero units", i),
                });
            }
        }
        if let Some(dim) = self.expert_output_dim {
            if dim == 0 {
                return Err(LayerError::ConfigError {
                    message: "Expert output dimension must be positive".to_string(),
                });
            }
        }
        if let Some(acts) = &self.expert_activations {
            if acts.len() != self.num_experts {
                return Err(LayerError::ConfigError {
                    message: "expert_activations length must match num_experts".to_string(),
                });
            }
        }
        if let Some(inits) = &self.expert_initializers {
            if inits.len() != self.num_experts {
                return Err(LayerError::ConfigError {
                    message: "expert_initializers length must match num_experts".to_string(),
                });
            }
        }
        if matches!(self.gate_type, GateType::TopK | GateType::NoiseTopK) && self.top_k == 0 {
            return Err(LayerError::ConfigError {
                message: "top_k must be positive for topk/noise_topk gates".to_string(),
            });
        }
        Ok(())
    }

    /// Builds the MMoE layer from this configuration.
    pub fn build(self) -> Result<MMoE, LayerError> {
        MMoE::from_config(self)
    }

    /// Returns the effective expert output dimension.
    fn effective_expert_output_dim(&self) -> usize {
        self.expert_output_dim.unwrap_or_else(|| {
            self.expert_hidden_units
                .last()
                .copied()
                .unwrap_or(self.input_dim)
        })
    }
}

/// Multi-gate Mixture of Experts (MMoE) layer.
///
/// MMoE is a multi-task learning architecture that uses a shared set of expert
/// networks and task-specific gating networks. Each task has its own gate that
/// learns to combine expert outputs in a task-specific way.
///
/// # Architecture
///
/// For each input:
/// 1. All experts process the input independently
/// 2. Each task's gate computes softmax weights over experts
/// 3. The task output is the weighted sum of expert outputs
///
/// # Example
///
/// ```
/// use monolith_layers::mmoe::{MMoE, MMoEConfig};
/// use monolith_layers::mlp::ActivationType;
/// use monolith_layers::tensor::Tensor;
///
/// // Create an MMoE with 4 experts for 2 tasks
/// let config = MMoEConfig::new(64, 4, 2)
///     .with_expert_hidden_units(vec![32, 32])
///     .with_expert_activation(ActivationType::relu());
///
/// let mmoe = MMoE::from_config(config).unwrap();
///
/// // Forward pass returns one output per task
/// let input = Tensor::rand(&[16, 64]);
/// let outputs = mmoe.forward_multi(&input).unwrap();
///
/// assert_eq!(outputs.len(), 2);  // 2 tasks
/// assert_eq!(outputs[0].shape(), &[16, 32]);  // expert output dim
/// ```
#[derive(Debug, Clone)]
pub struct MMoE {
    /// Shared expert networks
    experts: Vec<Expert>,
    /// Per-task gating networks
    gates: Vec<Gate>,
    /// Number of experts
    num_experts: usize,
    /// Number of tasks
    num_tasks: usize,
    /// Input dimension
    input_dim: usize,
    /// Gate input dimension (can differ from expert input)
    gate_input_dim: usize,
    /// Expert output dimension
    expert_output_dim: usize,
    /// Configuration used to build this MMoE
    config: MMoEConfig,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Cached gate input for backward pass
    cached_gate_input: Option<Tensor>,
    /// Cached expert outputs for backward pass
    cached_expert_outputs: Vec<Tensor>,
    /// Cached gate outputs for backward pass
    cached_gate_outputs: Vec<Tensor>,
    /// Cached importance loss (for topk/noise_topk gates)
    importance_loss: Option<f32>,
    /// Whether in training mode
    training: bool,
}

impl MMoE {
    /// Creates an MMoE layer from a configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The MMoE configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::mmoe::{MMoE, MMoEConfig};
    /// use monolith_layers::mlp::ActivationType;
    ///
    /// let config = MMoEConfig::new(64, 4, 2)
    ///     .with_expert_hidden_units(vec![32])
    ///     .with_expert_activation(ActivationType::relu());
    ///
    /// let mmoe = MMoE::from_config(config).unwrap();
    /// ```
    pub fn from_config(config: MMoEConfig) -> Result<Self, LayerError> {
        config.validate()?;

        let expert_output_dim = config.effective_expert_output_dim();

        let gate_input_dim = config.gate_input_dim.unwrap_or(config.input_dim);

        // Create experts
        let experts: Vec<Expert> = (0..config.num_experts)
            .map(|idx| {
                let activation = config
                    .expert_activations
                    .as_ref()
                    .map(|acts| acts[idx].clone())
                    .unwrap_or_else(|| config.expert_activation.clone());
                let initializer = config
                    .expert_initializers
                    .as_ref()
                    .map(|inits| inits[idx])
                    .unwrap_or(Initializer::GlorotUniform);
                Expert::new_with_options(
                    config.input_dim,
                    &config.expert_hidden_units,
                    expert_output_dim,
                    activation,
                    initializer,
                    config.use_bias,
                    config.kernel_regularizer.clone(),
                    config.bias_regularizer.clone(),
                    config.kernel_constraint.clone(),
                    config.bias_constraint.clone(),
                    config.use_weight_norm,
                    config.use_learnable_weight_norm,
                    config.enable_batch_normalization,
                    config.batch_normalization_momentum,
                    config.batch_normalization_renorm,
                    config.batch_normalization_renorm_clipping,
                    config.batch_normalization_renorm_momentum,
                )
            })
            .collect();

        // Create gates (one per task)
        let gates: Vec<Gate> = (0..config.num_tasks)
            .map(|_| {
                Gate::new_with_gate(
                    gate_input_dim,
                    config.num_experts,
                    config.gate_type,
                    config.top_k,
                )
            })
            .collect();

        Ok(Self {
            experts,
            gates,
            num_experts: config.num_experts,
            num_tasks: config.num_tasks,
            input_dim: config.input_dim,
            gate_input_dim,
            expert_output_dim,
            config,
            cached_input: None,
            cached_gate_input: None,
            cached_expert_outputs: Vec::new(),
            cached_gate_outputs: Vec::new(),
            importance_loss: None,
            training: true,
        })
    }

    /// Creates a simple MMoE layer.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `num_experts` - Number of experts
    /// * `num_tasks` - Number of tasks
    /// * `expert_hidden_units` - Hidden layer sizes for experts
    /// * `activation` - Activation function for expert hidden layers
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::mmoe::MMoE;
    /// use monolith_layers::mlp::ActivationType;
    ///
    /// let mmoe = MMoE::new(64, 4, 2, &[32, 32], ActivationType::relu()).unwrap();
    /// ```
    pub fn new(
        input_dim: usize,
        num_experts: usize,
        num_tasks: usize,
        expert_hidden_units: &[usize],
        activation: ActivationType,
    ) -> Result<Self, LayerError> {
        let config = MMoEConfig::new(input_dim, num_experts, num_tasks)
            .with_expert_hidden_units(expert_hidden_units.to_vec())
            .with_expert_activation(activation);
        Self::from_config(config)
    }

    /// Returns the number of experts.
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Returns the number of tasks.
    pub fn num_tasks(&self) -> usize {
        self.num_tasks
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the expert output dimension.
    pub fn expert_output_dim(&self) -> usize {
        self.expert_output_dim
    }

    /// Returns a reference to the experts.
    pub fn experts(&self) -> &[Expert] {
        &self.experts
    }

    /// Returns a reference to the gates.
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Returns the configuration.
    pub fn config(&self) -> &MMoEConfig {
        &self.config
    }

    /// Forward pass returning outputs for all tasks.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// A vector of tensors, one per task, each of shape [batch_size, expert_output_dim]
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::mmoe::{MMoE, MMoEConfig};
    /// use monolith_layers::mlp::ActivationType;
    /// use monolith_layers::tensor::Tensor;
    ///
    /// let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
    /// let input = Tensor::rand(&[8, 64]);
    /// let outputs = mmoe.forward_multi(&input).unwrap();
    ///
    /// assert_eq!(outputs.len(), 2);
    /// ```
    pub fn forward_multi(&self, input: &Tensor) -> Result<Vec<Tensor>, LayerError> {
        if self.gate_input_dim != self.input_dim {
            return Err(LayerError::ForwardError {
                message: "MMoE gate_input_dim differs; use forward_multi_with_gate_input"
                    .to_string(),
            });
        }
        self.forward_multi_with_gate_input(input, input)
    }

    /// Forward pass with separate expert and gate inputs.
    pub fn forward_multi_with_gate_input(
        &self,
        expert_input: &Tensor,
        gate_input: &Tensor,
    ) -> Result<Vec<Tensor>, LayerError> {
        if expert_input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("MMoE expects 2D expert input, got {}D", expert_input.ndim()),
            });
        }
        if expert_input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: expert_input.shape()[1],
            });
        }
        if gate_input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("MMoE gate input expects 2D input, got {}D", gate_input.ndim()),
            });
        }
        if gate_input.shape()[1] != self.gate_input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.gate_input_dim,
                actual: gate_input.shape()[1],
            });
        }

        let batch_size = expert_input.shape()[0];

        // Compute all expert outputs: [num_experts, batch_size, expert_output_dim]
        let expert_outputs: Vec<Tensor> = self
            .experts
            .iter()
            .map(|expert| expert.forward(expert_input))
            .collect::<Result<Vec<_>, _>>()?;

        // Compute output for each task
        let mut task_outputs = Vec::with_capacity(self.num_tasks);

        for gate in &self.gates {
            // Compute gate weights: [batch_size, num_experts]
            let gate_weights = gate.forward(gate_input)?;

            // Compute weighted sum of expert outputs
            // output = sum_e(gate_weight_e * expert_output_e)
            let task_output = self.weighted_sum_experts(&expert_outputs, &gate_weights, batch_size);
            task_outputs.push(task_output);
        }

        Ok(task_outputs)
    }

    /// Forward pass with training mode for multi-task output (same input for experts and gates).
    ///
    /// Caches intermediate values for backward pass.
    pub fn forward_multi_train(&mut self, input: &Tensor) -> Result<Vec<Tensor>, LayerError> {
        if self.gate_input_dim != self.input_dim {
            return Err(LayerError::ForwardError {
                message: "MMoE gate_input_dim differs; use forward_multi_train_with_gate_input"
                    .to_string(),
            });
        }
        self.forward_multi_train_with_gate_input(input, input)
    }

    /// Forward pass with training mode for multi-task output (separate expert and gate inputs).
    pub fn forward_multi_train_with_gate_input(
        &mut self,
        expert_input: &Tensor,
        gate_input: &Tensor,
    ) -> Result<Vec<Tensor>, LayerError> {
        if expert_input.ndim() != 2 || gate_input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: "MMoE expects 2D inputs for expert and gate".to_string(),
            });
        }
        if expert_input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: expert_input.shape()[1],
            });
        }
        if gate_input.shape()[1] != self.gate_input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.gate_input_dim,
                actual: gate_input.shape()[1],
            });
        }

        let batch_size = expert_input.shape()[0];

        // Cache inputs
        self.cached_input = Some(expert_input.clone());
        self.cached_gate_input = Some(gate_input.clone());

        // Compute all expert outputs
        self.cached_expert_outputs.clear();
        for expert in &mut self.experts {
            let output = expert.forward_train(expert_input)?;
            self.cached_expert_outputs.push(output);
        }

        // Compute gate outputs for each task
        self.cached_gate_outputs.clear();
        let mut gate_weights_list = Vec::with_capacity(self.num_tasks);

        for gate in &mut self.gates {
            let gate_weights = gate.forward_train(gate_input)?;
            self.cached_gate_outputs.push(gate_weights.clone());
            gate_weights_list.push(gate_weights);
        }

        // Importance loss for topk/noise_topk gates
        if matches!(self.config.gate_type, GateType::TopK | GateType::NoiseTopK) {
            let mut cv_sum = 0.0f32;
            for weights in &gate_weights_list {
                let importance = weights.sum_axis(0);
                let mean = importance.sum() / importance.shape()[0] as f32;
                let var = importance
                    .sub(&Tensor::from_data(&[1], vec![mean]))
                    .sqr()
                    .sum()
                    / importance.shape()[0] as f32;
                if mean.abs() > 1e-12 {
                    cv_sum += var / (mean * mean);
                }
            }
            self.importance_loss = Some(cv_sum);
        } else {
            self.importance_loss = None;
        }

        // Compute weighted sums for each task (after the mutable borrow is released)
        let mut task_outputs = Vec::with_capacity(self.num_tasks);
        for gate_weights in &gate_weights_list {
            let task_output =
                self.weighted_sum_experts(&self.cached_expert_outputs, gate_weights, batch_size);
            task_outputs.push(task_output);
        }

        Ok(task_outputs)
    }

    /// Computes weighted sum of expert outputs based on gate weights.
    fn weighted_sum_experts(
        &self,
        expert_outputs: &[Tensor],
        gate_weights: &Tensor,
        batch_size: usize,
    ) -> Tensor {
        let mut stacked = Vec::with_capacity(self.num_experts);
        for expert in expert_outputs {
            stacked.push(expert.reshape(&[batch_size, self.expert_output_dim, 1]));
        }
        let experts_stacked = Tensor::cat(&stacked, 2);

        let weights = gate_weights
            .reshape(&[batch_size, 1, self.num_experts])
            .broadcast_as(&[batch_size, self.expert_output_dim, self.num_experts]);
        let weighted = experts_stacked.mul(&weights);
        weighted.sum_axis(2)
    }

    /// Returns importance loss for topk/noise_topk gates.
    pub fn importance_loss(&self) -> Option<f32> {
        self.importance_loss
    }

    /// Backward pass for multi-task output.
    ///
    /// # Arguments
    ///
    /// * `grads` - Gradients for each task output, one tensor per task
    ///
    /// # Returns
    ///
    /// Gradient with respect to the input
    pub fn backward_multi(&mut self, grads: &[Tensor]) -> Result<Tensor, LayerError> {
        if self.gate_input_dim != self.input_dim {
            return Err(LayerError::BackwardError {
                message: "MMoE gate_input_dim differs; use backward_multi_with_gate_input"
                    .to_string(),
            });
        }
        let (expert_grad, gate_grad) = self.backward_multi_with_gate_input(grads)?;
        Ok(expert_grad.add(&gate_grad))
    }

    /// Backward pass for separate expert and gate inputs.
    pub fn backward_multi_with_gate_input(
        &mut self,
        grads: &[Tensor],
    ) -> Result<(Tensor, Tensor), LayerError> {
        if grads.len() != self.num_tasks {
            return Err(LayerError::BackwardError {
                message: format!(
                    "Expected {} gradients (one per task), got {}",
                    self.num_tasks,
                    grads.len()
                ),
            });
        }

        let expert_input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let _gate_input = self
            .cached_gate_input
            .as_ref()
            .unwrap_or(expert_input);

        let batch_size = expert_input.shape()[0];

        // Initialize gradients
        let mut expert_input_grad = Tensor::zeros(&[batch_size, self.input_dim]);
        let mut gate_input_grad = Tensor::zeros(&[batch_size, self.gate_input_dim]);
        let mut expert_grads: Vec<Tensor> = (0..self.num_experts)
            .map(|_| Tensor::zeros(&[batch_size, self.expert_output_dim]))
            .collect();

        // Backward through each task
        for (task_idx, grad) in grads.iter().enumerate() {
            let gate_weights = &self.cached_gate_outputs[task_idx];

            // Gradient w.r.t. expert outputs: d_expert_e = gate_weight_e * d_output
            let gate_data = gate_weights.data_ref();
            let grad_data = grad.data_ref();
            for (expert_idx, expert_grad) in expert_grads.iter_mut().enumerate() {
                let expert_output_dim = self.expert_output_dim;
                let num_experts = self.num_experts;
                expert_grad.modify_data(|expert_grad_data| {
                    for b in 0..batch_size {
                        let weight = gate_data[b * num_experts + expert_idx];
                        for d in 0..expert_output_dim {
                            let grad_val = grad_data[b * expert_output_dim + d];
                            expert_grad_data[b * expert_output_dim + d] += weight * grad_val;
                        }
                    }
                });
            }

            // Gradient w.r.t. gate weights: d_gate_e = expert_output_e dot d_output
            let mut gate_grad = vec![0.0; batch_size * self.num_experts];
            for expert_idx in 0..self.num_experts {
                let expert_output = &self.cached_expert_outputs[expert_idx];
                let expert_data = expert_output.data_ref();
                let grad_data = grad.data_ref();
                for b in 0..batch_size {
                    let mut dot = 0.0;
                    for d in 0..self.expert_output_dim {
                        dot += expert_data[b * self.expert_output_dim + d]
                            * grad_data[b * self.expert_output_dim + d];
                    }
                    gate_grad[b * self.num_experts + expert_idx] = dot;
                }
            }
            let gate_grad = Tensor::from_data(&[batch_size, self.num_experts], gate_grad);

            // Backward through gate
            let gate_grad_input = self.gates[task_idx].backward(&gate_grad)?;

            // Accumulate gate input gradient
            gate_input_grad = gate_input_grad.add(&gate_grad_input);
        }

        // Backward through experts
        for (expert_idx, expert_grad) in expert_grads.iter().enumerate() {
            let expert_grad_input = self.experts[expert_idx].backward(expert_grad)?;
            expert_input_grad = expert_input_grad.add(&expert_grad_input);
        }

        Ok((expert_input_grad, gate_input_grad))
    }
}

impl Layer for MMoE {
    /// Forward pass returning the first task's output.
    ///
    /// For multi-task output, use `forward_multi` instead.
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let outputs = self.forward_multi(input)?;
        Ok(outputs.into_iter().next().unwrap())
    }

    /// Backward pass for the first task's gradient.
    ///
    /// For multi-task backward, use `backward_multi` instead.
    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        // Create gradient vector with zeros for all tasks except the first
        let mut grads = vec![Tensor::zeros(grad.shape()); self.num_tasks];
        grads[0] = grad.clone();
        self.backward_multi(&grads)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        for gate in &self.gates {
            params.extend(gate.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for expert in &mut self.experts {
            params.extend(expert.parameters_mut());
        }
        for gate in &mut self.gates {
            params.extend(gate.parameters_mut());
        }
        params
    }

    fn name(&self) -> &str {
        "MMoE"
    }

    fn regularization_loss(&self) -> f32 {
        let mut loss: f32 = self
            .experts
            .iter()
            .map(|expert| expert.regularization_loss())
            .sum();
        if let Some(imp) = self.importance_loss {
            loss += imp;
        }
        loss
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
        for expert in &mut self.experts {
            expert.set_training(training);
        }
        for gate in &mut self.gates {
            gate.set_training(training);
        }
    }

    fn apply_constraints(&mut self) {
        for expert in &mut self.experts {
            expert.apply_constraints();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(64, &[32, 16], 8, ActivationType::relu());
        assert_eq!(expert.input_dim(), 64);
        assert_eq!(expert.output_dim(), 8);
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(64, &[32], 16, ActivationType::relu());
        let input = Tensor::rand(&[8, 64]);
        let output = expert.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8, 16]);
    }

    #[test]
    fn test_expert_forward_invalid_input() {
        let expert = Expert::new(64, &[32], 16, ActivationType::relu());
        let input = Tensor::rand(&[8, 128]); // wrong dimension
        let result = expert.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_expert_backward() {
        let mut expert = Expert::new(64, &[32], 16, ActivationType::relu());
        let input = Tensor::rand(&[8, 64]);

        let _output = expert.forward_train(&input).unwrap();
        let grad = Tensor::ones(&[8, 16]);
        let input_grad = expert.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[8, 64]);
    }

    #[test]
    fn test_gate_creation() {
        let gate = Gate::new(64, 4);
        assert_eq!(gate.input_dim(), 64);
        assert_eq!(gate.num_experts(), 4);
    }

    #[test]
    fn test_gate_forward() {
        let gate = Gate::new(64, 4);
        let input = Tensor::rand(&[8, 64]);
        let weights = gate.forward(&input).unwrap();

        assert_eq!(weights.shape(), &[8, 4]);

        // Check that weights sum to 1 for each sample (softmax property)
        for i in 0..8 {
            let sum: f32 = (0..4).map(|j| weights.data()[i * 4 + j]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Weights should sum to 1, got {}",
                sum
            );
        }
    }

    #[test]
    fn test_gate_backward() {
        let mut gate = Gate::new(64, 4);
        let input = Tensor::rand(&[8, 64]);

        let _weights = gate.forward_train(&input).unwrap();
        let grad = Tensor::ones(&[8, 4]);
        let input_grad = gate.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[8, 64]);
    }

    #[test]
    fn test_mmoe_config() {
        let config = MMoEConfig::new(64, 4, 2)
            .with_expert_hidden_units(vec![32, 16])
            .with_expert_activation(ActivationType::relu());

        assert_eq!(config.input_dim, 64);
        assert_eq!(config.num_experts, 4);
        assert_eq!(config.num_tasks, 2);
        assert_eq!(config.expert_hidden_units, vec![32, 16]);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_mmoe_config_invalid() {
        let config = MMoEConfig::new(0, 4, 2);
        assert!(config.validate().is_err());

        let config = MMoEConfig::new(64, 0, 2);
        assert!(config.validate().is_err());

        let config = MMoEConfig::new(64, 4, 0);
        assert!(config.validate().is_err());

        let config = MMoEConfig::new(64, 4, 2).with_expert_hidden_units(vec![0]);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_mmoe_creation() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
        assert_eq!(mmoe.num_experts(), 4);
        assert_eq!(mmoe.num_tasks(), 2);
        assert_eq!(mmoe.input_dim(), 64);
        assert_eq!(mmoe.expert_output_dim(), 32);
    }

    #[test]
    fn test_mmoe_forward_multi() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
        let input = Tensor::rand(&[8, 64]);
        let outputs = mmoe.forward_multi(&input).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[8, 32]);
        assert_eq!(outputs[1].shape(), &[8, 32]);
    }

    #[test]
    fn test_mmoe_forward() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
        let input = Tensor::rand(&[8, 64]);
        let output = mmoe.forward(&input).unwrap();

        assert_eq!(output.shape(), &[8, 32]);
    }

    #[test]
    fn test_mmoe_forward_invalid_input() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();

        // Wrong number of dimensions
        let input = Tensor::rand(&[8, 64, 1]);
        assert!(mmoe.forward(&input).is_err());

        // Wrong input dimension
        let input = Tensor::rand(&[8, 128]);
        assert!(mmoe.forward(&input).is_err());
    }

    #[test]
    fn test_mmoe_backward_multi() {
        let mut mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
        let input = Tensor::rand(&[8, 64]);

        let _outputs = mmoe.forward_multi_train(&input).unwrap();

        let grads = vec![Tensor::ones(&[8, 32]), Tensor::ones(&[8, 32])];
        let input_grad = mmoe.backward_multi(&grads).unwrap();

        assert_eq!(input_grad.shape(), &[8, 64]);
    }

    #[test]
    fn test_mmoe_parameters() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
        let params = mmoe.parameters();

        // 4 experts: each has 2 dense layers (64->32, 32->32), each with weights+bias = 4*4 = 16
        // 2 gates: each has 1 dense layer (64->4) with weights+bias = 2*2 = 4
        // Total: 16 + 4 = 20 parameters
        assert_eq!(params.len(), 20);
    }

    #[test]
    fn test_mmoe_training_mode() {
        let mut mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
        assert!(mmoe.is_training());

        mmoe.set_training(false);
        assert!(!mmoe.is_training());

        mmoe.set_training(true);
        assert!(mmoe.is_training());
    }

    #[test]
    fn test_mmoe_no_hidden_layers() {
        // Test MMoE without hidden layers in experts
        let config = MMoEConfig::new(64, 4, 2).with_expert_output_dim(32);
        let mmoe = MMoE::from_config(config).unwrap();

        let input = Tensor::rand(&[8, 64]);
        let outputs = mmoe.forward_multi(&input).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[8, 32]);
    }

    #[test]
    fn test_mmoe_different_activations() {
        for activation in [
            ActivationType::relu(),
            ActivationType::sigmoid(),
            ActivationType::tanh(),
            ActivationType::gelu(),
            ActivationType::None,
        ] {
            let mmoe = MMoE::new(64, 2, 2, &[32], activation).unwrap();
            let input = Tensor::rand(&[4, 64]);
            let outputs = mmoe.forward_multi(&input);
            assert!(outputs.is_ok());
        }
    }

    #[test]
    fn test_mmoe_layer_trait() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::relu()).unwrap();
        assert_eq!(mmoe.name(), "MMoE");
    }

    #[test]
    fn test_expert_no_hidden_layers() {
        let expert = Expert::new(64, &[], 32, ActivationType::relu());
        let input = Tensor::rand(&[8, 64]);
        let output = expert.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8, 32]);
    }

    #[test]
    fn test_gate_softmax_normalization() {
        let gate = Gate::new(16, 3);
        // Use a simple input to verify softmax behavior
        let input = Tensor::zeros(&[2, 16]);
        let weights = gate.forward(&input).unwrap();

        // With zero input, all gate weights should be approximately equal (1/3)
        for i in 0..2 {
            for j in 0..3 {
                let w = weights.data()[i * 3 + j];
                assert!(
                    (w - 1.0 / 3.0).abs() < 0.1,
                    "Expected uniform weights, got {}",
                    w
                );
            }
        }
    }
}
