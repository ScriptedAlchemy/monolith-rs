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
//!     .with_expert_activation(ActivationType::ReLU);
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

use crate::activation::{ReLU, Sigmoid, Softmax, Tanh, GELU};
use crate::dense::Dense;
use crate::error::LayerError;
use crate::layer::Layer;
use crate::mlp::ActivationType;
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
/// let expert = Expert::new(64, &[32], 32, ActivationType::ReLU);
/// let input = Tensor::rand(&[8, 64]);
/// let output = expert.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[8, 32]);
/// ```
#[derive(Debug, Clone)]
pub struct Expert {
    /// Dense layers of the expert network
    dense_layers: Vec<Dense>,
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
    Tanh(Tanh),
    GELU(GELU),
    None,
}

impl ActivationWrapper {
    fn from_type(activation_type: ActivationType) -> Self {
        match activation_type {
            ActivationType::ReLU => ActivationWrapper::ReLU(ReLU::new()),
            ActivationType::Sigmoid => ActivationWrapper::Sigmoid(Sigmoid::new()),
            ActivationType::Tanh => ActivationWrapper::Tanh(Tanh::new()),
            ActivationType::GELU => ActivationWrapper::GELU(GELU::new()),
            ActivationType::None => ActivationWrapper::None,
        }
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            ActivationWrapper::ReLU(a) => a.forward(input),
            ActivationWrapper::Sigmoid(a) => a.forward(input),
            ActivationWrapper::Tanh(a) => a.forward(input),
            ActivationWrapper::GELU(a) => a.forward(input),
            ActivationWrapper::None => Ok(input.clone()),
        }
    }

    fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            ActivationWrapper::ReLU(a) => a.forward_train(input),
            ActivationWrapper::Sigmoid(a) => a.forward_train(input),
            ActivationWrapper::Tanh(a) => a.forward_train(input),
            ActivationWrapper::GELU(a) => a.forward_train(input),
            ActivationWrapper::None => Ok(input.clone()),
        }
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            ActivationWrapper::ReLU(a) => a.backward(grad),
            ActivationWrapper::Sigmoid(a) => a.backward(grad),
            ActivationWrapper::Tanh(a) => a.backward(grad),
            ActivationWrapper::GELU(a) => a.backward(grad),
            ActivationWrapper::None => Ok(grad.clone()),
        }
    }
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
    /// let expert = Expert::new(64, &[32, 32], 16, ActivationType::ReLU);
    /// ```
    pub fn new(
        input_dim: usize,
        hidden_units: &[usize],
        output_dim: usize,
        activation: ActivationType,
    ) -> Self {
        let mut dense_layers = Vec::new();
        let mut activations = Vec::new();
        let mut prev_dim = input_dim;

        // Build hidden layers
        for &units in hidden_units {
            dense_layers.push(Dense::new(prev_dim, units));
            activations.push(ActivationWrapper::from_type(activation));
            prev_dim = units;
        }

        // Output layer (no activation)
        dense_layers.push(Dense::new(prev_dim, output_dim));
        activations.push(ActivationWrapper::None);

        Self {
            dense_layers,
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

        for (dense, activation) in self
            .dense_layers
            .iter_mut()
            .zip(self.activations.iter_mut())
        {
            self.cached_inputs.push(x.clone());
            x = dense.forward_train(&x)?;
            x = activation.forward_train(&x)?;
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
        for (dense, activation) in self.dense_layers.iter().zip(self.activations.iter()) {
            x = dense.forward(&x)?;
            x = activation.forward(&x)?;
        }
        Ok(x)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let mut g = grad.clone();

        // Backward through layers in reverse order
        for (dense, activation) in self
            .dense_layers
            .iter_mut()
            .zip(self.activations.iter_mut())
            .rev()
        {
            g = activation.backward(&g)?;
            g = dense.backward(&g)?;
        }

        Ok(g)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.dense_layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.dense_layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    fn name(&self) -> &str {
        "Expert"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
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
    pub fn new(input_dim: usize, num_experts: usize) -> Self {
        Self {
            linear: Dense::new(input_dim, num_experts),
            softmax: Softmax::new(),
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
        let logits = self.linear.forward_train(input)?;
        self.softmax.forward_train(&logits)
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

        let logits = self.linear.forward(input)?;
        self.softmax.forward(&logits)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let mut g = self.softmax.backward(grad)?;
        g = self.linear.backward(&g)?;
        Ok(g)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.linear.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.linear.parameters_mut()
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
///     .with_expert_activation(ActivationType::ReLU);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MMoEConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Number of expert networks
    pub num_experts: usize,
    /// Number of tasks (and gates)
    pub num_tasks: usize,
    /// Hidden layer units for each expert
    pub expert_hidden_units: Vec<usize>,
    /// Activation function for expert hidden layers
    pub expert_activation: ActivationType,
    /// Output dimension of each expert (defaults to last hidden unit size or input_dim)
    pub expert_output_dim: Option<usize>,
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
            num_experts,
            num_tasks,
            expert_hidden_units: Vec::new(),
            expert_activation: ActivationType::ReLU,
            expert_output_dim: None,
        }
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
///     .with_expert_activation(ActivationType::ReLU);
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
    /// Expert output dimension
    expert_output_dim: usize,
    /// Configuration used to build this MMoE
    config: MMoEConfig,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Cached expert outputs for backward pass
    cached_expert_outputs: Vec<Tensor>,
    /// Cached gate outputs for backward pass
    cached_gate_outputs: Vec<Tensor>,
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
    ///     .with_expert_activation(ActivationType::ReLU);
    ///
    /// let mmoe = MMoE::from_config(config).unwrap();
    /// ```
    pub fn from_config(config: MMoEConfig) -> Result<Self, LayerError> {
        config.validate()?;

        let expert_output_dim = config.effective_expert_output_dim();

        // Create experts
        let experts: Vec<Expert> = (0..config.num_experts)
            .map(|_| {
                Expert::new(
                    config.input_dim,
                    &config.expert_hidden_units,
                    expert_output_dim,
                    config.expert_activation,
                )
            })
            .collect();

        // Create gates (one per task)
        let gates: Vec<Gate> = (0..config.num_tasks)
            .map(|_| Gate::new(config.input_dim, config.num_experts))
            .collect();

        Ok(Self {
            experts,
            gates,
            num_experts: config.num_experts,
            num_tasks: config.num_tasks,
            input_dim: config.input_dim,
            expert_output_dim,
            config,
            cached_input: None,
            cached_expert_outputs: Vec::new(),
            cached_gate_outputs: Vec::new(),
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
    /// let mmoe = MMoE::new(64, 4, 2, &[32, 32], ActivationType::ReLU).unwrap();
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
    /// let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
    /// let input = Tensor::rand(&[8, 64]);
    /// let outputs = mmoe.forward_multi(&input).unwrap();
    ///
    /// assert_eq!(outputs.len(), 2);
    /// ```
    pub fn forward_multi(&self, input: &Tensor) -> Result<Vec<Tensor>, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("MMoE expects 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[1],
            });
        }

        let batch_size = input.shape()[0];

        // Compute all expert outputs: [num_experts, batch_size, expert_output_dim]
        let expert_outputs: Vec<Tensor> = self
            .experts
            .iter()
            .map(|expert| expert.forward(input))
            .collect::<Result<Vec<_>, _>>()?;

        // Compute output for each task
        let mut task_outputs = Vec::with_capacity(self.num_tasks);

        for gate in &self.gates {
            // Compute gate weights: [batch_size, num_experts]
            let gate_weights = gate.forward(input)?;

            // Compute weighted sum of expert outputs
            // output = sum_e(gate_weight_e * expert_output_e)
            let task_output = self.weighted_sum_experts(&expert_outputs, &gate_weights, batch_size);
            task_outputs.push(task_output);
        }

        Ok(task_outputs)
    }

    /// Forward pass with training mode for multi-task output.
    ///
    /// Caches intermediate values for backward pass.
    pub fn forward_multi_train(&mut self, input: &Tensor) -> Result<Vec<Tensor>, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("MMoE expects 2D input, got {}D", input.ndim()),
            });
        }
        if input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[1],
            });
        }

        let batch_size = input.shape()[0];

        // Cache input
        self.cached_input = Some(input.clone());

        // Compute all expert outputs
        self.cached_expert_outputs.clear();
        for expert in &mut self.experts {
            let output = expert.forward_train(input)?;
            self.cached_expert_outputs.push(output);
        }

        // Compute gate outputs for each task
        self.cached_gate_outputs.clear();
        let mut gate_weights_list = Vec::with_capacity(self.num_tasks);

        for gate in &mut self.gates {
            let gate_weights = gate.forward_train(input)?;
            self.cached_gate_outputs.push(gate_weights.clone());
            gate_weights_list.push(gate_weights);
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
        let mut result = vec![0.0; batch_size * self.expert_output_dim];

        for (expert_idx, expert_output) in expert_outputs.iter().enumerate() {
            for b in 0..batch_size {
                let weight = gate_weights.data()[b * self.num_experts + expert_idx];
                for d in 0..self.expert_output_dim {
                    result[b * self.expert_output_dim + d] +=
                        weight * expert_output.data()[b * self.expert_output_dim + d];
                }
            }
        }

        Tensor::from_data(&[batch_size, self.expert_output_dim], result)
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
        if grads.len() != self.num_tasks {
            return Err(LayerError::BackwardError {
                message: format!(
                    "Expected {} gradients (one per task), got {}",
                    self.num_tasks,
                    grads.len()
                ),
            });
        }

        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let batch_size = input.shape()[0];

        // Initialize gradients
        let mut input_grad = Tensor::zeros(&[batch_size, self.input_dim]);
        let mut expert_grads: Vec<Tensor> = (0..self.num_experts)
            .map(|_| Tensor::zeros(&[batch_size, self.expert_output_dim]))
            .collect();

        // Backward through each task
        for (task_idx, grad) in grads.iter().enumerate() {
            let gate_weights = &self.cached_gate_outputs[task_idx];

            // Gradient w.r.t. expert outputs: d_expert_e = gate_weight_e * d_output
            for (expert_idx, expert_grad) in expert_grads.iter_mut().enumerate() {
                for b in 0..batch_size {
                    let weight = gate_weights.data()[b * self.num_experts + expert_idx];
                    for d in 0..self.expert_output_dim {
                        let grad_val = grad.data()[b * self.expert_output_dim + d];
                        expert_grad.data_mut()[b * self.expert_output_dim + d] += weight * grad_val;
                    }
                }
            }

            // Gradient w.r.t. gate weights: d_gate_e = expert_output_e dot d_output
            let mut gate_grad = vec![0.0; batch_size * self.num_experts];
            for expert_idx in 0..self.num_experts {
                let expert_output = &self.cached_expert_outputs[expert_idx];
                for b in 0..batch_size {
                    let mut dot = 0.0;
                    for d in 0..self.expert_output_dim {
                        dot += expert_output.data()[b * self.expert_output_dim + d]
                            * grad.data()[b * self.expert_output_dim + d];
                    }
                    gate_grad[b * self.num_experts + expert_idx] = dot;
                }
            }
            let gate_grad = Tensor::from_data(&[batch_size, self.num_experts], gate_grad);

            // Backward through gate
            let gate_input_grad = self.gates[task_idx].backward(&gate_grad)?;

            // Accumulate gate input gradient
            input_grad = input_grad.add(&gate_input_grad);
        }

        // Backward through experts
        for (expert_idx, expert_grad) in expert_grads.iter().enumerate() {
            let expert_input_grad = self.experts[expert_idx].backward(expert_grad)?;
            input_grad = input_grad.add(&expert_input_grad);
        }

        Ok(input_grad)
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expert_creation() {
        let expert = Expert::new(64, &[32, 16], 8, ActivationType::ReLU);
        assert_eq!(expert.input_dim(), 64);
        assert_eq!(expert.output_dim(), 8);
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(64, &[32], 16, ActivationType::ReLU);
        let input = Tensor::rand(&[8, 64]);
        let output = expert.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8, 16]);
    }

    #[test]
    fn test_expert_forward_invalid_input() {
        let expert = Expert::new(64, &[32], 16, ActivationType::ReLU);
        let input = Tensor::rand(&[8, 128]); // wrong dimension
        let result = expert.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_expert_backward() {
        let mut expert = Expert::new(64, &[32], 16, ActivationType::ReLU);
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
            .with_expert_activation(ActivationType::ReLU);

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
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
        assert_eq!(mmoe.num_experts(), 4);
        assert_eq!(mmoe.num_tasks(), 2);
        assert_eq!(mmoe.input_dim(), 64);
        assert_eq!(mmoe.expert_output_dim(), 32);
    }

    #[test]
    fn test_mmoe_forward_multi() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
        let input = Tensor::rand(&[8, 64]);
        let outputs = mmoe.forward_multi(&input).unwrap();

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[8, 32]);
        assert_eq!(outputs[1].shape(), &[8, 32]);
    }

    #[test]
    fn test_mmoe_forward() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
        let input = Tensor::rand(&[8, 64]);
        let output = mmoe.forward(&input).unwrap();

        assert_eq!(output.shape(), &[8, 32]);
    }

    #[test]
    fn test_mmoe_forward_invalid_input() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();

        // Wrong number of dimensions
        let input = Tensor::rand(&[8, 64, 1]);
        assert!(mmoe.forward(&input).is_err());

        // Wrong input dimension
        let input = Tensor::rand(&[8, 128]);
        assert!(mmoe.forward(&input).is_err());
    }

    #[test]
    fn test_mmoe_backward_multi() {
        let mut mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
        let input = Tensor::rand(&[8, 64]);

        let _outputs = mmoe.forward_multi_train(&input).unwrap();

        let grads = vec![Tensor::ones(&[8, 32]), Tensor::ones(&[8, 32])];
        let input_grad = mmoe.backward_multi(&grads).unwrap();

        assert_eq!(input_grad.shape(), &[8, 64]);
    }

    #[test]
    fn test_mmoe_parameters() {
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
        let params = mmoe.parameters();

        // 4 experts: each has 2 dense layers (64->32, 32->32), each with weights+bias = 4*4 = 16
        // 2 gates: each has 1 dense layer (64->4) with weights+bias = 2*2 = 4
        // Total: 16 + 4 = 20 parameters
        assert_eq!(params.len(), 20);
    }

    #[test]
    fn test_mmoe_training_mode() {
        let mut mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
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
            ActivationType::ReLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::GELU,
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
        let mmoe = MMoE::new(64, 4, 2, &[32], ActivationType::ReLU).unwrap();
        assert_eq!(mmoe.name(), "MMoE");
    }

    #[test]
    fn test_expert_no_hidden_layers() {
        let expert = Expert::new(64, &[], 32, ActivationType::ReLU);
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
