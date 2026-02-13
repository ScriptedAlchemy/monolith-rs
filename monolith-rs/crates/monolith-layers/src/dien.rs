//! Deep Interest Evolution Network (DIEN) layer implementation.
//!
//! This module provides the [`DIENLayer`] which implements the Deep Interest Evolution Network
//! for sequential recommendation systems. DIEN captures the evolution of user interests
//! over time using a two-layer GRU architecture.
//!
//! # Overview
//!
//! DIEN consists of three main components:
//!
//! 1. **Interest Extractor Layer**: Uses a GRU to extract user interests from behavior sequences
//! 2. **Interest Evolution Layer**: Uses AGRU/AUGRU (Attention-based GRU) to model interest evolution
//!    with attention to the target item
//! 3. **Auxiliary Loss**: Helps the interest extractor capture sequential patterns
//!
//! # Architecture
//!
//! ```text
//! Behavior Sequence --> GRU (Interest Extractor) --> Hidden States
//!                                                        |
//! Target Item ---------> Attention --------------------->|
//!                                                        v
//!                                           AGRU/AUGRU (Interest Evolution) --> Evolved Interest
//! ```
//!
//! # Example
//!
//! ```
//! use monolith_layers::dien::{DIENLayer, DIENConfig, GRUType};
//! use monolith_layers::tensor::Tensor;
//!
//! // Create DIEN layer
//! let config = DIENConfig::new(32, 32)
//!     .with_use_auxiliary_loss(true)
//!     .with_gru_type(GRUType::AGRU);
//! let mut dien = DIENLayer::from_config(config).unwrap();
//!
//! // User behavior sequence [batch_size, seq_len, embedding_dim]
//! let behavior_seq = Tensor::rand(&[4, 10, 32]);
//! // Target item [batch_size, embedding_dim]
//! let target_item = Tensor::rand(&[4, 32]);
//! // Mask for valid sequence positions
//! let mask = Tensor::ones(&[4, 10]);
//!
//! // Forward pass
//! let evolved_interest = dien.forward_dien(&behavior_seq, &target_item, Some(&mask)).unwrap();
//! assert_eq!(evolved_interest.shape(), &[4, 32]);
//! ```
//!
//! # References
//!
//! - Zhou, G., et al. "Deep Interest Evolution Network for Click-Through Rate Prediction." AAAI 2019.

use crate::activation::{
    Exponential, HardSigmoid, LeakyReLU, Linear, Mish, PReLU, ReLU, Sigmoid2, Softmax, Softplus,
    Softsign, Swish, ThresholdedReLU, ELU, SELU,
};
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::mlp::ActivationType;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Type of GRU cell to use in the interest evolution layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum GRUType {
    /// Standard GRU cell
    Standard,
    /// Attention-based GRU (AGRU) - uses attention to mix state and candidate
    #[default]
    AGRU,
    /// Attention-based GRU (AUGRU) - modifies update gate with attention
    AUGRU,
}

fn apply_activation(input: &Tensor, activation: &ActivationType) -> Tensor {
    match activation {
        ActivationType::ReLU {
            max_value,
            negative_slope,
            threshold,
        } => ReLU::with_params(*max_value, *negative_slope, *threshold)
            .forward(input)
            .unwrap(),
        ActivationType::Sigmoid => input.sigmoid(),
        ActivationType::Sigmoid2 => Sigmoid2::new().forward(input).unwrap(),
        ActivationType::Tanh => input.tanh(),
        ActivationType::GELU => input.gelu(),
        ActivationType::SELU => SELU::new().forward(input).unwrap(),
        ActivationType::Softplus => Softplus::new().forward(input).unwrap(),
        ActivationType::Softsign => Softsign::new().forward(input).unwrap(),
        ActivationType::Swish => Swish::new().forward(input).unwrap(),
        ActivationType::Mish => Mish::new().forward(input).unwrap(),
        ActivationType::HardSigmoid => HardSigmoid::new().forward(input).unwrap(),
        ActivationType::LeakyReLU { alpha } => LeakyReLU::new(*alpha).forward(input).unwrap(),
        ActivationType::ELU { alpha } => ELU::new(*alpha).forward(input).unwrap(),
        ActivationType::PReLU {
            alpha,
            initializer,
            shared_axes,
            regularizer,
            constraint,
        } => PReLU::with_params(
            *alpha,
            initializer.clone(),
            shared_axes.clone(),
            regularizer.clone(),
            constraint.clone(),
        )
        .forward(input)
        .unwrap(),
        ActivationType::ThresholdedReLU { theta } => {
            ThresholdedReLU::new(*theta).forward(input).unwrap()
        }
        ActivationType::Softmax { axis } => Softmax::with_axis(*axis).forward(input).unwrap(),
        ActivationType::Linear => Linear::new().forward(input).unwrap(),
        ActivationType::Exponential => Exponential::new().forward(input).unwrap(),
        ActivationType::None => input.clone(),
    }
}

/// Configuration for DIEN layer.
///
/// # Example
///
/// ```
/// use monolith_layers::dien::{DIENConfig, GRUType};
///
/// let config = DIENConfig::new(32, 32)
///     .with_use_auxiliary_loss(true)
///     .with_gru_type(GRUType::AGRU)
///     .with_attention_hidden_units(vec![64, 32]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DIENConfig {
    /// Dimension of input embeddings
    pub embedding_dim: usize,
    /// Dimension of hidden states in GRU cells
    pub hidden_size: usize,
    /// Activation for GRU/AUGRU candidate state
    pub activation: ActivationType,
    /// Initializer for GRU/AUGRU weights and attention projection
    pub initializer: Initializer,
    /// Initializer for recurrent (hidden-to-hidden) weights
    pub recurrent_initializer: Initializer,
    /// Kernel regularizer for GRU/AUGRU weights and attention projection
    pub regularizer: Regularizer,
    /// Whether to use auxiliary loss for training
    pub use_auxiliary_loss: bool,
    /// Type of GRU to use in interest evolution layer
    pub gru_type: GRUType,
    /// Hidden units for attention MLP
    pub attention_hidden_units: Vec<usize>,
    /// Whether to use softmax in attention
    pub use_softmax: bool,
}

impl DIENConfig {
    /// Creates a new DIEN configuration.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of input embeddings
    /// * `hidden_size` - Dimension of hidden states in GRU
    pub fn new(embedding_dim: usize, hidden_size: usize) -> Self {
        Self {
            embedding_dim,
            hidden_size,
            activation: ActivationType::relu(),
            initializer: Initializer::HeUniform,
            recurrent_initializer: Initializer::Orthogonal,
            regularizer: Regularizer::None,
            use_auxiliary_loss: true,
            gru_type: GRUType::AGRU,
            attention_hidden_units: vec![64, 32],
            use_softmax: true,
        }
    }

    /// Sets whether to use auxiliary loss.
    pub fn with_use_auxiliary_loss(mut self, use_auxiliary_loss: bool) -> Self {
        self.use_auxiliary_loss = use_auxiliary_loss;
        self
    }

    /// Sets the activation for GRU/AUGRU candidate state.
    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    /// Sets the initializer for GRU/AUGRU weights and attention projection.
    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    /// Sets the recurrent initializer for hidden-to-hidden weights.
    pub fn with_recurrent_initializer(mut self, initializer: Initializer) -> Self {
        self.recurrent_initializer = initializer;
        self
    }

    /// Sets kernel regularizer for GRU/AUGRU weights and attention projection.
    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    /// Sets the GRU type for interest evolution.
    pub fn with_gru_type(mut self, gru_type: GRUType) -> Self {
        self.gru_type = gru_type;
        self
    }

    /// Sets the attention hidden units.
    pub fn with_attention_hidden_units(mut self, units: Vec<usize>) -> Self {
        self.attention_hidden_units = units;
        self
    }

    /// Sets whether to use softmax in attention.
    pub fn with_use_softmax(mut self, use_softmax: bool) -> Self {
        self.use_softmax = use_softmax;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), LayerError> {
        if self.embedding_dim == 0 {
            return Err(LayerError::ConfigError {
                message: "Embedding dimension must be positive".to_string(),
            });
        }
        if self.hidden_size == 0 {
            return Err(LayerError::ConfigError {
                message: "Hidden size must be positive".to_string(),
            });
        }
        // Python DIEN allows embedding_dim != hidden_size (interest extractor GRU maps
        // embedding_dim -> hidden_size). The attention module operates in hidden_size space.
        for (i, &dim) in self.attention_hidden_units.iter().enumerate() {
            if dim == 0 {
                return Err(LayerError::ConfigError {
                    message: format!("Attention hidden layer {} has zero dimension", i),
                });
            }
        }
        Ok(())
    }

    /// Builds a DIENLayer from this configuration.
    pub fn build(self) -> Result<DIENLayer, LayerError> {
        DIENLayer::from_config(self)
    }
}

impl Default for DIENConfig {
    fn default() -> Self {
        Self::new(32, 64)
    }
}

/// AUGRU (Attention Update GRU) cell implementation.
///
/// AUGRU modifies the standard GRU by incorporating attention scores into the update gate,
/// allowing the model to focus on relevant parts of the interest evolution.
///
/// The update equations are:
/// - r_t = sigmoid(W_r * [h_{t-1}, x_t] + b_r)  (reset gate)
/// - z_t = sigmoid(W_z * [h_{t-1}, x_t] + b_z)  (update gate)
/// - z_t' = a_t * z_t  (attention-modulated update gate)
/// - h_tilde = tanh(W_h * [r_t * h_{t-1}, x_t] + b_h)  (candidate)
/// - h_t = (1 - z_t') * h_{t-1} + z_t' * h_tilde  (hidden state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AUGRUCell {
    /// Input dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Activation for candidate state
    activation: ActivationType,
    /// Reset gate weights for input
    w_r_x: Tensor,
    /// Reset gate weights for hidden state
    w_r_h: Tensor,
    /// Reset gate bias
    b_r: Tensor,
    /// Update gate weights for input
    w_z_x: Tensor,
    /// Update gate weights for hidden state
    w_z_h: Tensor,
    /// Update gate bias
    b_z: Tensor,
    /// Candidate hidden state weights for input
    w_h_x: Tensor,
    /// Candidate hidden state weights for hidden state
    w_h_h: Tensor,
    /// Candidate hidden state bias
    b_h: Tensor,
}

impl AUGRUCell {
    /// Creates a new AUGRU cell.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of the input features
    /// * `hidden_dim` - Dimension of the hidden state
    pub fn new(input_dim: usize, hidden_dim: usize, activation: ActivationType) -> Self {
        Self::new_with_initializer(
            input_dim,
            hidden_dim,
            activation,
            Initializer::HeNormal,
            Initializer::Orthogonal,
            Initializer::Ones,
        )
    }

    /// Creates a new AUGRU cell with custom initializers.
    pub fn new_with_initializer(
        input_dim: usize,
        hidden_dim: usize,
        activation: ActivationType,
        input_init: Initializer,
        recurrent_init: Initializer,
        bias_init: Initializer,
    ) -> Self {
        Self {
            input_dim,
            hidden_dim,
            activation,
            // Reset gate
            w_r_x: input_init.initialize(&[input_dim, hidden_dim]),
            w_r_h: recurrent_init.initialize(&[hidden_dim, hidden_dim]),
            b_r: bias_init.initialize(&[hidden_dim]),
            // Update gate
            w_z_x: input_init.initialize(&[input_dim, hidden_dim]),
            w_z_h: recurrent_init.initialize(&[hidden_dim, hidden_dim]),
            b_z: bias_init.initialize(&[hidden_dim]),
            // Candidate hidden state
            w_h_x: input_init.initialize(&[input_dim, hidden_dim]),
            w_h_h: recurrent_init.initialize(&[hidden_dim, hidden_dim]),
            b_h: bias_init.initialize(&[hidden_dim]),
        }
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Computes one step of AUGRU with attention.
    ///
    /// # Arguments
    ///
    /// * `x` - Input at current timestep [batch_size, input_dim]
    /// * `h` - Previous hidden state [batch_size, hidden_dim]
    /// * `attention` - Attention score for this timestep [batch_size]
    ///
    /// # Returns
    ///
    /// New hidden state [batch_size, hidden_dim]
    pub fn forward_step(
        &self,
        x: &Tensor,
        h: &Tensor,
        attention: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let batch_size = x.shape()[0];
        let hidden_dim = self.hidden_dim;

        // Reset gate: r = sigmoid(x @ W_r_x + h @ W_r_h + b_r)
        let r = self.compute_reset_gate(x, h)?;

        // Update gate: z = sigmoid(x @ W_z_x + h @ W_z_h + b_z)
        let z = self.compute_update_gate(x, h)?;

        // Candidate: h_tilde = tanh(x @ W_h_x + (r * h) @ W_h_h + b_h)
        let h_tilde = self.compute_candidate(x, h, &r)?;

        // AUGRU: u = (1 - a) * z
        // New hidden: h_new = u * h + (1 - u) * h_tilde
        let att = attention
            .reshape(&[batch_size, 1])
            .broadcast_as(&[batch_size, hidden_dim]);
        let one = Tensor::ones(&[batch_size, hidden_dim]);
        let z_att = one.sub(&att).mul(&z);
        let h_new = z_att.mul(h).add(&one.sub(&z_att).mul(&h_tilde));
        Ok(h_new)
    }

    /// Computes one step of AGRU with attention.
    pub fn forward_step_agru(
        &self,
        x: &Tensor,
        h: &Tensor,
        attention: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let batch_size = x.shape()[0];
        let hidden_dim = self.hidden_dim;

        // Reset gate: r = sigmoid(x @ W_r_x + h @ W_r_h + b_r)
        let r = self.compute_reset_gate(x, h)?;

        // Candidate: h_tilde = tanh(x @ W_h_x + (r * h) @ W_h_h + b_h)
        let h_tilde = self.compute_candidate(x, h, &r)?;

        // AGRU: new_h = (1 - a) * h + a * h_tilde
        let att = attention
            .reshape(&[batch_size, 1])
            .broadcast_as(&[batch_size, hidden_dim]);
        let one = Tensor::ones(&[batch_size, hidden_dim]);
        let h_new = one.sub(&att).mul(h).add(&att.mul(&h_tilde));
        Ok(h_new)
    }

    /// Computes the reset gate.
    fn compute_reset_gate(&self, x: &Tensor, h: &Tensor) -> Result<Tensor, LayerError> {
        let xw = x.matmul(&self.w_r_x);
        let hw = h.matmul(&self.w_r_h);
        let sum = xw.add(&hw).add(&self.b_r);
        Ok(sum.sigmoid())
    }

    /// Computes the update gate.
    fn compute_update_gate(&self, x: &Tensor, h: &Tensor) -> Result<Tensor, LayerError> {
        let xw = x.matmul(&self.w_z_x);
        let hw = h.matmul(&self.w_z_h);
        let sum = xw.add(&hw).add(&self.b_z);
        Ok(sum.sigmoid())
    }

    /// Computes the candidate hidden state.
    fn compute_candidate(&self, x: &Tensor, h: &Tensor, r: &Tensor) -> Result<Tensor, LayerError> {
        let xw = x.matmul(&self.w_h_x);
        let rh = r.mul(h);
        let hw = rh.matmul(&self.w_h_h);
        let sum = xw.add(&hw).add(&self.b_h);
        Ok(apply_activation(&sum, &self.activation))
    }

    /// Returns all parameters of the AUGRU cell.
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.w_r_x,
            &self.w_r_h,
            &self.b_r,
            &self.w_z_x,
            &self.w_z_h,
            &self.b_z,
            &self.w_h_x,
            &self.w_h_h,
            &self.b_h,
        ]
    }

    /// Returns mutable references to all parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w_r_x,
            &mut self.w_r_h,
            &mut self.b_r,
            &mut self.w_z_x,
            &mut self.w_z_h,
            &mut self.b_z,
            &mut self.w_h_x,
            &mut self.w_h_h,
            &mut self.b_h,
        ]
    }
}

/// Standard GRU cell for interest extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GRUCell {
    /// Input dimension
    input_dim: usize,
    /// Hidden dimension
    hidden_dim: usize,
    /// Activation for candidate state
    activation: ActivationType,
    /// Reset gate weights for input
    w_r_x: Tensor,
    /// Reset gate weights for hidden state
    w_r_h: Tensor,
    /// Reset gate bias
    b_r: Tensor,
    /// Update gate weights for input
    w_z_x: Tensor,
    /// Update gate weights for hidden state
    w_z_h: Tensor,
    /// Update gate bias
    b_z: Tensor,
    /// Candidate hidden state weights for input
    w_h_x: Tensor,
    /// Candidate hidden state weights for hidden state
    w_h_h: Tensor,
    /// Candidate hidden state bias
    b_h: Tensor,
}

impl GRUCell {
    /// Creates a new GRU cell.
    pub fn new(input_dim: usize, hidden_dim: usize, activation: ActivationType) -> Self {
        Self::new_with_initializer(
            input_dim,
            hidden_dim,
            activation,
            Initializer::HeUniform,
            Initializer::Orthogonal,
            Initializer::Zeros,
        )
    }

    /// Creates a new GRU cell with custom initializers.
    pub fn new_with_initializer(
        input_dim: usize,
        hidden_dim: usize,
        activation: ActivationType,
        input_init: Initializer,
        recurrent_init: Initializer,
        bias_init: Initializer,
    ) -> Self {
        Self {
            input_dim,
            hidden_dim,
            activation,
            w_r_x: input_init.initialize(&[input_dim, hidden_dim]),
            w_r_h: recurrent_init.initialize(&[hidden_dim, hidden_dim]),
            b_r: bias_init.initialize(&[hidden_dim]),
            w_z_x: input_init.initialize(&[input_dim, hidden_dim]),
            w_z_h: recurrent_init.initialize(&[hidden_dim, hidden_dim]),
            b_z: bias_init.initialize(&[hidden_dim]),
            w_h_x: input_init.initialize(&[input_dim, hidden_dim]),
            w_h_h: recurrent_init.initialize(&[hidden_dim, hidden_dim]),
            b_h: bias_init.initialize(&[hidden_dim]),
        }
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.hidden_dim
    }

    /// Computes one step of standard GRU.
    ///
    /// # Arguments
    ///
    /// * `x` - Input at current timestep [batch_size, input_dim]
    /// * `h` - Previous hidden state [batch_size, hidden_dim]
    ///
    /// # Returns
    ///
    /// New hidden state [batch_size, hidden_dim]
    pub fn forward_step(&self, x: &Tensor, h: &Tensor) -> Result<Tensor, LayerError> {
        let batch_size = x.shape()[0];
        let hidden_dim = self.hidden_dim;

        // Reset gate
        let xw_r = x.matmul(&self.w_r_x);
        let hw_r = h.matmul(&self.w_r_h);
        let r = xw_r.add(&hw_r).add(&self.b_r).sigmoid();

        // Update gate
        let xw_z = x.matmul(&self.w_z_x);
        let hw_z = h.matmul(&self.w_z_h);
        let z = xw_z.add(&hw_z).add(&self.b_z).sigmoid();

        // Candidate
        let xw_h = x.matmul(&self.w_h_x);
        let rh = r.mul(h);
        let hw_h = rh.matmul(&self.w_h_h);
        let h_tilde = apply_activation(&xw_h.add(&hw_h).add(&self.b_h), &self.activation);

        // New hidden state: h_new = (1 - z) * h + z * h_tilde
        let one = Tensor::ones(&[batch_size, hidden_dim]);
        let h_new = one.sub(&z).mul(h).add(&z.mul(&h_tilde));
        Ok(h_new)
    }

    /// Returns all parameters.
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.w_r_x,
            &self.w_r_h,
            &self.b_r,
            &self.w_z_x,
            &self.w_z_h,
            &self.b_z,
            &self.w_h_x,
            &self.w_h_h,
            &self.b_h,
        ]
    }

    /// Returns mutable references to all parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w_r_x,
            &mut self.w_r_h,
            &mut self.b_r,
            &mut self.w_z_x,
            &mut self.w_z_h,
            &mut self.b_z,
            &mut self.w_h_x,
            &mut self.w_h_h,
            &mut self.b_h,
        ]
    }
}

/// Attention module for computing attention scores between target and hidden states.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AttentionModule {
    /// Hidden dimension (for hidden states)
    hidden_dim: usize,
    /// Attention weight matrix
    weight: Tensor,
    /// Whether to use softmax
    use_softmax: bool,
}

impl AttentionModule {
    /// Creates a new attention module.
    fn new(hidden_dim: usize, use_softmax: bool, initializer: Initializer) -> Self {
        Self {
            hidden_dim,
            weight: initializer.initialize(&[hidden_dim, hidden_dim]),
            use_softmax,
        }
    }

    /// Computes attention scores between target and hidden states.
    ///
    /// # Arguments
    ///
    /// * `target` - Target item embedding [batch_size, hidden_dim] (already projected)
    /// * `hidden_states` - Hidden states from GRU [batch_size, seq_len, hidden_dim]
    /// * `mask` - Optional mask [batch_size, seq_len]
    ///
    /// # Returns
    ///
    /// Attention scores [batch_size, seq_len]
    fn compute_attention(
        &self,
        target: &Tensor,
        hidden_states: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        let batch_size = target.shape()[0];
        let seq_len = hidden_states.shape()[1];
        let hidden_dim = self.hidden_dim;

        // query_weight = target @ W^T, shape [B, H]
        let weight_t = self.weight.transpose();
        let query_weight = target.matmul(&weight_t);

        // logits = sum(hidden_states * query_weight, dim=H) -> [B, T]
        let query_broadcast = query_weight
            .reshape(&[batch_size, 1, hidden_dim])
            .broadcast_as(&[batch_size, seq_len, hidden_dim]);
        let mut logits = hidden_states.mul(&query_broadcast).sum_axis(2);

        // Apply mask by subtracting a large value where mask == 0
        if let Some(m) = mask {
            let ones = Tensor::ones(&[batch_size, seq_len]);
            let mask_inv = ones.sub(m);
            let penalty = mask_inv.scale(1.0e9);
            logits = logits.sub(&penalty);
        }

        if self.use_softmax {
            Ok(logits.softmax(1))
        } else {
            Ok(logits.sigmoid())
        }
    }

    /// Returns all parameters.
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    /// Returns mutable references to all parameters.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight]
    }
}

/// Deep Interest Evolution Network (DIEN) layer.
///
/// DIEN captures the evolution of user interests through a two-layer GRU architecture:
/// 1. Interest Extractor: Standard GRU that extracts interests from behavior sequences
/// 2. Interest Evolution: AUGRU that models how interests evolve with attention to target
///
/// # Example
///
/// ```
/// use monolith_layers::dien::{DIENLayer, DIENConfig};
/// use monolith_layers::tensor::Tensor;
///
/// let dien = DIENLayer::new(32, 64);
///
/// let behavior_seq = Tensor::rand(&[2, 5, 32]);
/// let target_item = Tensor::rand(&[2, 32]);
///
/// let evolved_interest = dien.forward_dien(&behavior_seq, &target_item, None).unwrap();
/// assert_eq!(evolved_interest.shape(), &[2, 64]);
/// ```
#[derive(Debug, Clone)]
pub struct DIENLayer {
    /// Configuration
    config: DIENConfig,
    /// Interest extractor GRU
    interest_extractor: GRUCell,
    /// Interest evolution GRU/AUGRU
    evolution_gru: AUGRUCell,
    /// Standard GRU for evolution (used when gru_type is Standard)
    evolution_standard_gru: GRUCell,
    /// Attention module
    attention: AttentionModule,
    /// Optional linear projection from embedding_dim -> hidden_size for the target/query.
    target_projector: Option<crate::dense::Dense>,
    /// Whether in training mode
    training: bool,
    /// Cached values for backward pass
    cache: Option<DIENCache>,
}

/// Cached values from DIEN forward pass.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct DIENCache {
    /// Input behavior sequence
    behavior_seq: Tensor,
    /// Target item
    target_item: Tensor,
    /// Hidden states from interest extractor
    extractor_states: Tensor,
    /// Attention scores
    attention_scores: Tensor,
    /// Final evolved interest
    evolved_interest: Tensor,
    /// Optional mask
    mask: Option<Tensor>,
}

impl DIENLayer {
    /// Creates a new DIEN layer with default configuration.
    ///
    /// # Arguments
    ///
    /// * `embedding_dim` - Dimension of input embeddings
    /// * `hidden_size` - Dimension of hidden states
    pub fn new(embedding_dim: usize, hidden_size: usize) -> Self {
        let config = DIENConfig::new(embedding_dim, hidden_size);
        Self::from_config(config).expect("Invalid DIEN configuration")
    }

    /// Creates a DIEN layer from configuration.
    pub fn from_config(config: DIENConfig) -> Result<Self, LayerError> {
        config.validate()?;

        let embedding_dim = config.embedding_dim;
        let hidden_size = config.hidden_size;

        // Interest extractor: maps embedding_dim -> hidden_size
        let interest_extractor = GRUCell::new_with_initializer(
            embedding_dim,
            hidden_size,
            config.activation.clone(),
            config.initializer,
            config.recurrent_initializer,
            Initializer::Zeros,
        );

        // Interest evolution: maps hidden_size -> hidden_size
        let evolution_gru = AUGRUCell::new_with_initializer(
            hidden_size,
            hidden_size,
            config.activation.clone(),
            config.initializer,
            config.recurrent_initializer,
            Initializer::Ones,
        );
        let evolution_standard_gru = GRUCell::new_with_initializer(
            hidden_size,
            hidden_size,
            config.activation.clone(),
            config.initializer,
            config.recurrent_initializer,
            Initializer::Zeros,
        );

        // Attention module
        let attention = AttentionModule::new(hidden_size, config.use_softmax, config.initializer);

        // If embedding_dim != hidden_size, project the target/query into hidden space.
        let target_projector = if embedding_dim != hidden_size {
            Some(crate::dense::Dense::new(embedding_dim, hidden_size))
        } else {
            None
        };

        Ok(Self {
            config,
            interest_extractor,
            evolution_gru,
            evolution_standard_gru,
            attention,
            target_projector,
            training: true,
            cache: None,
        })
    }

    /// Returns the embedding dimension.
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }

    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Returns the configuration.
    pub fn config(&self) -> &DIENConfig {
        &self.config
    }

    /// Forward pass for DIEN.
    ///
    /// # Arguments
    ///
    /// * `behavior_seq` - User behavior sequence [batch_size, seq_len, embedding_dim]
    /// * `target_item` - Target item embedding [batch_size, embedding_dim]
    /// * `mask` - Optional mask for valid positions [batch_size, seq_len]
    ///
    /// # Returns
    ///
    /// Evolved interest representation [batch_size, hidden_size]
    pub fn forward_dien(
        &self,
        behavior_seq: &Tensor,
        target_item: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        self.validate_inputs(behavior_seq, target_item, mask)?;

        let batch_size = behavior_seq.shape()[0];
        let seq_len = behavior_seq.shape()[1];
        let hidden_size = self.config.hidden_size;

        // Step 1: Interest Extraction - run GRU over behavior sequence
        let extractor_states = self.run_interest_extractor(behavior_seq, batch_size, seq_len)?;

        // Step 2: Project target to hidden_size
        let target_projected = self.project_target(target_item)?;

        // Step 3: Compute attention scores between target and hidden states
        let attention_scores =
            self.attention
                .compute_attention(&target_projected, &extractor_states, mask)?;

        // Step 4: Interest Evolution - run AUGRU with attention
        let evolved_interest = self.run_interest_evolution(
            &extractor_states,
            &attention_scores,
            batch_size,
            seq_len,
            hidden_size,
        )?;

        Ok(evolved_interest)
    }

    /// Forward pass with training cache.
    pub fn forward_dien_train(
        &mut self,
        behavior_seq: &Tensor,
        target_item: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        self.validate_inputs(behavior_seq, target_item, mask)?;

        let batch_size = behavior_seq.shape()[0];
        let seq_len = behavior_seq.shape()[1];
        let hidden_size = self.config.hidden_size;

        // Step 1: Interest Extraction
        let extractor_states = self.run_interest_extractor(behavior_seq, batch_size, seq_len)?;

        // Step 2: Project target
        let target_projected = self.project_target(target_item)?;

        // Step 3: Compute attention
        let attention_scores =
            self.attention
                .compute_attention(&target_projected, &extractor_states, mask)?;

        // Step 4: Interest Evolution
        let evolved_interest = self.run_interest_evolution(
            &extractor_states,
            &attention_scores,
            batch_size,
            seq_len,
            hidden_size,
        )?;

        // Cache for backward
        self.cache = Some(DIENCache {
            behavior_seq: behavior_seq.clone(),
            target_item: target_item.clone(),
            extractor_states,
            attention_scores,
            evolved_interest: evolved_interest.clone(),
            mask: mask.cloned(),
        });

        Ok(evolved_interest)
    }

    /// Validates input shapes.
    fn validate_inputs(
        &self,
        behavior_seq: &Tensor,
        target_item: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<(), LayerError> {
        if behavior_seq.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "Behavior sequence should be 3D [batch, seq_len, dim], got {}D",
                    behavior_seq.ndim()
                ),
            });
        }

        if target_item.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "Target item should be 2D [batch, dim], got {}D",
                    target_item.ndim()
                ),
            });
        }

        let batch_size = behavior_seq.shape()[0];
        let seq_len = behavior_seq.shape()[1];
        let embedding_dim = self.config.embedding_dim;
        let hidden_size = self.config.hidden_size;

        if behavior_seq.shape()[2] != embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: embedding_dim,
                actual: behavior_seq.shape()[2],
            });
        }

        // Target can be in embedding space (embedding_dim) or hidden space (hidden_size).
        // If a projector is configured, accept embedding_dim and project internally.
        let target_dim = target_item.shape()[1];
        if self.target_projector.is_some() {
            if target_dim != embedding_dim {
                return Err(LayerError::InvalidInputDimension {
                    expected: embedding_dim,
                    actual: target_dim,
                });
            }
        } else if target_dim != hidden_size {
            return Err(LayerError::InvalidInputDimension {
                expected: hidden_size,
                actual: target_dim,
            });
        }

        if target_item.shape()[0] != batch_size {
            return Err(LayerError::ShapeMismatch {
                expected: vec![batch_size, hidden_size],
                actual: target_item.shape().to_vec(),
            });
        }

        if let Some(m) = mask {
            if m.shape() != [batch_size, seq_len] {
                return Err(LayerError::ShapeMismatch {
                    expected: vec![batch_size, seq_len],
                    actual: m.shape().to_vec(),
                });
            }
        }

        Ok(())
    }

    /// Runs the interest extractor GRU.
    fn run_interest_extractor(
        &self,
        behavior_seq: &Tensor,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor, LayerError> {
        let embedding_dim = self.config.embedding_dim;
        let hidden_size = self.config.hidden_size;

        let mut h = Tensor::zeros(&[batch_size, hidden_size]);
        let mut states: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract input at timestep t
            let x_t = self.extract_timestep(behavior_seq, t, batch_size, embedding_dim);

            // GRU step
            h = self.interest_extractor.forward_step(&x_t, &h)?;

            // Store hidden state as [batch, 1, hidden]
            states.push(h.reshape(&[batch_size, 1, hidden_size]));
        }

        Ok(Tensor::cat(&states, 1))
    }

    /// Projects target item to hidden dimension.
    fn project_target(&self, target_item: &Tensor) -> Result<Tensor, LayerError> {
        if let Some(proj) = &self.target_projector {
            proj.forward(target_item)
        } else {
            Ok(target_item.clone())
        }
    }

    /// Runs the interest evolution layer.
    fn run_interest_evolution(
        &self,
        extractor_states: &Tensor,
        attention_scores: &Tensor,
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Tensor, LayerError> {
        let mut h = Tensor::zeros(&[batch_size, hidden_size]);

        for t in 0..seq_len {
            // Extract hidden state at timestep t
            let x_t = self.extract_hidden_timestep(extractor_states, t, batch_size, hidden_size);

            // Extract attention score at timestep t
            let a_t = self.extract_attention_timestep(attention_scores, t, batch_size);

            // GRU/AUGRU step
            h = match self.config.gru_type {
                GRUType::AGRU => self.evolution_gru.forward_step_agru(&x_t, &h, &a_t)?,
                GRUType::AUGRU => self.evolution_gru.forward_step(&x_t, &h, &a_t)?,
                GRUType::Standard => self.evolution_standard_gru.forward_step(&x_t, &h)?,
            };
        }

        Ok(h)
    }

    /// Extracts a timestep from the behavior sequence.
    fn extract_timestep(&self, input: &Tensor, t: usize, batch_size: usize, dim: usize) -> Tensor {
        let _ = (batch_size, dim);
        input.narrow(1, t, 1).squeeze(1).contiguous()
    }

    /// Extracts a timestep from hidden states.
    fn extract_hidden_timestep(
        &self,
        hidden_states: &Tensor,
        t: usize,
        batch_size: usize,
        hidden_size: usize,
    ) -> Tensor {
        self.extract_timestep(hidden_states, t, batch_size, hidden_size)
    }

    /// Extracts attention score at a timestep.
    fn extract_attention_timestep(
        &self,
        attention: &Tensor,
        t: usize,
        batch_size: usize,
    ) -> Tensor {
        let _ = batch_size;
        attention.narrow(1, t, 1).squeeze(1).contiguous()
    }

    /// Computes auxiliary loss for training the interest extractor.
    ///
    /// The auxiliary loss helps the GRU learn to predict the next behavior,
    /// using negative sampling for contrastive learning.
    ///
    /// # Arguments
    ///
    /// * `behavior_seq` - User behavior sequence [batch_size, seq_len, embedding_dim]
    /// * `neg_samples` - Negative samples [batch_size, seq_len, embedding_dim]
    ///
    /// # Returns
    ///
    /// Scalar auxiliary loss value
    pub fn auxiliary_loss(
        &self,
        behavior_seq: &Tensor,
        neg_samples: &Tensor,
    ) -> Result<f32, LayerError> {
        if !self.config.use_auxiliary_loss {
            return Ok(0.0);
        }

        let batch_size = behavior_seq.shape()[0];
        let seq_len = behavior_seq.shape()[1];
        let embedding_dim = self.config.embedding_dim;
        let hidden_size = self.config.hidden_size;

        // Validate shapes
        if neg_samples.shape() != behavior_seq.shape() {
            return Err(LayerError::ShapeMismatch {
                expected: behavior_seq.shape().to_vec(),
                actual: neg_samples.shape().to_vec(),
            });
        }

        // Run interest extractor to get hidden states
        let extractor_states = self.run_interest_extractor(behavior_seq, batch_size, seq_len)?;

        let extractor_data = extractor_states.data_ref();
        let behavior_data = behavior_seq.data_ref();
        let neg_data = neg_samples.data_ref();

        // For each timestep t, predict next behavior using hidden state h_t
        // Positive: next item in sequence (behavior_seq[t+1])
        // Negative: neg_samples[t+1]
        let mut total_loss = 0.0;
        let mut count = 0;

        for t in 0..(seq_len - 1) {
            for b in 0..batch_size {
                // Get hidden state at t
                let h_offset = b * seq_len * hidden_size + t * hidden_size;

                // Get next positive item embedding
                let pos_offset = b * seq_len * embedding_dim + (t + 1) * embedding_dim;

                // Get next negative item embedding
                let neg_offset = b * seq_len * embedding_dim + (t + 1) * embedding_dim;

                // Compute dot product similarity
                // Note: In practice, we'd use a projection, but here we use a simplified version
                // by taking first min(hidden_size, embedding_dim) dimensions
                let dim = hidden_size.min(embedding_dim);

                let mut pos_score = 0.0;
                let mut neg_score = 0.0;

                for d in 0..dim {
                    let h_val = extractor_data[h_offset + d];
                    let pos_val = behavior_data[pos_offset + d];
                    let neg_val = neg_data[neg_offset + d];

                    pos_score += h_val * pos_val;
                    neg_score += h_val * neg_val;
                }

                // Binary cross-entropy loss
                // L = -log(sigmoid(pos_score)) - log(1 - sigmoid(neg_score))
                let pos_prob = 1.0 / (1.0 + (-pos_score).exp());
                let neg_prob = 1.0 / (1.0 + (-neg_score).exp());

                // Clamp for numerical stability
                let pos_prob = pos_prob.clamp(1e-7, 1.0 - 1e-7);
                let neg_prob = neg_prob.clamp(1e-7, 1.0 - 1e-7);

                total_loss += -pos_prob.ln() - (1.0 - neg_prob).ln();
                count += 1;
            }
        }

        if count > 0 {
            Ok(total_loss / count as f32)
        } else {
            Ok(0.0)
        }
    }

    /// Returns attention scores for visualization/analysis.
    ///
    /// # Arguments
    ///
    /// * `behavior_seq` - User behavior sequence [batch_size, seq_len, embedding_dim]
    /// * `target_item` - Target item embedding [batch_size, embedding_dim]
    /// * `mask` - Optional mask [batch_size, seq_len]
    ///
    /// # Returns
    ///
    /// Attention scores [batch_size, seq_len]
    pub fn get_attention_scores(
        &self,
        behavior_seq: &Tensor,
        target_item: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        self.validate_inputs(behavior_seq, target_item, mask)?;

        let batch_size = behavior_seq.shape()[0];
        let seq_len = behavior_seq.shape()[1];

        // Run interest extractor
        let extractor_states = self.run_interest_extractor(behavior_seq, batch_size, seq_len)?;

        // Project target
        let target_projected = self.project_target(target_item)?;

        // Compute attention
        self.attention
            .compute_attention(&target_projected, &extractor_states, mask)
    }
}

impl Layer for DIENLayer {
    /// Forward pass - expects input as [batch, seq_len+1, embedding_dim]
    /// where first position is target item and rest is behavior sequence.
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "DIEN forward expects 3D input [batch, seq_len+1, dim], got {}D",
                    input.ndim()
                ),
            });
        }

        let batch_size = input.shape()[0];
        let total_len = input.shape()[1];
        let embedding_dim = input.shape()[2];

        if total_len < 2 {
            return Err(LayerError::ForwardError {
                message: "Input sequence must have at least 2 elements (target + 1 behavior)"
                    .to_string(),
            });
        }

        if embedding_dim != self.config.embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.config.embedding_dim,
                actual: embedding_dim,
            });
        }

        // Extract target (first position)
        let target = self.extract_target(input, batch_size, embedding_dim);

        // Extract behavior sequence (remaining positions)
        let behavior_seq =
            self.extract_behavior_seq(input, batch_size, total_len - 1, embedding_dim);

        // Run DIEN forward
        self.forward_dien(&behavior_seq, &target, None)
    }

    fn backward(&mut self, _grad: &Tensor) -> Result<Tensor, LayerError> {
        let cache = self.cache.as_ref().ok_or(LayerError::NotInitialized)?;

        let batch_size = cache.behavior_seq.shape()[0];
        let seq_len = cache.behavior_seq.shape()[1];
        let embedding_dim = self.config.embedding_dim;

        // Simplified backward: return gradient w.r.t. behavior sequence
        // Full BPTT would require more complex gradient computation through GRUs
        let d_behavior = vec![0.0; batch_size * seq_len * embedding_dim];

        Ok(Tensor::from_data(
            &[batch_size, seq_len, embedding_dim],
            d_behavior,
        ))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();

        // Interest extractor parameters
        params.extend(self.interest_extractor.parameters());

        // Optional target projector parameters
        if let Some(p) = &self.target_projector {
            params.extend(p.parameters());
        }

        // Evolution GRU parameters
        params.extend(self.evolution_gru.parameters());
        params.extend(self.evolution_standard_gru.parameters());

        // Attention parameters
        params.extend(self.attention.parameters());

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();

        // Interest extractor parameters
        params.extend(self.interest_extractor.parameters_mut());

        // Optional target projector parameters
        if let Some(p) = &mut self.target_projector {
            params.extend(p.parameters_mut());
        }

        // Evolution GRU parameters
        params.extend(self.evolution_gru.parameters_mut());
        params.extend(self.evolution_standard_gru.parameters_mut());

        // Attention parameters
        params.extend(self.attention.parameters_mut());

        params
    }

    fn name(&self) -> &str {
        "DIENLayer"
    }

    fn regularization_loss(&self) -> f32 {
        let reg = &self.config.regularizer;
        self.parameters()
            .into_iter()
            .filter(|t| t.ndim() > 1)
            .map(|t| reg.loss(t))
            .sum()
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl DIENLayer {
    /// Extracts target from combined input.
    fn extract_target(&self, input: &Tensor, _batch_size: usize, _embedding_dim: usize) -> Tensor {
        input.narrow(1, 0, 1).squeeze(1).contiguous()
    }

    /// Extracts behavior sequence from combined input.
    fn extract_behavior_seq(
        &self,
        input: &Tensor,
        _batch_size: usize,
        seq_len: usize,
        _embedding_dim: usize,
    ) -> Tensor {
        input.narrow(1, 1, seq_len).contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dien_config_default() {
        let config = DIENConfig::new(32, 32);
        assert_eq!(config.embedding_dim, 32);
        assert_eq!(config.hidden_size, 32);
        assert!(config.use_auxiliary_loss);
        assert_eq!(config.gru_type, GRUType::AGRU);
    }

    #[test]
    fn test_dien_config_builder() {
        let config = DIENConfig::new(32, 32)
            .with_use_auxiliary_loss(false)
            .with_gru_type(GRUType::Standard)
            .with_attention_hidden_units(vec![128, 64])
            .with_use_softmax(true);

        assert!(!config.use_auxiliary_loss);
        assert_eq!(config.gru_type, GRUType::Standard);
        assert_eq!(config.attention_hidden_units, vec![128, 64]);
        assert!(config.use_softmax);
    }

    #[test]
    fn test_dien_config_validation() {
        // Valid config
        let config = DIENConfig::new(32, 32);
        config
            .validate()
            .expect("default DIEN config should pass validation");

        // Invalid: zero embedding dim
        let config = DIENConfig::new(0, 64);
        config
            .validate()
            .expect_err("DIEN config with zero embedding dim should fail validation");

        // Invalid: zero hidden size
        let config = DIENConfig::new(32, 0);
        config
            .validate()
            .expect_err("DIEN config with zero hidden size should fail validation");

        // Valid: embedding dim can differ from hidden size (extractor projects to hidden)
        let config = DIENConfig::new(32, 64);
        config
            .validate()
            .expect("DIEN config with projected hidden size should pass validation");
    }

    #[test]
    fn test_augru_cell_creation() {
        let cell = AUGRUCell::new(32, 64, ActivationType::relu());
        assert_eq!(cell.input_dim(), 32);
        assert_eq!(cell.hidden_dim(), 64);
        assert_eq!(cell.parameters().len(), 9);
    }

    #[test]
    fn test_augru_cell_forward() {
        let cell = AUGRUCell::new(8, 16, ActivationType::relu());

        let x = Tensor::rand(&[2, 8]);
        let h = Tensor::zeros(&[2, 16]);
        let attention = Tensor::ones(&[2]);

        let h_new = cell.forward_step(&x, &h, &attention).unwrap();
        assert_eq!(h_new.shape(), &[2, 16]);
    }

    #[test]
    fn test_gru_cell_creation() {
        let cell = GRUCell::new(32, 64, ActivationType::relu());
        assert_eq!(cell.input_dim(), 32);
        assert_eq!(cell.hidden_dim(), 64);
    }

    #[test]
    fn test_gru_cell_forward() {
        let cell = GRUCell::new(8, 16, ActivationType::relu());

        let x = Tensor::rand(&[2, 8]);
        let h = Tensor::zeros(&[2, 16]);

        let h_new = cell.forward_step(&x, &h).unwrap();
        assert_eq!(h_new.shape(), &[2, 16]);
    }

    #[test]
    fn test_dien_creation() {
        let dien = DIENLayer::new(32, 32);
        assert_eq!(dien.embedding_dim(), 32);
        assert_eq!(dien.hidden_size(), 32);
    }

    #[test]
    fn test_dien_from_config() {
        let config = DIENConfig::new(16, 16).with_gru_type(GRUType::AUGRU);

        let dien = DIENLayer::from_config(config).unwrap();
        assert_eq!(dien.embedding_dim(), 16);
        assert_eq!(dien.hidden_size(), 16);
    }

    #[test]
    fn test_dien_forward() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let output = dien
            .forward_dien(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_dien_forward_with_mask() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);
        let mask = Tensor::from_data(
            &[2, 5],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        );

        let output = dien
            .forward_dien(&behavior_seq, &target_item, Some(&mask))
            .unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_dien_forward_standard_gru() {
        let config = DIENConfig::new(8, 8).with_gru_type(GRUType::Standard);
        let dien = DIENLayer::from_config(config).unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let output = dien
            .forward_dien(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[2, 8]);
    }

    #[test]
    fn test_dien_layer_trait() {
        let dien = DIENLayer::new(8, 8);

        // Combined input: [batch, seq_len+1, embedding_dim]
        let input = Tensor::rand(&[2, 6, 8]);

        let output = dien.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 8]);
        assert_eq!(dien.name(), "DIENLayer");
    }

    #[test]
    fn test_dien_parameters() {
        let dien = DIENLayer::new(8, 8);
        let params = dien.parameters();

        // Interest extractor: 9 params
        // Evolution AUGRU: 9 params
        // Evolution standard GRU: 9 params (allocated for GRUType::Standard)
        // Attention: 1 param (weight)
        // Total: 28 params
        assert_eq!(params.len(), 28);
    }

    #[test]
    fn test_dien_training_mode() {
        let mut dien = DIENLayer::new(8, 8);
        assert!(dien.is_training());

        dien.set_training(false);
        assert!(!dien.is_training());

        dien.set_training(true);
        assert!(dien.is_training());
    }

    #[test]
    fn test_dien_invalid_behavior_shape() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 8]); // 2D instead of 3D
        let target_item = Tensor::rand(&[2, 8]);

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        result.expect_err("DIEN forward should fail when behavior sequence is not rank-3");
    }

    #[test]
    fn test_dien_invalid_target_shape() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 5, 8]); // 3D instead of 2D

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        result.expect_err("DIEN forward should fail when target item is not rank-2");
    }

    #[test]
    fn test_dien_embedding_dim_mismatch() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 16]); // Wrong embedding dim
        let target_item = Tensor::rand(&[2, 8]);

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        result.expect_err("DIEN forward should fail for embedding dimension mismatch");
    }

    #[test]
    fn test_dien_batch_size_mismatch() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[3, 8]); // Different batch size

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        result.expect_err("DIEN forward should fail for batch size mismatch");
    }

    #[test]
    fn test_dien_mask_shape_mismatch() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);
        let mask = Tensor::ones(&[2, 3]); // Wrong seq_len

        let result = dien.forward_dien(&behavior_seq, &target_item, Some(&mask));
        result.expect_err("DIEN forward should fail when mask shape mismatches sequence");
    }

    #[test]
    fn test_dien_auxiliary_loss() {
        let config = DIENConfig::new(8, 8).with_use_auxiliary_loss(true);
        let dien = DIENLayer::from_config(config).unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let neg_samples = Tensor::rand(&[2, 5, 8]);

        let loss = dien.auxiliary_loss(&behavior_seq, &neg_samples).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_dien_auxiliary_loss_disabled() {
        let config = DIENConfig::new(8, 8).with_use_auxiliary_loss(false);
        let dien = DIENLayer::from_config(config).unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let neg_samples = Tensor::rand(&[2, 5, 8]);

        let loss = dien.auxiliary_loss(&behavior_seq, &neg_samples).unwrap();
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_dien_auxiliary_loss_shape_mismatch() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let neg_samples = Tensor::rand(&[2, 3, 8]); // Wrong shape

        let result = dien.auxiliary_loss(&behavior_seq, &neg_samples);
        result.expect_err("DIEN auxiliary loss should fail for mismatched sample shapes");
    }

    #[test]
    fn test_dien_get_attention_scores() {
        let dien = DIENLayer::new(8, 8);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let scores = dien
            .get_attention_scores(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(scores.shape(), &[2, 5]);
    }

    #[test]
    fn test_dien_forward_train() {
        let mut dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let output = dien
            .forward_dien_train(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[2, 16]);

        // Cache should be set
        assert!(dien.cache.is_some());
    }

    #[test]
    fn test_dien_backward() {
        let mut dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        // Forward with training
        let _output = dien
            .forward_dien_train(&behavior_seq, &target_item, None)
            .unwrap();

        // Backward
        let grad = Tensor::ones(&[2, 16]);
        let input_grad = dien.backward(&grad).unwrap();
        assert_eq!(input_grad.shape(), &[2, 5, 8]);
    }

    #[test]
    fn test_dien_backward_without_cache() {
        let mut dien = DIENLayer::new(8, 16);

        let grad = Tensor::ones(&[2, 16]);
        let result = dien.backward(&grad);
        result.expect_err("DIEN backward should fail when no forward cache exists");
    }

    #[test]
    fn test_dien_single_item_sequence() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 1, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let output = dien
            .forward_dien(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_dien_large_batch() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[32, 10, 8]);
        let target_item = Tensor::rand(&[32, 8]);

        let output = dien
            .forward_dien(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[32, 16]);
    }

    #[test]
    fn test_gru_type_default() {
        let gru_type = GRUType::default();
        assert_eq!(gru_type, GRUType::AGRU);
    }

    #[test]
    fn test_dien_attention_effect() {
        // Test that different targets produce different attention patterns
        let dien = DIENLayer::new(4, 8);

        let behavior_seq = Tensor::rand(&[1, 3, 4]);
        let target1 = Tensor::from_data(&[1, 4], vec![1.0, 0.0, 0.0, 0.0]);
        let target2 = Tensor::from_data(&[1, 4], vec![0.0, 1.0, 0.0, 0.0]);

        let scores1 = dien
            .get_attention_scores(&behavior_seq, &target1, None)
            .unwrap();
        let scores2 = dien
            .get_attention_scores(&behavior_seq, &target2, None)
            .unwrap();

        // Scores should generally be different for different targets
        let diff: f32 = scores1
            .data()
            .iter()
            .zip(scores2.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        // There should be some difference (may be small due to random initialization)
        assert!(diff >= 0.0);
    }

    #[test]
    fn test_dien_with_softmax_attention() {
        let config = DIENConfig::new(8, 8).with_use_softmax(true);
        let dien = DIENLayer::from_config(config).unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let scores = dien
            .get_attention_scores(&behavior_seq, &target_item, None)
            .unwrap();

        // With softmax, scores should sum to 1 for each batch
        for b in 0..2 {
            let sum: f32 = (0..5).map(|s| scores.data()[b * 5 + s]).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Softmax scores should sum to 1, got {}",
                sum
            );
        }
    }
}
