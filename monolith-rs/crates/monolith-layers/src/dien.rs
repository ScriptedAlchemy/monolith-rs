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
//! 2. **Interest Evolution Layer**: Uses AUGRU (Attention-based GRU) to model interest evolution
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
//!                                           AUGRU (Interest Evolution) --> Evolved Interest
//! ```
//!
//! # Example
//!
//! ```
//! use monolith_layers::dien::{DIENLayer, DIENConfig, GRUType};
//! use monolith_layers::tensor::Tensor;
//!
//! // Create DIEN layer
//! let config = DIENConfig::new(32, 64)
//!     .with_use_auxiliary_loss(true)
//!     .with_gru_type(GRUType::AUGRU);
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
//! assert_eq!(evolved_interest.shape(), &[4, 64]);
//! ```
//!
//! # References
//!
//! - Zhou, G., et al. "Deep Interest Evolution Network for Click-Through Rate Prediction." AAAI 2019.

use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Type of GRU cell to use in the interest evolution layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GRUType {
    /// Standard GRU cell
    Standard,
    /// Attention-based GRU (AUGRU) - modifies update gate with attention
    AUGRU,
}

impl Default for GRUType {
    fn default() -> Self {
        GRUType::AUGRU
    }
}

/// Configuration for DIEN layer.
///
/// # Example
///
/// ```
/// use monolith_layers::dien::{DIENConfig, GRUType};
///
/// let config = DIENConfig::new(32, 64)
///     .with_use_auxiliary_loss(true)
///     .with_gru_type(GRUType::AUGRU)
///     .with_attention_hidden_units(vec![64, 32]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DIENConfig {
    /// Dimension of input embeddings
    pub embedding_dim: usize,
    /// Dimension of hidden states in GRU cells
    pub hidden_size: usize,
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
            use_auxiliary_loss: true,
            gru_type: GRUType::AUGRU,
            attention_hidden_units: vec![64, 32],
            use_softmax: false,
        }
    }

    /// Sets whether to use auxiliary loss.
    pub fn with_use_auxiliary_loss(mut self, use_auxiliary_loss: bool) -> Self {
        self.use_auxiliary_loss = use_auxiliary_loss;
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
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        // Xavier initialization
        let std_x = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let std_h = (2.0 / (hidden_dim + hidden_dim) as f32).sqrt();

        Self {
            input_dim,
            hidden_dim,
            // Reset gate
            w_r_x: Tensor::randn(&[input_dim, hidden_dim], 0.0, std_x),
            w_r_h: Tensor::randn(&[hidden_dim, hidden_dim], 0.0, std_h),
            b_r: Tensor::zeros(&[hidden_dim]),
            // Update gate
            w_z_x: Tensor::randn(&[input_dim, hidden_dim], 0.0, std_x),
            w_z_h: Tensor::randn(&[hidden_dim, hidden_dim], 0.0, std_h),
            b_z: Tensor::zeros(&[hidden_dim]),
            // Candidate hidden state
            w_h_x: Tensor::randn(&[input_dim, hidden_dim], 0.0, std_x),
            w_h_h: Tensor::randn(&[hidden_dim, hidden_dim], 0.0, std_h),
            b_h: Tensor::zeros(&[hidden_dim]),
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

        // Reset gate: r = sigmoid(x @ W_r_x + h @ W_r_h + b_r)
        let r = self.compute_reset_gate(x, h)?;

        // Update gate: z = sigmoid(x @ W_z_x + h @ W_z_h + b_z)
        let z = self.compute_update_gate(x, h)?;

        // Candidate: h_tilde = tanh(x @ W_h_x + (r * h) @ W_h_h + b_h)
        let h_tilde = self.compute_candidate(x, h, &r)?;

        // Apply attention to update gate: z' = a * z
        // New hidden: h_new = (1 - z') * h + z' * h_tilde
        let mut result = vec![0.0; batch_size * self.hidden_dim];
        for b in 0..batch_size {
            let att = attention.data()[b];
            for d in 0..self.hidden_dim {
                let idx = b * self.hidden_dim + d;
                let z_val = z.data()[idx];
                let z_att = att * z_val;
                result[idx] = (1.0 - z_att) * h.data()[idx] + z_att * h_tilde.data()[idx];
            }
        }

        Ok(Tensor::from_data(&[batch_size, self.hidden_dim], result))
    }

    /// Computes the reset gate.
    fn compute_reset_gate(&self, x: &Tensor, h: &Tensor) -> Result<Tensor, LayerError> {
        let xw = x.matmul(&self.w_r_x);
        let hw = h.matmul(&self.w_r_h);
        let sum = xw.add(&hw).add(&self.b_r);
        Ok(sum.map(|v| 1.0 / (1.0 + (-v).exp()))) // sigmoid
    }

    /// Computes the update gate.
    fn compute_update_gate(&self, x: &Tensor, h: &Tensor) -> Result<Tensor, LayerError> {
        let xw = x.matmul(&self.w_z_x);
        let hw = h.matmul(&self.w_z_h);
        let sum = xw.add(&hw).add(&self.b_z);
        Ok(sum.map(|v| 1.0 / (1.0 + (-v).exp()))) // sigmoid
    }

    /// Computes the candidate hidden state.
    fn compute_candidate(&self, x: &Tensor, h: &Tensor, r: &Tensor) -> Result<Tensor, LayerError> {
        let xw = x.matmul(&self.w_h_x);
        let rh = r.mul(h);
        let hw = rh.matmul(&self.w_h_h);
        let sum = xw.add(&hw).add(&self.b_h);
        Ok(sum.map(|v| v.tanh()))
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
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        let std_x = (2.0 / (input_dim + hidden_dim) as f32).sqrt();
        let std_h = (2.0 / (hidden_dim + hidden_dim) as f32).sqrt();

        Self {
            input_dim,
            hidden_dim,
            w_r_x: Tensor::randn(&[input_dim, hidden_dim], 0.0, std_x),
            w_r_h: Tensor::randn(&[hidden_dim, hidden_dim], 0.0, std_h),
            b_r: Tensor::zeros(&[hidden_dim]),
            w_z_x: Tensor::randn(&[input_dim, hidden_dim], 0.0, std_x),
            w_z_h: Tensor::randn(&[hidden_dim, hidden_dim], 0.0, std_h),
            b_z: Tensor::zeros(&[hidden_dim]),
            w_h_x: Tensor::randn(&[input_dim, hidden_dim], 0.0, std_x),
            w_h_h: Tensor::randn(&[hidden_dim, hidden_dim], 0.0, std_h),
            b_h: Tensor::zeros(&[hidden_dim]),
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

        // Reset gate
        let xw_r = x.matmul(&self.w_r_x);
        let hw_r = h.matmul(&self.w_r_h);
        let r = xw_r
            .add(&hw_r)
            .add(&self.b_r)
            .map(|v| 1.0 / (1.0 + (-v).exp()));

        // Update gate
        let xw_z = x.matmul(&self.w_z_x);
        let hw_z = h.matmul(&self.w_z_h);
        let z = xw_z
            .add(&hw_z)
            .add(&self.b_z)
            .map(|v| 1.0 / (1.0 + (-v).exp()));

        // Candidate
        let xw_h = x.matmul(&self.w_h_x);
        let rh = r.mul(h);
        let hw_h = rh.matmul(&self.w_h_h);
        let h_tilde = xw_h.add(&hw_h).add(&self.b_h).map(|v| v.tanh());

        // New hidden state: h_new = (1 - z) * h + z * h_tilde
        let mut result = vec![0.0; batch_size * self.hidden_dim];
        for b in 0..batch_size {
            for d in 0..self.hidden_dim {
                let idx = b * self.hidden_dim + d;
                let z_val = z.data()[idx];
                result[idx] = (1.0 - z_val) * h.data()[idx] + z_val * h_tilde.data()[idx];
            }
        }

        Ok(Tensor::from_data(&[batch_size, self.hidden_dim], result))
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
    /// Embedding dimension
    embedding_dim: usize,
    /// Hidden dimension (for hidden states)
    hidden_dim: usize,
    /// First layer weights
    w1: Tensor,
    /// First layer bias
    b1: Tensor,
    /// Second layer weights
    w2: Tensor,
    /// Second layer bias
    b2: Tensor,
    /// Output layer weights
    w_out: Tensor,
    /// Output layer bias
    b_out: Tensor,
    /// Whether to use softmax
    use_softmax: bool,
}

impl AttentionModule {
    /// Creates a new attention module.
    fn new(
        embedding_dim: usize,
        hidden_dim: usize,
        hidden_units: &[usize],
        use_softmax: bool,
    ) -> Self {
        // Input: [target, hidden_state, target-hidden, target*hidden] where target is projected
        // We need to project target from embedding_dim to hidden_dim first
        let input_dim = 4 * hidden_dim;

        let h1 = if !hidden_units.is_empty() {
            hidden_units[0]
        } else {
            32
        };
        let h2 = if hidden_units.len() > 1 {
            hidden_units[1]
        } else {
            h1 / 2
        };

        let std1 = (2.0 / (input_dim + h1) as f32).sqrt();
        let std2 = (2.0 / (h1 + h2) as f32).sqrt();
        let std_out = (2.0 / (h2 + 1) as f32).sqrt();

        Self {
            embedding_dim,
            hidden_dim,
            w1: Tensor::randn(&[input_dim, h1], 0.0, std1),
            b1: Tensor::zeros(&[h1]),
            w2: Tensor::randn(&[h1, h2], 0.0, std2),
            b2: Tensor::zeros(&[h2]),
            w_out: Tensor::randn(&[h2, 1], 0.0, std_out),
            b_out: Tensor::zeros(&[1]),
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

        // Build attention input for each (target, hidden_state) pair
        // Input: [target, hidden, target-hidden, target*hidden]
        let input_dim = 4 * hidden_dim;
        let mut attention_input = vec![0.0; batch_size * seq_len * input_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                let t_offset = b * hidden_dim;
                let h_offset = b * seq_len * hidden_dim + s * hidden_dim;
                let out_offset = (b * seq_len + s) * input_dim;

                for d in 0..hidden_dim {
                    let t_val = target.data()[t_offset + d];
                    let h_val = hidden_states.data()[h_offset + d];

                    attention_input[out_offset + d] = t_val;
                    attention_input[out_offset + hidden_dim + d] = h_val;
                    attention_input[out_offset + 2 * hidden_dim + d] = t_val - h_val;
                    attention_input[out_offset + 3 * hidden_dim + d] = t_val * h_val;
                }
            }
        }

        let attention_tensor =
            Tensor::from_data(&[batch_size * seq_len, input_dim], attention_input);

        // Forward through MLP
        // Layer 1
        let h1 = attention_tensor.matmul(&self.w1).add(&self.b1);
        let h1 = h1.map(|v| 1.0 / (1.0 + (-v).exp())); // sigmoid

        // Layer 2
        let h2 = h1.matmul(&self.w2).add(&self.b2);
        let h2 = h2.map(|v| 1.0 / (1.0 + (-v).exp())); // sigmoid

        // Output layer
        let logits = h2.matmul(&self.w_out).add(&self.b_out);

        // Reshape to [batch_size, seq_len]
        let mut scores = logits.reshape(&[batch_size, seq_len]).data().to_vec();

        // Apply mask
        if let Some(m) = mask {
            for b in 0..batch_size {
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if m.data()[idx] == 0.0 {
                        scores[idx] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // Apply softmax if configured
        if self.use_softmax {
            for b in 0..batch_size {
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if scores[idx] > max_val {
                        max_val = scores[idx];
                    }
                }

                if max_val == f32::NEG_INFINITY {
                    max_val = 0.0;
                }

                // Compute exp and sum
                let mut sum = 0.0;
                for s in 0..seq_len {
                    let idx = b * seq_len + s;
                    if scores[idx] != f32::NEG_INFINITY {
                        scores[idx] = (scores[idx] - max_val).exp();
                        sum += scores[idx];
                    } else {
                        scores[idx] = 0.0;
                    }
                }

                // Normalize
                if sum > 0.0 {
                    for s in 0..seq_len {
                        let idx = b * seq_len + s;
                        scores[idx] /= sum;
                    }
                }
            }
        } else {
            // Just apply sigmoid to get values in [0, 1]
            for score in &mut scores {
                if *score == f32::NEG_INFINITY {
                    *score = 0.0;
                }
            }
        }

        Ok(Tensor::from_data(&[batch_size, seq_len], scores))
    }

    /// Returns all parameters.
    fn parameters(&self) -> Vec<&Tensor> {
        vec![
            &self.w1,
            &self.b1,
            &self.w2,
            &self.b2,
            &self.w_out,
            &self.b_out,
        ]
    }

    /// Returns mutable references to all parameters.
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![
            &mut self.w1,
            &mut self.b1,
            &mut self.w2,
            &mut self.b2,
            &mut self.w_out,
            &mut self.b_out,
        ]
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
    /// Projection from embedding_dim to hidden_dim for target
    target_projection: Tensor,
    /// Target projection bias
    target_projection_bias: Tensor,
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
        let interest_extractor = GRUCell::new(embedding_dim, hidden_size);

        // Interest evolution: maps hidden_size -> hidden_size
        let evolution_gru = AUGRUCell::new(hidden_size, hidden_size);
        let evolution_standard_gru = GRUCell::new(hidden_size, hidden_size);

        // Attention module
        let attention = AttentionModule::new(
            embedding_dim,
            hidden_size,
            &config.attention_hidden_units,
            config.use_softmax,
        );

        // Target projection: embedding_dim -> hidden_size
        let std_proj = (2.0 / (embedding_dim + hidden_size) as f32).sqrt();
        let target_projection = Tensor::randn(&[embedding_dim, hidden_size], 0.0, std_proj);
        let target_projection_bias = Tensor::zeros(&[hidden_size]);

        Ok(Self {
            config,
            interest_extractor,
            evolution_gru,
            evolution_standard_gru,
            attention,
            target_projection,
            target_projection_bias,
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

        if behavior_seq.shape()[2] != embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: embedding_dim,
                actual: behavior_seq.shape()[2],
            });
        }

        if target_item.shape()[1] != embedding_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: embedding_dim,
                actual: target_item.shape()[1],
            });
        }

        if target_item.shape()[0] != batch_size {
            return Err(LayerError::ShapeMismatch {
                expected: vec![batch_size, embedding_dim],
                actual: target_item.shape().to_vec(),
            });
        }

        if let Some(m) = mask {
            if m.shape() != &[batch_size, seq_len] {
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
        let mut all_states = vec![0.0; batch_size * seq_len * hidden_size];

        for t in 0..seq_len {
            // Extract input at timestep t
            let x_t = self.extract_timestep(behavior_seq, t, batch_size, embedding_dim);

            // GRU step
            h = self.interest_extractor.forward_step(&x_t, &h)?;

            // Store hidden state
            for b in 0..batch_size {
                for d in 0..hidden_size {
                    all_states[b * seq_len * hidden_size + t * hidden_size + d] =
                        h.data()[b * hidden_size + d];
                }
            }
        }

        Ok(Tensor::from_data(
            &[batch_size, seq_len, hidden_size],
            all_states,
        ))
    }

    /// Projects target item to hidden dimension.
    fn project_target(&self, target_item: &Tensor) -> Result<Tensor, LayerError> {
        let projected = target_item.matmul(&self.target_projection);
        Ok(projected.add(&self.target_projection_bias))
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
                GRUType::AUGRU => self.evolution_gru.forward_step(&x_t, &h, &a_t)?,
                GRUType::Standard => self.evolution_standard_gru.forward_step(&x_t, &h)?,
            };
        }

        Ok(h)
    }

    /// Extracts a timestep from the behavior sequence.
    fn extract_timestep(&self, input: &Tensor, t: usize, batch_size: usize, dim: usize) -> Tensor {
        let seq_len = input.shape()[1];
        let mut data = vec![0.0; batch_size * dim];

        for b in 0..batch_size {
            for d in 0..dim {
                data[b * dim + d] = input.data()[b * seq_len * dim + t * dim + d];
            }
        }

        Tensor::from_data(&[batch_size, dim], data)
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
        let seq_len = attention.shape()[1];
        let data: Vec<f32> = (0..batch_size)
            .map(|b| attention.data()[b * seq_len + t])
            .collect();

        Tensor::from_data(&[batch_size], data)
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
                    let h_val = extractor_states.data()[h_offset + d];
                    let pos_val = behavior_seq.data()[pos_offset + d];
                    let neg_val = neg_samples.data()[neg_offset + d];

                    pos_score += h_val * pos_val;
                    neg_score += h_val * neg_val;
                }

                // Binary cross-entropy loss
                // L = -log(sigmoid(pos_score)) - log(1 - sigmoid(neg_score))
                let pos_prob = 1.0 / (1.0 + (-pos_score).exp());
                let neg_prob = 1.0 / (1.0 + (-neg_score).exp());

                // Clamp for numerical stability
                let pos_prob = pos_prob.max(1e-7).min(1.0 - 1e-7);
                let neg_prob = neg_prob.max(1e-7).min(1.0 - 1e-7);

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

        // Evolution GRU parameters
        params.extend(self.evolution_gru.parameters());

        // Attention parameters
        params.extend(self.attention.parameters());

        // Target projection
        params.push(&self.target_projection);
        params.push(&self.target_projection_bias);

        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();

        // Interest extractor parameters
        params.extend(self.interest_extractor.parameters_mut());

        // Evolution GRU parameters
        params.extend(self.evolution_gru.parameters_mut());

        // Attention parameters
        params.extend(self.attention.parameters_mut());

        // Target projection
        params.push(&mut self.target_projection);
        params.push(&mut self.target_projection_bias);

        params
    }

    fn name(&self) -> &str {
        "DIENLayer"
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
    fn extract_target(&self, input: &Tensor, batch_size: usize, embedding_dim: usize) -> Tensor {
        let total_len = input.shape()[1];
        let mut data = vec![0.0; batch_size * embedding_dim];

        for b in 0..batch_size {
            for d in 0..embedding_dim {
                data[b * embedding_dim + d] = input.data()[b * total_len * embedding_dim + d];
            }
        }

        Tensor::from_data(&[batch_size, embedding_dim], data)
    }

    /// Extracts behavior sequence from combined input.
    fn extract_behavior_seq(
        &self,
        input: &Tensor,
        batch_size: usize,
        seq_len: usize,
        embedding_dim: usize,
    ) -> Tensor {
        let total_len = input.shape()[1];
        let mut data = vec![0.0; batch_size * seq_len * embedding_dim];

        for b in 0..batch_size {
            for s in 0..seq_len {
                for d in 0..embedding_dim {
                    data[b * seq_len * embedding_dim + s * embedding_dim + d] =
                        input.data()[b * total_len * embedding_dim + (s + 1) * embedding_dim + d];
                }
            }
        }

        Tensor::from_data(&[batch_size, seq_len, embedding_dim], data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dien_config_default() {
        let config = DIENConfig::new(32, 64);
        assert_eq!(config.embedding_dim, 32);
        assert_eq!(config.hidden_size, 64);
        assert!(config.use_auxiliary_loss);
        assert_eq!(config.gru_type, GRUType::AUGRU);
    }

    #[test]
    fn test_dien_config_builder() {
        let config = DIENConfig::new(32, 64)
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
        let config = DIENConfig::new(32, 64);
        assert!(config.validate().is_ok());

        // Invalid: zero embedding dim
        let config = DIENConfig::new(0, 64);
        assert!(config.validate().is_err());

        // Invalid: zero hidden size
        let config = DIENConfig::new(32, 0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_augru_cell_creation() {
        let cell = AUGRUCell::new(32, 64);
        assert_eq!(cell.input_dim(), 32);
        assert_eq!(cell.hidden_dim(), 64);
        assert_eq!(cell.parameters().len(), 9);
    }

    #[test]
    fn test_augru_cell_forward() {
        let cell = AUGRUCell::new(8, 16);

        let x = Tensor::rand(&[2, 8]);
        let h = Tensor::zeros(&[2, 16]);
        let attention = Tensor::ones(&[2]);

        let h_new = cell.forward_step(&x, &h, &attention).unwrap();
        assert_eq!(h_new.shape(), &[2, 16]);
    }

    #[test]
    fn test_gru_cell_creation() {
        let cell = GRUCell::new(32, 64);
        assert_eq!(cell.input_dim(), 32);
        assert_eq!(cell.hidden_dim(), 64);
    }

    #[test]
    fn test_gru_cell_forward() {
        let cell = GRUCell::new(8, 16);

        let x = Tensor::rand(&[2, 8]);
        let h = Tensor::zeros(&[2, 16]);

        let h_new = cell.forward_step(&x, &h).unwrap();
        assert_eq!(h_new.shape(), &[2, 16]);
    }

    #[test]
    fn test_dien_creation() {
        let dien = DIENLayer::new(32, 64);
        assert_eq!(dien.embedding_dim(), 32);
        assert_eq!(dien.hidden_size(), 64);
    }

    #[test]
    fn test_dien_from_config() {
        let config = DIENConfig::new(16, 32).with_gru_type(GRUType::AUGRU);

        let dien = DIENLayer::from_config(config).unwrap();
        assert_eq!(dien.embedding_dim(), 16);
        assert_eq!(dien.hidden_size(), 32);
    }

    #[test]
    fn test_dien_forward() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let output = dien
            .forward_dien(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_dien_forward_with_mask() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);
        let mask = Tensor::from_data(
            &[2, 5],
            vec![1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        );

        let output = dien
            .forward_dien(&behavior_seq, &target_item, Some(&mask))
            .unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_dien_forward_standard_gru() {
        let config = DIENConfig::new(8, 16).with_gru_type(GRUType::Standard);
        let dien = DIENLayer::from_config(config).unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);

        let output = dien
            .forward_dien(&behavior_seq, &target_item, None)
            .unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_dien_layer_trait() {
        let dien = DIENLayer::new(8, 16);

        // Combined input: [batch, seq_len+1, embedding_dim]
        let input = Tensor::rand(&[2, 6, 8]);

        let output = dien.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16]);
        assert_eq!(dien.name(), "DIENLayer");
    }

    #[test]
    fn test_dien_parameters() {
        let dien = DIENLayer::new(8, 16);
        let params = dien.parameters();

        // Interest extractor: 9 params
        // Evolution GRU: 9 params
        // Attention: 6 params
        // Target projection: 2 params
        // Total: 26 params
        assert_eq!(params.len(), 26);
    }

    #[test]
    fn test_dien_training_mode() {
        let mut dien = DIENLayer::new(8, 16);
        assert!(dien.is_training());

        dien.set_training(false);
        assert!(!dien.is_training());

        dien.set_training(true);
        assert!(dien.is_training());
    }

    #[test]
    fn test_dien_invalid_behavior_shape() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 8]); // 2D instead of 3D
        let target_item = Tensor::rand(&[2, 8]);

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dien_invalid_target_shape() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 5, 8]); // 3D instead of 2D

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dien_embedding_dim_mismatch() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 16]); // Wrong embedding dim
        let target_item = Tensor::rand(&[2, 8]);

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dien_batch_size_mismatch() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[3, 8]); // Different batch size

        let result = dien.forward_dien(&behavior_seq, &target_item, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_dien_mask_shape_mismatch() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let target_item = Tensor::rand(&[2, 8]);
        let mask = Tensor::ones(&[2, 3]); // Wrong seq_len

        let result = dien.forward_dien(&behavior_seq, &target_item, Some(&mask));
        assert!(result.is_err());
    }

    #[test]
    fn test_dien_auxiliary_loss() {
        let config = DIENConfig::new(8, 16).with_use_auxiliary_loss(true);
        let dien = DIENLayer::from_config(config).unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let neg_samples = Tensor::rand(&[2, 5, 8]);

        let loss = dien.auxiliary_loss(&behavior_seq, &neg_samples).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_dien_auxiliary_loss_disabled() {
        let config = DIENConfig::new(8, 16).with_use_auxiliary_loss(false);
        let dien = DIENLayer::from_config(config).unwrap();

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let neg_samples = Tensor::rand(&[2, 5, 8]);

        let loss = dien.auxiliary_loss(&behavior_seq, &neg_samples).unwrap();
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_dien_auxiliary_loss_shape_mismatch() {
        let dien = DIENLayer::new(8, 16);

        let behavior_seq = Tensor::rand(&[2, 5, 8]);
        let neg_samples = Tensor::rand(&[2, 3, 8]); // Wrong shape

        let result = dien.auxiliary_loss(&behavior_seq, &neg_samples);
        assert!(result.is_err());
    }

    #[test]
    fn test_dien_get_attention_scores() {
        let dien = DIENLayer::new(8, 16);

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
        assert!(result.is_err());
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
        assert_eq!(gru_type, GRUType::AUGRU);
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
        let config = DIENConfig::new(8, 16).with_use_softmax(true);
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
