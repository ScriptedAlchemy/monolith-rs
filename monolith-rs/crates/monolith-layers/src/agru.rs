//! Attention GRU (AGRU) layer implementation.
//!
//! This module provides the [`AGRU`] layer, which is a variant of GRU that incorporates
//! attention mechanisms for sequential recommendation and user behavior modeling.

use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Attention GRU (AGRU) layer.
///
/// AGRU modifies the standard GRU by incorporating attention weights into the
/// update gate, allowing the model to selectively attend to relevant parts
/// of the user behavior sequence.
///
/// The forward pass computes:
/// - r_t = sigmoid(W_r * [h_{t-1}, x_t])  (reset gate)
/// - z_t = sigmoid(W_z * [h_{t-1}, x_t])  (update gate)
/// - h_tilde = tanh(W_h * [r_t * h_{t-1}, x_t])  (candidate hidden state)
/// - h_t = (1 - z_t * a_t) * h_{t-1} + z_t * a_t * h_tilde  (with attention)
///
/// where a_t is the attention weight for time step t.
///
/// # Example
///
/// ```
/// use monolith_layers::agru::AGRU;
/// use monolith_layers::tensor::Tensor;
///
/// let agru = AGRU::new(32, 64);
/// let input = Tensor::zeros(&[8, 10, 32]); // batch=8, seq_len=10, input_dim=32
/// let attention = Tensor::ones(&[8, 10]); // attention weights
/// let output = agru.forward_with_attention(&input, &attention).unwrap();
/// assert_eq!(output.shape(), &[8, 64]); // final hidden state
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AGRU {
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
    /// Cached values for backward pass
    cache: Option<AGRUCache>,
}

/// Cached values from AGRU forward pass for backward computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AGRUCache {
    /// Input sequence
    inputs: Tensor,
    /// Hidden states at each time step
    hidden_states: Vec<Tensor>,
    /// Reset gate values
    reset_gates: Vec<Tensor>,
    /// Update gate values
    update_gates: Vec<Tensor>,
    /// Candidate hidden states
    candidates: Vec<Tensor>,
    /// Attention weights
    attention_weights: Tensor,
}

impl AGRU {
    /// Creates a new AGRU layer.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Dimension of the input features
    /// * `hidden_dim` - Dimension of the hidden state
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self::new_with_initializer(
            input_dim,
            hidden_dim,
            Initializer::HeNormal,
            Initializer::Ones,
        )
    }

    /// Creates a new AGRU layer with custom initializers.
    pub fn new_with_initializer(
        input_dim: usize,
        hidden_dim: usize,
        weight_init: Initializer,
        bias_init: Initializer,
    ) -> Self {
        Self {
            input_dim,
            hidden_dim,
            // Reset gate
            w_r_x: weight_init.initialize(&[input_dim, hidden_dim]),
            w_r_h: weight_init.initialize(&[hidden_dim, hidden_dim]),
            b_r: bias_init.initialize(&[hidden_dim]),
            // Update gate
            w_z_x: weight_init.initialize(&[input_dim, hidden_dim]),
            w_z_h: weight_init.initialize(&[hidden_dim, hidden_dim]),
            b_z: bias_init.initialize(&[hidden_dim]),
            // Candidate hidden state
            w_h_x: weight_init.initialize(&[input_dim, hidden_dim]),
            w_h_h: weight_init.initialize(&[hidden_dim, hidden_dim]),
            b_h: bias_init.initialize(&[hidden_dim]),
            cache: None,
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

    /// Performs forward pass with attention weights.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, seq_len, input_dim]
    /// * `attention` - Attention weights of shape [batch_size, seq_len]
    ///
    /// # Returns
    ///
    /// The final hidden state of shape [batch_size, hidden_dim]
    pub fn forward_with_attention(
        &self,
        input: &Tensor,
        attention: &Tensor,
    ) -> Result<Tensor, LayerError> {
        // Validate input shapes
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!(
                    "AGRU expects 3D input [batch, seq, dim], got {}D",
                    input.ndim()
                ),
            });
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let input_dim = input.shape()[2];

        if input_dim != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input_dim,
            });
        }

        if attention.shape() != [batch_size, seq_len] {
            return Err(LayerError::ShapeMismatch {
                expected: vec![batch_size, seq_len],
                actual: attention.shape().to_vec(),
            });
        }

        // Initialize hidden state to zeros
        let mut h = Tensor::zeros(&[batch_size, self.hidden_dim]);

        // Process each time step
        for t in 0..seq_len {
            // Extract input at time t: [batch_size, input_dim]
            let x_t = self.extract_timestep(input, t, batch_size, input_dim);

            // Extract attention at time t: [batch_size]
            let a_t = self.extract_attention(attention, t, batch_size);

            // Compute gates
            // r_t = sigmoid(x_t @ W_r_x + h @ W_r_h + b_r)
            let r_t = self.compute_reset_gate(&x_t, &h)?;

            // z_t = sigmoid(x_t @ W_z_x + h @ W_z_h + b_z)
            let z_t = self.compute_update_gate(&x_t, &h)?;

            // h_tilde = tanh(x_t @ W_h_x + (r_t * h) @ W_h_h + b_h)
            let h_tilde = self.compute_candidate(&x_t, &h, &r_t)?;

            // h = (1 - z_t * a_t) * h + z_t * a_t * h_tilde
            h = self.compute_hidden_state(&h, &h_tilde, &z_t, &a_t)?;
        }

        Ok(h)
    }

    /// Extracts a single timestep from the input tensor.
    fn extract_timestep(
        &self,
        input: &Tensor,
        t: usize,
        batch_size: usize,
        input_dim: usize,
    ) -> Tensor {
        let _ = (batch_size, input_dim);
        input.narrow(1, t, 1).squeeze(1).contiguous()
    }

    /// Extracts attention weights for a single timestep.
    fn extract_attention(&self, attention: &Tensor, t: usize, batch_size: usize) -> Tensor {
        let _ = batch_size;
        attention.narrow(1, t, 1).squeeze(1).contiguous()
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
        Ok(sum.tanh())
    }

    /// Computes the new hidden state with attention.
    fn compute_hidden_state(
        &self,
        h: &Tensor,
        h_tilde: &Tensor,
        z: &Tensor,
        a: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let batch_size = h.shape()[0];
        let hidden_dim = h.shape()[1];
        let att = a
            .reshape(&[batch_size, 1])
            .broadcast_as(&[batch_size, hidden_dim]);
        let z_att = z.mul(&att);
        let one = Tensor::ones(&[batch_size, hidden_dim]);
        let h_new = one.sub(&z_att).mul(h).add(&z_att.mul(h_tilde));
        Ok(h_new)
    }

    /// Performs forward pass with uniform attention (standard GRU).
    pub fn forward_standard(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!("AGRU expects 3D input, got {}D", input.ndim()),
            });
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Use uniform attention weights (standard GRU behavior)
        let attention = Tensor::ones(&[batch_size, seq_len]);
        self.forward_with_attention(input, &attention)
    }

    /// Returns all hidden states for each time step.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, seq_len, input_dim]
    /// * `attention` - Attention weights of shape [batch_size, seq_len]
    ///
    /// # Returns
    ///
    /// Hidden states of shape [batch_size, seq_len, hidden_dim]
    pub fn forward_all_states(
        &self,
        input: &Tensor,
        attention: &Tensor,
    ) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!("AGRU expects 3D input, got {}D", input.ndim()),
            });
        }

        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let input_dim = input.shape()[2];

        if input_dim != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input_dim,
            });
        }

        // Collect all hidden states
        let mut states: Vec<Tensor> = Vec::with_capacity(seq_len);
        let mut h = Tensor::zeros(&[batch_size, self.hidden_dim]);

        for t in 0..seq_len {
            let x_t = self.extract_timestep(input, t, batch_size, input_dim);
            let a_t = self.extract_attention(attention, t, batch_size);

            let r_t = self.compute_reset_gate(&x_t, &h)?;
            let z_t = self.compute_update_gate(&x_t, &h)?;
            let h_tilde = self.compute_candidate(&x_t, &h, &r_t)?;
            h = self.compute_hidden_state(&h, &h_tilde, &z_t, &a_t)?;
            states.push(h.reshape(&[batch_size, 1, self.hidden_dim]));
        }

        Ok(Tensor::cat(&states, 1))
    }
}

impl Layer for AGRU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Default forward uses uniform attention (standard GRU behavior)
        self.forward_standard(input)
    }

    fn backward(&mut self, _grad: &Tensor) -> Result<Tensor, LayerError> {
        // Full BPTT implementation would go here
        // For now, return a placeholder
        Err(LayerError::BackwardError {
            message: "AGRU backward pass not yet implemented".to_string(),
        })
    }

    fn parameters(&self) -> Vec<&Tensor> {
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

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
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

    fn name(&self) -> &str {
        "AGRU"
    }
}

/// Configuration for AGRU layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AGRUConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Dropout rate (for regularization)
    pub dropout: f32,
    /// Whether to use bidirectional processing
    pub bidirectional: bool,
}

impl AGRUConfig {
    /// Creates a new AGRU configuration.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            dropout: 0.0,
            bidirectional: false,
        }
    }

    /// Sets the dropout rate.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Enables bidirectional processing.
    pub fn bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }

    /// Builds an AGRU layer from this configuration.
    pub fn build(self) -> AGRU {
        AGRU::new(self.input_dim, self.hidden_dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agru_creation() {
        let agru = AGRU::new(32, 64);
        assert_eq!(agru.input_dim(), 32);
        assert_eq!(agru.hidden_dim(), 64);
    }

    #[test]
    fn test_agru_forward_with_attention() {
        let agru = AGRU::new(8, 16);

        // Batch of 2, sequence length 5, input dim 8
        let input = Tensor::rand(&[2, 5, 8]);
        let attention = Tensor::ones(&[2, 5]);

        let output = agru.forward_with_attention(&input, &attention).unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_agru_forward_standard() {
        let agru = AGRU::new(8, 16);
        let input = Tensor::rand(&[2, 5, 8]);

        let output = agru.forward_standard(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_agru_forward_all_states() {
        let agru = AGRU::new(8, 16);
        let input = Tensor::rand(&[2, 5, 8]);
        let attention = Tensor::ones(&[2, 5]);

        let output = agru.forward_all_states(&input, &attention).unwrap();
        assert_eq!(output.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_agru_layer_trait() {
        let agru = AGRU::new(8, 16);
        let input = Tensor::rand(&[2, 5, 8]);

        let output = agru.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16]);
        assert_eq!(agru.name(), "AGRU");
    }

    #[test]
    fn test_agru_parameters() {
        let agru = AGRU::new(8, 16);
        let params = agru.parameters();

        // 3 gates * 3 tensors each (w_x, w_h, b)
        assert_eq!(params.len(), 9);
    }

    #[test]
    fn test_agru_invalid_input() {
        let agru = AGRU::new(8, 16);
        let input = Tensor::rand(&[2, 8]); // 2D instead of 3D

        let result = agru.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_agru_wrong_input_dim() {
        let agru = AGRU::new(8, 16);
        let input = Tensor::rand(&[2, 5, 16]); // Wrong input dim

        let result = agru.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_agru_attention_shape_mismatch() {
        let agru = AGRU::new(8, 16);
        let input = Tensor::rand(&[2, 5, 8]);
        let attention = Tensor::ones(&[2, 3]); // Wrong attention shape

        let result = agru.forward_with_attention(&input, &attention);
        assert!(result.is_err());
    }

    #[test]
    fn test_agru_config() {
        let config = AGRUConfig::new(32, 64).with_dropout(0.1).bidirectional();

        assert_eq!(config.input_dim, 32);
        assert_eq!(config.hidden_dim, 64);
        assert_eq!(config.dropout, 0.1);
        assert!(config.bidirectional);

        let agru = config.build();
        assert_eq!(agru.input_dim(), 32);
        assert_eq!(agru.hidden_dim(), 64);
    }

    #[test]
    fn test_agru_attention_effect() {
        let agru = AGRU::new(4, 8);
        let input = Tensor::rand(&[1, 3, 4]);

        // With full attention
        let full_attention = Tensor::ones(&[1, 3]);
        let output_full = agru
            .forward_with_attention(&input, &full_attention)
            .unwrap();

        // With zero attention (should preserve initial hidden state more)
        let zero_attention = Tensor::zeros(&[1, 3]);
        let output_zero = agru
            .forward_with_attention(&input, &zero_attention)
            .unwrap();

        // Outputs should be different
        let diff: f32 = output_full
            .data()
            .iter()
            .zip(output_zero.data().iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.0);
    }
}
