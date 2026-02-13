//! Adagrad optimizer.
//!
//! Adagrad adapts the learning rate for each parameter based on the
//! historical sum of squared gradients. This allows for larger updates
//! on infrequent parameters and smaller updates on frequent ones.
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Adagrad, OptimizerConfig};
//!
//! let config = OptimizerConfig::Adagrad {
//!     learning_rate: 0.01,
//!     initial_accumulator: 0.1,
//!     weight_decay: 0.0,
//! };
//! let mut adagrad = Adagrad::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! adagrad.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// Adagrad optimizer with per-parameter adaptive learning rates.
///
/// Updates embeddings using the formula:
/// ```text
/// accumulator = accumulator + gradient^2
/// embedding = embedding - learning_rate * gradient / sqrt(accumulator)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adagrad {
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Initial value for the accumulator.
    initial_accumulator: f32,
    /// Weight decay (L2 regularization) coefficient.
    weight_decay: f32,
    /// Accumulated squared gradients for each parameter.
    accumulator: Vec<f32>,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Adagrad {
    /// Creates a new Adagrad optimizer with the given parameters.
    pub fn with_params(learning_rate: f32, initial_accumulator: f32, weight_decay: f32) -> Self {
        let config = OptimizerConfig::Adagrad {
            learning_rate,
            initial_accumulator,
            weight_decay,
        };
        Self {
            learning_rate,
            initial_accumulator,
            weight_decay,
            accumulator: Vec::new(),
            config,
        }
    }

    /// Returns the current accumulator state.
    pub fn accumulator(&self) -> &[f32] {
        &self.accumulator
    }

    /// Resets the accumulator state.
    pub fn reset_state(&mut self) {
        self.accumulator.clear();
    }
}

impl Optimizer for Adagrad {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Adagrad {
                learning_rate,
                initial_accumulator,
                weight_decay,
            } => Ok(Self {
                learning_rate,
                initial_accumulator,
                weight_decay,
                accumulator: Vec::new(),
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Adagrad".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        // Initialize accumulator if needed
        if self.accumulator.len() != embedding.len() {
            self.accumulator = vec![self.initial_accumulator; embedding.len()];
        }

        for (i, (e, g)) in embedding.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad = *g + self.weight_decay * *e;

            // Update accumulator
            self.accumulator[i] += grad * grad;

            // Update embedding
            *e -= self.learning_rate * grad / self.accumulator[i].sqrt();
        }
    }

    fn config(&self) -> &OptimizerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adagrad_basic_update() {
        let config = OptimizerConfig::Adagrad {
            learning_rate: 0.1,
            initial_accumulator: 0.0,
            weight_decay: 0.0,
        };
        let mut adagrad = Adagrad::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![1.0, 1.0, 1.0];

        adagrad.apply_gradients(&mut embedding, &gradients);

        // After first update: accumulator = 1.0, update = 0.1 * 1 / sqrt(1) = 0.1
        assert!((embedding[0] - 0.9).abs() < 1e-6);
        assert!((embedding[1] - 1.9).abs() < 1e-6);
        assert!((embedding[2] - 2.9).abs() < 1e-6);
    }

    #[test]
    fn test_adagrad_accumulator_growth() {
        let config = OptimizerConfig::Adagrad {
            learning_rate: 0.1,
            initial_accumulator: 0.0,
            weight_decay: 0.0,
        };
        let mut adagrad = Adagrad::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        // First update
        adagrad.apply_gradients(&mut embedding, &gradients);
        let first_update = 1.0 - embedding[0];

        // Second update - should be smaller due to accumulator
        adagrad.apply_gradients(&mut embedding, &gradients);
        let second_update = (1.0 - first_update) - embedding[0];

        assert!(second_update < first_update);
    }

    #[test]
    fn test_adagrad_with_initial_accumulator() {
        let config = OptimizerConfig::Adagrad {
            learning_rate: 0.1,
            initial_accumulator: 1.0,
            weight_decay: 0.0,
        };
        let mut adagrad = Adagrad::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        adagrad.apply_gradients(&mut embedding, &gradients);

        // accumulator = 1.0 + 1.0 = 2.0, update = 0.1 * 1 / sqrt(2) ≈ 0.0707
        let expected = 1.0 - 0.1 / 2.0_f32.sqrt();
        assert!((embedding[0] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_adagrad_with_weight_decay() {
        let config = OptimizerConfig::Adagrad {
            learning_rate: 0.1,
            initial_accumulator: 0.0,
            weight_decay: 0.1,
        };
        let mut adagrad = Adagrad::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![0.0]; // Zero gradient, only weight decay

        adagrad.apply_gradients(&mut embedding, &gradients);

        // grad = 0 + 0.1 * 1.0 = 0.1
        // accumulator = 0.01
        // update = 0.1 * 0.1 / sqrt(0.01) = 0.1
        assert!((embedding[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_adagrad_config_mismatch() {
        let config = OptimizerConfig::Sgd {
            learning_rate: 0.01,
        };
        let result = Adagrad::new(config);
        result.expect_err("Adagrad constructor should fail when config variant is not Adagrad");
    }

    #[test]
    fn test_adagrad_reset_state() {
        let mut adagrad = Adagrad::with_params(0.1, 0.0, 0.0);
        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        adagrad.apply_gradients(&mut embedding, &gradients);
        assert_eq!(adagrad.accumulator().len(), 2);

        adagrad.reset_state();
        assert!(adagrad.accumulator().is_empty());
    }
}
