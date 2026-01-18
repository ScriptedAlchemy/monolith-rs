//! Adadelta optimizer.
//!
//! Adadelta is an extension of Adagrad that seeks to reduce its aggressive,
//! monotonically decreasing learning rate. Instead of accumulating all past
//! squared gradients, Adadelta restricts the window of accumulated past gradients
//! to some fixed size.
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Adadelta, OptimizerConfig};
//!
//! let config = OptimizerConfig::Adadelta {
//!     rho: 0.95,
//!     epsilon: 1e-6,
//!     weight_decay: 0.0,
//! };
//! let mut adadelta = Adadelta::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! adadelta.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// Adadelta optimizer with adaptive learning rates.
///
/// Updates embeddings using the formula:
/// ```text
/// accum = rho * accum + (1 - rho) * gradient^2
/// update = sqrt(accum_update + epsilon) / sqrt(accum + epsilon) * gradient
/// accum_update = rho * accum_update + (1 - rho) * update^2
/// embedding = embedding - update
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adadelta {
    /// Decay rate for running average of squared gradients.
    rho: f32,
    /// Small constant for numerical stability.
    epsilon: f32,
    /// Weight decay (L2 regularization) coefficient.
    weight_decay: f32,
    /// Running average of squared gradients.
    accum: Vec<f32>,
    /// Running average of squared parameter updates.
    accum_update: Vec<f32>,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Adadelta {
    /// Creates a new Adadelta optimizer with the given parameters.
    pub fn with_params(rho: f32, epsilon: f32, weight_decay: f32) -> Self {
        let config = OptimizerConfig::Adadelta {
            rho,
            epsilon,
            weight_decay,
        };
        Self {
            rho,
            epsilon,
            weight_decay,
            accum: Vec::new(),
            accum_update: Vec::new(),
            config,
        }
    }

    /// Returns the current gradient accumulator state.
    pub fn accum(&self) -> &[f32] {
        &self.accum
    }

    /// Returns the current update accumulator state.
    pub fn accum_update(&self) -> &[f32] {
        &self.accum_update
    }

    /// Resets the optimizer state.
    pub fn reset_state(&mut self) {
        self.accum.clear();
        self.accum_update.clear();
    }
}

impl Optimizer for Adadelta {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Adadelta {
                rho,
                epsilon,
                weight_decay,
            } => Ok(Self {
                rho,
                epsilon,
                weight_decay,
                accum: Vec::new(),
                accum_update: Vec::new(),
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Adadelta".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        // Initialize state if needed
        if self.accum.len() != embedding.len() {
            self.accum = vec![0.0; embedding.len()];
            self.accum_update = vec![0.0; embedding.len()];
        }

        for (i, (e, g)) in embedding.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad = *g + self.weight_decay * *e;

            // Update gradient accumulator
            self.accum[i] = self.rho * self.accum[i] + (1.0 - self.rho) * grad * grad;

            // Compute update using ratio of RMS of previous updates to RMS of gradients
            let update = ((self.accum_update[i] + self.epsilon).sqrt()
                / (self.accum[i] + self.epsilon).sqrt())
                * grad;

            // Update delta accumulator
            self.accum_update[i] =
                self.rho * self.accum_update[i] + (1.0 - self.rho) * update * update;

            // Update embedding
            *e -= update;
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
    fn test_adadelta_basic_update() {
        let config = OptimizerConfig::Adadelta {
            rho: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.0,
        };
        let mut adadelta = Adadelta::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![1.0, 1.0, 1.0];

        adadelta.apply_gradients(&mut embedding, &gradients);

        // All should decrease (positive gradient with negative update)
        assert!(embedding[0] < 1.0);
        assert!(embedding[1] < 2.0);
        assert!(embedding[2] < 3.0);
    }

    #[test]
    fn test_adadelta_accumulator_growth() {
        let config = OptimizerConfig::Adadelta {
            rho: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.0,
        };
        let mut adadelta = Adadelta::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        adadelta.apply_gradients(&mut embedding, &gradients);

        // Accumulators should have values after first update
        assert!(adadelta.accum()[0] > 0.0);
        assert!(adadelta.accum_update()[0] > 0.0);
    }

    #[test]
    fn test_adadelta_with_weight_decay() {
        let config = OptimizerConfig::Adadelta {
            rho: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.1,
        };
        let mut adadelta = Adadelta::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![0.0]; // Zero gradient, only weight decay

        adadelta.apply_gradients(&mut embedding, &gradients);

        // Should decrease due to weight decay
        assert!(embedding[0] < 1.0);
    }

    #[test]
    fn test_adadelta_zero_gradient() {
        let config = OptimizerConfig::Adadelta {
            rho: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.0,
        };
        let mut adadelta = Adadelta::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.0, 0.0, 0.0];

        adadelta.apply_gradients(&mut embedding, &gradients);

        assert!((embedding[0] - 1.0).abs() < 1e-6);
        assert!((embedding[1] - 2.0).abs() < 1e-6);
        assert!((embedding[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_adadelta_config_mismatch() {
        let config = OptimizerConfig::Sgd {
            learning_rate: 0.01,
        };
        let result = Adadelta::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_adadelta_reset_state() {
        let mut adadelta = Adadelta::with_params(0.95, 1e-6, 0.0);
        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        adadelta.apply_gradients(&mut embedding, &gradients);
        assert_eq!(adadelta.accum().len(), 2);
        assert_eq!(adadelta.accum_update().len(), 2);

        adadelta.reset_state();
        assert!(adadelta.accum().is_empty());
        assert!(adadelta.accum_update().is_empty());
    }

    #[test]
    fn test_adadelta_multiple_updates() {
        let config = OptimizerConfig::Adadelta {
            rho: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.0,
        };
        let mut adadelta = Adadelta::new(config).unwrap();

        let mut embedding = vec![0.0];
        let gradients = vec![1.0];

        // Apply several updates
        for _ in 0..10 {
            adadelta.apply_gradients(&mut embedding, &gradients);
        }

        // Adadelta updates are adaptive - just verify it moved in the right direction
        assert!(embedding[0] < 0.0);
    }
}
