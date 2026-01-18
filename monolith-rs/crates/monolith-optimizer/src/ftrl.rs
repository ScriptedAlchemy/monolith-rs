//! FTRL (Follow The Regularized Leader) optimizer.
//!
//! FTRL is an online learning algorithm that is particularly effective
//! for training sparse models. It produces sparser models than other
//! optimizers due to its L1 regularization.
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Ftrl, OptimizerConfig};
//!
//! let config = OptimizerConfig::Ftrl {
//!     learning_rate: 0.1,
//!     learning_rate_power: -0.5,
//!     l1_reg: 0.0,
//!     l2_reg: 0.0,
//! };
//! let mut ftrl = Ftrl::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! ftrl.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// FTRL optimizer for sparse model training.
///
/// Updates embeddings using the FTRL-Proximal algorithm:
/// ```text
/// n = n + gradient^2
/// sigma = (sqrt(n) - sqrt(n_prev)) / learning_rate
/// z = z + gradient - sigma * embedding
/// if |z| <= l1_reg:
///     embedding = 0
/// else:
///     embedding = -((beta + sqrt(n)) / learning_rate + l2_reg)^(-1) * (z - sign(z) * l1_reg)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ftrl {
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Power for learning rate schedule (typically -0.5).
    learning_rate_power: f32,
    /// L1 regularization strength.
    l1_reg: f32,
    /// L2 regularization strength.
    l2_reg: f32,
    /// Accumulated squared gradients (n in the algorithm).
    accumulator: Vec<f32>,
    /// Linear term (z in the algorithm).
    linear: Vec<f32>,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Ftrl {
    /// Creates a new FTRL optimizer with the given parameters.
    pub fn with_params(
        learning_rate: f32,
        learning_rate_power: f32,
        l1_reg: f32,
        l2_reg: f32,
    ) -> Self {
        let config = OptimizerConfig::Ftrl {
            learning_rate,
            learning_rate_power,
            l1_reg,
            l2_reg,
        };
        Self {
            learning_rate,
            learning_rate_power,
            l1_reg,
            l2_reg,
            accumulator: Vec::new(),
            linear: Vec::new(),
            config,
        }
    }

    /// Returns the current accumulator state.
    pub fn accumulator(&self) -> &[f32] {
        &self.accumulator
    }

    /// Returns the current linear state.
    pub fn linear(&self) -> &[f32] {
        &self.linear
    }

    /// Resets the optimizer state.
    pub fn reset_state(&mut self) {
        self.accumulator.clear();
        self.linear.clear();
    }

    /// Helper function to compute sign.
    fn sign(x: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else if x < 0.0 {
            -1.0
        } else {
            0.0
        }
    }
}

impl Optimizer for Ftrl {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Ftrl {
                learning_rate,
                learning_rate_power,
                l1_reg,
                l2_reg,
            } => Ok(Self {
                learning_rate,
                learning_rate_power,
                l1_reg,
                l2_reg,
                accumulator: Vec::new(),
                linear: Vec::new(),
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Ftrl".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        // Initialize state if needed
        if self.accumulator.len() != embedding.len() {
            self.accumulator = vec![0.0; embedding.len()];
            self.linear = vec![0.0; embedding.len()];
        }

        for (i, (e, g)) in embedding.iter_mut().zip(gradients.iter()).enumerate() {
            let n_prev = self.accumulator[i];
            let n_new = n_prev + g * g;
            self.accumulator[i] = n_new;

            // Compute sigma
            let sigma = (n_new.powf(-self.learning_rate_power)
                - n_prev.powf(-self.learning_rate_power))
                / self.learning_rate;

            // Update linear term
            self.linear[i] += *g - sigma * *e;

            // Compute new embedding using soft thresholding
            let z = self.linear[i];
            if z.abs() <= self.l1_reg {
                *e = 0.0;
            } else {
                let sign_z = Self::sign(z);
                let denominator =
                    n_new.powf(-self.learning_rate_power) / self.learning_rate + self.l2_reg;
                *e = -(z - sign_z * self.l1_reg) / denominator;
            }
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
    fn test_ftrl_basic_update() {
        let config = OptimizerConfig::Ftrl {
            learning_rate: 0.1,
            learning_rate_power: -0.5,
            l1_reg: 0.0,
            l2_reg: 0.0,
        };
        let mut ftrl = Ftrl::new(config).unwrap();

        let mut embedding = vec![0.0, 0.0, 0.0];
        let gradients = vec![1.0, 2.0, 3.0];

        ftrl.apply_gradients(&mut embedding, &gradients);

        // With zero initial embedding and l1_reg=0, embeddings should be non-zero
        assert!(embedding[0] != 0.0);
        assert!(embedding[1] != 0.0);
        assert!(embedding[2] != 0.0);
    }

    #[test]
    fn test_ftrl_sparsity_with_l1() {
        let config = OptimizerConfig::Ftrl {
            learning_rate: 0.1,
            learning_rate_power: -0.5,
            l1_reg: 1.0,
            l2_reg: 0.0,
        };
        let mut ftrl = Ftrl::new(config).unwrap();

        let mut embedding = vec![0.0];
        let gradients = vec![0.1]; // Small gradient

        ftrl.apply_gradients(&mut embedding, &gradients);

        // With high L1 regularization and small gradient, embedding might stay zero
        // This is the sparsity-inducing property of FTRL
        assert!(embedding[0].abs() < 1.0);
    }

    #[test]
    fn test_ftrl_state_accumulation() {
        let config = OptimizerConfig::Ftrl {
            learning_rate: 0.1,
            learning_rate_power: -0.5,
            l1_reg: 0.0,
            l2_reg: 0.0,
        };
        let mut ftrl = Ftrl::new(config).unwrap();

        let mut embedding = vec![0.0];
        let gradients = vec![1.0];

        ftrl.apply_gradients(&mut embedding, &gradients);
        let acc_after_first = ftrl.accumulator()[0];

        ftrl.apply_gradients(&mut embedding, &gradients);
        let acc_after_second = ftrl.accumulator()[0];

        // Accumulator should grow
        assert!(acc_after_second > acc_after_first);
    }

    #[test]
    fn test_ftrl_with_l2() {
        let config = OptimizerConfig::Ftrl {
            learning_rate: 0.1,
            learning_rate_power: -0.5,
            l1_reg: 0.0,
            l2_reg: 1.0,
        };
        let mut ftrl = Ftrl::new(config).unwrap();

        let mut embedding = vec![0.0];
        let gradients = vec![1.0];

        ftrl.apply_gradients(&mut embedding, &gradients);

        // L2 regularization should affect the magnitude
        assert!(embedding[0] != 0.0);
    }

    #[test]
    fn test_ftrl_config_mismatch() {
        let config = OptimizerConfig::Sgd {
            learning_rate: 0.01,
        };
        let result = Ftrl::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_ftrl_reset_state() {
        let mut ftrl = Ftrl::with_params(0.1, -0.5, 0.0, 0.0);
        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        ftrl.apply_gradients(&mut embedding, &gradients);
        assert_eq!(ftrl.accumulator().len(), 2);
        assert_eq!(ftrl.linear().len(), 2);

        ftrl.reset_state();
        assert!(ftrl.accumulator().is_empty());
        assert!(ftrl.linear().is_empty());
    }

    #[test]
    fn test_ftrl_sign_function() {
        assert_eq!(Ftrl::sign(1.0), 1.0);
        assert_eq!(Ftrl::sign(-1.0), -1.0);
        assert_eq!(Ftrl::sign(0.0), 0.0);
    }
}
