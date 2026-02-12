//! Stochastic Gradient Descent (SGD) optimizer.
//!
//! SGD is a simple optimizer that updates parameters by subtracting
//! the gradient scaled by the learning rate.
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Sgd, OptimizerConfig};
//!
//! let config = OptimizerConfig::Sgd { learning_rate: 0.01 };
//! let mut sgd = Sgd::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! sgd.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// Stochastic Gradient Descent optimizer.
///
/// Updates embeddings using the formula:
/// `embedding = embedding - learning_rate * gradient`
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sgd {
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Sgd {
    /// Creates a new SGD optimizer with the given learning rate.
    pub fn with_learning_rate(learning_rate: f32) -> Self {
        let config = OptimizerConfig::Sgd { learning_rate };
        Self {
            learning_rate,
            config,
        }
    }
}

impl Optimizer for Sgd {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Sgd { learning_rate } => Ok(Self {
                learning_rate,
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Sgd".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        for (e, g) in embedding.iter_mut().zip(gradients.iter()) {
            *e -= self.learning_rate * g;
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
    fn test_sgd_basic_update() {
        let config = OptimizerConfig::Sgd { learning_rate: 0.1 };
        let mut sgd = Sgd::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![1.0, 1.0, 1.0];

        sgd.apply_gradients(&mut embedding, &gradients);

        assert!((embedding[0] - 0.9).abs() < 1e-6);
        assert!((embedding[1] - 1.9).abs() < 1e-6);
        assert!((embedding[2] - 2.9).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_zero_gradient() {
        let config = OptimizerConfig::Sgd { learning_rate: 0.1 };
        let mut sgd = Sgd::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.0, 0.0, 0.0];

        sgd.apply_gradients(&mut embedding, &gradients);

        assert!((embedding[0] - 1.0).abs() < 1e-6);
        assert!((embedding[1] - 2.0).abs() < 1e-6);
        assert!((embedding[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_config_mismatch() {
        let config = OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let result = Sgd::new(config);
        result.expect_err("SGD constructor should fail when config variant is not SGD");
    }

    #[test]
    fn test_sgd_with_learning_rate() {
        let sgd = Sgd::with_learning_rate(0.05);
        assert!(matches!(
            sgd.config(),
            OptimizerConfig::Sgd { learning_rate } if (*learning_rate - 0.05).abs() < 1e-6
        ));
    }
}
