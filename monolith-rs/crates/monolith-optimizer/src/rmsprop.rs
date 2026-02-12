//! RMSprop optimizer.
//!
//! RMSprop maintains a moving average of squared gradients to normalize
//! the gradient. It adapts the learning rate for each parameter, making
//! it well-suited for non-stationary objectives and online settings.
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Rmsprop, OptimizerConfig};
//!
//! let config = OptimizerConfig::Rmsprop {
//!     learning_rate: 0.001,
//!     decay: 0.9,
//!     momentum: 0.0,
//!     epsilon: 1e-8,
//! };
//! let mut rmsprop = Rmsprop::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! rmsprop.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// RMSprop optimizer with adaptive learning rates.
///
/// Updates embeddings using the formula:
/// ```text
/// mean_square = decay * mean_square + (1 - decay) * gradient^2
/// if momentum > 0:
///     mom = momentum * mom + learning_rate * gradient / sqrt(mean_square + epsilon)
///     embedding = embedding - mom
/// else:
///     embedding = embedding - learning_rate * gradient / sqrt(mean_square + epsilon)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rmsprop {
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Decay rate for the moving average of squared gradients.
    decay: f32,
    /// Momentum coefficient.
    momentum: f32,
    /// Small constant for numerical stability.
    epsilon: f32,
    /// Moving average of squared gradients.
    mean_square: Vec<f32>,
    /// Momentum buffer.
    mom: Vec<f32>,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Rmsprop {
    /// Creates a new RMSprop optimizer with the given parameters.
    pub fn with_params(learning_rate: f32, decay: f32, momentum: f32, epsilon: f32) -> Self {
        let config = OptimizerConfig::Rmsprop {
            learning_rate,
            decay,
            momentum,
            epsilon,
        };
        Self {
            learning_rate,
            decay,
            momentum,
            epsilon,
            mean_square: Vec::new(),
            mom: Vec::new(),
            config,
        }
    }

    /// Returns the current mean square state.
    pub fn mean_square(&self) -> &[f32] {
        &self.mean_square
    }

    /// Returns the current momentum buffer.
    pub fn momentum_buffer(&self) -> &[f32] {
        &self.mom
    }

    /// Resets the optimizer state.
    pub fn reset_state(&mut self) {
        self.mean_square.clear();
        self.mom.clear();
    }
}

impl Optimizer for Rmsprop {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Rmsprop {
                learning_rate,
                decay,
                momentum,
                epsilon,
            } => Ok(Self {
                learning_rate,
                decay,
                momentum,
                epsilon,
                mean_square: Vec::new(),
                mom: Vec::new(),
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Rmsprop".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        // Initialize state if needed
        if self.mean_square.len() != embedding.len() {
            self.mean_square = vec![0.0; embedding.len()];
            if self.momentum > 0.0 {
                self.mom = vec![0.0; embedding.len()];
            }
        }

        for (i, (e, g)) in embedding.iter_mut().zip(gradients.iter()).enumerate() {
            // Update moving average of squared gradients
            self.mean_square[i] = self.decay * self.mean_square[i] + (1.0 - self.decay) * g * g;

            if self.momentum > 0.0 {
                // With momentum
                self.mom[i] = self.momentum * self.mom[i]
                    + self.learning_rate * g / (self.mean_square[i] + self.epsilon).sqrt();
                *e -= self.mom[i];
            } else {
                // Without momentum
                *e -= self.learning_rate * g / (self.mean_square[i] + self.epsilon).sqrt();
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
    fn test_rmsprop_basic_update() {
        let config = OptimizerConfig::Rmsprop {
            learning_rate: 0.1,
            decay: 0.9,
            momentum: 0.0,
            epsilon: 1e-8,
        };
        let mut rmsprop = Rmsprop::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![1.0, 1.0, 1.0];

        rmsprop.apply_gradients(&mut embedding, &gradients);

        // All should decrease (positive gradient)
        assert!(embedding[0] < 1.0);
        assert!(embedding[1] < 2.0);
        assert!(embedding[2] < 3.0);
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let config = OptimizerConfig::Rmsprop {
            learning_rate: 0.1,
            decay: 0.9,
            momentum: 0.9,
            epsilon: 1e-8,
        };
        let mut rmsprop = Rmsprop::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        // First update
        rmsprop.apply_gradients(&mut embedding, &gradients);

        // Momentum buffer should be non-zero
        assert!(rmsprop.momentum_buffer()[0] > 0.0);
    }

    #[test]
    fn test_rmsprop_mean_square_decay() {
        let config = OptimizerConfig::Rmsprop {
            learning_rate: 0.01,
            decay: 0.9,
            momentum: 0.0,
            epsilon: 1e-8,
        };
        let mut rmsprop = Rmsprop::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        rmsprop.apply_gradients(&mut embedding, &gradients);
        // mean_square = 0.9 * 0 + 0.1 * 1 = 0.1
        assert!((rmsprop.mean_square()[0] - 0.1).abs() < 1e-6);

        rmsprop.apply_gradients(&mut embedding, &gradients);
        // mean_square = 0.9 * 0.1 + 0.1 * 1 = 0.19
        assert!((rmsprop.mean_square()[0] - 0.19).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_zero_gradient() {
        let config = OptimizerConfig::Rmsprop {
            learning_rate: 0.1,
            decay: 0.9,
            momentum: 0.0,
            epsilon: 1e-8,
        };
        let mut rmsprop = Rmsprop::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.0, 0.0, 0.0];

        rmsprop.apply_gradients(&mut embedding, &gradients);

        assert!((embedding[0] - 1.0).abs() < 1e-6);
        assert!((embedding[1] - 2.0).abs() < 1e-6);
        assert!((embedding[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmsprop_adaptive_learning() {
        let config = OptimizerConfig::Rmsprop {
            learning_rate: 0.1,
            decay: 0.9,
            momentum: 0.0,
            epsilon: 1e-8,
        };
        let mut rmsprop = Rmsprop::new(config).unwrap();

        let mut embedding = vec![0.0, 0.0];
        let gradients = vec![1.0, 10.0]; // Different gradient magnitudes

        rmsprop.apply_gradients(&mut embedding, &gradients);

        // The larger gradient should not result in 10x the update
        // due to adaptive scaling
        let ratio = embedding[1].abs() / embedding[0].abs();
        assert!(ratio < 10.0);
    }

    #[test]
    fn test_rmsprop_config_mismatch() {
        let config = OptimizerConfig::Sgd {
            learning_rate: 0.01,
        };
        let result = Rmsprop::new(config);
        result.expect_err("RMSprop constructor should fail when config variant is not RMSprop");
    }

    #[test]
    fn test_rmsprop_reset_state() {
        let mut rmsprop = Rmsprop::with_params(0.01, 0.9, 0.9, 1e-8);
        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        rmsprop.apply_gradients(&mut embedding, &gradients);
        assert_eq!(rmsprop.mean_square().len(), 2);
        assert_eq!(rmsprop.momentum_buffer().len(), 2);

        rmsprop.reset_state();
        assert!(rmsprop.mean_square().is_empty());
        assert!(rmsprop.momentum_buffer().is_empty());
    }
}
