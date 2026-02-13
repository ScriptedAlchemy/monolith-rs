//! Momentum optimizer.
//!
//! Momentum accelerates gradient descent by accumulating a velocity vector
//! in the direction of persistent reduction in the objective. This helps
//! overcome local minima and speeds up convergence.
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Momentum, OptimizerConfig};
//!
//! let config = OptimizerConfig::Momentum {
//!     learning_rate: 0.01,
//!     momentum: 0.9,
//!     weight_decay: 0.0,
//!     use_nesterov: false,
//! };
//! let mut momentum = Momentum::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! momentum.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// Momentum optimizer with optional Nesterov acceleration.
///
/// Updates embeddings using the formula:
/// ```text
/// velocity = momentum * velocity + gradient
/// embedding = embedding - learning_rate * velocity  (standard)
/// ```
///
/// With Nesterov momentum:
/// ```text
/// velocity = momentum * velocity + gradient
/// embedding = embedding - learning_rate * (momentum * velocity + gradient)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Momentum {
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Momentum coefficient.
    momentum: f32,
    /// Weight decay (L2 regularization) coefficient.
    weight_decay: f32,
    /// Whether to use Nesterov momentum.
    use_nesterov: bool,
    /// Velocity buffer for momentum.
    velocity: Vec<f32>,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Momentum {
    /// Creates a new Momentum optimizer with the given parameters.
    pub fn with_params(
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        use_nesterov: bool,
    ) -> Self {
        let config = OptimizerConfig::Momentum {
            learning_rate,
            momentum,
            weight_decay,
            use_nesterov,
        };
        Self {
            learning_rate,
            momentum,
            weight_decay,
            use_nesterov,
            velocity: Vec::new(),
            config,
        }
    }

    /// Returns the current velocity state.
    pub fn velocity(&self) -> &[f32] {
        &self.velocity
    }

    /// Returns whether Nesterov momentum is enabled.
    pub fn is_nesterov(&self) -> bool {
        self.use_nesterov
    }

    /// Resets the optimizer state.
    pub fn reset_state(&mut self) {
        self.velocity.clear();
    }
}

impl Optimizer for Momentum {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Momentum {
                learning_rate,
                momentum,
                weight_decay,
                use_nesterov,
            } => Ok(Self {
                learning_rate,
                momentum,
                weight_decay,
                use_nesterov,
                velocity: Vec::new(),
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Momentum".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        // Initialize velocity if needed
        if self.velocity.len() != embedding.len() {
            self.velocity = vec![0.0; embedding.len()];
        }

        for (i, (e, g)) in embedding.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad = *g + self.weight_decay * *e;

            // Update velocity
            self.velocity[i] = self.momentum * self.velocity[i] + grad;

            // Update embedding
            if self.use_nesterov {
                // Nesterov: look ahead with the velocity
                *e -= self.learning_rate * (self.momentum * self.velocity[i] + grad);
            } else {
                // Standard momentum
                *e -= self.learning_rate * self.velocity[i];
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
    fn test_momentum_basic_update() {
        let config = OptimizerConfig::Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
            use_nesterov: false,
        };
        let mut momentum = Momentum::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![1.0, 1.0, 1.0];

        momentum.apply_gradients(&mut embedding, &gradients);

        // All should decrease (positive gradient with negative update)
        assert!((embedding[0] - 0.9).abs() < 1e-6);
        assert!((embedding[1] - 1.9).abs() < 1e-6);
        assert!((embedding[2] - 2.9).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_accumulation() {
        let config = OptimizerConfig::Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
            use_nesterov: false,
        };
        let mut momentum = Momentum::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        // First update: velocity = 1.0, embedding = 1.0 - 0.1 * 1.0 = 0.9
        momentum.apply_gradients(&mut embedding, &gradients);
        let first_update = 1.0 - embedding[0];

        // Second update: velocity = 0.9 * 1.0 + 1.0 = 1.9, embedding = 0.9 - 0.1 * 1.9 = 0.71
        momentum.apply_gradients(&mut embedding, &gradients);
        let second_update = 0.9 - embedding[0];

        // Momentum should make second update larger
        assert!(second_update > first_update);
    }

    #[test]
    fn test_momentum_nesterov() {
        let config_standard = OptimizerConfig::Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
            use_nesterov: false,
        };
        let config_nesterov = OptimizerConfig::Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
            use_nesterov: true,
        };

        let mut standard = Momentum::new(config_standard).unwrap();
        let mut nesterov = Momentum::new(config_nesterov).unwrap();

        let mut embedding_standard = vec![1.0];
        let mut embedding_nesterov = vec![1.0];
        let gradients = vec![1.0];

        standard.apply_gradients(&mut embedding_standard, &gradients);
        nesterov.apply_gradients(&mut embedding_nesterov, &gradients);

        // Nesterov should have a larger update on first step
        assert!(embedding_nesterov[0] < embedding_standard[0]);
        assert!(nesterov.is_nesterov());
        assert!(!standard.is_nesterov());
    }

    #[test]
    fn test_momentum_with_weight_decay() {
        let config = OptimizerConfig::Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.1,
            use_nesterov: false,
        };
        let mut momentum = Momentum::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![0.0]; // Zero gradient, only weight decay

        momentum.apply_gradients(&mut embedding, &gradients);

        // Should decrease due to weight decay
        // grad = 0 + 0.1 * 1.0 = 0.1
        // velocity = 0.1
        // embedding = 1.0 - 0.1 * 0.1 = 0.99
        assert!((embedding[0] - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_zero_gradient() {
        let config = OptimizerConfig::Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
            use_nesterov: false,
        };
        let mut momentum = Momentum::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.0, 0.0, 0.0];

        momentum.apply_gradients(&mut embedding, &gradients);

        assert!((embedding[0] - 1.0).abs() < 1e-6);
        assert!((embedding[1] - 2.0).abs() < 1e-6);
        assert!((embedding[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_momentum_config_mismatch() {
        let config = OptimizerConfig::Sgd {
            learning_rate: 0.01,
        };
        let result = Momentum::new(config);
        result.expect_err("Momentum constructor should fail when config variant is not Momentum");
    }

    #[test]
    fn test_momentum_reset_state() {
        let mut momentum = Momentum::with_params(0.1, 0.9, 0.0, false);
        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        momentum.apply_gradients(&mut embedding, &gradients);
        assert_eq!(momentum.velocity().len(), 2);

        momentum.reset_state();
        assert!(momentum.velocity().is_empty());
    }

    #[test]
    fn test_momentum_velocity_decay() {
        let config = OptimizerConfig::Momentum {
            learning_rate: 0.1,
            momentum: 0.9,
            weight_decay: 0.0,
            use_nesterov: false,
        };
        let mut momentum = Momentum::new(config).unwrap();

        let mut embedding = vec![0.0];

        // Apply gradient
        momentum.apply_gradients(&mut embedding, &[1.0]);
        let velocity_after_grad = momentum.velocity()[0];

        // Apply zero gradient multiple times
        for _ in 0..10 {
            momentum.apply_gradients(&mut embedding, &[0.0]);
        }

        // Velocity should decay towards zero
        assert!(momentum.velocity()[0].abs() < velocity_after_grad.abs());
    }
}
