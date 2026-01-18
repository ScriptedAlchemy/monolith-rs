//! Adam optimizer.
//!
//! Adam (Adaptive Moment Estimation) combines the benefits of momentum
//! and RMSprop by maintaining exponential moving averages of both the
//! gradients (first moment) and squared gradients (second moment).
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Adam, OptimizerConfig};
//!
//! let config = OptimizerConfig::Adam {
//!     learning_rate: 0.001,
//!     beta1: 0.9,
//!     beta2: 0.999,
//!     epsilon: 1e-8,
//! };
//! let mut adam = Adam::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! adam.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// Adam optimizer with adaptive learning rates and momentum.
///
/// Updates embeddings using the formula:
/// ```text
/// m = beta1 * m + (1 - beta1) * gradient
/// v = beta2 * v + (1 - beta2) * gradient^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// embedding = embedding - learning_rate * m_hat / (sqrt(v_hat) + epsilon)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adam {
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Exponential decay rate for first moment estimates.
    beta1: f32,
    /// Exponential decay rate for second moment estimates.
    beta2: f32,
    /// Small constant for numerical stability.
    epsilon: f32,
    /// First moment estimates (mean of gradients).
    m: Vec<f32>,
    /// Second moment estimates (mean of squared gradients).
    v: Vec<f32>,
    /// Current timestep for bias correction.
    t: u64,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Adam {
    /// Creates a new Adam optimizer with the given parameters.
    pub fn with_params(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        let config = OptimizerConfig::Adam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
        };
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
            config,
        }
    }

    /// Returns the current first moment state.
    pub fn first_moment(&self) -> &[f32] {
        &self.m
    }

    /// Returns the current second moment state.
    pub fn second_moment(&self) -> &[f32] {
        &self.v
    }

    /// Returns the current timestep.
    pub fn timestep(&self) -> u64 {
        self.t
    }

    /// Resets the optimizer state.
    pub fn reset_state(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t = 0;
    }
}

impl Optimizer for Adam {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => Ok(Self {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                m: Vec::new(),
                v: Vec::new(),
                t: 0,
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Adam".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        // Initialize state if needed
        if self.m.len() != embedding.len() {
            self.m = vec![0.0; embedding.len()];
            self.v = vec![0.0; embedding.len()];
        }

        // Increment timestep
        self.t += 1;

        // Compute bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (e, g)) in embedding.iter_mut().zip(gradients.iter()).enumerate() {
            // Update first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;

            // Update second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Compute bias-corrected estimates
            let m_hat = self.m[i] / bias_correction1;
            let v_hat = self.v[i] / bias_correction2;

            // Update embedding
            *e -= self.learning_rate * m_hat / (v_hat.sqrt() + self.epsilon);
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
    fn test_adam_basic_update() {
        let config = OptimizerConfig::Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut adam = Adam::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![1.0, 1.0, 1.0];

        adam.apply_gradients(&mut embedding, &gradients);

        // All should decrease (positive gradient with negative update)
        assert!(embedding[0] < 1.0);
        assert!(embedding[1] < 2.0);
        assert!(embedding[2] < 3.0);
    }

    #[test]
    fn test_adam_timestep_increment() {
        let config = OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut adam = Adam::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        assert_eq!(adam.timestep(), 0);

        adam.apply_gradients(&mut embedding, &gradients);
        assert_eq!(adam.timestep(), 1);

        adam.apply_gradients(&mut embedding, &gradients);
        assert_eq!(adam.timestep(), 2);
    }

    #[test]
    fn test_adam_momentum_effect() {
        let config = OptimizerConfig::Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut adam = Adam::new(config).unwrap();

        let mut embedding = vec![0.0];
        let gradients = vec![1.0];

        // Apply several updates
        for _ in 0..10 {
            adam.apply_gradients(&mut embedding, &gradients);
        }

        // First moment should accumulate
        assert!(adam.first_moment()[0] > 0.0);
        assert!(adam.second_moment()[0] > 0.0);
    }

    #[test]
    fn test_adam_zero_gradient() {
        let config = OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut adam = Adam::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.0, 0.0, 0.0];

        adam.apply_gradients(&mut embedding, &gradients);

        assert!((embedding[0] - 1.0).abs() < 1e-6);
        assert!((embedding[1] - 2.0).abs() < 1e-6);
        assert!((embedding[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_adam_config_mismatch() {
        let config = OptimizerConfig::Sgd {
            learning_rate: 0.01,
        };
        let result = Adam::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_adam_reset_state() {
        let mut adam = Adam::with_params(0.001, 0.9, 0.999, 1e-8);
        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        adam.apply_gradients(&mut embedding, &gradients);
        assert_eq!(adam.timestep(), 1);
        assert_eq!(adam.first_moment().len(), 2);

        adam.reset_state();
        assert_eq!(adam.timestep(), 0);
        assert!(adam.first_moment().is_empty());
        assert!(adam.second_moment().is_empty());
    }
}
