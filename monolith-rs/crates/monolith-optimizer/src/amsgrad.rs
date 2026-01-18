//! AMSGrad optimizer.
//!
//! AMSGrad is a variant of Adam that uses the maximum of past squared gradients
//! rather than the exponential average. This modification addresses the convergence
//! issues that can occur with Adam in certain settings.
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Amsgrad, OptimizerConfig};
//!
//! let config = OptimizerConfig::Amsgrad {
//!     learning_rate: 0.001,
//!     beta1: 0.9,
//!     beta2: 0.999,
//!     epsilon: 1e-8,
//!     weight_decay: 0.0,
//! };
//! let mut amsgrad = Amsgrad::new(config).unwrap();
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//! amsgrad.apply_gradients(&mut embedding, &gradients);
//! ```

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// AMSGrad optimizer (Adam variant with max of second moment).
///
/// Updates embeddings using the formula:
/// ```text
/// m = beta1 * m + (1 - beta1) * gradient
/// v = beta2 * v + (1 - beta2) * gradient^2
/// vhat = max(vhat, v)
/// m_hat = m / (1 - beta1^t)
/// embedding = embedding - learning_rate * m_hat / (sqrt(vhat) + epsilon)
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Amsgrad {
    /// Learning rate for gradient updates.
    learning_rate: f32,
    /// Exponential decay rate for first moment estimates.
    beta1: f32,
    /// Exponential decay rate for second moment estimates.
    beta2: f32,
    /// Small constant for numerical stability.
    epsilon: f32,
    /// Weight decay (L2 regularization) coefficient.
    weight_decay: f32,
    /// First moment estimates (mean of gradients).
    m: Vec<f32>,
    /// Second moment estimates (mean of squared gradients).
    v: Vec<f32>,
    /// Maximum of second moment estimates.
    vhat: Vec<f32>,
    /// Current timestep for bias correction.
    t: u64,
    /// Configuration used to create this optimizer.
    config: OptimizerConfig,
}

impl Amsgrad {
    /// Creates a new AMSGrad optimizer with the given parameters.
    pub fn with_params(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        let config = OptimizerConfig::Amsgrad {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
        };
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            vhat: Vec::new(),
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

    /// Returns the current max second moment state.
    pub fn max_second_moment(&self) -> &[f32] {
        &self.vhat
    }

    /// Returns the current timestep.
    pub fn timestep(&self) -> u64 {
        self.t
    }

    /// Resets the optimizer state.
    pub fn reset_state(&mut self) {
        self.m.clear();
        self.v.clear();
        self.vhat.clear();
        self.t = 0;
    }
}

impl Optimizer for Amsgrad {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Amsgrad {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
            } => Ok(Self {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                m: Vec::new(),
                v: Vec::new(),
                vhat: Vec::new(),
                t: 0,
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Amsgrad".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        // Initialize state if needed
        if self.m.len() != embedding.len() {
            self.m = vec![0.0; embedding.len()];
            self.v = vec![0.0; embedding.len()];
            self.vhat = vec![0.0; embedding.len()];
        }

        // Increment timestep
        self.t += 1;

        // Compute bias correction factor for first moment
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);

        for (i, (e, g)) in embedding.iter_mut().zip(gradients.iter()).enumerate() {
            // Apply weight decay
            let grad = *g + self.weight_decay * *e;

            // Update first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad;

            // Update second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grad * grad;

            // Update max of second moment (the key difference from Adam)
            self.vhat[i] = self.vhat[i].max(self.v[i]);

            // Compute bias-corrected first moment
            let m_hat = self.m[i] / bias_correction1;

            // Update embedding using max of second moment (no bias correction for vhat)
            *e -= self.learning_rate * m_hat / (self.vhat[i].sqrt() + self.epsilon);
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
    fn test_amsgrad_basic_update() {
        let config = OptimizerConfig::Amsgrad {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let mut amsgrad = Amsgrad::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![1.0, 1.0, 1.0];

        amsgrad.apply_gradients(&mut embedding, &gradients);

        // All should decrease (positive gradient with negative update)
        assert!(embedding[0] < 1.0);
        assert!(embedding[1] < 2.0);
        assert!(embedding[2] < 3.0);
    }

    #[test]
    fn test_amsgrad_timestep_increment() {
        let config = OptimizerConfig::Amsgrad {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let mut amsgrad = Amsgrad::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![1.0];

        assert_eq!(amsgrad.timestep(), 0);

        amsgrad.apply_gradients(&mut embedding, &gradients);
        assert_eq!(amsgrad.timestep(), 1);

        amsgrad.apply_gradients(&mut embedding, &gradients);
        assert_eq!(amsgrad.timestep(), 2);
    }

    #[test]
    fn test_amsgrad_vhat_increases() {
        let config = OptimizerConfig::Amsgrad {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let mut amsgrad = Amsgrad::new(config).unwrap();

        let mut embedding = vec![0.0];

        // Apply with large gradient
        amsgrad.apply_gradients(&mut embedding, &[10.0]);
        let vhat_after_large = amsgrad.max_second_moment()[0];

        // Apply with small gradient - vhat should not decrease
        amsgrad.apply_gradients(&mut embedding, &[0.1]);
        let vhat_after_small = amsgrad.max_second_moment()[0];

        assert!(vhat_after_small >= vhat_after_large);
    }

    #[test]
    fn test_amsgrad_with_weight_decay() {
        let config = OptimizerConfig::Amsgrad {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.1,
        };
        let mut amsgrad = Amsgrad::new(config).unwrap();

        let mut embedding = vec![1.0];
        let gradients = vec![0.0]; // Zero gradient, only weight decay

        amsgrad.apply_gradients(&mut embedding, &gradients);

        // Should decrease due to weight decay
        assert!(embedding[0] < 1.0);
    }

    #[test]
    fn test_amsgrad_zero_gradient() {
        let config = OptimizerConfig::Amsgrad {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let mut amsgrad = Amsgrad::new(config).unwrap();

        let mut embedding = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.0, 0.0, 0.0];

        amsgrad.apply_gradients(&mut embedding, &gradients);

        assert!((embedding[0] - 1.0).abs() < 1e-6);
        assert!((embedding[1] - 2.0).abs() < 1e-6);
        assert!((embedding[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_amsgrad_config_mismatch() {
        let config = OptimizerConfig::Sgd {
            learning_rate: 0.01,
        };
        let result = Amsgrad::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_amsgrad_reset_state() {
        let mut amsgrad = Amsgrad::with_params(0.001, 0.9, 0.999, 1e-8, 0.0);
        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        amsgrad.apply_gradients(&mut embedding, &gradients);
        assert_eq!(amsgrad.timestep(), 1);
        assert_eq!(amsgrad.first_moment().len(), 2);

        amsgrad.reset_state();
        assert_eq!(amsgrad.timestep(), 0);
        assert!(amsgrad.first_moment().is_empty());
        assert!(amsgrad.second_moment().is_empty());
        assert!(amsgrad.max_second_moment().is_empty());
    }

    #[test]
    fn test_amsgrad_momentum_effect() {
        let config = OptimizerConfig::Amsgrad {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        };
        let mut amsgrad = Amsgrad::new(config).unwrap();

        let mut embedding = vec![0.0];
        let gradients = vec![1.0];

        // Apply several updates
        for _ in 0..10 {
            amsgrad.apply_gradients(&mut embedding, &gradients);
        }

        // First moment should accumulate
        assert!(amsgrad.first_moment()[0] > 0.0);
        assert!(amsgrad.second_moment()[0] > 0.0);
        assert!(amsgrad.max_second_moment()[0] > 0.0);
    }
}
