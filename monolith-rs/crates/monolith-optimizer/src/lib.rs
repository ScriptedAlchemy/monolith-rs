//! Embedding optimizers for Monolith.
//!
//! This crate provides various optimization algorithms commonly used for
//! training embedding models. Each optimizer implements the [`Optimizer`] trait.
//!
//! # Available Optimizers
//!
//! - [`Sgd`] - Stochastic Gradient Descent
//! - [`Adagrad`] - Adaptive Gradient Algorithm
//! - [`Adam`] - Adaptive Moment Estimation
//! - [`Ftrl`] - Follow The Regularized Leader
//! - [`Rmsprop`] - Root Mean Square Propagation
//! - [`Adadelta`] - Adadelta optimizer
//! - [`Amsgrad`] - AMSGrad (Adam variant)
//! - [`Momentum`] - Momentum optimizer
//!
//! # Example
//!
//! ```
//! use monolith_optimizer::{Optimizer, Sgd, OptimizerConfig};
//!
//! let config = OptimizerConfig::Sgd { learning_rate: 0.01 };
//! let mut optimizer = Sgd::new(config).unwrap();
//!
//! let mut embedding = vec![1.0, 2.0, 3.0];
//! let gradients = vec![0.1, 0.2, 0.3];
//!
//! optimizer.apply_gradients(&mut embedding, &gradients);
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;

mod adadelta;
mod adagrad;
mod adam;
mod amsgrad;
mod ftrl;
mod momentum;
mod rmsprop;
mod sgd;

pub use adadelta::Adadelta;
pub use adagrad::Adagrad;
pub use adam::Adam;
pub use amsgrad::Amsgrad;
pub use ftrl::Ftrl;
pub use momentum::Momentum;
pub use rmsprop::Rmsprop;
pub use sgd::Sgd;

/// Errors that can occur when working with optimizers.
#[derive(Debug, Error)]
pub enum OptimizerError {
    /// Configuration type does not match the optimizer type.
    #[error("Config mismatch: expected {expected}, got {got}")]
    ConfigMismatch { expected: String, got: String },

    /// Invalid configuration parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Configuration for different optimizer types.
///
/// This enum contains the configuration parameters for each supported
/// optimizer type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizerConfig {
    /// Stochastic Gradient Descent configuration.
    Sgd {
        /// Learning rate for gradient updates.
        learning_rate: f32,
    },

    /// Adagrad configuration.
    Adagrad {
        /// Learning rate for gradient updates.
        learning_rate: f32,
        /// Initial value for the accumulator.
        initial_accumulator: f32,
        /// Weight decay (L2 regularization) coefficient.
        weight_decay: f32,
    },

    /// Adam configuration.
    Adam {
        /// Learning rate for gradient updates.
        learning_rate: f32,
        /// Exponential decay rate for first moment estimates.
        beta1: f32,
        /// Exponential decay rate for second moment estimates.
        beta2: f32,
        /// Small constant for numerical stability.
        epsilon: f32,
    },

    /// FTRL configuration.
    Ftrl {
        /// Learning rate for gradient updates.
        learning_rate: f32,
        /// Power for learning rate schedule (typically -0.5).
        learning_rate_power: f32,
        /// L1 regularization strength.
        l1_reg: f32,
        /// L2 regularization strength.
        l2_reg: f32,
    },

    /// RMSprop configuration.
    Rmsprop {
        /// Learning rate for gradient updates.
        learning_rate: f32,
        /// Decay rate for the moving average of squared gradients.
        decay: f32,
        /// Momentum coefficient.
        momentum: f32,
        /// Small constant for numerical stability.
        epsilon: f32,
    },

    /// Adadelta configuration.
    Adadelta {
        /// Decay rate for running average of squared gradients.
        rho: f32,
        /// Small constant for numerical stability.
        epsilon: f32,
        /// Weight decay (L2 regularization) coefficient.
        weight_decay: f32,
    },

    /// AMSGrad configuration.
    Amsgrad {
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
    },

    /// Momentum configuration.
    Momentum {
        /// Learning rate for gradient updates.
        learning_rate: f32,
        /// Momentum coefficient.
        momentum: f32,
        /// Weight decay (L2 regularization) coefficient.
        weight_decay: f32,
        /// Whether to use Nesterov momentum.
        use_nesterov: bool,
    },
}

impl OptimizerConfig {
    /// Returns the name of the optimizer type.
    pub fn name(&self) -> &'static str {
        match self {
            OptimizerConfig::Sgd { .. } => "Sgd",
            OptimizerConfig::Adagrad { .. } => "Adagrad",
            OptimizerConfig::Adam { .. } => "Adam",
            OptimizerConfig::Ftrl { .. } => "Ftrl",
            OptimizerConfig::Rmsprop { .. } => "Rmsprop",
            OptimizerConfig::Adadelta { .. } => "Adadelta",
            OptimizerConfig::Amsgrad { .. } => "Amsgrad",
            OptimizerConfig::Momentum { .. } => "Momentum",
        }
    }

    /// Returns the learning rate for the optimizer.
    ///
    /// Note: Adadelta does not use a traditional learning rate, so this returns 1.0 for it.
    pub fn learning_rate(&self) -> f32 {
        match self {
            OptimizerConfig::Sgd { learning_rate } => *learning_rate,
            OptimizerConfig::Adagrad { learning_rate, .. } => *learning_rate,
            OptimizerConfig::Adam { learning_rate, .. } => *learning_rate,
            OptimizerConfig::Ftrl { learning_rate, .. } => *learning_rate,
            OptimizerConfig::Rmsprop { learning_rate, .. } => *learning_rate,
            OptimizerConfig::Adadelta { .. } => 1.0, // Adadelta doesn't use a traditional learning rate
            OptimizerConfig::Amsgrad { learning_rate, .. } => *learning_rate,
            OptimizerConfig::Momentum { learning_rate, .. } => *learning_rate,
        }
    }
}

/// Trait for embedding optimizers.
///
/// This trait defines the interface that all optimizers must implement.
/// Optimizers are responsible for updating embedding vectors based on
/// computed gradients.
pub trait Optimizer: Sized {
    /// Creates a new optimizer from the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The optimizer configuration.
    ///
    /// # Returns
    ///
    /// A new optimizer instance, or an error if the configuration is invalid.
    ///
    /// # Errors
    ///
    /// Returns [`OptimizerError::ConfigMismatch`] if the configuration type
    /// does not match the optimizer type.
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError>;

    /// Applies gradients to update the embedding vector.
    ///
    /// # Arguments
    ///
    /// * `embedding` - The embedding vector to update (modified in place).
    /// * `gradients` - The gradient vector to apply.
    ///
    /// # Panics
    ///
    /// May panic if `embedding` and `gradients` have different lengths.
    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]);

    /// Returns a reference to the optimizer's configuration.
    fn config(&self) -> &OptimizerConfig;
}

/// Creates an optimizer from the given configuration.
///
/// This is a convenience function that creates the appropriate optimizer
/// type based on the configuration variant.
///
/// # Example
///
/// ```
/// use monolith_optimizer::{create_optimizer, OptimizerConfig, Optimizer};
///
/// let config = OptimizerConfig::Adam {
///     learning_rate: 0.001,
///     beta1: 0.9,
///     beta2: 0.999,
///     epsilon: 1e-8,
/// };
///
/// let optimizer = create_optimizer(config);
/// ```
pub fn create_optimizer(config: OptimizerConfig) -> Box<dyn OptimizerDyn> {
    match &config {
        OptimizerConfig::Sgd { .. } => Box::new(Sgd::new(config).unwrap()),
        OptimizerConfig::Adagrad { .. } => Box::new(Adagrad::new(config).unwrap()),
        OptimizerConfig::Adam { .. } => Box::new(Adam::new(config).unwrap()),
        OptimizerConfig::Ftrl { .. } => Box::new(Ftrl::new(config).unwrap()),
        OptimizerConfig::Rmsprop { .. } => Box::new(Rmsprop::new(config).unwrap()),
        OptimizerConfig::Adadelta { .. } => Box::new(Adadelta::new(config).unwrap()),
        OptimizerConfig::Amsgrad { .. } => Box::new(Amsgrad::new(config).unwrap()),
        OptimizerConfig::Momentum { .. } => Box::new(Momentum::new(config).unwrap()),
    }
}

/// Dynamic dispatch version of the Optimizer trait.
///
/// This trait enables using optimizers as trait objects when dynamic
/// dispatch is needed.
pub trait OptimizerDyn {
    /// Applies gradients to update the embedding vector.
    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]);

    /// Returns a reference to the optimizer's configuration.
    fn config(&self) -> &OptimizerConfig;
}

impl<T: Optimizer> OptimizerDyn for T {
    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        Optimizer::apply_gradients(self, embedding, gradients)
    }

    fn config(&self) -> &OptimizerConfig {
        Optimizer::config(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_config_name() {
        let sgd = OptimizerConfig::Sgd { learning_rate: 0.01 };
        assert_eq!(sgd.name(), "Sgd");

        let adam = OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        assert_eq!(adam.name(), "Adam");
    }

    #[test]
    fn test_optimizer_config_learning_rate() {
        let sgd = OptimizerConfig::Sgd { learning_rate: 0.01 };
        assert!((sgd.learning_rate() - 0.01).abs() < 1e-6);

        let adagrad = OptimizerConfig::Adagrad {
            learning_rate: 0.05,
            initial_accumulator: 0.1,
            weight_decay: 0.0,
        };
        assert!((adagrad.learning_rate() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_create_optimizer() {
        let config = OptimizerConfig::Sgd { learning_rate: 0.01 };
        let mut optimizer = create_optimizer(config);

        let mut embedding = vec![1.0, 2.0];
        let gradients = vec![1.0, 1.0];

        optimizer.apply_gradients(&mut embedding, &gradients);

        assert!(embedding[0] < 1.0);
        assert!(embedding[1] < 2.0);
    }

    #[test]
    fn test_create_all_optimizer_types() {
        let configs = vec![
            OptimizerConfig::Sgd { learning_rate: 0.01 },
            OptimizerConfig::Adagrad {
                learning_rate: 0.01,
                initial_accumulator: 0.1,
                weight_decay: 0.0,
            },
            OptimizerConfig::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            OptimizerConfig::Ftrl {
                learning_rate: 0.1,
                learning_rate_power: -0.5,
                l1_reg: 0.0,
                l2_reg: 0.0,
            },
            OptimizerConfig::Rmsprop {
                learning_rate: 0.001,
                decay: 0.9,
                momentum: 0.0,
                epsilon: 1e-8,
            },
            OptimizerConfig::Adadelta {
                rho: 0.95,
                epsilon: 1e-6,
                weight_decay: 0.0,
            },
            OptimizerConfig::Amsgrad {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: 0.0,
            },
            OptimizerConfig::Momentum {
                learning_rate: 0.01,
                momentum: 0.9,
                weight_decay: 0.0,
                use_nesterov: false,
            },
        ];

        for config in configs {
            let optimizer = create_optimizer(config.clone());
            assert_eq!(optimizer.config().name(), config.name());
        }
    }

    #[test]
    fn test_optimizer_config_serialization() {
        let config = OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: OptimizerConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.name(), deserialized.name());
        assert!((config.learning_rate() - deserialized.learning_rate()).abs() < 1e-6);
    }
}
