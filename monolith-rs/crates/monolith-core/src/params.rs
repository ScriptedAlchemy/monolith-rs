//! Configuration types and hyperparameters for Monolith.
//!
//! This module provides configuration structures and traits for defining
//! model hyperparameters, embedding configurations, and initializer settings.
//!
//! # Overview
//!
//! - [`EmbeddingConfig`]: Configuration for embedding tables.
//! - [`InitializerConfig`]: Configuration for weight initialization.
//! - [`Params`]: Trait for hyperparameter containers.

use serde::{Deserialize, Serialize};

use crate::error::{MonolithError, Result};
use crate::fid::SlotId;

/// Configuration for embedding tables.
///
/// This structure defines the configuration for an embedding table,
/// including its dimension, initialization method, and learning rate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// The slot ID this embedding configuration applies to.
    slot_id: SlotId,

    /// The embedding dimension.
    dim: usize,

    /// The initializer configuration for the embedding weights.
    initializer: InitializerConfig,

    /// The learning rate for this embedding table.
    learning_rate: f32,

    /// L2 regularization coefficient.
    l2_regularization: f32,

    /// Whether to enable gradient clipping.
    gradient_clipping: Option<f32>,

    /// The optimizer type to use.
    optimizer: OptimizerType,
}

impl EmbeddingConfig {
    /// Creates a new embedding configuration with default settings.
    ///
    /// # Arguments
    ///
    /// * `slot_id` - The slot ID this configuration applies to.
    /// * `dim` - The embedding dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_core::params::EmbeddingConfig;
    ///
    /// let config = EmbeddingConfig::new(1, 64);
    /// assert_eq!(config.slot_id(), 1);
    /// assert_eq!(config.dim(), 64);
    /// ```
    pub fn new(slot_id: SlotId, dim: usize) -> Self {
        Self {
            slot_id,
            dim,
            initializer: InitializerConfig::default(),
            learning_rate: 0.001,
            l2_regularization: 0.0,
            gradient_clipping: None,
            optimizer: OptimizerType::default(),
        }
    }

    /// Creates a builder for embedding configuration.
    pub fn builder(slot_id: SlotId, dim: usize) -> EmbeddingConfigBuilder {
        EmbeddingConfigBuilder::new(slot_id, dim)
    }

    /// Returns the slot ID.
    #[inline]
    pub fn slot_id(&self) -> SlotId {
        self.slot_id
    }

    /// Returns the embedding dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the initializer configuration.
    #[inline]
    pub fn initializer(&self) -> &InitializerConfig {
        &self.initializer
    }

    /// Returns the learning rate.
    #[inline]
    pub fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Returns the L2 regularization coefficient.
    #[inline]
    pub fn l2_regularization(&self) -> f32 {
        self.l2_regularization
    }

    /// Returns the gradient clipping threshold.
    #[inline]
    pub fn gradient_clipping(&self) -> Option<f32> {
        self.gradient_clipping
    }

    /// Returns the optimizer type.
    #[inline]
    pub fn optimizer(&self) -> &OptimizerType {
        &self.optimizer
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.dim == 0 {
            return Err(MonolithError::ConfigError {
                message: "Embedding dimension must be greater than 0".to_string(),
            });
        }

        if self.learning_rate <= 0.0 {
            return Err(MonolithError::ConfigError {
                message: "Learning rate must be positive".to_string(),
            });
        }

        if self.l2_regularization < 0.0 {
            return Err(MonolithError::ConfigError {
                message: "L2 regularization must be non-negative".to_string(),
            });
        }

        self.initializer.validate()?;

        Ok(())
    }
}

/// Builder for [`EmbeddingConfig`].
#[derive(Debug)]
pub struct EmbeddingConfigBuilder {
    config: EmbeddingConfig,
}

impl EmbeddingConfigBuilder {
    /// Creates a new builder.
    pub fn new(slot_id: SlotId, dim: usize) -> Self {
        Self {
            config: EmbeddingConfig::new(slot_id, dim),
        }
    }

    /// Sets the initializer.
    pub fn initializer(mut self, initializer: InitializerConfig) -> Self {
        self.config.initializer = initializer;
        self
    }

    /// Sets the learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.config.learning_rate = lr;
        self
    }

    /// Sets the L2 regularization coefficient.
    pub fn l2_regularization(mut self, l2: f32) -> Self {
        self.config.l2_regularization = l2;
        self
    }

    /// Sets the gradient clipping threshold.
    pub fn gradient_clipping(mut self, clip: f32) -> Self {
        self.config.gradient_clipping = Some(clip);
        self
    }

    /// Sets the optimizer type.
    pub fn optimizer(mut self, optimizer: OptimizerType) -> Self {
        self.config.optimizer = optimizer;
        self
    }

    /// Builds the configuration, validating it first.
    pub fn build(self) -> Result<EmbeddingConfig> {
        self.config.validate()?;
        Ok(self.config)
    }
}

/// Configuration for weight initialization.
///
/// This enum defines different initialization strategies for model weights.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InitializerConfig {
    /// Initialize all weights to zero.
    Zeros,

    /// Initialize all weights to one.
    Ones,

    /// Initialize all weights to a constant value.
    Constant {
        /// The constant value to use.
        value: f32,
    },

    /// Initialize weights with uniform random values.
    RandomUniform {
        /// The minimum value.
        min: f32,
        /// The maximum value.
        max: f32,
    },

    /// Initialize weights with normal (Gaussian) random values.
    RandomNormal {
        /// The mean of the distribution.
        mean: f32,
        /// The standard deviation of the distribution.
        stddev: f32,
    },

    /// Xavier/Glorot uniform initialization.
    XavierUniform {
        /// The gain factor.
        gain: f32,
    },

    /// Xavier/Glorot normal initialization.
    XavierNormal {
        /// The gain factor.
        gain: f32,
    },

    /// Truncated normal initialization.
    TruncatedNormal {
        /// The mean of the distribution.
        mean: f32,
        /// The standard deviation of the distribution.
        stddev: f32,
    },
}

impl Default for InitializerConfig {
    fn default() -> Self {
        InitializerConfig::RandomUniform {
            min: -0.05,
            max: 0.05,
        }
    }
}

impl InitializerConfig {
    /// Creates a zeros initializer.
    pub fn zeros() -> Self {
        InitializerConfig::Zeros
    }

    /// Creates a ones initializer.
    pub fn ones() -> Self {
        InitializerConfig::Ones
    }

    /// Creates a constant initializer.
    pub fn constant(value: f32) -> Self {
        InitializerConfig::Constant { value }
    }

    /// Creates a uniform random initializer.
    pub fn uniform(min: f32, max: f32) -> Self {
        InitializerConfig::RandomUniform { min, max }
    }

    /// Creates a normal random initializer.
    pub fn normal(mean: f32, stddev: f32) -> Self {
        InitializerConfig::RandomNormal { mean, stddev }
    }

    /// Creates a Xavier uniform initializer.
    pub fn xavier_uniform(gain: f32) -> Self {
        InitializerConfig::XavierUniform { gain }
    }

    /// Creates a Xavier normal initializer.
    pub fn xavier_normal(gain: f32) -> Self {
        InitializerConfig::XavierNormal { gain }
    }

    /// Creates a truncated normal initializer.
    pub fn truncated_normal(mean: f32, stddev: f32) -> Self {
        InitializerConfig::TruncatedNormal { mean, stddev }
    }

    /// Validates the initializer configuration.
    pub fn validate(&self) -> Result<()> {
        match self {
            InitializerConfig::RandomUniform { min, max } => {
                if min >= max {
                    return Err(MonolithError::InvalidInitializer {
                        message: format!(
                            "RandomUniform min ({}) must be less than max ({})",
                            min, max
                        ),
                    });
                }
            }
            InitializerConfig::RandomNormal { stddev, .. } => {
                if *stddev <= 0.0 {
                    return Err(MonolithError::InvalidInitializer {
                        message: format!("RandomNormal stddev ({}) must be positive", stddev),
                    });
                }
            }
            InitializerConfig::XavierUniform { gain }
            | InitializerConfig::XavierNormal { gain } => {
                if *gain <= 0.0 {
                    return Err(MonolithError::InvalidInitializer {
                        message: format!("Xavier gain ({}) must be positive", gain),
                    });
                }
            }
            InitializerConfig::TruncatedNormal { stddev, .. } => {
                if *stddev <= 0.0 {
                    return Err(MonolithError::InvalidInitializer {
                        message: format!("TruncatedNormal stddev ({}) must be positive", stddev),
                    });
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Optimizer type configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent.
    Sgd {
        /// Momentum factor.
        momentum: f32,
    },

    /// Adam optimizer.
    Adam {
        /// Beta1 coefficient for moving average of gradients.
        beta1: f32,
        /// Beta2 coefficient for moving average of squared gradients.
        beta2: f32,
        /// Small constant for numerical stability.
        epsilon: f32,
    },

    /// Adagrad optimizer.
    Adagrad {
        /// Initial accumulator value.
        initial_accumulator_value: f32,
    },

    /// FTRL optimizer (Follow The Regularized Leader).
    Ftrl {
        /// Learning rate power.
        learning_rate_power: f32,
        /// L1 regularization strength.
        l1_regularization: f32,
        /// L2 regularization strength.
        l2_regularization: f32,
    },
}

impl Default for OptimizerType {
    fn default() -> Self {
        OptimizerType::Adam {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// A trait for hyperparameter containers.
///
/// This trait defines the interface for accessing and modifying
/// hyperparameters during training.
pub trait Params {
    /// Returns the learning rate.
    fn learning_rate(&self) -> f32;

    /// Sets the learning rate.
    fn set_learning_rate(&mut self, lr: f32);

    /// Returns the batch size.
    fn batch_size(&self) -> usize;

    /// Sets the batch size.
    fn set_batch_size(&mut self, batch_size: usize);

    /// Returns the number of training epochs.
    fn num_epochs(&self) -> usize;

    /// Sets the number of training epochs.
    fn set_num_epochs(&mut self, epochs: usize);

    /// Validates all parameters.
    fn validate(&self) -> Result<()>;
}

/// Default training parameters.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingParams {
    /// The learning rate.
    learning_rate: f32,

    /// The batch size.
    batch_size: usize,

    /// The number of training epochs.
    num_epochs: usize,

    /// Whether to shuffle data each epoch.
    shuffle: bool,

    /// Random seed for reproducibility.
    seed: Option<u64>,

    /// Gradient accumulation steps.
    gradient_accumulation_steps: usize,

    /// Warmup steps for learning rate.
    warmup_steps: usize,

    /// Maximum gradient norm for clipping.
    max_grad_norm: Option<f32>,
}

impl Default for TrainingParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            num_epochs: 1,
            shuffle: true,
            seed: None,
            gradient_accumulation_steps: 1,
            warmup_steps: 0,
            max_grad_norm: None,
        }
    }
}

impl TrainingParams {
    /// Creates new training parameters with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether to shuffle data.
    pub fn shuffle(&self) -> bool {
        self.shuffle
    }

    /// Sets whether to shuffle data.
    pub fn set_shuffle(&mut self, shuffle: bool) {
        self.shuffle = shuffle;
    }

    /// Returns the random seed.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// Sets the random seed.
    pub fn set_seed(&mut self, seed: Option<u64>) {
        self.seed = seed;
    }

    /// Returns the gradient accumulation steps.
    pub fn gradient_accumulation_steps(&self) -> usize {
        self.gradient_accumulation_steps
    }

    /// Sets the gradient accumulation steps.
    pub fn set_gradient_accumulation_steps(&mut self, steps: usize) {
        self.gradient_accumulation_steps = steps;
    }

    /// Returns the warmup steps.
    pub fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// Sets the warmup steps.
    pub fn set_warmup_steps(&mut self, steps: usize) {
        self.warmup_steps = steps;
    }

    /// Returns the maximum gradient norm.
    pub fn max_grad_norm(&self) -> Option<f32> {
        self.max_grad_norm
    }

    /// Sets the maximum gradient norm.
    pub fn set_max_grad_norm(&mut self, norm: Option<f32>) {
        self.max_grad_norm = norm;
    }
}

impl Params for TrainingParams {
    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn set_batch_size(&mut self, batch_size: usize) {
        self.batch_size = batch_size;
    }

    fn num_epochs(&self) -> usize {
        self.num_epochs
    }

    fn set_num_epochs(&mut self, epochs: usize) {
        self.num_epochs = epochs;
    }

    fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(MonolithError::ConfigError {
                message: "Learning rate must be positive".to_string(),
            });
        }

        if self.batch_size == 0 {
            return Err(MonolithError::ConfigError {
                message: "Batch size must be greater than 0".to_string(),
            });
        }

        if self.num_epochs == 0 {
            return Err(MonolithError::ConfigError {
                message: "Number of epochs must be greater than 0".to_string(),
            });
        }

        if self.gradient_accumulation_steps == 0 {
            return Err(MonolithError::ConfigError {
                message: "Gradient accumulation steps must be greater than 0".to_string(),
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_config_new() {
        let config = EmbeddingConfig::new(1, 64);
        assert_eq!(config.slot_id(), 1);
        assert_eq!(config.dim(), 64);
        assert_eq!(config.learning_rate(), 0.001);
        assert_eq!(config.l2_regularization(), 0.0);
        assert!(config.gradient_clipping().is_none());
    }

    #[test]
    fn test_embedding_config_builder() {
        let config = EmbeddingConfig::builder(1, 64)
            .learning_rate(0.01)
            .l2_regularization(0.001)
            .gradient_clipping(1.0)
            .initializer(InitializerConfig::zeros())
            .build()
            .unwrap();

        assert_eq!(config.learning_rate(), 0.01);
        assert_eq!(config.l2_regularization(), 0.001);
        assert_eq!(config.gradient_clipping(), Some(1.0));
        assert_eq!(config.initializer(), &InitializerConfig::Zeros);
    }

    #[test]
    fn test_embedding_config_validate() {
        // Valid config
        let config = EmbeddingConfig::new(1, 64);
        config
            .validate()
            .expect("default embedding config should pass validation");

        // Invalid: zero dimension
        let mut config = EmbeddingConfig::new(1, 0);
        config.learning_rate = 0.001;
        config
            .validate()
            .expect_err("embedding config with zero dimension should fail validation");
    }

    #[test]
    fn test_initializer_config() {
        let init = InitializerConfig::zeros();
        assert_eq!(init, InitializerConfig::Zeros);
        init.validate()
            .expect("zero initializer should pass validation");

        let init = InitializerConfig::uniform(-1.0, 1.0);
        init.validate()
            .expect("uniform initializer with ordered bounds should pass validation");

        let init = InitializerConfig::uniform(1.0, -1.0);
        init.validate()
            .expect_err("uniform initializer with inverted bounds should fail validation");

        let init = InitializerConfig::normal(0.0, 1.0);
        init.validate()
            .expect("normal initializer with positive stddev should pass validation");

        let init = InitializerConfig::normal(0.0, -1.0);
        init.validate()
            .expect_err("normal initializer with negative stddev should fail validation");
    }

    #[test]
    fn test_initializer_config_default() {
        let init = InitializerConfig::default();
        assert!(
            matches!(
                init,
                InitializerConfig::RandomUniform { min, max }
                    if (min + 0.05).abs() < f32::EPSILON
                        && (max - 0.05).abs() < f32::EPSILON
            ),
            "default initializer should be RandomUniform(-0.05, 0.05)"
        );
    }

    #[test]
    fn test_optimizer_type() {
        let opt = OptimizerType::default();
        assert!(
            matches!(
                opt,
                OptimizerType::Adam {
                    beta1,
                    beta2,
                    epsilon,
                } if (beta1 - 0.9).abs() < f32::EPSILON
                    && (beta2 - 0.999).abs() < f32::EPSILON
                    && (epsilon - 1e-8).abs() < f32::EPSILON
            ),
            "default optimizer should be Adam with expected hyperparameters"
        );

        let opt = OptimizerType::Sgd { momentum: 0.9 };
        assert!(
            matches!(opt, OptimizerType::Sgd { momentum } if (momentum - 0.9).abs() < f32::EPSILON),
            "constructed optimizer should be SGD with configured momentum"
        );
    }

    #[test]
    fn test_training_params_default() {
        let params = TrainingParams::default();
        assert_eq!(params.learning_rate(), 0.001);
        assert_eq!(params.batch_size(), 32);
        assert_eq!(params.num_epochs(), 1);
        assert!(params.shuffle());
        assert!(params.seed().is_none());
    }

    #[test]
    fn test_training_params_validate() {
        let params = TrainingParams::default();
        params
            .validate()
            .expect("default training params should pass validation");

        let mut params = TrainingParams::default();
        params.set_learning_rate(-0.001);
        params
            .validate()
            .expect_err("negative learning rate should fail training params validation");

        let mut params = TrainingParams::default();
        params.set_batch_size(0);
        params
            .validate()
            .expect_err("zero batch size should fail training params validation");

        let mut params = TrainingParams::default();
        params.set_num_epochs(0);
        params
            .validate()
            .expect_err("zero num_epochs should fail training params validation");
    }

    #[test]
    fn test_training_params_setters() {
        let mut params = TrainingParams::new();

        params.set_learning_rate(0.01);
        assert_eq!(params.learning_rate(), 0.01);

        params.set_batch_size(64);
        assert_eq!(params.batch_size(), 64);

        params.set_num_epochs(10);
        assert_eq!(params.num_epochs(), 10);

        params.set_shuffle(false);
        assert!(!params.shuffle());

        params.set_seed(Some(42));
        assert_eq!(params.seed(), Some(42));

        params.set_gradient_accumulation_steps(4);
        assert_eq!(params.gradient_accumulation_steps(), 4);

        params.set_warmup_steps(100);
        assert_eq!(params.warmup_steps(), 100);

        params.set_max_grad_norm(Some(1.0));
        assert_eq!(params.max_grad_norm(), Some(1.0));
    }
}
