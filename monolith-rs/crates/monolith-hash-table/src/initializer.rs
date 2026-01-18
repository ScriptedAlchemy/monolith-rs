//! Embedding initializers for hash tables.
//!
//! This module provides various initialization strategies for embedding vectors.
//! Initializers are used to set the initial values of embeddings when new entries
//! are created in the hash table.
//!
//! # Overview
//!
//! The main components are:
//!
//! - [`Initializer`] - The core trait defining the interface for initializers
//! - [`ZerosInitializer`] - Initialize all values to zero
//! - [`OnesInitializer`] - Initialize all values to one
//! - [`ConstantInitializer`] - Initialize to a constant value
//! - [`RandomUniformInitializer`] - Initialize with uniform random values
//! - [`RandomNormalInitializer`] - Initialize with normal distribution values
//! - [`TruncatedNormalInitializer`] - Initialize with truncated normal distribution
//! - [`XavierUniformInitializer`] - Xavier/Glorot uniform initialization
//! - [`InitializerFactory`] - Factory for creating initializers from config
//!
//! # Example
//!
//! ```
//! use monolith_hash_table::initializer::{Initializer, RandomUniformInitializer};
//!
//! let initializer = RandomUniformInitializer::new(-0.05, 0.05);
//! let embedding = initializer.initialize(64);
//! assert_eq!(embedding.len(), 64);
//! ```

use monolith_core::params::InitializerConfig;
use rand::Rng;
use rand_distr::{Distribution, Normal};

/// A trait for embedding initializers.
///
/// Initializers generate initial values for embedding vectors when new entries
/// are created in the hash table. Different initialization strategies can
/// significantly impact model training convergence and performance.
///
/// # Thread Safety
///
/// All initializers must be `Send + Sync` to allow use in concurrent contexts.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, ZerosInitializer};
///
/// let initializer = ZerosInitializer;
/// let embedding = initializer.initialize(4);
/// assert_eq!(embedding, vec![0.0, 0.0, 0.0, 0.0]);
/// ```
pub trait Initializer: Send + Sync {
    /// Initialize an embedding vector with the given dimension.
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension of the embedding vector to create
    ///
    /// # Returns
    ///
    /// A vector of `f32` values with the specified dimension.
    fn initialize(&self, dim: usize) -> Vec<f32>;

    /// Returns the name of this initializer.
    ///
    /// This is useful for logging and debugging purposes.
    fn name(&self) -> &str;
}

/// Initializer that sets all values to zero.
///
/// This is the simplest initializer but may not be suitable for all use cases
/// as it can lead to symmetry issues during training.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, ZerosInitializer};
///
/// let initializer = ZerosInitializer;
/// let embedding = initializer.initialize(4);
/// assert_eq!(embedding, vec![0.0, 0.0, 0.0, 0.0]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct ZerosInitializer;

impl Initializer for ZerosInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        vec![0.0; dim]
    }

    fn name(&self) -> &str {
        "zeros"
    }
}

/// Initializer that sets all values to one.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, OnesInitializer};
///
/// let initializer = OnesInitializer;
/// let embedding = initializer.initialize(4);
/// assert_eq!(embedding, vec![1.0, 1.0, 1.0, 1.0]);
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct OnesInitializer;

impl Initializer for OnesInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        vec![1.0; dim]
    }

    fn name(&self) -> &str {
        "ones"
    }
}

/// Initializer that sets all values to a constant.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, ConstantInitializer};
///
/// let initializer = ConstantInitializer::new(0.5);
/// let embedding = initializer.initialize(4);
/// assert_eq!(embedding, vec![0.5, 0.5, 0.5, 0.5]);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ConstantInitializer {
    /// The constant value to use for initialization.
    value: f32,
}

impl ConstantInitializer {
    /// Creates a new constant initializer with the given value.
    ///
    /// # Arguments
    ///
    /// * `value` - The constant value to use for all elements
    pub fn new(value: f32) -> Self {
        Self { value }
    }

    /// Returns the constant value.
    pub fn value(&self) -> f32 {
        self.value
    }
}

impl Default for ConstantInitializer {
    fn default() -> Self {
        Self { value: 0.0 }
    }
}

impl Initializer for ConstantInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        vec![self.value; dim]
    }

    fn name(&self) -> &str {
        "constant"
    }
}

/// Initializer that samples from a uniform distribution.
///
/// Values are sampled uniformly from the range `[min_val, max_val)`.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, RandomUniformInitializer};
///
/// let initializer = RandomUniformInitializer::new(-0.05, 0.05);
/// let embedding = initializer.initialize(64);
/// assert_eq!(embedding.len(), 64);
/// for &val in &embedding {
///     assert!(val >= -0.05 && val < 0.05);
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RandomUniformInitializer {
    /// The minimum value (inclusive).
    min_val: f32,
    /// The maximum value (exclusive).
    max_val: f32,
}

impl RandomUniformInitializer {
    /// Creates a new uniform random initializer.
    ///
    /// # Arguments
    ///
    /// * `min_val` - The minimum value (inclusive)
    /// * `max_val` - The maximum value (exclusive)
    ///
    /// # Panics
    ///
    /// Panics if `min_val >= max_val`.
    pub fn new(min_val: f32, max_val: f32) -> Self {
        assert!(
            min_val < max_val,
            "min_val ({}) must be less than max_val ({})",
            min_val,
            max_val
        );
        Self { min_val, max_val }
    }

    /// Returns the minimum value.
    pub fn min_val(&self) -> f32 {
        self.min_val
    }

    /// Returns the maximum value.
    pub fn max_val(&self) -> f32 {
        self.max_val
    }
}

impl Default for RandomUniformInitializer {
    fn default() -> Self {
        Self {
            min_val: -0.05,
            max_val: 0.05,
        }
    }
}

impl Initializer for RandomUniformInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        (0..dim)
            .map(|_| rng.gen_range(self.min_val..self.max_val))
            .collect()
    }

    fn name(&self) -> &str {
        "random_uniform"
    }
}

/// Initializer that samples from a normal (Gaussian) distribution.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, RandomNormalInitializer};
///
/// let initializer = RandomNormalInitializer::new(0.0, 0.01);
/// let embedding = initializer.initialize(64);
/// assert_eq!(embedding.len(), 64);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct RandomNormalInitializer {
    /// The mean of the distribution.
    mean: f32,
    /// The standard deviation of the distribution.
    stddev: f32,
}

impl RandomNormalInitializer {
    /// Creates a new normal random initializer.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean of the distribution
    /// * `stddev` - The standard deviation of the distribution
    ///
    /// # Panics
    ///
    /// Panics if `stddev <= 0`.
    pub fn new(mean: f32, stddev: f32) -> Self {
        assert!(stddev > 0.0, "stddev ({}) must be positive", stddev);
        Self { mean, stddev }
    }

    /// Returns the mean.
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Returns the standard deviation.
    pub fn stddev(&self) -> f32 {
        self.stddev
    }
}

impl Default for RandomNormalInitializer {
    fn default() -> Self {
        Self {
            mean: 0.0,
            stddev: 0.01,
        }
    }
}

impl Initializer for RandomNormalInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(self.mean as f64, self.stddev as f64)
            .expect("Failed to create normal distribution");
        (0..dim).map(|_| normal.sample(&mut rng) as f32).collect()
    }

    fn name(&self) -> &str {
        "random_normal"
    }
}

/// Initializer that samples from a truncated normal distribution.
///
/// Values are sampled from a normal distribution but values more than
/// 2 standard deviations from the mean are discarded and resampled.
/// This helps avoid extreme outliers in the initial weights.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, TruncatedNormalInitializer};
///
/// let initializer = TruncatedNormalInitializer::new(0.0, 0.01);
/// let embedding = initializer.initialize(64);
/// assert_eq!(embedding.len(), 64);
/// for &val in &embedding {
///     assert!(val >= -0.02 && val <= 0.02); // Within 2 stddev
/// }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TruncatedNormalInitializer {
    /// The mean of the distribution.
    mean: f32,
    /// The standard deviation of the distribution.
    stddev: f32,
}

impl TruncatedNormalInitializer {
    /// Creates a new truncated normal initializer.
    ///
    /// Values more than 2 standard deviations from the mean are resampled.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean of the distribution
    /// * `stddev` - The standard deviation of the distribution
    ///
    /// # Panics
    ///
    /// Panics if `stddev <= 0`.
    pub fn new(mean: f32, stddev: f32) -> Self {
        assert!(stddev > 0.0, "stddev ({}) must be positive", stddev);
        Self { mean, stddev }
    }

    /// Returns the mean.
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Returns the standard deviation.
    pub fn stddev(&self) -> f32 {
        self.stddev
    }
}

impl Default for TruncatedNormalInitializer {
    fn default() -> Self {
        Self {
            mean: 0.0,
            stddev: 0.01,
        }
    }
}

impl Initializer for TruncatedNormalInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(self.mean as f64, self.stddev as f64)
            .expect("Failed to create normal distribution");

        let lower = self.mean - 2.0 * self.stddev;
        let upper = self.mean + 2.0 * self.stddev;

        (0..dim)
            .map(|_| loop {
                let val = normal.sample(&mut rng) as f32;
                if val >= lower && val <= upper {
                    return val;
                }
            })
            .collect()
    }

    fn name(&self) -> &str {
        "truncated_normal"
    }
}

/// Xavier/Glorot uniform initializer.
///
/// This initializer is designed to keep the scale of gradients roughly the same
/// in all layers. It draws samples from a uniform distribution within
/// `[-limit, limit]` where `limit = sqrt(6 / (fan_in + fan_out))`.
///
/// For embeddings, we use `fan_in = 1` (single input) and `fan_out = dim`.
///
/// # Reference
///
/// Glorot, X. & Bengio, Y. (2010). Understanding the difficulty of training
/// deep feedforward neural networks. AISTATS.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, XavierUniformInitializer};
///
/// let initializer = XavierUniformInitializer::new(1.0);
/// let embedding = initializer.initialize(64);
/// assert_eq!(embedding.len(), 64);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct XavierUniformInitializer {
    /// The gain factor applied to the limit.
    gain: f32,
}

impl XavierUniformInitializer {
    /// Creates a new Xavier uniform initializer.
    ///
    /// # Arguments
    ///
    /// * `gain` - The gain factor (typically 1.0, or sqrt(2) for ReLU)
    ///
    /// # Panics
    ///
    /// Panics if `gain <= 0`.
    pub fn new(gain: f32) -> Self {
        assert!(gain > 0.0, "gain ({}) must be positive", gain);
        Self { gain }
    }

    /// Returns the gain factor.
    pub fn gain(&self) -> f32 {
        self.gain
    }
}

impl Default for XavierUniformInitializer {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl Initializer for XavierUniformInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        // For embeddings: fan_in = 1, fan_out = dim
        let fan_in = 1.0_f32;
        let fan_out = dim as f32;
        let limit = self.gain * (6.0 / (fan_in + fan_out)).sqrt();

        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen_range(-limit..limit)).collect()
    }

    fn name(&self) -> &str {
        "xavier_uniform"
    }
}

/// Xavier/Glorot normal initializer.
///
/// This initializer draws samples from a truncated normal distribution centered
/// on 0 with `stddev = gain * sqrt(2 / (fan_in + fan_out))`.
///
/// For embeddings, we use `fan_in = 1` (single input) and `fan_out = dim`.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{Initializer, XavierNormalInitializer};
///
/// let initializer = XavierNormalInitializer::new(1.0);
/// let embedding = initializer.initialize(64);
/// assert_eq!(embedding.len(), 64);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct XavierNormalInitializer {
    /// The gain factor applied to the standard deviation.
    gain: f32,
}

impl XavierNormalInitializer {
    /// Creates a new Xavier normal initializer.
    ///
    /// # Arguments
    ///
    /// * `gain` - The gain factor (typically 1.0, or sqrt(2) for ReLU)
    ///
    /// # Panics
    ///
    /// Panics if `gain <= 0`.
    pub fn new(gain: f32) -> Self {
        assert!(gain > 0.0, "gain ({}) must be positive", gain);
        Self { gain }
    }

    /// Returns the gain factor.
    pub fn gain(&self) -> f32 {
        self.gain
    }
}

impl Default for XavierNormalInitializer {
    fn default() -> Self {
        Self { gain: 1.0 }
    }
}

impl Initializer for XavierNormalInitializer {
    fn initialize(&self, dim: usize) -> Vec<f32> {
        // For embeddings: fan_in = 1, fan_out = dim
        let fan_in = 1.0_f64;
        let fan_out = dim as f64;
        let stddev = (self.gain as f64) * (2.0 / (fan_in + fan_out)).sqrt();

        let mut rng = rand::thread_rng();
        let normal = Normal::new(0.0, stddev).expect("Failed to create normal distribution");
        (0..dim).map(|_| normal.sample(&mut rng) as f32).collect()
    }

    fn name(&self) -> &str {
        "xavier_normal"
    }
}

/// Factory for creating initializers from configuration.
///
/// This factory creates boxed initializer instances based on the provided
/// configuration. It supports all standard initializer types.
///
/// # Example
///
/// ```
/// use monolith_hash_table::initializer::{InitializerFactory, Initializer};
/// use monolith_core::params::InitializerConfig;
///
/// let config = InitializerConfig::zeros();
/// let initializer = InitializerFactory::create_initializer(&config);
/// assert_eq!(initializer.name(), "zeros");
/// ```
pub struct InitializerFactory;

impl InitializerFactory {
    /// Creates an initializer from the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The initializer configuration
    ///
    /// # Returns
    ///
    /// A boxed initializer implementing the specified strategy.
    pub fn create_initializer(config: &InitializerConfig) -> Box<dyn Initializer> {
        match config {
            InitializerConfig::Zeros => Box::new(ZerosInitializer),
            InitializerConfig::Ones => Box::new(OnesInitializer),
            InitializerConfig::Constant { value } => Box::new(ConstantInitializer::new(*value)),
            InitializerConfig::RandomUniform { min, max } => {
                Box::new(RandomUniformInitializer::new(*min, *max))
            }
            InitializerConfig::RandomNormal { mean, stddev } => {
                Box::new(RandomNormalInitializer::new(*mean, *stddev))
            }
            InitializerConfig::TruncatedNormal { mean, stddev } => {
                Box::new(TruncatedNormalInitializer::new(*mean, *stddev))
            }
            InitializerConfig::XavierUniform { gain } => {
                Box::new(XavierUniformInitializer::new(*gain))
            }
            InitializerConfig::XavierNormal { gain } => {
                Box::new(XavierNormalInitializer::new(*gain))
            }
        }
    }

    /// Creates a default initializer (random uniform with range [-0.05, 0.05]).
    pub fn create_default() -> Box<dyn Initializer> {
        Box::new(RandomUniformInitializer::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zeros_initializer() {
        let initializer = ZerosInitializer;
        let embedding = initializer.initialize(4);
        assert_eq!(embedding, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(initializer.name(), "zeros");
    }

    #[test]
    fn test_ones_initializer() {
        let initializer = OnesInitializer;
        let embedding = initializer.initialize(4);
        assert_eq!(embedding, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(initializer.name(), "ones");
    }

    #[test]
    fn test_constant_initializer() {
        let initializer = ConstantInitializer::new(0.5);
        let embedding = initializer.initialize(4);
        assert_eq!(embedding, vec![0.5, 0.5, 0.5, 0.5]);
        assert_eq!(initializer.name(), "constant");
        assert_eq!(initializer.value(), 0.5);
    }

    #[test]
    fn test_constant_initializer_default() {
        let initializer = ConstantInitializer::default();
        let embedding = initializer.initialize(4);
        assert_eq!(embedding, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_random_uniform_initializer() {
        let initializer = RandomUniformInitializer::new(-0.1, 0.1);
        let embedding = initializer.initialize(100);
        assert_eq!(embedding.len(), 100);

        for &val in &embedding {
            assert!(
                val >= -0.1 && val < 0.1,
                "Value {} out of range [-0.1, 0.1)",
                val
            );
        }

        assert_eq!(initializer.name(), "random_uniform");
        assert_eq!(initializer.min_val(), -0.1);
        assert_eq!(initializer.max_val(), 0.1);
    }

    #[test]
    fn test_random_uniform_initializer_default() {
        let initializer = RandomUniformInitializer::default();
        let embedding = initializer.initialize(100);

        for &val in &embedding {
            assert!(
                val >= -0.05 && val < 0.05,
                "Value {} out of range [-0.05, 0.05)",
                val
            );
        }
    }

    #[test]
    #[should_panic(expected = "min_val")]
    fn test_random_uniform_initializer_invalid_range() {
        RandomUniformInitializer::new(0.1, -0.1);
    }

    #[test]
    fn test_random_normal_initializer() {
        let initializer = RandomNormalInitializer::new(0.0, 0.1);
        let embedding = initializer.initialize(1000);
        assert_eq!(embedding.len(), 1000);

        // Check that mean is approximately 0
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        assert!(mean.abs() < 0.05, "Mean {} too far from 0", mean);

        assert_eq!(initializer.name(), "random_normal");
        assert_eq!(initializer.mean(), 0.0);
        assert_eq!(initializer.stddev(), 0.1);
    }

    #[test]
    fn test_random_normal_initializer_default() {
        let initializer = RandomNormalInitializer::default();
        assert_eq!(initializer.mean(), 0.0);
        assert_eq!(initializer.stddev(), 0.01);
    }

    #[test]
    #[should_panic(expected = "stddev")]
    fn test_random_normal_initializer_invalid_stddev() {
        RandomNormalInitializer::new(0.0, -0.1);
    }

    #[test]
    fn test_truncated_normal_initializer() {
        let initializer = TruncatedNormalInitializer::new(0.0, 0.1);
        let embedding = initializer.initialize(1000);
        assert_eq!(embedding.len(), 1000);

        // Check that all values are within 2 stddev
        let lower = -0.2;
        let upper = 0.2;
        for &val in &embedding {
            assert!(
                val >= lower && val <= upper,
                "Value {} out of truncated range [{}, {}]",
                val,
                lower,
                upper
            );
        }

        assert_eq!(initializer.name(), "truncated_normal");
        assert_eq!(initializer.mean(), 0.0);
        assert_eq!(initializer.stddev(), 0.1);
    }

    #[test]
    fn test_truncated_normal_initializer_default() {
        let initializer = TruncatedNormalInitializer::default();
        assert_eq!(initializer.mean(), 0.0);
        assert_eq!(initializer.stddev(), 0.01);
    }

    #[test]
    #[should_panic(expected = "stddev")]
    fn test_truncated_normal_initializer_invalid_stddev() {
        TruncatedNormalInitializer::new(0.0, 0.0);
    }

    #[test]
    fn test_xavier_uniform_initializer() {
        let initializer = XavierUniformInitializer::new(1.0);
        let dim = 64;
        let embedding = initializer.initialize(dim);
        assert_eq!(embedding.len(), dim);

        // Check that values are within expected range
        // limit = sqrt(6 / (1 + 64)) = sqrt(6/65) ~ 0.304
        let limit = (6.0_f32 / (1.0 + dim as f32)).sqrt();
        for &val in &embedding {
            assert!(
                val >= -limit && val < limit,
                "Value {} out of range [{}, {})",
                val,
                -limit,
                limit
            );
        }

        assert_eq!(initializer.name(), "xavier_uniform");
        assert_eq!(initializer.gain(), 1.0);
    }

    #[test]
    fn test_xavier_uniform_initializer_default() {
        let initializer = XavierUniformInitializer::default();
        assert_eq!(initializer.gain(), 1.0);
    }

    #[test]
    #[should_panic(expected = "gain")]
    fn test_xavier_uniform_initializer_invalid_gain() {
        XavierUniformInitializer::new(0.0);
    }

    #[test]
    fn test_xavier_normal_initializer() {
        let initializer = XavierNormalInitializer::new(1.0);
        let embedding = initializer.initialize(64);
        assert_eq!(embedding.len(), 64);

        assert_eq!(initializer.name(), "xavier_normal");
        assert_eq!(initializer.gain(), 1.0);
    }

    #[test]
    fn test_xavier_normal_initializer_default() {
        let initializer = XavierNormalInitializer::default();
        assert_eq!(initializer.gain(), 1.0);
    }

    #[test]
    #[should_panic(expected = "gain")]
    fn test_xavier_normal_initializer_invalid_gain() {
        XavierNormalInitializer::new(-1.0);
    }

    #[test]
    fn test_initializer_factory_zeros() {
        let config = InitializerConfig::Zeros;
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "zeros");
        assert_eq!(initializer.initialize(4), vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_initializer_factory_ones() {
        let config = InitializerConfig::Ones;
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "ones");
        assert_eq!(initializer.initialize(4), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_initializer_factory_constant() {
        let config = InitializerConfig::Constant { value: 0.5 };
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "constant");
        assert_eq!(initializer.initialize(4), vec![0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_initializer_factory_random_uniform() {
        let config = InitializerConfig::RandomUniform {
            min: -0.1,
            max: 0.1,
        };
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "random_uniform");

        let embedding = initializer.initialize(100);
        for &val in &embedding {
            assert!(val >= -0.1 && val < 0.1);
        }
    }

    #[test]
    fn test_initializer_factory_random_normal() {
        let config = InitializerConfig::RandomNormal {
            mean: 0.0,
            stddev: 0.1,
        };
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "random_normal");
    }

    #[test]
    fn test_initializer_factory_truncated_normal() {
        let config = InitializerConfig::TruncatedNormal {
            mean: 0.0,
            stddev: 0.1,
        };
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "truncated_normal");

        let embedding = initializer.initialize(100);
        for &val in &embedding {
            assert!(val >= -0.2 && val <= 0.2);
        }
    }

    #[test]
    fn test_initializer_factory_xavier_uniform() {
        let config = InitializerConfig::XavierUniform { gain: 1.0 };
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "xavier_uniform");
    }

    #[test]
    fn test_initializer_factory_xavier_normal() {
        let config = InitializerConfig::XavierNormal { gain: 1.0 };
        let initializer = InitializerFactory::create_initializer(&config);
        assert_eq!(initializer.name(), "xavier_normal");
    }

    #[test]
    fn test_initializer_factory_default() {
        let initializer = InitializerFactory::create_default();
        assert_eq!(initializer.name(), "random_uniform");
    }

    #[test]
    fn test_initializers_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}

        assert_send_sync::<ZerosInitializer>();
        assert_send_sync::<OnesInitializer>();
        assert_send_sync::<ConstantInitializer>();
        assert_send_sync::<RandomUniformInitializer>();
        assert_send_sync::<RandomNormalInitializer>();
        assert_send_sync::<TruncatedNormalInitializer>();
        assert_send_sync::<XavierUniformInitializer>();
        assert_send_sync::<XavierNormalInitializer>();
    }

    #[test]
    fn test_empty_dimension() {
        let initializer = ZerosInitializer;
        let embedding = initializer.initialize(0);
        assert!(embedding.is_empty());
    }

    #[test]
    fn test_large_dimension() {
        let initializer = RandomUniformInitializer::default();
        let embedding = initializer.initialize(10000);
        assert_eq!(embedding.len(), 10000);
    }
}
