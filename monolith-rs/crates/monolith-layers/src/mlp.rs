//! Multi-layer perceptron (MLP) implementation.
//!
//! This module provides the [`MLP`] struct, which is a stack of dense layers
//! with activation functions between them.

use crate::activation::{ReLU, Sigmoid, Tanh, GELU};
use crate::dense::Dense;
use crate::error::LayerError;
use crate::layer::Layer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Activation function types supported by MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid function
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Gaussian Error Linear Unit
    GELU,
    /// No activation (identity)
    None,
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::ReLU
    }
}

/// Configuration for building an MLP.
///
/// # Example
///
/// ```
/// use monolith_layers::mlp::{MLPConfig, ActivationType};
///
/// let config = MLPConfig::new(128)
///     .add_layer(64, ActivationType::ReLU)
///     .add_layer(32, ActivationType::ReLU)
///     .add_layer(10, ActivationType::None);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Layer configurations: (output_dim, activation)
    pub layers: Vec<(usize, ActivationType)>,
    /// Whether to use bias in dense layers
    pub use_bias: bool,
    /// Dropout rate (0.0 to disable)
    pub dropout_rate: f32,
}

impl MLPConfig {
    /// Creates a new MLP configuration with the specified input dimension.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - The dimension of the input features
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            layers: Vec::new(),
            use_bias: true,
            dropout_rate: 0.0,
        }
    }

    /// Adds a layer to the MLP configuration.
    ///
    /// # Arguments
    ///
    /// * `output_dim` - The output dimension of this layer
    /// * `activation` - The activation function to use after this layer
    pub fn add_layer(mut self, output_dim: usize, activation: ActivationType) -> Self {
        self.layers.push((output_dim, activation));
        self
    }

    /// Sets whether to use bias in dense layers.
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Sets the dropout rate.
    pub fn with_dropout(mut self, rate: f32) -> Self {
        self.dropout_rate = rate;
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), LayerError> {
        if self.input_dim == 0 {
            return Err(LayerError::ConfigError {
                message: "Input dimension must be positive".to_string(),
            });
        }
        if self.layers.is_empty() {
            return Err(LayerError::ConfigError {
                message: "MLP must have at least one layer".to_string(),
            });
        }
        for (i, (dim, _)) in self.layers.iter().enumerate() {
            if *dim == 0 {
                return Err(LayerError::ConfigError {
                    message: format!("Layer {} has zero output dimension", i),
                });
            }
        }
        if self.dropout_rate < 0.0 || self.dropout_rate >= 1.0 {
            return Err(LayerError::ConfigError {
                message: "Dropout rate must be in [0, 1)".to_string(),
            });
        }
        Ok(())
    }

    /// Builds the MLP from this configuration.
    pub fn build(self) -> Result<MLP, LayerError> {
        MLP::from_config(self)
    }
}

/// Internal enum to hold different activation layer types.
#[derive(Debug, Clone)]
enum ActivationLayer {
    ReLU(ReLU),
    Sigmoid(Sigmoid),
    Tanh(Tanh),
    GELU(GELU),
    None,
}

impl ActivationLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            Self::ReLU(a) => a.forward(input),
            Self::Sigmoid(a) => a.forward(input),
            Self::Tanh(a) => a.forward(input),
            Self::GELU(a) => a.forward(input),
            Self::None => Ok(input.clone()),
        }
    }

    fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            Self::ReLU(a) => a.forward_train(input),
            Self::Sigmoid(a) => a.forward_train(input),
            Self::Tanh(a) => a.forward_train(input),
            Self::GELU(a) => a.forward_train(input),
            Self::None => Ok(input.clone()),
        }
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        match self {
            Self::ReLU(a) => a.backward(grad),
            Self::Sigmoid(a) => a.backward(grad),
            Self::Tanh(a) => a.backward(grad),
            Self::GELU(a) => a.backward(grad),
            Self::None => Ok(grad.clone()),
        }
    }
}

/// A multi-layer perceptron (MLP) neural network.
///
/// An MLP consists of multiple dense (fully connected) layers with
/// activation functions between them.
///
/// # Example
///
/// ```
/// use monolith_layers::mlp::{MLP, MLPConfig, ActivationType};
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let mlp = MLPConfig::new(128)
///     .add_layer(64, ActivationType::ReLU)
///     .add_layer(10, ActivationType::None)
///     .build()
///     .unwrap();
///
/// let input = Tensor::zeros(&[32, 128]);
/// let output = mlp.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[32, 10]);
/// ```
#[derive(Debug, Clone)]
pub struct MLP {
    /// Dense layers
    dense_layers: Vec<Dense>,
    /// Activation layers (one per dense layer)
    activations: Vec<ActivationLayer>,
    /// Configuration used to build this MLP
    config: MLPConfig,
    /// Whether in training mode
    training: bool,
}

impl MLP {
    /// Creates an MLP from a configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The MLP configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid
    pub fn from_config(config: MLPConfig) -> Result<Self, LayerError> {
        config.validate()?;

        let mut dense_layers = Vec::new();
        let mut activations = Vec::new();

        let mut prev_dim = config.input_dim;
        for (output_dim, activation_type) in &config.layers {
            let dense = if config.use_bias {
                Dense::new(prev_dim, *output_dim)
            } else {
                Dense::new_no_bias(prev_dim, *output_dim)
            };
            dense_layers.push(dense);

            let activation = match activation_type {
                ActivationType::ReLU => ActivationLayer::ReLU(ReLU::new()),
                ActivationType::Sigmoid => ActivationLayer::Sigmoid(Sigmoid::new()),
                ActivationType::Tanh => ActivationLayer::Tanh(Tanh::new()),
                ActivationType::GELU => ActivationLayer::GELU(GELU::new()),
                ActivationType::None => ActivationLayer::None,
            };
            activations.push(activation);

            prev_dim = *output_dim;
        }

        Ok(Self {
            dense_layers,
            activations,
            config,
            training: true,
        })
    }

    /// Creates a simple MLP with uniform hidden layer sizes.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Input feature dimension
    /// * `hidden_dims` - Sizes of hidden layers
    /// * `output_dim` - Output dimension
    /// * `activation` - Activation function for hidden layers
    pub fn new(
        input_dim: usize,
        hidden_dims: &[usize],
        output_dim: usize,
        activation: ActivationType,
    ) -> Result<Self, LayerError> {
        let mut config = MLPConfig::new(input_dim);
        for &dim in hidden_dims {
            config = config.add_layer(dim, activation);
        }
        config = config.add_layer(output_dim, ActivationType::None);
        Self::from_config(config)
    }

    /// Returns the number of layers in the MLP.
    pub fn num_layers(&self) -> usize {
        self.dense_layers.len()
    }

    /// Returns a reference to the dense layers.
    pub fn dense_layers(&self) -> &[Dense] {
        &self.dense_layers
    }

    /// Returns the configuration used to build this MLP.
    pub fn config(&self) -> &MLPConfig {
        &self.config
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.config.input_dim
    }

    /// Returns the output dimension.
    pub fn output_dim(&self) -> usize {
        self.config.layers.last().map(|(d, _)| *d).unwrap_or(0)
    }

    /// Performs forward pass with training mode (caches activations).
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut x = input.clone();

        for (dense, activation) in self
            .dense_layers
            .iter_mut()
            .zip(self.activations.iter_mut())
        {
            x = dense.forward_train(&x)?;
            x = activation.forward_train(&x)?;
        }

        Ok(x)
    }
}

impl Layer for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut x = input.clone();

        for (dense, activation) in self.dense_layers.iter().zip(self.activations.iter()) {
            x = dense.forward(&x)?;
            x = activation.forward(&x)?;
        }

        Ok(x)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let mut g = grad.clone();

        // Backward pass through layers in reverse order
        for (dense, activation) in self
            .dense_layers
            .iter_mut()
            .zip(self.activations.iter_mut())
            .rev()
        {
            g = activation.backward(&g)?;
            g = dense.backward(&g)?;
        }

        Ok(g)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        self.dense_layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.dense_layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    fn name(&self) -> &str {
        "MLP"
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_config() {
        let config = MLPConfig::new(128)
            .add_layer(64, ActivationType::ReLU)
            .add_layer(32, ActivationType::ReLU)
            .add_layer(10, ActivationType::None);

        assert_eq!(config.input_dim, 128);
        assert_eq!(config.layers.len(), 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_mlp_config_invalid() {
        let config = MLPConfig::new(0);
        assert!(config.validate().is_err());

        let config = MLPConfig::new(128);
        assert!(config.validate().is_err()); // No layers

        let config = MLPConfig::new(128).add_layer(0, ActivationType::ReLU);
        assert!(config.validate().is_err()); // Zero dimension
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MLPConfig::new(10)
            .add_layer(5, ActivationType::ReLU)
            .add_layer(2, ActivationType::None)
            .build()
            .unwrap();

        let input = Tensor::ones(&[3, 10]); // batch of 3
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 2]);
    }

    #[test]
    fn test_mlp_new() {
        let mlp = MLP::new(128, &[64, 32], 10, ActivationType::ReLU).unwrap();
        assert_eq!(mlp.num_layers(), 3);
        assert_eq!(mlp.input_dim(), 128);
        assert_eq!(mlp.output_dim(), 10);
    }

    #[test]
    fn test_mlp_backward() {
        let mut mlp = MLPConfig::new(10)
            .add_layer(5, ActivationType::ReLU)
            .add_layer(2, ActivationType::None)
            .build()
            .unwrap();

        let input = Tensor::ones(&[3, 10]);
        let _output = mlp.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[3, 2]);
        let input_grad = mlp.backward(&grad).unwrap();
        assert_eq!(input_grad.shape(), &[3, 10]);
    }

    #[test]
    fn test_mlp_parameters() {
        let mlp = MLPConfig::new(10)
            .add_layer(5, ActivationType::ReLU)
            .add_layer(2, ActivationType::None)
            .build()
            .unwrap();

        // 2 layers with bias: 2 * 2 = 4 parameter tensors
        let params = mlp.parameters();
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_mlp_training_mode() {
        let mut mlp = MLPConfig::new(10)
            .add_layer(5, ActivationType::ReLU)
            .build()
            .unwrap();

        assert!(mlp.is_training());
        mlp.set_training(false);
        assert!(!mlp.is_training());
    }

    #[test]
    fn test_mlp_different_activations() {
        // Test each activation type
        for activation in [
            ActivationType::ReLU,
            ActivationType::Sigmoid,
            ActivationType::Tanh,
            ActivationType::GELU,
            ActivationType::None,
        ] {
            let mlp = MLPConfig::new(10).add_layer(5, activation).build().unwrap();

            let input = Tensor::rand(&[2, 10]);
            let output = mlp.forward(&input);
            assert!(output.is_ok());
        }
    }
}
