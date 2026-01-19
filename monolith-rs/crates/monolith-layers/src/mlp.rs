//! Multi-layer perceptron (MLP) implementation.
//!
//! This module provides the [`MLP`] struct, which is a stack of dense layers
//! with activation functions between them.

use crate::activation_layer::ActivationLayer;
use crate::constraint::Constraint;
use crate::initializer::Initializer;
use crate::normalization::BatchNorm;
use crate::dense::Dense;
use crate::error::LayerError;
use crate::layer::Layer;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Activation function types supported by MLP.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU {
        max_value: Option<f32>,
        negative_slope: f32,
        threshold: f32,
    },
    /// Sigmoid function
    Sigmoid,
    /// Sigmoid2: 2 * sigmoid(x)
    Sigmoid2,
    /// Hyperbolic tangent
    Tanh,
    /// Gaussian Error Linear Unit
    GELU,
    /// Scaled Exponential Linear Unit
    SELU,
    /// Softplus activation
    Softplus,
    /// Softsign activation
    Softsign,
    /// Swish activation
    Swish,
    /// Mish activation
    Mish,
    /// Hard sigmoid activation
    HardSigmoid,
    /// Leaky ReLU activation
    LeakyReLU {
        alpha: f32,
    },
    /// ELU activation
    ELU {
        alpha: f32,
    },
    /// PReLU activation
    PReLU {
        alpha: f32,
        #[serde(default)]
        initializer: Option<Initializer>,
        #[serde(default)]
        shared_axes: Option<Vec<isize>>,
        #[serde(default)]
        regularizer: Option<crate::regularizer::Regularizer>,
        #[serde(default)]
        constraint: Option<crate::constraint::Constraint>,
    },
    /// Thresholded ReLU activation
    ThresholdedReLU {
        theta: f32,
    },
    /// Softmax activation (axis defaults to -1 if negative)
    Softmax {
        axis: isize,
    },
    /// Linear activation (identity)
    Linear,
    /// Exponential activation
    Exponential,
    /// No activation (identity)
    None,
}

impl ActivationType {
    pub fn relu() -> Self {
        Self::ReLU {
            max_value: None,
            negative_slope: 0.0,
            threshold: 0.0,
        }
    }

    pub fn relu_with(max_value: Option<f32>, negative_slope: f32, threshold: f32) -> Self {
        Self::ReLU {
            max_value,
            negative_slope,
            threshold,
        }
    }

    pub fn sigmoid() -> Self {
        Self::Sigmoid
    }

    pub fn sigmoid2() -> Self {
        Self::Sigmoid2
    }

    pub fn tanh() -> Self {
        Self::Tanh
    }

    pub fn gelu() -> Self {
        Self::GELU
    }

    pub fn none() -> Self {
        Self::None
    }
}

impl Default for ActivationType {
    fn default() -> Self {
        Self::relu()
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
///     .add_layer(64, ActivationType::relu())
///     .add_layer(32, ActivationType::relu())
///     .add_layer(10, ActivationType::None);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Layer configurations: (output_dim, activation)
    pub layers: Vec<(usize, ActivationType)>,
    /// Optional per-layer initializers (defaults to GlorotUniform)
    pub initializers: Vec<Initializer>,
    /// Whether to use bias in dense layers
    pub use_bias: bool,
    /// Kernel regularizer for dense layers
    pub kernel_regularizer: Regularizer,
    /// Bias regularizer for dense layers
    pub bias_regularizer: Regularizer,
    /// Kernel constraint for dense layers
    pub kernel_constraint: Constraint,
    /// Bias constraint for dense layers
    pub bias_constraint: Constraint,
    /// Whether to enable kernel norm (weight normalization)
    pub use_weight_norm: bool,
    /// Whether kernel norm scale is trainable
    pub use_learnable_weight_norm: bool,
    /// Whether to enable batch normalization
    pub enable_batch_normalization: bool,
    /// Batch normalization momentum
    pub batch_normalization_momentum: f32,
    /// Whether to use batch renorm
    pub batch_normalization_renorm: bool,
    /// Batch renorm clipping (rmin, rmax, dmax)
    pub batch_normalization_renorm_clipping: Option<(f32, f32, f32)>,
    /// Batch renorm momentum
    pub batch_normalization_renorm_momentum: f32,
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
            initializers: Vec::new(),
            use_bias: true,
            kernel_regularizer: Regularizer::None,
            bias_regularizer: Regularizer::None,
            kernel_constraint: Constraint::None,
            bias_constraint: Constraint::None,
            use_weight_norm: true,
            use_learnable_weight_norm: true,
            enable_batch_normalization: false,
            batch_normalization_momentum: 0.99,
            batch_normalization_renorm: false,
            batch_normalization_renorm_clipping: None,
            batch_normalization_renorm_momentum: 0.99,
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

    /// Adds a layer with a custom initializer.
    pub fn add_layer_with_initializer(
        mut self,
        output_dim: usize,
        activation: ActivationType,
        initializer: Initializer,
    ) -> Self {
        self.layers.push((output_dim, activation));
        self.initializers.push(initializer);
        self
    }

    /// Sets whether to use bias in dense layers.
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Sets kernel and bias regularizers.
    pub fn with_regularizers(
        mut self,
        kernel_regularizer: Regularizer,
        bias_regularizer: Regularizer,
    ) -> Self {
        self.kernel_regularizer = kernel_regularizer;
        self.bias_regularizer = bias_regularizer;
        self
    }

    /// Sets kernel and bias constraints.
    pub fn with_constraints(mut self, kernel_constraint: Constraint, bias_constraint: Constraint) -> Self {
        self.kernel_constraint = kernel_constraint;
        self.bias_constraint = bias_constraint;
        self
    }

    /// Enables kernel norm (weight normalization) in dense layers.
    pub fn with_weight_norm(mut self, use_weight_norm: bool) -> Self {
        self.use_weight_norm = use_weight_norm;
        self
    }

    /// Sets whether kernel norm scale is trainable.
    pub fn with_learnable_weight_norm(mut self, use_learnable: bool) -> Self {
        self.use_learnable_weight_norm = use_learnable;
        self
    }

    /// Enables batch normalization with optional renorm settings.
    pub fn with_batch_normalization(
        mut self,
        enable: bool,
        momentum: f32,
        renorm: bool,
        renorm_clipping: Option<(f32, f32, f32)>,
        renorm_momentum: f32,
    ) -> Self {
        self.enable_batch_normalization = enable;
        self.batch_normalization_momentum = momentum;
        self.batch_normalization_renorm = renorm;
        self.batch_normalization_renorm_clipping = renorm_clipping;
        self.batch_normalization_renorm_momentum = renorm_momentum;
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
        if self.batch_normalization_momentum < 0.0 || self.batch_normalization_momentum > 1.0 {
            return Err(LayerError::ConfigError {
                message: "BatchNorm momentum must be in [0, 1]".to_string(),
            });
        }
        if !self.initializers.is_empty() && self.initializers.len() != self.layers.len() {
            return Err(LayerError::ConfigError {
                message: "Initializers length must match layers length".to_string(),
            });
        }
        Ok(())
    }

    /// Builds the MLP from this configuration.
    pub fn build(self) -> Result<MLP, LayerError> {
        MLP::from_config(self)
    }

    /// Creates an MLPConfig from output dimensions with optional activations/initializers.
    ///
    /// If `activations` is None, defaults to ReLU for hidden layers and None for the last layer.
    /// If `activations` has length 1, uses it for hidden layers and None for the last layer.
    pub fn from_dims(
        input_dim: usize,
        output_dims: &[usize],
        activations: Option<Vec<ActivationType>>,
        initializers: Option<Vec<Initializer>>,
    ) -> Result<Self, LayerError> {
        if output_dims.is_empty() {
            return Err(LayerError::ConfigError {
                message: "output_dims must be non-empty".to_string(),
            });
        }

        let mut config = MLPConfig::new(input_dim);

        let acts = if let Some(acts) = activations {
            if acts.len() == 1 {
                let mut expanded = vec![acts[0].clone(); output_dims.len()];
                expanded[output_dims.len() - 1] = ActivationType::None;
                expanded
            } else if acts.len() == output_dims.len() {
                acts
            } else {
                return Err(LayerError::ConfigError {
                    message: "activations length must be 1 or match output_dims length"
                        .to_string(),
                });
            }
        } else {
            let mut expanded = vec![ActivationType::relu(); output_dims.len()];
            expanded[output_dims.len() - 1] = ActivationType::None;
            expanded
        };

        let inits = if let Some(inits) = initializers {
            if inits.len() == 1 {
                vec![inits[0]; output_dims.len()]
            } else if inits.len() == output_dims.len() {
                inits
            } else {
                return Err(LayerError::ConfigError {
                    message: "initializers length must be 1 or match output_dims length"
                        .to_string(),
                });
            }
        } else {
            vec![Initializer::GlorotUniform; output_dims.len()]
        };

        for ((&dim, act), init) in output_dims.iter().zip(acts).zip(inits) {
            config = config.add_layer_with_initializer(dim, act, init);
        }

        Ok(config)
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
///     .add_layer(64, ActivationType::relu())
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
    /// Optional batch norm on input
    input_batch_norm: Option<BatchNorm>,
    /// Optional batch norm after each dense layer (except last when disabled)
    batch_norms: Vec<Option<BatchNorm>>,
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
        let mut batch_norms = Vec::new();

        let input_batch_norm = if config.enable_batch_normalization {
            let mut bn = BatchNorm::with_momentum(
                config.input_dim,
                config.batch_normalization_momentum,
                1e-5,
            );
            bn = bn.with_renorm(
                config.batch_normalization_renorm,
                config.batch_normalization_renorm_clipping,
                config.batch_normalization_renorm_momentum,
            );
            Some(bn)
        } else {
            None
        };

        let mut prev_dim = config.input_dim;
        for (idx, (output_dim, activation_type)) in config.layers.iter().enumerate() {
            let is_final_layer = idx == config.layers.len() - 1;
            let initializer = if config.initializers.is_empty() {
                Initializer::GlorotUniform
            } else {
                config.initializers[idx]
            };
            let dense = if config.use_bias {
                Dense::new_with_initializer(
                    prev_dim,
                    *output_dim,
                    initializer,
                    Initializer::Zeros,
                    true,
                )
            } else {
                Dense::new_with_initializer(
                    prev_dim,
                    *output_dim,
                    initializer,
                    Initializer::Zeros,
                    false,
                )
            };
            let dense = if config.use_weight_norm {
                dense.with_kernel_norm(config.use_learnable_weight_norm)
            } else {
                dense
            }
            .with_kernel_regularizer(config.kernel_regularizer.clone())
            .with_bias_regularizer(config.bias_regularizer.clone())
            .with_kernel_constraint(config.kernel_constraint.clone())
            .with_bias_constraint(config.bias_constraint.clone());
            dense_layers.push(dense);

            let bn = if config.enable_batch_normalization && !is_final_layer {
                let mut bn = BatchNorm::with_momentum(
                    *output_dim,
                    config.batch_normalization_momentum,
                    1e-5,
                );
                bn = bn.with_renorm(
                    config.batch_normalization_renorm,
                    config.batch_normalization_renorm_clipping,
                    config.batch_normalization_renorm_momentum,
                );
                Some(bn)
            } else {
                None
            };
            batch_norms.push(bn);

            let activation = ActivationLayer::from_activation_type(activation_type.clone());
            activations.push(activation);

            prev_dim = *output_dim;
        }

        Ok(Self {
            dense_layers,
            input_batch_norm,
            batch_norms,
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
            config = config.add_layer(dim, activation.clone());
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

    /// Returns a mutable reference to the dense layers.
    pub fn dense_layers_mut(&mut self) -> &mut [Dense] {
        &mut self.dense_layers
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

        if let Some(bn) = self.input_batch_norm.as_mut() {
            x = bn.forward_train(&x)?;
        }

        for idx in 0..self.dense_layers.len() {
            x = self.dense_layers[idx].forward_train(&x)?;
            if let Some(bn) = self.batch_norms[idx].as_mut() {
                x = bn.forward_train(&x)?;
            }
            x = self.activations[idx].forward_train(&x)?;
        }

        Ok(x)
    }
}

impl Layer for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut x = input.clone();

        if let Some(bn) = &self.input_batch_norm {
            x = bn.forward(&x)?;
        }

        for idx in 0..self.dense_layers.len() {
            x = self.dense_layers[idx].forward(&x)?;
            if let Some(bn) = &self.batch_norms[idx] {
                x = bn.forward(&x)?;
            }
            x = self.activations[idx].forward(&x)?;
        }

        Ok(x)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let mut g = grad.clone();

        // Backward pass through layers in reverse order
        for idx in (0..self.dense_layers.len()).rev() {
            g = self.activations[idx].backward(&g)?;
            if let Some(bn) = self.batch_norms[idx].as_mut() {
                g = bn.backward(&g)?;
            }
            g = self.dense_layers[idx].backward(&g)?;
        }

        if let Some(bn) = self.input_batch_norm.as_mut() {
            g = bn.backward(&g)?;
        }

        Ok(g)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params: Vec<&Tensor> = self
            .dense_layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect();
        if let Some(bn) = &self.input_batch_norm {
            params.extend(bn.parameters());
        }
        for bn in &self.batch_norms {
            if let Some(bn) = bn {
                params.extend(bn.parameters());
            }
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params: Vec<&mut Tensor> = self
            .dense_layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect();
        if let Some(bn) = self.input_batch_norm.as_mut() {
            params.extend(bn.parameters_mut());
        }
        for bn in &mut self.batch_norms {
            if let Some(bn) = bn {
                params.extend(bn.parameters_mut());
            }
        }
        params
    }

    fn name(&self) -> &str {
        "MLP"
    }

    fn regularization_loss(&self) -> f32 {
        self.dense_layers
            .iter()
            .map(|layer| layer.regularization_loss())
            .sum()
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn apply_constraints(&mut self) {
        for layer in &mut self.dense_layers {
            layer.apply_constraints();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_config() {
        let config = MLPConfig::new(128)
            .add_layer(64, ActivationType::relu())
            .add_layer(32, ActivationType::relu())
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

        let config = MLPConfig::new(128).add_layer(0, ActivationType::relu());
        assert!(config.validate().is_err()); // Zero dimension
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MLPConfig::new(10)
            .add_layer(5, ActivationType::relu())
            .add_layer(2, ActivationType::None)
            .build()
            .unwrap();

        let input = Tensor::ones(&[3, 10]); // batch of 3
        let output = mlp.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 2]);
    }

    #[test]
    fn test_mlp_new() {
        let mlp = MLP::new(128, &[64, 32], 10, ActivationType::relu()).unwrap();
        assert_eq!(mlp.num_layers(), 3);
        assert_eq!(mlp.input_dim(), 128);
        assert_eq!(mlp.output_dim(), 10);
    }

    #[test]
    fn test_mlp_backward() {
        let mut mlp = MLPConfig::new(10)
            .add_layer(5, ActivationType::relu())
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
            .add_layer(5, ActivationType::relu())
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
            .add_layer(5, ActivationType::relu())
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
            ActivationType::relu(),
            ActivationType::sigmoid(),
            ActivationType::tanh(),
            ActivationType::gelu(),
            ActivationType::None,
        ] {
            let mlp = MLPConfig::new(10).add_layer(5, activation).build().unwrap();

            let input = Tensor::rand(&[2, 10]);
            let output = mlp.forward(&input);
            assert!(output.is_ok());
        }
    }
}
