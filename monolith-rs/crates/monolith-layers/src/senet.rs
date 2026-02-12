//! SENet (Squeeze-and-Excitation Network) layer for feature importance.
//!
//! This module provides the [`SENetLayer`] which implements the squeeze-and-excitation
//! mechanism for learning feature importance weights. The SE block consists of:
//!
//! 1. **Squeeze**: Global average pooling on the feature dimension
//! 2. **Excitation**: Two FC layers (reduction -> expansion) with ReLU and Sigmoid
//! 3. **Reweight**: Element-wise multiplication of input features by attention weights
//!
//! # Reference
//!
//! Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks.
//! In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition.
//!
//! # Example
//!
//! ```
//! use monolith_layers::senet::{SENetLayer, SENetConfig};
//! use monolith_layers::tensor::Tensor;
//! use monolith_layers::layer::Layer;
//!
//! let config = SENetConfig::new(64)
//!     .with_reduction_ratio(4)
//!     .with_bias(true);
//!
//! let senet = SENetLayer::from_config(config).unwrap();
//! let input = Tensor::rand(&[32, 64]);  // batch of 32, 64 features
//! let output = senet.forward(&input).unwrap();
//! assert_eq!(output.shape(), &[32, 64]);
//! ```

use crate::constraint::Constraint;
use crate::dense::Dense;
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::merge::{merge_tensor_list, merge_tensor_list_tensor, MergeOutput, MergeType};
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration for the SENet layer.
///
/// # Example
///
/// ```
/// use monolith_layers::senet::SENetConfig;
///
/// let config = SENetConfig::new(128)
///     .with_reduction_ratio(8)
///     .with_bias(true);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SENetConfig {
    /// Number of input features.
    pub input_dim: usize,
    /// Reduction ratio for the squeeze operation (default: 4).
    /// The bottleneck dimension will be input_dim / reduction_ratio.
    pub reduction_ratio: usize,
    /// Explicit compression dimension (overrides reduction_ratio when set).
    pub cmp_dim: Option<usize>,
    /// Whether to use bias in FC layers (default: true).
    pub use_bias: bool,
    /// Initializer for FC layers.
    pub initializer: Initializer,
    /// Kernel regularizer for FC layers.
    pub regularizer: Regularizer,
    /// Merge output type (stack/concat/none).
    pub out_type: MergeType,
    /// Whether to keep list output (only used by forward_with_merge).
    pub keep_list: bool,
    /// Optional num_feature override for 2D inputs.
    pub num_feature: Option<usize>,
    /// Whether to use GPU-optimized path (kept for parity, no-op here).
    pub on_gpu: bool,
    /// Whether to use the compression tower (fc1/fc2). When false, weights are identity.
    pub use_cmp_tower: bool,
}

impl SENetConfig {
    /// Creates a new SENet configuration with the specified input dimension.
    ///
    /// Default values:
    /// - reduction_ratio: 4
    /// - use_bias: true
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features
    pub fn new(input_dim: usize) -> Self {
        Self {
            input_dim,
            reduction_ratio: 4,
            cmp_dim: None,
            use_bias: true,
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
            out_type: MergeType::Concat,
            keep_list: false,
            num_feature: Some(input_dim),
            on_gpu: false,
            use_cmp_tower: true,
        }
    }

    /// Sets the reduction ratio for the squeeze operation.
    ///
    /// The bottleneck dimension will be input_dim / reduction_ratio.
    /// A higher ratio means more compression in the bottleneck.
    ///
    /// # Arguments
    ///
    /// * `ratio` - The reduction ratio (typically 4, 8, or 16)
    pub fn with_reduction_ratio(mut self, ratio: usize) -> Self {
        self.reduction_ratio = ratio;
        self
    }

    /// Sets an explicit compression dimension (overrides reduction_ratio).
    pub fn with_cmp_dim(mut self, cmp_dim: usize) -> Self {
        self.cmp_dim = Some(cmp_dim);
        self
    }

    /// Sets whether to use bias in the FC layers.
    ///
    /// # Arguments
    ///
    /// * `use_bias` - Whether to include bias terms
    pub fn with_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Sets initializer for FC layers.
    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    /// Sets kernel regularizer for FC layers.
    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    /// Sets the merge output type.
    pub fn with_out_type(mut self, out_type: MergeType) -> Self {
        self.out_type = out_type;
        self
    }

    /// Sets whether to keep list output.
    pub fn with_keep_list(mut self, keep_list: bool) -> Self {
        self.keep_list = keep_list;
        self
    }

    /// Sets num_feature override for 2D inputs.
    pub fn with_num_feature(mut self, num_feature: usize) -> Self {
        self.num_feature = Some(num_feature);
        self
    }

    /// Sets on_gpu flag (no-op, kept for parity).
    pub fn with_on_gpu(mut self, on_gpu: bool) -> Self {
        self.on_gpu = on_gpu;
        self
    }

    /// Disables the compression tower (fc1/fc2); uses identity weights.
    pub fn with_use_cmp_tower(mut self, use_cmp_tower: bool) -> Self {
        self.use_cmp_tower = use_cmp_tower;
        self
    }

    /// Builds a SENetLayer from this configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid (e.g., reduction_ratio is 0
    /// or larger than input_dim).
    pub fn build(self) -> Result<SENetLayer, LayerError> {
        SENetLayer::from_config(self)
    }
}

/// Squeeze-and-Excitation Network layer for learning feature importance.
///
/// The SE block adaptively recalibrates channel-wise (feature-wise) responses
/// by explicitly modeling interdependencies between features. It consists of:
///
/// 1. **Squeeze**: Compute global statistics via average pooling
/// 2. **Excitation**: Learn feature importance via two FC layers with non-linearities
/// 3. **Scale**: Reweight the original features by the learned importance weights
///
/// For 2D inputs of shape `[batch_size, features]`:
/// - Squeeze: Compute mean across batch (conceptually, we treat each feature as a channel)
/// - Actually, for feature recalibration, we keep the batch dimension and compute
///   attention weights per sample
///
/// # Architecture
///
/// ```text
/// Input [B, C] --> Squeeze (identity for 2D) --> FC1 [C -> C/r] --> ReLU
///                                                     |
///                                                     v
///                                           FC2 [C/r -> C] --> Sigmoid
///                                                     |
///                                                     v
///                                           Attention weights [B, C]
///                                                     |
///                                                     v
/// Output = Input * Attention weights [B, C]
/// ```
///
/// # Example
///
/// ```
/// use monolith_layers::senet::SENetLayer;
/// use monolith_layers::tensor::Tensor;
/// use monolith_layers::layer::Layer;
///
/// let senet = SENetLayer::new(64, 4, true);
/// let input = Tensor::rand(&[8, 64]);
/// let output = senet.forward(&input).unwrap();
/// assert_eq!(output.shape(), &[8, 64]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SENetLayer {
    /// First FC layer for reduction (squeeze)
    fc1: Option<Dense>,
    /// Second FC layer for expansion (excitation)
    fc2: Option<Dense>,
    /// Input dimension (number of features)
    input_dim: usize,
    /// Bottleneck dimension (input_dim / reduction_ratio or cmp_dim)
    bottleneck_dim: usize,
    /// Reduction ratio
    reduction_ratio: usize,
    /// Explicit compression dimension (if set)
    cmp_dim: Option<usize>,
    /// Output merge type
    out_type: MergeType,
    /// Keep list output
    keep_list: bool,
    /// Optional num_feature override for 2D inputs
    num_feature: Option<usize>,
    /// Whether to use compression tower
    use_cmp_tower: bool,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Cached list inputs for backward pass
    cached_list_inputs: Option<Vec<Tensor>>,
    /// Cached list squeeze embedding
    cached_list_squeeze: Option<Tensor>,
    /// Cached attention weights for backward pass
    cached_attention: Option<Tensor>,
    /// Cached intermediate activations for backward pass
    cached_fc1_output: Option<Tensor>,
    /// Cached original input shape
    cached_input_shape: Option<Vec<usize>>,
    /// Whether input was reshaped to 3D
    cached_input_was_reshaped: bool,
    /// Training mode flag
    training: bool,
}

impl SENetLayer {
    /// Creates a new SENet layer with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `input_dim` - Number of input features
    /// * `reduction_ratio` - Reduction ratio for bottleneck (typically 4, 8, or 16)
    /// * `use_bias` - Whether to use bias in FC layers
    ///
    /// # Panics
    ///
    /// Panics if reduction_ratio is 0 or if input_dim / reduction_ratio is 0.
    ///
    /// # Example
    ///
    /// ```
    /// use monolith_layers::senet::SENetLayer;
    ///
    /// let senet = SENetLayer::new(128, 4, true);
    /// ```
    pub fn new(input_dim: usize, reduction_ratio: usize, use_bias: bool) -> Self {
        assert!(reduction_ratio > 0, "reduction_ratio must be positive");
        let bottleneck_dim = (input_dim / reduction_ratio).max(1);

        let fc1 = Dense::new_with_options(
            input_dim,
            bottleneck_dim,
            Initializer::GlorotUniform,
            Initializer::Zeros,
            use_bias,
            Regularizer::None,
            Regularizer::None,
            Constraint::None,
            Constraint::None,
        );

        let fc2 = Dense::new_with_options(
            bottleneck_dim,
            input_dim,
            Initializer::GlorotUniform,
            Initializer::Zeros,
            use_bias,
            Regularizer::None,
            Regularizer::None,
            Constraint::None,
            Constraint::None,
        );

        Self {
            fc1: Some(fc1),
            fc2: Some(fc2),
            input_dim,
            bottleneck_dim,
            reduction_ratio,
            cmp_dim: None,
            out_type: MergeType::Concat,
            keep_list: false,
            num_feature: Some(input_dim),
            use_cmp_tower: true,
            cached_input: None,
            cached_list_inputs: None,
            cached_list_squeeze: None,
            cached_attention: None,
            cached_fc1_output: None,
            cached_input_shape: None,
            cached_input_was_reshaped: false,
            training: true,
        }
    }

    /// Creates a SENet layer from a configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The SENet configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn from_config(config: SENetConfig) -> Result<Self, LayerError> {
        if config.reduction_ratio == 0 {
            return Err(LayerError::ConfigError {
                message: "reduction_ratio must be positive".to_string(),
            });
        }
        if config.input_dim == 0 {
            return Err(LayerError::ConfigError {
                message: "input_dim must be positive".to_string(),
            });
        }
        let use_cmp_tower = config.use_cmp_tower;
        let bottleneck_dim = if use_cmp_tower {
            if let Some(cmp_dim) = config.cmp_dim {
                cmp_dim
            } else {
                config.input_dim / config.reduction_ratio
            }
        } else {
            config.input_dim
        };
        if use_cmp_tower && bottleneck_dim == 0 {
            return Err(LayerError::ConfigError {
                message: format!(
                    "Bottleneck dimension is 0: input_dim={} / reduction_ratio={} = 0. \
                     Use a smaller reduction_ratio.",
                    config.input_dim, config.reduction_ratio
                ),
            });
        }
        let fc1 = if use_cmp_tower {
            Some(Dense::new_with_options(
                config.input_dim,
                bottleneck_dim,
                config.initializer,
                Initializer::Zeros,
                config.use_bias,
                config.regularizer.clone(),
                Regularizer::None,
                Constraint::None,
                Constraint::None,
            ))
        } else {
            None
        };
        let fc2 = if use_cmp_tower {
            Some(Dense::new_with_options(
                bottleneck_dim,
                config.input_dim,
                config.initializer,
                Initializer::Zeros,
                config.use_bias,
                config.regularizer,
                Regularizer::None,
                Constraint::None,
                Constraint::None,
            ))
        } else {
            None
        };

        Ok(Self {
            fc1,
            fc2,
            input_dim: config.input_dim,
            bottleneck_dim,
            reduction_ratio: config.reduction_ratio,
            cmp_dim: config.cmp_dim,
            out_type: config.out_type,
            keep_list: config.keep_list,
            num_feature: config.num_feature,
            use_cmp_tower,
            cached_input: None,
            cached_list_inputs: None,
            cached_list_squeeze: None,
            cached_attention: None,
            cached_fc1_output: None,
            cached_input_shape: None,
            cached_input_was_reshaped: false,
            training: true,
        })
    }

    /// Returns the input dimension.
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// Returns the bottleneck dimension.
    pub fn bottleneck_dim(&self) -> usize {
        self.bottleneck_dim
    }

    /// Returns the reduction ratio.
    pub fn reduction_ratio(&self) -> usize {
        self.reduction_ratio
    }

    /// Returns the learned attention weights from the last forward pass.
    ///
    /// This can be useful for analyzing which features the model considers important.
    pub fn last_attention_weights(&self) -> Option<&Tensor> {
        self.cached_attention.as_ref()
    }

    /// Returns mutable references to the dense layers (fc1, fc2) if present.
    pub fn layers_mut(&mut self) -> Vec<&mut Dense> {
        let mut layers = Vec::new();
        if let Some(fc1) = &mut self.fc1 {
            layers.push(fc1);
        }
        if let Some(fc2) = &mut self.fc2 {
            layers.push(fc2);
        }
        layers
    }

    /// Clears cached values.
    pub fn clear_cache(&mut self) {
        self.cached_input = None;
        self.cached_list_inputs = None;
        self.cached_list_squeeze = None;
        self.cached_attention = None;
        self.cached_fc1_output = None;
        self.cached_input_shape = None;
        self.cached_input_was_reshaped = false;
    }

    /// Applies ReLU activation element-wise.
    fn relu(x: &Tensor) -> Tensor {
        x.map(|v| v.max(0.0))
    }

    /// Applies Sigmoid activation element-wise.
    fn sigmoid(x: &Tensor) -> Tensor {
        x.map(|v| 1.0 / (1.0 + (-v).exp()))
    }

    /// Computes the derivative of ReLU.
    fn relu_grad(x: &Tensor) -> Tensor {
        x.map(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }

    /// Computes the derivative of Sigmoid given the sigmoid output.
    fn sigmoid_grad(sigmoid_output: &Tensor) -> Tensor {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let one_minus = sigmoid_output.map(|v| 1.0 - v);
        sigmoid_output.mul(&one_minus)
    }

    /// Performs forward pass and caches intermediate values for backward pass.
    ///
    /// Use this method during training to enable gradient computation.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    ///
    /// Output tensor of the same shape as input, with features reweighted
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let (input3d, used_3d, original_shape) = if input.ndim() == 3 {
            (input.clone(), true, input.shape().to_vec())
        } else if input.ndim() == 2 {
            if let Some(num_feature) = self.num_feature {
                if num_feature > 1 {
                    let emb_size = input.shape()[1] / num_feature;
                    (
                        input.reshape(&[input.shape()[0], num_feature, emb_size]),
                        true,
                        input.shape().to_vec(),
                    )
                } else {
                    (input.clone(), false, input.shape().to_vec())
                }
            } else {
                (input.clone(), false, input.shape().to_vec())
            }
        } else {
            return Err(LayerError::ForwardError {
                message: format!("Expected 2D/3D input, got {}D", input.ndim()),
            });
        };

        self.cached_input = Some(input3d.clone());
        self.cached_input_shape = Some(original_shape.clone());
        self.cached_input_was_reshaped = used_3d && input.ndim() == 2;

        let (batch, num_feat, emb_dim) = if used_3d {
            (input3d.shape()[0], input3d.shape()[1], input3d.shape()[2])
        } else {
            (input3d.shape()[0], input3d.shape()[1], 1)
        };

        let squeeze = if used_3d {
            input3d.mean_axis(2)
        } else {
            input3d.clone()
        };

        let (attention, fc1_cache) = if let (Some(fc1), Some(fc2)) = (&mut self.fc1, &mut self.fc2)
        {
            let fc1_out = fc1.forward_train(&squeeze)?;
            let fc1_relu = Self::relu(&fc1_out);
            let fc2_out = fc2.forward_train(&fc1_relu)?;
            let attention = Self::sigmoid(&fc2_out);
            (attention, Some(fc1_out))
        } else {
            (squeeze.clone(), None)
        };
        self.cached_fc1_output = fc1_cache;
        self.cached_attention = Some(attention.clone());

        let output = if used_3d {
            let attn_b = attention
                .reshape(&[batch, num_feat, 1])
                .broadcast_as(&[batch, num_feat, emb_dim]);
            input3d.mul(&attn_b)
        } else {
            input3d.mul(&attention)
        };

        let merged = if used_3d {
            match self.out_type {
                MergeType::Concat => {
                    merge_tensor_list_tensor(vec![output], MergeType::Concat, None, 1)?
                }
                MergeType::Stack => output,
                MergeType::None => {
                    return Err(LayerError::ForwardError {
                        message: "SENet forward cannot return list when out_type is None"
                            .to_string(),
                    })
                }
            }
        } else {
            output
        };

        Ok(merged)
    }

    fn squeeze_list_inputs(inputs: &[Tensor]) -> Result<Tensor, LayerError> {
        if inputs.is_empty() {
            return Err(LayerError::ForwardError {
                message: "SENet expects non-empty input list".to_string(),
            });
        }
        let batch = inputs[0].shape()[0];
        let mut parts = Vec::with_capacity(inputs.len());
        for input in inputs {
            if input.ndim() != 2 {
                return Err(LayerError::ForwardError {
                    message: "SENet list inputs must be 2D tensors".to_string(),
                });
            }
            if input.shape()[0] != batch {
                return Err(LayerError::ShapeMismatch {
                    expected: vec![batch],
                    actual: vec![input.shape()[0]],
                });
            }
            let mean = input.mean_axis(1);
            let mean = mean.reshape(&[batch, 1]);
            parts.push(mean);
        }
        Ok(Tensor::cat(&parts, 1))
    }

    fn split_attention_2d(attention: &Tensor, num_feat: usize) -> Result<Vec<Tensor>, LayerError> {
        if attention.ndim() != 2 || attention.shape()[1] != num_feat {
            return Err(LayerError::ShapeMismatch {
                expected: vec![attention.shape()[0], num_feat],
                actual: attention.shape().to_vec(),
            });
        }
        let mut weights = Vec::with_capacity(num_feat);
        for i in 0..num_feat {
            weights.push(attention.narrow(1, i, 1));
        }
        Ok(weights)
    }

    fn broadcast_weight_2d(weight: &Tensor, target: &Tensor) -> Tensor {
        let batch = target.shape()[0];
        let mut shape = vec![1usize; target.ndim()];
        shape[0] = batch;
        if weight.ndim() == 2 {
            shape[1] = weight.shape()[1];
        }
        let w = if weight.shape().len() == shape.len() {
            weight.clone()
        } else {
            weight.reshape(&shape)
        };
        w.broadcast_as(target.shape())
    }

    /// Forward pass for list inputs (each tensor must be 2D).
    pub fn forward_train_with_list(
        &mut self,
        inputs: &[Tensor],
    ) -> Result<MergeOutput, LayerError> {
        let squeeze = Self::squeeze_list_inputs(inputs)?;
        self.cached_list_inputs = Some(inputs.to_vec());
        self.cached_list_squeeze = Some(squeeze.clone());

        let (attention, fc1_cache) = if let (Some(fc1), Some(fc2)) = (&mut self.fc1, &mut self.fc2)
        {
            let fc1_out = fc1.forward_train(&squeeze)?;
            let fc1_relu = Self::relu(&fc1_out);
            let fc2_out = fc2.forward_train(&fc1_relu)?;
            let attention = Self::sigmoid(&fc2_out);
            (attention, Some(fc1_out))
        } else {
            (squeeze.clone(), None)
        };

        self.cached_fc1_output = fc1_cache;
        self.cached_attention = Some(attention.clone());

        let weights = Self::split_attention_2d(&attention, inputs.len())?;
        let mut outputs = Vec::with_capacity(inputs.len());
        for (input, weight) in inputs.iter().zip(weights.iter()) {
            let weight_b = Self::broadcast_weight_2d(weight, input);
            outputs.push(input.mul(&weight_b));
        }

        Ok(merge_tensor_list(
            outputs,
            self.out_type,
            Some(inputs.len()),
            1,
            self.keep_list,
        )?)
    }

    /// Backward pass for list inputs (expects gradients for each output).
    pub fn backward_with_list(&mut self, grads: &[Tensor]) -> Result<Vec<Tensor>, LayerError> {
        let inputs = self
            .cached_list_inputs
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        if grads.len() != inputs.len() {
            return Err(LayerError::BackwardError {
                message: "SENet backward expects one grad per input tensor".to_string(),
            });
        }
        let attention = self
            .cached_attention
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let weights = Self::split_attention_2d(attention, inputs.len())?;

        let mut grad_attention_parts = Vec::with_capacity(inputs.len());
        let mut input_grads = Vec::with_capacity(inputs.len());
        for ((grad, input), weight) in grads.iter().zip(inputs.iter()).zip(weights.iter()) {
            if input.ndim() != 2 {
                return Err(LayerError::BackwardError {
                    message: "SENet list backward only supports 2D inputs".to_string(),
                });
            }
            let weight_b = Self::broadcast_weight_2d(weight, input);
            input_grads.push(grad.mul(&weight_b));

            let grad_weight = grad.mul(input).sum_axis(1).reshape(&[input.shape()[0], 1]);
            grad_attention_parts.push(grad_weight);
        }

        let grad_attention = Tensor::cat(&grad_attention_parts, 1);

        let grad_squeeze = if let (Some(fc1), Some(fc2), Some(fc1_output)) = (
            &mut self.fc1,
            &mut self.fc2,
            self.cached_fc1_output.as_ref(),
        ) {
            let attention_grad = Self::sigmoid_grad(attention);
            let grad_fc2 = grad_attention.mul(&attention_grad);
            let grad_fc1_relu = fc2.backward(&grad_fc2)?;
            let relu_grad = Self::relu_grad(fc1_output);
            let grad_fc1 = grad_fc1_relu.mul(&relu_grad);
            fc1.backward(&grad_fc1)?
        } else {
            grad_attention
        };

        // propagate squeeze gradient back to inputs (mean over axis 1)
        for (idx, input) in inputs.iter().enumerate() {
            let grad_slice = grad_squeeze.narrow(1, idx, 1);
            let denom = input.shape()[1] as f32;
            let grad_b = grad_slice
                .reshape(&[input.shape()[0], 1])
                .broadcast_as(&[input.shape()[0], input.shape()[1]])
                .scale(1.0 / denom);
            input_grads[idx] = input_grads[idx].add(&grad_b);
        }

        Ok(input_grads)
    }
}

impl Layer for SENetLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        // Validate the expected input dimension early to mirror Python/Keras behavior
        // and to avoid panicking in reshape when the dimensions don't match.
        if input.ndim() == 2 && input.shape()[1] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[1],
            });
        }
        if input.ndim() == 3 && input.shape()[2] != self.input_dim {
            return Err(LayerError::InvalidInputDimension {
                expected: self.input_dim,
                actual: input.shape()[2],
            });
        }

        let (input3d, used_3d) = if input.ndim() == 3 {
            (input.clone(), true)
        } else if input.ndim() == 2 {
            if let Some(num_feature) = self.num_feature {
                if num_feature > 1 {
                    if input.shape()[1] % num_feature != 0 {
                        return Err(LayerError::ForwardError {
                            message: format!(
                                "Input dim {} not divisible by num_feature {}",
                                input.shape()[1],
                                num_feature
                            ),
                        });
                    }
                    let emb_size = input.shape()[1] / num_feature;
                    (
                        input.reshape(&[input.shape()[0], num_feature, emb_size]),
                        true,
                    )
                } else {
                    (input.clone(), false)
                }
            } else {
                (input.clone(), false)
            }
        } else {
            return Err(LayerError::ForwardError {
                message: format!("Expected 2D/3D input, got {}D", input.ndim()),
            });
        };

        let (batch, num_feat, emb_dim) = if used_3d {
            (input3d.shape()[0], input3d.shape()[1], input3d.shape()[2])
        } else {
            (input3d.shape()[0], input3d.shape()[1], 1)
        };

        let squeeze = if used_3d {
            input3d.mean_axis(2)
        } else {
            input3d.clone()
        };

        let attention = if let (Some(fc1), Some(fc2)) = (&self.fc1, &self.fc2) {
            let fc1_out = fc1.forward(&squeeze)?;
            let fc1_relu = Self::relu(&fc1_out);
            let fc2_out = fc2.forward(&fc1_relu)?;
            Self::sigmoid(&fc2_out)
        } else {
            squeeze.clone()
        };

        let output = if used_3d {
            let attn_b = attention
                .reshape(&[batch, num_feat, 1])
                .broadcast_as(&[batch, num_feat, emb_dim]);
            input3d.mul(&attn_b)
        } else {
            input3d.mul(&attention)
        };

        if used_3d {
            match self.out_type {
                MergeType::Concat => Ok(merge_tensor_list_tensor(
                    vec![output],
                    MergeType::Concat,
                    None,
                    1,
                )?),
                MergeType::Stack => Ok(output),
                MergeType::None => Err(LayerError::ForwardError {
                    message: "SENet forward cannot return list when out_type is None".to_string(),
                }),
            }
        } else {
            Ok(output)
        }
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let attention = self
            .cached_attention
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let fc1_output = self.cached_fc1_output.as_ref();

        let input_is_3d = input.ndim() == 3;
        let (batch, num_feat, emb_dim) = if input_is_3d {
            (input.shape()[0], input.shape()[1], input.shape()[2])
        } else {
            (input.shape()[0], input.shape()[1], 1)
        };

        let grad_out = if input_is_3d {
            match self.out_type {
                MergeType::Concat => {
                    let shape = vec![batch, num_feat, emb_dim];
                    grad.reshape(&shape)
                }
                MergeType::Stack => grad.clone(),
                MergeType::None => {
                    return Err(LayerError::BackwardError {
                        message: "SENet backward cannot accept list output".to_string(),
                    })
                }
            }
        } else {
            grad.clone()
        };

        let (grad_input_scale, grad_attention) = if input_is_3d {
            let attn_b = attention
                .reshape(&[batch, num_feat, 1])
                .broadcast_as(&[batch, num_feat, emb_dim]);
            let grad_input_scale = grad_out.mul(&attn_b);
            let grad_attention = grad_out.mul(input).sum_axis(2);
            (grad_input_scale, grad_attention)
        } else {
            (grad_out.mul(attention), grad_out.mul(input))
        };

        let grad_squeeze = if let (Some(fc1), Some(fc2), Some(fc1_output)) =
            (&mut self.fc1, &mut self.fc2, fc1_output)
        {
            let attention_grad = Self::sigmoid_grad(attention);
            let grad_fc2 = grad_attention.mul(&attention_grad);
            let grad_fc1_relu = fc2.backward(&grad_fc2)?;
            let relu_grad = Self::relu_grad(fc1_output);
            let grad_fc1 = grad_fc1_relu.mul(&relu_grad);
            fc1.backward(&grad_fc1)?
        } else {
            grad_attention.clone()
        };

        let grad_input_fc1 = if input_is_3d {
            grad_squeeze
                .reshape(&[batch, num_feat, 1])
                .broadcast_as(&[batch, num_feat, emb_dim])
                .scale(1.0 / emb_dim as f32)
        } else {
            grad_squeeze
        };

        let mut grad_input = grad_input_scale.add(&grad_input_fc1);
        if self.cached_input_was_reshaped {
            if let Some(orig_shape) = &self.cached_input_shape {
                grad_input = grad_input.reshape(orig_shape);
            }
        }

        Ok(grad_input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(fc1) = &self.fc1 {
            params.extend(fc1.parameters());
        }
        if let Some(fc2) = &self.fc2 {
            params.extend(fc2.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(fc1) = &mut self.fc1 {
            params.extend(fc1.parameters_mut());
        }
        if let Some(fc2) = &mut self.fc2 {
            params.extend(fc2.parameters_mut());
        }
        params
    }

    fn name(&self) -> &str {
        "SENetLayer"
    }

    fn regularization_loss(&self) -> f32 {
        let mut loss = 0.0;
        if let Some(fc1) = &self.fc1 {
            loss += fc1.regularization_loss();
        }
        if let Some(fc2) = &self.fc2 {
            loss += fc2.regularization_loss();
        }
        loss
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn apply_constraints(&mut self) {
        if let Some(fc1) = &mut self.fc1 {
            fc1.apply_constraints();
        }
        if let Some(fc2) = &mut self.fc2 {
            fc2.apply_constraints();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_senet_config_defaults() {
        let config = SENetConfig::new(64);
        assert_eq!(config.input_dim, 64);
        assert_eq!(config.reduction_ratio, 4);
        assert!(config.use_bias);
    }

    #[test]
    fn test_senet_config_builder() {
        let config = SENetConfig::new(128)
            .with_reduction_ratio(8)
            .with_bias(false);

        assert_eq!(config.input_dim, 128);
        assert_eq!(config.reduction_ratio, 8);
        assert!(!config.use_bias);
    }

    #[test]
    fn test_senet_creation() {
        let senet = SENetLayer::new(64, 4, true);
        assert_eq!(senet.input_dim(), 64);
        assert_eq!(senet.bottleneck_dim(), 16);
        assert_eq!(senet.reduction_ratio(), 4);
    }

    #[test]
    fn test_senet_from_config() {
        let config = SENetConfig::new(128).with_reduction_ratio(8);
        let senet = SENetLayer::from_config(config).unwrap();

        assert_eq!(senet.input_dim(), 128);
        assert_eq!(senet.bottleneck_dim(), 16);
    }

    #[test]
    fn test_senet_from_config_error() {
        // Test invalid reduction ratio
        let config = SENetConfig {
            reduction_ratio: 0,
            ..SENetConfig::new(64)
        };
        SENetLayer::from_config(config)
            .expect_err("SENet config with zero reduction ratio should fail");

        // Test bottleneck dimension becoming 0
        let config = SENetConfig {
            reduction_ratio: 8,
            ..SENetConfig::new(4)
        };
        SENetLayer::from_config(config)
            .expect_err("SENet config producing zero bottleneck dimension should fail");
    }

    #[test]
    fn test_senet_forward_shape() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 64]);

        let output = senet.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8, 64]);
    }

    #[test]
    fn test_senet_forward_preserves_batch_size() {
        let senet = SENetLayer::new(32, 4, true);

        // Test with different batch sizes
        for batch_size in [1, 4, 16, 32] {
            let input = Tensor::rand(&[batch_size, 32]);
            let output = senet.forward(&input).unwrap();
            assert_eq!(output.shape(), &[batch_size, 32]);
        }
    }

    #[test]
    fn test_senet_forward_invalid_input_dim() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 32]); // Wrong input dimension

        let result = senet.forward(&input);
        result.expect_err("SENet forward should fail for mismatched input dimension");
    }

    #[test]
    fn test_senet_forward_invalid_input_ndim() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 4, 64]); // 3D instead of 2D

        let result = senet.forward(&input);
        result.expect_err("SENet forward should fail for non-2D inputs");
    }

    #[test]
    fn test_senet_attention_range() {
        let senet = SENetLayer::new(64, 4, true);
        let input = Tensor::rand(&[8, 64]);

        let _output = senet.forward(&input).unwrap();

        // After forward, we can't access cached_attention since forward doesn't cache
        // Use forward_train instead
        let mut senet_train = SENetLayer::new(64, 4, true);
        let _output = senet_train.forward_train(&input).unwrap();

        if let Some(attention) = senet_train.last_attention_weights() {
            // All attention weights should be between 0 and 1 (sigmoid output)
            for val in attention.data() {
                assert!(
                    val >= 0.0 && val <= 1.0,
                    "Attention weight {} out of range",
                    val
                );
            }
        }
    }

    #[test]
    fn test_senet_parameters() {
        let senet = SENetLayer::new(64, 4, true);
        let params = senet.parameters();

        // Dense defaults include a trainable kernel-norm scale; each dense has:
        // weights, bias, kernel_norm => 3 tensors.
        assert_eq!(params.len(), 6);
    }

    #[test]
    fn test_senet_parameters_no_bias() {
        let senet = SENetLayer::new(64, 4, false);
        let params = senet.parameters();

        assert_eq!(params.len(), 4);
    }

    #[test]
    fn test_senet_backward() {
        let mut senet = SENetLayer::new(32, 4, true);
        let input = Tensor::rand(&[4, 32]);

        // Forward pass with caching
        let _output = senet.forward_train(&input).unwrap();

        // Backward pass
        let grad = Tensor::ones(&[4, 32]);
        let input_grad = senet.backward(&grad).unwrap();

        assert_eq!(input_grad.shape(), &[4, 32]);
    }

    #[test]
    fn test_senet_backward_without_forward() {
        let mut senet = SENetLayer::new(32, 4, true);
        let grad = Tensor::ones(&[4, 32]);

        let result = senet.backward(&grad);
        result.expect_err("SENet backward should fail when no forward cache exists");
    }

    #[test]
    fn test_senet_training_mode() {
        let mut senet = SENetLayer::new(64, 4, true);
        assert!(senet.is_training());

        senet.set_training(false);
        assert!(!senet.is_training());

        senet.set_training(true);
        assert!(senet.is_training());
    }

    #[test]
    fn test_senet_name() {
        let senet = SENetLayer::new(64, 4, true);
        assert_eq!(senet.name(), "SENetLayer");
    }

    #[test]
    fn test_senet_clear_cache() {
        let mut senet = SENetLayer::new(32, 4, true);
        let input = Tensor::rand(&[4, 32]);

        let _output = senet.forward_train(&input).unwrap();
        assert!(senet.last_attention_weights().is_some());

        senet.clear_cache();
        assert!(senet.last_attention_weights().is_none());
    }

    #[test]
    fn test_senet_different_reduction_ratios() {
        for ratio in [2, 4, 8, 16] {
            let senet = SENetLayer::new(64, ratio, true);
            let expected_bottleneck = 64 / ratio;
            assert_eq!(senet.bottleneck_dim(), expected_bottleneck);

            let input = Tensor::rand(&[4, 64]);
            let output = senet.forward(&input).unwrap();
            assert_eq!(output.shape(), &[4, 64]);
        }
    }

    #[test]
    fn test_senet_with_ones_input() {
        let senet = SENetLayer::new(16, 4, true);
        let input = Tensor::ones(&[2, 16]);

        let output = senet.forward(&input).unwrap();

        // Output should be input * attention
        // Since input is all ones, output should equal attention weights
        // (scaled by the learned attention)
        assert_eq!(output.shape(), &[2, 16]);
    }

    #[test]
    fn test_senet_config_build() {
        let config = SENetConfig::new(64).with_reduction_ratio(4);
        let senet = config.build().unwrap();

        assert_eq!(senet.input_dim(), 64);
        assert_eq!(senet.bottleneck_dim(), 16);
    }

    #[test]
    fn test_senet_minimum_bottleneck() {
        // When input_dim / reduction_ratio < 1, bottleneck should be at least 1
        let senet = SENetLayer::new(3, 4, true);
        assert_eq!(senet.bottleneck_dim(), 1); // max(3/4, 1) = max(0, 1) = 1
    }
}
