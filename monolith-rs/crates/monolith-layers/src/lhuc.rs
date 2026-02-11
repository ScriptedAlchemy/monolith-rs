//! LHUC tower implementation.

use crate::activation_layer::ActivationLayer;
use crate::constraint::Constraint;
use crate::dense::Dense;
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::mlp::{ActivationType, MLPConfig, MLP};
use crate::normalization::BatchNorm;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// LHUC output dimension specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LHUCOutputDims {
    Shared(Vec<usize>),
    PerLayer(Vec<Vec<usize>>),
}

/// LHUC override options for LHUC-side MLPs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LHUCOverrides {
    pub use_bias: Option<bool>,
    pub use_weight_norm: Option<bool>,
    pub use_learnable_weight_norm: Option<bool>,
    pub enable_batch_normalization: Option<bool>,
    pub batch_normalization_momentum: Option<f32>,
    pub batch_normalization_renorm: Option<bool>,
    pub batch_normalization_renorm_clipping: Option<(f32, f32, f32)>,
    pub batch_normalization_renorm_momentum: Option<f32>,
    pub initializers: Option<Vec<Initializer>>,
    pub kernel_regularizer: Option<Regularizer>,
    pub bias_regularizer: Option<Regularizer>,
    pub kernel_constraint: Option<Constraint>,
    pub bias_constraint: Option<Constraint>,
}

/// Configuration for LHUC tower.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LHUCConfig {
    pub input_dim: usize,
    pub output_dims: Vec<usize>,
    pub lhuc_output_dims: Option<LHUCOutputDims>,
    pub activations: Option<Vec<ActivationType>>,
    pub initializers: Option<Vec<Initializer>>,
    pub use_bias: bool,
    pub kernel_regularizer: Regularizer,
    pub bias_regularizer: Regularizer,
    pub kernel_constraint: Constraint,
    pub bias_constraint: Constraint,
    pub use_weight_norm: bool,
    pub use_learnable_weight_norm: bool,
    pub enable_batch_normalization: bool,
    pub batch_normalization_momentum: f32,
    pub batch_normalization_renorm: bool,
    pub batch_normalization_renorm_clipping: Option<(f32, f32, f32)>,
    pub batch_normalization_renorm_momentum: f32,
    pub lhuc_overrides: LHUCOverrides,
}

impl LHUCConfig {
    pub fn new(input_dim: usize, output_dims: Vec<usize>) -> Self {
        Self {
            input_dim,
            output_dims,
            lhuc_output_dims: None,
            activations: None,
            initializers: None,
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
            lhuc_overrides: LHUCOverrides::default(),
        }
    }

    pub fn with_activations(mut self, activations: Vec<ActivationType>) -> Self {
        self.activations = Some(activations);
        self
    }

    pub fn with_initializers(mut self, initializers: Vec<Initializer>) -> Self {
        self.initializers = Some(initializers);
        self
    }

    pub fn with_regularizers(
        mut self,
        kernel_regularizer: Regularizer,
        bias_regularizer: Regularizer,
    ) -> Self {
        self.kernel_regularizer = kernel_regularizer;
        self.bias_regularizer = bias_regularizer;
        self
    }

    pub fn with_constraints(
        mut self,
        kernel_constraint: Constraint,
        bias_constraint: Constraint,
    ) -> Self {
        self.kernel_constraint = kernel_constraint;
        self.bias_constraint = bias_constraint;
        self
    }

    pub fn with_lhuc_output_dims(mut self, dims: LHUCOutputDims) -> Self {
        self.lhuc_output_dims = Some(dims);
        self
    }

    pub fn with_overrides(mut self, overrides: LHUCOverrides) -> Self {
        self.lhuc_overrides = overrides;
        self
    }
}

#[derive(Debug, Clone)]
struct LHUCBlock {
    dense: Dense,
    batch_norm: Option<BatchNorm>,
    activation: Option<ActivationLayer>,
}

impl LHUCBlock {
    fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut out = self.dense.forward_train(input)?;
        if let Some(bn) = &mut self.batch_norm {
            out = bn.forward_train(&out)?;
        }
        if let Some(act) = &mut self.activation {
            out = act.forward_train(&out)?;
        }
        Ok(out)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let mut g = grad.clone();
        if let Some(act) = &mut self.activation {
            g = act.backward(&g)?;
        }
        if let Some(bn) = &mut self.batch_norm {
            g = bn.backward(&g)?;
        }
        self.dense.backward(&g)
    }
}

/// LHUC tower layer.
#[derive(Debug, Clone)]
pub struct LHUCTower {
    #[allow(dead_code)]
    config: LHUCConfig,
    input_batch_norm: Option<BatchNorm>,
    layers: Vec<LHUCBlock>,
    lhuc_layers: Vec<MLP>,
    cached_outputs: Vec<Tensor>,
    cached_gates: Vec<Tensor>,
    cached_lhuc_input: Option<Tensor>,
}

impl LHUCTower {
    pub fn from_config(config: LHUCConfig) -> Result<Self, LayerError> {
        if config.output_dims.is_empty() {
            return Err(LayerError::ConfigError {
                message: "LHUCTower requires at least one output dim".to_string(),
            });
        }

        let n_layers = config.output_dims.len();
        let activations = if let Some(acts) = &config.activations {
            if acts.len() == 1 {
                let mut expanded = vec![acts[0].clone(); n_layers];
                expanded[n_layers - 1] = ActivationType::None;
                expanded
            } else if acts.len() == n_layers {
                acts.clone()
            } else {
                return Err(LayerError::ConfigError {
                    message: "activations length must be 1 or match output_dims".to_string(),
                });
            }
        } else {
            let mut expanded = vec![ActivationType::relu(); n_layers];
            expanded[n_layers - 1] = ActivationType::None;
            expanded
        };

        let initializers = if let Some(inits) = &config.initializers {
            if inits.len() == 1 {
                vec![inits[0]; n_layers]
            } else if inits.len() == n_layers {
                inits.clone()
            } else {
                return Err(LayerError::ConfigError {
                    message: "initializers length must be 1 or match output_dims".to_string(),
                });
            }
        } else {
            vec![Initializer::GlorotUniform; n_layers]
        };

        let input_batch_norm = if config.enable_batch_normalization {
            let mut bn = BatchNorm::with_momentum(
                config.input_dim,
                config.batch_normalization_momentum,
                1e-6,
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

        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;
        for (idx, output_dim) in config.output_dims.iter().enumerate() {
            let mut dense = Dense::new_with_options(
                prev_dim,
                *output_dim,
                initializers[idx],
                Initializer::Zeros,
                config.use_bias,
                config.kernel_regularizer.clone(),
                config.bias_regularizer.clone(),
                config.kernel_constraint.clone(),
                config.bias_constraint.clone(),
            );
            if config.use_weight_norm {
                dense = dense.with_kernel_norm(config.use_learnable_weight_norm);
            }

            let bn = if config.enable_batch_normalization && idx != n_layers - 1 {
                let mut bn = BatchNorm::with_momentum(
                    *output_dim,
                    config.batch_normalization_momentum,
                    1e-6,
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

            let activation = if activations[idx] == ActivationType::None {
                None
            } else {
                Some(ActivationLayer::from_activation_type(
                    activations[idx].clone(),
                ))
            };

            layers.push(LHUCBlock {
                dense,
                batch_norm: bn,
                activation,
            });

            prev_dim = *output_dim;
        }

        let lhuc_output_dims = build_lhuc_output_dims(&config)?;
        let mut lhuc_layers = Vec::new();
        for (idx, dims) in lhuc_output_dims.iter().enumerate() {
            let mut mlp_config = MLPConfig::new(config.input_dim);
            let override_bias = config.lhuc_overrides.use_bias.unwrap_or(config.use_bias);
            let override_weight_norm = config
                .lhuc_overrides
                .use_weight_norm
                .unwrap_or(config.use_weight_norm);
            let override_learnable = config
                .lhuc_overrides
                .use_learnable_weight_norm
                .unwrap_or(config.use_learnable_weight_norm);
            let override_bn = config
                .lhuc_overrides
                .enable_batch_normalization
                .unwrap_or(config.enable_batch_normalization);
            let override_bn_momentum = config
                .lhuc_overrides
                .batch_normalization_momentum
                .unwrap_or(config.batch_normalization_momentum);
            let override_bn_renorm = config
                .lhuc_overrides
                .batch_normalization_renorm
                .unwrap_or(config.batch_normalization_renorm);
            let override_bn_clip = config
                .lhuc_overrides
                .batch_normalization_renorm_clipping
                .or(config.batch_normalization_renorm_clipping);
            let override_bn_mom = config
                .lhuc_overrides
                .batch_normalization_renorm_momentum
                .unwrap_or(config.batch_normalization_renorm_momentum);
            let override_kernel_reg = config
                .lhuc_overrides
                .kernel_regularizer
                .clone()
                .unwrap_or_else(|| config.kernel_regularizer.clone());
            let override_bias_reg = config
                .lhuc_overrides
                .bias_regularizer
                .clone()
                .unwrap_or_else(|| config.bias_regularizer.clone());
            let override_kernel_constraint = config
                .lhuc_overrides
                .kernel_constraint
                .clone()
                .unwrap_or_else(|| config.kernel_constraint.clone());
            let override_bias_constraint = config
                .lhuc_overrides
                .bias_constraint
                .clone()
                .unwrap_or_else(|| config.bias_constraint.clone());

            mlp_config.use_bias = override_bias;
            mlp_config.use_weight_norm = override_weight_norm;
            mlp_config.use_learnable_weight_norm = override_learnable;
            mlp_config.enable_batch_normalization = override_bn;
            mlp_config.batch_normalization_momentum = override_bn_momentum;
            mlp_config.batch_normalization_renorm = override_bn_renorm;
            mlp_config.batch_normalization_renorm_clipping = override_bn_clip;
            mlp_config.batch_normalization_renorm_momentum = override_bn_mom;
            mlp_config = mlp_config
                .with_regularizers(override_kernel_reg, override_bias_reg)
                .with_constraints(override_kernel_constraint, override_bias_constraint);
            let mut lhuc_acts = Vec::new();
            for i in 0..dims.len() {
                if i + 1 == dims.len() {
                    lhuc_acts.push(ActivationType::Sigmoid2);
                } else {
                    lhuc_acts.push(ActivationType::relu());
                }
            }

            let lhuc_inits = if let Some(inits) = &config.lhuc_overrides.initializers {
                if inits.len() == 1 {
                    vec![inits[0]; dims.len()]
                } else if inits.len() == dims.len() {
                    inits.clone()
                } else {
                    return Err(LayerError::ConfigError {
                        message: "lhuc initializers length must be 1 or match lhuc dims"
                            .to_string(),
                    });
                }
            } else {
                vec![initializers[idx]; dims.len()]
            };

            for ((&d, act), init) in dims.iter().zip(lhuc_acts).zip(lhuc_inits) {
                mlp_config = mlp_config.add_layer_with_initializer(d, act, init);
            }

            let mlp = mlp_config.build()?;
            lhuc_layers.push(mlp);
        }

        Ok(Self {
            config,
            input_batch_norm,
            layers,
            lhuc_layers,
            cached_outputs: Vec::new(),
            cached_gates: Vec::new(),
            cached_lhuc_input: None,
        })
    }

    /// Forward pass with optional separate LHUC input.
    pub fn forward_with_inputs(
        &mut self,
        dense_input: &Tensor,
        lhuc_input: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        let lhuc_input = lhuc_input.unwrap_or(dense_input).clone();
        self.cached_outputs.clear();
        self.cached_gates.clear();
        self.cached_lhuc_input = Some(lhuc_input.clone());

        let mut input_t = dense_input.clone();
        if let Some(bn) = &mut self.input_batch_norm {
            input_t = bn.forward_train(&input_t)?;
        }

        for (layer, lhuc_layer) in self.layers.iter_mut().zip(self.lhuc_layers.iter_mut()) {
            let layer_out = layer.forward_train(&input_t)?;
            let gate = lhuc_layer.forward_train(&lhuc_input)?;
            self.cached_outputs.push(layer_out.clone());
            self.cached_gates.push(gate.clone());
            input_t = layer_out.mul(&gate);
        }

        Ok(input_t)
    }

    /// Backward pass returning gradients for dense input and LHUC input.
    pub fn backward_with_inputs(&mut self, grad: &Tensor) -> Result<(Tensor, Tensor), LayerError> {
        let lhuc_input = self
            .cached_lhuc_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let mut grad_dense = grad.clone();
        let mut grad_lhuc = Tensor::zeros(lhuc_input.shape());

        for i in (0..self.layers.len()).rev() {
            let layer_out = &self.cached_outputs[i];
            let gate = &self.cached_gates[i];
            let grad_layer = grad_dense.mul(gate);
            let grad_gate = grad_dense.mul(layer_out);

            grad_dense = self.layers[i].backward(&grad_layer)?;
            let grad_l = self.lhuc_layers[i].backward(&grad_gate)?;
            grad_lhuc = grad_lhuc.add(&grad_l);
        }

        if let Some(bn) = &mut self.input_batch_norm {
            grad_dense = bn.backward(&grad_dense)?;
        }

        Ok((grad_dense, grad_lhuc))
    }
}

impl Layer for LHUCTower {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut layer = self.clone();
        layer.forward_with_inputs(input, None)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let (grad_dense, grad_lhuc) = self.backward_with_inputs(grad)?;
        Ok(grad_dense.add(&grad_lhuc))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.dense.parameters());
        }
        for mlp in &self.lhuc_layers {
            params.extend(mlp.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.dense.parameters_mut());
        }
        for mlp in &mut self.lhuc_layers {
            params.extend(mlp.parameters_mut());
        }
        params
    }

    fn name(&self) -> &str {
        "LHUCTower"
    }

    fn regularization_loss(&self) -> f32 {
        let mut loss = 0.0;
        for layer in &self.layers {
            loss += layer.dense.regularization_loss();
        }
        for mlp in &self.lhuc_layers {
            loss += mlp.regularization_loss();
        }
        loss
    }

    fn apply_constraints(&mut self) {
        for layer in &mut self.layers {
            layer.dense.apply_constraints();
        }
        for mlp in &mut self.lhuc_layers {
            mlp.apply_constraints();
        }
    }
}

fn build_lhuc_output_dims(config: &LHUCConfig) -> Result<Vec<Vec<usize>>, LayerError> {
    let mut out = Vec::new();
    match &config.lhuc_output_dims {
        None => {
            for &dim in &config.output_dims {
                out.push(vec![dim]);
            }
        }
        Some(LHUCOutputDims::Shared(shared)) => {
            for &dim in &config.output_dims {
                let mut dims = shared.clone();
                dims.push(dim);
                out.push(dims);
            }
        }
        Some(LHUCOutputDims::PerLayer(per_layer)) => {
            if per_layer.len() != config.output_dims.len() {
                return Err(LayerError::ConfigError {
                    message: "lhuc_output_dims length must match output_dims".to_string(),
                });
            }
            for (dims, &dim) in per_layer.iter().zip(config.output_dims.iter()) {
                if *dims.last().unwrap_or(&0) != dim {
                    return Err(LayerError::ConfigError {
                        message: "lhuc_output_dims last dim must match layer output".to_string(),
                    });
                }
                out.push(dims.clone());
            }
        }
    }
    Ok(out)
}
