//! Activation function layers.
//!
//! This module provides common activation functions as neural network layers,
//! including ReLU, Sigmoid, Tanh, and GELU.

use crate::constraint::Constraint;
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Rectified Linear Unit (ReLU) activation function.
///
/// Computes `f(x) = max(0, x)` element-wise.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::ReLU;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let relu = ReLU::new();
/// let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
/// let output = relu.forward(&input).unwrap();
/// assert_eq!(output.data(), &[0.0, 0.0, 1.0, 2.0]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReLU {
    max_value: Option<f32>,
    negative_slope: f32,
    threshold: f32,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl ReLU {
    /// Creates a new ReLU activation layer.
    pub fn new() -> Self {
        Self {
            max_value: None,
            negative_slope: 0.0,
            threshold: 0.0,
            cached_input: None,
        }
    }

    /// Creates a ReLU with custom parameters.
    pub fn with_params(max_value: Option<f32>, negative_slope: f32, threshold: f32) -> Self {
        Self {
            max_value,
            negative_slope,
            threshold,
            cached_input: None,
        }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let threshold = Tensor::from_data(&[1], vec![self.threshold]);
        let shifted = input.sub(&threshold);
        // Keras ReLU uses x > threshold for the positive branch (x == threshold is zeroed).
        let pos_mask = input.gt_scalar(self.threshold);
        let ones = Tensor::ones(input.shape());
        let neg_mask = ones.sub(&pos_mask);
        let pos_out = input.clone();
        let neg_out = shifted.scale(self.negative_slope);
        let mut out = pos_out.mul(&pos_mask).add(&neg_out.mul(&neg_mask));

        if let Some(max_value) = self.max_value {
            out = out.clamp(-1.0e20, max_value);
        }

        Ok(out)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Match forward predicate (`>`), so threshold values have zero gradient.
        let pos_mask = input.gt_scalar(self.threshold);
        let ones = Tensor::ones(input.shape());
        let neg_mask = ones.sub(&pos_mask);
        let mut grad_multiplier = pos_mask.add(&neg_mask.scale(self.negative_slope));

        if let Some(max_value) = self.max_value {
            let max_mask = input.lt_scalar(max_value);
            grad_multiplier = grad_multiplier.mul(&max_mask);
        }

        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![] // No learnable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "ReLU"
    }
}

/// Sigmoid activation function.
///
/// Computes `f(x) = 1 / (1 + exp(-x))` element-wise.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::Sigmoid;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let sigmoid = Sigmoid::new();
/// let input = Tensor::zeros(&[2, 2]);
/// let output = sigmoid.forward(&input).unwrap();
/// // sigmoid(0) = 0.5
/// assert!((output.data()[0] - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Sigmoid {
    /// Cached output for backward pass (more efficient than caching input)
    cached_output: Option<Tensor>,
}

impl Sigmoid {
    /// Creates a new Sigmoid activation layer.
    pub fn new() -> Self {
        Self {
            cached_output: None,
        }
    }

    /// Performs forward pass and caches output for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let output = self.forward(input)?;
        self.cached_output = Some(output.clone());
        Ok(output)
    }
}

impl Layer for Sigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.sigmoid())
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let output = self
            .cached_output
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x)) = output * (1 - output)
        let grad_multiplier = output.map(|y| y * (1.0 - y));
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Sigmoid"
    }
}

/// Sigmoid2 activation function (2 * sigmoid(x)).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Sigmoid2 {
    cached_output: Option<Tensor>,
}

impl Sigmoid2 {
    pub fn new() -> Self {
        Self {
            cached_output: None,
        }
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let output = self.forward(input)?;
        self.cached_output = Some(output.clone());
        Ok(output)
    }
}

impl Layer for Sigmoid2 {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.sigmoid().scale(2.0))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let output = self
            .cached_output
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // y = 2 * sigmoid(x); dy/dx = 2 * s * (1 - s)
        let half = output.scale(0.5);
        let ones = Tensor::ones(output.shape());
        let grad_multiplier = half.mul(&ones.sub(&half)).scale(2.0);
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Sigmoid2"
    }
}

/// Hyperbolic tangent (Tanh) activation function.
///
/// Computes `f(x) = tanh(x)` element-wise.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::Tanh;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let tanh = Tanh::new();
/// let input = Tensor::zeros(&[2, 2]);
/// let output = tanh.forward(&input).unwrap();
/// // tanh(0) = 0
/// assert!(output.data()[0].abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Tanh {
    /// Cached output for backward pass
    cached_output: Option<Tensor>,
}

impl Tanh {
    /// Creates a new Tanh activation layer.
    pub fn new() -> Self {
        Self {
            cached_output: None,
        }
    }

    /// Performs forward pass and caches output for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let output = self.forward(input)?;
        self.cached_output = Some(output.clone());
        Ok(output)
    }
}

impl Layer for Tanh {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.tanh())
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let output = self
            .cached_output
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Tanh gradient: 1 - tanh(x)^2 = 1 - output^2
        let grad_multiplier = output.map(|y| 1.0 - y * y);
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Tanh"
    }
}

/// Softsign activation function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Softsign {
    cached_input: Option<Tensor>,
}

impl Softsign {
    pub fn new() -> Self {
        Self { cached_input: None }
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for Softsign {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let abs = input.abs();
        let ones = Tensor::ones(input.shape());
        let denom = abs.add(&ones);
        Ok(input.div(&denom))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let abs = input.abs();
        let ones = Tensor::ones(input.shape());
        let denom = abs.add(&ones);
        let grad_multiplier = denom.sqr().recip();
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Softsign"
    }
}

/// Linear activation function (identity).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Linear;

impl Linear {
    pub fn new() -> Self {
        Self
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.clone())
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.clone())
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        Ok(grad.clone())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Linear"
    }
}

/// Exponential activation function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Exponential {
    cached_output: Option<Tensor>,
}

impl Exponential {
    pub fn new() -> Self {
        Self {
            cached_output: None,
        }
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let output = self.forward(input)?;
        self.cached_output = Some(output.clone());
        Ok(output)
    }
}

impl Layer for Exponential {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.exp())
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let output = self
            .cached_output
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        Ok(grad.mul(output))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Exponential"
    }
}

/// Thresholded ReLU activation function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdedReLU {
    theta: f32,
    cached_input: Option<Tensor>,
}

impl ThresholdedReLU {
    pub fn new(theta: f32) -> Self {
        Self {
            theta,
            cached_input: None,
        }
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Default for ThresholdedReLU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Layer for ThresholdedReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mask = input.gt_scalar(self.theta);
        Ok(input.mul(&mask))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let mask = input.gt_scalar(self.theta);
        Ok(grad.mul(&mask))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "ThresholdedReLU"
    }
}

/// Gaussian Error Linear Unit (GELU) activation function.
///
/// Computes `f(x) = x * Phi(x)` where Phi is the CDF of the standard normal distribution.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::GELU;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let gelu = GELU::new();
/// let input = Tensor::zeros(&[2, 2]);
/// let output = gelu.forward(&input).unwrap();
/// // GELU(0) = 0
/// assert!(output.data()[0].abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GELU {
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl GELU {
    /// Creates a new GELU activation layer.
    pub fn new() -> Self {
        Self { cached_input: None }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.gelu())
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Exact derivative: d/dx gelu(x) = Phi(x) + x * phi(x)
        // Phi(x) = gelu(x) / x (limit 0.5 at x=0)
        // phi(x) = exp(-x^2/2) / sqrt(2*pi)
        let gelu_out = input.gelu();
        let phi = input.sqr().scale(-0.5).exp().scale(0.3989422804014327_f32); // 1/sqrt(2π)

        let abs = input.abs();
        let small = abs.lt_scalar(1e-6);
        let ones = Tensor::ones(input.shape());
        let safe = input.add(&small.scale(1e-6));
        let mut phi_cdf = gelu_out.div(&safe);
        phi_cdf = phi_cdf.mul(&ones.sub(&small)).add(&small.scale(0.5));

        let grad_multiplier = phi_cdf.add(&input.mul(&phi));
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "GELU"
    }
}

/// Leaky ReLU activation function.
///
/// Computes `f(x) = max(alpha * x, x)` element-wise.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakyReLU {
    /// Negative slope (default: 0.01)
    alpha: f32,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl LeakyReLU {
    /// Creates a new LeakyReLU with the specified negative slope.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            cached_input: None,
        }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Layer for LeakyReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let relu = input.relu();
        let neg = input.sub(&relu);
        Ok(relu.add(&neg.scale(self.alpha)))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let pos_mask = input.gt_scalar(0.0);
        let ones = Tensor::ones(input.shape());
        let neg_mask = ones.sub(&pos_mask);
        let mask = pos_mask.add(&neg_mask.scale(self.alpha));
        Ok(grad.mul(&mask))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "LeakyReLU"
    }
}

/// Exponential Linear Unit (ELU) activation function.
///
/// Computes `f(x) = x` if x > 0, else `alpha * (exp(x) - 1)`.
///
/// ELU helps mitigate the vanishing gradient problem and produces negative outputs
/// which can help push mean unit activations closer to zero.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::ELU;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let elu = ELU::new(1.0);
/// let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
/// let output = elu.forward(&input).unwrap();
/// // Positive values pass through, negative values are transformed
/// assert!((output.data()[2] - 1.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ELU {
    /// Alpha parameter controlling the saturation value for negative inputs
    alpha: f32,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl ELU {
    /// Creates a new ELU activation layer with the specified alpha.
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            cached_input: None,
        }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Layer for ELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let relu = input.relu();
        let neg = input.sub(&relu);
        let ones = Tensor::ones(input.shape());
        let neg_out = neg.exp().sub(&ones).scale(self.alpha);
        Ok(relu.add(&neg_out))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // ELU gradient: 1 if x > 0, else alpha * exp(x) = output + alpha for x <= 0
        let pos_mask = input.gt_scalar(0.0);
        let ones = Tensor::ones(input.shape());
        let neg_mask = ones.sub(&pos_mask);
        let neg = input.exp().scale(self.alpha);
        let grad_multiplier = pos_mask.add(&neg.mul(&neg_mask));
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "ELU"
    }
}

/// Parametric ReLU (PReLU) activation function.
///
/// Computes `f(x) = x` if x > 0, else `alpha * x`, where alpha is a learnable parameter.
///
/// Unlike LeakyReLU, PReLU learns the optimal slope for negative inputs during training.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::PReLU;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let prelu = PReLU::new(0.25);
/// let input = Tensor::from_data(&[2, 2], vec![-2.0, -1.0, 1.0, 2.0]);
/// let output = prelu.forward(&input).unwrap();
/// assert!((output.data()[0] - (-0.5)).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PReLU {
    /// Learnable alpha parameter (negative slope)
    alpha: Tensor,
    /// Gradient of alpha
    alpha_grad: Option<Tensor>,
    /// Alpha initializer
    alpha_initializer: Initializer,
    /// Alpha regularizer
    alpha_regularizer: Regularizer,
    /// Alpha constraint
    alpha_constraint: Constraint,
    /// Shared axes for alpha (input axes excluding batch)
    shared_axes: Vec<isize>,
    /// Whether alpha has been built for input shape
    built: bool,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl PReLU {
    /// Creates a new PReLU activation layer with the specified initial alpha.
    pub fn new(initial_alpha: f32) -> Self {
        Self::with_params(
            initial_alpha,
            None,
            None,
            Some(Regularizer::None),
            Some(Constraint::None),
        )
    }

    /// Creates a new PReLU with optional initializer/shared_axes/regularizer/constraint.
    pub fn with_params(
        initial_alpha: f32,
        initializer: Option<Initializer>,
        shared_axes: Option<Vec<isize>>,
        regularizer: Option<Regularizer>,
        constraint: Option<Constraint>,
    ) -> Self {
        Self {
            alpha: Tensor::from_data(&[1], vec![initial_alpha]),
            alpha_grad: None,
            alpha_initializer: initializer.unwrap_or(Initializer::Constant(initial_alpha)),
            alpha_regularizer: regularizer.unwrap_or(Regularizer::None),
            alpha_constraint: constraint.unwrap_or(Constraint::None),
            shared_axes: shared_axes.unwrap_or_default(),
            built: false,
            cached_input: None,
        }
    }

    fn build_if_needed(&mut self, input: &Tensor) -> Result<(), LayerError> {
        if self.built {
            return Ok(());
        }
        if input.ndim() < 2 {
            return Err(LayerError::ForwardError {
                message: format!("PReLU expects >=2D input, got {}D", input.ndim()),
            });
        }
        let ndim = input.ndim();
        let mut shared = Vec::new();
        for &axis in &self.shared_axes {
            let ax = if axis < 0 {
                (ndim as isize + axis) as usize
            } else {
                axis as usize
            };
            if ax != 0 && ax < ndim {
                shared.push(ax);
            }
        }

        let mut alpha_shape = Vec::with_capacity(ndim - 1);
        for axis in 1..ndim {
            if shared.contains(&axis) {
                alpha_shape.push(1);
            } else {
                alpha_shape.push(input.shape()[axis]);
            }
        }
        self.alpha = self.alpha_initializer.initialize(&alpha_shape);
        self.built = true;
        Ok(())
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.build_if_needed(input)?;
        self.cached_input = Some(input.clone());
        self.forward(input)
    }

    /// Returns the current alpha value.
    pub fn alpha(&self) -> f32 {
        self.alpha.data_ref()[0]
    }

    /// Returns the alpha gradient if computed.
    pub fn alpha_grad(&self) -> Option<&Tensor> {
        self.alpha_grad.as_ref()
    }
}

impl Default for PReLU {
    fn default() -> Self {
        Self::new(0.25)
    }
}

impl Layer for PReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let relu = input.relu();
        let neg = input.sub(&relu);
        let mut shape = self.alpha.shape().to_vec();
        shape.insert(0, 1);
        let alpha_b = self.alpha.reshape(&shape).broadcast_as(input.shape());
        Ok(relu.add(&neg.mul(&alpha_b)))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .clone()
            .ok_or(LayerError::NotInitialized)?;

        self.build_if_needed(&input)?;

        let relu = input.relu();
        let neg = input.sub(&relu);
        let mut shape = self.alpha.shape().to_vec();
        shape.insert(0, 1);
        let alpha_b = self.alpha.reshape(&shape).broadcast_as(input.shape());

        let pos_mask = input.gt_scalar(0.0);
        let ones = Tensor::ones(input.shape());
        let neg_mask = ones.sub(&pos_mask);
        let mask = pos_mask.add(&neg_mask.mul(&alpha_b));
        let input_grad = grad.mul(&mask);

        let mut reduce_axes = Vec::new();
        for axis in 0..input.ndim() {
            if axis == 0 {
                reduce_axes.push(axis);
                continue;
            }
            let alpha_axis = axis - 1;
            if self.alpha.shape()[alpha_axis] == 1 {
                reduce_axes.push(axis);
            }
        }

        let mut alpha_grad = grad.mul(&neg);
        if !reduce_axes.is_empty() {
            alpha_grad = alpha_grad.sum_axes(&reduce_axes);
        }

        if let Some(reg_grad) = self.alpha_regularizer.grad(&self.alpha) {
            alpha_grad = alpha_grad.add(&reg_grad);
        }
        self.alpha_grad = Some(alpha_grad);

        Ok(input_grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.alpha]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.alpha]
    }

    fn name(&self) -> &str {
        "PReLU"
    }

    fn regularization_loss(&self) -> f32 {
        self.alpha_regularizer.loss(&self.alpha)
    }

    fn apply_constraints(&mut self) {
        self.alpha = self.alpha_constraint.apply(&self.alpha);
    }
}

/// Scaled Exponential Linear Unit (SELU) activation function.
///
/// Computes `f(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))`.
///
/// SELU uses fixed alpha and scale values that enable self-normalizing neural networks,
/// where the activations automatically converge to zero mean and unit variance.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::SELU;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let selu = SELU::new();
/// let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
/// let output = selu.forward(&input).unwrap();
/// // Positive values are scaled
/// assert!((output.data()[2] - 1.0507).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SELU {
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl SELU {
    /// SELU alpha constant for self-normalizing properties
    const ALPHA: f32 = 1.673_263_2;
    /// SELU scale constant for self-normalizing properties
    const SCALE: f32 = 1.050_701;

    /// Creates a new SELU activation layer.
    pub fn new() -> Self {
        Self { cached_input: None }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for SELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let relu = input.relu();
        let neg = input.sub(&relu);
        let ones = Tensor::ones(input.shape());
        let neg_out = neg.exp().sub(&ones).scale(Self::ALPHA);
        Ok(relu.add(&neg_out).scale(Self::SCALE))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // SELU gradient: scale if x > 0, else scale * alpha * exp(x)
        let pos_mask = input.gt_scalar(0.0);
        let ones = Tensor::ones(input.shape());
        let neg_mask = ones.sub(&pos_mask);
        let pos = pos_mask.scale(Self::SCALE);
        let neg = input.exp().scale(Self::SCALE * Self::ALPHA);
        let grad_multiplier = pos.add(&neg.mul(&neg_mask));
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "SELU"
    }
}

/// Swish activation function.
///
/// Computes `f(x) = x * sigmoid(x)`.
///
/// Swish (also known as SiLU) is a smooth, non-monotonic activation function that
/// has been shown to outperform ReLU on deeper models.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::Swish;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let swish = Swish::new();
/// let input = Tensor::zeros(&[2, 2]);
/// let output = swish.forward(&input).unwrap();
/// // swish(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
/// assert!(output.data()[0].abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Swish {
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl Swish {
    /// Creates a new Swish activation layer.
    pub fn new() -> Self {
        Self { cached_input: None }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for Swish {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.mul(&input.sigmoid()))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Swish gradient: swish(x) + sigmoid(x) * (1 - swish(x))
        // = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let sig = input.sigmoid();
        let ones = Tensor::ones(input.shape());
        let one_minus_sig = ones.sub(&sig);
        let grad_multiplier = sig.add(&input.mul(&sig).mul(&one_minus_sig));
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Swish"
    }
}

/// Softplus activation function.
///
/// Computes `f(x) = log(1 + exp(x))`.
///
/// Softplus is a smooth approximation of the ReLU function, differentiable everywhere
/// and always positive.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::Softplus;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let softplus = Softplus::new();
/// let input = Tensor::zeros(&[2, 2]);
/// let output = softplus.forward(&input).unwrap();
/// // softplus(0) = log(1 + 1) = log(2) ≈ 0.693
/// assert!((output.data()[0] - 0.693).abs() < 0.01);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Softplus {
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Threshold for numerical stability (default: 20.0)
    threshold: f32,
}

impl Softplus {
    /// Creates a new Softplus activation layer.
    pub fn new() -> Self {
        Self {
            cached_input: None,
            threshold: 20.0,
        }
    }

    /// Creates a new Softplus with a custom threshold.
    /// For x > threshold, returns x directly for numerical stability.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            cached_input: None,
            threshold,
        }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for Softplus {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let relu = input.relu();
        let abs = input.abs();
        let ones = Tensor::ones(input.shape());
        let log1p = abs.scale(-1.0).exp().add(&ones).log();
        Ok(relu.add(&log1p))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let threshold = self.threshold;
        // Softplus gradient: sigmoid(x) = 1 / (1 + exp(-x))
        let sig = input.sigmoid();
        let ones = Tensor::ones(input.shape());
        let mask = input.gt_scalar(threshold);
        let inv = ones.sub(&mask);
        let grad_multiplier = mask.add(&sig.mul(&inv));
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Softplus"
    }
}

/// Hard Sigmoid activation function.
///
/// Computes `f(x) = clip(0.2 * x + 0.5, 0, 1)`.
///
/// HardSigmoid is a piecewise linear approximation of the sigmoid function that
/// is computationally cheaper while maintaining similar properties.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::HardSigmoid;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let hard_sigmoid = HardSigmoid::new();
/// let input = Tensor::zeros(&[2, 2]);
/// let output = hard_sigmoid.forward(&input).unwrap();
/// // hard_sigmoid(0) = 0.2 * 0 + 0.5 = 0.5
/// assert!((output.data()[0] - 0.5).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HardSigmoid {
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl HardSigmoid {
    /// Creates a new HardSigmoid activation layer.
    pub fn new() -> Self {
        Self { cached_input: None }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for HardSigmoid {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let scaled = input.scale(0.2).add(&Tensor::from_data(&[1], vec![0.5]));
        Ok(scaled.clamp(0.0, 1.0))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // HardSigmoid gradient: 0.2 if -2.5 <= x <= 2.5, else 0
        let ge_low = input.ge_scalar(-2.5);
        let le_high = input.le_scalar(2.5);
        let grad_multiplier = ge_low.mul(&le_high).scale(0.2);
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "HardSigmoid"
    }
}

/// Mish activation function.
///
/// Computes `f(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))`.
///
/// Mish is a self-regularizing, non-monotonic activation function that has shown
/// strong performance in various deep learning tasks.
///
/// # Example
///
/// ```
/// use monolith_layers::activation::Mish;
/// use monolith_layers::layer::Layer;
/// use monolith_layers::tensor::Tensor;
///
/// let mish = Mish::new();
/// let input = Tensor::zeros(&[2, 2]);
/// let output = mish.forward(&input).unwrap();
/// // mish(0) = 0 * tanh(log(2)) ≈ 0
/// assert!(output.data()[0].abs() < 1e-6);
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Mish {
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl Mish {
    /// Creates a new Mish activation layer.
    pub fn new() -> Self {
        Self { cached_input: None }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }

    /// Computes softplus(x) = log(1 + exp(x)) with numerical stability
    #[inline]
    fn softplus_tensor(input: &Tensor) -> Tensor {
        let relu = input.relu();
        let abs = input.abs();
        let ones = Tensor::ones(input.shape());
        let log1p = abs.scale(-1.0).exp().add(&ones).log();
        relu.add(&log1p)
    }
}

impl Layer for Mish {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let sp = Self::softplus_tensor(input);
        Ok(input.mul(&sp.tanh()))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Mish gradient: complex derivative
        // Let sp = softplus(x), sig = sigmoid(x)
        // d/dx mish(x) = tanh(sp) + x * sech^2(sp) * sig
        let sp = Self::softplus_tensor(input);
        let tanh_sp = sp.tanh();
        let ones = Tensor::ones(input.shape());
        let sech2 = ones.sub(&tanh_sp.sqr());
        let sig = input.sigmoid();
        let grad_multiplier = tanh_sp.add(&input.mul(&sech2).mul(&sig));
        Ok(grad.mul(&grad_multiplier))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Mish"
    }
}

/// Softmax activation function.
///
/// Computes softmax along the last dimension.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Softmax {
    /// Cached output for backward pass
    cached_output: Option<Tensor>,
    axis: isize,
}

impl Softmax {
    /// Creates a new Softmax activation layer.
    pub fn new() -> Self {
        Self {
            cached_output: None,
            axis: -1,
        }
    }

    pub fn with_axis(axis: isize) -> Self {
        Self {
            cached_output: None,
            axis,
        }
    }

    /// Performs forward pass and caches output for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        let output = self.forward(input)?;
        self.cached_output = Some(output.clone());
        Ok(output)
    }
}

impl Layer for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let ndim = input.ndim();
        if ndim == 0 {
            return Err(LayerError::ForwardError {
                message: "Softmax expects at least 1D input".to_string(),
            });
        }
        let axis = if self.axis < 0 {
            (ndim as isize + self.axis) as usize
        } else {
            self.axis as usize
        };
        Ok(input.softmax(axis))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let output = self
            .cached_output
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Softmax backward: s_i * (grad_i - sum_j(s_j * grad_j))
        let ndim = output.ndim();
        let axis = if self.axis < 0 {
            (ndim as isize + self.axis) as usize
        } else {
            self.axis as usize
        };
        let dot = output.mul(grad).sum_axis(axis);
        let mut dot_shape = dot.shape().to_vec();
        dot_shape.insert(axis, 1);
        let dot_b = dot.reshape(&dot_shape).broadcast_as(output.shape());
        Ok(output.mul(&grad.sub(&dot_b)))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "Softmax"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relu_forward() {
        let relu = ReLU::new();
        let input = Tensor::from_data(&[2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let output = relu.forward(&input).unwrap();
        assert_eq!(output.data(), &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_relu_backward() {
        let mut relu = ReLU::new();
        let input = Tensor::from_data(&[2, 3], vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let _output = relu.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 3]);
        let input_grad = relu.backward(&grad).unwrap();
        assert_eq!(input_grad.data(), &[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Sigmoid::new();
        let input = Tensor::zeros(&[2, 2]);
        let output = sigmoid.forward(&input).unwrap();

        for val in output.data() {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh::new();
        let input = Tensor::zeros(&[2, 2]);
        let output = tanh.forward(&input).unwrap();

        for val in output.data() {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_gelu_forward() {
        let gelu = GELU::new();
        let input = Tensor::zeros(&[2, 2]);
        let output = gelu.forward(&input).unwrap();

        for val in output.data() {
            assert!(val.abs() < 1e-6);
        }

        // GELU should be approximately identity for large positive values
        let input = Tensor::from_data(&[1, 1], vec![3.0]);
        let output = gelu.forward(&input).unwrap();
        assert!((output.data()[0] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_leaky_relu_forward() {
        let leaky_relu = LeakyReLU::new(0.1);
        let input = Tensor::from_data(&[2, 2], vec![-2.0, -1.0, 1.0, 2.0]);
        let output = leaky_relu.forward(&input).unwrap();
        assert_eq!(output.data(), &[-0.2, -0.1, 1.0, 2.0]);
    }

    #[test]
    fn test_softmax_forward() {
        let softmax = Softmax::new();
        let input = Tensor::from_data(&[1, 3], vec![1.0, 2.0, 3.0]);
        let output = softmax.forward(&input).unwrap();

        // Output should sum to 1
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Largest input should have largest output
        assert!(output.data()[2] > output.data()[1]);
        assert!(output.data()[1] > output.data()[0]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let softmax = Softmax::new();
        // Large values that could cause overflow without proper handling
        let input = Tensor::from_data(&[1, 3], vec![1000.0, 1001.0, 1002.0]);
        let output = softmax.forward(&input).unwrap();

        // Should still be valid probabilities
        let sum: f32 = output.data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for val in output.data() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_activation_no_parameters() {
        let relu = ReLU::new();
        assert!(relu.parameters().is_empty());

        let sigmoid = Sigmoid::new();
        assert!(sigmoid.parameters().is_empty());

        let tanh = Tanh::new();
        assert!(tanh.parameters().is_empty());

        let gelu = GELU::new();
        assert!(gelu.parameters().is_empty());
    }

    #[test]
    fn test_elu_forward() {
        let elu = ELU::new(1.0);
        let input = Tensor::from_data(&[2, 2], vec![-2.0, 0.0, 1.0, 2.0]);
        let output = elu.forward(&input).unwrap();

        // Positive values pass through
        assert!((output.data()[2] - 1.0).abs() < 1e-6);
        assert!((output.data()[3] - 2.0).abs() < 1e-6);

        // Zero stays zero
        assert!((output.data()[1]).abs() < 1e-6);

        // Negative values: alpha * (exp(x) - 1)
        let expected_neg = 1.0 * ((-2.0_f32).exp() - 1.0);
        assert!((output.data()[0] - expected_neg).abs() < 1e-6);
    }

    #[test]
    fn test_elu_backward() {
        let mut elu = ELU::new(1.0);
        let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
        let _output = elu.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 2]);
        let input_grad = elu.backward(&grad).unwrap();

        // For x > 0, gradient is 1
        assert!((input_grad.data()[2] - 1.0).abs() < 1e-6);
        assert!((input_grad.data()[3] - 1.0).abs() < 1e-6);

        // For x <= 0, gradient is alpha * exp(x)
        let expected_grad_neg = 1.0 * (-1.0_f32).exp();
        assert!((input_grad.data()[0] - expected_grad_neg).abs() < 1e-6);
    }

    #[test]
    fn test_prelu_forward() {
        let prelu = PReLU::new(0.25);
        let input = Tensor::from_data(&[2, 2], vec![-2.0, -1.0, 1.0, 2.0]);
        let output = prelu.forward(&input).unwrap();

        assert!((output.data()[0] - (-0.5)).abs() < 1e-6);
        assert!((output.data()[1] - (-0.25)).abs() < 1e-6);
        assert!((output.data()[2] - 1.0).abs() < 1e-6);
        assert!((output.data()[3] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_prelu_backward() {
        let mut prelu = PReLU::new(0.25);
        let input = Tensor::from_data(&[2, 2], vec![-2.0, -1.0, 1.0, 2.0]);
        let _output = prelu.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 2]);
        let input_grad = prelu.backward(&grad).unwrap();

        // For x > 0, gradient is 1
        assert!((input_grad.data()[2] - 1.0).abs() < 1e-6);
        assert!((input_grad.data()[3] - 1.0).abs() < 1e-6);

        // For x <= 0, gradient is alpha
        assert!((input_grad.data()[0] - 0.25).abs() < 1e-6);
        assert!((input_grad.data()[1] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_prelu_has_learnable_parameters() {
        let prelu = PReLU::new(0.25);
        assert_eq!(prelu.parameters().len(), 1);
        assert!((prelu.alpha() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_selu_forward() {
        let selu = SELU::new();
        let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
        let output = selu.forward(&input).unwrap();

        // Positive values are scaled by SCALE
        assert!((output.data()[2] - 1.0507009873554805).abs() < 1e-4);
        assert!((output.data()[3] - 2.1014019747109610).abs() < 1e-4);

        // Zero stays approximately zero
        assert!(output.data()[1].abs() < 1e-6);

        // Negative values: scale * alpha * (exp(x) - 1)
        let expected_neg = 1.0507009873554805 * 1.6732632423543772 * ((-1.0_f32).exp() - 1.0);
        assert!((output.data()[0] - expected_neg).abs() < 1e-4);
    }

    #[test]
    fn test_selu_backward() {
        let mut selu = SELU::new();
        let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
        let _output = selu.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 2]);
        let input_grad = selu.backward(&grad).unwrap();

        // For x > 0, gradient is SCALE
        assert!((input_grad.data()[2] - 1.0507009873554805).abs() < 1e-4);
        assert!((input_grad.data()[3] - 1.0507009873554805).abs() < 1e-4);
    }

    #[test]
    fn test_swish_forward() {
        let swish = Swish::new();
        let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
        let output = swish.forward(&input).unwrap();

        // swish(0) = 0 * sigmoid(0) = 0
        assert!(output.data()[1].abs() < 1e-6);

        // swish(x) = x * sigmoid(x)
        let sigmoid_1 = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!((output.data()[2] - 1.0 * sigmoid_1).abs() < 1e-6);

        // For large positive values, swish approaches x
        let large_input = Tensor::from_data(&[1, 1], vec![10.0]);
        let large_output = swish.forward(&large_input).unwrap();
        assert!((large_output.data()[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_swish_backward() {
        let mut swish = Swish::new();
        let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
        let _output = swish.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 2]);
        let input_grad = swish.backward(&grad).unwrap();

        // At x = 0, gradient = sigmoid(0) + 0 * sigmoid(0) * (1 - sigmoid(0)) = 0.5
        assert!((input_grad.data()[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softplus_forward() {
        let softplus = Softplus::new();
        let input = Tensor::from_data(&[2, 2], vec![-2.0, 0.0, 1.0, 2.0]);
        let output = softplus.forward(&input).unwrap();

        // softplus(0) = log(2) ≈ 0.693
        assert!((output.data()[1] - 0.693).abs() < 0.01);

        // All outputs should be positive
        for val in output.data() {
            assert!(val > 0.0);
        }

        // For large x, softplus(x) ≈ x
        let large_input = Tensor::from_data(&[1, 1], vec![30.0]);
        let large_output = softplus.forward(&large_input).unwrap();
        assert!((large_output.data()[0] - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_softplus_backward() {
        let mut softplus = Softplus::new();
        let input = Tensor::from_data(&[2, 2], vec![-2.0, 0.0, 1.0, 2.0]);
        let _output = softplus.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 2]);
        let input_grad = softplus.backward(&grad).unwrap();

        // Gradient is sigmoid(x)
        // At x = 0, sigmoid(0) = 0.5
        assert!((input_grad.data()[1] - 0.5).abs() < 1e-6);

        // All gradients should be in (0, 1)
        for val in input_grad.data() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_hard_sigmoid_forward() {
        let hard_sigmoid = HardSigmoid::new();
        let input = Tensor::from_data(&[1, 5], vec![-5.0, -2.5, 0.0, 2.5, 5.0]);
        let output = hard_sigmoid.forward(&input).unwrap();

        // x <= -2.5: output = 0
        assert!((output.data()[0]).abs() < 1e-6);
        assert!((output.data()[1]).abs() < 1e-6);

        // x = 0: output = 0.5
        assert!((output.data()[2] - 0.5).abs() < 1e-6);

        // x >= 2.5: output = 1
        assert!((output.data()[3] - 1.0).abs() < 1e-6);
        assert!((output.data()[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hard_sigmoid_backward() {
        let mut hard_sigmoid = HardSigmoid::new();
        let input = Tensor::from_data(&[1, 5], vec![-5.0, -2.0, 0.0, 2.0, 5.0]);
        let _output = hard_sigmoid.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[1, 5]);
        let input_grad = hard_sigmoid.backward(&grad).unwrap();

        // Gradient is 0.2 for -2.5 <= x <= 2.5, else 0
        assert!((input_grad.data()[0]).abs() < 1e-6); // x = -5
        assert!((input_grad.data()[1] - 0.2).abs() < 1e-6); // x = -2
        assert!((input_grad.data()[2] - 0.2).abs() < 1e-6); // x = 0
        assert!((input_grad.data()[3] - 0.2).abs() < 1e-6); // x = 2
        assert!((input_grad.data()[4]).abs() < 1e-6); // x = 5
    }

    #[test]
    fn test_mish_forward() {
        let mish = Mish::new();
        let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
        let output = mish.forward(&input).unwrap();

        // mish(0) = 0 * tanh(log(2)) = 0
        assert!(output.data()[1].abs() < 1e-6);

        // mish(x) = x * tanh(softplus(x))
        let sp_1 = (1.0 + 1.0_f32.exp()).ln();
        let expected_1 = 1.0 * sp_1.tanh();
        assert!((output.data()[2] - expected_1).abs() < 1e-4);

        // For large positive x, mish approaches x
        let large_input = Tensor::from_data(&[1, 1], vec![10.0]);
        let large_output = mish.forward(&large_input).unwrap();
        assert!((large_output.data()[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_mish_backward() {
        let mut mish = Mish::new();
        let input = Tensor::from_data(&[2, 2], vec![-1.0, 0.0, 1.0, 2.0]);
        let _output = mish.forward_train(&input).unwrap();

        let grad = Tensor::ones(&[2, 2]);
        let input_grad = mish.backward(&grad).unwrap();

        // Gradient at x = 0
        // sp = log(2), tanh_sp = tanh(log(2)), sech2_sp = 1 - tanh(log(2))^2
        // sig = 0.5
        // grad = tanh(sp) + 0 * sech2(sp) * sig = tanh(log(2))
        let sp_0 = (2.0_f32).ln();
        let expected_grad_0 = sp_0.tanh();
        assert!((input_grad.data()[1] - expected_grad_0).abs() < 1e-4);
    }

    #[test]
    fn test_new_activations_no_parameters() {
        // ELU, SELU, Swish, Softplus, HardSigmoid, Mish should have no learnable parameters
        let elu = ELU::new(1.0);
        assert!(elu.parameters().is_empty());

        let selu = SELU::new();
        assert!(selu.parameters().is_empty());

        let swish = Swish::new();
        assert!(swish.parameters().is_empty());

        let softplus = Softplus::new();
        assert!(softplus.parameters().is_empty());

        let hard_sigmoid = HardSigmoid::new();
        assert!(hard_sigmoid.parameters().is_empty());

        let mish = Mish::new();
        assert!(mish.parameters().is_empty());

        // PReLU has learnable parameters
        let prelu = PReLU::new(0.25);
        assert_eq!(prelu.parameters().len(), 1);
    }

    #[test]
    fn test_activation_names() {
        assert_eq!(ELU::new(1.0).name(), "ELU");
        assert_eq!(PReLU::new(0.25).name(), "PReLU");
        assert_eq!(SELU::new().name(), "SELU");
        assert_eq!(Swish::new().name(), "Swish");
        assert_eq!(Softplus::new().name(), "Softplus");
        assert_eq!(HardSigmoid::new().name(), "HardSigmoid");
        assert_eq!(Mish::new().name(), "Mish");
    }
}
