//! Activation function layers.
//!
//! This module provides common activation functions as neural network layers,
//! including ReLU, Sigmoid, Tanh, and GELU.

use crate::error::LayerError;
use crate::layer::Layer;
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
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReLU {
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl ReLU {
    /// Creates a new ReLU activation layer.
    pub fn new() -> Self {
        Self { cached_input: None }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }
}

impl Layer for ReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.map(|x| x.max(0.0)))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // ReLU gradient: 1 if x > 0, else 0
        let mask = input.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        Ok(grad.mul(&mask))
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
        Ok(input.map(|x| 1.0 / (1.0 + (-x).exp())))
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
        Ok(input.map(|x| x.tanh()))
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

/// Gaussian Error Linear Unit (GELU) activation function.
///
/// Computes `f(x) = x * Phi(x)` where Phi is the CDF of the standard normal distribution.
/// We use the approximation: `GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
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

    /// GELU approximation constant
    const SQRT_2_OVER_PI: f32 = 0.797_884_6; // sqrt(2/pi)
    const GELU_COEF: f32 = 0.044715;
}

impl Layer for GELU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.map(|x| {
            let inner = Self::SQRT_2_OVER_PI * (x + Self::GELU_COEF * x * x * x);
            0.5 * x * (1.0 + inner.tanh())
        }))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // GELU derivative is complex, using numerical approximation
        // d/dx GELU(x) ≈ 0.5 * (1 + tanh(inner)) + 0.5 * x * (1 - tanh(inner)^2) * d_inner
        // where d_inner = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        let grad_multiplier = input.map(|x| {
            let x3 = x * x * x;
            let inner = Self::SQRT_2_OVER_PI * (x + Self::GELU_COEF * x3);
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let d_inner = Self::SQRT_2_OVER_PI * (1.0 + 3.0 * Self::GELU_COEF * x * x);
            0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * d_inner
        });

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
        let alpha = self.alpha;
        Ok(input.map(|x| if x > 0.0 { x } else { alpha * x }))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let alpha = self.alpha;
        let mask = input.map(|x| if x > 0.0 { 1.0 } else { alpha });
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
        let alpha = self.alpha;
        Ok(input.map(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let alpha = self.alpha;
        // ELU gradient: 1 if x > 0, else alpha * exp(x) = output + alpha for x <= 0
        let grad_multiplier = input.map(|x| if x > 0.0 { 1.0 } else { alpha * x.exp() });
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
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
}

impl PReLU {
    /// Creates a new PReLU activation layer with the specified initial alpha.
    pub fn new(initial_alpha: f32) -> Self {
        Self {
            alpha: Tensor::from_data(&[1], vec![initial_alpha]),
            cached_input: None,
        }
    }

    /// Performs forward pass and caches input for backward pass.
    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        self.forward(input)
    }

    /// Returns the current alpha value.
    pub fn alpha(&self) -> f32 {
        self.alpha.data()[0]
    }
}

impl Default for PReLU {
    fn default() -> Self {
        Self::new(0.25)
    }
}

impl Layer for PReLU {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let alpha = self.alpha.data()[0];
        Ok(input.map(|x| if x > 0.0 { x } else { alpha * x }))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let alpha = self.alpha.data()[0];
        let mask = input.map(|x| if x > 0.0 { 1.0 } else { alpha });
        Ok(grad.mul(&mask))
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
        Ok(input.map(|x| {
            if x > 0.0 {
                Self::SCALE * x
            } else {
                Self::SCALE * Self::ALPHA * (x.exp() - 1.0)
            }
        }))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // SELU gradient: scale if x > 0, else scale * alpha * exp(x)
        let grad_multiplier = input.map(|x| {
            if x > 0.0 {
                Self::SCALE
            } else {
                Self::SCALE * Self::ALPHA * x.exp()
            }
        });
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

    /// Computes sigmoid(x)
    #[inline]
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Layer for Swish {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.map(|x| x * Self::sigmoid(x)))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Swish gradient: swish(x) + sigmoid(x) * (1 - swish(x))
        // = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        let grad_multiplier = input.map(|x| {
            let sig = Self::sigmoid(x);
            sig + x * sig * (1.0 - sig)
        });
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
        let threshold = self.threshold;
        Ok(input.map(|x| {
            if x > threshold {
                x // For numerical stability
            } else {
                (1.0 + x.exp()).ln()
            }
        }))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let threshold = self.threshold;
        // Softplus gradient: sigmoid(x) = 1 / (1 + exp(-x))
        let grad_multiplier = input.map(|x| {
            if x > threshold {
                1.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        });
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
/// Computes `f(x) = clip((x + 3) / 6, 0, 1)`.
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
/// // hard_sigmoid(0) = (0 + 3) / 6 = 0.5
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
        Ok(input.map(|x| ((x + 3.0) / 6.0).clamp(0.0, 1.0)))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // HardSigmoid gradient: 1/6 if -3 <= x <= 3, else 0
        let grad_multiplier = input.map(|x| {
            if (-3.0..=3.0).contains(&x) {
                1.0 / 6.0
            } else {
                0.0
            }
        });
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
    fn softplus(x: f32) -> f32 {
        if x > 20.0 {
            x
        } else {
            (1.0 + x.exp()).ln()
        }
    }
}

impl Layer for Mish {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        Ok(input.map(|x| x * Self::softplus(x).tanh()))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Mish gradient: complex derivative
        // Let sp = softplus(x), sig = sigmoid(x)
        // d/dx mish(x) = tanh(sp) + x * sech^2(sp) * sig
        let grad_multiplier = input.map(|x| {
            let sp = Self::softplus(x);
            let tanh_sp = sp.tanh();
            let sech2_sp = 1.0 - tanh_sp * tanh_sp;
            let sig = 1.0 / (1.0 + (-x).exp());
            tanh_sp + x * sech2_sp * sig
        });
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
}

impl Softmax {
    /// Creates a new Softmax activation layer.
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

impl Layer for Softmax {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("Softmax expects 2D input, got {}D", input.ndim()),
            });
        }

        let batch_size = input.shape()[0];
        let dim = input.shape()[1];
        let mut result = vec![0.0; input.numel()];

        for i in 0..batch_size {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..dim {
                max_val = max_val.max(input.data()[i * dim + j]);
            }

            // Compute exp(x - max) and sum
            let mut sum = 0.0;
            for j in 0..dim {
                let exp_val = (input.data()[i * dim + j] - max_val).exp();
                result[i * dim + j] = exp_val;
                sum += exp_val;
            }

            // Normalize
            for j in 0..dim {
                result[i * dim + j] /= sum;
            }
        }

        Ok(Tensor::from_data(input.shape(), result))
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let output = self
            .cached_output
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        // Softmax backward: s_i * (grad_i - sum_j(s_j * grad_j))
        let batch_size = output.shape()[0];
        let dim = output.shape()[1];
        let mut result = vec![0.0; output.numel()];

        for i in 0..batch_size {
            // Compute dot product: sum_j(s_j * grad_j)
            let mut dot = 0.0;
            for j in 0..dim {
                dot += output.data()[i * dim + j] * grad.data()[i * dim + j];
            }

            // Compute gradient
            for j in 0..dim {
                let s = output.data()[i * dim + j];
                result[i * dim + j] = s * (grad.data()[i * dim + j] - dot);
            }
        }

        Ok(Tensor::from_data(output.shape(), result))
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

        for &val in output.data() {
            assert!((val - 0.5).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Tanh::new();
        let input = Tensor::zeros(&[2, 2]);
        let output = tanh.forward(&input).unwrap();

        for &val in output.data() {
            assert!(val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_gelu_forward() {
        let gelu = GELU::new();
        let input = Tensor::zeros(&[2, 2]);
        let output = gelu.forward(&input).unwrap();

        for &val in output.data() {
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
        for &val in output.data() {
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
        for &val in output.data() {
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
        for &val in input_grad.data() {
            assert!(val > 0.0 && val < 1.0);
        }
    }

    #[test]
    fn test_hard_sigmoid_forward() {
        let hard_sigmoid = HardSigmoid::new();
        let input = Tensor::from_data(&[1, 5], vec![-5.0, -3.0, 0.0, 3.0, 5.0]);
        let output = hard_sigmoid.forward(&input).unwrap();

        // x <= -3: output = 0
        assert!((output.data()[0]).abs() < 1e-6);
        assert!((output.data()[1]).abs() < 1e-6);

        // x = 0: output = 0.5
        assert!((output.data()[2] - 0.5).abs() < 1e-6);

        // x >= 3: output = 1
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

        // Gradient is 1/6 for -3 <= x <= 3, else 0
        assert!((input_grad.data()[0]).abs() < 1e-6); // x = -5
        assert!((input_grad.data()[1] - 1.0 / 6.0).abs() < 1e-6); // x = -2
        assert!((input_grad.data()[2] - 1.0 / 6.0).abs() < 1e-6); // x = 0
        assert!((input_grad.data()[3] - 1.0 / 6.0).abs() < 1e-6); // x = 2
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
