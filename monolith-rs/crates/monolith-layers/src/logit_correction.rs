//! Logit correction layer.

use crate::activation_layer::ActivationLayer;
use crate::error::LayerError;
use crate::layer::Layer;
use crate::mlp::ActivationType;
use crate::tensor::Tensor;
/// LogitCorrection layer.
#[derive(Debug, Clone)]
pub struct LogitCorrection {
    #[allow(dead_code)]
    activation: ActivationType,
    sample_bias: bool,
    cached_logits: Option<Tensor>,
    cached_sample_rate: Option<Tensor>,
    activation_layer: Option<ActivationLayer>,
}

impl LogitCorrection {
    /// Creates a new LogitCorrection layer.
    pub fn new(activation: ActivationType, sample_bias: bool) -> Self {
        let activation_layer = if activation == ActivationType::None {
            None
        } else {
            Some(ActivationLayer::from_activation_type(activation.clone()))
        };
        Self {
            activation,
            sample_bias,
            cached_logits: None,
            cached_sample_rate: None,
            activation_layer,
        }
    }

    fn safe_log_sigmoid(logits: &Tensor) -> Tensor {
        let cond = logits.ge_scalar(0.0);
        let ones = Tensor::ones(logits.shape());
        let relu_logits = logits.mul(&cond);
        let neg_abs = logits.mul(&cond.scale(-1.0).add(&ones.sub(&cond)));
        let log1p = neg_abs.exp().add(&ones).log();
        relu_logits.sub(logits).add(&log1p).scale(-1.0)
    }

    fn get_sample_logits(
        logits: &Tensor,
        sample_rate: Option<&Tensor>,
        sample_bias: bool,
    ) -> Tensor {
        match (sample_rate, sample_bias) {
            (None, true) => Self::safe_log_sigmoid(logits),
            (Some(rate), false) => logits.sub(&rate.log()),
            (Some(rate), true) => Self::safe_log_sigmoid(logits).sub(&rate.log()),
            (None, false) => logits.clone(),
        }
    }

    /// Forward with optional sample_rate.
    pub fn forward_with_sample_rate(
        &mut self,
        logits: &Tensor,
        sample_rate: Option<&Tensor>,
    ) -> Result<Tensor, LayerError> {
        self.cached_logits = Some(logits.clone());
        self.cached_sample_rate = sample_rate.cloned();
        let mut corrected = Self::get_sample_logits(logits, sample_rate, self.sample_bias);
        if let Some(act) = &mut self.activation_layer {
            corrected = act.forward_train(&corrected)?;
        }
        Ok(corrected)
    }
}

impl Default for LogitCorrection {
    fn default() -> Self {
        Self::new(ActivationType::None, false)
    }
}

impl Layer for LogitCorrection {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut layer = self.clone();
        layer.forward_with_sample_rate(input, None)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let logits = self
            .cached_logits
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let mut grad_out = grad.clone();
        if let Some(act) = &mut self.activation_layer {
            grad_out = act.backward(&grad_out)?;
        }

        let grad_logits = if self.sample_bias {
            let sig = logits.sigmoid();
            let ones = Tensor::ones(sig.shape());
            grad_out.mul(&ones.sub(&sig))
        } else {
            grad_out
        };
        Ok(grad_logits)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "LogitCorrection"
    }
}
