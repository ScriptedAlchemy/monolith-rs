//! Sub-Network Routing (SNR) layer.

use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// SNR type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SNRType {
    Trans,
    Aver,
}

/// Configuration for SNR.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SNRConfig {
    pub num_out_subnet: usize,
    pub out_subnet_dim: usize,
    pub snr_type: SNRType,
    pub zeta: f32,
    pub gamma: f32,
    pub beta: f32,
    pub use_ste: bool,
    pub training: bool,
    pub initializer: Initializer,
    pub regularizer: Regularizer,
}

impl SNRConfig {
    pub fn new(num_out_subnet: usize, out_subnet_dim: usize) -> Self {
        Self {
            num_out_subnet,
            out_subnet_dim,
            snr_type: SNRType::Trans,
            zeta: 1.1,
            gamma: -0.1,
            beta: 0.5,
            use_ste: false,
            training: true,
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
        }
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }
}

/// SNR layer.
#[derive(Debug, Clone)]
pub struct SNR {
    config: SNRConfig,
    num_in_subnet: Option<usize>,
    in_subnet_dim: Option<usize>,
    weight: Option<Tensor>,
    weight_grad: Option<Tensor>,
    log_alpha: Option<Tensor>,
    log_alpha_grad: Option<Tensor>,
    cached_inputs: Option<Tensor>,
    cached_z: Option<Tensor>,
    cached_s: Option<Tensor>,
}

impl SNR {
    pub fn new(config: SNRConfig) -> Self {
        Self {
            config,
            num_in_subnet: None,
            in_subnet_dim: None,
            weight: None,
            weight_grad: None,
            log_alpha: None,
            log_alpha_grad: None,
            cached_inputs: None,
            cached_z: None,
            cached_s: None,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.config.training = training;
    }

    fn build_if_needed(&mut self, inputs: &[Tensor]) -> Result<(), LayerError> {
        if inputs.is_empty() {
            return Err(LayerError::ForwardError {
                message: "SNR requires non-empty input list".to_string(),
            });
        }
        if self.num_in_subnet.is_some() {
            return Ok(());
        }
        let in_dim = inputs[0].shape()[1];
        for input in inputs {
            if input.ndim() != 2 {
                return Err(LayerError::ForwardError {
                    message: "SNR expects 2D inputs".to_string(),
                });
            }
            if input.shape()[1] != in_dim {
                return Err(LayerError::ShapeMismatch {
                    expected: vec![in_dim],
                    actual: vec![input.shape()[1]],
                });
            }
        }
        let num_in = inputs.len();
        self.num_in_subnet = Some(num_in);
        self.in_subnet_dim = Some(in_dim);

        let num_route = num_in * self.config.num_out_subnet;
        let block_size = in_dim * self.config.out_subnet_dim;

        let weight = if self.config.snr_type == SNRType::Trans {
            Some(self.config.initializer.initialize(&[num_route, block_size]))
        } else {
            let mut block = vec![0.0f32; block_size];
            for i in 0..in_dim.min(self.config.out_subnet_dim) {
                block[i * self.config.out_subnet_dim + i] = 1.0;
            }
            let mut w = Vec::with_capacity(num_route * block_size);
            for _ in 0..num_route {
                w.extend_from_slice(&block);
            }
            Some(Tensor::from_data(&[num_route, block_size], w))
        };

        let log_alpha = Some(Tensor::zeros(&[num_route, 1]));
        self.weight = weight;
        self.log_alpha = log_alpha;
        Ok(())
    }

    fn sample(&mut self) -> Result<Tensor, LayerError> {
        let log_alpha = self.log_alpha.as_ref().ok_or(LayerError::NotInitialized)?;
        let s = if self.config.training {
            let u = Tensor::rand(log_alpha.shape());
            let logit = u
                .log()
                .sub(&Tensor::ones(u.shape()).sub(&u).log())
                .add(log_alpha);
            logit.scale(1.0 / self.config.beta).sigmoid()
        } else {
            log_alpha.sigmoid()
        };
        let s_scaled = s
            .scale(self.config.zeta - self.config.gamma)
            .add(&Tensor::from_data(&[1], vec![self.config.gamma]));
        let z = s_scaled.clamp(0.0, 1.0);
        self.cached_s = Some(s);
        self.cached_z = Some(z.clone());
        Ok(z)
    }

    /// Returns the L0 loss term.
    pub fn l0_loss(&self) -> Result<f32, LayerError> {
        let log_alpha = self.log_alpha.as_ref().ok_or(LayerError::NotInitialized)?;
        let factor = self.config.beta * (-self.config.gamma / self.config.zeta).ln();
        let loss = log_alpha
            .sub(&Tensor::from_data(&[1], vec![factor]))
            .sigmoid()
            .sum();
        Ok(loss)
    }

    /// Returns regularization loss (L0 + weight regularizer if any).
    pub fn regularization_loss(&self) -> Result<f32, LayerError> {
        let mut loss = self.l0_loss()?;
        if let Some(weight) = &self.weight {
            loss += self.config.regularizer.loss(weight);
        }
        Ok(loss)
    }

    /// Forward with list inputs, returns list outputs.
    pub fn forward_with_inputs(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>, LayerError> {
        self.build_if_needed(inputs)?;
        let num_in = self.num_in_subnet.unwrap();
        let in_dim = self.in_subnet_dim.unwrap();
        let num_out = self.config.num_out_subnet;
        let out_dim = self.config.out_subnet_dim;

        let z = self.sample()?;
        let weight = self.weight.as_ref().ok_or(LayerError::NotInitialized)?;
        let weight = weight.mul(&z.broadcast_as(weight.shape()));

        let weight4 = weight.reshape(&[num_in, num_out, in_dim, out_dim]);
        let weight4 = weight4.permute(&[0, 2, 1, 3]);
        let weight_mat = weight4.reshape(&[num_in * in_dim, num_out * out_dim]);

        let input_concat = Tensor::cat(inputs, 1);
        self.cached_inputs = Some(input_concat.clone());

        let out = input_concat.matmul(&weight_mat);
        let out_3d = out.reshape(&[out.shape()[0], num_out, out_dim]);
        Ok(out_3d.unstack(1))
    }

    /// Backward with list gradients, returns list input gradients.
    pub fn backward_with_inputs(&mut self, grads: &[Tensor]) -> Result<Vec<Tensor>, LayerError> {
        let input_concat = self
            .cached_inputs
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let z = self.cached_z.as_ref().ok_or(LayerError::NotInitialized)?;
        let s = self.cached_s.as_ref().ok_or(LayerError::NotInitialized)?;
        let weight = self.weight.as_ref().ok_or(LayerError::NotInitialized)?;
        let num_in = self.num_in_subnet.unwrap();
        let in_dim = self.in_subnet_dim.unwrap();
        let num_out = self.config.num_out_subnet;
        let out_dim = self.config.out_subnet_dim;

        let grad_concat = Tensor::cat(grads, 1);

        let weight4 = weight
            .mul(&z.broadcast_as(weight.shape()))
            .reshape(&[num_in, num_out, in_dim, out_dim]);
        let weight4 = weight4.permute(&[0, 2, 1, 3]);
        let weight_mat = weight4.reshape(&[num_in * in_dim, num_out * out_dim]);

        let grad_input = grad_concat.matmul(&weight_mat.transpose());
        let mut input_grads = grad_input
            .reshape(&[grad_input.shape()[0], num_in, in_dim])
            .unstack(1);

        let grad_weight_mat = input_concat.transpose().matmul(&grad_concat);
        let grad_weight4 = grad_weight_mat
            .reshape(&[num_in, in_dim, num_out, out_dim])
            .permute(&[0, 2, 1, 3]);
        let grad_weight = grad_weight4.reshape(&[num_in * num_out, in_dim * out_dim]);

        if self.config.snr_type == SNRType::Trans {
            let z_b = z.broadcast_as(grad_weight.shape());
            let mut weight_grad = grad_weight.mul(&z_b);
            if let Some(reg_grad) = self.config.regularizer.grad(weight) {
                weight_grad = weight_grad.add(&reg_grad);
            }
            self.weight_grad = Some(weight_grad);

            let mut grad_z = grad_weight.mul(weight).sum_axis(1).reshape(z.shape());
            if !self.config.use_ste {
                let mask = z.gt_scalar(0.0).mul(&z.lt_scalar(1.0));
                grad_z = grad_z.mul(&mask);
            }
            let dz_ds = self.config.zeta - self.config.gamma;
            grad_z = grad_z.scale(dz_ds);
            let ds_dlog = s
                .mul(&Tensor::ones(s.shape()).sub(s))
                .scale(1.0 / self.config.beta);
            let grad_log_alpha = grad_z.mul(&ds_dlog);
            self.log_alpha_grad = Some(grad_log_alpha);
        }

        Ok(input_grads.drain(..).collect())
    }
}
