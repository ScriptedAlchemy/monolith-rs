//! DMR_U2I layer (Deep Match to Rank, User-to-Item).

use crate::activation_layer::ActivationLayer;
use crate::dense::Dense;
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::mlp::ActivationType;
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// DMR_U2I configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DMRU2IConfig {
    pub cmp_dim: usize,
    pub activation: ActivationType,
    pub initializer: Initializer,
    pub regularizer: Regularizer,
}

impl DMRU2IConfig {
    pub fn new(cmp_dim: usize) -> Self {
        Self {
            cmp_dim,
            activation: ActivationType::PReLU {
                alpha: 0.25,
                initializer: None,
                shared_axes: None,
                regularizer: None,
                constraint: None,
            },
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
        }
    }

    pub fn with_activation(mut self, activation: ActivationType) -> Self {
        self.activation = activation;
        self
    }

    pub fn with_initializer(mut self, initializer: Initializer) -> Self {
        self.initializer = initializer;
        self
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }
}

/// DMR_U2I layer.
#[derive(Debug, Clone)]
pub struct DMRU2I {
    cmp_dim: usize,
    regularizer: Regularizer,
    pos_emb: Tensor,
    emb_weight: Tensor,
    z_weight: Tensor,
    bias: Tensor,
    pos_emb_grad: Option<Tensor>,
    emb_weight_grad: Option<Tensor>,
    z_weight_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    linear: Dense,
    activation: ActivationLayer,
    cached_items: Option<Tensor>,
    cached_user_seq: Option<Tensor>,
    cached_alpha: Option<Tensor>,
    cached_comped: Option<Tensor>,
    cached_linear_out: Option<Tensor>,
}

impl DMRU2I {
    pub fn from_config(
        config: DMRU2IConfig,
        seq_len: usize,
        user_emb_dim: usize,
        item_emb_dim: usize,
    ) -> Self {
        let pos_emb = config.initializer.initialize(&[seq_len, config.cmp_dim]);
        let emb_weight = config
            .initializer
            .initialize(&[user_emb_dim, config.cmp_dim]);
        let z_weight = Initializer::Ones.initialize(&[config.cmp_dim, 1]);
        let bias = Initializer::Zeros.initialize(&[config.cmp_dim]);
        let linear = Dense::new_with_initializer(
            user_emb_dim,
            item_emb_dim,
            config.initializer,
            Initializer::Zeros,
            true,
        )
        .with_kernel_regularizer(config.regularizer.clone());
        let activation = ActivationLayer::from_activation_type(config.activation);
        Self {
            cmp_dim: config.cmp_dim,
            regularizer: config.regularizer,
            pos_emb,
            emb_weight,
            z_weight,
            bias,
            pos_emb_grad: None,
            emb_weight_grad: None,
            z_weight_grad: None,
            bias_grad: None,
            linear,
            activation,
            cached_items: None,
            cached_user_seq: None,
            cached_alpha: None,
            cached_comped: None,
            cached_linear_out: None,
        }
    }

    pub fn forward_train(
        &mut self,
        items: &Tensor,
        user_seq: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let (b, seq_len, _ue_size) = (
            user_seq.shape()[0],
            user_seq.shape()[1],
            user_seq.shape()[2],
        );
        let emb_cmp = user_seq.matmul(&self.emb_weight);
        let pos = self
            .pos_emb
            .reshape(&[1, seq_len, self.cmp_dim])
            .broadcast_as(&[b, seq_len, self.cmp_dim]);
        let bias_b =
            self.bias
                .reshape(&[1, 1, self.cmp_dim])
                .broadcast_as(&[b, seq_len, self.cmp_dim]);
        let comped = emb_cmp.add(&pos).add(&bias_b);
        let alpha = comped.matmul(&self.z_weight).softmax(1);

        let user_seq_trans = user_seq.transpose_dims(1, 2);
        let user_seq_merged = user_seq_trans.matmul(&alpha).squeeze(2);
        let linear_out = self.linear.forward_train(&user_seq_merged)?;
        let activated = self.activation.forward_train(&linear_out)?;
        let output = activated.mul(items);

        self.cached_items = Some(items.clone());
        self.cached_user_seq = Some(user_seq.clone());
        self.cached_alpha = Some(alpha);
        self.cached_comped = Some(comped);
        self.cached_linear_out = Some(linear_out);

        Ok(output)
    }

    pub fn backward(&mut self, grad: &Tensor) -> Result<(Tensor, Tensor), LayerError> {
        let items = self
            .cached_items
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let user_seq = self
            .cached_user_seq
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let alpha = self
            .cached_alpha
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let comped = self
            .cached_comped
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let linear_out = self
            .cached_linear_out
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let grad_items = grad.mul(&self.activation.forward(linear_out)?);
        let grad_act = grad.mul(items);
        let grad_linear = self.activation.backward(&grad_act)?;
        let grad_user_seq_merged = self.linear.backward(&grad_linear)?;

        let (b, _seq_len, ue_size) = (
            user_seq.shape()[0],
            user_seq.shape()[1],
            user_seq.shape()[2],
        );
        let grad_y = grad_user_seq_merged.reshape(&[b, ue_size, 1]);
        let grad_user_seq_trans = grad_y.matmul(&alpha.transpose_dims(1, 2));
        let grad_alpha = user_seq.transpose_dims(1, 2).matmul(&grad_y);

        let alpha_grad = softmax_backward_3d_axis1(alpha, &grad_alpha);
        let grad_comped = alpha_grad.matmul(&self.z_weight.transpose());
        let grad_z_weight_batch = comped.transpose_dims(1, 2).matmul(&alpha_grad);
        self.z_weight_grad = Some(grad_z_weight_batch.sum_axis(0));

        let grad_pos = grad_comped.sum_axis(0);
        self.pos_emb_grad = Some(grad_pos.clone());
        let grad_bias = grad_pos.sum_axis(0);
        self.bias_grad = Some(grad_bias);

        let grad_emb_cmp = grad_comped;
        let grad_user_seq_from_emb = grad_emb_cmp.matmul(&self.emb_weight.transpose());
        let grad_emb_weight_batch = user_seq.transpose_dims(1, 2).matmul(&grad_emb_cmp);
        self.emb_weight_grad = Some(grad_emb_weight_batch.sum_axis(0));

        let grad_user_seq = grad_user_seq_trans
            .transpose_dims(1, 2)
            .add(&grad_user_seq_from_emb);
        Ok((grad_items, grad_user_seq))
    }

    /// Returns regularization loss for trainable weights.
    pub fn regularization_loss(&self) -> f32 {
        let mut loss = 0.0;
        loss += self.regularizer.loss(&self.pos_emb);
        loss += self.regularizer.loss(&self.emb_weight);
        loss += self.linear.regularization_loss();
        loss
    }
}

fn softmax_backward_3d_axis1(softmax: &Tensor, grad: &Tensor) -> Tensor {
    let dot = grad.mul(softmax).sum_axis(1);
    let (b, s, h) = (softmax.shape()[0], softmax.shape()[1], softmax.shape()[2]);
    let dot_b = dot.reshape(&[b, 1, h]).broadcast_as(&[b, s, h]);
    softmax.mul(&grad.sub(&dot_b))
}
