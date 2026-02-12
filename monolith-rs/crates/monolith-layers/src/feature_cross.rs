//! Feature cross layers (GroupInt, AllInt, CDot, CAN, CIN).

use crate::activation_layer::ActivationLayer;
use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::merge::{merge_tensor_list, merge_tensor_list_tensor, MergeOutput, MergeType};
use crate::mlp::{ActivationType, MLPConfig, MLP};
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;

#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::tensor::{gpu_available, is_gpu_enabled};
use serde::{Deserialize, Serialize};

#[cfg(any(feature = "metal", feature = "cuda"))]
use monolith_tensor::Tensor as CandleTensorOps;

/// Interaction type for GroupInt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupIntType {
    Multiply,
    Dot,
}

/// Group Interaction layer (GroupInt / FFM).
#[derive(Debug, Clone)]
pub struct GroupInt {
    interaction_type: GroupIntType,
    use_attention: bool,
    attention_units: Vec<usize>,
    activation: ActivationType,
    initializer: Initializer,
    regularizer: Regularizer,
    out_type: MergeType,
    keep_list: bool,
    mlp: Option<MLP>,
    mlp_input_dim: Option<usize>,
    cached_left: Option<Tensor>,
    cached_right: Option<Tensor>,
    cached_attention: Option<Tensor>,
}

impl GroupInt {
    pub fn new(interaction_type: GroupIntType) -> Self {
        Self {
            interaction_type,
            use_attention: false,
            attention_units: Vec::new(),
            activation: ActivationType::relu(),
            initializer: Initializer::GlorotNormal,
            regularizer: Regularizer::None,
            out_type: MergeType::Concat,
            keep_list: false,
            mlp: None,
            mlp_input_dim: None,
            cached_left: None,
            cached_right: None,
            cached_attention: None,
        }
    }

    pub fn with_attention(
        mut self,
        attention_units: Vec<usize>,
        activation: ActivationType,
        initializer: Initializer,
    ) -> Self {
        self.use_attention = true;
        self.attention_units = attention_units.clone();
        self.activation = activation;
        self.initializer = initializer;
        self.mlp = None;
        self.mlp_input_dim = None;
        self
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    pub fn with_out_type(mut self, out_type: MergeType) -> Self {
        self.out_type = out_type;
        self
    }

    pub fn with_keep_list(mut self, keep_list: bool) -> Self {
        self.keep_list = keep_list;
        self
    }

    pub fn forward_with_groups(
        &mut self,
        left_fields: &[Tensor],
        right_fields: &[Tensor],
    ) -> Result<Tensor, LayerError> {
        let left = concat_fields(left_fields)?;
        let right = concat_fields(right_fields)?;
        self.cached_left = Some(left.clone());
        self.cached_right = Some(right.clone());

        let ffm = ffm_interaction(&left, &right, self.interaction_type)?;
        let output = if self.interaction_type == GroupIntType::Multiply && self.use_attention {
            let num_feature = left.shape()[1] * right.shape()[1];
            let emb_dim = left.shape()[2];
            let stacked = ffm.reshape(&[left.shape()[0], num_feature, emb_dim]);
            if self.mlp.is_none() || self.mlp_input_dim != Some(emb_dim) {
                let mut config = MLPConfig::new(emb_dim);
                for (i, &dim) in self.attention_units.iter().enumerate() {
                    let act = if i + 1 == self.attention_units.len() {
                        ActivationType::None
                    } else {
                        self.activation.clone()
                    };
                    config = config.add_layer_with_initializer(dim, act, self.initializer);
                }
                config = config.with_regularizers(self.regularizer.clone(), Regularizer::None);
                self.mlp = Some(MLP::from_config(config)?);
                self.mlp_input_dim = Some(emb_dim);
            }
            let mlp = self.mlp.as_mut().ok_or(LayerError::NotInitialized)?;
            let att_in = stacked.reshape(&[left.shape()[0] * num_feature, emb_dim]);
            let attention = mlp.forward_train(&att_in)?;
            let attention = attention.reshape(&[left.shape()[0], num_feature, 1]);
            self.cached_attention = Some(attention.clone());
            let weighted = stacked.mul(
                &attention
                    .reshape(&[left.shape()[0], num_feature, 1])
                    .broadcast_as(&[left.shape()[0], num_feature, emb_dim]),
            );
            weighted.reshape(&[left.shape()[0], num_feature * emb_dim])
        } else {
            ffm
        };

        Ok(output)
    }

    pub fn forward_with_groups_merge(
        &mut self,
        left_fields: &[Tensor],
        right_fields: &[Tensor],
    ) -> Result<MergeOutput, LayerError> {
        let out = self.forward_with_groups(left_fields, right_fields)?;
        if self.keep_list {
            Ok(MergeOutput::List(vec![out]))
        } else {
            Ok(MergeOutput::Tensor(out))
        }
    }

    pub fn regularization_loss(&self) -> f32 {
        self.mlp
            .as_ref()
            .map(|mlp| mlp.regularization_loss())
            .unwrap_or(0.0)
    }
}

impl GroupInt {
    pub fn backward(&mut self, grad: &Tensor) -> Result<(Tensor, Tensor), LayerError> {
        let left = self
            .cached_left
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let right = self
            .cached_right
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let (b, l, d) = (left.shape()[0], left.shape()[1], left.shape()[2]);
        let r = right.shape()[1];

        if self.interaction_type == GroupIntType::Dot {
            let grad_scores = grad.reshape(&[b, l, r]);
            let grad_left = grad_scores.matmul(&right);
            let grad_right = grad_scores.transpose_dims(1, 2).matmul(&left);
            return Ok((grad_left, grad_right));
        }

        let grad_mul = if self.use_attention {
            let num_feature = l * r;
            let emb_dim = d;
            let grad_reshaped = grad.reshape(&[b, num_feature, emb_dim]);
            let attention = self
                .cached_attention
                .as_ref()
                .ok_or(LayerError::NotInitialized)?;
            let grad_stacked =
                grad_reshaped.mul(&attention.reshape(&[b, num_feature, 1]).broadcast_as(&[
                    b,
                    num_feature,
                    emb_dim,
                ]));

            let stacked = ffm_multiply(left, right)?;
            let grad_attention = grad_reshaped.mul(&stacked).sum_axis(2);
            if let Some(mlp) = self.mlp.as_mut() {
                let grad_att_in = grad_attention.reshape(&[b * num_feature, 1]);
                let grad_att = mlp.backward(&grad_att_in)?;
                let grad_att = grad_att.reshape(&[b, num_feature, emb_dim]);
                grad_stacked.add(&grad_att)
            } else {
                grad_stacked
            }
        } else {
            grad.reshape(&[b, l * r, d])
        };

        let mut grad_left = vec![0.0; b * l * d];
        let mut grad_right = vec![0.0; b * r * d];
        let grad_data = grad_mul.data_ref();
        let left_data = left.data_ref();
        let right_data = right.data_ref();

        for bi in 0..b {
            for li in 0..l {
                for ri in 0..r {
                    let idx = bi * l * r * d + (li * r + ri) * d;
                    for di in 0..d {
                        let g = grad_data[idx + di];
                        let l_idx = bi * l * d + li * d + di;
                        let r_idx = bi * r * d + ri * d + di;
                        grad_left[l_idx] += g * right_data[r_idx];
                        grad_right[r_idx] += g * left_data[l_idx];
                    }
                }
            }
        }

        Ok((
            Tensor::from_data(left.shape(), grad_left),
            Tensor::from_data(right.shape(), grad_right),
        ))
    }
}

fn concat_fields(fields: &[Tensor]) -> Result<Tensor, LayerError> {
    if fields.is_empty() {
        return Err(LayerError::ForwardError {
            message: "GroupInt requires non-empty fields".to_string(),
        });
    }
    let mut tensors = Vec::with_capacity(fields.len());
    for t in fields.iter() {
        if t.ndim() == 2 {
            tensors.push(t.reshape(&[t.shape()[0], 1, t.shape()[1]]));
        } else if t.ndim() == 3 {
            tensors.push(t.clone());
        } else {
            return Err(LayerError::ForwardError {
                message: "GroupInt expects 2D or 3D tensors".to_string(),
            });
        }
    }
    Ok(Tensor::cat(&tensors, 1))
}

fn ffm_interaction(
    left: &Tensor,
    right: &Tensor,
    interaction_type: GroupIntType,
) -> Result<Tensor, LayerError> {
    if left.ndim() != 3 || right.ndim() != 3 {
        return Err(LayerError::ForwardError {
            message: "FFM interaction expects 3D tensors".to_string(),
        });
    }
    let (b, l, d) = (left.shape()[0], left.shape()[1], left.shape()[2]);
    let r = right.shape()[1];
    match interaction_type {
        GroupIntType::Dot => {
            let scores = left.matmul(&right.transpose_dims(1, 2));
            Ok(scores.reshape(&[b, l * r]))
        }
        GroupIntType::Multiply => {
            let prod = ffm_multiply(left, right)?;
            Ok(prod.reshape(&[b, l * r * d]))
        }
    }
}

fn ffm_multiply(left: &Tensor, right: &Tensor) -> Result<Tensor, LayerError> {
    let (b, l, d) = (left.shape()[0], left.shape()[1], left.shape()[2]);
    let r = right.shape()[1];

    #[cfg(any(feature = "metal", feature = "cuda"))]
    if is_gpu_enabled() && gpu_available() {
        let l_gpu = left.to_candle();
        let r_gpu = right.to_candle();
        let l4 = l_gpu.reshape(&[b, l, 1, d]);
        let r4 = r_gpu.reshape(&[b, 1, r, d]);
        let prod = l4.mul(&r4);
        let prod = prod.reshape(&[b, l * r, d]);
        return Ok(Tensor::from_candle(prod));
    }

    let left_data = left.data_ref();
    let right_data = right.data_ref();
    let mut out = vec![0.0; b * l * r * d];
    for bi in 0..b {
        for li in 0..l {
            for ri in 0..r {
                let out_base = bi * l * r * d + (li * r + ri) * d;
                let l_base = bi * l * d + li * d;
                let r_base = bi * r * d + ri * d;
                for di in 0..d {
                    out[out_base + di] = left_data[l_base + di] * right_data[r_base + di];
                }
            }
        }
    }
    Ok(Tensor::from_data(&[b, l * r, d], out))
}

/// AllInt layer (All Interaction).
#[derive(Debug, Clone)]
pub struct AllInt {
    cmp_dim: usize,
    initializer: Initializer,
    regularizer: Regularizer,
    kernel: Tensor,
    bias: Option<Tensor>,
    kernel_grad: Option<Tensor>,
    bias_grad: Option<Tensor>,
    out_type: MergeType,
    keep_list: bool,
    cached_input: Option<Tensor>,
    cached_feature_comp: Option<Tensor>,
}

impl AllInt {
    pub fn new(cmp_dim: usize, initializer: Initializer, use_bias: bool) -> Self {
        Self {
            cmp_dim,
            initializer,
            regularizer: Regularizer::None,
            kernel: initializer.initialize(&[1, cmp_dim]),
            bias: if use_bias {
                Some(Tensor::zeros(&[cmp_dim]))
            } else {
                None
            },
            kernel_grad: None,
            bias_grad: None,
            out_type: MergeType::Concat,
            keep_list: false,
            cached_input: None,
            cached_feature_comp: None,
        }
    }

    pub fn with_out_type(mut self, out_type: MergeType) -> Self {
        self.out_type = out_type;
        self
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    pub fn with_keep_list(mut self, keep_list: bool) -> Self {
        self.keep_list = keep_list;
        self
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: "AllInt expects 3D input".to_string(),
            });
        }
        let num_feat = input.shape()[1];
        let kernel = if self.kernel.shape()[0] != num_feat {
            self.kernel = self.initializer.initialize(&[num_feat, self.cmp_dim]);
            self.kernel.clone()
        } else {
            self.kernel.clone()
        };

        let transposed = input.transpose_dims(1, 2);
        let mut feature_comp = transposed.matmul(&kernel);
        if let Some(bias) = &self.bias {
            let bias_b = bias
                .reshape(&[1, 1, self.cmp_dim])
                .broadcast_as(feature_comp.shape());
            feature_comp = feature_comp.add(&bias_b);
        }
        let interaction = input.matmul(&feature_comp);
        self.cached_input = Some(input.clone());
        self.cached_feature_comp = Some(feature_comp);

        if self.out_type == MergeType::None {
            return Ok(interaction);
        }
        Ok(merge_tensor_list_tensor(
            vec![interaction],
            self.out_type,
            None,
            1,
        )?)
    }

    pub fn forward_with_merge(&mut self, input: &Tensor) -> Result<MergeOutput, LayerError> {
        let out = self.forward_train(input)?;
        Ok(merge_tensor_list(
            vec![out],
            self.out_type,
            None,
            1,
            self.keep_list,
        )?)
    }
}

impl Layer for AllInt {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut layer = self.clone();
        layer.forward_train(input)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let feature_comp = self
            .cached_feature_comp
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let (b, f, _d) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let grad_3d = match self.out_type {
            MergeType::Concat => grad.reshape(&[b, f, self.cmp_dim]),
            MergeType::Stack => grad.clone(),
            MergeType::None => {
                return Err(LayerError::BackwardError {
                    message: "AllInt backward cannot accept list output".to_string(),
                })
            }
        };

        let grad_input_from_inter = grad_3d.matmul(&feature_comp.transpose_dims(1, 2));
        let grad_feature_comp = input.transpose_dims(1, 2).matmul(&grad_3d);

        let grad_kernel_batch = input.matmul(&grad_feature_comp);
        let mut grad_kernel = grad_kernel_batch.sum_axis(0);
        if let Some(reg_grad) = self.regularizer.grad(&self.kernel) {
            grad_kernel = grad_kernel.add(&reg_grad);
        }
        self.kernel_grad = Some(grad_kernel);

        if let Some(bias) = &self.bias {
            let grad_bias = grad_feature_comp.sum_axis(0).sum_axis(0);
            self.bias_grad = Some(grad_bias);
            let _ = bias;
        }

        let grad_input_from_comp = grad_feature_comp
            .matmul(&self.kernel.transpose())
            .transpose_dims(1, 2);
        Ok(grad_input_from_inter.add(&grad_input_from_comp))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.kernel];
        if let Some(bias) = &self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.kernel];
        if let Some(bias) = &mut self.bias {
            params.push(bias);
        }
        params
    }

    fn name(&self) -> &str {
        "AllInt"
    }

    fn regularization_loss(&self) -> f32 {
        self.regularizer.loss(&self.kernel)
    }
}

/// CDot layer.
#[derive(Debug, Clone)]
pub struct CDot {
    project_dim: usize,
    compress_units: Vec<usize>,
    activation: ActivationType,
    initializer: Initializer,
    regularizer: Regularizer,
    project_weight: Tensor,
    project_grad: Option<Tensor>,
    compress_tower: Option<MLP>,
    compress_input_dim: Option<usize>,
    cached_input: Option<Tensor>,
    cached_projected: Option<Tensor>,
}

impl CDot {
    pub fn new(project_dim: usize, compress_units: Vec<usize>) -> Result<Self, LayerError> {
        let activation = ActivationType::relu();
        let initializer = Initializer::GlorotNormal;
        Ok(Self {
            project_dim,
            compress_units,
            activation,
            initializer,
            regularizer: Regularizer::None,
            project_weight: initializer.initialize(&[1, project_dim]),
            project_grad: None,
            compress_tower: None,
            compress_input_dim: None,
            cached_input: None,
            cached_projected: None,
        })
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    fn ensure_tower(&mut self, input_dim: usize, output_dim: usize) -> Result<(), LayerError> {
        if self.compress_tower.is_some() && self.compress_input_dim == Some(input_dim) {
            return Ok(());
        }
        let mut config = MLPConfig::new(input_dim);
        for &dim in self.compress_units.iter() {
            config =
                config.add_layer_with_initializer(dim, self.activation.clone(), self.initializer);
        }
        config =
            config.add_layer_with_initializer(output_dim, ActivationType::None, self.initializer);
        config = config.with_regularizers(self.regularizer.clone(), Regularizer::None);
        self.compress_tower = Some(MLP::from_config(config)?);
        self.compress_input_dim = Some(input_dim);
        Ok(())
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: "CDot expects 3D input".to_string(),
            });
        }
        let (b, num_feat, emb_size) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if self.project_weight.shape()[0] != num_feat {
            self.project_weight = self.initializer.initialize(&[num_feat, self.project_dim]);
        }
        let transposed = input.transpose_dims(1, 2);
        let projected = transposed.matmul(&self.project_weight);
        let concated = projected.reshape(&[b, emb_size * self.project_dim]);
        let compress_out_dim = emb_size * self.project_dim;
        self.ensure_tower(compress_out_dim, compress_out_dim)?;
        let compressed = self
            .compress_tower
            .as_mut()
            .ok_or(LayerError::NotInitialized)?
            .forward_train(&concated)?;
        let reshaped = compressed.reshape(&[b, emb_size, self.project_dim]);
        let crossed = input.matmul(&reshaped);
        let crossed_flat = crossed.reshape(&[b, num_feat * self.project_dim]);
        let output = Tensor::cat(&[crossed_flat, compressed], 1);

        self.cached_input = Some(input.clone());
        self.cached_projected = Some(projected);

        Ok(output)
    }
}

impl Layer for CDot {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut layer = self.clone();
        layer.forward_train(input)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let (b, num_feat, emb_size) = (input.shape()[0], input.shape()[1], input.shape()[2]);

        let crossed_size = num_feat * self.project_dim;
        let grad_crossed_flat = grad.narrow(1, 0, crossed_size);
        let grad_compressed = grad.narrow(1, crossed_size, emb_size * self.project_dim);

        let grad_crossed = grad_crossed_flat.reshape(&[b, num_feat, self.project_dim]);
        let reshaped = grad_compressed.reshape(&[b, emb_size, self.project_dim]);

        let grad_input_from_cross = grad_crossed.matmul(&reshaped.transpose_dims(1, 2));
        let grad_reshaped = input.transpose_dims(1, 2).matmul(&grad_crossed);

        let grad_compressed_total =
            grad_compressed.add(&grad_reshaped.reshape(&[b, emb_size * self.project_dim]));
        let grad_concated = self
            .compress_tower
            .as_mut()
            .ok_or(LayerError::NotInitialized)?
            .backward(&grad_compressed_total)?;

        let grad_projected = grad_concated.reshape(&[b, emb_size, self.project_dim]);
        let grad_input_from_proj = grad_projected
            .matmul(&self.project_weight.transpose())
            .transpose_dims(1, 2);
        let grad_weight_batch = input.matmul(&grad_projected);
        let mut grad_weight = grad_weight_batch.sum_axis(0);
        if let Some(reg_grad) = self.regularizer.grad(&self.project_weight) {
            grad_weight = grad_weight.add(&reg_grad);
        }
        self.project_grad = Some(grad_weight);

        Ok(grad_input_from_cross.add(&grad_input_from_proj))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = if let Some(mlp) = &self.compress_tower {
            mlp.parameters()
        } else {
            Vec::new()
        };
        params.push(&self.project_weight);
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = if let Some(mlp) = &mut self.compress_tower {
            mlp.parameters_mut()
        } else {
            Vec::new()
        };
        params.push(&mut self.project_weight);
        params
    }

    fn name(&self) -> &str {
        "CDot"
    }

    fn regularization_loss(&self) -> f32 {
        let mut loss = self.regularizer.loss(&self.project_weight);
        if let Some(mlp) = &self.compress_tower {
            loss += mlp.regularization_loss();
        }
        loss
    }
}

/// CAN layer.
#[derive(Debug, Clone)]
pub struct CAN {
    layer_num: usize,
    activation: ActivationLayer,
    is_seq: bool,
    is_stacked: bool,
    splits: Vec<usize>,
    cached_users: Vec<Tensor>,
    cached_weights: Vec<Tensor>,
    cached_biases: Vec<Tensor>,
    cached_activations: Vec<ActivationLayer>,
    cached_user_was_expanded: bool,
}

impl CAN {
    pub fn new(
        layer_num: usize,
        activation: ActivationType,
        is_seq: bool,
        is_stacked: bool,
    ) -> Self {
        Self {
            layer_num,
            activation: ActivationLayer::from_activation_type(activation),
            is_seq,
            is_stacked,
            splits: Vec::new(),
            cached_users: Vec::new(),
            cached_weights: Vec::new(),
            cached_biases: Vec::new(),
            cached_activations: Vec::new(),
            cached_user_was_expanded: false,
        }
    }

    pub fn forward_with_inputs(
        &mut self,
        user_emb: &Tensor,
        item_emb: &Tensor,
    ) -> Result<Tensor, LayerError> {
        let u_emb_size = *user_emb.shape().last().unwrap();
        let required = (u_emb_size * (u_emb_size + 1)) * self.layer_num;
        if item_emb.shape()[1] != required {
            return Err(LayerError::ShapeMismatch {
                expected: vec![required],
                actual: vec![item_emb.shape()[1]],
            });
        }
        self.splits = vec![u_emb_size * u_emb_size, u_emb_size].repeat(self.layer_num);
        let params = split_1d(item_emb, &self.splits)?;

        let mut user = if !self.is_seq && !self.is_stacked && user_emb.ndim() == 2 {
            self.cached_user_was_expanded = true;
            user_emb.reshape(&[user_emb.shape()[0], 1, user_emb.shape()[1]])
        } else {
            self.cached_user_was_expanded = false;
            user_emb.clone()
        };

        self.cached_users.clear();
        self.cached_weights.clear();
        self.cached_biases.clear();
        self.cached_activations.clear();

        for i in 0..self.layer_num {
            let weight = params[2 * i].reshape(&weight_shape(
                user_emb,
                u_emb_size,
                self.is_seq,
                self.is_stacked,
            ));
            let bias = params[2 * i + 1].reshape(&bias_shape(
                user_emb,
                u_emb_size,
                self.is_seq,
                self.is_stacked,
            ));
            let mut out = user.matmul(&weight);
            let bias_b = bias.broadcast_as(out.shape());
            out = out.add(&bias_b);
            let mut act = self.activation.clone();
            out = act.forward_train(&out)?;
            self.cached_activations.push(act);
            self.cached_users.push(user.clone());
            self.cached_weights.push(weight.clone());
            self.cached_biases.push(bias.clone());
            user = out;
        }

        if self.is_seq && self.is_stacked {
            Ok(user.sum_axis(2))
        } else if self.is_seq && !self.is_stacked {
            Ok(user.sum_axis(1))
        } else if !self.is_seq && !self.is_stacked {
            Ok(user.squeeze(1))
        } else {
            Ok(user)
        }
    }

    pub fn backward(&mut self, grad: &Tensor) -> Result<(Tensor, Tensor), LayerError> {
        if self.cached_users.is_empty() {
            return Err(LayerError::NotInitialized);
        }
        let last_user = self.cached_users.last().ok_or(LayerError::NotInitialized)?;
        let mut grad_user = if self.is_seq && self.is_stacked {
            let (b, num_feat, seq_len, d) = (
                last_user.shape()[0],
                last_user.shape()[1],
                last_user.shape()[2],
                last_user.shape()[3],
            );
            grad.reshape(&[b, num_feat, 1, d])
                .broadcast_as(&[b, num_feat, seq_len, d])
        } else if self.is_seq && !self.is_stacked {
            let (b, seq_len, d) = (
                last_user.shape()[0],
                last_user.shape()[1],
                last_user.shape()[2],
            );
            grad.reshape(&[b, 1, d]).broadcast_as(&[b, seq_len, d])
        } else if !self.is_seq && !self.is_stacked {
            let (b, d) = (grad.shape()[0], grad.shape()[1]);
            grad.reshape(&[b, 1, d])
        } else {
            grad.clone()
        };

        let mut grad_item_parts: Vec<Tensor> = Vec::new();

        for idx in (0..self.layer_num).rev() {
            let user_in = &self.cached_users[idx];
            let weight = &self.cached_weights[idx];
            let act = &mut self.cached_activations[idx];

            let grad_act = act.backward(&grad_user)?;
            let weight_t = weight.transpose_dims(weight.ndim() - 2, weight.ndim() - 1);
            let grad_user_prev = grad_act.matmul(&weight_t);

            let user_t = user_in.transpose_dims(user_in.ndim() - 2, user_in.ndim() - 1);
            let mut grad_weight = user_t.matmul(&grad_act);
            if weight.ndim() == 4 && weight.shape()[1] == 1 && grad_weight.shape()[1] > 1 {
                grad_weight = grad_weight.sum_axis(1);
                grad_weight =
                    grad_weight.reshape(&[weight.shape()[0], weight.shape()[2], weight.shape()[3]]);
            }

            let grad_bias = match grad_act.ndim() {
                4 => grad_act.sum_axis(2).sum_axis(1),
                3 => grad_act.sum_axis(1),
                2 => grad_act.clone(),
                _ => grad_act.clone(),
            };

            let grad_weight_flat = if grad_weight.ndim() == 4 {
                grad_weight.squeeze(1).reshape(&[
                    grad_weight.shape()[0],
                    grad_weight.shape()[2] * grad_weight.shape()[3],
                ])
            } else {
                grad_weight.reshape(&[
                    grad_weight.shape()[0],
                    grad_weight.shape()[1] * grad_weight.shape()[2],
                ])
            };
            let grad_bias_flat = if grad_bias.ndim() == 2 {
                grad_bias
            } else {
                let last = *grad_bias.shape().last().unwrap_or(&1);
                grad_bias.reshape(&[grad_bias.shape()[0], last])
            };

            grad_item_parts.push(grad_bias_flat);
            grad_item_parts.push(grad_weight_flat);

            grad_user = grad_user_prev;
        }

        grad_item_parts.reverse();
        let grad_item = Tensor::cat(&grad_item_parts, 1);

        if self.cached_user_was_expanded {
            grad_user = grad_user.squeeze(1);
        }

        Ok((grad_user, grad_item))
    }
}

fn weight_shape(user_emb: &Tensor, dims: usize, is_seq: bool, is_stacked: bool) -> Vec<usize> {
    let batch = user_emb.shape()[0];
    if is_seq && is_stacked {
        vec![batch, 1, dims, dims]
    } else {
        vec![batch, dims, dims]
    }
}

fn bias_shape(user_emb: &Tensor, dims: usize, is_seq: bool, is_stacked: bool) -> Vec<usize> {
    let batch = user_emb.shape()[0];
    if is_seq && is_stacked {
        vec![batch, 1, 1, dims]
    } else {
        vec![batch, 1, dims]
    }
}

fn split_1d(input: &Tensor, splits: &[usize]) -> Result<Vec<Tensor>, LayerError> {
    let mut outputs = Vec::with_capacity(splits.len());
    let mut start = 0;
    for &len in splits {
        outputs.push(input.narrow(1, start, len));
        start += len;
    }
    Ok(outputs)
}

/// CIN layer.
#[derive(Debug, Clone)]
pub struct CIN {
    hidden_units: Vec<usize>,
    activation: Option<ActivationType>,
    initializer: Initializer,
    regularizer: Regularizer,
    conv_weights: Vec<Tensor>,
    conv_bias: Vec<Tensor>,
    conv_weight_grads: Vec<Option<Tensor>>,
    conv_bias_grads: Vec<Option<Tensor>>,
    cached_x0: Option<Tensor>,
    cached_xl: Vec<Tensor>,
    cached_zl: Vec<Tensor>,
    cached_activations: Vec<ActivationLayer>,
}

impl CIN {
    pub fn new(hidden_units: Vec<usize>, activation: Option<ActivationType>) -> Self {
        Self {
            hidden_units,
            activation,
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
            conv_weights: Vec::new(),
            conv_bias: Vec::new(),
            conv_weight_grads: Vec::new(),
            conv_bias_grads: Vec::new(),
            cached_x0: None,
            cached_xl: Vec::new(),
            cached_zl: Vec::new(),
            cached_activations: Vec::new(),
        }
    }

    pub fn with_regularizer(mut self, regularizer: Regularizer) -> Self {
        self.regularizer = regularizer;
        self
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: "CIN expects 3D input".to_string(),
            });
        }
        let (b, num_feat, emb_size) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let mut xl = input.transpose_dims(1, 2);
        let x0 = xl.clone();
        self.cached_x0 = Some(x0.clone());
        let mut outputs = Vec::new();
        self.cached_xl.clear();
        self.cached_zl.clear();
        self.cached_activations.clear();

        for (i, &units) in self.hidden_units.iter().enumerate() {
            let last_hidden = if i == 0 {
                num_feat
            } else {
                self.hidden_units[i - 1]
            };
            if self.conv_weights.len() <= i {
                self.conv_weights.push(
                    self.initializer
                        .initialize(&[last_hidden * num_feat, units]),
                );
                self.conv_bias.push(Tensor::zeros(&[units]));
                self.conv_weight_grads.push(None);
                self.conv_bias_grads.push(None);
            }

            let zl = cin_outer(&xl, &x0)?;
            let zl2d = zl.reshape(&[b * emb_size, last_hidden * num_feat]);
            let mut hl = zl2d.matmul(&self.conv_weights[i]).add(
                &self.conv_bias[i]
                    .reshape(&[1, units])
                    .broadcast_as(&[b * emb_size, units]),
            );
            if let Some(act) = &self.activation {
                if i + 1 == self.hidden_units.len() {
                    self.cached_activations.push(ActivationLayer::None);
                } else {
                    let mut layer = ActivationLayer::from_activation_type(act.clone());
                    hl = layer.forward_train(&hl)?;
                    self.cached_activations.push(layer);
                }
            } else {
                self.cached_activations.push(ActivationLayer::None);
            }
            let hl3d = hl.reshape(&[b, emb_size, units]);
            outputs.push(hl3d.clone());
            self.cached_xl.push(xl.clone());
            self.cached_zl.push(zl);
            xl = hl3d;
        }

        let mut pooled = Vec::new();
        for h in outputs.iter() {
            pooled.push(h.sum_axis(1));
        }
        Ok(Tensor::cat(&pooled, 1))
    }
}

impl Layer for CIN {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        let mut layer = self.clone();
        layer.forward_train(input)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let x0 = self.cached_x0.as_ref().ok_or(LayerError::NotInitialized)?;
        let (b, emb, num_feat) = (x0.shape()[0], x0.shape()[1], x0.shape()[2]);
        let layer_num = self.hidden_units.len();
        if layer_num == 0 {
            return Ok(Tensor::zeros(x0.shape()));
        }

        let mut grad_segments = Vec::new();
        let mut offset = 0;
        for &units in self.hidden_units.iter() {
            grad_segments.push(grad.narrow(1, offset, units));
            offset += units;
        }

        let mut grad_xl = Tensor::zeros(&[b, emb, *self.hidden_units.last().unwrap()]);
        let mut grad_x0_acc = vec![0.0; b * emb * num_feat];

        for i in (0..layer_num).rev() {
            let units = self.hidden_units[i];
            let grad_seg = &grad_segments[i];
            let grad_from_out = grad_seg
                .reshape(&[b, 1, units])
                .broadcast_as(&[b, emb, units]);
            let grad_total = if i == layer_num - 1 {
                grad_from_out
            } else {
                grad_from_out.add(&grad_xl)
            };

            let grad_total_2d = grad_total.reshape(&[b * emb, units]);
            let grad_act = self.cached_activations[i].backward(&grad_total_2d)?;

            let last_hidden = if i == 0 {
                num_feat
            } else {
                self.hidden_units[i - 1]
            };
            let zl = &self.cached_zl[i];
            let zl2d = zl.reshape(&[b * emb, last_hidden * num_feat]);
            let grad_w = zl2d.transpose().matmul(&grad_act);
            let mut grad_w = grad_w;
            if let Some(reg_grad) = self.regularizer.grad(&self.conv_weights[i]) {
                grad_w = grad_w.add(&reg_grad);
            }
            self.conv_weight_grads[i] = Some(grad_w);
            let grad_b = grad_act.sum_axis(0);
            self.conv_bias_grads[i] = Some(grad_b);

            let grad_zl2d = grad_act.matmul(&self.conv_weights[i].transpose());
            let grad_zl = grad_zl2d.reshape(&[b, emb, last_hidden * num_feat]);

            let xl_prev = &self.cached_xl[i];
            let xl_data = xl_prev.data_ref();
            let x0_data = x0.data_ref();
            let grad_zl_data = grad_zl.data_ref();
            let mut grad_xl_prev = vec![0.0; b * emb * last_hidden];

            for bi in 0..b {
                for ei in 0..emb {
                    for hi in 0..last_hidden {
                        let xl_idx = bi * emb * last_hidden + ei * last_hidden + hi;
                        for mi in 0..num_feat {
                            let gz_idx = bi * emb * last_hidden * num_feat
                                + ei * last_hidden * num_feat
                                + hi * num_feat
                                + mi;
                            let g = grad_zl_data[gz_idx];
                            let x0_idx = bi * emb * num_feat + ei * num_feat + mi;
                            grad_xl_prev[xl_idx] += g * x0_data[x0_idx];
                            grad_x0_acc[x0_idx] += g * xl_data[xl_idx];
                        }
                    }
                }
            }

            grad_xl = Tensor::from_data(&[b, emb, last_hidden], grad_xl_prev);
        }

        let grad_x0 = Tensor::from_data(&[b, emb, num_feat], grad_x0_acc);
        Ok(grad_x0.transpose_dims(1, 2))
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        for w in self.conv_weights.iter() {
            params.push(w);
        }
        for b in self.conv_bias.iter() {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        for w in self.conv_weights.iter_mut() {
            params.push(w);
        }
        for b in self.conv_bias.iter_mut() {
            params.push(b);
        }
        params
    }

    fn name(&self) -> &str {
        "CIN"
    }

    fn regularization_loss(&self) -> f32 {
        self.conv_weights
            .iter()
            .map(|w| self.regularizer.loss(w))
            .sum()
    }
}

fn cin_outer(xl: &Tensor, x0: &Tensor) -> Result<Tensor, LayerError> {
    let (b, emb, h) = (xl.shape()[0], xl.shape()[1], xl.shape()[2]);
    let m = x0.shape()[2];
    let xl_data = xl.data_ref();
    let x0_data = x0.data_ref();
    let mut out = vec![0.0; b * emb * h * m];
    for bi in 0..b {
        for ei in 0..emb {
            for hi in 0..h {
                let xl_idx = bi * emb * h + ei * h + hi;
                for mi in 0..m {
                    let x0_idx = bi * emb * m + ei * m + mi;
                    out[bi * emb * h * m + ei * h * m + hi * m + mi] =
                        xl_data[xl_idx] * x0_data[x0_idx];
                }
            }
        }
    }
    Ok(Tensor::from_data(&[b, emb, h * m], out))
}
