//! Feature transformation layers (AutoInt, iRazor).

use crate::error::LayerError;
use crate::initializer::Initializer;
use crate::layer::Layer;
use crate::merge::{merge_tensor_list, merge_tensor_list_tensor, MergeOutput, MergeType};
use crate::regularizer::Regularizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// Configuration for AutoInt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoIntConfig {
    pub layer_num: usize,
    pub out_type: MergeType,
    pub keep_list: bool,
}

impl AutoIntConfig {
    pub fn new(layer_num: usize) -> Self {
        Self {
            layer_num,
            out_type: MergeType::Concat,
            keep_list: false,
        }
    }

    pub fn with_out_type(mut self, out_type: MergeType) -> Self {
        self.out_type = out_type;
        self
    }

    pub fn with_keep_list(mut self, keep_list: bool) -> Self {
        self.keep_list = keep_list;
        self
    }
}

/// AutoInt layer: self-attention based feature interaction.
#[derive(Debug, Clone)]
pub struct AutoInt {
    layer_num: usize,
    out_type: MergeType,
    keep_list: bool,
    cached_inputs: Vec<Tensor>,
    cached_attn: Vec<Tensor>,
    cached_out_shape: Option<Vec<usize>>,
}

impl AutoInt {
    pub fn new(layer_num: usize) -> Self {
        Self {
            layer_num,
            out_type: MergeType::Concat,
            keep_list: false,
            cached_inputs: Vec::new(),
            cached_attn: Vec::new(),
            cached_out_shape: None,
        }
    }

    pub fn from_config(config: AutoIntConfig) -> Self {
        Self {
            layer_num: config.layer_num,
            out_type: config.out_type,
            keep_list: config.keep_list,
            cached_inputs: Vec::new(),
            cached_attn: Vec::new(),
            cached_out_shape: None,
        }
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_inputs.clear();
        self.cached_attn.clear();
        self.cached_out_shape = None;
        let output = self.forward_internal(input, true)?;
        Ok(output)
    }

    pub fn forward_with_merge(&mut self, input: &Tensor) -> Result<MergeOutput, LayerError> {
        let out = self.forward_internal(input, false)?;
        let output = merge_tensor_list(vec![out], self.out_type, None, 1, self.keep_list);
        Ok(output)
    }

    fn forward_internal(&mut self, input: &Tensor, cache: bool) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!("AutoInt expects 3D input, got {}D", input.ndim()),
            });
        }
        let mut x = input.clone();
        for _ in 0..self.layer_num {
            if cache {
                self.cached_inputs.push(x.clone());
            }
            let scores = x.matmul(&x.transpose_dims(1, 2));
            let attn = scores.softmax(2);
            if cache {
                self.cached_attn.push(attn.clone());
            }
            x = attn.matmul(&x);
        }
        if cache {
            self.cached_out_shape = Some(x.shape().to_vec());
        }
        Ok(x)
    }

    fn softmax_backward_3d_axis2(softmax: &Tensor, grad: &Tensor) -> Tensor {
        let dot = grad.mul(softmax).sum_axis(2);
        let (b, f, _) = (softmax.shape()[0], softmax.shape()[1], softmax.shape()[2]);
        let dot_b = dot
            .reshape(&[b, f, 1])
            .broadcast_as(&[b, f, softmax.shape()[2]]);
        softmax.mul(&grad.sub(&dot_b))
    }
}

impl Layer for AutoInt {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!("AutoInt expects 3D input, got {}D", input.ndim()),
            });
        }
        let mut x = input.clone();
        for _ in 0..self.layer_num {
            let scores = x.matmul(&x.transpose_dims(1, 2));
            let attn = scores.softmax(2);
            x = attn.matmul(&x);
        }
        match self.out_type {
            MergeType::Concat => Ok(merge_tensor_list_tensor(
                vec![x],
                MergeType::Concat,
                None,
                1,
            )),
            MergeType::Stack => Ok(x),
            MergeType::None => Err(LayerError::ForwardError {
                message: "AutoInt forward cannot return list when out_type is None".to_string(),
            }),
        }
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        if self.cached_inputs.is_empty() || self.cached_attn.is_empty() {
            return Err(LayerError::NotInitialized);
        }
        let mut g = match self.out_type {
            MergeType::Concat => {
                let shape = self
                    .cached_out_shape
                    .clone()
                    .ok_or(LayerError::NotInitialized)?;
                grad.reshape(&shape)
            }
            MergeType::Stack => grad.clone(),
            MergeType::None => {
                return Err(LayerError::BackwardError {
                    message: "AutoInt backward cannot accept list output".to_string(),
                })
            }
        };

        for idx in (0..self.layer_num).rev() {
            let x_prev = &self.cached_inputs[idx];
            let attn = &self.cached_attn[idx];

            let grad_attn = g.matmul(&x_prev.transpose_dims(1, 2));
            let grad_x_from_out = attn.transpose_dims(1, 2).matmul(&g);

            let grad_scores = Self::softmax_backward_3d_axis2(attn, &grad_attn);
            let grad_x_from_scores = grad_scores
                .matmul(x_prev)
                .add(&grad_scores.transpose_dims(1, 2).matmul(x_prev));

            g = grad_x_from_out.add(&grad_x_from_scores);
        }

        Ok(g)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn name(&self) -> &str {
        "AutoInt"
    }
}

/// Configuration for iRazor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRazorConfig {
    pub num_feature: usize,
    pub nas_space: Vec<usize>,
    pub t: f32,
    pub initializer: Initializer,
    pub regularizer: Regularizer,
    pub out_type: MergeType,
    pub keep_list: bool,
    pub feature_weight: Option<Vec<f32>>,
}

impl IRazorConfig {
    pub fn new(num_feature: usize, nas_space: Vec<usize>) -> Self {
        Self {
            num_feature,
            nas_space,
            t: 0.05,
            initializer: Initializer::GlorotUniform,
            regularizer: Regularizer::None,
            out_type: MergeType::Concat,
            keep_list: false,
            feature_weight: None,
        }
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.t = t;
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

    pub fn with_out_type(mut self, out_type: MergeType) -> Self {
        self.out_type = out_type;
        self
    }

    pub fn with_keep_list(mut self, keep_list: bool) -> Self {
        self.keep_list = keep_list;
        self
    }

    pub fn with_feature_weight(mut self, feature_weight: Vec<f32>) -> Self {
        self.feature_weight = Some(feature_weight);
        self
    }
}

/// iRazor layer for feature selection and embedding dimension search.
#[derive(Debug, Clone)]
pub struct IRazor {
    num_feature: usize,
    nas_space: Vec<usize>,
    t: f32,
    regularizer: Regularizer,
    nas_logits: Tensor,
    nas_logits_grad: Option<Tensor>,
    rigid_masks: Tensor,
    out_type: MergeType,
    keep_list: bool,
    feature_weight: Option<Tensor>,
    cached_input: Option<Tensor>,
    cached_soft_masks: Option<Tensor>,
    cached_nas_weight: Option<Tensor>,
    aux_loss: Option<f32>,
}

impl IRazor {
    pub fn from_config(config: IRazorConfig) -> Self {
        let nas_len = config.nas_space.len();
        let emb_size = *config.nas_space.iter().max().unwrap_or(&0);
        let nas_logits = config
            .initializer
            .initialize(&[config.num_feature, nas_len]);

        let mut masks = vec![0.0f32; nas_len * emb_size];
        for i in 1..nas_len {
            let start = config.nas_space[i - 1];
            let end = config.nas_space[i];
            for j in start..end {
                masks[i * emb_size + j] = 1.0;
            }
        }
        let rigid_masks = Tensor::from_data(&[nas_len, emb_size], masks);

        let feature_weight = config
            .feature_weight
            .map(|fw| Tensor::from_data(&[1, fw.len()], fw));

        Self {
            num_feature: config.num_feature,
            nas_space: config.nas_space,
            t: config.t,
            regularizer: config.regularizer,
            nas_logits,
            nas_logits_grad: None,
            rigid_masks,
            out_type: config.out_type,
            keep_list: config.keep_list,
            feature_weight,
            cached_input: None,
            cached_soft_masks: None,
            cached_nas_weight: None,
            aux_loss: None,
        }
    }

    pub fn forward_train(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        self.cached_input = Some(input.clone());
        let output = self.forward_internal(input)?;
        Ok(output)
    }

    pub fn forward_with_merge(&mut self, input: &Tensor) -> Result<MergeOutput, LayerError> {
        let out = self.forward_internal(input)?;
        Ok(merge_tensor_list(
            vec![out],
            self.out_type,
            Some(self.num_feature),
            1,
            self.keep_list,
        ))
    }

    pub fn aux_loss(&self) -> Option<f32> {
        self.aux_loss
    }

    /// Returns regularization loss including auxiliary loss if present.
    pub fn regularization_loss(&self) -> f32 {
        let mut loss = self.regularizer.loss(&self.nas_logits);
        if let Some(aux) = self.aux_loss {
            loss += aux;
        }
        loss
    }

    fn forward_internal(&mut self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!("iRazor expects 3D input, got {}D", input.ndim()),
            });
        }
        let emb_size = *self.nas_space.iter().max().unwrap_or(&0);
        if input.shape()[1] != self.num_feature || input.shape()[2] != emb_size {
            return Err(LayerError::ShapeMismatch {
                expected: vec![self.num_feature, emb_size],
                actual: vec![input.shape()[1], input.shape()[2]],
            });
        }

        let nas_weight = self.nas_logits.scale(1.0 / self.t).softmax(1);
        let soft_masks = nas_weight.matmul(&self.rigid_masks);

        if let Some(feature_weight) = &self.feature_weight {
            let sum_masks = soft_masks.sum_axis(1).reshape(&[self.num_feature, 1]);
            let loss = feature_weight.matmul(&sum_masks).sum();
            self.aux_loss = Some(loss);
        } else {
            self.aux_loss = None;
        }

        let (b, f, d) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let masks_b = soft_masks.reshape(&[1, f, d]).broadcast_as(&[b, f, d]);
        let output = input.mul(&masks_b);

        self.cached_soft_masks = Some(soft_masks);
        self.cached_nas_weight = Some(nas_weight);

        Ok(output)
    }

    fn softmax_backward_2d(softmax: &Tensor, grad: &Tensor) -> Tensor {
        let dot = grad.mul(softmax).sum_axis(1);
        let m = softmax.shape()[0];
        let n = softmax.shape()[1];
        let dot_b = dot.reshape(&[m, 1]).broadcast_as(&[m, n]);
        softmax.mul(&grad.sub(&dot_b))
    }
}

impl Layer for IRazor {
    fn forward(&self, input: &Tensor) -> Result<Tensor, LayerError> {
        if input.ndim() != 3 {
            return Err(LayerError::ForwardError {
                message: format!("iRazor expects 3D input, got {}D", input.ndim()),
            });
        }
        let mut layer = self.clone();
        layer.forward_internal(input)
    }

    fn backward(&mut self, grad: &Tensor) -> Result<Tensor, LayerError> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let soft_masks = self
            .cached_soft_masks
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;
        let nas_weight = self
            .cached_nas_weight
            .as_ref()
            .ok_or(LayerError::NotInitialized)?;

        let (b, f, d) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        let masks_b = soft_masks.reshape(&[1, f, d]).broadcast_as(&[b, f, d]);
        let grad_input = grad.mul(&masks_b);

        let grad_soft_masks = grad.mul(input).sum_axis(0);
        let grad_nas_weight = grad_soft_masks.matmul(&self.rigid_masks.transpose());
        let grad_z = Self::softmax_backward_2d(nas_weight, &grad_nas_weight).scale(1.0 / self.t);
        self.nas_logits_grad = Some(grad_z);

        Ok(grad_input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.nas_logits]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.nas_logits]
    }

    fn name(&self) -> &str {
        "IRazor"
    }
}
