//! Candle-backed inference models for Monolith serving.
//!
//! This module is intentionally focused on inference-time parity with the Python
//! Monolith serving behavior, but implemented using Candle (GPU-friendly) rather
//! than hand-rolled tensor math.
//!
//! The key idea is:
//! - Training/export produces a "SavedModel-like" directory (see monolith-checkpoint),
//!   containing `dense/params.json` plus `model_spec.json` describing the network.
//! - Serving loads `model_spec.json` and creates an inference graph backed by Candle.
//!
//! This does NOT attempt to execute arbitrary TensorFlow graphs. It targets the
//! Rust-native model families we support: MLP, DCN, MMoE (and will extend to DIN/DIEN).

use crate::error::{ServingError, ServingResult};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model specification stored next to an exported model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ModelSpec {
    /// Simple feed-forward network.
    Mlp(MlpSpec),
    /// Deep & Cross Network.
    Dcn(DcnSpec),
    /// Multi-gate Mixture-of-Experts.
    Mmoe(MmoeSpec),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlpSpec {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub output_dim: usize,
    #[serde(default)]
    pub activation: Activation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DcnSpec {
    pub input_dim: usize,
    pub cross_layers: usize,
    /// Deep tower hidden dims.
    #[serde(default)]
    pub deep_hidden_dims: Vec<usize>,
    pub output_dim: usize,
    #[serde(default)]
    pub activation: Activation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MmoeSpec {
    pub input_dim: usize,
    pub num_experts: usize,
    pub expert_hidden_dims: Vec<usize>,
    pub num_tasks: usize,
    pub gate_hidden_dims: Vec<usize>,
    pub task_output_dim: usize,
    #[serde(default)]
    pub activation: Activation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    #[default]
    Relu,
    Tanh,
    Sigmoid,
    None,
}

impl Activation {
    fn apply(&self, t: Tensor) -> candle_core::Result<Tensor> {
        match self {
            Activation::Relu => t.relu(),
            Activation::Tanh => t.tanh(),
            Activation::Sigmoid => candle_nn::ops::sigmoid(&t),
            Activation::None => Ok(t),
        }
    }
}

/// Inference model interface (single-example inferences for now).
pub trait InferenceModel: Send + Sync {
    fn input_dim(&self) -> usize;
    fn predict(&self, input: &Tensor) -> ServingResult<Tensor>;
}

/// Load a candle model from a spec + dense params (flat f32 arrays).
pub fn build_model(
    spec: &ModelSpec,
    params: &HashMap<String, Vec<f32>>,
    device: &Device,
) -> ServingResult<Box<dyn InferenceModel>> {
    match spec {
        ModelSpec::Mlp(s) => Ok(Box::new(MlpModel::from_params(s, params, device)?)),
        ModelSpec::Dcn(s) => Ok(Box::new(DcnModel::from_params(s, params, device)?)),
        ModelSpec::Mmoe(s) => Ok(Box::new(MmoeModel::from_params(s, params, device)?)),
    }
}

fn tensor_from_vec(
    params: &HashMap<String, Vec<f32>>,
    name: &str,
    shape: &[usize],
    device: &Device,
) -> ServingResult<Tensor> {
    let data = params
        .get(name)
        .ok_or_else(|| ServingError::ModelLoadError(format!("Missing dense param {:?}", name)))?;
    let numel: usize = shape.iter().product();
    if data.len() != numel {
        return Err(ServingError::ModelLoadError(format!(
            "Param {:?} has len {}, expected {} for shape {:?}",
            name,
            data.len(),
            numel,
            shape
        )));
    }
    Tensor::from_slice(data, shape, device)
        .map_err(|e| ServingError::ModelLoadError(format!("Candle tensor init failed: {e}")))
}

fn linear(x: &Tensor, w: &Tensor, b: Option<&Tensor>) -> ServingResult<Tensor> {
    // x: [B, in], w: [out, in] (Candle's matmul expects [B,in] @ [in,out]).
    let wt = w
        .t()
        .map_err(|e| ServingError::PredictionError(format!("transpose failed: {e}")))?;
    let y = x
        .matmul(&wt)
        .map_err(|e| ServingError::PredictionError(format!("matmul failed: {e}")))?;
    let y = match b {
        Some(b) => y
            .broadcast_add(b)
            .map_err(|e| ServingError::PredictionError(format!("bias add failed: {e}")))?,
        None => y,
    };
    Ok(y)
}

#[derive(Debug)]
struct MlpModel {
    spec: MlpSpec,
    // layers: (w, b)
    weights: Vec<(Tensor, Tensor)>,
    device: Device,
}

impl MlpModel {
    fn from_params(
        spec: &MlpSpec,
        params: &HashMap<String, Vec<f32>>,
        device: &Device,
    ) -> ServingResult<Self> {
        let mut weights: Vec<(Tensor, Tensor)> = Vec::new();
        let mut in_dim = spec.input_dim;

        let mut all_layers: Vec<usize> = spec.hidden_dims.clone();
        all_layers.push(spec.output_dim);

        for (i, &out_dim) in all_layers.iter().enumerate() {
            let w_name = format!("mlp.layers.{i}.weight");
            let b_name = format!("mlp.layers.{i}.bias");
            let w = tensor_from_vec(params, &w_name, &[out_dim, in_dim], device)?;
            let b = tensor_from_vec(params, &b_name, &[out_dim], device)?;
            weights.push((w, b));
            in_dim = out_dim;
        }

        Ok(Self {
            spec: spec.clone(),
            weights,
            device: device.clone(),
        })
    }
}

impl InferenceModel for MlpModel {
    fn input_dim(&self) -> usize {
        self.spec.input_dim
    }

    fn predict(&self, input: &Tensor) -> ServingResult<Tensor> {
        let mut x = input.clone();
        for (i, (w, b)) in self.weights.iter().enumerate() {
            x = linear(&x, w, Some(b))?;
            let is_last = i + 1 == self.weights.len();
            if !is_last {
                x = self.spec.activation.apply(x).map_err(|e| {
                    ServingError::PredictionError(format!("activation failed: {e}"))
                })?;
            }
        }
        Ok(x.to_device(&self.device)
            .map_err(|e| ServingError::PredictionError(format!("device move failed: {e}")))?)
    }
}

#[derive(Debug)]
struct DcnModel {
    spec: DcnSpec,
    // Cross layers: w: [in], b: [in]
    cross_w: Vec<Tensor>,
    cross_b: Vec<Tensor>,
    // Deep tower MLP (optional)
    deep: Option<MlpModel>,
    // Final projection
    out_w: Tensor,
    out_b: Tensor,
    device: Device,
}

impl DcnModel {
    fn from_params(
        spec: &DcnSpec,
        params: &HashMap<String, Vec<f32>>,
        device: &Device,
    ) -> ServingResult<Self> {
        let mut cross_w = Vec::new();
        let mut cross_b = Vec::new();
        for i in 0..spec.cross_layers {
            let w_name = format!("dcn.cross.{i}.w");
            let b_name = format!("dcn.cross.{i}.b");
            cross_w.push(tensor_from_vec(params, &w_name, &[spec.input_dim], device)?);
            cross_b.push(tensor_from_vec(params, &b_name, &[spec.input_dim], device)?);
        }

        let deep = if spec.deep_hidden_dims.is_empty() {
            None
        } else {
            let mlp_spec = MlpSpec {
                input_dim: spec.input_dim,
                hidden_dims: spec.deep_hidden_dims.clone(),
                output_dim: spec
                    .deep_hidden_dims
                    .last()
                    .copied()
                    .unwrap_or(spec.input_dim),
                activation: spec.activation,
            };
            Some(MlpModel::from_params(&mlp_spec, params, device)?)
        };

        // output head takes concat([cross_out, deep_out]) if deep exists else cross_out
        let head_in = if let Some(ref deep) = deep {
            spec.input_dim
                + deep
                    .weights
                    .last()
                    .map(|(w, _)| w.dims()[0])
                    .unwrap_or(spec.input_dim)
        } else {
            spec.input_dim
        };

        let out_w = tensor_from_vec(
            params,
            "dcn.out.weight",
            &[spec.output_dim, head_in],
            device,
        )?;
        let out_b = tensor_from_vec(params, "dcn.out.bias", &[spec.output_dim], device)?;

        Ok(Self {
            spec: spec.clone(),
            cross_w,
            cross_b,
            deep,
            out_w,
            out_b,
            device: device.clone(),
        })
    }

    fn cross_forward(&self, x0: &Tensor) -> ServingResult<Tensor> {
        // x0: [B, D]
        let mut x = x0.clone();
        for (w, b) in self.cross_w.iter().zip(self.cross_b.iter()) {
            // x_{l+1} = x0 * (x_l · w) + b + x_l
            // where (x_l · w) is a scalar per batch: [B, D] * [D] -> [B, 1]
            let w2 = w
                .reshape((self.spec.input_dim, 1))
                .map_err(|e| ServingError::PredictionError(format!("reshape failed: {e}")))?;
            let xlw = x
                .matmul(&w2)
                .map_err(|e| ServingError::PredictionError(format!("cross matmul failed: {e}")))?; // [B,1]
            let prod = x0
                .broadcast_mul(&xlw)
                .map_err(|e| ServingError::PredictionError(format!("broadcast mul failed: {e}")))?;
            let prod = prod
                .broadcast_add(b)
                .map_err(|e| ServingError::PredictionError(format!("broadcast add failed: {e}")))?;
            x = (prod + x)
                .map_err(|e| ServingError::PredictionError(format!("add failed: {e}")))?;
        }
        Ok(x)
    }
}

impl InferenceModel for DcnModel {
    fn input_dim(&self) -> usize {
        self.spec.input_dim
    }

    fn predict(&self, input: &Tensor) -> ServingResult<Tensor> {
        let x0 = input.clone();
        let cross_out = self.cross_forward(&x0)?;
        let head_in = if let Some(ref deep) = self.deep {
            let deep_out = deep.predict(&x0)?;
            Tensor::cat(&[cross_out, deep_out], 1)
                .map_err(|e| ServingError::PredictionError(format!("cat failed: {e}")))?
        } else {
            cross_out
        };
        let y = linear(&head_in, &self.out_w, Some(&self.out_b))?;
        Ok(y.to_device(&self.device)
            .map_err(|e| ServingError::PredictionError(format!("device move failed: {e}")))?)
    }
}

#[derive(Debug)]
struct MmoeModel {
    spec: MmoeSpec,
    experts: Vec<MlpModel>,
    // Each gate is an MLP producing logits over experts: output_dim = num_experts
    gates: Vec<MlpModel>,
    // Task heads: linear from expert-mixture hidden -> task_output_dim
    task_w: Vec<Tensor>,
    task_b: Vec<Tensor>,
    device: Device,
}

impl MmoeModel {
    fn from_params(
        spec: &MmoeSpec,
        params: &HashMap<String, Vec<f32>>,
        device: &Device,
    ) -> ServingResult<Self> {
        let mut experts = Vec::new();
        for e in 0..spec.num_experts {
            let mlp_spec = MlpSpec {
                input_dim: spec.input_dim,
                hidden_dims: spec.expert_hidden_dims.clone(),
                output_dim: spec
                    .expert_hidden_dims
                    .last()
                    .copied()
                    .unwrap_or(spec.input_dim),
                activation: spec.activation,
            };
            experts.push(MlpModel::from_params_with_prefix(
                &mlp_spec,
                params,
                device,
                &format!("mmoe.expert.{e}"),
            )?);
        }

        let mut gates = Vec::new();
        for t in 0..spec.num_tasks {
            let gate_spec = MlpSpec {
                input_dim: spec.input_dim,
                hidden_dims: spec.gate_hidden_dims.clone(),
                output_dim: spec.num_experts,
                activation: spec.activation,
            };
            gates.push(MlpModel::from_params_with_prefix(
                &gate_spec,
                params,
                device,
                &format!("mmoe.gate.{t}"),
            )?);
        }

        let hidden_dim = spec
            .expert_hidden_dims
            .last()
            .copied()
            .unwrap_or(spec.input_dim);
        let mut task_w = Vec::new();
        let mut task_b = Vec::new();
        for t in 0..spec.num_tasks {
            task_w.push(tensor_from_vec(
                params,
                &format!("mmoe.task.{t}.weight"),
                &[spec.task_output_dim, hidden_dim],
                device,
            )?);
            task_b.push(tensor_from_vec(
                params,
                &format!("mmoe.task.{t}.bias"),
                &[spec.task_output_dim],
                device,
            )?);
        }

        Ok(Self {
            spec: spec.clone(),
            experts,
            gates,
            task_w,
            task_b,
            device: device.clone(),
        })
    }
}

impl MlpModel {
    fn from_params_with_prefix(
        spec: &MlpSpec,
        params: &HashMap<String, Vec<f32>>,
        device: &Device,
        prefix: &str,
    ) -> ServingResult<Self> {
        let mut weights: Vec<(Tensor, Tensor)> = Vec::new();
        let mut in_dim = spec.input_dim;
        let mut all_layers: Vec<usize> = spec.hidden_dims.clone();
        all_layers.push(spec.output_dim);

        for (i, &out_dim) in all_layers.iter().enumerate() {
            let w_name = format!("{prefix}.layers.{i}.weight");
            let b_name = format!("{prefix}.layers.{i}.bias");
            let w = tensor_from_vec(params, &w_name, &[out_dim, in_dim], device)?;
            let b = tensor_from_vec(params, &b_name, &[out_dim], device)?;
            weights.push((w, b));
            in_dim = out_dim;
        }

        Ok(Self {
            spec: spec.clone(),
            weights,
            device: device.clone(),
        })
    }
}

impl InferenceModel for MmoeModel {
    fn input_dim(&self) -> usize {
        self.spec.input_dim
    }

    fn predict(&self, input: &Tensor) -> ServingResult<Tensor> {
        // Compute expert outputs: [E] of [B, H]
        let mut expert_outs: Vec<Tensor> = Vec::with_capacity(self.experts.len());
        for e in &self.experts {
            expert_outs.push(e.predict(input)?);
        }

        // Stack experts: [B, E, H]
        let stacked = Tensor::stack(&expert_outs, 1)
            .map_err(|e| ServingError::PredictionError(format!("stack failed: {e}")))?;

        // Per-task: gate softmax over experts, then weighted sum.
        let mut task_outputs: Vec<Tensor> = Vec::with_capacity(self.spec.num_tasks);
        for t in 0..self.spec.num_tasks {
            let gate_logits = self.gates[t].predict(input)?; // [B, E]
            let gate = candle_nn::ops::softmax(&gate_logits, 1)
                .map_err(|e| ServingError::PredictionError(format!("softmax failed: {e}")))?; // [B, E]
            let gate = gate
                .unsqueeze(2)
                .map_err(|e| ServingError::PredictionError(format!("unsqueeze failed: {e}")))?; // [B,E,1]
            let weighted = stacked
                .broadcast_mul(&gate)
                .map_err(|e| ServingError::PredictionError(format!("broadcast mul failed: {e}")))?;
            let mixed = weighted
                .sum(1)
                .map_err(|e| ServingError::PredictionError(format!("sum failed: {e}")))?; // [B,H]

            let head = linear(&mixed, &self.task_w[t], Some(&self.task_b[t]))?; // [B, out]
            task_outputs.push(head);
        }

        // For now, return the first task head as the primary score.
        // Python monolith commonly serves a single primary head for ranking.
        let y = task_outputs
            .into_iter()
            .next()
            .ok_or_else(|| ServingError::PredictionError("MMoE has no tasks".into()))?;

        Ok(y.to_device(&self.device)
            .map_err(|e| ServingError::PredictionError(format!("device move failed: {e}")))?)
    }
}
