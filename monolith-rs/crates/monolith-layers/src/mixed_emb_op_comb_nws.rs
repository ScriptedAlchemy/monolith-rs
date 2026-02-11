//! Port of Python `monolith/core/mixed_emb_op_comb_nws.py` (sampling-based NWS path).
//!
//! The Python layer is TF/Keras oriented and returns either masked embeddings or
//! (optionally) distillation outputs. In Rust we implement the core math for the
//! sampling-based masking path:
//! - Build per-slot choice probabilities from trainable logits (softmax)
//! - During pretraining steps, use uniform probabilities (0.5, 0.5) for 2 choices
//!   (mirrors Python test setups; the original code comments mention 4-choice too)
//! - Sample a single choice per slot and build a binary mask over the embedding slice
//! - Apply the straight-through estimator multiplier:
//!     mask * (1 + p_chosen - stop_gradient(p_chosen))
//!   In Rust forward-only, `stop_gradient(p_chosen)` is just `p_chosen`, so the multiplier is 1.
//!
//! This module is forward-only (no autograd) and intended for parity with masking semantics.

use crate::error::LayerError;
use crate::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct MixedEmbedOpCombNws {
    slot_names: Vec<String>,
    embedding_size_choices_list: Vec<Vec<usize>>,
    warmup_steps: usize,
    pretraining_steps: usize,

    num_choices_per_embedding: Vec<usize>,
    max_num_choices: usize,
    max_choice_per_embedding: Vec<usize>,
    total_emb_size: usize,

    /// Trainable architecture logits flattened across slots.
    arch_embedding_weights: Tensor, // shape [sum(num_choices)]
}

impl MixedEmbedOpCombNws {
    pub fn new(
        slot_names: Vec<String>,
        embedding_size_choices_list: Vec<Vec<usize>>,
        warmup_steps: usize,
        pretraining_steps: usize,
    ) -> Result<Self, LayerError> {
        if slot_names.len() != embedding_size_choices_list.len() {
            return Err(LayerError::ConfigError {
                message: "slot_names and embedding_size_choices_list must have same length"
                    .to_string(),
            });
        }
        if slot_names.is_empty() {
            return Err(LayerError::ConfigError {
                message: "must provide at least one slot".to_string(),
            });
        }

        let mut num_choices_per_embedding = Vec::with_capacity(slot_names.len());
        let mut max_choice_per_embedding = Vec::with_capacity(slot_names.len());
        let mut max_num_choices = 0usize;
        let mut total_emb_size = 0usize;
        for choices in embedding_size_choices_list.iter() {
            if choices.is_empty() {
                return Err(LayerError::ConfigError {
                    message: "each slot must have at least one embedding choice".to_string(),
                });
            }
            max_num_choices = max_num_choices.max(choices.len());
            num_choices_per_embedding.push(choices.len());
            let sum: usize = choices.iter().sum();
            max_choice_per_embedding.push(sum);
            total_emb_size += sum;
        }

        let total_choices: usize = num_choices_per_embedding.iter().sum();
        // Python initializes with uniform(-1e-3, 1e-3)
        let mut data = Vec::with_capacity(total_choices);
        let mut seed: u64 = 42;
        for _ in 0..total_choices {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let u = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
            let v = (2.0 * u - 1.0) * 1e-3;
            data.push(v);
        }
        let arch_embedding_weights = Tensor::from_data(&[total_choices], data);

        Ok(Self {
            slot_names,
            embedding_size_choices_list,
            warmup_steps,
            pretraining_steps,
            num_choices_per_embedding,
            max_num_choices,
            max_choice_per_embedding,
            total_emb_size,
            arch_embedding_weights,
        })
    }

    pub fn total_emb_size(&self) -> usize {
        self.total_emb_size
    }

    pub fn arch_embedding_weights(&self) -> &Tensor {
        &self.arch_embedding_weights
    }

    /// Forward pass: returns `masked_embedding` with the sampled mask applied.
    ///
    /// `global_step` mirrors TF global step for warmup/pretraining behavior.
    pub fn forward(&self, embedding: &Tensor, global_step: usize) -> Result<Tensor, LayerError> {
        if embedding.ndim() != 2 {
            return Err(LayerError::ForwardError {
                message: format!("expected 2D embedding input, got {}D", embedding.ndim()),
            });
        }
        if embedding.shape()[1] != self.total_emb_size {
            return Err(LayerError::InvalidInputDimension {
                expected: self.total_emb_size,
                actual: embedding.shape()[1],
            });
        }

        // Build and apply mask multipliers concatenated across slots (shape [total_emb_size])
        let mask = self.build_arch_embedding_mask(global_step)?;
        let mask_b = mask
            .reshape(&[1, self.total_emb_size])
            .broadcast_as(embedding.shape());
        Ok(embedding.mul(&mask_b))
    }

    fn build_arch_embedding_mask(&self, global_step: usize) -> Result<Tensor, LayerError> {
        let mut masks: Vec<Tensor> = Vec::with_capacity(self.slot_names.len());
        let mut current_idx = 0usize;

        for (slot_i, choices) in self.embedding_size_choices_list.iter().enumerate() {
            let num_choices = self.num_choices_per_embedding[slot_i];
            let max_emb_choice = self.max_choice_per_embedding[slot_i];
            let logits = self
                .arch_embedding_weights
                .narrow(0, current_idx, num_choices);
            current_idx += num_choices;

            // Per-slot softmax selection.
            let mut probs = logits.softmax(0);

            // Warmup for 0-size embedding first option.
            if choices[0] == 0 {
                let warm = (global_step as f32 - self.warmup_steps as f32)
                    .max(0.0)
                    .min(1.0);
                // Multiply first prob by warmup scalar.
                let mut p = probs.data().to_vec();
                p[0] *= warm;
                probs = Tensor::from_data(&[p.len()], p);
                // Renormalize (avoid div by 0 for warmup==0).
                let sum = probs.data().iter().copied().sum::<f32>().max(1e-12);
                probs = probs.scale(1.0 / sum);
            }

            // Pretraining override: for now support the common 2-choice case used in the python code.
            let probs_for_sampling = if global_step < self.pretraining_steps {
                if self.max_num_choices != 2 {
                    return Err(LayerError::ForwardError {
                        message:
                            "pretraining uniform sampling currently implemented for 2 choices only"
                                .to_string(),
                    });
                }
                Tensor::from_data(&[2], vec![0.5, 0.5])
            } else {
                // Pad to max_num_choices if needed.
                if num_choices < self.max_num_choices {
                    let mut p = probs.data().to_vec();
                    p.resize(self.max_num_choices, 0.0);
                    Tensor::from_data(&[self.max_num_choices], p)
                } else {
                    probs.clone()
                }
            };

            let sampled = sample_categorical_1d(&probs_for_sampling, slot_i as u64)?;
            let p_chosen = probs_for_sampling.data()[sampled];

            // Build binary mask of length `max_emb_choice` for this slot.
            let mut one_slot = vec![0.0f32; max_emb_choice];
            let mut lower = 0usize;
            for (j, &size) in choices.iter().enumerate() {
                let upper = lower + size;
                if j == sampled {
                    for k in lower..upper {
                        if k < max_emb_choice {
                            one_slot[k] = 1.0;
                        }
                    }
                }
                lower = upper;
            }

            // Straight-through estimator multiplier:
            // `1 + p - stop_gradient(p)` => `1` in a forward-only implementation.
            // Keep the multiplier explicit for readability/parity.
            let _ste_multiplier = 1.0f32 + p_chosen - p_chosen;
            masks.push(Tensor::from_data(&[max_emb_choice], one_slot));
        }

        Ok(Tensor::cat(&masks, 0))
    }
}

fn sample_categorical_1d(probs: &Tensor, salt: u64) -> Result<usize, LayerError> {
    if probs.ndim() != 1 {
        return Err(LayerError::ForwardError {
            message: "categorical expects 1D probs".to_string(),
        });
    }
    let p = probs.data();
    let mut sum = 0.0f32;
    for &v in p.iter() {
        if v < 0.0 {
            return Err(LayerError::ForwardError {
                message: "categorical probs must be non-negative".to_string(),
            });
        }
        sum += v;
    }
    if sum <= 0.0 {
        return Err(LayerError::ForwardError {
            message: "categorical probs sum to zero".to_string(),
        });
    }
    // Deterministic RNG for tests.
    let mut seed: u64 = 42 ^ salt.wrapping_mul(0x9E3779B97F4A7C15);
    seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    let u = ((seed >> 16) & 0x7fff) as f32 / 32768.0;
    let mut r = u * sum;
    for (i, &v) in p.iter().enumerate() {
        if r < v {
            return Ok(i);
        }
        r -= v;
    }
    Ok(p.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_output_shape_and_dim_validation() {
        let layer =
            MixedEmbedOpCombNws::new(vec!["a".to_string()], vec![vec![2, 3]], 0, 0).unwrap();

        let x = Tensor::ones(&[4, 5]);
        let y = layer.forward(&x, 10).unwrap();
        assert_eq!(y.shape(), &[4, 5]);

        let bad = Tensor::ones(&[4, 6]);
        assert!(layer.forward(&bad, 10).is_err());
    }

    #[test]
    fn test_masking_is_binary_per_slot_slice() {
        let layer =
            MixedEmbedOpCombNws::new(vec!["a".to_string()], vec![vec![2, 3]], 0, 0).unwrap();

        let x = Tensor::ones(&[1, 5]);
        let y = layer.forward(&x, 10).unwrap();
        let d = y.data();
        // Exactly 2 or 3 positions should be non-zero, depending on sampled choice.
        let nnz = d.iter().filter(|&&v| v != 0.0).count();
        assert!(nnz == 2 || nnz == 3);
    }
}
