//! Monolith RMSprop optimizer.
//!
//! This matches the custom TensorFlow kernels used by
//! `monolith/native_training/optimizers/rmsprop.py`.
//!
//! There are two variants controlled by `use_v2`:
//! - v1: `ResourceApplyRmsprop` (slot update uses `(grad^2 - v) * (1 - beta2)` and rsqrt(v+eps))
//! - v2: `ResourceApplyRmspropV2` (slot update uses `v = beta2*v + grad^2` and `1/(sqrt(v)+eps)`)

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// Monolith RMSprop optimizer mirroring the TF custom op behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonolithRmsprop {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    use_v2: bool,
    // Slot state.
    m: Vec<f32>,
    v: Vec<f32>,
    config: OptimizerConfig,
}

impl MonolithRmsprop {
    pub fn reset_state(&mut self) {
        self.m.clear();
        self.v.clear();
    }

    pub fn m(&self) -> &[f32] {
        &self.m
    }

    pub fn v(&self) -> &[f32] {
        &self.v
    }
}

impl Optimizer for MonolithRmsprop {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::MonolithRmsprop {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                use_v2,
            } => Ok(Self {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                use_v2,
                m: Vec::new(),
                v: Vec::new(),
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "MonolithRmsprop".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        if self.m.len() != embedding.len() {
            self.m = vec![0.0; embedding.len()];
            self.v = vec![0.0; embedding.len()];
        }

        for i in 0..embedding.len() {
            let var = embedding[i];
            let grad_after_decay = self.weight_decay * var + gradients[i];

            if self.use_v2 {
                // v = beta2*v + grad^2
                self.v[i] = self.beta2 * self.v[i] + grad_after_decay * grad_after_decay;
                // m = beta1*m + (grad*lr)/(sqrt(v)+eps)
                self.m[i] = self.beta1 * self.m[i]
                    + (grad_after_decay * self.learning_rate) / (self.v[i].sqrt() + self.epsilon);
            } else {
                // v += (grad^2 - v) * (1 - beta2)
                self.v[i] += (grad_after_decay * grad_after_decay - self.v[i]) * (1.0 - self.beta2);
                // m = beta1*m + (grad*lr) * rsqrt(v+eps)
                self.m[i] = self.beta1 * self.m[i]
                    + (grad_after_decay * self.learning_rate) / (self.v[i] + self.epsilon).sqrt();
            }

            embedding[i] = var - self.m[i];
        }
    }

    fn config(&self) -> &OptimizerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monolith_rmsprop_matches_python_test_values_v1() {
        // Mirrors monolith/native_training/optimizers/rmsprop_test.py::testBasic
        let mut opt = MonolithRmsprop::new(OptimizerConfig::MonolithRmsprop {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.9,
            epsilon: 0.1,
            weight_decay: 1.0,
            use_v2: false,
        })
        .unwrap();

        let mut var = vec![0.1_f32];
        let grad = vec![0.12_f32]; // loss = 0.12 * var
        opt.apply_gradients(&mut var, &grad);

        let eps = 1e-8_f32;
        assert!((opt.m()[0] - 0.067_945_26).abs() < eps, "{:?}", opt.m());
        assert!((opt.v()[0] - 0.004_84).abs() < eps, "{:?}", opt.v());
        assert!((var[0] - 0.032_054_738).abs() < eps, "{var:?}");
    }

    #[test]
    fn monolith_rmsprop_matches_python_test_values_v2() {
        // Mirrors monolith/native_training/optimizers/rmspropv2_test.py::testBasic
        let mut opt = MonolithRmsprop::new(OptimizerConfig::MonolithRmsprop {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.9,
            epsilon: 0.1,
            weight_decay: 1.0,
            use_v2: true,
        })
        .unwrap();

        let mut var = vec![0.1_f32];
        let grad = vec![0.12_f32];
        opt.apply_gradients(&mut var, &grad);

        let eps = 1e-8_f32;
        assert!((opt.m()[0] - 0.068_75).abs() < eps, "{:?}", opt.m());
        assert!((opt.v()[0] - 0.048_4).abs() < eps, "{:?}", opt.v());
        assert!((var[0] - 0.031_25).abs() < eps, "{var:?}");
    }
}
