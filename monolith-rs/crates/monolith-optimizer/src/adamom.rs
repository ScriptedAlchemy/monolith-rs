//! Adamom optimizer.
//!
//! This implements the same update rules as the custom TensorFlow op used by
//! `monolith/native_training/optimizers/adamom.py`.

use crate::{Optimizer, OptimizerConfig, OptimizerError};
use serde::{Deserialize, Serialize};

/// A dedicated config type for convenience when constructing Adamom directly.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AdamomConfig {
    pub learning_rate: f32,
    pub ada_decay: f32,
    pub mom_decay: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for AdamomConfig {
    fn default() -> Self {
        Self {
            learning_rate: 5e-6,
            ada_decay: 0.9999,
            mom_decay: 0.99,
            epsilon: 1e-6,
            weight_decay: 0.0,
        }
    }
}

/// Adamom optimizer.
///
/// Mirrors CPU kernel `ApplyAdamom` in
/// `monolith/native_training/optimizers/cc/kernels/training_ops.cc`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Adamom {
    learning_rate: f32,
    ada_decay: f32,
    mom_decay: f32,
    epsilon: f32,
    weight_decay: f32,
    // Slot state, per-parameter.
    m: Vec<f32>,
    v: Vec<f32>,
    c: Vec<f32>,
    config: OptimizerConfig,
}

impl Adamom {
    pub fn with_params(cfg: AdamomConfig) -> Self {
        let config = OptimizerConfig::Adamom {
            learning_rate: cfg.learning_rate,
            ada_decay: cfg.ada_decay,
            mom_decay: cfg.mom_decay,
            epsilon: cfg.epsilon,
            weight_decay: cfg.weight_decay,
        };
        Self {
            learning_rate: cfg.learning_rate,
            ada_decay: cfg.ada_decay,
            mom_decay: cfg.mom_decay,
            epsilon: cfg.epsilon,
            weight_decay: cfg.weight_decay,
            m: Vec::new(),
            v: Vec::new(),
            c: Vec::new(),
            config,
        }
    }

    pub fn first_moment(&self) -> &[f32] {
        &self.m
    }

    pub fn second_moment(&self) -> &[f32] {
        &self.v
    }

    pub fn c(&self) -> &[f32] {
        &self.c
    }

    pub fn reset_state(&mut self) {
        self.m.clear();
        self.v.clear();
        self.c.clear();
    }
}

impl Optimizer for Adamom {
    fn new(config: OptimizerConfig) -> Result<Self, OptimizerError> {
        match config {
            OptimizerConfig::Adamom {
                learning_rate,
                ada_decay,
                mom_decay,
                epsilon,
                weight_decay,
            } => Ok(Self {
                learning_rate,
                ada_decay,
                mom_decay,
                epsilon,
                weight_decay,
                m: Vec::new(),
                v: Vec::new(),
                c: Vec::new(),
                config,
            }),
            _ => Err(OptimizerError::ConfigMismatch {
                expected: "Adamom".to_string(),
                got: config.name().to_string(),
            }),
        }
    }

    fn apply_gradients(&mut self, embedding: &mut [f32], gradients: &[f32]) {
        if self.m.len() != embedding.len() {
            self.m = vec![0.0; embedding.len()];
            self.v = vec![0.0; embedding.len()];
            self.c = vec![0.0; embedding.len()];
        }

        // ApplyAdamom updates slots first, then applies update to var using updated slots.
        for i in 0..embedding.len() {
            let var = embedding[i];
            let grad_after_decay = self.weight_decay * var + gradients[i];

            self.m[i] = self.mom_decay * self.m[i] + (1.0 - self.mom_decay) * grad_after_decay;
            self.v[i] = self.ada_decay * self.v[i] + grad_after_decay * grad_after_decay;
            self.c[i] = self.ada_decay * self.c[i] + 1.0;

            // var -= m * lr * rsqrt(v / c + epsilon)
            let denom = (self.v[i] / self.c[i] + self.epsilon).sqrt();
            embedding[i] = var - self.m[i] * self.learning_rate / denom;
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
    fn adamom_matches_python_test_values_single_step() {
        // Mirrors monolith/native_training/optimizers/adamom_test.py::testBasic
        //
        // v starts at 0.1. loss = 0.12*v so grad=0.12.
        // Config: lr=0.1, weight_decay=0.01, ada_decay=0.99, mom_decay=0.9.
        let cfg = AdamomConfig {
            learning_rate: 0.1,
            ada_decay: 0.99,
            mom_decay: 0.9,
            epsilon: 1e-6, // python default
            weight_decay: 0.01,
        };
        let mut opt = Adamom::with_params(cfg);

        let mut v = vec![0.1_f32];
        let grad = vec![0.12_f32];
        opt.apply_gradients(&mut v, &grad);

        let eps = 1e-8_f32;
        assert!(
            (opt.first_moment()[0] - 0.0121).abs() < eps,
            "{:?}",
            opt.first_moment()
        );
        assert!((opt.c()[0] - 1.0).abs() < eps, "{:?}", opt.c());
        assert!(
            (opt.second_moment()[0] - 0.014_641).abs() < eps,
            "{:?}",
            opt.second_moment()
        );
        assert!((v[0] - 0.090000336).abs() < eps, "{v:?}");
    }
}
