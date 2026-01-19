//! Weight initialization utilities matching Python Monolith defaults.

use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum Initializer {
    /// Glorot/Xavier uniform initialization.
    #[default]
    GlorotUniform,
    /// Glorot/Xavier normal initialization.
    GlorotNormal,
    /// He/Kaiming uniform initialization.
    HeUniform,
    /// He/Kaiming normal initialization.
    HeNormal,
    /// Orthogonal initialization.
    Orthogonal,
    /// All zeros.
    Zeros,
    /// All ones.
    Ones,
    /// Constant value.
    Constant(f32),
}

impl Initializer {
    pub fn initialize(&self, shape: &[usize]) -> Tensor {
        match self {
            Initializer::Zeros => Tensor::zeros(shape),
            Initializer::Ones => Tensor::ones(shape),
            Initializer::Constant(value) => Tensor::from_data(shape, vec![*value; shape.iter().product()]),
            Initializer::GlorotUniform => {
                let (fan_in, fan_out) = fan_in_out(shape);
                let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::rand(shape)
                    .scale(2.0 * limit)
                    .sub(&Tensor::from_data(&[1], vec![limit]))
            }
            Initializer::GlorotNormal => {
                let (fan_in, fan_out) = fan_in_out(shape);
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                Tensor::randn(shape, 0.0, std)
            }
            Initializer::HeUniform => {
                let (fan_in, _) = fan_in_out(shape);
                let limit = (6.0 / fan_in as f32).sqrt();
                Tensor::rand(shape)
                    .scale(2.0 * limit)
                    .sub(&Tensor::from_data(&[1], vec![limit]))
            }
            Initializer::HeNormal => {
                let (fan_in, _) = fan_in_out(shape);
                let std = (2.0 / fan_in as f32).sqrt();
                Tensor::randn(shape, 0.0, std)
            }
            Initializer::Orthogonal => orthogonal(shape),
        }
    }
}

fn fan_in_out(shape: &[usize]) -> (usize, usize) {
    if shape.len() >= 2 {
        let fan_in = shape[0];
        let fan_out = shape[1];
        (fan_in.max(1), fan_out.max(1))
    } else if shape.len() == 1 {
        let dim = shape[0].max(1);
        (dim, dim)
    } else {
        (1, 1)
    }
}

fn orthogonal(shape: &[usize]) -> Tensor {
    assert!(shape.len() == 2, "orthogonal initializer expects 2D shape");
    let rows = shape[0];
    let cols = shape[1];

    let (m, n, transpose) = if rows >= cols {
        (rows, cols, false)
    } else {
        (cols, rows, true)
    };

    let mut seed: u64 = 42;
    let data = randn_vec(m * n, 0.0, 1.0, &mut seed);

    // Gram-Schmidt orthogonalization on columns
    let mut q = vec![0.0f32; m * n];
    for j in 0..n {
        // v = a[:, j]
        let mut v = vec![0.0f32; m];
        for i in 0..m {
            v[i] = data[i * n + j];
        }

        for k in 0..j {
            let mut dot = 0.0f32;
            for i in 0..m {
                dot += q[i * n + k] * v[i];
            }
            for i in 0..m {
                v[i] -= dot * q[i * n + k];
            }
        }

        let mut norm = 0.0f32;
        for i in 0..m {
            norm += v[i] * v[i];
        }
        norm = norm.sqrt().max(1e-6);
        for i in 0..m {
            q[i * n + j] = v[i] / norm;
        }
    }

    if transpose {
        // transpose q from (m x n) to (rows x cols)
        let mut out = vec![0.0f32; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[i * cols + j] = q[j * rows + i];
            }
        }
        Tensor::from_data(shape, out)
    } else {
        Tensor::from_data(shape, q)
    }
}

fn randn_vec(n: usize, mean: f32, std: f32, seed: &mut u64) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let u1 = ((*seed >> 16) & 0x7fff) as f32 / 32768.0 + 1e-10;
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let u2 = ((*seed >> 16) & 0x7fff) as f32 / 32768.0;
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        out.push(z * std + mean);
    }
    out
}
