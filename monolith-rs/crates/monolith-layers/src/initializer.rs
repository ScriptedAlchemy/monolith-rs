//! Weight initialization utilities matching Python Monolith defaults.

use std::time::{SystemTime, UNIX_EPOCH};

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
    /// VarianceScaling initializer matching Python `monolith/core/variance_scaling.py`.
    ///
    /// Note: This is distinct from Glorot/He helpers above; it supports mode/distribution
    /// and (optionally) a fixed seed.
    VarianceScaling {
        scale: f32,
        mode: VarianceScalingMode,
        distribution: VarianceScalingDistribution,
        seed: Option<u64>,
    },
}

impl Initializer {
    pub fn initialize(&self, shape: &[usize]) -> Tensor {
        match self {
            Initializer::Zeros => Tensor::zeros(shape),
            Initializer::Ones => Tensor::ones(shape),
            Initializer::Constant(value) => {
                Tensor::from_data(shape, vec![*value; shape.iter().product()])
            }
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
            Initializer::VarianceScaling {
                scale,
                mode,
                distribution,
                seed,
            } => variance_scaling_initialize(*scale, *mode, *distribution, *seed, shape),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarianceScalingMode {
    FanIn,
    FanOut,
    FanAvg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VarianceScalingDistribution {
    TruncatedNormal,
    UntruncatedNormal,
    Uniform,
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

fn compute_fans(shape: &[usize]) -> (f32, f32) {
    match shape.len() {
        2 => (shape[0] as f32, shape[1] as f32),
        3 | 4 | 5 => {
            // Keras channels_last receptive field: prod(shape[:-2])
            let receptive: usize = shape[..shape.len() - 2].iter().product();
            let fan_in = shape[shape.len() - 2] as f32 * receptive as f32;
            let fan_out = shape[shape.len() - 1] as f32 * receptive as f32;
            (fan_in, fan_out)
        }
        0 | 1 => {
            let p: f32 = shape.iter().product::<usize>() as f32;
            let v = p.sqrt();
            (v, v)
        }
        _ => {
            let p: f32 = shape.iter().product::<usize>() as f32;
            let v = p.sqrt();
            (v, v)
        }
    }
}

fn variance_scaling_initialize(
    scale: f32,
    mode: VarianceScalingMode,
    distribution: VarianceScalingDistribution,
    seed: Option<u64>,
    shape: &[usize],
) -> Tensor {
    assert!(scale > 0.0, "VarianceScaling scale must be positive");

    let (fan_in, fan_out) = compute_fans(shape);
    let mut adj = scale;
    match mode {
        VarianceScalingMode::FanIn => adj /= fan_in.max(1.0),
        VarianceScalingMode::FanOut => adj /= fan_out.max(1.0),
        VarianceScalingMode::FanAvg => adj /= ((fan_in + fan_out) / 2.0).max(1.0),
    }

    // Python (`monolith/core/variance_scaling.py`) calls `np.random.seed(self.seed)` on each
    // invocation. When `seed` is None, numpy reseeds from OS entropy/time; when set, it is
    // deterministic. Mirror that behavior here using a local RNG.
    let actual_seed = seed.unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64
    });
    let mut rng = Lcg::new(actual_seed);
    match distribution {
        VarianceScalingDistribution::Uniform => {
            let limit = (3.0 * adj).sqrt();
            let n: usize = shape.iter().product();
            let mut data = Vec::with_capacity(n);
            for _ in 0..n {
                let u = rng.next_f32(); // [0,1)
                data.push((2.0 * u - 1.0) * limit);
            }
            Tensor::from_data(shape, data)
        }
        VarianceScalingDistribution::UntruncatedNormal => {
            let std = adj.sqrt();
            tensor_randn_seeded(shape, 0.0, std, &mut rng)
        }
        VarianceScalingDistribution::TruncatedNormal => {
            // 0.879... matches scipy.stats.truncnorm.std(a=-2,b=2,loc=0,scale=1).
            let std = adj.sqrt() / 0.879_625_661_034_239_78_f32;
            tensor_randn_truncated_seeded(shape, 0.0, std, -2.0, 2.0, &mut rng)
        }
    }
}

#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.state >> 16) & 0x7fff) as u32
    }

    fn next_f32(&mut self) -> f32 {
        // [0, 1)
        self.next_u32() as f32 / 32768.0
    }

    fn next_f32_nonzero(&mut self) -> f32 {
        self.next_f32().max(1e-10)
    }

    fn next_standard_normal(&mut self) -> f32 {
        // Box-Muller transform, deterministic from the LCG.
        let u1 = self.next_f32_nonzero();
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

fn tensor_randn_seeded(shape: &[usize], mean: f32, std: f32, rng: &mut Lcg) -> Tensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let z = rng.next_standard_normal();
        data.push(z * std + mean);
    }
    Tensor::from_data(shape, data)
}

fn tensor_randn_truncated_seeded(
    shape: &[usize],
    mean: f32,
    std: f32,
    a: f32,
    b: f32,
    rng: &mut Lcg,
) -> Tensor {
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        // Rejection sample from N(0,1) truncated to [a,b].
        // a/b are in standard deviations (as in Python's truncnorm after standardization).
        let mut z = rng.next_standard_normal();
        let mut tries = 0;
        while (z < a || z > b) && tries < 64 {
            z = rng.next_standard_normal();
            tries += 1;
        }
        // If extremely unlucky, clamp.
        z = z.clamp(a, b);
        data.push(z * std + mean);
    }
    Tensor::from_data(shape, data)
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
