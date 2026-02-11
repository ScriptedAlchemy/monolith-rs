//! TF-free learning rate schedules mirroring `monolith/native_training/learning_rate_functions.py`.
//!
//! The Python code uses TF's global step as an implicit input. In Rust we take
//! the global step explicitly when computing the current learning rate.

use std::fmt;

/// A learning rate schedule function.
pub trait LearningRateFunction: Send + Sync + fmt::Display {
    /// Returns the learning rate at the given `global_step`.
    fn value(&self, global_step: u64) -> f32;
}

fn py_bool(v: bool) -> &'static str {
    if v {
        "True"
    } else {
        "False"
    }
}

fn py_float(v: f32) -> String {
    // Match Python's `str(float)` for the values we use in parity tests:
    // integers keep a trailing ".0".
    if v.is_finite() && v.fract() == 0.0 {
        format!("{:.1}", v)
    } else {
        // Rust's default float formatting is close enough for non-integers
        // (e.g. 0.01, 0.11, 1e-6).
        format!("{v}")
    }
}

/// Polynomial decay learning rate schedule.
///
/// Mirrors TF v1 `tf.compat.v1.train.polynomial_decay`.
#[derive(Debug, Clone, PartialEq)]
pub struct PolynomialDecay {
    pub initial_learning_rate: f32,
    pub decay_steps: u64,
    pub end_learning_rate: f32,
    pub power: f32,
    pub cycle: bool,
    pub name: Option<String>,
}

impl PolynomialDecay {
    /// Creates a new polynomial decay schedule.
    pub fn new(initial_learning_rate: f32, decay_steps: u64) -> Self {
        Self {
            initial_learning_rate,
            decay_steps,
            end_learning_rate: 0.0001,
            power: 1.0,
            cycle: false,
            name: None,
        }
    }

    pub fn with_end_learning_rate(mut self, end_learning_rate: f32) -> Self {
        self.end_learning_rate = end_learning_rate;
        self
    }

    pub fn with_power(mut self, power: f32) -> Self {
        self.power = power;
        self
    }

    pub fn with_cycle(mut self, cycle: bool) -> Self {
        self.cycle = cycle;
        self
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

impl LearningRateFunction for PolynomialDecay {
    fn value(&self, global_step: u64) -> f32 {
        let decay_steps = self.decay_steps.max(1);
        let gs = global_step;

        // Mirrors TF:
        // if cycle:
        //   decay_steps = decay_steps * ceil(gs / decay_steps)
        // else:
        //   gs = min(gs, decay_steps)
        let (gs, decay_steps) = if self.cycle {
            let div = (gs + decay_steps - 1) / decay_steps; // ceil(gs/decay_steps)
            let div = div.max(1);
            (gs, decay_steps.saturating_mul(div))
        } else {
            (gs.min(decay_steps), decay_steps)
        };

        let step = gs as f32;
        let ds = decay_steps as f32;
        let frac = (1.0 - step / ds).clamp(0.0, 1.0);
        (self.initial_learning_rate - self.end_learning_rate) * frac.powf(self.power)
            + self.end_learning_rate
    }
}

impl fmt::Display for PolynomialDecay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Match Python's `LearningRateFunction.__str__` which sorts by `__dict__` keys.
        let name = self.name.as_deref().unwrap_or("None");
        write!(
            f,
            "LearningRateFunction(\"PolynomialDecay\",Params:cycle={},decay_steps={},end_learning_rate={},initial_learning_rate={},name={},power={})",
            py_bool(self.cycle),
            self.decay_steps,
            py_float(self.end_learning_rate),
            py_float(self.initial_learning_rate),
            name,
            py_float(self.power),
        )
    }
}
