//! Base task definitions mirroring Python `BaseTask`.

use crate::error::{MonolithError, Result};
use crate::hyperparams::{ParamValue, Params};

/// Accelerator options for tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accelerator {
    None,
    Tpu,
    Horovod,
}

impl Accelerator {
    /// Parses an accelerator from string.
    pub fn from_str(value: &str) -> Result<Self> {
        match value {
            "tpu" => Ok(Accelerator::Tpu),
            "horovod" => Ok(Accelerator::Horovod),
            "none" | "" => Ok(Accelerator::None),
            _ => Err(MonolithError::ConfigError {
                message: format!("Unknown accelerator {}", value),
            }),
        }
    }

    /// Returns the string name of the accelerator.
    pub fn as_str(&self) -> &'static str {
        match self {
            Accelerator::None => "none",
            Accelerator::Tpu => "tpu",
            Accelerator::Horovod => "horovod",
        }
    }
}

/// Task mode for building inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskMode {
    Train,
    Eval,
    Predict,
}

/// Builds the default BaseTask params tree.
pub fn base_task_params() -> Params {
    let mut p = Params::new();
    let _ = p.define(
        "accelerator",
        ParamValue::None,
        "Accelerator to use. One of [None, \"tpu\", \"horovod\"].",
    );

    let mut input = Params::new();
    let _ = input.define("eval_examples", ParamValue::None, "Eval examples.");
    let _ = input.define("train_examples", ParamValue::None, "Train examples.");
    let _ = p.define("input", ParamValue::from(input), "Input params.");

    let mut eval = Params::new();
    let _ = eval.define(
        "per_replica_batch_size",
        ParamValue::None,
        "Per replica batch size",
    );
    let _ = eval.define("steps_per_eval", 10000_i64, "Steps between evals");
    let _ = eval.define("steps", ParamValue::None, "Eval steps");
    let _ = p.define("eval", ParamValue::from(eval), "Eval params");

    let mut train = Params::new();
    let _ = train.define("steps", ParamValue::None, "Train steps");
    let _ = train.define("max_steps", ParamValue::None, "Max train steps");
    let _ = train.define(
        "per_replica_batch_size",
        ParamValue::None,
        "Per replica batch size",
    );
    let _ = train.define("file_pattern", ParamValue::None, "Training input data.");
    let _ = train.define("repeat", false, "Repeat input");
    let _ = train.define("label_key", "label", "Label field key");
    let _ = train.define(
        "save_checkpoints_steps",
        ParamValue::None,
        "Save checkpoints every N steps",
    );
    let _ = train.define(
        "save_checkpoints_secs",
        ParamValue::None,
        "Save checkpoints every N seconds",
    );
    let _ = train.define(
        "dense_only_save_checkpoints_secs",
        ParamValue::None,
        "Save dense-only checkpoints every N seconds",
    );
    let _ = train.define(
        "dense_only_save_checkpoints_steps",
        ParamValue::None,
        "Save dense-only checkpoints every N steps",
    );
    let _ = p.define("train", ParamValue::from(train), "Train params");

    p
}

/// BaseTask trait: concrete tasks supply input/model creation.
pub trait BaseTask {
    /// Returns task params.
    fn params(&self) -> &Params;

    /// Returns mutable task params.
    fn params_mut(&mut self) -> &mut Params;

    /// Creates input given mode.
    fn create_input(&self, mode: TaskMode) -> Result<()>;

    /// Creates model function or model state.
    fn create_model(&self) -> Result<()>;
}
