//! Estimator pattern for training orchestration.
//!
//! This module provides the `Estimator` abstraction, inspired by TensorFlow's
//! Estimator API, which orchestrates training, evaluation, and prediction.

use crate::hooks::{Hook, HookAction, HookList};
use crate::metrics::{Metrics, MetricsRecorder};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Errors that can occur during estimator operations.
#[derive(Debug, Error)]
pub enum EstimatorError {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A hook error occurred.
    #[error("Hook error: {0}")]
    Hook(#[from] crate::hooks::HookError),

    /// Training was stopped by a hook.
    #[error("Training stopped by hook at step {0}")]
    StoppedByHook(u64),

    /// The model directory is invalid.
    #[error("Invalid model directory: {0}")]
    InvalidModelDir(String),

    /// A training error occurred.
    #[error("Training error: {0}")]
    Training(String),

    /// An evaluation error occurred.
    #[error("Evaluation error: {0}")]
    Evaluation(String),

    /// A prediction error occurred.
    #[error("Prediction error: {0}")]
    Prediction(String),

    /// EstimatorSpec update attempted to change mode.
    #[error("Cannot change EstimatorSpec mode from {current:?} to {requested:?}")]
    SpecModeChange {
        current: EstimatorMode,
        requested: EstimatorMode,
    },
}

/// Result type for estimator operations.
pub type EstimatorResult<T> = Result<T, EstimatorError>;

/// Configuration for the Estimator.
///
/// # Examples
///
/// ```
/// use monolith_training::estimator::EstimatorConfig;
/// use std::path::PathBuf;
///
/// let config = EstimatorConfig::new(PathBuf::from("/tmp/model"))
///     .with_train_steps(10000)
///     .with_eval_steps(100)
///     .with_save_checkpoints_steps(1000);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EstimatorConfig {
    /// Directory to save model checkpoints and exports.
    pub model_dir: PathBuf,
    /// Number of training steps to run.
    pub train_steps: Option<u64>,
    /// Number of evaluation steps to run.
    pub eval_steps: Option<u64>,
    /// Save checkpoints every N steps.
    pub save_checkpoints_steps: Option<u64>,
    /// Log metrics every N steps.
    pub log_step_count_steps: u64,
    /// Warm start from a previous checkpoint.
    pub warm_start_from: Option<PathBuf>,
    /// Random seed for reproducibility.
    pub random_seed: Option<u64>,
}

impl EstimatorConfig {
    /// Creates a new EstimatorConfig with the given model directory.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - The directory to save model checkpoints.
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            model_dir,
            train_steps: None,
            eval_steps: None,
            save_checkpoints_steps: None,
            log_step_count_steps: 100,
            warm_start_from: None,
            random_seed: None,
        }
    }

    /// Sets the number of training steps.
    pub fn with_train_steps(mut self, steps: u64) -> Self {
        self.train_steps = Some(steps);
        self
    }

    /// Sets the number of evaluation steps.
    pub fn with_eval_steps(mut self, steps: u64) -> Self {
        self.eval_steps = Some(steps);
        self
    }

    /// Sets the checkpoint saving interval.
    pub fn with_save_checkpoints_steps(mut self, steps: u64) -> Self {
        self.save_checkpoints_steps = Some(steps);
        self
    }

    /// Sets the logging interval.
    pub fn with_log_step_count_steps(mut self, steps: u64) -> Self {
        self.log_step_count_steps = steps;
        self
    }

    /// Sets the warm start path.
    pub fn with_warm_start_from(mut self, path: PathBuf) -> Self {
        self.warm_start_from = Some(path);
        self
    }

    /// Sets the random seed.
    pub fn with_random_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }
}

/// Mode of operation for the estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorMode {
    /// Training mode.
    Train,
    /// Evaluation mode.
    Eval,
    /// Prediction mode.
    Predict,
}

/// Python-style estimator output spec.
///
/// Mirrors `EstimatorSpec` defaults from Python:
/// - `head_name=None`
/// - `loss=None`
/// - `optimizer=None`
/// - `classification=True`
#[derive(Debug, Clone, PartialEq)]
pub struct EstimatorSpec {
    pub mode: EstimatorMode,
    pub label: Option<String>,
    pub pred: Vec<f32>,
    pub head_name: Option<String>,
    pub loss: Option<f64>,
    pub optimizer: Option<String>,
    pub classification: bool,
}

impl EstimatorSpec {
    pub fn new(mode: EstimatorMode, label: Option<String>, pred: Vec<f32>) -> Self {
        Self {
            mode,
            label,
            pred,
            head_name: None,
            loss: None,
            optimizer: None,
            classification: true,
        }
    }

    /// Parity helper mirroring Python `namedtuple._replace(...)`.
    ///
    /// Mode can only be supplied when it equals the current mode.
    pub fn replace(&self, update: EstimatorSpecUpdate) -> EstimatorResult<Self> {
        if let Some(mode) = update.mode {
            if mode != self.mode {
                return Err(EstimatorError::SpecModeChange {
                    current: self.mode,
                    requested: mode,
                });
            }
        }
        Ok(Self {
            mode: self.mode,
            label: update.label.unwrap_or_else(|| self.label.clone()),
            pred: update.pred.unwrap_or_else(|| self.pred.clone()),
            head_name: update.head_name.unwrap_or_else(|| self.head_name.clone()),
            loss: update.loss.unwrap_or(self.loss),
            optimizer: update.optimizer.unwrap_or_else(|| self.optimizer.clone()),
            classification: update.classification.unwrap_or(self.classification),
        })
    }
}

/// Update payload for [`EstimatorSpec::replace`].
#[derive(Debug, Clone, Default)]
pub struct EstimatorSpecUpdate {
    pub mode: Option<EstimatorMode>,
    pub label: Option<Option<String>>,
    pub pred: Option<Vec<f32>>,
    pub head_name: Option<Option<String>>,
    pub loss: Option<Option<f64>>,
    pub optimizer: Option<Option<String>>,
    pub classification: Option<bool>,
}

/// Result of a training run.
#[derive(Debug, Clone)]
pub struct TrainResult {
    /// The final global step.
    pub global_step: u64,
    /// The final metrics.
    pub final_metrics: Option<Metrics>,
    /// Whether training was stopped early by a hook.
    pub stopped_early: bool,
}

/// Result of an evaluation run.
#[derive(Debug, Clone)]
pub struct EvalResult {
    /// The global step at which evaluation was performed.
    pub global_step: u64,
    /// The evaluation metrics.
    pub metrics: Metrics,
    /// Number of evaluation steps.
    pub eval_steps: u64,
}

/// Result of a prediction run.
#[derive(Debug, Clone)]
pub struct PredictResult {
    /// The predictions (as raw f32 values).
    pub predictions: Vec<Vec<f32>>,
    /// Number of examples predicted.
    pub num_examples: usize,
}

/// Trait for model functions that define forward pass and loss computation.
///
/// This is a simplified version that would be extended with actual tensor types
/// in a real implementation.
pub trait ModelFn: Send + Sync {
    /// Performs a forward pass and computes loss.
    ///
    /// # Arguments
    ///
    /// * `mode` - The estimator mode (train, eval, or predict).
    ///
    /// # Returns
    ///
    /// The loss value for training/eval, or predictions for predict mode.
    fn call(&self, mode: EstimatorMode) -> EstimatorResult<f64>;

    /// Produces predictions for a single input example.
    ///
    /// Default implementation keeps backward compatibility with legacy `call(Predict)`
    /// behavior by returning a single scalar prediction.
    fn predict(&self, _features: &[f32]) -> EstimatorResult<Vec<f32>> {
        Ok(vec![self.call(EstimatorMode::Predict)? as f32])
    }
}

/// A simple model function that returns a constant loss (for testing).
pub struct ConstantModelFn {
    loss: f64,
}

impl ConstantModelFn {
    /// Creates a new constant model function.
    pub fn new(loss: f64) -> Self {
        Self { loss }
    }
}

impl ModelFn for ConstantModelFn {
    fn call(&self, _mode: EstimatorMode) -> EstimatorResult<f64> {
        Ok(self.loss)
    }
}

/// The main Estimator struct for orchestrating training.
///
/// The Estimator pattern provides a high-level API for training, evaluation,
/// and prediction, handling common concerns like checkpointing, logging,
/// and distributed training.
///
/// # Examples
///
/// ```
/// use monolith_training::estimator::{Estimator, EstimatorConfig, ConstantModelFn};
/// use std::path::PathBuf;
///
/// let config = EstimatorConfig::new(PathBuf::from("/tmp/model"))
///     .with_train_steps(100);
///
/// let model_fn = ConstantModelFn::new(0.5);
/// let mut estimator = Estimator::new(config, model_fn);
///
/// // Training would be called like this:
/// // let result = estimator.train().unwrap();
/// ```
pub struct Estimator<M: ModelFn> {
    /// Estimator configuration.
    config: EstimatorConfig,
    /// The model function.
    model_fn: M,
    /// Training hooks.
    hooks: HookList,
    /// Current global step.
    global_step: u64,
    /// Metrics recorder.
    metrics_recorder: MetricsRecorder,
}

impl<M: ModelFn> Estimator<M> {
    /// Creates a new Estimator with the given configuration and model function.
    ///
    /// # Arguments
    ///
    /// * `config` - The estimator configuration.
    /// * `model_fn` - The model function that defines the model.
    pub fn new(config: EstimatorConfig, model_fn: M) -> Self {
        Self {
            config,
            model_fn,
            hooks: HookList::new(),
            global_step: 0,
            metrics_recorder: MetricsRecorder::new(),
        }
    }

    /// Creates an estimator from execution-time runner configuration.
    pub fn from_runner_config(runner_conf: &crate::run_config::RunnerConfig, model_fn: M) -> Self {
        Self::new(runner_conf.to_estimator_config(), model_fn)
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &EstimatorConfig {
        &self.config
    }

    /// Returns the current global step.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }

    /// Adds a training hook.
    ///
    /// # Arguments
    ///
    /// * `hook` - The hook to add.
    pub fn add_hook<H: Hook + 'static>(&mut self, hook: H) {
        self.hooks.add(hook);
    }

    fn resolve_train_target(&self, steps: Option<u64>, max_steps: Option<u64>) -> u64 {
        let from_steps = steps.map(|s| self.global_step.saturating_add(s));
        match (from_steps, max_steps) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => self.config.train_steps.unwrap_or(u64::MAX),
        }
    }

    /// Runs training.
    ///
    /// # Returns
    ///
    /// A `TrainResult` containing the final step and metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if training fails or a hook returns an error.
    pub fn train(&mut self) -> EstimatorResult<TrainResult> {
        self.train_with_limits(None, None)
    }

    /// Runs training with Python-style `steps`/`max_steps` semantics.
    ///
    /// - `steps`: relative number of additional steps from current global step.
    /// - `max_steps`: absolute global-step cap.
    /// - if both are provided, training stops at `min(global_step + steps, max_steps)`.
    pub fn train_with_limits(
        &mut self,
        steps: Option<u64>,
        max_steps: Option<u64>,
    ) -> EstimatorResult<TrainResult> {
        let max_steps = self.resolve_train_target(steps, max_steps);
        let mut stopped_early = false;
        let mut last_metrics: Option<Metrics> = None;

        tracing::info!(
            "Starting training for {} steps",
            if max_steps == u64::MAX {
                "unlimited".to_string()
            } else {
                max_steps.to_string()
            }
        );

        while self.global_step < max_steps {
            // Before step hooks
            self.hooks.before_step(self.global_step)?;

            // Run one training step
            let loss = self.model_fn.call(EstimatorMode::Train)?;
            let metrics = Metrics::new(loss, self.global_step);

            // Record metrics
            self.metrics_recorder.record(&metrics);
            last_metrics = Some(metrics.clone());

            // After step hooks
            match self.hooks.after_step(self.global_step, &metrics)? {
                HookAction::Continue => {}
                HookAction::Stop => {
                    tracing::info!("Training stopped by hook at step {}", self.global_step);
                    stopped_early = true;
                    break;
                }
            }

            self.global_step += 1;
        }

        // End hooks
        self.hooks.end(self.global_step, last_metrics.as_ref())?;

        Ok(TrainResult {
            global_step: self.global_step,
            final_metrics: last_metrics,
            stopped_early,
        })
    }

    /// Runs evaluation.
    ///
    /// # Returns
    ///
    /// An `EvalResult` containing the evaluation metrics.
    ///
    /// # Errors
    ///
    /// Returns an error if evaluation fails.
    pub fn evaluate(&mut self) -> EstimatorResult<EvalResult> {
        self.evaluate_with_steps(None)
    }

    /// Runs evaluation with an optional step override.
    pub fn evaluate_with_steps(&mut self, steps: Option<u64>) -> EstimatorResult<EvalResult> {
        let eval_steps = steps.or(self.config.eval_steps).unwrap_or(1);
        let mut recorder = MetricsRecorder::new();

        tracing::info!("Starting evaluation for {} steps", eval_steps);

        for step in 0..eval_steps {
            let loss = self.model_fn.call(EstimatorMode::Eval)?;
            let metrics = Metrics::new(loss, step);
            recorder.record(&metrics);
        }

        let metrics = recorder.aggregate(self.global_step);

        tracing::info!("Evaluation complete: loss = {:.6}", metrics.loss);

        Ok(EvalResult {
            global_step: self.global_step,
            metrics,
            eval_steps,
        })
    }

    /// Runs prediction for a fixed number of examples.
    ///
    /// This method preserves compatibility with existing callsites by producing
    /// predictions from empty feature vectors.
    ///
    /// # Arguments
    ///
    /// * `num_examples` - The number of examples to predict.
    ///
    /// # Returns
    ///
    /// A `PredictResult` containing the predictions.
    ///
    /// # Errors
    ///
    /// Returns an error if prediction fails.
    pub fn predict(&mut self, num_examples: usize) -> EstimatorResult<PredictResult> {
        let empty_inputs = vec![Vec::<f32>::new(); num_examples];
        self.predict_with_inputs(&empty_inputs)
    }

    /// Runs prediction using explicit input features.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Per-example feature vectors.
    ///
    /// # Returns
    ///
    /// A `PredictResult` containing one output vector per input example.
    pub fn predict_with_inputs(&mut self, inputs: &[Vec<f32>]) -> EstimatorResult<PredictResult> {
        tracing::info!("Running prediction for {} examples", inputs.len());
        let mut predictions = Vec::with_capacity(inputs.len());

        for features in inputs {
            predictions.push(self.model_fn.predict(features)?);
        }

        Ok(PredictResult {
            predictions,
            num_examples: inputs.len(),
        })
    }

    /// Trains and evaluates in an interleaved fashion.
    ///
    /// # Arguments
    ///
    /// * `train_steps_per_eval` - Number of training steps between evaluations.
    /// * `num_evals` - Maximum number of evaluations to perform.
    ///
    /// # Returns
    ///
    /// A tuple of the final training result and the last evaluation result.
    pub fn train_and_evaluate(
        &mut self,
        train_steps_per_eval: u64,
        num_evals: u64,
    ) -> EstimatorResult<(TrainResult, EvalResult)> {
        let mut last_eval: Option<EvalResult> = None;
        let mut last_train: Option<TrainResult> = None;

        for eval_num in 0..num_evals {
            // Train one relative slice per round.
            let train_result = self.train_with_limits(Some(train_steps_per_eval), None)?;
            last_train = Some(train_result.clone());

            // Evaluate
            let eval_result = self.evaluate()?;
            tracing::info!(
                "Eval {}: loss = {:.6}",
                eval_num + 1,
                eval_result.metrics.loss
            );

            last_eval = Some(eval_result);

            if train_result.stopped_early {
                break;
            }
        }

        let final_train = last_train.unwrap_or(TrainResult {
            global_step: self.global_step,
            final_metrics: self.metrics_recorder.aggregate(self.global_step).into(),
            stopped_early: false,
        });

        let final_eval = last_eval
            .ok_or_else(|| EstimatorError::Evaluation("No evaluation was performed".to_string()))?;

        Ok((final_train, final_eval))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hooks::{EarlyStoppingHook, LoggingHook};

    #[test]
    fn test_estimator_config() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"))
            .with_train_steps(1000)
            .with_eval_steps(100)
            .with_save_checkpoints_steps(500)
            .with_random_seed(42);

        assert_eq!(config.train_steps, Some(1000));
        assert_eq!(config.eval_steps, Some(100));
        assert_eq!(config.save_checkpoints_steps, Some(500));
        assert_eq!(config.random_seed, Some(42));
    }

    #[test]
    fn test_estimator_train() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model")).with_train_steps(10);

        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        let result = estimator.train().unwrap();
        assert_eq!(result.global_step, 10);
        assert!(!result.stopped_early);
        assert!(result.final_metrics.is_some());
    }

    #[test]
    fn test_estimator_from_runner_config() {
        let runner = crate::run_config::RunnerConfig {
            model_dir: PathBuf::from("/tmp/model_from_runner"),
            log_step_count_steps: 7,
            restore_ckpt: Some("model.ckpt-88".to_string()),
            ..crate::run_config::RunnerConfig::default()
        };
        let model_fn = ConstantModelFn::new(0.5);
        let estimator = Estimator::from_runner_config(&runner, model_fn);
        assert_eq!(estimator.config().model_dir, PathBuf::from("/tmp/model_from_runner"));
        assert_eq!(estimator.config().log_step_count_steps, 7);
        assert_eq!(
            estimator.config().warm_start_from,
            Some(PathBuf::from("model.ckpt-88"))
        );
    }

    #[test]
    fn test_estimator_with_hooks() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model")).with_train_steps(100);

        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        estimator.add_hook(LoggingHook::new(10));

        let result = estimator.train().unwrap();
        assert_eq!(result.global_step, 100);
    }

    #[test]
    fn test_estimator_early_stopping() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model")).with_train_steps(1000);

        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        // Will stop after 5 steps with no improvement
        estimator.add_hook(EarlyStoppingHook::new("loss", 5, 0.001));

        let result = estimator.train().unwrap();
        assert!(result.stopped_early);
        assert!(result.global_step < 1000);
    }

    #[test]
    fn test_estimator_evaluate() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model")).with_eval_steps(10);

        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        let result = estimator.evaluate().unwrap();
        assert_eq!(result.eval_steps, 10);
        assert!((result.metrics.loss - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_train_with_limits_relative_steps() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"));
        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        let r1 = estimator.train_with_limits(Some(3), None).unwrap();
        assert_eq!(r1.global_step, 3);
        let r2 = estimator.train_with_limits(Some(2), None).unwrap();
        assert_eq!(r2.global_step, 5);
    }

    #[test]
    fn test_estimator_train_with_limits_max_steps_absolute() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"));
        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        estimator.train_with_limits(Some(5), None).unwrap();
        let r = estimator.train_with_limits(None, Some(7)).unwrap();
        assert_eq!(r.global_step, 7);
    }

    #[test]
    fn test_estimator_train_with_limits_steps_and_max_steps() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"));
        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        estimator.train_with_limits(Some(5), None).unwrap();
        let r = estimator.train_with_limits(Some(10), Some(12)).unwrap();
        assert_eq!(r.global_step, 12);
    }

    #[test]
    fn test_estimator_evaluate_with_steps_override() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model")).with_eval_steps(10);
        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        let result = estimator.evaluate_with_steps(Some(3)).unwrap();
        assert_eq!(result.eval_steps, 3);
        assert!((result.metrics.loss - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_estimator_predict() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"));
        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        let result = estimator.predict(5).unwrap();
        assert_eq!(result.num_examples, 5);
        assert_eq!(result.predictions.len(), 5);
    }

    #[test]
    fn test_estimator_spec_defaults() {
        let spec = EstimatorSpec::new(EstimatorMode::Predict, None, vec![1.0, 2.0]);
        assert_eq!(spec.head_name, None);
        assert_eq!(spec.loss, None);
        assert_eq!(spec.optimizer, None);
        assert!(spec.classification);
    }

    #[test]
    fn test_estimator_spec_replace_same_mode_allowed() {
        let spec = EstimatorSpec::new(EstimatorMode::Train, Some("l".to_string()), vec![1.0]);
        let next = spec
            .replace(EstimatorSpecUpdate {
                mode: Some(EstimatorMode::Train),
                loss: Some(Some(0.5)),
                classification: Some(false),
                ..EstimatorSpecUpdate::default()
            })
            .unwrap();
        assert_eq!(next.loss, Some(0.5));
        assert!(!next.classification);
    }

    #[test]
    fn test_estimator_spec_replace_mode_change_rejected() {
        let spec = EstimatorSpec::new(EstimatorMode::Train, None, vec![]);
        let err = spec
            .replace(EstimatorSpecUpdate {
                mode: Some(EstimatorMode::Eval),
                ..EstimatorSpecUpdate::default()
            })
            .unwrap_err();
        assert!(matches!(
            err,
            EstimatorError::SpecModeChange {
                current: EstimatorMode::Train,
                requested: EstimatorMode::Eval
            }
        ));
    }

    struct InputAwareModelFn;

    impl ModelFn for InputAwareModelFn {
        fn call(&self, mode: EstimatorMode) -> EstimatorResult<f64> {
            match mode {
                EstimatorMode::Train | EstimatorMode::Eval => Ok(0.1),
                EstimatorMode::Predict => Ok(0.0),
            }
        }

        fn predict(&self, features: &[f32]) -> EstimatorResult<Vec<f32>> {
            let sum = features.iter().copied().sum::<f32>();
            Ok(vec![sum, features.len() as f32])
        }
    }

    #[test]
    fn test_estimator_predict_with_inputs() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"));
        let model_fn = InputAwareModelFn;
        let mut estimator = Estimator::new(config, model_fn);

        let inputs = vec![vec![1.0, 2.0, 3.0], vec![4.0]];
        let result = estimator.predict_with_inputs(&inputs).unwrap();
        assert_eq!(result.num_examples, 2);
        assert_eq!(result.predictions, vec![vec![6.0, 3.0], vec![4.0, 1.0]]);
    }
}
