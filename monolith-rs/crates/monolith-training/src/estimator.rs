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
        let max_steps = self.config.train_steps.unwrap_or(u64::MAX);
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
        let eval_steps = self.config.eval_steps.unwrap_or(1);
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
        let original_train_steps = self.config.train_steps;

        for eval_num in 0..num_evals {
            // Set train steps for this round
            let target_step = (eval_num + 1) * train_steps_per_eval;
            self.config.train_steps = Some(target_step);

            // Train
            let train_result = self.train()?;

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

        // Restore original config
        self.config.train_steps = original_train_steps;

        let final_train = TrainResult {
            global_step: self.global_step,
            final_metrics: self.metrics_recorder.aggregate(self.global_step).into(),
            stopped_early: false,
        };

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
    fn test_estimator_predict() {
        let config = EstimatorConfig::new(PathBuf::from("/tmp/model"));
        let model_fn = ConstantModelFn::new(0.5);
        let mut estimator = Estimator::new(config, model_fn);

        let result = estimator.predict(5).unwrap();
        assert_eq!(result.num_examples, 5);
        assert_eq!(result.predictions.len(), 5);
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
