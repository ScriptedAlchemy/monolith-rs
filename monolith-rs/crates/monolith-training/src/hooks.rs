//! Training hooks for customizing the training loop.
//!
//! Hooks allow injecting custom behavior at various points during training,
//! such as logging, checkpointing, and early stopping.

use crate::metrics::Metrics;
use serde_json::json;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;
use tracing::{debug, info, warn};

/// Errors that can occur during hook execution.
#[derive(Debug, Error)]
pub enum HookError {
    /// An I/O error occurred.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A checkpoint error occurred.
    #[error("Checkpoint error: {0}")]
    Checkpoint(String),

    /// A custom hook error.
    #[error("Hook error: {0}")]
    Custom(String),
}

/// Result type for hook operations.
pub type HookResult<T> = Result<T, HookError>;

/// Action to take after a hook runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HookAction {
    /// Continue training normally.
    Continue,
    /// Stop training early.
    Stop,
}

/// Trait for training hooks.
///
/// Hooks are called at various points during the training loop to allow
/// custom behavior such as logging, checkpointing, and early stopping.
///
/// # Examples
///
/// ```
/// use monolith_training::hooks::{Hook, HookAction, HookResult};
/// use monolith_training::metrics::Metrics;
///
/// struct MyHook;
///
/// impl Hook for MyHook {
///     fn name(&self) -> &str {
///         "my_hook"
///     }
///
///     fn after_step(&mut self, step: u64, metrics: &Metrics) -> HookResult<HookAction> {
///         println!("Step {}: loss = {}", step, metrics.loss);
///         Ok(HookAction::Continue)
///     }
/// }
/// ```
pub trait Hook: Send + Sync {
    /// Returns the name of this hook for logging purposes.
    fn name(&self) -> &str;

    /// Called before each training step.
    ///
    /// # Arguments
    ///
    /// * `step` - The current global step (0-indexed).
    fn before_step(&mut self, _step: u64) -> HookResult<()> {
        Ok(())
    }

    /// Called after each training step with the step's metrics.
    ///
    /// # Arguments
    ///
    /// * `step` - The current global step.
    /// * `metrics` - The metrics from the training step.
    ///
    /// # Returns
    ///
    /// A `HookAction` indicating whether to continue or stop training.
    fn after_step(&mut self, _step: u64, _metrics: &Metrics) -> HookResult<HookAction> {
        Ok(HookAction::Continue)
    }

    /// Called at the end of training.
    ///
    /// # Arguments
    ///
    /// * `step` - The final global step.
    /// * `metrics` - The final metrics (if available).
    fn end(&mut self, _step: u64, _metrics: Option<&Metrics>) -> HookResult<()> {
        Ok(())
    }
}

/// A hook that logs training metrics at regular intervals.
///
/// # Examples
///
/// ```
/// use monolith_training::hooks::LoggingHook;
///
/// // Log every 100 steps
/// let hook = LoggingHook::new(100);
/// ```
#[derive(Debug)]
pub struct LoggingHook {
    /// Log every N steps.
    every_n_steps: u64,
    /// Whether to log on the first step.
    log_first_step: bool,
}

impl LoggingHook {
    /// Creates a new logging hook that logs every N steps.
    ///
    /// # Arguments
    ///
    /// * `every_n_steps` - The interval at which to log metrics.
    pub fn new(every_n_steps: u64) -> Self {
        Self {
            every_n_steps: every_n_steps.max(1),
            log_first_step: true,
        }
    }

    /// Sets whether to log on the first step.
    pub fn with_log_first_step(mut self, log_first: bool) -> Self {
        self.log_first_step = log_first;
        self
    }
}

impl Hook for LoggingHook {
    fn name(&self) -> &str {
        "logging_hook"
    }

    fn after_step(&mut self, step: u64, metrics: &Metrics) -> HookResult<HookAction> {
        let should_log = (step == 0 && self.log_first_step) || (step % self.every_n_steps == 0);

        if should_log {
            let mut msg = format!("Step {}: loss = {:.6}", step, metrics.loss);

            if let Some(acc) = metrics.accuracy {
                msg.push_str(&format!(", accuracy = {:.4}", acc));
            }

            if let Some(auc) = metrics.auc {
                msg.push_str(&format!(", AUC = {:.4}", auc));
            }

            for (name, value) in &metrics.custom {
                msg.push_str(&format!(", {} = {:.4}", name, value));
            }

            info!("{}", msg);
        }

        Ok(HookAction::Continue)
    }

    fn end(&mut self, step: u64, metrics: Option<&Metrics>) -> HookResult<()> {
        if let Some(m) = metrics {
            info!(
                "Training finished at step {}: final loss = {:.6}",
                step, m.loss
            );
        } else {
            info!("Training finished at step {}", step);
        }
        Ok(())
    }
}

/// A hook that saves checkpoints at regular intervals.
///
/// # Examples
///
/// ```
/// use monolith_training::hooks::CheckpointHook;
/// use std::path::PathBuf;
///
/// // Save checkpoint every 1000 steps
/// let hook = CheckpointHook::new(PathBuf::from("/tmp/model"), 1000);
/// ```
pub struct CheckpointHook {
    /// Directory to save checkpoints.
    model_dir: PathBuf,
    /// Save every N steps.
    save_every_n_steps: u64,
    /// Maximum number of checkpoints to keep.
    max_to_keep: usize,
    /// Ordered list of saved checkpoint paths (oldest -> newest).
    saved_paths: VecDeque<PathBuf>,
    /// Checkpoint function (stub for now).
    checkpoint_fn: Option<Arc<dyn Fn(&PathBuf, u64) -> HookResult<()> + Send + Sync>>,
}

impl std::fmt::Debug for CheckpointHook {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointHook")
            .field("model_dir", &self.model_dir)
            .field("save_every_n_steps", &self.save_every_n_steps)
            .field("max_to_keep", &self.max_to_keep)
            .field("checkpoint_fn", &self.checkpoint_fn.is_some())
            .finish()
    }
}

impl CheckpointHook {
    /// Creates a new checkpoint hook.
    ///
    /// # Arguments
    ///
    /// * `model_dir` - The directory to save checkpoints.
    /// * `save_every_n_steps` - The interval at which to save checkpoints.
    pub fn new(model_dir: PathBuf, save_every_n_steps: u64) -> Self {
        Self {
            model_dir,
            save_every_n_steps: save_every_n_steps.max(1),
            max_to_keep: 5,
            saved_paths: VecDeque::new(),
            checkpoint_fn: None,
        }
    }

    /// Sets the maximum number of checkpoints to keep.
    pub fn with_max_to_keep(mut self, max: usize) -> Self {
        self.max_to_keep = max;
        self
    }

    /// Sets a custom checkpoint function.
    pub fn with_checkpoint_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&PathBuf, u64) -> HookResult<()> + Send + Sync + 'static,
    {
        self.checkpoint_fn = Some(Arc::new(f));
        self
    }

    fn save_checkpoint(&self, step: u64) -> HookResult<()> {
        let checkpoint_path = self.model_dir.join(format!("checkpoint-{}.json", step));

        if let Some(ref f) = self.checkpoint_fn {
            f(&checkpoint_path, step)?;
        } else {
            std::fs::create_dir_all(&self.model_dir)?;
            let payload = json!({
                "step": step,
                "format": "monolith_training::hooks::CheckpointHook",
            });
            std::fs::write(&checkpoint_path, payload.to_string())?;
            debug!(
                "Saved default checkpoint payload to {:?} at step {}",
                checkpoint_path, step
            );
        }

        info!("Saved checkpoint at step {} to {:?}", step, checkpoint_path);
        Ok(())
    }

    fn prune_old_checkpoints(&mut self) -> HookResult<()> {
        // `max_to_keep = 0` means "keep all checkpoints".
        if self.max_to_keep == 0 {
            return Ok(());
        }

        while self.saved_paths.len() > self.max_to_keep {
            if let Some(path) = self.saved_paths.pop_front() {
                match std::fs::remove_file(&path) {
                    Ok(()) => {
                        info!("Removed old checkpoint {:?}", path);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(HookError::Io(e)),
                }
            }
        }
        Ok(())
    }
}

impl Hook for CheckpointHook {
    fn name(&self) -> &str {
        "checkpoint_hook"
    }

    fn after_step(&mut self, step: u64, _metrics: &Metrics) -> HookResult<HookAction> {
        if step > 0 && step % self.save_every_n_steps == 0 {
            self.save_checkpoint(step)?;
            self.saved_paths
                .push_back(self.model_dir.join(format!("checkpoint-{}.json", step)));
            self.prune_old_checkpoints()?;
        }
        Ok(HookAction::Continue)
    }

    fn end(&mut self, step: u64, _metrics: Option<&Metrics>) -> HookResult<()> {
        // Always save a final checkpoint
        self.save_checkpoint(step)?;
        self.saved_paths
            .push_back(self.model_dir.join(format!("checkpoint-{}.json", step)));
        self.prune_old_checkpoints()
    }
}

/// A hook that stops training early based on a metric.
///
/// # Examples
///
/// ```
/// use monolith_training::hooks::EarlyStoppingHook;
///
/// // Stop if loss doesn't improve for 10 steps, with 0.001 min delta
/// let hook = EarlyStoppingHook::new("loss", 10, 0.001);
/// ```
#[derive(Debug)]
pub struct EarlyStoppingHook {
    /// Name of the metric to monitor.
    metric_name: String,
    /// Number of steps with no improvement before stopping.
    patience: u64,
    /// Minimum change to qualify as an improvement.
    min_delta: f64,
    /// Whether lower is better for this metric.
    lower_is_better: bool,
    /// Best metric value seen so far.
    best_value: Option<f64>,
    /// Step at which the best value was seen.
    best_step: u64,
    /// Current step count since last improvement.
    steps_without_improvement: u64,
}

impl EarlyStoppingHook {
    /// Creates a new early stopping hook.
    ///
    /// # Arguments
    ///
    /// * `metric_name` - The name of the metric to monitor ("loss", "accuracy", "auc", or custom).
    /// * `patience` - Number of steps to wait for improvement before stopping.
    /// * `min_delta` - Minimum change to qualify as an improvement.
    pub fn new(metric_name: impl Into<String>, patience: u64, min_delta: f64) -> Self {
        let metric_name = metric_name.into();
        let lower_is_better = metric_name == "loss";

        Self {
            metric_name,
            patience,
            min_delta,
            lower_is_better,
            best_value: None,
            best_step: 0,
            steps_without_improvement: 0,
        }
    }

    /// Sets whether lower values are better for this metric.
    ///
    /// By default, "loss" is the only metric where lower is better.
    pub fn with_lower_is_better(mut self, lower_is_better: bool) -> Self {
        self.lower_is_better = lower_is_better;
        self
    }

    fn get_metric_value(&self, metrics: &Metrics) -> Option<f64> {
        match self.metric_name.as_str() {
            "loss" => Some(metrics.loss),
            "accuracy" => metrics.accuracy,
            "auc" => metrics.auc,
            name => metrics.custom.get(name).copied(),
        }
    }

    fn is_improvement(&self, current: f64) -> bool {
        match self.best_value {
            None => true,
            Some(best) => {
                if self.lower_is_better {
                    current < best - self.min_delta
                } else {
                    current > best + self.min_delta
                }
            }
        }
    }
}

impl Hook for EarlyStoppingHook {
    fn name(&self) -> &str {
        "early_stopping_hook"
    }

    fn after_step(&mut self, step: u64, metrics: &Metrics) -> HookResult<HookAction> {
        let current = match self.get_metric_value(metrics) {
            Some(v) => v,
            None => {
                warn!(
                    "EarlyStoppingHook: metric '{}' not found in metrics",
                    self.metric_name
                );
                return Ok(HookAction::Continue);
            }
        };

        if self.is_improvement(current) {
            debug!(
                "EarlyStoppingHook: {} improved from {:?} to {} at step {}",
                self.metric_name, self.best_value, current, step
            );
            self.best_value = Some(current);
            self.best_step = step;
            self.steps_without_improvement = 0;
        } else {
            self.steps_without_improvement += 1;
            debug!(
                "EarlyStoppingHook: no improvement for {} steps (patience: {})",
                self.steps_without_improvement, self.patience
            );

            if self.steps_without_improvement >= self.patience {
                info!(
                    "EarlyStoppingHook: stopping early at step {} (no improvement since step {})",
                    step, self.best_step
                );
                return Ok(HookAction::Stop);
            }
        }

        Ok(HookAction::Continue)
    }
}

/// A collection of hooks that are run together.
#[derive(Default)]
pub struct HookList {
    hooks: Vec<Box<dyn Hook>>,
}

impl HookList {
    /// Creates a new empty hook list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a hook to the list.
    pub fn add<H: Hook + 'static>(&mut self, hook: H) {
        self.hooks.push(Box::new(hook));
    }

    /// Runs `before_step` on all hooks.
    pub fn before_step(&mut self, step: u64) -> HookResult<()> {
        for hook in &mut self.hooks {
            hook.before_step(step)?;
        }
        Ok(())
    }

    /// Runs `after_step` on all hooks.
    ///
    /// Returns `HookAction::Stop` if any hook requests stopping.
    pub fn after_step(&mut self, step: u64, metrics: &Metrics) -> HookResult<HookAction> {
        for hook in &mut self.hooks {
            if hook.after_step(step, metrics)? == HookAction::Stop {
                return Ok(HookAction::Stop);
            }
        }
        Ok(HookAction::Continue)
    }

    /// Runs `end` on all hooks.
    pub fn end(&mut self, step: u64, metrics: Option<&Metrics>) -> HookResult<()> {
        for hook in &mut self.hooks {
            hook.end(step, metrics)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_logging_hook() {
        let mut hook = LoggingHook::new(10);
        let metrics = Metrics::new(0.5, 0);

        assert!(hook.before_step(0).is_ok());
        assert_eq!(hook.after_step(0, &metrics).unwrap(), HookAction::Continue);
        assert_eq!(hook.after_step(5, &metrics).unwrap(), HookAction::Continue);
        assert_eq!(hook.after_step(10, &metrics).unwrap(), HookAction::Continue);
    }

    #[test]
    fn test_checkpoint_hook() {
        let dir = tempdir().unwrap();
        let mut hook = CheckpointHook::new(dir.path().to_path_buf(), 100);
        let metrics = Metrics::new(0.5, 0);

        // Should not checkpoint at step 50
        assert_eq!(hook.after_step(50, &metrics).unwrap(), HookAction::Continue);
        assert!(!dir.path().join("checkpoint-50.json").exists());

        // Should checkpoint at step 100
        assert_eq!(
            hook.after_step(100, &metrics).unwrap(),
            HookAction::Continue
        );
        assert!(dir.path().join("checkpoint-100.json").exists());
    }

    #[test]
    fn test_early_stopping_hook_improvement() {
        let mut hook = EarlyStoppingHook::new("loss", 3, 0.01);

        // First step establishes baseline
        let metrics = Metrics::new(1.0, 0);
        assert_eq!(hook.after_step(0, &metrics).unwrap(), HookAction::Continue);
        assert_eq!(hook.best_value, Some(1.0));

        // Improvement
        let metrics = Metrics::new(0.5, 1);
        assert_eq!(hook.after_step(1, &metrics).unwrap(), HookAction::Continue);
        assert_eq!(hook.best_value, Some(0.5));
        assert_eq!(hook.steps_without_improvement, 0);
    }

    #[test]
    fn test_early_stopping_hook_stop() {
        let mut hook = EarlyStoppingHook::new("loss", 3, 0.01);

        // Establish baseline
        hook.after_step(0, &Metrics::new(0.5, 0)).unwrap();

        // No improvement for 3 steps
        assert_eq!(
            hook.after_step(1, &Metrics::new(0.5, 1)).unwrap(),
            HookAction::Continue
        );
        assert_eq!(
            hook.after_step(2, &Metrics::new(0.5, 2)).unwrap(),
            HookAction::Continue
        );
        assert_eq!(
            hook.after_step(3, &Metrics::new(0.5, 3)).unwrap(),
            HookAction::Stop
        );
    }

    #[test]
    fn test_hook_list() {
        let dir = tempdir().unwrap();
        let mut hooks = HookList::new();
        hooks.add(LoggingHook::new(10));
        hooks.add(CheckpointHook::new(dir.path().to_path_buf(), 100));

        let metrics = Metrics::new(0.5, 0);
        assert!(hooks.before_step(0).is_ok());
        assert_eq!(hooks.after_step(0, &metrics).unwrap(), HookAction::Continue);
        assert!(hooks.end(100, Some(&metrics)).is_ok());
        assert!(dir.path().join("checkpoint-100.json").exists());
    }

    #[test]
    fn test_checkpoint_hook_max_to_keep() {
        let dir = tempdir().unwrap();
        let mut hook = CheckpointHook::new(dir.path().to_path_buf(), 1).with_max_to_keep(2);
        let metrics = Metrics::new(0.5, 0);

        hook.after_step(1, &metrics).unwrap();
        hook.after_step(2, &metrics).unwrap();
        hook.after_step(3, &metrics).unwrap();

        assert!(!dir.path().join("checkpoint-1.json").exists());
        assert!(dir.path().join("checkpoint-2.json").exists());
        assert!(dir.path().join("checkpoint-3.json").exists());
    }
}
