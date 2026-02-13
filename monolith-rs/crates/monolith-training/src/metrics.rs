//! Training metrics collection and recording.
//!
//! This module provides types for tracking training metrics like loss, accuracy, and AUC,
//! as well as a recorder for accumulating metrics over training steps.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Training metrics collected during a training step or evaluation.
///
/// Contains common metrics used in machine learning training loops.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    /// The loss value for this step/evaluation.
    pub loss: f64,
    /// Classification accuracy (0.0 to 1.0).
    pub accuracy: Option<f64>,
    /// Area Under the ROC Curve.
    pub auc: Option<f64>,
    /// Custom metrics with string keys.
    pub custom: HashMap<String, f64>,
    /// The global step at which these metrics were recorded.
    pub global_step: u64,
}

impl Metrics {
    /// Creates a new `Metrics` instance with the given loss and step.
    ///
    /// # Arguments
    ///
    /// * `loss` - The loss value for this step.
    /// * `global_step` - The global training step number.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_training::metrics::Metrics;
    ///
    /// let metrics = Metrics::new(0.5, 100);
    /// assert_eq!(metrics.loss, 0.5);
    /// assert_eq!(metrics.global_step, 100);
    /// ```
    pub fn new(loss: f64, global_step: u64) -> Self {
        Self {
            loss,
            accuracy: None,
            auc: None,
            custom: HashMap::new(),
            global_step,
        }
    }

    /// Sets the accuracy metric.
    ///
    /// # Arguments
    ///
    /// * `accuracy` - The accuracy value (should be between 0.0 and 1.0).
    pub fn with_accuracy(mut self, accuracy: f64) -> Self {
        self.accuracy = Some(accuracy);
        self
    }

    /// Sets the AUC metric.
    ///
    /// # Arguments
    ///
    /// * `auc` - The AUC value (should be between 0.0 and 1.0).
    pub fn with_auc(mut self, auc: f64) -> Self {
        self.auc = Some(auc);
        self
    }

    /// Adds a custom metric.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the custom metric.
    /// * `value` - The value of the custom metric.
    pub fn with_custom(mut self, name: impl Into<String>, value: f64) -> Self {
        self.custom.insert(name.into(), value);
        self
    }
}

/// Accumulates metrics over multiple training steps.
///
/// Useful for computing running averages of metrics during training.
#[derive(Debug, Clone, Default)]
pub struct MetricsRecorder {
    /// Accumulated loss values.
    loss_sum: f64,
    /// Accumulated accuracy values.
    accuracy_sum: f64,
    /// Accumulated AUC values.
    auc_sum: f64,
    /// Number of samples with accuracy.
    accuracy_count: u64,
    /// Number of samples with AUC.
    auc_count: u64,
    /// Accumulated custom metrics.
    custom_sums: HashMap<String, f64>,
    /// Counts for custom metrics.
    custom_counts: HashMap<String, u64>,
    /// Total number of samples recorded.
    count: u64,
}

impl MetricsRecorder {
    /// Creates a new empty `MetricsRecorder`.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_training::metrics::MetricsRecorder;
    ///
    /// let recorder = MetricsRecorder::new();
    /// assert_eq!(recorder.count(), 0);
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a set of metrics.
    ///
    /// # Arguments
    ///
    /// * `metrics` - The metrics to record.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_training::metrics::{Metrics, MetricsRecorder};
    ///
    /// let mut recorder = MetricsRecorder::new();
    /// recorder.record(&Metrics::new(0.5, 1));
    /// recorder.record(&Metrics::new(0.3, 2));
    /// assert_eq!(recorder.count(), 2);
    /// ```
    pub fn record(&mut self, metrics: &Metrics) {
        self.loss_sum += metrics.loss;
        self.count += 1;

        if let Some(acc) = metrics.accuracy {
            self.accuracy_sum += acc;
            self.accuracy_count += 1;
        }

        if let Some(auc) = metrics.auc {
            self.auc_sum += auc;
            self.auc_count += 1;
        }

        for (name, value) in &metrics.custom {
            *self.custom_sums.entry(name.clone()).or_insert(0.0) += value;
            *self.custom_counts.entry(name.clone()).or_insert(0) += 1;
        }
    }

    /// Returns the number of metrics recorded.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Returns the average loss.
    ///
    /// Returns 0.0 if no metrics have been recorded.
    pub fn average_loss(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.loss_sum / self.count as f64
        }
    }

    /// Returns the average accuracy, if any accuracy values were recorded.
    pub fn average_accuracy(&self) -> Option<f64> {
        if self.accuracy_count == 0 {
            None
        } else {
            Some(self.accuracy_sum / self.accuracy_count as f64)
        }
    }

    /// Returns the average AUC, if any AUC values were recorded.
    pub fn average_auc(&self) -> Option<f64> {
        if self.auc_count == 0 {
            None
        } else {
            Some(self.auc_sum / self.auc_count as f64)
        }
    }

    /// Returns the average value of a custom metric.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the custom metric.
    pub fn average_custom(&self, name: &str) -> Option<f64> {
        match (self.custom_sums.get(name), self.custom_counts.get(name)) {
            (Some(&sum), Some(&count)) if count > 0 => Some(sum / count as f64),
            _ => None,
        }
    }

    /// Computes aggregate metrics at the given global step.
    ///
    /// # Arguments
    ///
    /// * `global_step` - The global step to assign to the aggregated metrics.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_training::metrics::{Metrics, MetricsRecorder};
    ///
    /// let mut recorder = MetricsRecorder::new();
    /// recorder.record(&Metrics::new(0.5, 1).with_accuracy(0.8));
    /// recorder.record(&Metrics::new(0.3, 2).with_accuracy(0.9));
    ///
    /// let avg = recorder.aggregate(2);
    /// assert!((avg.loss - 0.4).abs() < 1e-10);
    /// assert!(
    ///     (avg
    ///         .accuracy
    ///         .expect("accuracy should be present after recording accuracy metrics")
    ///         - 0.85)
    ///         .abs()
    ///         < 1e-10
    /// );
    /// ```
    pub fn aggregate(&self, global_step: u64) -> Metrics {
        let mut metrics = Metrics::new(self.average_loss(), global_step);

        if let Some(acc) = self.average_accuracy() {
            metrics = metrics.with_accuracy(acc);
        }

        if let Some(auc) = self.average_auc() {
            metrics = metrics.with_auc(auc);
        }

        for name in self.custom_sums.keys() {
            if let Some(avg) = self.average_custom(name) {
                metrics = metrics.with_custom(name.clone(), avg);
            }
        }

        metrics
    }

    /// Resets the recorder to its initial state.
    ///
    /// # Examples
    ///
    /// ```
    /// use monolith_training::metrics::{Metrics, MetricsRecorder};
    ///
    /// let mut recorder = MetricsRecorder::new();
    /// recorder.record(&Metrics::new(0.5, 1));
    /// recorder.reset();
    /// assert_eq!(recorder.count(), 0);
    /// ```
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new() {
        let metrics = Metrics::new(0.5, 100);
        assert_eq!(metrics.loss, 0.5);
        assert_eq!(metrics.global_step, 100);
        assert!(metrics.accuracy.is_none());
        assert!(metrics.auc.is_none());
        assert!(metrics.custom.is_empty());
    }

    #[test]
    fn test_metrics_builder() {
        let metrics = Metrics::new(0.5, 100)
            .with_accuracy(0.9)
            .with_auc(0.95)
            .with_custom("precision", 0.88);

        assert_eq!(metrics.accuracy, Some(0.9));
        assert_eq!(metrics.auc, Some(0.95));
        assert_eq!(metrics.custom.get("precision"), Some(&0.88));
    }

    #[test]
    fn test_metrics_recorder_empty() {
        let recorder = MetricsRecorder::new();
        assert_eq!(recorder.count(), 0);
        assert_eq!(recorder.average_loss(), 0.0);
        assert!(recorder.average_accuracy().is_none());
        assert!(recorder.average_auc().is_none());
    }

    #[test]
    fn test_metrics_recorder_record() {
        let mut recorder = MetricsRecorder::new();
        recorder.record(&Metrics::new(0.5, 1).with_accuracy(0.8));
        recorder.record(&Metrics::new(0.3, 2).with_accuracy(0.9));

        assert_eq!(recorder.count(), 2);
        assert!((recorder.average_loss() - 0.4).abs() < 1e-10);
        assert!(
            (recorder
                .average_accuracy()
                .expect("average accuracy should be present after accuracy-bearing records")
                - 0.85)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_metrics_recorder_aggregate() {
        let mut recorder = MetricsRecorder::new();
        recorder.record(
            &Metrics::new(0.5, 1)
                .with_accuracy(0.8)
                .with_custom("f1", 0.7),
        );
        recorder.record(
            &Metrics::new(0.3, 2)
                .with_accuracy(0.9)
                .with_custom("f1", 0.9),
        );

        let agg = recorder.aggregate(100);
        assert_eq!(agg.global_step, 100);
        assert!((agg.loss - 0.4).abs() < 1e-10);
        assert!(
            (agg.accuracy
                .expect("aggregated accuracy should be present after accuracy-bearing records")
                - 0.85)
                .abs()
                < 1e-10
        );
        assert!(
            (agg.custom
                .get("f1")
                .expect("aggregated custom f1 metric should be present")
                - 0.8)
                .abs()
                < 1e-10
        );
    }

    #[test]
    fn test_metrics_recorder_reset() {
        let mut recorder = MetricsRecorder::new();
        recorder.record(&Metrics::new(0.5, 1));
        recorder.reset();
        assert_eq!(recorder.count(), 0);
        assert_eq!(recorder.average_loss(), 0.0);
    }
}
