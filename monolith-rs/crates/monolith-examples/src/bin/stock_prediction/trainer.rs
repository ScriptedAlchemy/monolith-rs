use super::config::StockPredictorConfig;
use super::data::RandomGenerator;
use super::instances::{create_batches, FeatureIndex, StockInstance};
use super::model::StockPredictionModel;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    pub step: usize,
    pub total_loss: f32,
    pub direction_loss: f32,
    pub magnitude_loss: f32,
    pub profitable_loss: f32,
    pub direction_accuracy: f32,
    pub samples_processed: usize,
}

#[derive(Debug, Clone, Default)]
pub struct DeviceMetrics {
    pub forward_time: Duration,
    pub backward_time: Duration,
    pub loss_time: Duration,
    pub data_prep_time: Duration,
    pub total_time: Duration,
    pub samples: usize,
}

impl DeviceMetrics {
    pub fn throughput(&self) -> f32 {
        if self.total_time.as_secs_f32() > 0.0 {
            self.samples as f32 / self.total_time.as_secs_f32()
        } else {
            0.0
        }
    }

    pub fn gpu_activity_pct(&self) -> f32 {
        // Estimate GPU activity as forward+backward time / total time
        // This is an approximation since forward/backward use GPU heavily
        let compute_time = self.forward_time + self.backward_time;
        if self.total_time.as_secs_f32() > 0.0 {
            100.0 * compute_time.as_secs_f32() / self.total_time.as_secs_f32()
        } else {
            0.0
        }
    }
}

pub struct Trainer {
    model: StockPredictionModel,
    config: StockPredictorConfig,
    learning_rate: f32,
    initial_lr: f32,
    global_step: usize,
    epoch: usize,
    best_eval_loss: f32,
    patience_counter: usize,
    rng: RandomGenerator,
    grad_clip: f32,
    warmup_steps: usize,
    device_metrics: DeviceMetrics,
}

impl Trainer {
    pub fn new(config: &StockPredictorConfig, indicator_dim: usize) -> Self {
        let model = StockPredictionModel::new(config, indicator_dim);
        let warmup_steps = 50;

        Self {
            model,
            config: config.clone(),
            learning_rate: config.learning_rate,
            initial_lr: config.learning_rate,
            global_step: 0,
            epoch: 0,
            best_eval_loss: f32::MAX,
            patience_counter: 0,
            rng: RandomGenerator::new(config.seed),
            grad_clip: 1.0,
            warmup_steps,
            device_metrics: DeviceMetrics::default(),
        }
    }

    pub fn device_metrics(&self) -> &DeviceMetrics {
        &self.device_metrics
    }

    pub fn reset_device_metrics(&mut self) {
        self.device_metrics = DeviceMetrics::default();
    }

    pub fn update_lr(&mut self) {
        if self.global_step < self.warmup_steps {
            let warmup_progress = self.global_step as f32 / self.warmup_steps as f32;
            self.learning_rate = self.initial_lr * warmup_progress;
        } else {
            let progress = self.epoch as f32 / self.config.num_epochs as f32;
            let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.learning_rate = self.initial_lr * decay.max(0.01);
        }
    }

    pub fn decay_lr(&mut self) {
        self.epoch += 1;
        self.update_lr();
    }

    pub fn current_lr(&self) -> f32 {
        self.learning_rate
    }

    pub fn train_epoch(
        &mut self,
        train_instances: &[StockInstance],
        features: &FeatureIndex,
    ) -> TrainingMetrics {
        let epoch_start = Instant::now();
        let mut indices: Vec<usize> = (0..train_instances.len()).collect();
        self.rng.shuffle(&mut indices);

        let mut epoch_metrics = TrainingMetrics::default();
        let mut direction_correct = 0;
        let mut total_samples = 0;

        let mut forward_time = Duration::ZERO;
        let mut backward_time = Duration::ZERO;
        let mut loss_time = Duration::ZERO;
        let mut data_prep_time = Duration::ZERO;

        for chunk in indices.chunks(self.config.batch_size) {
            let t0 = Instant::now();
            let batch: Vec<&StockInstance> = chunk.iter().map(|&i| &train_instances[i]).collect();
            data_prep_time += t0.elapsed();

            // Forward pass with gradient tracking
            let t1 = Instant::now();
            let (output, cache) = self.model.forward_train(&batch, features);
            forward_time += t1.elapsed();

            let t2 = Instant::now();
            let (total_loss, direction_loss, magnitude_loss, profitable_loss) =
                self.model.compute_loss(&output, &batch);
            loss_time += t2.elapsed();

            epoch_metrics.total_loss += total_loss * batch.len() as f32;
            epoch_metrics.direction_loss += direction_loss * batch.len() as f32;
            epoch_metrics.magnitude_loss += magnitude_loss * batch.len() as f32;
            epoch_metrics.profitable_loss += profitable_loss * batch.len() as f32;

            for (i, instance) in batch.iter().enumerate() {
                let pred = if output.direction[i] > 0.5 { 1.0 } else { 0.0 };
                if (pred - instance.direction_label).abs() < 0.1 {
                    direction_correct += 1;
                }
            }
            total_samples += batch.len();

            // Backpropagation step
            let t3 = Instant::now();
            self.backprop_step(&batch, &output, &cache);
            backward_time += t3.elapsed();

            self.global_step += 1;

            if self.config.verbose && self.global_step.is_multiple_of(self.config.log_every_n_steps) {
                println!(
                    "  [Step {}] Loss: {:.4} | Dir: {:.4} | Mag: {:.4} | Prof: {:.4}",
                    self.global_step, total_loss, direction_loss, magnitude_loss, profitable_loss
                );
            }
        }

        // Update device metrics
        self.device_metrics.forward_time += forward_time;
        self.device_metrics.backward_time += backward_time;
        self.device_metrics.loss_time += loss_time;
        self.device_metrics.data_prep_time += data_prep_time;
        self.device_metrics.total_time += epoch_start.elapsed();
        self.device_metrics.samples += total_samples;

        epoch_metrics.step = self.global_step;
        epoch_metrics.samples_processed = total_samples;
        if total_samples > 0 {
            epoch_metrics.total_loss /= total_samples as f32;
            epoch_metrics.direction_loss /= total_samples as f32;
            epoch_metrics.magnitude_loss /= total_samples as f32;
            epoch_metrics.profitable_loss /= total_samples as f32;
            epoch_metrics.direction_accuracy = direction_correct as f32 / total_samples as f32;
        }

        epoch_metrics
    }

    fn backprop_step(
        &mut self,
        batch: &[&StockInstance],
        output: &super::model::ModelOutput,
        cache: &super::model::ForwardCache,
    ) {
        self.update_lr();

        // Compute loss gradients
        let loss_grads = self.model.compute_loss_gradients(output, batch);

        // Backward pass (computes gradients for all layers)
        self.model.backward(&loss_grads, cache);

        // Apply gradients using SGD with gradient clipping
        self.model.apply_gradients(self.learning_rate, self.grad_clip);
    }

    pub fn evaluate(
        &self,
        eval_instances: &[StockInstance],
        features: &FeatureIndex,
    ) -> TrainingMetrics {
        let batches = create_batches(eval_instances, self.config.batch_size);

        let mut metrics = TrainingMetrics::default();
        let mut direction_correct = 0;
        let mut total_samples = 0;

        for batch in batches {
            let output = self.model.forward(&batch, features);
            let (total_loss, direction_loss, magnitude_loss, profitable_loss) =
                self.model.compute_loss(&output, &batch);

            metrics.total_loss += total_loss * batch.len() as f32;
            metrics.direction_loss += direction_loss * batch.len() as f32;
            metrics.magnitude_loss += magnitude_loss * batch.len() as f32;
            metrics.profitable_loss += profitable_loss * batch.len() as f32;

            for (i, instance) in batch.iter().enumerate() {
                let pred = if output.direction[i] > 0.5 { 1.0 } else { 0.0 };
                if (pred - instance.direction_label).abs() < 0.1 {
                    direction_correct += 1;
                }
            }
            total_samples += batch.len();
        }

        metrics.step = self.global_step;
        metrics.samples_processed = total_samples;
        if total_samples > 0 {
            metrics.total_loss /= total_samples as f32;
            metrics.direction_loss /= total_samples as f32;
            metrics.magnitude_loss /= total_samples as f32;
            metrics.profitable_loss /= total_samples as f32;
            metrics.direction_accuracy = direction_correct as f32 / total_samples as f32;
        }

        metrics
    }

    pub fn check_early_stopping(&mut self, eval_loss: f32) -> bool {
        if self.config.early_stopping_patience == 0 {
            return false;
        }

        let min_delta = self.config.early_stopping_min_delta.max(0.0);
        if eval_loss < self.best_eval_loss - min_delta {
            self.best_eval_loss = eval_loss;
            self.patience_counter = 0;
            false
        } else {
            self.patience_counter += 1;
            self.patience_counter >= self.config.early_stopping_patience
        }
    }

    pub fn model(&self) -> &StockPredictionModel {
        &self.model
    }
}
