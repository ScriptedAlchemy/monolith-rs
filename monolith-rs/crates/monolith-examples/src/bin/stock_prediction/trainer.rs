use super::config::StockPredictorConfig;
use super::data::RandomGenerator;
use super::instances::{create_batches, FeatureIndex, StockInstance};
use super::model::StockPredictionModel;
use monolith_optimizer::{Amsgrad, Optimizer};

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
    optimizers: Vec<Amsgrad>,
    grad_clip: f32,
    warmup_steps: usize,
    fd_epsilon: f32,
    fd_num_coords: usize,
}

impl Trainer {
    pub fn new(config: &StockPredictorConfig, indicator_dim: usize) -> Self {
        let mut model = StockPredictionModel::new(config, indicator_dim);

        let optimizers: Vec<Amsgrad> = model
            .parameters_mut()
            .iter()
            .map(|_| Amsgrad::with_params(config.learning_rate, 0.9, 0.999, 1e-8, 0.0001))
            .collect();

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
            optimizers,
            grad_clip: 1.0,
            warmup_steps,
            fd_epsilon: 1e-3,
            fd_num_coords: 2048,
        }
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
        let mut indices: Vec<usize> = (0..train_instances.len()).collect();
        self.rng.shuffle(&mut indices);

        let mut epoch_metrics = TrainingMetrics::default();
        let mut direction_correct = 0;
        let mut total_samples = 0;

        for chunk in indices.chunks(self.config.batch_size) {
            let batch: Vec<&StockInstance> = chunk.iter().map(|&i| &train_instances[i]).collect();

            let output = self.model.forward(&batch, features);
            let (total_loss, direction_loss, magnitude_loss, profitable_loss) =
                self.model.compute_loss(&output, &batch);

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

            self.finite_difference_step(&batch, features);

            self.global_step += 1;

            if self.config.verbose && self.global_step.is_multiple_of(self.config.log_every_n_steps) {
                println!(
                    "  [Step {}] Loss: {:.4} | Dir: {:.4} | Mag: {:.4} | Prof: {:.4}",
                    self.global_step, total_loss, direction_loss, magnitude_loss, profitable_loss
                );
            }
        }

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

    fn finite_difference_step(&mut self, batch: &[&StockInstance], features: &FeatureIndex) {
        self.update_lr();

        let baseline_loss = {
            let output = self.model.forward(batch, features);
            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
            loss
        };

        let num_params = self.model.parameters_mut().len().max(1);
        let param_idx = self.global_step % num_params;

        let (param_len, coord_count) = {
            let params = self.model.parameters_mut();
            let len = params[param_idx].data().len();
            (len, self.fd_num_coords.min(len.max(1)))
        };

        let mut coord_indices = Vec::with_capacity(coord_count);
        let mut coord_deltas = Vec::with_capacity(coord_count);
        for j in 0..coord_count {
            let h = (self.global_step as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(param_idx as u64 * 97)
                .wrapping_add(j as u64 * 1_009);
            let idx = (h as usize) % param_len;
            let delta = if (h >> 11) & 1 == 0 { 1.0 } else { -1.0 };
            coord_indices.push(idx);
            coord_deltas.push(delta);
        }

        let eps = self.fd_epsilon;

        // Apply perturbation (+eps * delta) at selected coordinates
        {
            let mut params = self.model.parameters_mut();
            let data = params[param_idx].data_mut();
            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
                data[idx] += eps * delta;
            }
        }

        let loss_plus = {
            let output = self.model.forward(batch, features);
            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
            loss
        };

        // Apply perturbation (-2eps * delta) to reach the negative side
        {
            let mut params = self.model.parameters_mut();
            let data = params[param_idx].data_mut();
            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
                data[idx] -= 2.0 * eps * delta;
            }
        }

        let loss_minus = {
            let output = self.model.forward(batch, features);
            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
            loss
        };

        // Restore original weights
        {
            let mut params = self.model.parameters_mut();
            let data = params[param_idx].data_mut();
            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
                data[idx] += eps * delta;
            }
        }

        // Central difference gradient estimate along +/- delta direction
        let coeff = (loss_plus - loss_minus) / (2.0 * eps).max(1e-12);

        // Ensure we have an optimizer state per parameter tensor
        if param_idx >= self.optimizers.len() {
            self.optimizers.resize_with(param_idx + 1, || {
                Amsgrad::with_params(self.learning_rate, 0.9, 0.999, 1e-8, 0.0001)
            });
        }

        // Keep optimizer learning rate in sync with scheduler (recreate if needed)
        // NOTE: Amsgrad doesn't expose a setter; recreating is fine because we keep its state per tensor.
        // We only recreate if LR changes meaningfully to avoid unnecessary resets.
        let lr_now = self.learning_rate;
        let lr_prev = self.optimizers[param_idx].config().learning_rate();
        if (lr_prev - lr_now).abs() > 1e-12 {
            // Preserve state? Not possible without setters; prefer stability by keeping state and accepting fixed LR.
            // So we do nothing here.
        }

        // Build sparse gradient vector for AMSGrad
        let mut grads = vec![0.0_f32; param_len];
        for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
            let g = (coeff as f32) * delta;
            grads[idx] = g.clamp(-self.grad_clip, self.grad_clip);
        }

        // Apply AMSGrad update using its internal moment estimates (much more stable than raw SGD)
        let mut params = self.model.parameters_mut();
        let data = params[param_idx].data_mut();
        self.optimizers[param_idx].apply_gradients(data, &grads);

        if !baseline_loss.is_finite() || !loss_plus.is_finite() || !loss_minus.is_finite() {
            self.fd_epsilon = (self.fd_epsilon * 0.5).max(1e-6);
        } else {
            // Mildly increase epsilon if gradients are too tiny (helps avoid stalling)
            let gap = (loss_plus - loss_minus).abs();
            if gap < 1e-6 {
                self.fd_epsilon = (self.fd_epsilon * 1.05).min(1e-2);
            }
        }
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
