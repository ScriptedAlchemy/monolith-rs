use std::sync::Arc;

use rayon::prelude::*;

use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable, XavierNormalInitializer};
use monolith_layers::{
    dcn::{CrossNetwork, DCNMode},
    dien::{DIENConfig, DIENLayer},
    din::{DINAttention, DINConfig},
    layer::Layer,
    mlp::{ActivationType, MLPConfig, MLP},
    mmoe::{MMoE, MMoEConfig},
    normalization::LayerNorm,
    senet::SENetLayer,
    tensor::Tensor,
};

use super::config::StockPredictorConfig;
use super::data::Sector;
use super::instances::{FeatureIndex, StockInstance};

pub struct EmbeddingTables {
    ticker_table: CuckooEmbeddingHashTable,
    sector_table: CuckooEmbeddingHashTable,
}

impl EmbeddingTables {
    pub fn new(
        num_tickers: usize,
        ticker_dim: usize,
        num_sectors: usize,
        sector_dim: usize,
    ) -> Self {
        let ticker_initializer = Arc::new(XavierNormalInitializer::new(1.0));
        let mut ticker_table = CuckooEmbeddingHashTable::with_initializer(
            num_tickers * 2,
            ticker_dim,
            ticker_initializer,
        );

        let sector_initializer = Arc::new(XavierNormalInitializer::new(1.0));
        let mut sector_table = CuckooEmbeddingHashTable::with_initializer(
            num_sectors * 2,
            sector_dim,
            sector_initializer,
        );

        for i in 0..num_tickers {
            ticker_table.get_or_initialize(i as i64).unwrap();
        }

        for sector in Sector::all() {
            sector_table.get_or_initialize(sector.id()).unwrap();
        }

        Self {
            ticker_table,
            sector_table,
        }
    }

    pub fn lookup_batch(&self, ticker_ids: &[i64], sector_ids: &[i64]) -> (Vec<f32>, Vec<f32>) {
        let batch_size = ticker_ids.len();
        let mut ticker_embeddings = vec![0.0; batch_size * self.ticker_table.dim()];
        let mut sector_embeddings = vec![0.0; batch_size * self.sector_table.dim()];

        self.ticker_table
            .lookup(ticker_ids, &mut ticker_embeddings)
            .unwrap();
        self.sector_table
            .lookup(sector_ids, &mut sector_embeddings)
            .unwrap();

        (ticker_embeddings, sector_embeddings)
    }

    pub fn ticker_dim(&self) -> usize {
        self.ticker_table.dim()
    }

    pub fn sector_dim(&self) -> usize {
        self.sector_table.dim()
    }
}

pub struct StockPredictionModel {
    embeddings: EmbeddingTables,
    senet: SENetLayer,
    dien: DIENLayer,
    din: DINAttention,
    dcn: CrossNetwork,
    layer_norm: LayerNorm,
    residual_norm: LayerNorm,
    mmoe: MMoE,
    direction_head: MLP,
    magnitude_head: MLP,
    profitable_head: MLP,
    config: StockPredictorConfig,
}

impl StockPredictionModel {
    pub fn new(config: &StockPredictorConfig, indicator_dim: usize) -> Self {
        let embeddings = EmbeddingTables::new(
            config.num_tickers,
            config.ticker_embedding_dim,
            Sector::all().len(),
            config.sector_embedding_dim,
        );

        let seq_feature_dim = 4 + indicator_dim;
        let pooled_dim = seq_feature_dim * config.lookbacks.len().max(1);

        // SENet for feature recalibration
        let senet = SENetLayer::new(indicator_dim, 4, true);

        // DIEN for sequential modeling
        let dien = DIENConfig::new(seq_feature_dim, config.dien_hidden_size)
            .with_use_auxiliary_loss(false)
            .build()
            .unwrap();

        // DIN for attention-based sequence modeling
        let din = DINConfig::new(seq_feature_dim)
            .with_attention_hidden_units(vec![64, 32])
            .with_activation(ActivationType::Sigmoid)
            .with_use_softmax(true)
            .build()
            .unwrap();

        // Calculate combined feature dimension (must match `concatenate_enhanced_features`)
        // combined = ticker + sector + dien + din(seq_feature_dim) + senet_indicators(indicator_dim) + pooled(pooled_dim)
        let combined_dim = config.ticker_embedding_dim
            + config.sector_embedding_dim
            + config.dien_hidden_size
            + seq_feature_dim
            + indicator_dim
            + pooled_dim;

        // Deep cross network with residual connections
        let dcn = CrossNetwork::new(combined_dim, config.dcn_cross_layers, DCNMode::Matrix);
        let layer_norm = LayerNorm::new(combined_dim);
        let residual_norm = LayerNorm::new(combined_dim);

        // MMoE for multi-task learning
        let mmoe = MMoEConfig::new(combined_dim, 4, 3)
            .with_expert_hidden_units(vec![128, 64, 32])  // Deeper experts
            .with_expert_activation(ActivationType::GELU)
            .with_expert_output_dim(48)  // Larger output dimension
            .build()
            .unwrap();

        let mmoe_output_dim = 48;  // Updated to match MMoE output
        let direction_head = MLPConfig::new(mmoe_output_dim)
            .add_layer(32, ActivationType::GELU)
            .add_layer(16, ActivationType::GELU)
            .add_layer(1, ActivationType::Sigmoid)
            .build()
            .unwrap();

        let magnitude_head = MLPConfig::new(mmoe_output_dim)
            .add_layer(32, ActivationType::GELU)
            .add_layer(16, ActivationType::GELU)
            .add_layer(1, ActivationType::None)
            .build()
            .unwrap();

        let profitable_head = MLPConfig::new(mmoe_output_dim)
            .add_layer(32, ActivationType::GELU)
            .add_layer(16, ActivationType::GELU)
            .add_layer(1, ActivationType::Sigmoid)
            .build()
            .unwrap();

        Self {
            embeddings,
            senet,
            dien,
            din,
            dcn,
            layer_norm,
            residual_norm,
            mmoe,
            direction_head,
            magnitude_head,
            profitable_head,
            config: config.clone(),
        }
    }

    pub fn forward(&self, batch: &[&StockInstance], features: &FeatureIndex) -> ModelOutput {
        let batch_size = batch.len();
        let lookback = self.config.lookback_window;
        let seq_feature_dim = 4 + features.indicator_dim();

        let ticker_ids: Vec<i64> = batch.iter().map(|i| i.ticker_fid).collect();
        let sector_ids: Vec<i64> = batch.iter().map(|i| i.sector_fid).collect();

        let (ticker_embs, sector_embs) = self.embeddings.lookup_batch(&ticker_ids, &sector_ids);
        let ticker_tensor =
            Tensor::from_data(&[batch_size, self.embeddings.ticker_dim()], ticker_embs);
        let sector_tensor =
            Tensor::from_data(&[batch_size, self.embeddings.sector_dim()], sector_embs);

        let indicator_dim = features.indicator_dim();
        let mut indicator_data = vec![0.0; batch_size * indicator_dim];
        indicator_data
            .par_chunks_mut(indicator_dim)
            .zip(batch.par_iter())
            .for_each(|(chunk, instance)| {
                features.write_indicator_features(instance, chunk);
            });
        let indicator_tensor = Tensor::from_data(&[batch_size, indicator_dim], indicator_data);

        let senet_indicators = self.senet.forward(&indicator_tensor).unwrap();

        let mut seq_data = vec![0.0; batch_size * lookback * seq_feature_dim];
        let per_batch = lookback * seq_feature_dim;
        seq_data
            .par_chunks_mut(per_batch)
            .zip(batch.par_iter())
            .for_each(|(chunk, instance)| {
                features.write_sequence_features(instance, lookback, chunk);
            });
        let mut seq_tensor = Tensor::from_data(&[batch_size, lookback, seq_feature_dim], seq_data);

        // Multi-lookback pooled features: for each configured lookback L, compute a mean pool
        // over the last L steps of the sequence tensor. This gives multi-horizon context without
        // storing extra sequences.
        let lookbacks = if self.config.lookbacks.is_empty() {
            vec![lookback]
        } else {
            self.config.lookbacks.clone()
        };
        let pooled_dim = seq_feature_dim * lookbacks.len();
        let mut pooled_data = vec![0.0; batch_size * pooled_dim];
        let seq_flat = seq_tensor.data();
        pooled_data
            .par_chunks_mut(pooled_dim)
            .enumerate()
            .for_each(|(b, out)| {
                let base = b * lookback * seq_feature_dim;
                for (li, &l_raw) in lookbacks.iter().enumerate() {
                    let l = l_raw.min(lookback).max(1);
                    let start_step = lookback - l;
                    let out_off = li * seq_feature_dim;
                    for d in 0..seq_feature_dim {
                        let mut sum = 0.0f32;
                        for step in start_step..lookback {
                            sum += seq_flat[base + step * seq_feature_dim + d];
                        }
                        out[out_off + d] = sum / l as f32;
                    }
                }
            });
        let pooled_tensor = Tensor::from_data(&[batch_size, pooled_dim], pooled_data);

        let mut target_data = vec![0.0; batch_size * seq_feature_dim];
        for b in 0..batch_size {
            let seq_offset = b * lookback * seq_feature_dim + (lookback - 1) * seq_feature_dim;
            let target_offset = b * seq_feature_dim;
            target_data[target_offset..target_offset + seq_feature_dim]
                .copy_from_slice(&seq_tensor.data()[seq_offset..seq_offset + seq_feature_dim]);
        }
        let target_tensor = Tensor::from_data(&[batch_size, seq_feature_dim], target_data);

        let dien_output = self
            .dien
            .forward_dien(&seq_tensor, &target_tensor, None)
            .unwrap();

        let din_output = self
            .din
            .forward_attention(&target_tensor, &seq_tensor, &seq_tensor, None)
            .unwrap();

        let combined = self.concatenate_enhanced_features(
            &ticker_tensor,
            &sector_tensor,
            &dien_output,
            &din_output,
            &senet_indicators,
            &pooled_tensor,
        );

        // Residual connection through DCN
        let dcn_output = self.dcn.forward(&combined).unwrap();
        let residual_combined = self.add_residual(&combined, &dcn_output);
        let normalized = self.layer_norm.forward(&residual_combined).unwrap();

        let mmoe_outputs = self.mmoe.forward_multi(&normalized).unwrap();

        let direction_pred = self.direction_head.forward(&mmoe_outputs[0]).unwrap();
        let magnitude_pred = self.magnitude_head.forward(&mmoe_outputs[1]).unwrap();
        let profitable_pred = self.profitable_head.forward(&mmoe_outputs[2]).unwrap();

        ModelOutput {
            direction: direction_pred.data().to_vec(),
            magnitude: magnitude_pred.data().to_vec(),
            profitable: profitable_pred.data().to_vec(),
        }
    }

    fn concatenate_enhanced_features(
        &self,
        ticker: &Tensor,
        sector: &Tensor,
        dien: &Tensor,
        din: &Tensor,
        senet_indicators: &Tensor,
        pooled: &Tensor,
    ) -> Tensor {
        let batch_size = ticker.shape()[0];
        let total_dim = ticker.shape()[1]
            + sector.shape()[1]
            + dien.shape()[1]
            + din.shape()[1]
            + senet_indicators.shape()[1]
            + pooled.shape()[1];

        let mut data = vec![0.0; batch_size * total_dim];

        for b in 0..batch_size {
            let mut offset = b * total_dim;

            for i in 0..ticker.shape()[1] {
                data[offset + i] = ticker.data()[b * ticker.shape()[1] + i];
            }
            offset += ticker.shape()[1];

            for i in 0..sector.shape()[1] {
                data[offset + i] = sector.data()[b * sector.shape()[1] + i];
            }
            offset += sector.shape()[1];

            for i in 0..dien.shape()[1] {
                data[offset + i] = dien.data()[b * dien.shape()[1] + i];
            }
            offset += dien.shape()[1];

            for i in 0..din.shape()[1] {
                data[offset + i] = din.data()[b * din.shape()[1] + i];
            }
            offset += din.shape()[1];

            for i in 0..senet_indicators.shape()[1] {
                data[offset + i] = senet_indicators.data()[b * senet_indicators.shape()[1] + i];
            }
            offset += senet_indicators.shape()[1];

            for i in 0..pooled.shape()[1] {
                data[offset + i] = pooled.data()[b * pooled.shape()[1] + i];
            }
        }

        Tensor::from_data(&[batch_size, total_dim], data)
    }

    fn add_residual(&self, input: &Tensor, residual: &Tensor) -> Tensor {
        let batch_size = input.shape()[0];
        let dim = input.shape()[1];
        let mut data = vec![0.0; batch_size * dim];

        for b in 0..batch_size {
            for d in 0..dim {
                let idx = b * dim + d;
                data[idx] = input.data()[idx] + residual.data()[idx];
            }
        }

        Tensor::from_data(&[batch_size, dim], data)
    }

    pub fn compute_loss(
        &self,
        output: &ModelOutput,
        batch: &[&StockInstance],
    ) -> (f32, f32, f32, f32) {
        let batch_size = batch.len() as f32;

        // Direction loss (binary cross-entropy)
        let mut direction_loss = 0.0;
        for (i, instance) in batch.iter().enumerate() {
            let pred = output.direction[i].clamp(1e-7, 1.0 - 1e-7);
            let label = instance.direction_label;
            direction_loss += -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln();
        }
        direction_loss /= batch_size;

        // Magnitude loss (improved Huber loss with dynamic scaling)
        let mut magnitude_loss = 0.0;
        let mut magnitude_mae = 0.0;
        for (i, instance) in batch.iter().enumerate() {
            let diff = (output.magnitude[i] - instance.magnitude_label).abs();
            magnitude_mae += diff;

            // Adaptive delta based on label magnitude
            let delta = (instance.magnitude_label.abs() * 0.5).max(0.01).min(2.0);
            if diff <= delta {
                magnitude_loss += 0.5 * diff * diff / delta;  // Normalized L2
            } else {
                magnitude_loss += diff - 0.5 * delta;  // L1 for outliers
            }
        }
        magnitude_loss /= batch_size;
        magnitude_mae /= batch_size;

        // Scale magnitude loss based on current MAE to prevent vanishing gradients
        let magnitude_scale = (magnitude_mae * 10.0).max(0.1).min(5.0);
        magnitude_loss *= magnitude_scale;

        // Profitable loss (binary cross-entropy with label smoothing)
        let mut profitable_loss = 0.0;
        let label_smoothing = 0.1;
        for (i, instance) in batch.iter().enumerate() {
            let pred = output.profitable[i].clamp(1e-7, 1.0 - 1e-7);
            let label = instance.profitable_label;
            let smooth_label = label * (1.0 - label_smoothing) + 0.5 * label_smoothing;
            profitable_loss += -smooth_label * pred.ln() - (1.0 - smooth_label) * (1.0 - pred).ln();
        }
        profitable_loss /= batch_size;

        // Balanced total loss with adaptive weighting
        let direction_weight = 0.5;
        let magnitude_weight = 0.3;
        let profitable_weight = 0.2;

        let total_loss = direction_weight * direction_loss +
                        magnitude_weight * magnitude_loss +
                        profitable_weight * profitable_loss;

        (total_loss, direction_loss, magnitude_loss, profitable_loss)
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.senet.parameters_mut());
        params.extend(self.dien.parameters_mut());
        params.extend(self.din.parameters_mut());
        params.extend(self.dcn.parameters_mut());
        params.extend(self.layer_norm.parameters_mut());
        params.extend(self.residual_norm.parameters_mut());
        params.extend(self.mmoe.parameters_mut());
        params.extend(self.direction_head.parameters_mut());
        params.extend(self.magnitude_head.parameters_mut());
        params.extend(self.profitable_head.parameters_mut());
        params
    }
}

#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub direction: Vec<f32>,
    pub magnitude: Vec<f32>,
    pub profitable: Vec<f32>,
}
