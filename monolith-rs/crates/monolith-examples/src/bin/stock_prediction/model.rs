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
use super::indicators::TechnicalIndicators;
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
        let senet = SENetLayer::new(indicator_dim, 4, true);

        let dien = DIENConfig::new(seq_feature_dim, config.dien_hidden_size)
            .with_use_auxiliary_loss(false)
            .build()
            .unwrap();

        let din = DINConfig::new(seq_feature_dim)
            .with_attention_hidden_units(vec![64, 32])
            .with_activation(ActivationType::Sigmoid)
            .with_use_softmax(true)
            .build()
            .unwrap();

        let combined_dim = config.ticker_embedding_dim
            + config.sector_embedding_dim
            + config.dien_hidden_size
            + seq_feature_dim
            + indicator_dim;

        let dcn = CrossNetwork::new(combined_dim, config.dcn_cross_layers, DCNMode::Matrix);
        let layer_norm = LayerNorm::new(combined_dim);

        let mmoe = MMoEConfig::new(combined_dim, 4, 3)
            .with_expert_hidden_units(vec![64, 32])
            .with_expert_activation(ActivationType::GELU)
            .with_expert_output_dim(32)
            .build()
            .unwrap();

        let mmoe_output_dim = 32;
        let direction_head = MLPConfig::new(mmoe_output_dim)
            .add_layer(16, ActivationType::GELU)
            .add_layer(1, ActivationType::Sigmoid)
            .build()
            .unwrap();

        let magnitude_head = MLPConfig::new(mmoe_output_dim)
            .add_layer(16, ActivationType::GELU)
            .add_layer(1, ActivationType::None)
            .build()
            .unwrap();

        let profitable_head = MLPConfig::new(mmoe_output_dim)
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
        let seq_tensor = Tensor::from_data(&[batch_size, lookback, seq_feature_dim], seq_data);

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
        );

        let dcn_output = self.dcn.forward(&combined).unwrap();
        let normalized = self.layer_norm.forward(&dcn_output).unwrap();
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
    ) -> Tensor {
        let batch_size = ticker.shape()[0];
        let total_dim = ticker.shape()[1]
            + sector.shape()[1]
            + dien.shape()[1]
            + din.shape()[1]
            + senet_indicators.shape()[1];

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
        }

        Tensor::from_data(&[batch_size, total_dim], data)
    }

    pub fn compute_loss(
        &self,
        output: &ModelOutput,
        batch: &[&StockInstance],
    ) -> (f32, f32, f32, f32) {
        let batch_size = batch.len() as f32;

        let mut direction_loss = 0.0;
        for (i, instance) in batch.iter().enumerate() {
            let pred = output.direction[i].clamp(1e-7, 1.0 - 1e-7);
            let label = instance.direction_label;
            direction_loss += -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln();
        }
        direction_loss /= batch_size;

        let mut magnitude_loss = 0.0;
        let delta = 5.0;
        for (i, instance) in batch.iter().enumerate() {
            let diff = (output.magnitude[i] - instance.magnitude_label).abs();
            if diff <= delta {
                magnitude_loss += 0.5 * diff * diff;
            } else {
                magnitude_loss += delta * (diff - 0.5 * delta);
            }
        }
        magnitude_loss /= batch_size;
        magnitude_loss /= 10.0;

        let mut profitable_loss = 0.0;
        for (i, instance) in batch.iter().enumerate() {
            let pred = output.profitable[i].clamp(1e-7, 1.0 - 1e-7);
            let label = instance.profitable_label;
            profitable_loss += -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln();
        }
        profitable_loss /= batch_size;

        let total_loss = 0.4 * direction_loss + 0.4 * magnitude_loss + 0.2 * profitable_loss;

        (total_loss, direction_loss, magnitude_loss, profitable_loss)
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.senet.parameters_mut());
        params.extend(self.dien.parameters_mut());
        params.extend(self.din.parameters_mut());
        params.extend(self.dcn.parameters_mut());
        params.extend(self.layer_norm.parameters_mut());
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
