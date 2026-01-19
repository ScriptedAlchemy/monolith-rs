use rayon::prelude::*;

use super::data::{StockBar, TickerInfo};
use super::indicators::TechnicalIndicators;

#[derive(Debug, Clone)]
pub struct StockInstance {
    pub ticker_idx: usize,
    pub ticker_fid: i64,
    pub sector_fid: i64,
    pub t: usize,
    pub direction_label: f32,
    pub magnitude_label: f32,
    pub profitable_label: f32,
}

pub struct FeatureIndex {
    pub tickers: Vec<TickerInfo>,
    pub bars_by_ticker: Vec<Vec<StockBar>>,
    // [ticker][timeframe_idx][indicator_idx]
    pub indicators_by_ticker: Vec<Vec<Vec<TechnicalIndicators>>>,
    pub timeframes: Vec<usize>,
}

impl FeatureIndex {
    pub fn new(
        tickers: Vec<TickerInfo>,
        bars_by_ticker: Vec<Vec<StockBar>>,
        indicators_by_ticker: Vec<Vec<Vec<TechnicalIndicators>>>,
        timeframes: Vec<usize>,
    ) -> Self {
        Self {
            tickers,
            bars_by_ticker,
            indicators_by_ticker,
            timeframes,
        }
    }

    pub fn indicator_dim(&self) -> usize {
        TechnicalIndicators::NUM_FEATURES * self.timeframes.len()
    }

    pub fn bar_at(&self, inst: &StockInstance, t: usize) -> &StockBar {
        &self.bars_by_ticker[inst.ticker_idx][t]
    }

    pub fn indicator_at(
        &self,
        inst: &StockInstance,
        t: usize,
        timeframe_idx: usize,
    ) -> Option<&TechnicalIndicators> {
        let tf = self.timeframes[timeframe_idx].max(1);
        let series = &self.indicators_by_ticker[inst.ticker_idx][timeframe_idx];
        indicator_index_for_timeframe(t, tf, series.len()).map(|idx| &series[idx])
    }

    pub fn write_indicator_features(&self, inst: &StockInstance, out: &mut [f32]) {
        let mut offset = 0;
        for tf_idx in 0..self.timeframes.len() {
            if let Some(indicators) = self.indicator_at(inst, inst.t, tf_idx) {
                let vals = indicators.to_vec();
                let end = offset + vals.len();
                if end <= out.len() {
                    out[offset..end].copy_from_slice(&vals);
                }
                offset = end;
            } else {
                offset += TechnicalIndicators::NUM_FEATURES;
            }
        }
    }

    pub fn write_sequence_features(&self, inst: &StockInstance, lookback: usize, out: &mut [f32]) {
        let seq_feature_dim = 4 + self.indicator_dim();
        let needed = lookback * seq_feature_dim;
        if out.len() < needed {
            return;
        }

        let start = inst.t + 1 - lookback;
        for i in 0..lookback {
            let t = start + i;
            let bar = self.bar_at(inst, t);
            let base = i * seq_feature_dim;
            let close = bar.close.max(1e-6);
            out[base] = bar.open / close - 1.0;
            out[base + 1] = bar.high / close - 1.0;
            out[base + 2] = bar.low / close - 1.0;
            out[base + 3] = 1.0;
            let mut offset = base + 4;
            for tf_idx in 0..self.timeframes.len() {
                if let Some(indicators) = self.indicator_at(inst, t, tf_idx) {
                    let ind_vals = indicators.to_vec();
                    for (j, v) in ind_vals.iter().enumerate() {
                        out[offset + j] = *v;
                    }
                }
                offset += TechnicalIndicators::NUM_FEATURES;
            }
        }
    }
}

fn indicator_index_for_timeframe(t: usize, timeframe: usize, series_len: usize) -> Option<usize> {
    if series_len == 0 {
        return None;
    }
    if t + 1 < timeframe {
        return None;
    }
    let completed = (t + 1) / timeframe;
    let idx = completed.saturating_sub(1);
    Some(idx.min(series_len.saturating_sub(1)))
}

pub struct InstanceCreator {
    lookback_window: usize,
    forward_horizon: usize,
    profit_threshold: f32,
}

impl InstanceCreator {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            lookback_window,
            forward_horizon: 5,
            profit_threshold: 0.02,
        }
    }

    pub fn create_instances(
        &self,
        ticker_idx: usize,
        ticker: &TickerInfo,
        bars: &[StockBar],
        indicators: &[TechnicalIndicators],
    ) -> Vec<StockInstance> {
        let n = bars.len().min(indicators.len());
        let mut instances = Vec::new();

        if n <= self.lookback_window + self.forward_horizon {
            return instances;
        }

        for i in self.lookback_window..(n - self.forward_horizon) {
            let future_price = bars[i + self.forward_horizon].close;
            let current_price = bars[i].close;
            let forward_return = (future_price - current_price) / current_price;

            let direction_label = if forward_return > 0.0 { 1.0 } else { 0.0 };
            let magnitude_label = forward_return * 100.0;
            let profitable_label = if forward_return > self.profit_threshold {
                1.0
            } else {
                0.0
            };

            instances.push(StockInstance {
                ticker_idx,
                ticker_fid: ticker.ticker_id,
                sector_fid: ticker.sector.id(),
                t: i,
                direction_label,
                magnitude_label,
                profitable_label,
            });
        }

        instances
    }
}

pub fn train_eval_split(
    instances: &[StockInstance],
    train_ratio: f32,
) -> (Vec<StockInstance>, Vec<StockInstance>) {
    let split_idx = (instances.len() as f32 * train_ratio) as usize;
    let train = instances[..split_idx].to_vec();
    let eval = instances[split_idx..].to_vec();
    (train, eval)
}

pub fn create_instances_parallel(
    feature_index: &FeatureIndex,
    creator: &InstanceCreator,
) -> Vec<StockInstance> {
    let instances_by_ticker: Vec<Vec<StockInstance>> = feature_index
        .tickers
        .par_iter()
        .enumerate()
        .map(|(idx, ticker)| {
            let bars = &feature_index.bars_by_ticker[idx];
            let indicators = &feature_index.indicators_by_ticker[idx][0];
            creator.create_instances(idx, ticker, bars, indicators)
        })
        .collect();

    instances_by_ticker.into_iter().flatten().collect()
}

pub fn create_batches(instances: &[StockInstance], batch_size: usize) -> Vec<Vec<&StockInstance>> {
    instances
        .chunks(batch_size)
        .map(|c| c.iter().collect())
        .collect()
}
