use rayon::prelude::*;
use std::collections::HashMap;

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
                    for (dst, v) in out[offset..end].iter_mut().zip(vals.iter()) {
                        *dst = if v.is_finite() { *v } else { 0.0 };
                    }
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
            let open = if bar.open.is_finite() { bar.open } else { 0.0 };
            let high = if bar.high.is_finite() { bar.high } else { 0.0 };
            let low = if bar.low.is_finite() { bar.low } else { 0.0 };
            out[base] = (open / close - 1.0).clamp(-10.0, 10.0);
            out[base + 1] = (high / close - 1.0).clamp(-10.0, 10.0);
            out[base + 2] = (low / close - 1.0).clamp(-10.0, 10.0);
            out[base + 3] = 1.0;
            let mut offset = base + 4;
            for tf_idx in 0..self.timeframes.len() {
                if let Some(indicators) = self.indicator_at(inst, t, tf_idx) {
                    let ind_vals = indicators.to_vec();
                    for (j, v) in ind_vals.iter().enumerate() {
                        out[offset + j] = if v.is_finite() { *v } else { 0.0 };
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
    profit_atr_mult: f32,
    magnitude_clip: f32,
}

impl InstanceCreator {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            lookback_window,
            forward_horizon: 5,
            profit_threshold: 0.02,
            // For intraday bars, a fixed threshold is often wrong; ATR-scaled threshold adapts.
            // Profit threshold used is max(profit_threshold, profit_atr_mult * atr_14).
            profit_atr_mult: 1.0,
            // Clip extreme forward returns to reduce label noise/outliers dominating the loss.
            magnitude_clip: 0.20,
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
            if !current_price.is_finite() || current_price.abs() < 1e-9 {
                continue;
            }
            if !future_price.is_finite() {
                continue;
            }
            let forward_return = (future_price - current_price) / current_price;
            if !forward_return.is_finite() {
                continue;
            }

            let direction_label = if forward_return > 0.0 { 1.0 } else { 0.0 };
            let clipped_return = forward_return.clamp(-self.magnitude_clip, self.magnitude_clip);
            // Keep magnitude in "percent-ish" units but avoid huge outliers.
            let magnitude_label = clipped_return * 100.0;

            // Volatility-aware profitable threshold (ratio units)
            let atr_rel = indicators[i].atr_14.abs().clamp(0.0, 1.0);
            let dyn_threshold = self
                .profit_threshold
                .max(self.profit_atr_mult * atr_rel);

            let profitable_label = if forward_return > dyn_threshold {
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

/// Time-based split per ticker: for each ticker we keep the earliest `train_ratio` portion
/// in train, and the latest portion in eval. This avoids evaluating on entirely different
/// tickers (which can happen if you split the flattened list).
pub fn train_eval_split_time_by_ticker(
    instances: &[StockInstance],
    train_ratio: f32,
) -> (Vec<StockInstance>, Vec<StockInstance>) {
    let mut by_ticker: HashMap<usize, Vec<&StockInstance>> = HashMap::new();
    for inst in instances {
        by_ticker.entry(inst.ticker_idx).or_default().push(inst);
    }

    let mut train = Vec::new();
    let mut eval = Vec::new();

    for (_ticker, mut group) in by_ticker {
        group.sort_by_key(|i| i.t);
        let n = group.len();
        if n < 2 {
            continue;
        }
        let mut split_idx = (n as f32 * train_ratio) as usize;
        split_idx = split_idx.clamp(1, n - 1);

        train.extend(group[..split_idx].iter().map(|i| (*i).clone()));
        eval.extend(group[split_idx..].iter().map(|i| (*i).clone()));
    }

    // Deterministic ordering for reproducibility
    train.sort_by_key(|i| (i.ticker_idx, i.t));
    eval.sort_by_key(|i| (i.ticker_idx, i.t));
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
