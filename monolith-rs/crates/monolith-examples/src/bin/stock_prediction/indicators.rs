use ta::indicators::{
    AverageTrueRange, BollingerBands, ExponentialMovingAverage, Maximum, Minimum, MoneyFlowIndex,
    OnBalanceVolume, RateOfChange, RelativeStrengthIndex, SimpleMovingAverage, SlowStochastic,
    StandardDeviation,
};
use ta::Next;

use super::data::StockBar;

#[derive(Debug, Clone, Default)]
pub struct TechnicalIndicators {
    pub sma_5: f32,
    pub sma_20: f32,
    pub sma_50: f32,
    pub ema_12: f32,
    pub ema_26: f32,

    pub rsi_14: f32,
    pub macd: f32,
    pub macd_signal: f32,
    pub macd_histogram: f32,
    pub roc_10: f32,
    pub roc_20: f32,

    pub stoch_k: f32,
    pub stoch_d: f32,

    pub bollinger_upper: f32,
    pub bollinger_lower: f32,
    pub bollinger_width: f32,
    pub atr_14: f32,
    pub stddev_20: f32,

    pub volume_sma_20: f32,
    pub volume_ratio: f32,
    pub obv: f32,
    pub mfi_14: f32,

    pub max_14: f32,
    pub min_14: f32,
    pub price_vs_range: f32,

    pub price_vs_sma20: f32,
    pub price_vs_bollinger: f32,
    pub high_low_range: f32,
    pub body_ratio: f32,
    pub upper_shadow: f32,
    pub lower_shadow: f32,
}

impl TechnicalIndicators {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.sma_5,
            self.sma_20,
            self.sma_50,
            self.ema_12,
            self.ema_26,
            self.rsi_14,
            self.macd,
            self.macd_signal,
            self.macd_histogram,
            self.roc_10,
            self.roc_20,
            self.stoch_k,
            self.stoch_d,
            self.bollinger_upper,
            self.bollinger_lower,
            self.bollinger_width,
            self.atr_14,
            self.stddev_20,
            self.volume_sma_20,
            self.volume_ratio,
            self.obv,
            self.mfi_14,
            self.max_14,
            self.min_14,
            self.price_vs_range,
            self.price_vs_sma20,
            self.price_vs_bollinger,
            self.high_low_range,
            self.body_ratio,
            self.upper_shadow,
            self.lower_shadow,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    pub const NUM_FEATURES: usize = 36;
}

pub struct IndicatorCalculator;

impl IndicatorCalculator {
    pub fn new() -> Self {
        Self {}
    }

    pub fn compute_indicators(&mut self, bars: &[StockBar]) -> Vec<TechnicalIndicators> {
        if bars.is_empty() {
            return Vec::new();
        }

        let n = bars.len();
        let mut indicators = vec![TechnicalIndicators::default(); n];

        let mut sma_5 = SimpleMovingAverage::new(5).unwrap();
        let mut sma_20 = SimpleMovingAverage::new(20).unwrap();
        let mut sma_50 = SimpleMovingAverage::new(50).unwrap();
        let mut ema_12 = ExponentialMovingAverage::new(12).unwrap();
        let mut ema_26 = ExponentialMovingAverage::new(26).unwrap();
        let mut ema_9 = ExponentialMovingAverage::new(9).unwrap();

        let mut rsi = RelativeStrengthIndex::new(14).unwrap();
        let mut roc_10 = RateOfChange::new(10).unwrap();
        let mut roc_20 = RateOfChange::new(20).unwrap();
        let mut stoch_k = SlowStochastic::new(14, 3).unwrap();
        let mut stoch_d_sma = SimpleMovingAverage::new(3).unwrap();

        let mut bb = BollingerBands::new(20, 2.0_f64).unwrap();
        let mut atr = AverageTrueRange::new(14).unwrap();
        let mut stddev = StandardDeviation::new(20).unwrap();

        let mut volume_sma = SimpleMovingAverage::new(20).unwrap();
        let mut obv = OnBalanceVolume::new();
        let mut mfi = MoneyFlowIndex::new(14).unwrap();

        let mut max_14 = Maximum::new(14).unwrap();
        let mut min_14 = Minimum::new(14).unwrap();

        let mut sma_5_vals = vec![0.0_f64; n];
        let mut sma_20_vals = vec![0.0_f64; n];
        let mut sma_50_vals = vec![0.0_f64; n];
        let mut ema_12_vals = vec![0.0_f64; n];
        let mut ema_26_vals = vec![0.0_f64; n];
        let mut rsi_vals = vec![50.0_f64; n];
        let mut roc_10_vals = vec![0.0_f64; n];
        let mut roc_20_vals = vec![0.0_f64; n];
        let mut stoch_k_vals = vec![50.0_f64; n];
        let mut stoch_d_vals = vec![50.0_f64; n];
        let mut bb_upper = vec![0.0_f64; n];
        let mut bb_lower = vec![0.0_f64; n];
        let mut atr_vals = vec![0.0_f64; n];
        let mut stddev_vals = vec![0.0_f64; n];
        let mut vol_sma_vals = vec![0.0_f64; n];
        let mut obv_vals = vec![0.0_f64; n];
        let mut mfi_vals = vec![50.0_f64; n];
        let mut max_14_vals = vec![0.0_f64; n];
        let mut min_14_vals = vec![0.0_f64; n];

        for (i, bar) in bars.iter().enumerate() {
            let close = bar.close as f64;
            let high = bar.high as f64;
            let low = bar.low as f64;
            let open = bar.open as f64;
            let volume = bar.volume;

            let data_item = ta::DataItem::builder()
                .high(high)
                .low(low)
                .close(close)
                .open(open)
                .volume(volume)
                .build()
                .unwrap();

            sma_5_vals[i] = sma_5.next(close);
            sma_20_vals[i] = sma_20.next(close);
            sma_50_vals[i] = sma_50.next(close);
            ema_12_vals[i] = ema_12.next(close);
            ema_26_vals[i] = ema_26.next(close);

            rsi_vals[i] = rsi.next(close);
            roc_10_vals[i] = roc_10.next(close);
            roc_20_vals[i] = roc_20.next(close);

            let k_val = stoch_k.next(close);
            stoch_k_vals[i] = k_val;
            stoch_d_vals[i] = stoch_d_sma.next(k_val);

            let bb_val = bb.next(close);
            bb_upper[i] = bb_val.upper;
            bb_lower[i] = bb_val.lower;

            atr_vals[i] = atr.next(&data_item);
            stddev_vals[i] = stddev.next(close);
            vol_sma_vals[i] = volume_sma.next(volume);
            obv_vals[i] = obv.next(&data_item);
            mfi_vals[i] = mfi.next(&data_item);

            max_14_vals[i] = max_14.next(high);
            min_14_vals[i] = min_14.next(low);
        }

        let mut macd_vals = vec![0.0_f64; n];
        let mut macd_signal_vals = vec![0.0_f64; n];

        for i in 0..n {
            macd_vals[i] = ema_12_vals[i] - ema_26_vals[i];
            macd_signal_vals[i] = ema_9.next(macd_vals[i]);
        }

        for i in 0..n {
            let close = bars[i].close as f64;
            let high = bars[i].high as f64;
            let low = bars[i].low as f64;
            let open = bars[i].open as f64;

            let macd_histogram = macd_vals[i] - macd_signal_vals[i];
            let bb_width = if sma_20_vals[i] > 0.0 {
                (bb_upper[i] - bb_lower[i]) / sma_20_vals[i]
            } else {
                0.0
            };

            let vol_ratio = if vol_sma_vals[i] > 0.0 {
                bars[i].volume / vol_sma_vals[i]
            } else {
                1.0
            };

            let price_range = max_14_vals[i] - min_14_vals[i];
            let price_vs_range = if price_range > 0.0 {
                ((close - min_14_vals[i]) / price_range - 0.5) * 2.0
            } else {
                0.0
            };

            indicators[i] = TechnicalIndicators {
                sma_5: (sma_5_vals[i] / close - 1.0) as f32,
                sma_20: (sma_20_vals[i] / close - 1.0) as f32,
                sma_50: (sma_50_vals[i] / close - 1.0) as f32,
                ema_12: (ema_12_vals[i] / close - 1.0) as f32,
                ema_26: (ema_26_vals[i] / close - 1.0) as f32,

                rsi_14: ((rsi_vals[i] - 50.0) / 50.0) as f32,
                macd: (macd_vals[i] / close * 100.0) as f32,
                macd_signal: (macd_signal_vals[i] / close * 100.0) as f32,
                macd_histogram: (macd_histogram / close * 100.0) as f32,
                roc_10: (roc_10_vals[i] / 100.0).clamp(-1.0, 1.0) as f32,
                roc_20: (roc_20_vals[i] / 100.0).clamp(-1.0, 1.0) as f32,

                stoch_k: ((stoch_k_vals[i] - 50.0) / 50.0) as f32,
                stoch_d: ((stoch_d_vals[i] - 50.0) / 50.0) as f32,

                bollinger_upper: (bb_upper[i] / close - 1.0) as f32,
                bollinger_lower: (bb_lower[i] / close - 1.0) as f32,
                bollinger_width: bb_width as f32,
                atr_14: (atr_vals[i] / close) as f32,
                stddev_20: (stddev_vals[i] / close) as f32,

                volume_sma_20: (vol_sma_vals[i] / 1_000_000.0) as f32,
                volume_ratio: (vol_ratio.min(5.0) / 5.0 - 0.5) as f32,
                obv: (obv_vals[i] / 1e9) as f32,
                mfi_14: ((mfi_vals[i] - 50.0) / 50.0) as f32,

                max_14: (max_14_vals[i] / close - 1.0) as f32,
                min_14: (min_14_vals[i] / close - 1.0) as f32,
                price_vs_range: price_vs_range as f32,

                price_vs_sma20: ((close / sma_20_vals[i].max(0.01) - 1.0).clamp(-0.5, 0.5)) as f32,
                price_vs_bollinger: if bb_upper[i] > bb_lower[i] {
                    (((close - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) - 0.5) * 2.0) as f32
                } else {
                    0.0
                },
                high_low_range: ((high - low) / close) as f32,
                body_ratio: if high > low {
                    ((close - open).abs() / (high - low)) as f32
                } else {
                    0.0
                },
                upper_shadow: if high > low {
                    ((high - close.max(open)) / (high - low)) as f32
                } else {
                    0.0
                },
                lower_shadow: if high > low {
                    ((close.min(open) - low) / (high - low)) as f32
                } else {
                    0.0
                },
            };
        }

        indicators
    }
}
