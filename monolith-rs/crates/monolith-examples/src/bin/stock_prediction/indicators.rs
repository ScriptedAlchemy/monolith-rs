use serde::{Deserialize, Serialize};
use ta::indicators::{
    AverageTrueRange, BollingerBands, CommodityChannelIndex, ExponentialMovingAverage, Maximum,
    Minimum, MoneyFlowIndex, OnBalanceVolume, RateOfChange, RelativeStrengthIndex,
    SimpleMovingAverage, SlowStochastic, StandardDeviation,
};
use ta::Next;

use super::data::StockBar;

#[cfg(feature = "talib")]
mod talib {
    // Raw FFI bindings to TA-Lib (C). The `ta-lib-sys` crate vendors/links TA-Lib.
    use ta_lib_sys as sys;

    pub fn try_init() -> bool {
        unsafe { sys::Initialize() == sys::RetCode::SUCCESS }
    }

    pub fn shutdown() {
        let _ = unsafe { sys::Shutdown() };
    }

    fn fill_1out(len: usize, out_beg: i32, out_nb: i32, out: &[f64], dst: &mut [f64]) {
        if out_nb <= 0 || out.is_empty() || dst.is_empty() {
            return;
        }
        let start = out_beg.max(0) as usize;
        let count = out_nb.max(0) as usize;
        let end = (start + count).min(len).min(dst.len());
        let copy_len = end.saturating_sub(start).min(out.len());
        if copy_len > 0 {
            dst[start..start + copy_len].copy_from_slice(&out[..copy_len]);
        }
    }

    pub fn adx_14(high: &[f64], low: &[f64], close: &[f64], dst: &mut [f64]) {
        let len = high.len().min(low.len()).min(close.len()).min(dst.len());
        if len == 0 {
            return;
        }
        let mut out_beg: i32 = 0;
        let mut out_nb: i32 = 0;
        let mut out = vec![0.0_f64; len];
        let in_high: Vec<f32> = high[..len].iter().map(|v| *v as f32).collect();
        let in_low: Vec<f32> = low[..len].iter().map(|v| *v as f32).collect();
        let in_close: Vec<f32> = close[..len].iter().map(|v| *v as f32).collect();
        unsafe {
            let _ = sys::S_ADX(
                0,
                (len as i32) - 1,
                in_high.as_ptr(),
                in_low.as_ptr(),
                in_close.as_ptr(),
                14,
                &mut out_beg,
                &mut out_nb,
                out.as_mut_ptr(),
            );
        }
        fill_1out(len, out_beg, out_nb, &out, dst);
    }

    pub fn plus_di_14(high: &[f64], low: &[f64], close: &[f64], dst: &mut [f64]) {
        let len = high.len().min(low.len()).min(close.len()).min(dst.len());
        if len == 0 {
            return;
        }
        let mut out_beg: i32 = 0;
        let mut out_nb: i32 = 0;
        let mut out = vec![0.0_f64; len];
        let in_high: Vec<f32> = high[..len].iter().map(|v| *v as f32).collect();
        let in_low: Vec<f32> = low[..len].iter().map(|v| *v as f32).collect();
        let in_close: Vec<f32> = close[..len].iter().map(|v| *v as f32).collect();
        unsafe {
            let _ = sys::S_PLUS_DI(
                0,
                (len as i32) - 1,
                in_high.as_ptr(),
                in_low.as_ptr(),
                in_close.as_ptr(),
                14,
                &mut out_beg,
                &mut out_nb,
                out.as_mut_ptr(),
            );
        }
        fill_1out(len, out_beg, out_nb, &out, dst);
    }

    pub fn minus_di_14(high: &[f64], low: &[f64], close: &[f64], dst: &mut [f64]) {
        let len = high.len().min(low.len()).min(close.len()).min(dst.len());
        if len == 0 {
            return;
        }
        let mut out_beg: i32 = 0;
        let mut out_nb: i32 = 0;
        let mut out = vec![0.0_f64; len];
        let in_high: Vec<f32> = high[..len].iter().map(|v| *v as f32).collect();
        let in_low: Vec<f32> = low[..len].iter().map(|v| *v as f32).collect();
        let in_close: Vec<f32> = close[..len].iter().map(|v| *v as f32).collect();
        unsafe {
            let _ = sys::S_MINUS_DI(
                0,
                (len as i32) - 1,
                in_high.as_ptr(),
                in_low.as_ptr(),
                in_close.as_ptr(),
                14,
                &mut out_beg,
                &mut out_nb,
                out.as_mut_ptr(),
            );
        }
        fill_1out(len, out_beg, out_nb, &out, dst);
    }

    pub fn aroon_14(high: &[f64], low: &[f64], out_up: &mut [f64], out_down: &mut [f64]) {
        let len = high.len().min(low.len()).min(out_up.len()).min(out_down.len());
        if len == 0 {
            return;
        }
        let mut out_beg: i32 = 0;
        let mut out_nb: i32 = 0;
        let mut up = vec![0.0_f64; len];
        let mut down = vec![0.0_f64; len];
        let in_high: Vec<f32> = high[..len].iter().map(|v| *v as f32).collect();
        let in_low: Vec<f32> = low[..len].iter().map(|v| *v as f32).collect();
        unsafe {
            let _ = sys::S_AROON(
                0,
                (len as i32) - 1,
                in_high.as_ptr(),
                in_low.as_ptr(),
                14,
                &mut out_beg,
                &mut out_nb,
                down.as_mut_ptr(),
                up.as_mut_ptr(),
            );
        }
        fill_1out(len, out_beg, out_nb, &up, out_up);
        fill_1out(len, out_beg, out_nb, &down, out_down);
    }

    pub fn ultosc(high: &[f64], low: &[f64], close: &[f64], dst: &mut [f64]) {
        let len = high.len().min(low.len()).min(close.len()).min(dst.len());
        if len == 0 {
            return;
        }
        let mut out_beg: i32 = 0;
        let mut out_nb: i32 = 0;
        let mut out = vec![0.0_f64; len];
        let in_high: Vec<f32> = high[..len].iter().map(|v| *v as f32).collect();
        let in_low: Vec<f32> = low[..len].iter().map(|v| *v as f32).collect();
        let in_close: Vec<f32> = close[..len].iter().map(|v| *v as f32).collect();
        unsafe {
            let _ = sys::S_ULTOSC(
                0,
                (len as i32) - 1,
                in_high.as_ptr(),
                in_low.as_ptr(),
                in_close.as_ptr(),
                7,
                14,
                28,
                &mut out_beg,
                &mut out_nb,
                out.as_mut_ptr(),
            );
        }
        fill_1out(len, out_beg, out_nb, &out, dst);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TechnicalIndicators {
    // Trend Indicators
    pub sma_5: f32,
    pub sma_20: f32,
    pub sma_50: f32,
    pub ema_12: f32,
    pub ema_26: f32,
    pub sma_ratio_20_50: f32, // SMA 20/50 ratio
    pub ema_ratio_12_26: f32, // EMA 12/26 ratio
    
    // Momentum Indicators
    pub rsi_14: f32,
    pub rsi_7: f32,   // Faster RSI
    pub rsi_21: f32,  // Slower RSI
    pub macd: f32,
    pub macd_signal: f32,
    pub macd_histogram: f32,
    pub roc_10: f32,
    pub roc_20: f32,
    pub mfi_14: f32,
    pub cci_20: f32,  // Commodity Channel Index
    
    // Volatility Indicators
    pub bollinger_upper: f32,
    pub bollinger_lower: f32,
    pub bollinger_width: f32,
    pub bollinger_position: f32, // Position within Bollinger Bands
    pub atr_14: f32,
    pub atr_7: f32,   // Shorter ATR
    pub stddev_20: f32,
    pub stddev_10: f32, // Shorter standard deviation
    
    // Volume Indicators
    pub volume_sma_20: f32,
    pub volume_sma_5: f32,
    pub volume_ratio: f32,
    pub volume_roc: f32, // Volume Rate of Change
    pub obv: f32,
    pub obv_slope: f32, // OBV trend
    pub volume_weighted_price: f32,
    
    // Price Action
    pub stoch_k: f32,
    pub stoch_d: f32,
    pub stoch_rsi: f32, // Stochastic RSI
    pub max_14: f32,
    pub min_14: f32,
    pub price_vs_range: f32,
    
    // Pattern Recognition
    pub price_vs_sma20: f32,
    pub price_vs_sma50: f32,
    pub price_vs_bollinger: f32,
    pub high_low_range: f32,
    pub body_ratio: f32,
    pub upper_shadow: f32,
    pub lower_shadow: f32,
    pub doji_score: f32, // Doji pattern strength
    
    // Market Microstructure
    pub vwap: f32, // Volume Weighted Average Price
    pub vwap_deviation: f32, // Deviation from VWAP
    pub pivot_support: f32,
    pub pivot_resistance: f32,
    pub pivot_point: f32,
    pub pivot_range: f32, // (resistance - support) relative to price
    
    // Advanced Features
    pub trend_strength: f32,
    pub volatility_regime: f32,
    pub volume_profile_high: f32,
    pub volume_profile_low: f32,

    // TA-Lib extras (computed only when the `talib` Cargo feature is enabled)
    pub talib_adx_14: f32,
    pub talib_plus_di_14: f32,
    pub talib_minus_di_14: f32,
    pub talib_aroon_up_14: f32,
    pub talib_aroon_down_14: f32,
    pub talib_ultosc: f32,
}

impl TechnicalIndicators {
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            // Trend Indicators (7)
            self.sma_5, self.sma_20, self.sma_50, self.ema_12, self.ema_26,
            self.sma_ratio_20_50, self.ema_ratio_12_26,

            // Momentum Indicators (10)
            self.rsi_14, self.rsi_7, self.rsi_21, self.macd, self.macd_signal,
            self.macd_histogram, self.roc_10, self.roc_20, self.mfi_14, self.cci_20,
            
            // Volatility Indicators (7)
            self.bollinger_upper, self.bollinger_lower, self.bollinger_width,
            self.bollinger_position, self.atr_14, self.atr_7, self.stddev_20,
            self.stddev_10,
            
            // Volume Indicators (7)
            self.volume_sma_20, self.volume_sma_5, self.volume_ratio, self.volume_roc,
            self.obv, self.obv_slope, self.volume_weighted_price,
            
            // Price Action (8)
            self.stoch_k, self.stoch_d, self.stoch_rsi, self.max_14, self.min_14,
            self.price_vs_range, self.price_vs_sma20, self.price_vs_sma50,
            
            // Pattern Recognition (6)
            self.price_vs_bollinger, self.high_low_range, self.body_ratio,
            self.upper_shadow, self.lower_shadow, self.doji_score,
            
            // Market Microstructure (6)
            self.vwap, self.vwap_deviation, self.pivot_support, self.pivot_resistance,
            self.pivot_point, self.pivot_range,
            
            // Advanced Features (4)
            self.trend_strength, self.volatility_regime, self.volume_profile_high,
            self.volume_profile_low,

            // TA-Lib extras (6)
            self.talib_adx_14,
            self.talib_plus_di_14,
            self.talib_minus_di_14,
            self.talib_aroon_up_14,
            self.talib_aroon_down_14,
            self.talib_ultosc,
        ]
    }

    pub const NUM_FEATURES: usize = 62; // Updated feature count
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

        // Trend indicators
        let mut sma_5 = SimpleMovingAverage::new(5).unwrap();
        let mut sma_20 = SimpleMovingAverage::new(20).unwrap();
        let mut sma_50 = SimpleMovingAverage::new(50).unwrap();
        let mut ema_12 = ExponentialMovingAverage::new(12).unwrap();
        let mut ema_26 = ExponentialMovingAverage::new(26).unwrap();
        let mut ema_9 = ExponentialMovingAverage::new(9).unwrap();

        // Momentum indicators
        let mut rsi_14 = RelativeStrengthIndex::new(14).unwrap();
        let mut rsi_7 = RelativeStrengthIndex::new(7).unwrap();
        let mut rsi_21 = RelativeStrengthIndex::new(21).unwrap();
        let mut roc_10 = RateOfChange::new(10).unwrap();
        let mut roc_20 = RateOfChange::new(20).unwrap();
        let mut stoch_k = SlowStochastic::new(14, 3).unwrap();
        let mut stoch_d_sma = SimpleMovingAverage::new(3).unwrap();
        let mut cci = CommodityChannelIndex::new(20).unwrap();

        // Volatility indicators
        let mut bb = BollingerBands::new(20, 2.0_f64).unwrap();
        let mut atr_14 = AverageTrueRange::new(14).unwrap();
        let mut atr_7 = AverageTrueRange::new(7).unwrap();
        let mut stddev_20 = StandardDeviation::new(20).unwrap();
        let mut stddev_10 = StandardDeviation::new(10).unwrap();

        // Volume indicators
        let mut volume_sma_20 = SimpleMovingAverage::new(20).unwrap();
        let mut volume_sma_5 = SimpleMovingAverage::new(5).unwrap();
        let mut obv = OnBalanceVolume::new();
        let mut mfi = MoneyFlowIndex::new(14).unwrap();

        // Price action indicators
        let mut max_14 = Maximum::new(14).unwrap();
        let mut min_14 = Minimum::new(14).unwrap();

        // Initialize value arrays for all indicators
        let mut sma_5_vals = vec![0.0_f64; n];
        let mut sma_20_vals = vec![0.0_f64; n];
        let mut sma_50_vals = vec![0.0_f64; n];
        let mut ema_12_vals = vec![0.0_f64; n];
        let mut ema_26_vals = vec![0.0_f64; n];
        let mut rsi_14_vals = vec![50.0_f64; n];
        let mut rsi_7_vals = vec![50.0_f64; n];
        let mut rsi_21_vals = vec![50.0_f64; n];
        let mut roc_10_vals = vec![0.0_f64; n];
        let mut roc_20_vals = vec![0.0_f64; n];
        let mut stoch_k_vals = vec![50.0_f64; n];
        let mut stoch_d_vals = vec![50.0_f64; n];
        let mut cci_vals = vec![0.0_f64; n];
        let mut bb_upper = vec![0.0_f64; n];
        let mut bb_lower = vec![0.0_f64; n];
        let mut atr_14_vals = vec![0.0_f64; n];
        let mut atr_7_vals = vec![0.0_f64; n];
        let mut stddev_20_vals = vec![0.0_f64; n];
        let mut stddev_10_vals = vec![0.0_f64; n];
        let mut vol_sma_20_vals = vec![0.0_f64; n];
        let mut vol_sma_5_vals = vec![0.0_f64; n];
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

            // Trend indicators
            sma_5_vals[i] = sma_5.next(close);
            sma_20_vals[i] = sma_20.next(close);
            sma_50_vals[i] = sma_50.next(close);
            ema_12_vals[i] = ema_12.next(close);
            ema_26_vals[i] = ema_26.next(close);

            // Momentum indicators
            rsi_14_vals[i] = rsi_14.next(close);
            rsi_7_vals[i] = rsi_7.next(close);
            rsi_21_vals[i] = rsi_21.next(close);
            roc_10_vals[i] = roc_10.next(close);
            roc_20_vals[i] = roc_20.next(close);
            cci_vals[i] = cci.next(&data_item);

            // Stochastic
            let k_val = stoch_k.next(close);
            stoch_k_vals[i] = k_val;
            stoch_d_vals[i] = stoch_d_sma.next(k_val);

            // Volatility indicators
            let bb_val = bb.next(close);
            bb_upper[i] = bb_val.upper;
            bb_lower[i] = bb_val.lower;
            atr_14_vals[i] = atr_14.next(&data_item);
            atr_7_vals[i] = atr_7.next(&data_item);
            stddev_20_vals[i] = stddev_20.next(close);
            stddev_10_vals[i] = stddev_10.next(close);

            // Volume indicators
            vol_sma_20_vals[i] = volume_sma_20.next(volume);
            vol_sma_5_vals[i] = volume_sma_5.next(volume);
            obv_vals[i] = obv.next(&data_item);
            mfi_vals[i] = mfi.next(&data_item);

            // Price action indicators
            max_14_vals[i] = max_14.next(high);
            min_14_vals[i] = min_14.next(low);
        }

        // TA-Lib extras (batch computed; useful for "advanced" features without having to
        // hand-implement each indicator). These remain zeros unless the `talib` feature is enabled.
        let mut talib_adx_vals = vec![0.0_f64; n];
        let mut talib_plus_di_vals = vec![0.0_f64; n];
        let mut talib_minus_di_vals = vec![0.0_f64; n];
        let mut talib_aroon_up_vals = vec![0.0_f64; n];
        let mut talib_aroon_down_vals = vec![0.0_f64; n];
        let mut talib_ultosc_vals = vec![0.0_f64; n];

        #[cfg(feature = "talib")]
        {
            if talib::try_init() {
                let highs: Vec<f64> = bars.iter().map(|b| b.high as f64).collect();
                let lows: Vec<f64> = bars.iter().map(|b| b.low as f64).collect();
                let closes: Vec<f64> = bars.iter().map(|b| b.close as f64).collect();

                talib::adx_14(&highs, &lows, &closes, &mut talib_adx_vals);
                talib::plus_di_14(&highs, &lows, &closes, &mut talib_plus_di_vals);
                talib::minus_di_14(&highs, &lows, &closes, &mut talib_minus_di_vals);
                talib::aroon_14(
                    &highs,
                    &lows,
                    &mut talib_aroon_up_vals,
                    &mut talib_aroon_down_vals,
                );
                talib::ultosc(&highs, &lows, &closes, &mut talib_ultosc_vals);

                talib::shutdown();
            }
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

            let _macd_histogram = macd_vals[i] - macd_signal_vals[i];
            let _bb_width = if sma_20_vals[i] > 0.0 {
                (bb_upper[i] - bb_lower[i]) / sma_20_vals[i]
            } else {
                0.0
            };

            let _vol_ratio = if vol_sma_20_vals[i] > 0.0 {
                bars[i].volume / vol_sma_20_vals[i]
            } else {
                1.0
            };

            let price_range = max_14_vals[i] - min_14_vals[i];
            let _price_vs_range = if price_range > 0.0 {
                ((close - min_14_vals[i]) / price_range - 0.5) * 2.0
            } else {
                0.0
            };

            // Compute advanced features
            let macd_histogram = macd_vals[i] - macd_signal_vals[i];
            let bb_width = if sma_20_vals[i] > 0.0 {
                (bb_upper[i] - bb_lower[i]) / sma_20_vals[i]
            } else {
                0.0
            };

            let vol_ratio = if vol_sma_20_vals[i] > 0.0 {
                bars[i].volume / vol_sma_20_vals[i]
            } else {
                1.0
            };

            let price_range = max_14_vals[i] - min_14_vals[i];
            let price_vs_range = if price_range > 0.0 {
                ((close - min_14_vals[i]) / price_range - 0.5) * 2.0
            } else {
                0.0
            };

            // Bollinger position (-1 to 1, where 0 is middle)
            let bollinger_position = if bb_upper[i] > bb_lower[i] {
                (((close - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) - 0.5) * 2.0) as f32
            } else {
                0.0
            };

            // Volume rate of change (10-period)
            let volume_roc = if i >= 10 && vol_sma_20_vals[i-10] > 0.0 {
                (vol_sma_20_vals[i] / vol_sma_20_vals[i-10] - 1.0) as f32
            } else {
                0.0
            };

            // OBV slope (trend)
            let obv_slope = if i >= 5 && obv_vals[i-5] != 0.0 {
                ((obv_vals[i] - obv_vals[i-5]) / obv_vals[i-5].abs()) as f32
            } else {
                0.0
            };

            // Stochastic RSI
            let stoch_rsi = if rsi_14_vals[i] != 50.0 {
                ((stoch_k_vals[i] * rsi_14_vals[i] / 100.0 - 50.0) / 50.0) as f32
            } else {
                0.0
            };

            // Doji score (how close open/close are)
            let doji_score = if high > low {
                1.0 - ((close - open).abs() / (high - low)) as f32
            } else {
                0.0
            };

            // VWAP approximation (cumulative)
            let vwap = if i > 0 {
                let cum_vol: f64 = bars[..=i].iter().map(|b| b.volume).sum();
                let cum_price_vol: f64 = bars[..=i].iter().enumerate()
                    .map(|(j, b)| (bars[j].high + bars[j].low + bars[j].close) as f64 / 3.0 * b.volume)
                    .sum();
                if cum_vol > 0.0 {
                    (cum_price_vol / cum_vol / close - 1.0) as f32
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Pivot points
            let pivot_point = if i >= 14 {
                let prev_high = max_14_vals[i - 1];
                let prev_low = min_14_vals[i - 1];
                let prev_close = bars[i-1].close as f64;
                ((prev_high + prev_low + prev_close) / 3.0 / close - 1.0) as f32
            } else {
                0.0
            };

            let pivot_support = if i >= 14 {
                let prev_high = max_14_vals[i - 1];
                let prev_low = min_14_vals[i - 1];
                ((prev_low * 2.0 + prev_high) / 3.0 / close - 1.0) as f32
            } else {
                0.0
            };

            let pivot_resistance = if i >= 14 {
                let prev_high = max_14_vals[i - 1];
                let prev_low = min_14_vals[i - 1];
                ((prev_high * 2.0 + prev_low) / 3.0 / close - 1.0) as f32
            } else {
                0.0
            };

            let pivot_range = (pivot_resistance - pivot_support).clamp(-1.0, 1.0);

            // Trend strength (SMA alignment)
            let trend_strength = if sma_20_vals[i] > 0.0 && sma_50_vals[i] > 0.0 {
                ((sma_20_vals[i] / sma_50_vals[i] - 1.0) * 2.0).clamp(-1.0, 1.0) as f32
            } else {
                0.0
            };

            // Volatility regime (high ATR = high volatility)
            let volatility_regime = if atr_14_vals[i] > 0.0 {
                ((atr_14_vals[i] / close - 0.02) / 0.04).clamp(-1.0, 1.0) as f32
            } else {
                0.0
            };

            indicators[i] = TechnicalIndicators {
                // Trend Indicators
                sma_5: (sma_5_vals[i] / close - 1.0) as f32,
                sma_20: (sma_20_vals[i] / close - 1.0) as f32,
                sma_50: (sma_50_vals[i] / close - 1.0) as f32,
                ema_12: (ema_12_vals[i] / close - 1.0) as f32,
                ema_26: (ema_26_vals[i] / close - 1.0) as f32,
                sma_ratio_20_50: if sma_50_vals[i] > 0.0 {
                    ((sma_20_vals[i] / sma_50_vals[i] - 1.0) * 2.0).clamp(-1.0, 1.0) as f32
                } else {
                    0.0
                },
                ema_ratio_12_26: if ema_26_vals[i] > 0.0 {
                    ((ema_12_vals[i] / ema_26_vals[i] - 1.0) * 2.0).clamp(-1.0, 1.0) as f32
                } else {
                    0.0
                },

                // Momentum Indicators
                rsi_14: ((rsi_14_vals[i] - 50.0) / 50.0) as f32,
                rsi_7: ((rsi_7_vals[i] - 50.0) / 50.0) as f32,
                rsi_21: ((rsi_21_vals[i] - 50.0) / 50.0) as f32,
                macd: (macd_vals[i] / close * 100.0) as f32,
                macd_signal: (macd_signal_vals[i] / close * 100.0) as f32,
                macd_histogram: (macd_histogram / close * 100.0) as f32,
                roc_10: (roc_10_vals[i] / 100.0).clamp(-1.0, 1.0) as f32,
                roc_20: (roc_20_vals[i] / 100.0).clamp(-1.0, 1.0) as f32,
                mfi_14: ((mfi_vals[i] - 50.0) / 50.0) as f32,
                cci_20: (cci_vals[i] / 100.0).clamp(-1.0, 1.0) as f32,

                // Volatility Indicators
                bollinger_upper: (bb_upper[i] / close - 1.0) as f32,
                bollinger_lower: (bb_lower[i] / close - 1.0) as f32,
                bollinger_width: bb_width as f32,
                bollinger_position,
                atr_14: (atr_14_vals[i] / close) as f32,
                atr_7: (atr_7_vals[i] / close) as f32,
                stddev_20: (stddev_20_vals[i] / close) as f32,
                stddev_10: (stddev_10_vals[i] / close) as f32,

                // Volume Indicators
                volume_sma_20: (vol_sma_20_vals[i] / 1_000_000.0) as f32,
                volume_sma_5: (vol_sma_5_vals[i] / 1_000_000.0) as f32,
                volume_ratio: (vol_ratio.min(5.0) / 5.0 - 0.5) as f32,
                volume_roc,
                obv: (obv_vals[i] / 1e9) as f32,
                obv_slope,
                volume_weighted_price: (close * bars[i].volume as f64 / 1e6) as f32,

                // Price Action
                stoch_k: ((stoch_k_vals[i] - 50.0) / 50.0) as f32,
                stoch_d: ((stoch_d_vals[i] - 50.0) / 50.0) as f32,
                stoch_rsi,
                max_14: (max_14_vals[i] / close - 1.0) as f32,
                min_14: (min_14_vals[i] / close - 1.0) as f32,
                price_vs_range: price_vs_range as f32,

                // Pattern Recognition
                price_vs_sma20: ((close / sma_20_vals[i].max(0.01) - 1.0).clamp(-0.5, 0.5)) as f32,
                price_vs_sma50: ((close / sma_50_vals[i].max(0.01) - 1.0).clamp(-0.5, 0.5)) as f32,
                price_vs_bollinger: bollinger_position,
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
                doji_score,

                // Market Microstructure
                vwap,
                vwap_deviation: ((close as f64 - vwap as f64 * close as f64) / close as f64) as f32,
                pivot_support,
                pivot_resistance,
                pivot_point,
                pivot_range,

                // Advanced Features
                trend_strength,
                volatility_regime,
                volume_profile_high: (bb_upper[i] / close - 1.0) as f32,
                volume_profile_low: (bb_lower[i] / close - 1.0) as f32,

                // TA-Lib extras (scaled into roughly [-1, 1] / [0, 1] ranges)
                talib_adx_14: (talib_adx_vals[i] / 100.0).clamp(0.0, 1.0) as f32,
                talib_plus_di_14: (talib_plus_di_vals[i] / 100.0).clamp(0.0, 1.0) as f32,
                talib_minus_di_14: (talib_minus_di_vals[i] / 100.0).clamp(0.0, 1.0) as f32,
                talib_aroon_up_14: (talib_aroon_up_vals[i] / 100.0).clamp(0.0, 1.0) as f32,
                talib_aroon_down_14: (talib_aroon_down_vals[i] / 100.0).clamp(0.0, 1.0) as f32,
                talib_ultosc: ((talib_ultosc_vals[i] - 50.0) / 50.0).clamp(-1.0, 1.0) as f32,
            };
        }

        indicators
    }
}
