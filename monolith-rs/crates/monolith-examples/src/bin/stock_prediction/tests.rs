use super::config::{Mode, StockPredictorConfig};
use super::data::{RandomGenerator, Sector, StockBar, TickerInfo};
use super::indicators::{IndicatorCalculator, TechnicalIndicators};
use super::instances::{FeatureIndex, InstanceCreator};
use super::model::{EmbeddingTables, StockPredictionModel};

/// Create minimal test bars for a single ticker.
fn create_test_bars(ticker_id: i64, num_bars: usize) -> Vec<StockBar> {
    let mut bars = Vec::with_capacity(num_bars);
    let mut price = 100.0_f32;
    let mut prev_close = price;

    for i in 0..num_bars {
        let change = ((i % 7) as f32 - 3.0) * 0.01;
        price *= 1.0 + change;
        let returns = if i == 0 { 0.0 } else { (price - prev_close) / prev_close };
        prev_close = price;

        bars.push(StockBar {
            ticker_id,
            timestamp: i as i64,
            day_index: i,
            open: price * 0.999,
            high: price * 1.01,
            low: price * 0.99,
            close: price,
            volume: 1_000_000.0 + (i as f64 * 10_000.0),
            returns,
        });
    }
    bars
}

/// Create a test ticker.
fn create_test_ticker(id: i64, symbol: &str) -> TickerInfo {
    TickerInfo {
        ticker_id: id,
        name: format!("{} Inc.", symbol),
        symbol: symbol.to_string(),
        sector: Sector::Technology,
        beta: 1.0,
        base_volatility: 0.02,
        drift: 0.0001,
    }
}

#[test]
fn test_config_default() {
    let config = StockPredictorConfig::default();
    assert_eq!(config.num_tickers, 50);
    assert_eq!(config.days_of_history, 252);
    assert_eq!(config.lookback_window, 20);
}

#[test]
fn test_indicator_calculation() {
    let bars = create_test_bars(0, 30);
    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);

    assert_eq!(indicators.len(), 30);
}

#[test]
fn test_instance_creation() {
    let ticker = create_test_ticker(0, "TEST");
    let bars = create_test_bars(0, 50);

    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);

    let creator = InstanceCreator::new(10);
    let instances = creator.create_instances(0, &ticker, &bars, &indicators);

    assert!(!instances.is_empty());
}

#[test]
fn test_embedding_tables() {
    let tables = EmbeddingTables::new(10, 16, 11, 8);
    let (ticker_embs, sector_embs) = tables.lookup_batch(&[0, 1], &[0, 1]);
    assert_eq!(ticker_embs.len(), 2 * tables.ticker_dim());
    assert_eq!(sector_embs.len(), 2 * tables.sector_dim());
}

#[test]
fn test_model_forward() {
    let config = StockPredictorConfig {
        num_tickers: 1,
        days_of_history: 50,
        lookback_window: 10,
        ticker_embedding_dim: 8,
        sector_embedding_dim: 4,
        dien_hidden_size: 16,
        dcn_cross_layers: 2,
        timeframes: vec![1],
        ..Default::default()
    };

    let indicator_dim = TechnicalIndicators::NUM_FEATURES * config.timeframes.len();
    let model = StockPredictionModel::new(&config, indicator_dim);

    let ticker = create_test_ticker(0, "TEST");
    let bars = create_test_bars(0, 30);

    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);
    let feature_index = FeatureIndex::new(
        vec![ticker.clone()],
        vec![bars.clone()],
        vec![vec![indicators.clone()]],
        vec![1],
    );

    let creator = InstanceCreator::new(config.lookback_window);
    let instances = creator.create_instances(0, &ticker, &bars, &indicators);
    let instance = instances.first().unwrap().clone();

    let batch = vec![&instance];
    let output = model.forward(&batch, &feature_index);

    assert_eq!(output.direction.len(), 1);
    assert_eq!(output.magnitude.len(), 1);
    assert_eq!(output.profitable.len(), 1);
    assert!(output.direction[0] >= 0.0 && output.direction[0] <= 1.0);
    assert!(output.profitable[0] >= 0.0 && output.profitable[0] <= 1.0);
}

#[test]
fn test_technical_indicators_to_vec() {
    let indicators = TechnicalIndicators::default();
    let vec = indicators.to_vec();
    assert_eq!(vec.len(), TechnicalIndicators::NUM_FEATURES);
}

#[test]
#[cfg(feature = "talib")]
fn test_talib_features_are_finite() {
    let bars = create_test_bars(0, 120);

    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);

    // After warm-up, TA-Lib features should be finite (may be 0.0 early due to lookback).
    for ind in indicators.iter().skip(50) {
        assert!(ind.talib_adx_14.is_finite());
        assert!(ind.talib_plus_di_14.is_finite());
        assert!(ind.talib_minus_di_14.is_finite());
        assert!(ind.talib_aroon_up_14.is_finite());
        assert!(ind.talib_aroon_down_14.is_finite());
        assert!(ind.talib_ultosc.is_finite());
    }
}

#[test]
fn test_random_generator() {
    let mut rng = RandomGenerator::new(42);
    let u = rng.uniform();
    assert!(u >= 0.0 && u < 1.0);
}

#[test]
fn test_mode_parsing() {
    assert_eq!(Mode::from_str("train"), Some(Mode::Train));
    assert_eq!(Mode::from_str("TRAIN"), Some(Mode::Train));
    assert_eq!(Mode::from_str("predict"), Some(Mode::Predict));
    assert_eq!(Mode::from_str("backtest"), Some(Mode::Backtest));
    assert_eq!(Mode::from_str("invalid"), None);
}

#[test]
fn test_sector() {
    assert_eq!(Sector::all().len(), 11);
    assert_eq!(Sector::Technology.id(), 0);
    assert_eq!(Sector::Technology.name(), "Technology");
}

#[test]
fn test_feature_index_indicator_dim_multi_tf() {
    let ticker = create_test_ticker(0, "TEST");
    let bars = create_test_bars(0, 40);

    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);

    let feature_index = FeatureIndex::new(
        vec![ticker],
        vec![bars],
        vec![vec![indicators.clone(), indicators]],
        vec![1, 5],
    );

    assert_eq!(
        feature_index.indicator_dim(),
        TechnicalIndicators::NUM_FEATURES * 2
    );
}

#[test]
fn test_sequence_features_are_finite() {
    let ticker = create_test_ticker(0, "TEST");
    let bars = create_test_bars(0, 60);

    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);
    let feature_index = FeatureIndex::new(
        vec![ticker.clone()],
        vec![bars.clone()],
        vec![vec![indicators.clone()]],
        vec![1],
    );

    let creator = InstanceCreator::new(10);
    let instances = creator.create_instances(0, &ticker, &bars, &indicators);
    let instance = instances.first().expect("instance").clone();

    let seq_dim = 4 + feature_index.indicator_dim();
    let mut buf = vec![0.0; seq_dim * 10];
    feature_index.write_sequence_features(&instance, 10, &mut buf);

    assert!(buf.iter().all(|v| v.is_finite()));
    assert!(instances.iter().all(|inst| {
        inst.direction_label.is_finite()
            && inst.magnitude_label.is_finite()
            && inst.profitable_label.is_finite()
    }));
}
