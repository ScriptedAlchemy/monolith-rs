use super::config::{Mode, StockPredictorConfig};
use super::data::{RandomGenerator, Sector, StockDataGenerator};
use super::indicators::{IndicatorCalculator, TechnicalIndicators};
use super::instances::{FeatureIndex, InstanceCreator};
use super::model::{EmbeddingTables, StockPredictionModel};

#[test]
fn test_config_default() {
    let config = StockPredictorConfig::default();
    assert_eq!(config.num_tickers, 50);
    assert_eq!(config.days_of_history, 252);
    assert_eq!(config.lookback_window, 20);
}

#[test]
fn test_stock_data_generation() {
    let mut generator = StockDataGenerator::new(42);
    generator.generate_tickers(10);
    generator.generate_bars(50);

    assert_eq!(generator.tickers().len(), 10);
    assert_eq!(generator.bars().len(), 10 * 50);
}

#[test]
fn test_indicator_calculation() {
    let mut generator = StockDataGenerator::new(42);
    generator.generate_tickers(1);
    generator.generate_bars(30);

    let bars = generator.bars().to_vec();
    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);

    assert_eq!(indicators.len(), 30);
}

#[test]
fn test_instance_creation() {
    let mut generator = StockDataGenerator::new(42);
    generator.generate_tickers(1);
    generator.generate_bars(50);

    let ticker = &generator.tickers()[0];
    let bars = generator.bars().to_vec();

    let mut calc = IndicatorCalculator::new();
    let indicators = calc.compute_indicators(&bars);

    let creator = InstanceCreator::new(10);
    let instances = creator.create_instances(0, ticker, &bars, &indicators);

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

    let model = StockPredictionModel::new(&config);

    let mut generator = StockDataGenerator::new(42);
    generator.generate_tickers(1);
    generator.generate_bars(30);
    let ticker = generator.tickers()[0].clone();
    let bars = generator.bars().to_vec();

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
fn test_random_generator() {
    let mut rng = RandomGenerator::new(42);
    let u = rng.uniform();
    assert!(u >= 0.0 && u < 1.0);

    let n = rng.normal();
    assert!(n.is_finite());
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
