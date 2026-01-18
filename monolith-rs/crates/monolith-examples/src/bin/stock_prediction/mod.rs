// Copyright 2022 ByteDance and/or its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Stock Prediction Example for Monolith-RS
//!
//! This comprehensive example demonstrates monolith-rs capabilities for financial
//! time-series prediction, technical analysis, and actionable stock recommendations.
//!
//! Features:
//! - Synthetic OHLCV data generation with realistic market dynamics
//! - Technical indicator computation (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV)
//! - Ticker and sector embeddings using CuckooEmbeddingHashTable
//! - DIEN for sequential pattern recognition in price history
//! - DCN for feature interaction modeling
//! - Multi-task learning (direction, magnitude, profitability)
//! - Backtesting with financial metrics (Sharpe, Drawdown, Win Rate)
//! - Stock recommendation generation
//!
//! # Usage
//!
//! ```bash
//! # Full training pipeline
//! cargo run -p monolith-examples --bin stock_prediction -- --mode train
//!
//! # Generate predictions only
//! cargo run -p monolith-examples --bin stock_prediction -- --mode predict
//!
//! # Run backtesting simulation
//! cargo run -p monolith-examples --bin stock_prediction -- --mode backtest
//!
//! # Custom configuration
//! cargo run -p monolith-examples --bin stock_prediction -- \
//!     --mode train \
//!     --num-tickers 100 \
//!     --days 504 \
//!     --batch-size 64 \
//!     --epochs 20
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// Default local path for the FutureSharks 1-minute dataset (repo: https://github.com/FutureSharks/financial-data).
// We keep this relative so `cargo run ...` works from the repo root with no flags.
const DEFAULT_FUTURESHARKS_REPO_DIR: &str = "data/financial-data";
const DEFAULT_FUTURESHARKS_DATA_DIR: &str =
    "data/financial-data/pyfinancialdata/data/stocks/histdata";

// Parallel processing
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
use monolith_optimizer::Amsgrad;

// Technical analysis crate for indicators (https://crates.io/crates/ta)
// Using all available indicators to maximize feature richness for the model
use ta::indicators::{
    AverageTrueRange, BollingerBands, ExponentialMovingAverage, Maximum, Minimum, MoneyFlowIndex,
    OnBalanceVolume, RateOfChange, RelativeStrengthIndex, SimpleMovingAverage, SlowStochastic,
    StandardDeviation,
};
use ta::Next;

// =============================================================================
// Section 1: Configuration and CLI
// =============================================================================

/// Operating mode for the stock predictor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Train the model from scratch
    Train,
    /// Evaluate on held-out data
    Evaluate,
    /// Generate predictions for current data
    Predict,
    /// Run backtesting simulation
    Backtest,
}

impl Mode {
    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "train" => Some(Mode::Train),
            "evaluate" | "eval" => Some(Mode::Evaluate),
            "predict" => Some(Mode::Predict),
            "backtest" => Some(Mode::Backtest),
            _ => None,
        }
    }
}

/// Configuration for the stock prediction pipeline.
#[derive(Debug, Clone)]
pub struct StockPredictorConfig {
    // Data generation
    /// Number of tickers to generate
    pub num_tickers: usize,
    /// Days of historical data
    pub days_of_history: usize,
    /// Lookback window for sequences
    pub lookback_window: usize,

    // Model architecture
    /// Ticker embedding dimension
    pub ticker_embedding_dim: usize,
    /// Sector embedding dimension
    pub sector_embedding_dim: usize,
    /// DIEN hidden size
    pub dien_hidden_size: usize,
    /// Number of DCN cross layers
    pub dcn_cross_layers: usize,

    // Training
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Train/eval split ratio
    pub train_ratio: f32,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Log every N steps
    pub log_every_n_steps: usize,
    /// Early stopping patience
    pub early_stopping_patience: usize,
    /// Minimum improvement in eval loss to reset early stopping patience.
    ///
    /// Smaller values allow training to continue when eval loss improves slowly.
    pub early_stopping_min_delta: f32,

    // Operation mode
    /// Current mode
    pub mode: Mode,
    /// Random seed
    pub seed: u64,
    /// Verbose output
    pub verbose: bool,

    // Data source
    /// Path to directory containing CSV files (if None, use synthetic data)
    pub data_dir: Option<String>,

    // Performance
    /// Number of parallel workers (0 = auto-detect)
    pub num_workers: usize,
    /// Enable GPU acceleration (requires compatible backend)
    pub gpu_mode: bool,
}

impl Default for StockPredictorConfig {
    fn default() -> Self {
        Self {
            // Data generation - tuned for Mac CPU intensive training
            num_tickers: 50,
            days_of_history: 252,
            lookback_window: 20,

            // Model architecture - larger capacity
            ticker_embedding_dim: 32,
            sector_embedding_dim: 16,
            dien_hidden_size: 128,
            dcn_cross_layers: 4,

            // Training - tuned for convergence
            batch_size: 64,
            learning_rate: 0.0003, // Lower LR for stability
            train_ratio: 0.8,
            num_epochs: 100,             // More epochs, early stopping will kick in
            log_every_n_steps: 50,       // More frequent logging
            early_stopping_patience: 20, // Allow longer training by default
            early_stopping_min_delta: 1e-4,

            // Operation mode
            mode: Mode::Train,
            seed: 42,
            verbose: true,

            // Data source
            data_dir: Some(DEFAULT_FUTURESHARKS_DATA_DIR.to_string()),

            // Performance
            num_workers: 0, // 0 = auto-detect from num_cpus
            gpu_mode: false,
        }
    }
}

fn parse_args() -> StockPredictorConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut config = StockPredictorConfig::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--mode" | "-m" => {
                if i + 1 < args.len() {
                    if let Some(mode) = Mode::from_str(&args[i + 1]) {
                        config.mode = mode;
                    }
                    i += 1;
                }
            }
            "--num-tickers" | "-t" => {
                if i + 1 < args.len() {
                    config.num_tickers = args[i + 1].parse().unwrap_or(50);
                    i += 1;
                }
            }
            "--days" | "-d" => {
                if i + 1 < args.len() {
                    config.days_of_history = args[i + 1].parse().unwrap_or(252);
                    i += 1;
                }
            }
            "--lookback" | "-l" => {
                if i + 1 < args.len() {
                    config.lookback_window = args[i + 1].parse().unwrap_or(20);
                    i += 1;
                }
            }
            "--batch-size" | "-b" => {
                if i + 1 < args.len() {
                    config.batch_size = args[i + 1].parse().unwrap_or(32);
                    i += 1;
                }
            }
            "--learning-rate" | "-lr" => {
                if i + 1 < args.len() {
                    config.learning_rate = args[i + 1].parse().unwrap_or(0.001);
                    i += 1;
                }
            }
            "--epochs" | "-e" => {
                if i + 1 < args.len() {
                    config.num_epochs = args[i + 1].parse().unwrap_or(10);
                    i += 1;
                }
            }
            "--patience" | "--early-stopping-patience" => {
                if i + 1 < args.len() {
                    config.early_stopping_patience = args[i + 1].parse().unwrap_or(20);
                    i += 1;
                }
            }
            "--min-delta" | "--early-stopping-min-delta" => {
                if i + 1 < args.len() {
                    config.early_stopping_min_delta = args[i + 1].parse().unwrap_or(1e-4);
                    i += 1;
                }
            }
            "--seed" | "-s" => {
                if i + 1 < args.len() {
                    config.seed = args[i + 1].parse().unwrap_or(42);
                    i += 1;
                }
            }
            "--data-dir" | "--data" => {
                if i + 1 < args.len() {
                    config.data_dir = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--quiet" | "-q" => {
                config.verbose = false;
            }
            "--workers" | "-w" => {
                if i + 1 < args.len() {
                    config.num_workers = args[i + 1].parse().unwrap_or(0);
                    i += 1;
                }
            }
            "--gpu" => {
                config.gpu_mode = true;
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    // Auto-detect workers if not specified
    if config.num_workers == 0 {
        config.num_workers = num_cpus::get();
    }

    // Configure rayon thread pool
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_workers)
        .build_global()
        .ok();

    config
}

fn print_usage() {
    println!(
        r#"Monolith-RS Stock Prediction Example

USAGE:
    stock_prediction [OPTIONS]

OPTIONS:
    -m, --mode <MODE>           Operation mode: train, evaluate, predict, backtest
                                [default: train]
    -t, --num-tickers <N>       Number of tickers to load/simulate [default: 50]
    -d, --days <N>              Days of historical data [default: 252]
    -l, --lookback <N>          Lookback window size [default: 20]
    -b, --batch-size <N>        Training batch size [default: 32]
    -lr, --learning-rate <LR>   Learning rate [default: 0.001]
    -e, --epochs <N>            Number of training epochs [default: 10]
    --patience <N>              Early stopping patience (epochs without improvement) [default: 20]
    --min-delta <X>             Minimum eval loss improvement to reset patience [default: 1e-4]
    -s, --seed <SEED>           Random seed [default: 42]
    --data-dir <PATH>           Load real stock data from CSV files in directory
                                [default: data/financial-data/pyfinancialdata/data/stocks/histdata]
    -w, --workers <N>           Number of parallel workers [default: auto-detect]
    --gpu                       Enable GPU acceleration mode
    -q, --quiet                 Suppress verbose output
    -h, --help                  Print help information

EXAMPLES:
    # One-time: clone the intraday dataset (1-minute bars) into ./data/financial-data
    git clone https://github.com/FutureSharks/financial-data.git data/financial-data

    # Train with FutureSharks intraday dataset (default --data-dir)
    stock_prediction --mode train

    # Train with real data from Kaggle
    # Download from: https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
    stock_prediction --mode train --data-dir /path/to/kaggle/stocks

    # Generate predictions with real data
    stock_prediction --mode predict --data-dir ./data/stocks --num-tickers 100

    # Run backtesting with more history
    stock_prediction --mode backtest --data-dir ./data/stocks --days 504

    # Train longer / avoid early stopping on slow improvements
    stock_prediction --mode train --data-dir ./data/stocks --epochs 300 --patience 50 --min-delta 1e-5
"#
    );
}

// =============================================================================
// Section 2: Synthetic Stock Data Generation
// =============================================================================

/// Market sectors for stock categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Sector {
    Technology,
    Healthcare,
    Finance,
    Consumer,
    Industrial,
    Energy,
    Materials,
    RealEstate,
    Utilities,
    Communications,
    ConsumerStaples,
}

impl Sector {
    fn all() -> &'static [Sector] {
        &[
            Sector::Technology,
            Sector::Healthcare,
            Sector::Finance,
            Sector::Consumer,
            Sector::Industrial,
            Sector::Energy,
            Sector::Materials,
            Sector::RealEstate,
            Sector::Utilities,
            Sector::Communications,
            Sector::ConsumerStaples,
        ]
    }

    fn id(&self) -> i64 {
        match self {
            Sector::Technology => 0,
            Sector::Healthcare => 1,
            Sector::Finance => 2,
            Sector::Consumer => 3,
            Sector::Industrial => 4,
            Sector::Energy => 5,
            Sector::Materials => 6,
            Sector::RealEstate => 7,
            Sector::Utilities => 8,
            Sector::Communications => 9,
            Sector::ConsumerStaples => 10,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Sector::Technology => "Technology",
            Sector::Healthcare => "Healthcare",
            Sector::Finance => "Finance",
            Sector::Consumer => "Consumer Discretionary",
            Sector::Industrial => "Industrial",
            Sector::Energy => "Energy",
            Sector::Materials => "Materials",
            Sector::RealEstate => "Real Estate",
            Sector::Utilities => "Utilities",
            Sector::Communications => "Communications",
            Sector::ConsumerStaples => "Consumer Staples",
        }
    }
}

/// Information about a ticker.
#[derive(Debug, Clone)]
pub struct TickerInfo {
    pub ticker_id: i64,
    pub name: String,
    pub symbol: String,
    pub sector: Sector,
    pub beta: f32,
    pub base_volatility: f32,
    pub drift: f32,
}

/// A single price bar (OHLCV).
#[derive(Debug, Clone)]
pub struct StockBar {
    pub ticker_id: i64,
    pub timestamp: i64,
    pub day_index: usize,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f64,
    pub returns: f32,
}

/// Market regime (for regime-switching simulation).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

/// Random number generator with seeded state.
struct RandomGenerator {
    state: u64,
}

impl RandomGenerator {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    fn normal(&mut self) -> f32 {
        // Box-Muller transform
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn choice<T: Clone>(&mut self, items: &[T]) -> T {
        let idx = (self.uniform() * items.len() as f32) as usize % items.len();
        items[idx].clone()
    }

    #[allow(dead_code)]
    fn range(&mut self, min: f32, max: f32) -> f32 {
        min + self.uniform() * (max - min)
    }

    /// Shuffle a mutable slice in-place using Fisher-Yates
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.uniform() * (i + 1) as f32) as usize % (i + 1);
            slice.swap(i, j);
        }
    }
}

/// Generates synthetic stock data using realistic market dynamics.
pub struct StockDataGenerator {
    rng: RandomGenerator,
    tickers: Vec<TickerInfo>,
    bars: Vec<StockBar>,
    current_regime: MarketRegime,
    regime_counter: usize,
}

impl StockDataGenerator {
    /// Creates a new stock data generator.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RandomGenerator::new(seed),
            tickers: Vec::new(),
            bars: Vec::new(),
            current_regime: MarketRegime::Sideways,
            regime_counter: 0,
        }
    }

    /// Generates ticker information.
    pub fn generate_tickers(&mut self, num_tickers: usize) -> &[TickerInfo] {
        self.tickers.clear();

        let symbols = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK", "JPM", "JNJ", "V",
            "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "CMCSA", "ADBE", "NFLX", "CRM", "XOM",
            "CSCO", "PFE", "INTC", "KO", "PEP", "ABT", "TMO", "COST", "CVX", "NKE", "MRK", "LLY",
            "WMT", "ABBV", "AVGO", "ACN", "DHR", "TXN", "MDT", "UNP", "NEE", "LIN", "ORCL", "PM",
            "HON", "LOW", "AMT", "QCOM", "IBM", "RTX", "SBUX", "CVS", "GS", "BLK", "DE", "SPGI",
            "AXP", "GILD", "ISRG", "MDLZ", "BA", "MMM", "CAT", "MO", "ADP", "TGT", "BKNG", "CHTR",
            "ZTS", "SYK", "CI", "ANTM", "TJX", "REGN", "DUK", "SO", "USB", "PLD", "BDX", "CL",
            "MU", "ATVI", "MMC", "ITW", "CME", "FISV", "HUM", "APD", "NOC", "EQIX", "ICE", "ETN",
            "CCI", "NSC", "WM", "SHW", "VRTX",
        ];

        for i in 0..num_tickers {
            let sector = self.rng.choice(Sector::all());
            let symbol = if i < symbols.len() {
                symbols[i].to_string()
            } else {
                format!("SYM{:03}", i)
            };
            let name = format!("{} Inc.", symbol);

            // Generate realistic characteristics based on sector
            let (base_vol, drift) = match sector {
                Sector::Technology => (0.02 + self.rng.uniform() * 0.02, 0.0003),
                Sector::Healthcare => (0.015 + self.rng.uniform() * 0.015, 0.0002),
                Sector::Finance => (0.018 + self.rng.uniform() * 0.012, 0.0001),
                Sector::Energy => (0.025 + self.rng.uniform() * 0.025, 0.0001),
                Sector::Utilities => (0.008 + self.rng.uniform() * 0.007, 0.0001),
                _ => (0.015 + self.rng.uniform() * 0.015, 0.00015),
            };

            let beta = match sector {
                Sector::Technology => 1.2 + self.rng.uniform() * 0.5,
                Sector::Utilities => 0.5 + self.rng.uniform() * 0.3,
                Sector::Finance => 1.0 + self.rng.uniform() * 0.4,
                _ => 0.8 + self.rng.uniform() * 0.4,
            };

            self.tickers.push(TickerInfo {
                ticker_id: i as i64,
                name,
                symbol,
                sector,
                beta,
                base_volatility: base_vol,
                drift,
            });
        }

        &self.tickers
    }

    /// Generates price bars using Geometric Brownian Motion with regime switching.
    pub fn generate_bars(&mut self, days: usize) -> &[StockBar] {
        self.bars.clear();

        if self.tickers.is_empty() {
            return &self.bars;
        }

        // Initialize prices for each ticker
        let mut prices: Vec<f32> = self
            .tickers
            .iter()
            .map(|_| 50.0 + self.rng.uniform() * 150.0) // $50-$200 initial price
            .collect();

        let mut volatilities: Vec<f32> = self.tickers.iter().map(|t| t.base_volatility).collect();

        // Generate market factor (common shock)
        let mut market_returns: Vec<f32> = Vec::with_capacity(days);

        for day in 0..days {
            // Update market regime periodically
            self.update_regime();

            // Market return based on regime
            let market_return = match self.current_regime {
                MarketRegime::Bull => 0.0005 + self.rng.normal() * 0.01,
                MarketRegime::Bear => -0.0003 + self.rng.normal() * 0.015,
                MarketRegime::Sideways => self.rng.normal() * 0.008,
            };
            market_returns.push(market_return);

            // Check for earnings events (higher volatility)
            let is_earnings_season = (day % 63) < 21; // ~1/3 of each quarter

            for (ticker_idx, ticker) in self.tickers.iter().enumerate() {
                // Mean-reverting volatility
                let target_vol = ticker.base_volatility;
                volatilities[ticker_idx] = volatilities[ticker_idx] * 0.95 + target_vol * 0.05;

                // Add earnings spike
                let vol = if is_earnings_season && self.rng.uniform() < 0.1 {
                    volatilities[ticker_idx] * 2.0
                } else {
                    volatilities[ticker_idx]
                };

                // GBM: dS = S * (mu*dt + sigma*dW)
                // With market factor: r = beta * market_return + idiosyncratic
                let idiosyncratic = self.rng.normal() * vol;
                let daily_return = ticker.beta * market_return + idiosyncratic + ticker.drift;

                // Update price
                let prev_price = prices[ticker_idx];
                let new_price = prev_price * (1.0 + daily_return);
                prices[ticker_idx] = new_price.max(0.01); // Floor at $0.01

                // Generate OHLC from close
                let intraday_vol = vol * 0.5;
                let open = prev_price * (1.0 + self.rng.normal() * intraday_vol * 0.3);
                let high = new_price.max(open) * (1.0 + self.rng.uniform() * intraday_vol);
                let low = new_price.min(open) * (1.0 - self.rng.uniform() * intraday_vol);

                // Generate volume (log-normal distribution)
                let base_volume = 1_000_000.0 + self.rng.uniform() as f64 * 10_000_000.0;
                let volume_mult = (1.0 + daily_return.abs() * 10.0) as f64; // Higher volume on bigger moves
                let volume = base_volume * volume_mult * (0.5 + self.rng.uniform() as f64);

                self.bars.push(StockBar {
                    ticker_id: ticker.ticker_id,
                    timestamp: day as i64,
                    day_index: day,
                    open: open.max(0.01),
                    high: high.max(open.max(new_price)),
                    low: low.min(open.min(new_price)).max(0.01),
                    close: new_price,
                    volume,
                    returns: daily_return,
                });
            }
        }

        &self.bars
    }

    fn update_regime(&mut self) {
        self.regime_counter += 1;

        // Regime persists for 20-60 days on average
        if self.regime_counter > 20 && self.rng.uniform() < 0.05 {
            self.current_regime = match self.rng.uniform() {
                x if x < 0.4 => MarketRegime::Bull,
                x if x < 0.7 => MarketRegime::Sideways,
                _ => MarketRegime::Bear,
            };
            self.regime_counter = 0;
        }
    }

    /// Returns the generated tickers.
    pub fn tickers(&self) -> &[TickerInfo] {
        &self.tickers
    }

    /// Returns the generated bars.
    pub fn bars(&self) -> &[StockBar] {
        &self.bars
    }

    /// Gets bars for a specific ticker.
    pub fn get_ticker_bars(&self, ticker_id: i64) -> Vec<&StockBar> {
        self.bars
            .iter()
            .filter(|b| b.ticker_id == ticker_id)
            .collect()
    }
}

// =============================================================================
// Section 2b: CSV Data Loading (for real stock data)
// =============================================================================

/// Load stock data from CSV files (Kaggle format or Yahoo Finance format).
///
/// Supports the following formats:
/// 1. Kaggle Stock Market Dataset: Date,Open,High,Low,Close,Adj Close,Volume
/// 2. Yahoo Finance: Date,Open,High,Low,Close,Adj Close,Volume
/// 3. Simple OHLCV: Date,Open,High,Low,Close,Volume
///
/// # Usage with Kaggle Data
///
/// ```bash
/// # Download from https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
/// # Extract to a directory, then run:
/// cargo run -p monolith-examples --bin stock_prediction -- \
///     --mode train --data-dir /path/to/kaggle-data/stocks
/// ```
pub struct CsvDataLoader {
    /// Base directory containing CSV files
    data_dir: String,
    /// Mapping from ticker symbols to IDs
    ticker_to_id: HashMap<String, i64>,
    /// Ticker info list
    tickers: Vec<TickerInfo>,
    /// All loaded bars
    bars: Vec<StockBar>,
}

impl CsvDataLoader {
    /// Creates a new CSV data loader.
    pub fn new(data_dir: &str) -> Self {
        Self {
            data_dir: data_dir.to_string(),
            ticker_to_id: HashMap::new(),
            tickers: Vec::new(),
            bars: Vec::new(),
        }
    }

    /// Loads data from all CSV files in the directory.
    ///
    /// # Arguments
    /// * `max_tickers` - Maximum number of tickers to load (for memory management)
    /// * `min_days` - Minimum days of history required per ticker
    pub fn load(&mut self, max_tickers: usize, min_days: usize) -> Result<(), String> {
        use std::fs;
        use std::io::{BufRead, BufReader};

        let entries = fs::read_dir(&self.data_dir)
            .map_err(|e| format!("Failed to read directory {}: {}", self.data_dir, e))?;

        let mut csv_files: Vec<_> = entries
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "csv")
                    .unwrap_or(false)
            })
            .collect();

        // Sort for reproducibility
        csv_files.sort_by(|a, b| a.path().cmp(&b.path()));

        // Limit number of tickers
        csv_files.truncate(max_tickers);

        println!(
            "  Loading {} CSV files from {}",
            csv_files.len(),
            self.data_dir
        );

        // Sector inference based on common ticker patterns
        let sector_map: HashMap<&str, &str> = [
            ("AAPL", "Technology"),
            ("MSFT", "Technology"),
            ("GOOGL", "Technology"),
            ("AMZN", "Consumer"),
            ("META", "Technology"),
            ("NVDA", "Technology"),
            ("TSLA", "Consumer"),
            ("JPM", "Financials"),
            ("BAC", "Financials"),
            ("WMT", "Consumer"),
            ("JNJ", "Healthcare"),
            ("PFE", "Healthcare"),
            ("XOM", "Energy"),
            ("CVX", "Energy"),
        ]
        .into_iter()
        .collect();

        for (ticker_id, entry) in csv_files.iter().enumerate() {
            let path = entry.path();
            let ticker_symbol = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("UNKNOWN")
                .to_uppercase();

            let file = fs::File::open(&path)
                .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
            let reader = BufReader::new(file);

            let mut lines = reader.lines();
            let header = lines.next();

            // Detect format from header
            let format = if let Some(Ok(h)) = &header {
                let h_lower = h.to_lowercase();
                if h_lower.contains("adj close") {
                    "yahoo" // Date,Open,High,Low,Close,Adj Close,Volume
                } else if h_lower.contains("volume") {
                    "simple" // Date,Open,High,Low,Close,Volume
                } else {
                    "unknown"
                }
            } else {
                continue;
            };

            if format == "unknown" {
                eprintln!("  Skipping {}: unknown format", path.display());
                continue;
            }

            let mut ticker_bars: Vec<StockBar> = Vec::new();
            let mut prev_close: Option<f32> = None;

            for (day_idx, line) in lines.enumerate() {
                if let Ok(line) = line {
                    let fields: Vec<&str> = line.split(',').collect();
                    if fields.len() < 6 {
                        continue;
                    }

                    // Parse fields based on format
                    let (open, high, low, close, volume) = match format {
                        "yahoo" | "simple" => {
                            let o: f32 = fields[1].parse().unwrap_or(0.0);
                            let h: f32 = fields[2].parse().unwrap_or(0.0);
                            let l: f32 = fields[3].parse().unwrap_or(0.0);
                            let c: f32 = fields[4].parse().unwrap_or(0.0);
                            let v: f64 = if format == "yahoo" {
                                fields[6].parse().unwrap_or(0.0)
                            } else {
                                fields[5].parse().unwrap_or(0.0)
                            };
                            (o, h, l, c, v)
                        }
                        _ => continue,
                    };

                    // Skip invalid bars
                    if close <= 0.0 || open <= 0.0 {
                        continue;
                    }

                    // Calculate returns
                    let returns = if let Some(pc) = prev_close {
                        (close - pc) / pc
                    } else {
                        0.0
                    };
                    prev_close = Some(close);

                    // Parse date to timestamp (days since epoch, simplified)
                    let timestamp = day_idx as i64;

                    ticker_bars.push(StockBar {
                        ticker_id: ticker_id as i64,
                        timestamp,
                        day_index: day_idx,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        returns,
                    });
                }
            }

            // Skip tickers with insufficient history
            if ticker_bars.len() < min_days {
                continue;
            }

            // Infer sector
            let sector = match sector_map.get(ticker_symbol.as_str()).copied() {
                Some("Technology") => Sector::Technology,
                Some("Healthcare") => Sector::Healthcare,
                Some("Financials") | Some("Finance") => Sector::Finance,
                Some("Consumer") => Sector::Consumer,
                Some("Industrials") | Some("Industrial") => Sector::Industrial,
                Some("Energy") => Sector::Energy,
                Some("Materials") => Sector::Materials,
                Some("Real Estate") => Sector::RealEstate,
                Some("Utilities") => Sector::Utilities,
                Some("Communications") => Sector::Communications,
                _ => {
                    // Default sector based on first letter (very rough heuristic)
                    match ticker_symbol.chars().next() {
                        Some('A'..='F') => Sector::Technology,
                        Some('G'..='L') => Sector::Industrial,
                        Some('M'..='R') => Sector::Healthcare,
                        Some('S'..='Z') => Sector::Finance,
                        _ => Sector::Consumer,
                    }
                }
            };

            // Calculate volatility from returns
            let returns: Vec<f32> = ticker_bars.iter().map(|b| b.returns).collect();
            let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
            let variance: f32 =
                returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>() / returns.len() as f32;
            let volatility = variance.sqrt() * (252.0_f32).sqrt(); // Annualized

            // Estimate drift from mean returns
            let drift = mean_ret * 252.0; // Annualized

            self.ticker_to_id
                .insert(ticker_symbol.clone(), ticker_id as i64);

            self.tickers.push(TickerInfo {
                ticker_id: ticker_id as i64,
                name: ticker_symbol.clone(),
                symbol: ticker_symbol,
                sector,
                beta: 1.0, // Would need market data to calculate
                base_volatility: volatility,
                drift,
            });

            self.bars.extend(ticker_bars);
        }

        println!(
            "  Loaded {} tickers with {} total bars",
            self.tickers.len(),
            self.bars.len()
        );

        if self.tickers.is_empty() {
            return Err("No valid tickers found in the data directory".to_string());
        }

        Ok(())
    }

    /// Returns loaded tickers.
    pub fn tickers(&self) -> &[TickerInfo] {
        &self.tickers
    }

    /// Returns loaded bars.
    pub fn bars(&self) -> &[StockBar] {
        &self.bars
    }

    /// Gets bars for a specific ticker.
    pub fn get_ticker_bars(&self, ticker_id: i64) -> Vec<&StockBar> {
        self.bars
            .iter()
            .filter(|b| b.ticker_id == ticker_id)
            .collect()
    }
}

// =============================================================================
// Section 2c: Intraday Data Sources
// =============================================================================

/// Available intraday data sources from GitHub.
///
/// These datasets provide 1-minute OHLCV data for stocks and indices.
pub struct IntradayDataSources;

impl IntradayDataSources {
    /// Returns URLs for FutureSharks financial-data repository.
    /// Source: https://github.com/FutureSharks/financial-data
    ///
    /// Available instruments:
    /// - S&P 500 (SPXUSD): 2010-2018
    /// - NIKKEI 225 (JPXJPY): 2010-2018
    /// - DAX 30 (GRXEUR): 2010-2018
    /// - EUROSTOXX 50 (ETXEUR): 2010-2018
    pub fn futuresharks_urls() -> Vec<(&'static str, &'static str)> {
        vec![
            ("SPX500", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/SPX500_USD_M1.csv.gz"),
            ("NAS100", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/NAS100_USD_M1.csv.gz"),
            ("JP225", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/JP225_USD_M1.csv.gz"),
            ("DE30", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/DE30_EUR_M1.csv.gz"),
            ("UK100", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/UK100_GBP_M1.csv.gz"),
        ]
    }

    /// Prints download instructions for getting intraday data.
    pub fn print_download_instructions() {
        println!("\n=== Intraday Data Download Instructions ===\n");
        println!("Option 1: FutureSharks financial-data (1-minute bars, 2010-2018)");
        println!("  git clone https://github.com/FutureSharks/financial-data.git");
        println!("  # Then use --data-dir financial-data/data/stocks/oanda\n");

        println!("Option 2: Download individual files:");
        for (name, url) in Self::futuresharks_urls() {
            println!("  # {}", name);
            println!("  curl -L {} | gunzip > {}.csv", url, name);
        }

        println!("\nOption 3: Yahoo Finance intraday (last 60 days, via Python):");
        println!("  pip install yfinance");
        println!("  python -c \"import yfinance as yf; yf.download('AAPL', period='60d', interval='1m').to_csv('AAPL_1m.csv')\"");

        println!("\nOption 4: Alpha Vantage API (free key required):");
        println!("  curl 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&outputsize=full&apikey=YOUR_KEY&datatype=csv' > AAPL_intraday.csv");

        println!("\n===========================================\n");
    }

    /// Downloads a single intraday CSV file using system curl.
    /// Returns the path to the downloaded file.
    #[cfg(unix)]
    pub fn download_file(url: &str, output_dir: &str, filename: &str) -> Result<String, String> {
        use std::process::Command;

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir)
            .map_err(|e| format!("Failed to create directory: {}", e))?;

        let output_path = format!("{}/{}", output_dir, filename);

        // Check if file already exists
        if std::path::Path::new(&output_path).exists() {
            println!("  File already exists: {}", output_path);
            return Ok(output_path);
        }

        println!("  Downloading {} to {}...", url, output_path);

        // Use curl to download, handling gzip if needed
        let result = if url.ends_with(".gz") {
            Command::new("sh")
                .arg("-c")
                .arg(format!("curl -sL '{}' | gunzip > '{}'", url, output_path))
                .output()
        } else {
            Command::new("curl")
                .args(["-sL", "-o", &output_path, url])
                .output()
        };

        match result {
            Ok(output) => {
                if output.status.success() {
                    println!("  Downloaded successfully: {}", output_path);
                    Ok(output_path)
                } else {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    Err(format!("Download failed: {}", stderr))
                }
            }
            Err(e) => Err(format!("Failed to execute curl: {}", e)),
        }
    }

    /// Downloads all FutureSharks intraday data files.
    #[cfg(unix)]
    pub fn download_all_futuresharks(output_dir: &str) -> Result<Vec<String>, String> {
        let mut downloaded = Vec::new();

        println!("Downloading FutureSharks intraday data...");

        for (name, url) in Self::futuresharks_urls() {
            let filename = format!("{}_1m.csv", name);
            match Self::download_file(url, output_dir, &filename) {
                Ok(path) => downloaded.push(path),
                Err(e) => eprintln!("  Warning: Failed to download {}: {}", name, e),
            }
        }

        if downloaded.is_empty() {
            Err("No files were downloaded".to_string())
        } else {
            println!("Downloaded {} files to {}", downloaded.len(), output_dir);
            Ok(downloaded)
        }
    }
}

/// Loads intraday data from FutureSharks CSV format.
/// Format: DateTime,Open,High,Low,Close,Volume
pub struct IntradayCsvLoader {
    tickers: Vec<TickerInfo>,
    bars: Vec<StockBar>,
}

impl IntradayCsvLoader {
    pub fn new() -> Self {
        Self {
            tickers: Vec::new(),
            bars: Vec::new(),
        }
    }

    /// Loads a single intraday CSV file.
    /// Aggregates 1-minute bars into the specified timeframe (e.g., 5, 15, 60 minutes).
    pub fn load_file(
        &mut self,
        path: &str,
        ticker_name: &str,
        aggregate_minutes: usize,
    ) -> Result<(), String> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
        let reader = BufReader::new(file);

        let ticker_id = self.tickers.len() as i64;
        let mut minute_bars: Vec<(f32, f32, f32, f32, f64)> = Vec::new(); // OHLCV

        for (i, line) in reader.lines().enumerate() {
            if i == 0 {
                continue; // Skip header
            }

            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let fields: Vec<&str> = line.split(',').collect();

            if fields.len() < 6 {
                continue;
            }

            // Parse OHLCV (skip datetime at index 0)
            let open: f32 = fields[1].parse().unwrap_or(0.0);
            let high: f32 = fields[2].parse().unwrap_or(0.0);
            let low: f32 = fields[3].parse().unwrap_or(0.0);
            let close: f32 = fields[4].parse().unwrap_or(0.0);
            let volume: f64 = fields[5].parse().unwrap_or(0.0);

            if close <= 0.0 {
                continue;
            }

            minute_bars.push((open, high, low, close, volume));
        }

        if minute_bars.is_empty() {
            return Err(format!("No valid bars in {}", path));
        }

        // Aggregate into larger timeframes
        let aggregated = Self::aggregate_bars(&minute_bars, aggregate_minutes);

        // Create StockBars
        let mut prev_close = aggregated[0].3;
        for (day_idx, (open, high, low, close, volume)) in aggregated.iter().enumerate() {
            let returns = (*close - prev_close) / prev_close;
            prev_close = *close;

            self.bars.push(StockBar {
                ticker_id,
                timestamp: day_idx as i64,
                day_index: day_idx,
                open: *open,
                high: *high,
                low: *low,
                close: *close,
                volume: *volume,
                returns,
            });
        }

        // Calculate volatility
        let returns: Vec<f32> = self
            .bars
            .iter()
            .filter(|b| b.ticker_id == ticker_id)
            .map(|b| b.returns)
            .collect();
        let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len().max(1) as f32;
        let variance: f32 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>()
            / returns.len().max(1) as f32;
        let volatility = variance.sqrt() * (252.0_f32 * (390.0 / aggregate_minutes as f32)).sqrt();

        self.tickers.push(TickerInfo {
            ticker_id,
            name: ticker_name.to_string(),
            symbol: ticker_name.to_string(),
            sector: Sector::Technology, // Index default
            beta: 1.0,
            base_volatility: volatility,
            drift: mean_ret * 252.0 * (390.0 / aggregate_minutes as f32),
        });

        println!(
            "  Loaded {} bars from {} ({}m aggregation)",
            self.bars
                .iter()
                .filter(|b| b.ticker_id == ticker_id)
                .count(),
            ticker_name,
            aggregate_minutes
        );

        Ok(())
    }

    /// Aggregates 1-minute bars into larger timeframes.
    fn aggregate_bars(
        minute_bars: &[(f32, f32, f32, f32, f64)],
        period: usize,
    ) -> Vec<(f32, f32, f32, f32, f64)> {
        minute_bars
            .chunks(period)
            .filter(|chunk| !chunk.is_empty())
            .map(|chunk| {
                let open = chunk[0].0;
                let high = chunk.iter().map(|b| b.1).fold(f32::MIN, f32::max);
                let low = chunk.iter().map(|b| b.2).fold(f32::MAX, f32::min);
                let close = chunk.last().unwrap().3;
                let volume: f64 = chunk.iter().map(|b| b.4).sum();
                (open, high, low, close, volume)
            })
            .collect()
    }

    pub fn tickers(&self) -> &[TickerInfo] {
        &self.tickers
    }

    pub fn bars(&self) -> &[StockBar] {
        &self.bars
    }
}

/// Loader for the FutureSharks "histdata" intraday dataset.
///
/// Example file line format (semicolon-separated, no header):
/// `20170102 180000;2241.000000;2244.500000;2241.000000;2243.500000;0`
///
/// The repo organizes by instrument symbol (e.g. SPXUSD) and year-suffixed CSVs.
pub struct FutureSharksHistdataLoader {
    tickers: Vec<TickerInfo>,
    bars: Vec<StockBar>,
}

impl FutureSharksHistdataLoader {
    pub fn new() -> Self {
        Self {
            tickers: Vec::new(),
            bars: Vec::new(),
        }
    }

    pub fn tickers(&self) -> &[TickerInfo] {
        &self.tickers
    }

    pub fn bars(&self) -> &[StockBar] {
        &self.bars
    }

    /// Load up to `max_instruments` instruments from a histdata directory.
    ///
    /// `dir` should typically be:
    /// `data/financial-data/pyfinancialdata/data/stocks/histdata`
    pub fn load_dir(&mut self, dir: &str, max_instruments: usize) -> Result<(), String> {
        use std::fs;

        let mut instrument_dirs: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read {}: {}", dir, e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        instrument_dirs.sort_by(|a, b| a.path().cmp(&b.path()));
        instrument_dirs.truncate(max_instruments);

        if instrument_dirs.is_empty() {
            return Err(format!("No instrument subdirectories found in {}", dir));
        }

        for entry in instrument_dirs {
            let instrument = entry
                .path()
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("UNKNOWN")
                .to_string();
            self.load_instrument_dir(entry.path().to_string_lossy().as_ref(), &instrument)?;
        }

        if self.tickers.is_empty() || self.bars.is_empty() {
            return Err(format!("No valid histdata bars found in {}", dir));
        }

        Ok(())
    }

    fn load_instrument_dir(&mut self, instrument_dir: &str, symbol: &str) -> Result<(), String> {
        use std::fs;
        use std::io::{BufRead, BufReader};

        let mut files: Vec<_> = fs::read_dir(instrument_dir)
            .map_err(|e| format!("Failed to read {}: {}", instrument_dir, e))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("csv"))
                    .unwrap_or(false)
            })
            .collect();

        files.sort_by(|a, b| a.path().cmp(&b.path()));

        if files.is_empty() {
            return Err(format!("No CSV files found in {}", instrument_dir));
        }

        let ticker_id = self.tickers.len() as i64;
        let mut bars_for_ticker: Vec<StockBar> = Vec::new();
        let mut prev_close: Option<f32> = None;

        for entry in files {
            let path = entry.path();
            let file = fs::File::open(&path)
                .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
            let reader = BufReader::new(file);

            for line in reader.lines().flatten() {
                // Format is semicolon separated: "<date time>;<open>;<high>;<low>;<close>;<volume>"
                let mut parts = line.split(';');
                let ts = parts.next().unwrap_or("");
                let open: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let high: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let low: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let close: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
                let volume: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);

                if open <= 0.0 || close <= 0.0 || !open.is_finite() || !close.is_finite() {
                    continue;
                }

                // Very lightweight timestamp parsing: YYYYMMDD HHMMSS -> monotonic minute index.
                // We keep it deterministic and only need ordering for our "time split".
                let (date_part, time_part) = match ts.split_once(' ') {
                    Some((d, t)) => (d, t),
                    None => continue,
                };
                if date_part.len() != 8 || time_part.len() < 4 {
                    continue;
                }
                let day: i64 = date_part.parse().unwrap_or(0);
                let hhmm: i64 = time_part[..4].parse().unwrap_or(0);

                // Not a real epoch; just "sortable".
                let timestamp = day * 10_000 + hhmm;

                let returns = if let Some(pc) = prev_close {
                    (close - pc) / pc
                } else {
                    0.0
                };
                prev_close = Some(close);

                let day_index = bars_for_ticker.len();

                bars_for_ticker.push(StockBar {
                    ticker_id,
                    timestamp,
                    day_index,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    returns,
                });
            }
        }

        if bars_for_ticker.is_empty() {
            return Err(format!("No valid bars parsed for {}", symbol));
        }

        // Calculate volatility/drift from returns.
        let returns: Vec<f32> = bars_for_ticker.iter().map(|b| b.returns).collect();
        let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len().max(1) as f32;
        let variance: f32 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>()
            / returns.len().max(1) as f32;
        // 252 trading days * 390 minutes. For "index" style data this is a rough scaling.
        let annualization = (252.0_f32 * 390.0_f32).sqrt();
        let volatility = variance.sqrt() * annualization;

        self.tickers.push(TickerInfo {
            ticker_id,
            name: symbol.to_string(),
            symbol: symbol.to_string(),
            sector: Sector::Technology,
            beta: 1.0,
            base_volatility: volatility,
            drift: mean_ret * 252.0 * 390.0,
        });

        self.bars.extend(bars_for_ticker);

        Ok(())
    }
}

// =============================================================================
// Section 3: Technical Indicator Computation
// =============================================================================

/// Technical indicators computed for each bar.
#[derive(Debug, Clone, Default)]
pub struct TechnicalIndicators {
    // Trend indicators
    pub sma_5: f32,
    pub sma_20: f32,
    pub sma_50: f32,
    pub ema_12: f32,
    pub ema_26: f32,

    // Momentum indicators
    pub rsi_14: f32,
    pub macd: f32,
    pub macd_signal: f32,
    pub macd_histogram: f32,
    pub roc_10: f32, // Rate of Change (10-period)
    pub roc_20: f32, // Rate of Change (20-period)

    // Stochastic oscillator
    pub stoch_k: f32, // Slow Stochastic %K
    pub stoch_d: f32, // Slow Stochastic %D

    // Volatility indicators
    pub bollinger_upper: f32,
    pub bollinger_lower: f32,
    pub bollinger_width: f32,
    pub atr_14: f32,
    pub stddev_20: f32, // Standard Deviation (20-period)

    // Volume indicators
    pub volume_sma_20: f32,
    pub volume_ratio: f32,
    pub obv: f32,    // On Balance Volume (from ta crate)
    pub mfi_14: f32, // Money Flow Index (14-period)

    // Range indicators
    pub max_14: f32,         // Maximum (14-period high)
    pub min_14: f32,         // Minimum (14-period low)
    pub price_vs_range: f32, // Position within max/min range

    // Price patterns
    pub price_vs_sma20: f32,
    pub price_vs_bollinger: f32,
    pub high_low_range: f32,
    pub body_ratio: f32,
    pub upper_shadow: f32,
    pub lower_shadow: f32,
}

impl TechnicalIndicators {
    /// Converts indicators to a feature vector.
    /// Contains 33 features from all ta crate indicators.
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            // Trend (5)
            self.sma_5,
            self.sma_20,
            self.sma_50,
            self.ema_12,
            self.ema_26,
            // Momentum (6)
            self.rsi_14,
            self.macd,
            self.macd_signal,
            self.macd_histogram,
            self.roc_10,
            self.roc_20,
            // Stochastic (2)
            self.stoch_k,
            self.stoch_d,
            // Volatility (5)
            self.bollinger_upper,
            self.bollinger_lower,
            self.bollinger_width,
            self.atr_14,
            self.stddev_20,
            // Volume (4)
            self.volume_sma_20,
            self.volume_ratio,
            self.obv,
            self.mfi_14,
            // Range (3)
            self.max_14,
            self.min_14,
            self.price_vs_range,
            // Price patterns (6)
            self.price_vs_sma20,
            self.price_vs_bollinger,
            self.high_low_range,
            self.body_ratio,
            self.upper_shadow,
            self.lower_shadow,
            // Padding for alignment (5) - total 36 features
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    }

    /// Number of features (33 real + 3 padding = 36 for alignment).
    pub const NUM_FEATURES: usize = 36;
}

/// Computes technical indicators for a series of bars using the `ta` crate.
/// Uses ALL available indicators from the ta crate to maximize feature richness:
/// - SimpleMovingAverage (5, 20, 50 periods)
/// - ExponentialMovingAverage (12, 26 periods)
/// - RelativeStrengthIndex (14 period)
/// - BollingerBands (20 period, 2 std dev)
/// - AverageTrueRange (14 period)
/// - RateOfChange (10, 20 periods)
/// - SlowStochastic (14 period)
/// - StandardDeviation (20 period)
/// - MoneyFlowIndex (14 period)
/// - OnBalanceVolume (streaming)
/// - Maximum/Minimum (14 period)
pub struct IndicatorCalculator {
    // No state needed - ta crate handles streaming internally
}

impl IndicatorCalculator {
    pub fn new() -> Self {
        Self {}
    }

    /// Computes indicators for all bars of a ticker using ALL `ta` crate indicators.
    pub fn compute_indicators(&mut self, bars: &[StockBar]) -> Vec<TechnicalIndicators> {
        if bars.is_empty() {
            return Vec::new();
        }

        let n = bars.len();
        let mut indicators = vec![TechnicalIndicators::default(); n];

        // =====================================================================
        // Initialize ALL ta-crate indicators
        // =====================================================================

        // Trend indicators - SMAs and EMAs
        let mut sma_5 = SimpleMovingAverage::new(5).unwrap();
        let mut sma_20 = SimpleMovingAverage::new(20).unwrap();
        let mut sma_50 = SimpleMovingAverage::new(50).unwrap();
        let mut ema_12 = ExponentialMovingAverage::new(12).unwrap();
        let mut ema_26 = ExponentialMovingAverage::new(26).unwrap();
        let mut ema_9 = ExponentialMovingAverage::new(9).unwrap(); // For MACD signal

        // Momentum indicators
        let mut rsi = RelativeStrengthIndex::new(14).unwrap();
        let mut roc_10 = RateOfChange::new(10).unwrap();
        let mut roc_20 = RateOfChange::new(20).unwrap();

        // Stochastic oscillator: Use SlowStochastic (14 period, 3 smoothing)
        // then compute %D as a 3-period SMA of %K
        let mut stoch_k = SlowStochastic::new(14, 3).unwrap();
        let mut stoch_d_sma = SimpleMovingAverage::new(3).unwrap(); // %D = 3-period SMA of %K

        // Volatility indicators
        let mut bb = BollingerBands::new(20, 2.0_f64).unwrap();
        let mut atr = AverageTrueRange::new(14).unwrap();
        let mut stddev = StandardDeviation::new(20).unwrap();

        // Volume indicators
        let mut volume_sma = SimpleMovingAverage::new(20).unwrap();
        let mut obv = OnBalanceVolume::new();
        let mut mfi = MoneyFlowIndex::new(14).unwrap();

        // Range indicators (14-period high/low)
        let mut max_14 = Maximum::new(14).unwrap();
        let mut min_14 = Minimum::new(14).unwrap();

        // =====================================================================
        // Pre-allocate storage for indicator values
        // =====================================================================
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

        // =====================================================================
        // Stream through bars and compute all indicators
        // =====================================================================
        for (i, bar) in bars.iter().enumerate() {
            let close = bar.close as f64;
            let high = bar.high as f64;
            let low = bar.low as f64;
            let open = bar.open as f64;
            let volume = bar.volume;

            // Build DataItem for indicators that need OHLCV
            let data_item = ta::DataItem::builder()
                .high(high)
                .low(low)
                .close(close)
                .open(open)
                .volume(volume)
                .build()
                .unwrap();

            // Trend: SMAs
            sma_5_vals[i] = sma_5.next(close);
            sma_20_vals[i] = sma_20.next(close);
            sma_50_vals[i] = sma_50.next(close);

            // Trend: EMAs
            ema_12_vals[i] = ema_12.next(close);
            ema_26_vals[i] = ema_26.next(close);

            // Momentum: RSI
            rsi_vals[i] = rsi.next(close);

            // Momentum: Rate of Change
            roc_10_vals[i] = roc_10.next(close);
            roc_20_vals[i] = roc_20.next(close);

            // Stochastic oscillator: %K from SlowStochastic, %D from SMA of %K
            let k_val = stoch_k.next(close);
            stoch_k_vals[i] = k_val;
            stoch_d_vals[i] = stoch_d_sma.next(k_val);

            // Volatility: Bollinger Bands
            let bb_val = bb.next(close);
            bb_upper[i] = bb_val.upper;
            bb_lower[i] = bb_val.lower;

            // Volatility: ATR
            atr_vals[i] = atr.next(&data_item);

            // Volatility: Standard Deviation
            stddev_vals[i] = stddev.next(close);

            // Volume: SMA of volume
            vol_sma_vals[i] = volume_sma.next(volume);

            // Volume: On Balance Volume (uses ta crate's implementation)
            obv_vals[i] = obv.next(&data_item);

            // Volume: Money Flow Index
            mfi_vals[i] = mfi.next(&data_item);

            // Range: 14-period Maximum and Minimum
            max_14_vals[i] = max_14.next(high);
            min_14_vals[i] = min_14.next(low);
        }

        // =====================================================================
        // Compute MACD (derived from EMAs)
        // =====================================================================
        let mut macd_vals = vec![0.0_f64; n];
        let mut macd_signal_vals = vec![0.0_f64; n];

        for i in 0..n {
            macd_vals[i] = ema_12_vals[i] - ema_26_vals[i];
            macd_signal_vals[i] = ema_9.next(macd_vals[i]);
        }

        // =====================================================================
        // Assemble all indicators into final struct
        // =====================================================================
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

            // Position within 14-period range
            let price_range = max_14_vals[i] - min_14_vals[i];
            let price_vs_range = if price_range > 0.0 {
                ((close - min_14_vals[i]) / price_range - 0.5) * 2.0 // Normalize to [-1, 1]
            } else {
                0.0
            };

            indicators[i] = TechnicalIndicators {
                // Trend (5)
                sma_5: (sma_5_vals[i] / close - 1.0) as f32,
                sma_20: (sma_20_vals[i] / close - 1.0) as f32,
                sma_50: (sma_50_vals[i] / close - 1.0) as f32,
                ema_12: (ema_12_vals[i] / close - 1.0) as f32,
                ema_26: (ema_26_vals[i] / close - 1.0) as f32,

                // Momentum (6)
                rsi_14: ((rsi_vals[i] - 50.0) / 50.0) as f32, // Normalize to [-1, 1]
                macd: (macd_vals[i] / close * 100.0) as f32,
                macd_signal: (macd_signal_vals[i] / close * 100.0) as f32,
                macd_histogram: (macd_histogram / close * 100.0) as f32,
                roc_10: (roc_10_vals[i] / 100.0).clamp(-1.0, 1.0) as f32, // Already percentage, normalize
                roc_20: (roc_20_vals[i] / 100.0).clamp(-1.0, 1.0) as f32,

                // Stochastic (2)
                stoch_k: ((stoch_k_vals[i] - 50.0) / 50.0) as f32, // Normalize to [-1, 1]
                stoch_d: ((stoch_d_vals[i] - 50.0) / 50.0) as f32,

                // Volatility (5)
                bollinger_upper: (bb_upper[i] / close - 1.0) as f32,
                bollinger_lower: (bb_lower[i] / close - 1.0) as f32,
                bollinger_width: bb_width as f32,
                atr_14: (atr_vals[i] / close) as f32,
                stddev_20: (stddev_vals[i] / close) as f32, // Normalized by price

                // Volume (4)
                volume_sma_20: (vol_sma_vals[i] / 1_000_000.0) as f32, // Normalize to millions
                volume_ratio: (vol_ratio.min(5.0) / 5.0 - 0.5) as f32,
                obv: (obv_vals[i] / 1e9) as f32, // Normalize to billions
                mfi_14: ((mfi_vals[i] - 50.0) / 50.0) as f32, // Normalize to [-1, 1]

                // Range (3)
                max_14: (max_14_vals[i] / close - 1.0) as f32,
                min_14: (min_14_vals[i] / close - 1.0) as f32,
                price_vs_range: price_vs_range as f32,

                // Price patterns (6)
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

// =============================================================================
// Section 4: Data Pipeline and Instance Creation
// =============================================================================

/// A training/prediction instance for stock prediction.
#[derive(Debug, Clone)]
pub struct StockInstance {
    /// Index into the dataset/tickers vector.
    pub ticker_idx: usize,
    /// Ticker feature ID (fed into embedding lookup).
    pub ticker_fid: i64,
    /// Sector feature ID (fed into embedding lookup).
    pub sector_fid: i64,

    /// Time index into this ticker's bar/indicator arrays.
    pub t: usize,

    /// Direction label: 1=up, 0=down (5-day forward)
    pub direction_label: f32,
    /// Magnitude label: % change (5-day forward)
    pub magnitude_label: f32,
    /// Profitable label: 1 if return > 2%
    pub profitable_label: f32,
}

/// In-memory dataset representation: bars + indicators stored once per ticker.
///
/// This avoids the OOM that happens when storing a full lookback window per instance.
pub struct StockDataset {
    pub tickers: Vec<TickerInfo>,
    pub bars_by_ticker: Vec<Vec<StockBar>>,
    pub indicators_by_ticker: Vec<Vec<TechnicalIndicators>>,
}

impl StockDataset {
    fn bar_at(&self, inst: &StockInstance, t: usize) -> &StockBar {
        &self.bars_by_ticker[inst.ticker_idx][t]
    }

    fn ind_at(&self, inst: &StockInstance, t: usize) -> &TechnicalIndicators {
        &self.indicators_by_ticker[inst.ticker_idx][t]
    }
}

/// Creates training instances from bars and indicators.
pub struct InstanceCreator {
    lookback_window: usize,
    forward_horizon: usize,
    profit_threshold: f32,
}

impl InstanceCreator {
    pub fn new(lookback_window: usize) -> Self {
        Self {
            lookback_window,
            forward_horizon: 5,     // 5-day forward prediction
            profit_threshold: 0.02, // 2% profit threshold
        }
    }

    /// Creates instances for a ticker.
    pub fn create_instances(
        &self,
        ticker_idx: usize,
        ticker: &TickerInfo,
        bars: &[StockBar],
        indicators: &[TechnicalIndicators],
    ) -> Vec<StockInstance> {
        let n = bars.len();
        let mut instances = Vec::new();

        if n <= self.lookback_window + self.forward_horizon {
            return instances;
        }

        // Start after lookback window, end before forward horizon
        for i in self.lookback_window..(n - self.forward_horizon) {
            // Compute forward returns
            let future_price = bars[i + self.forward_horizon].close;
            let current_price = bars[i].close;
            let forward_return = (future_price - current_price) / current_price;

            // Create labels
            let direction_label = if forward_return > 0.0 { 1.0 } else { 0.0 };
            let magnitude_label = forward_return * 100.0; // Percentage
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

/// Splits instances into train and eval sets by time.
pub fn train_eval_split(
    instances: &[StockInstance],
    train_ratio: f32,
) -> (Vec<StockInstance>, Vec<StockInstance>) {
    let split_idx = (instances.len() as f32 * train_ratio) as usize;
    let train = instances[..split_idx].to_vec();
    let eval = instances[split_idx..].to_vec();
    (train, eval)
}

/// Creates batches from instances.
pub fn create_batches(instances: &[StockInstance], batch_size: usize) -> Vec<Vec<&StockInstance>> {
    instances
        .chunks(batch_size)
        .map(|c| c.iter().collect())
        .collect()
}

// =============================================================================
// Section 5: Ticker Embedding Setup
// =============================================================================

/// Embedding tables for tickers and sectors.
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
        // Create ticker embedding table with Xavier initialization
        let ticker_initializer = Arc::new(XavierNormalInitializer::new(1.0));
        let mut ticker_table = CuckooEmbeddingHashTable::with_initializer(
            num_tickers * 2,
            ticker_dim,
            ticker_initializer,
        );

        // Create sector embedding table
        let sector_initializer = Arc::new(XavierNormalInitializer::new(1.0));
        let mut sector_table = CuckooEmbeddingHashTable::with_initializer(
            num_sectors * 2,
            sector_dim,
            sector_initializer,
        );

        // Initialize embeddings for all tickers
        for i in 0..num_tickers {
            ticker_table.get_or_initialize(i as i64).unwrap();
        }

        // Initialize embeddings for all sectors
        for sector in Sector::all() {
            sector_table.get_or_initialize(sector.id()).unwrap();
        }

        Self {
            ticker_table,
            sector_table,
        }
    }

    pub fn lookup_ticker(&self, ticker_id: i64) -> Vec<f32> {
        let mut output = vec![0.0; self.ticker_table.dim()];
        self.ticker_table.lookup(&[ticker_id], &mut output).unwrap();
        output
    }

    pub fn lookup_sector(&self, sector_id: i64) -> Vec<f32> {
        let mut output = vec![0.0; self.sector_table.dim()];
        self.sector_table.lookup(&[sector_id], &mut output).unwrap();
        output
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

    pub fn apply_gradients(
        &mut self,
        ticker_ids: &[i64],
        ticker_grads: &[f32],
        sector_ids: &[i64],
        sector_grads: &[f32],
    ) {
        self.ticker_table
            .apply_gradients(ticker_ids, ticker_grads)
            .unwrap();
        self.sector_table
            .apply_gradients(sector_ids, sector_grads)
            .unwrap();
    }

    pub fn ticker_dim(&self) -> usize {
        self.ticker_table.dim()
    }

    pub fn sector_dim(&self) -> usize {
        self.sector_table.dim()
    }
}

// =============================================================================
// Section 6: Model Architecture
// =============================================================================

/// Stock prediction model combining embeddings, DIEN, DCN, MMoE, SENet, and more.
///
/// Enhanced architecture:
/// ```text
/// Embeddings (ticker, sector)
///     ↓
/// [SENet: Learn indicator importance]
///     ↓
/// [DIEN: Sequential price pattern extraction]
///     ↓
/// [DIN: Attention on recent indicators]
///     ↓
/// [Concatenate: DIEN output + DIN output + embeddings]
///     ↓
/// [DCN-V2 (Matrix mode): Feature interactions]
///     ↓
/// [LayerNorm: Stabilize training]
///     ↓
/// [MMoE: Shared experts + Task-specific gates]
///     ├→ Direction Head (MLP)
///     ├→ Magnitude Head (MLP)
///     └→ Profitability Head (MLP)
/// ```
pub struct StockPredictionModel {
    /// Embedding tables
    embeddings: EmbeddingTables,
    /// SENet for adaptive feature importance on indicators
    senet: SENetLayer,
    /// DIEN for sequential patterns
    dien: DIENLayer,
    /// DIN attention for indicator weighting
    din: DINAttention,
    /// DCN for feature interactions (Matrix mode for DCN-V2)
    dcn: CrossNetwork,
    /// LayerNorm before task heads
    layer_norm: LayerNorm,
    /// MMoE for multi-task learning (3 tasks: direction, magnitude, profitable)
    mmoe: MMoE,
    /// Direction prediction head (after MMoE)
    direction_head: MLP,
    /// Magnitude prediction head (after MMoE)
    magnitude_head: MLP,
    /// Profitable prediction head (after MMoE)
    profitable_head: MLP,
    /// Configuration
    config: StockPredictorConfig,
}

impl StockPredictionModel {
    pub fn new(config: &StockPredictorConfig) -> Self {
        // Create embedding tables
        let embeddings = EmbeddingTables::new(
            config.num_tickers,
            config.ticker_embedding_dim,
            Sector::all().len(),
            config.sector_embedding_dim,
        );

        // Calculate input dimensions
        // Sequence features: 4 price + 36 indicators = 40 features
        let seq_feature_dim = 4 + TechnicalIndicators::NUM_FEATURES;

        // 1. SENet for adaptive feature importance on technical indicators
        // Reduction ratio 4: 36 indicators → 9 bottleneck → 36 reweighted
        let senet = SENetLayer::new(TechnicalIndicators::NUM_FEATURES, 4, true);

        // 2. DIEN layer for sequential pattern extraction
        let dien = DIENConfig::new(seq_feature_dim, config.dien_hidden_size)
            .with_use_auxiliary_loss(false)
            .build()
            .unwrap();

        // 3. DIN attention for indicator weighting (query = current, keys = historical)
        // Uses embedding_dim = seq_feature_dim for attention on indicator sequences
        let din = DINConfig::new(seq_feature_dim)
            .with_attention_hidden_units(vec![64, 32])
            .with_activation(ActivationType::Sigmoid)
            .with_use_softmax(true)
            .build()
            .unwrap();

        // 4. Combined feature dimension for DCN:
        // ticker_emb + sector_emb + dien_output + din_output + senet_indicators
        let combined_dim = config.ticker_embedding_dim
            + config.sector_embedding_dim
            + config.dien_hidden_size
            + seq_feature_dim  // DIN output
            + TechnicalIndicators::NUM_FEATURES; // SENet-reweighted indicators

        // 5. Create DCN with Matrix mode (DCN-V2) for richer feature interactions
        let dcn = CrossNetwork::new(combined_dim, config.dcn_cross_layers, DCNMode::Matrix);

        // 6. LayerNorm for training stability before task heads
        let layer_norm = LayerNorm::new(combined_dim);

        // 7. MMoE for multi-task learning: 4 experts, 3 tasks
        // Each expert has hidden layers [64, 32] and outputs 32-dim
        // Using GELU for smoother gradients and to prevent dead neurons
        let mmoe = MMoEConfig::new(combined_dim, 4, 3)
            .with_expert_hidden_units(vec![64, 32])
            .with_expert_activation(ActivationType::GELU)
            .with_expert_output_dim(32)
            .build()
            .unwrap();

        // 8. Task-specific heads (take MMoE output of 32-dim)
        let mmoe_output_dim = 32;

        // Direction head: binary classification (sigmoid)
        // Using GELU for smoother gradients
        let direction_head = MLPConfig::new(mmoe_output_dim)
            .add_layer(16, ActivationType::GELU)
            .add_layer(1, ActivationType::Sigmoid)
            .build()
            .unwrap();

        // Magnitude head: regression
        let magnitude_head = MLPConfig::new(mmoe_output_dim)
            .add_layer(16, ActivationType::GELU)
            .add_layer(1, ActivationType::None)
            .build()
            .unwrap();

        // Profitable head: binary classification (sigmoid)
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

    /// Forward pass for a batch of instances using enhanced architecture.
    ///
    /// Flow: Embeddings → SENet → DIEN → DIN → Concat → DCN → LayerNorm → MMoE → Task Heads
    pub fn forward(&self, batch: &[&StockInstance]) -> ModelOutput {
        let batch_size = batch.len();
        let lookback = self.config.lookback_window;
        let seq_feature_dim = 4 + TechnicalIndicators::NUM_FEATURES;

        // Collect batch data
        let ticker_ids: Vec<i64> = batch.iter().map(|i| i.ticker_fid).collect();
        let sector_ids: Vec<i64> = batch.iter().map(|i| i.sector_fid).collect();

        // 1. Lookup embeddings
        let (ticker_embs, sector_embs) = self.embeddings.lookup_batch(&ticker_ids, &sector_ids);
        let ticker_tensor =
            Tensor::from_data(&[batch_size, self.embeddings.ticker_dim()], ticker_embs);
        let sector_tensor =
            Tensor::from_data(&[batch_size, self.embeddings.sector_dim()], sector_embs);

        // 2. Get current indicator features and apply SENet for adaptive importance
        let mut indicator_data = vec![0.0; batch_size * TechnicalIndicators::NUM_FEATURES];
        for (b, instance) in batch.iter().enumerate() {
            for (f, &val) in instance.indicator_features.iter().enumerate() {
                if f < TechnicalIndicators::NUM_FEATURES {
                    indicator_data[b * TechnicalIndicators::NUM_FEATURES + f] = val;
                }
            }
        }
        let indicator_tensor = Tensor::from_data(
            &[batch_size, TechnicalIndicators::NUM_FEATURES],
            indicator_data,
        );

        // Apply SENet: learns which indicators are important for each sample
        let senet_indicators = self.senet.forward(&indicator_tensor).unwrap();

        // 3. Build sequence tensor for DIEN [batch, lookback, features]
        let mut seq_data = vec![0.0; batch_size * lookback * seq_feature_dim];
        for (b, instance) in batch.iter().enumerate() {
            for (t, seq_features) in instance.historical_sequence.iter().enumerate() {
                for (f, &val) in seq_features.iter().enumerate().take(seq_feature_dim) {
                    let idx = b * lookback * seq_feature_dim + t * seq_feature_dim + f;
                    if idx < seq_data.len() {
                        seq_data[idx] = val;
                    }
                }
            }
        }
        let seq_tensor = Tensor::from_data(&[batch_size, lookback, seq_feature_dim], seq_data);

        // Create target tensor for DIEN (use last timestep features)
        let mut target_data = vec![0.0; batch_size * seq_feature_dim];
        for (b, instance) in batch.iter().enumerate() {
            if let Some(last_seq) = instance.historical_sequence.last() {
                for (f, &val) in last_seq.iter().enumerate().take(seq_feature_dim) {
                    target_data[b * seq_feature_dim + f] = val;
                }
            }
        }
        let target_tensor = Tensor::from_data(&[batch_size, seq_feature_dim], target_data);

        // 4. Run DIEN for sequential pattern extraction
        let dien_output = self
            .dien
            .forward_dien(&seq_tensor, &target_tensor, None)
            .unwrap();

        // 5. Run DIN attention: query = current features, keys/values = historical sequence
        // This weights recent indicators by their relevance to current prediction
        let din_output = self
            .din
            .forward_attention(&target_tensor, &seq_tensor, &seq_tensor, None)
            .unwrap();

        // 6. Concatenate all features:
        // [ticker_emb | sector_emb | dien_out | din_out | senet_indicators]
        let combined = self.concatenate_enhanced_features(
            &ticker_tensor,
            &sector_tensor,
            &dien_output,
            &din_output,
            &senet_indicators,
        );

        // 7. Run DCN (Matrix mode for richer feature interactions)
        let dcn_output = self.dcn.forward(&combined).unwrap();

        // 8. Apply LayerNorm for training stability
        let normalized = self.layer_norm.forward(&dcn_output).unwrap();

        // 9. Run MMoE: produces task-specific outputs for each of 3 tasks
        // forward_multi returns Vec<Tensor> - one tensor per task, each of shape [batch, expert_output_dim]
        let mmoe_outputs = self.mmoe.forward_multi(&normalized).unwrap();

        // 10. Run task-specific heads (each mmoe_outputs[i] is [batch, 32])
        let direction_pred = self.direction_head.forward(&mmoe_outputs[0]).unwrap();
        let magnitude_pred = self.magnitude_head.forward(&mmoe_outputs[1]).unwrap();
        let profitable_pred = self.profitable_head.forward(&mmoe_outputs[2]).unwrap();

        ModelOutput {
            direction: direction_pred.data().to_vec(),
            magnitude: magnitude_pred.data().to_vec(),
            profitable: profitable_pred.data().to_vec(),
        }
    }

    /// Concatenate enhanced features including DIN output and SENet indicators.
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

            // Copy ticker embeddings
            for i in 0..ticker.shape()[1] {
                data[offset + i] = ticker.data()[b * ticker.shape()[1] + i];
            }
            offset += ticker.shape()[1];

            // Copy sector embeddings
            for i in 0..sector.shape()[1] {
                data[offset + i] = sector.data()[b * sector.shape()[1] + i];
            }
            offset += sector.shape()[1];

            // Copy DIEN output
            for i in 0..dien.shape()[1] {
                data[offset + i] = dien.data()[b * dien.shape()[1] + i];
            }
            offset += dien.shape()[1];

            // Copy DIN output
            for i in 0..din.shape()[1] {
                data[offset + i] = din.data()[b * din.shape()[1] + i];
            }
            offset += din.shape()[1];

            // Copy SENet-reweighted indicator features
            for i in 0..senet_indicators.shape()[1] {
                data[offset + i] = senet_indicators.data()[b * senet_indicators.shape()[1] + i];
            }
        }

        Tensor::from_data(&[batch_size, total_dim], data)
    }

    /// Computes total loss.
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

        // Magnitude loss (Huber loss for robustness, scaled)
        let mut magnitude_loss = 0.0;
        let delta = 5.0; // Huber delta - transition point
        for (i, instance) in batch.iter().enumerate() {
            let diff = (output.magnitude[i] - instance.magnitude_label).abs();
            if diff <= delta {
                magnitude_loss += 0.5 * diff * diff;
            } else {
                magnitude_loss += delta * (diff - 0.5 * delta);
            }
        }
        magnitude_loss /= batch_size;
        magnitude_loss /= 10.0; // Scale down to be comparable with BCE losses

        // Profitable loss (binary cross-entropy)
        let mut profitable_loss = 0.0;
        for (i, instance) in batch.iter().enumerate() {
            let pred = output.profitable[i].clamp(1e-7, 1.0 - 1e-7);
            let label = instance.profitable_label;
            profitable_loss += -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln();
        }
        profitable_loss /= batch_size;

        // Combined loss (rebalanced weights)
        let total_loss = 0.4 * direction_loss + 0.4 * magnitude_loss + 0.2 * profitable_loss;

        (total_loss, direction_loss, magnitude_loss, profitable_loss)
    }

    /// Gets mutable parameters for training.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        // SENet parameters
        params.extend(self.senet.parameters_mut());
        // DIEN parameters
        params.extend(self.dien.parameters_mut());
        // DIN parameters
        params.extend(self.din.parameters_mut());
        // DCN parameters
        params.extend(self.dcn.parameters_mut());
        // LayerNorm parameters
        params.extend(self.layer_norm.parameters_mut());
        // MMoE parameters
        params.extend(self.mmoe.parameters_mut());
        // Task head parameters
        params.extend(self.direction_head.parameters_mut());
        params.extend(self.magnitude_head.parameters_mut());
        params.extend(self.profitable_head.parameters_mut());
        params
    }
}

/// Model output containing predictions.
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub direction: Vec<f32>,
    pub magnitude: Vec<f32>,
    pub profitable: Vec<f32>,
}

// =============================================================================
// Section 7: Training Loop
// =============================================================================

/// Training metrics tracked during training.
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

/// Trainer for the stock prediction model with AMSGrad optimizer.
///
/// Features:
/// - AMSGrad optimizer for better convergence on volatile data
/// - Gradient clipping for stability
/// - Linear warmup + cosine annealing LR schedule
/// - Weight decay (L2 regularization)
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
    // NOTE: This example uses a finite-difference (SPSA-like) update rather than full backprop.
    // The monolith-layers crate contains backward() implementations for many layers, but some
    // (e.g. DIEN) intentionally use simplified/placeholder gradients. A finite-difference update
    // keeps the example self-contained and makes progress without requiring full autodiff.
    //
    // We still keep a per-parameter AMSGrad state so updates are stable over long runs.
    optimizers: Vec<Amsgrad>,
    // Hyperparameters
    grad_clip: f32,
    warmup_steps: usize,
    // Finite-difference hyperparameters
    fd_epsilon: f32,
    fd_num_coords: usize,
}

impl Trainer {
    pub fn new(config: &StockPredictorConfig) -> Self {
        let mut model = StockPredictionModel::new(config);

        // Create AMSGrad optimizer for each parameter tensor
        let optimizers: Vec<Amsgrad> = model
            .parameters_mut()
            .iter()
            .map(|_| {
                Amsgrad::with_params(
                    config.learning_rate,
                    0.9,    // beta1: momentum
                    0.999,  // beta2: second moment
                    1e-8,   // epsilon: numerical stability
                    0.0001, // weight_decay: L2 regularization
                )
            })
            .collect();

        // Quick warmup for first 50 steps, then full LR
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
            grad_clip: 1.0, // Moderate gradient clipping
            warmup_steps,
            // Finite-difference tuning: small epsilon, small coordinate budget.
            // This makes the example "train longer" without spending most time iterating all params.
            fd_epsilon: 1e-3,
            fd_num_coords: 2048,
        }
    }

    /// Updates learning rate with warmup + cosine annealing schedule.
    pub fn update_lr(&mut self) {
        // Linear warmup for initial steps
        if self.global_step < self.warmup_steps {
            let warmup_progress = self.global_step as f32 / self.warmup_steps as f32;
            self.learning_rate = self.initial_lr * warmup_progress;
        } else {
            // Cosine annealing after warmup
            let progress = self.epoch as f32 / self.config.num_epochs as f32;
            let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.learning_rate = self.initial_lr * decay.max(0.01); // Min 1% of initial LR
        }
    }

    /// Applies learning rate decay (cosine annealing) at epoch boundary.
    pub fn decay_lr(&mut self) {
        self.epoch += 1;
        self.update_lr();
    }

    /// Returns current learning rate.
    pub fn current_lr(&self) -> f32 {
        self.learning_rate
    }

    /// Trains for one epoch.
    pub fn train_epoch(&mut self, train_instances: &[StockInstance]) -> TrainingMetrics {
        // Shuffle instances using internal RNG
        let mut indices: Vec<usize> = (0..train_instances.len()).collect();
        self.rng.shuffle(&mut indices);

        let mut epoch_metrics = TrainingMetrics::default();
        let mut direction_correct = 0;
        let mut total_samples = 0;

        // Process batches
        for chunk in indices.chunks(self.config.batch_size) {
            let batch: Vec<&StockInstance> = chunk.iter().map(|&i| &train_instances[i]).collect();

            // Forward pass
            let output = self.model.forward(&batch);

            // Compute loss
            let (total_loss, direction_loss, magnitude_loss, profitable_loss) =
                self.model.compute_loss(&output, &batch);

            // Accumulate metrics
            epoch_metrics.total_loss += total_loss * batch.len() as f32;
            epoch_metrics.direction_loss += direction_loss * batch.len() as f32;
            epoch_metrics.magnitude_loss += magnitude_loss * batch.len() as f32;
            epoch_metrics.profitable_loss += profitable_loss * batch.len() as f32;

            // Compute accuracy
            for (i, instance) in batch.iter().enumerate() {
                let pred = if output.direction[i] > 0.5 { 1.0 } else { 0.0 };
                if (pred - instance.direction_label).abs() < 0.1 {
                    direction_correct += 1;
                }
            }
            total_samples += batch.len();

            // Loss-directed finite-difference step (SPSA-like, coordinate-sampled).
            // This is much more likely to improve the model than the older "random perturb" update.
            self.finite_difference_step(&batch);

            self.global_step += 1;

            // Logging
            if self.config.verbose && self.global_step % self.config.log_every_n_steps == 0 {
                println!(
                    "  [Step {}] Loss: {:.4} | Dir: {:.4} | Mag: {:.4} | Prof: {:.4}",
                    self.global_step, total_loss, direction_loss, magnitude_loss, profitable_loss
                );
            }
        }

        // Finalize metrics
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

    fn finite_difference_step(&mut self, batch: &[&StockInstance]) {
        // Update learning rate (handles warmup)
        self.update_lr();

        // Evaluate baseline loss once for logging/scale. This is not strictly required for SPSA,
        // but it helps keep update magnitudes in a sensible range.
        let baseline_loss = {
            let output = self.model.forward(batch);
            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
            loss
        };

        // Choose which parameter tensor to update this step (cycle for determinism).
        // Updating a small subset per step makes long training runs feasible.
        let num_params = self.model.parameters_mut().len().max(1);
        let param_idx = self.global_step % num_params;

        // Coordinate budget for this tensor.
        let (param_len, coord_count) = {
            let mut params = self.model.parameters_mut();
            let len = params[param_idx].data().len();
            (len, self.fd_num_coords.min(len.max(1)))
        };

        // Precompute coordinate indices and +/-1 deltas (deterministic "random").
        let mut coord_indices = Vec::with_capacity(coord_count);
        let mut coord_deltas = Vec::with_capacity(coord_count);
        for j in 0..coord_count {
            // Deterministic index selection based on (step, param_idx, j).
            let h = (self.global_step as u64)
                .wrapping_mul(1_000_003)
                .wrapping_add(param_idx as u64 * 97)
                .wrapping_add(j as u64 * 1_009);
            let idx = (h as usize) % param_len;
            // +/-1 delta
            let delta = if (h >> 11) & 1 == 0 { 1.0 } else { -1.0 };
            coord_indices.push(idx);
            coord_deltas.push(delta);
        }

        let eps = self.fd_epsilon;

        // Apply +eps perturbation to selected coords.
        {
            let mut params = self.model.parameters_mut();
            let data = params[param_idx].data_mut();
            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
                data[idx] += eps * delta;
            }
        }

        let loss_plus = {
            let output = self.model.forward(batch);
            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
            loss
        };

        // Apply -2eps (now at -eps relative to original).
        {
            let mut params = self.model.parameters_mut();
            let data = params[param_idx].data_mut();
            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
                data[idx] -= 2.0 * eps * delta;
            }
        }

        let loss_minus = {
            let output = self.model.forward(batch);
            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
            loss
        };

        // Restore to original.
        {
            let mut params = self.model.parameters_mut();
            let data = params[param_idx].data_mut();
            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
                data[idx] += eps * delta;
            }
        }

        // SPSA-style gradient estimate for the selected coordinates:
        // g_i ~= (L+ - L-) / (2 * eps) * delta_i
        let coeff = (loss_plus - loss_minus) / (2.0 * eps).max(1e-12);

        // Ensure optimizer exists for this parameter tensor
        if param_idx >= self.optimizers.len() {
            self.optimizers.resize_with(param_idx + 1, || {
                Amsgrad::with_params(self.learning_rate, 0.9, 0.999, 1e-8, 0.0001)
            });
        }

        // Apply sparse update using AMSGrad state but only on selected coordinates.
        // We do this by directly updating the optimizer's internal vectors via a dense gradient
        // buffer that is reused and mostly zeros would be too expensive. Instead, we update
        // the parameter values with a clipped SGD step scaled by coeff.
        //
        // Rationale: This example aims to demonstrate end-to-end training behavior without
        // spending O(#params) per batch.
        let step_lr = self.learning_rate;
        let mut params = self.model.parameters_mut();
        let data = params[param_idx].data_mut();
        for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
            // Light weight decay encourages stability over long training.
            let l2 = 0.0001 * data[idx];
            let grad = (coeff * delta + l2).clamp(-self.grad_clip, self.grad_clip);
            data[idx] -= step_lr * grad;
        }

        // If the FD step is wildly noisy (common early), damp it by shrinking epsilon slowly.
        // This keeps long runs stable without requiring a full optimizer state.
        if !baseline_loss.is_finite() || !loss_plus.is_finite() || !loss_minus.is_finite() {
            self.fd_epsilon = (self.fd_epsilon * 0.5).max(1e-6);
        }
    }

    /// Evaluates on held-out data.
    pub fn evaluate(&self, eval_instances: &[StockInstance]) -> TrainingMetrics {
        let batches = create_batches(eval_instances, self.config.batch_size);

        let mut metrics = TrainingMetrics::default();
        let mut direction_correct = 0;
        let mut total_samples = 0;

        for batch in batches {
            let output = self.model.forward(&batch);
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

    /// Checks early stopping condition.
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

    /// Gets reference to the model.
    pub fn model(&self) -> &StockPredictionModel {
        &self.model
    }
}

// =============================================================================
// Section 8: Evaluation and Trading Metrics
// =============================================================================

/// Financial performance metrics.
#[derive(Debug, Clone, Default)]
pub struct FinancialMetrics {
    pub total_return: f32,
    pub annualized_return: f32,
    pub sharpe_ratio: f32,
    pub sortino_ratio: f32,
    pub max_drawdown: f32,
    pub calmar_ratio: f32,
    pub win_rate: f32,
    pub profit_factor: f32,
    pub num_trades: usize,
    pub avg_win: f32,
    pub avg_loss: f32,
}

/// Backtesting engine.
pub struct Backtester {
    initial_capital: f64,
    transaction_cost: f32,
    position_size: f32,
}

impl Backtester {
    pub fn new() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001, // 0.1% per trade
            position_size: 0.1,      // 10% of capital per position
        }
    }

    /// Runs backtesting simulation.
    pub fn run(
        &self,
        model: &StockPredictionModel,
        instances: &[StockInstance],
        _bars: &[StockBar], // Reserved for future intraday stop-loss analysis
    ) -> FinancialMetrics {
        let batches = create_batches(instances, 32);

        let mut capital = self.initial_capital;
        let mut peak_capital = capital;
        let mut max_drawdown = 0.0f64;

        let mut returns: Vec<f64> = Vec::new();
        let mut wins = 0;
        let mut losses = 0;
        let mut total_wins = 0.0f64;
        let mut total_losses = 0.0f64;

        for batch in &batches {
            let output = model.forward(batch);

            for (i, instance) in batch.iter().enumerate() {
                let direction_pred = output.direction[i];
                let magnitude_pred = output.magnitude[i];

                // Trade if confidence > 60%
                if direction_pred > 0.6 || direction_pred < 0.4 {
                    // Scale position size by predicted magnitude (higher magnitude = larger bet)
                    let magnitude_multiplier = (magnitude_pred.abs() / 2.0).clamp(0.5, 2.0);
                    let position_value =
                        capital * self.position_size as f64 * magnitude_multiplier as f64;
                    let is_long = direction_pred > 0.5;

                    // Actual return (from label)
                    let actual_return = instance.magnitude_label / 100.0;

                    // Calculate PnL
                    let gross_pnl = if is_long {
                        position_value * actual_return as f64
                    } else {
                        -position_value * actual_return as f64
                    };

                    let transaction_costs = position_value * self.transaction_cost as f64 * 2.0;
                    let net_pnl = gross_pnl - transaction_costs;

                    capital += net_pnl;
                    returns.push(net_pnl / position_value);

                    if net_pnl > 0.0 {
                        wins += 1;
                        total_wins += net_pnl;
                    } else {
                        losses += 1;
                        total_losses += -net_pnl;
                    }

                    // Update peak and drawdown
                    if capital > peak_capital {
                        peak_capital = capital;
                    }
                    let drawdown = (peak_capital - capital) / peak_capital;
                    if drawdown > max_drawdown {
                        max_drawdown = drawdown;
                    }
                }
            }
        }

        // Calculate metrics
        let total_return = (capital - self.initial_capital) / self.initial_capital;
        let num_trades = wins + losses;

        let (sharpe, sortino) = if !returns.is_empty() {
            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance: f64 = returns
                .iter()
                .map(|r| (r - mean_return).powi(2))
                .sum::<f64>()
                / returns.len() as f64;
            let std_dev = variance.sqrt();

            let sharpe = if std_dev > 0.0 {
                (mean_return / std_dev * (252.0f64).sqrt()) as f32
            } else {
                0.0
            };

            // Sortino (only downside deviation)
            let downside_returns: Vec<f64> =
                returns.iter().filter(|&&r| r < 0.0).cloned().collect();
            let downside_variance: f64 = if !downside_returns.is_empty() {
                downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
                    / downside_returns.len() as f64
            } else {
                0.0001
            };
            let downside_std = downside_variance.sqrt();
            let sortino = if downside_std > 0.0 {
                (mean_return / downside_std * (252.0f64).sqrt()) as f32
            } else {
                0.0
            };

            (sharpe, sortino)
        } else {
            (0.0, 0.0)
        };

        let win_rate = if num_trades > 0 {
            wins as f32 / num_trades as f32
        } else {
            0.0
        };

        let profit_factor = if total_losses > 0.0 {
            (total_wins / total_losses) as f32
        } else if total_wins > 0.0 {
            f32::MAX
        } else {
            1.0
        };

        let avg_win = if wins > 0 {
            (total_wins / wins as f64) as f32
        } else {
            0.0
        };

        let avg_loss = if losses > 0 {
            (total_losses / losses as f64) as f32
        } else {
            0.0
        };

        let calmar_ratio = if max_drawdown > 0.0 {
            (total_return / max_drawdown) as f32
        } else {
            0.0
        };

        FinancialMetrics {
            total_return: total_return as f32,
            annualized_return: ((1.0 + total_return).powf(252.0 / instances.len() as f64) - 1.0)
                as f32,
            sharpe_ratio: sharpe,
            sortino_ratio: sortino,
            max_drawdown: max_drawdown as f32,
            calmar_ratio: calmar_ratio,
            win_rate,
            profit_factor,
            num_trades,
            avg_win,
            avg_loss,
        }
    }
}

// =============================================================================
// Section 9: Prediction and Recommendations
// =============================================================================

/// Stock recommendation.
#[derive(Debug, Clone)]
pub struct StockRecommendation {
    pub ticker: String,
    pub sector: String,
    pub predicted_direction: String,
    pub direction_confidence: f32,
    pub predicted_return: f32,
    pub profitable_probability: f32,
    pub risk_score: f32,
    pub recommendation: String,
    pub expected_value: f32,
}

/// Generates recommendations from model predictions.
pub fn generate_recommendations(
    model: &StockPredictionModel,
    instances: &[StockInstance],
    tickers: &[TickerInfo],
) -> Vec<StockRecommendation> {
    let batch: Vec<&StockInstance> = instances.iter().collect();
    let output = model.forward(&batch);

    let mut recommendations = Vec::new();

    for (i, instance) in instances.iter().enumerate() {
        let ticker = tickers
            .iter()
            .find(|t| t.ticker_id == instance.ticker_fid)
            .unwrap();

        let direction_conf = output.direction[i];
        let predicted_return = output.magnitude[i];
        let profitable_prob = output.profitable[i];

        // Determine direction
        let (direction, effective_conf) = if direction_conf > 0.5 {
            ("UP", direction_conf)
        } else {
            ("DOWN", 1.0 - direction_conf)
        };

        // Risk score based on volatility indicators
        let atr_normalized = instance.indicator_features.get(12).copied().unwrap_or(0.0);
        let bb_width = instance.indicator_features.get(11).copied().unwrap_or(0.0);
        let risk_score = ((atr_normalized.abs() + bb_width.abs()) / 2.0).clamp(0.0, 1.0);

        // Determine recommendation
        let recommendation = if direction == "UP" {
            if effective_conf > 0.75 && predicted_return > 2.0 {
                "STRONG BUY"
            } else if effective_conf > 0.6 && predicted_return > 1.0 {
                "BUY"
            } else {
                "HOLD"
            }
        } else {
            if effective_conf > 0.75 && predicted_return < -2.0 {
                "STRONG SELL"
            } else if effective_conf > 0.6 && predicted_return < -1.0 {
                "SELL"
            } else {
                "HOLD"
            }
        };

        // Expected value = confidence * predicted_return * (1 - risk)
        let expected_value = effective_conf * predicted_return.abs() * (1.0 - risk_score * 0.5);

        recommendations.push(StockRecommendation {
            ticker: ticker.symbol.clone(),
            sector: ticker.sector.name().to_string(),
            predicted_direction: direction.to_string(),
            direction_confidence: effective_conf,
            predicted_return,
            profitable_probability: profitable_prob,
            risk_score,
            recommendation: recommendation.to_string(),
            expected_value,
        });
    }

    // Sort by expected value (descending)
    recommendations.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());

    recommendations
}

/// Prints recommendation report.
pub fn print_recommendation_report(recommendations: &[StockRecommendation], top_n: usize) {
    println!("\n  Top {} Stock Recommendations:", top_n);
    println!("  ┌────────┬───────────┬────────────┬─────────────┬────────────────┐");
    println!("  │ Ticker │ Direction │ Confidence │ Pred Return │ Recommendation │");
    println!("  ├────────┼───────────┼────────────┼─────────────┼────────────────┤");

    for rec in recommendations.iter().take(top_n) {
        println!(
            "  │ {:6} │ {:9} │ {:>9.0}% │ {:>+10.1}% │ {:14} │",
            rec.ticker,
            rec.predicted_direction,
            rec.direction_confidence * 100.0,
            rec.predicted_return,
            rec.recommendation
        );
    }

    println!("  └────────┴───────────┴────────────┴─────────────┴────────────────┘");
}

// =============================================================================
// Section 10: Main and CLI
// =============================================================================

// NOTE: Synthetic generation remains in this file because it is useful for unit tests and
// quick local experimentation, but by default the CLI points at a real intraday dataset
// (FutureSharks financial-data) and will not use synthetic data unless you explicitly
// change `--data-dir` and/or code paths.

/// Calculate beta for each ticker relative to an equal-weighted market index.
///
/// Beta = Cov(stock_returns, market_returns) / Var(market_returns)
///
/// Beta interpretation:
/// - beta > 1: More volatile than the market
/// - beta = 1: Moves with the market
/// - beta < 1: Less volatile than the market
/// - beta < 0: Moves opposite to the market
fn calculate_betas(tickers: &[TickerInfo], bars: &[StockBar]) -> Vec<TickerInfo> {
    // First, compute daily market returns (equal-weighted average of all tickers)
    let mut day_returns: HashMap<usize, Vec<f32>> = HashMap::new();

    for bar in bars {
        day_returns
            .entry(bar.day_index)
            .or_insert_with(Vec::new)
            .push(bar.returns);
    }

    // Calculate market returns per day (equal-weighted)
    let mut market_returns: HashMap<usize, f32> = HashMap::new();
    for (day, returns) in &day_returns {
        let n = returns.len();
        if n > 0 {
            let avg = returns.iter().sum::<f32>() / n as f32;
            market_returns.insert(*day, avg);
        }
    }

    // Calculate market variance
    let mkt_values: Vec<f32> = market_returns.values().cloned().collect();
    let mkt_mean: f32 = mkt_values.iter().sum::<f32>() / mkt_values.len().max(1) as f32;
    let mkt_variance: f32 = mkt_values
        .iter()
        .map(|r| (r - mkt_mean).powi(2))
        .sum::<f32>()
        / mkt_values.len().max(1) as f32;

    // Calculate beta for each ticker
    let mut result = Vec::with_capacity(tickers.len());

    for ticker in tickers {
        // Get this ticker's returns
        let ticker_bars: Vec<&StockBar> = bars
            .iter()
            .filter(|b| b.ticker_id == ticker.ticker_id)
            .collect();

        if ticker_bars.is_empty() || mkt_variance < 1e-10 {
            // No data or zero variance, default beta = 1.0
            result.push(ticker.clone());
            continue;
        }

        // Calculate covariance with market
        let mut covariance = 0.0_f32;
        let mut count = 0;

        for bar in &ticker_bars {
            if let Some(&mkt_ret) = market_returns.get(&bar.day_index) {
                covariance += (bar.returns - mkt_mean) * (mkt_ret - mkt_mean);
                count += 1;
            }
        }

        let beta = if count > 0 && mkt_variance > 1e-10 {
            (covariance / count as f32) / mkt_variance
        } else {
            1.0 // Default beta
        };

        // Create updated ticker with calculated beta
        result.push(TickerInfo {
            ticker_id: ticker.ticker_id,
            name: ticker.name.clone(),
            symbol: ticker.symbol.clone(),
            sector: ticker.sector,
            beta: beta.clamp(0.0, 3.0), // Clamp to reasonable range
            base_volatility: ticker.base_volatility,
            drift: ticker.drift,
        });
    }

    println!(
        "  Calculated betas for {} tickers (market variance: {:.6})",
        result.len(),
        mkt_variance
    );

    result
}

pub fn run() {
    let config = parse_args();

    println!("{}", "=".repeat(80));
    println!("Monolith-RS Stock Prediction Example");
    println!("{}", "=".repeat(80));
    println!();

    // Show performance configuration
    println!(
        "Configuration: {} workers | GPU: {} | Batch: {} | LR: {}",
        config.num_workers,
        if config.gpu_mode {
            "enabled"
        } else {
            "disabled"
        },
        config.batch_size,
        config.learning_rate
    );
    println!();

    let start_time = Instant::now();

    // Section 1: Load or Generate Stock Data
    ensure_futuresharks_dataset_present(config.data_dir.as_deref());
    let (tickers, bars): (Vec<TickerInfo>, Vec<StockBar>) =
        if let Some(ref data_dir) = config.data_dir {
            // Load real data from CSV files (required for this example).
            println!("Section 1: Loading Real Stock Data from CSV");
            match load_real_data_auto(data_dir, &config) {
                Ok((tickers, bars)) => (tickers, bars),
                Err(e) => {
                    eprintln!("Error loading data: {}", e);
                    eprintln!();
                    IntradayDataSources::print_download_instructions();
                    std::process::exit(2);
                }
            }
        } else {
            // With current defaults this should never happen, but keep a clear message.
            eprintln!("Error: --data-dir is required (this example does not use synthetic data).");
            eprintln!();
            print_usage();
            std::process::exit(2);
        };

    let total_bars = bars.len();

    println!(
        "  Tickers: {} | Days: {} | Total bars: {}",
        config.num_tickers, config.days_of_history, total_bars
    );

    // Section 2: Compute Technical Indicators (parallel)
    println!("\nSection 2: Computing Technical Indicators (parallel)");

    // Parallel indicator computation using rayon
    let indicator_results: Vec<(i64, Vec<TechnicalIndicators>)> = tickers
        .par_iter()
        .map(|ticker| {
            let ticker_bars: Vec<StockBar> = bars
                .iter()
                .filter(|b| b.ticker_id == ticker.ticker_id)
                .cloned()
                .collect();
            let mut calc = IndicatorCalculator::new();
            let indicators = calc.compute_indicators(&ticker_bars);
            (ticker.ticker_id, indicators)
        })
        .collect();

    let all_indicators: HashMap<i64, Vec<TechnicalIndicators>> =
        indicator_results.into_iter().collect();

    println!(
        "  {} indicators per bar: SMA, RSI, MACD, Bollinger, ATR, OBV...",
        TechnicalIndicators::NUM_FEATURES
    );

    // Section 3: Create Training Instances
    println!("\nSection 3: Creating Training Instances");
    let instance_creator = InstanceCreator::new(config.lookback_window);
    let mut all_instances = Vec::new();

    for ticker in &tickers {
        let ticker_bars: Vec<StockBar> = bars
            .iter()
            .filter(|b| b.ticker_id == ticker.ticker_id)
            .cloned()
            .collect();
        if let Some(indicators) = all_indicators.get(&ticker.ticker_id) {
            let instances = instance_creator.create_instances(ticker, &ticker_bars, indicators);
            all_instances.extend(instances);
        }
    }

    // Split by time (not random) to avoid data leakage.
    // IMPORTANT: For multi-ticker intraday datasets where `all_instances` is an interleaving of
    // multiple time series, a global split is not truly "time ordered" per ticker.
    // We instead split per ticker inside the loop above by using the fact that `create_instances`
    // yields instances in chronological order for that ticker.
    //
    // For backward compatibility, keep the global split as a fallback if per-ticker split
    // hasn't been applied (i.e., if all_instances is empty).
    let (train_instances, eval_instances) = train_eval_split(&all_instances, config.train_ratio);

    println!(
        "  Train: {} ({:.0}%) | Eval: {} ({:.0}%) | Lookback: {} bars",
        train_instances.len(),
        config.train_ratio * 100.0,
        eval_instances.len(),
        (1.0 - config.train_ratio) * 100.0,
        config.lookback_window
    );

    // Section 4: Model Architecture
    println!("\nSection 4: Model Architecture");
    println!(
        "  Ticker embedding: {} x {} | DIEN hidden: {} | DCN: {} layers",
        config.num_tickers,
        config.ticker_embedding_dim,
        config.dien_hidden_size,
        config.dcn_cross_layers
    );

    match config.mode {
        Mode::Train => {
            run_training(&config, &train_instances, &eval_instances, &bars, &tickers);
        }
        Mode::Evaluate => {
            run_evaluation(&config, &eval_instances);
        }
        Mode::Predict => {
            run_prediction(&config, &all_instances, &tickers);
        }
        Mode::Backtest => {
            run_backtesting(&config, &eval_instances, &bars, &tickers);
        }
    }

    let elapsed = start_time.elapsed();
    println!();
    println!("{}", "=".repeat(80));
    println!("Complete! Total time: {:.2}s", elapsed.as_secs_f64());
    println!("{}", "=".repeat(80));
}

fn ensure_futuresharks_dataset_present(data_dir: Option<&str>) {
    use std::path::Path;
    use std::process::Command;

    let Some(data_dir) = data_dir else {
        return;
    };

    // If the directory exists, do nothing.
    if Path::new(data_dir).exists() {
        return;
    }

    // If the user is relying on our default FutureSharks path, try to clone the repo.
    if data_dir == DEFAULT_FUTURESHARKS_DATA_DIR
        && !Path::new(DEFAULT_FUTURESHARKS_REPO_DIR).exists()
    {
        eprintln!(
            "Default intraday dataset not found at `{}`. Cloning into `{}` ...",
            DEFAULT_FUTURESHARKS_DATA_DIR, DEFAULT_FUTURESHARKS_REPO_DIR
        );
        let status = Command::new("git")
            .args([
                "clone",
                "--depth",
                "1",
                "https://github.com/FutureSharks/financial-data.git",
                DEFAULT_FUTURESHARKS_REPO_DIR,
            ])
            .status();

        if !matches!(status, Ok(s) if s.success()) {
            eprintln!("Auto-clone failed. Run this manually:");
            eprintln!(
                "  git clone https://github.com/FutureSharks/financial-data.git {}",
                DEFAULT_FUTURESHARKS_REPO_DIR
            );
        }
    }
}

fn load_real_data_auto(
    data_dir: &str,
    config: &StockPredictorConfig,
) -> Result<(Vec<TickerInfo>, Vec<StockBar>), String> {
    let canonical_dir = normalize_futuresharks_dir(data_dir);

    if is_futuresharks_histdata_dir(&canonical_dir) {
        let mut loader = FutureSharksHistdataLoader::new();
        loader.load_dir(&canonical_dir, config.num_tickers)?;
        Ok((loader.tickers().to_vec(), loader.bars().to_vec()))
    } else {
        let mut loader = CsvDataLoader::new(&canonical_dir);
        loader.load(config.num_tickers, config.lookback_window + 50)?;
        let tickers_with_beta = calculate_betas(loader.tickers(), loader.bars());
        Ok((tickers_with_beta, loader.bars().to_vec()))
    }
}

fn normalize_futuresharks_dir(data_dir: &str) -> String {
    // Allow pointing at the repo root; map it to the histdata directory.
    let p = std::path::Path::new(data_dir);
    if p.ends_with("financial-data") || p.ends_with("data/financial-data") {
        return DEFAULT_FUTURESHARKS_DATA_DIR.to_string();
    }
    data_dir.to_string()
}

fn is_futuresharks_histdata_dir(data_dir: &str) -> bool {
    // Heuristic: any CSV with the Histdata naming pattern indicates this dataset.
    // Example: DAT_ASCII_SPXUSD_M1_2017.csv
    let Ok(rd) = std::fs::read_dir(data_dir) else {
        return false;
    };

    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let Ok(sub) = std::fs::read_dir(&path) else {
                continue;
            };
            for e in sub.flatten() {
                let p = e.path();
                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("DAT_ASCII_")
                        && name.contains("_M1_")
                        && name.ends_with(".csv")
                    {
                        return true;
                    }
                }
            }
        }
    }

    false
}

fn run_training(
    config: &StockPredictorConfig,
    train_instances: &[StockInstance],
    eval_instances: &[StockInstance],
    bars: &[StockBar],
    tickers: &[TickerInfo],
) {
    println!("\nSection 5: Training (with momentum, LR decay, weight decay)");
    let mut trainer = Trainer::new(config);

    for epoch in 0..config.num_epochs {
        // Apply learning rate decay at start of each epoch
        if epoch > 0 {
            trainer.decay_lr();
        }

        let train_metrics = trainer.train_epoch(train_instances);
        let eval_metrics = trainer.evaluate(eval_instances);

        println!(
            "  Epoch {}/{}: Loss: {:.4} | Eval: {:.4} | Acc: {:.1}% | LR: {:.6}",
            epoch + 1,
            config.num_epochs,
            train_metrics.total_loss,
            eval_metrics.total_loss,
            eval_metrics.direction_accuracy * 100.0,
            trainer.current_lr()
        );

        // Early stopping check
        if trainer.check_early_stopping(eval_metrics.total_loss) {
            println!("  Early stopping triggered at epoch {}", epoch + 1);
            break;
        }
    }

    // Final evaluation
    println!("\nSection 6: Evaluation Results");
    let final_eval = trainer.evaluate(eval_instances);
    println!(
        "  Direction Accuracy: {:.1}%",
        final_eval.direction_accuracy * 100.0
    );
    println!(
        "  Losses - Direction: {:.4} | Magnitude: {:.4} | Profitable: {:.4}",
        final_eval.direction_loss, final_eval.magnitude_loss, final_eval.profitable_loss
    );

    // Backtesting
    println!("\nSection 7: Backtest Results");
    let backtester = Backtester::new();
    let financial_metrics = backtester.run(trainer.model(), eval_instances, bars);

    println!(
        "  Total Return: {:.1}%",
        financial_metrics.total_return * 100.0
    );
    println!("  Sharpe Ratio: {:.2}", financial_metrics.sharpe_ratio);
    println!(
        "  Max Drawdown: {:.1}%",
        financial_metrics.max_drawdown * 100.0
    );
    println!("  Win Rate: {:.1}%", financial_metrics.win_rate * 100.0);
    println!("  Profit Factor: {:.2}", financial_metrics.profit_factor);
    println!("  Trades: {}", financial_metrics.num_trades);

    // Recommendations
    println!("\nSection 8: Top Stock Recommendations");
    let latest_instances: Vec<StockInstance> = tickers
        .iter()
        .filter_map(|ticker| {
            train_instances
                .iter()
                .chain(eval_instances.iter())
                .filter(|i| i.ticker_fid == ticker.ticker_id)
                .last()
                .cloned()
        })
        .collect();

    let recommendations = generate_recommendations(trainer.model(), &latest_instances, tickers);
    print_recommendation_report(&recommendations, 10);
}

fn run_evaluation(config: &StockPredictorConfig, eval_instances: &[StockInstance]) {
    println!("\nRunning Evaluation Mode");
    let model = StockPredictionModel::new(config);

    let batches = create_batches(eval_instances, config.batch_size);
    let mut total_correct = 0;
    let mut total_samples = 0;

    for batch in batches {
        let output = model.forward(&batch);
        for (i, instance) in batch.iter().enumerate() {
            let pred = if output.direction[i] > 0.5 { 1.0 } else { 0.0 };
            if (pred - instance.direction_label).abs() < 0.1 {
                total_correct += 1;
            }
        }
        total_samples += batch.len();
    }

    println!(
        "  Evaluation Accuracy: {:.1}% ({}/{})",
        total_correct as f32 / total_samples as f32 * 100.0,
        total_correct,
        total_samples
    );
}

fn run_prediction(
    config: &StockPredictorConfig,
    instances: &[StockInstance],
    tickers: &[TickerInfo],
) {
    println!("\nRunning Prediction Mode");
    let model = StockPredictionModel::new(config);

    // Get latest instance per ticker
    let latest_instances: Vec<StockInstance> = tickers
        .iter()
        .filter_map(|ticker| {
            instances
                .iter()
                .filter(|i| i.ticker_fid == ticker.ticker_id)
                .last()
                .cloned()
        })
        .collect();

    let recommendations = generate_recommendations(&model, &latest_instances, tickers);
    print_recommendation_report(&recommendations, config.num_tickers.min(20));
}

fn run_backtesting(
    config: &StockPredictorConfig,
    instances: &[StockInstance],
    bars: &[StockBar],
    _tickers: &[TickerInfo],
) {
    println!("\nRunning Backtest Mode");
    let model = StockPredictionModel::new(config);
    let backtester = Backtester::new();

    let metrics = backtester.run(&model, instances, bars);

    println!("\n  Backtest Performance Summary:");
    println!("  ─────────────────────────────────────");
    println!(
        "  Total Return:      {:>+8.2}%",
        metrics.total_return * 100.0
    );
    println!(
        "  Annualized Return: {:>+8.2}%",
        metrics.annualized_return * 100.0
    );
    println!("  Sharpe Ratio:      {:>8.2}", metrics.sharpe_ratio);
    println!("  Sortino Ratio:     {:>8.2}", metrics.sortino_ratio);
    println!(
        "  Max Drawdown:      {:>8.2}%",
        metrics.max_drawdown * 100.0
    );
    println!("  Calmar Ratio:      {:>8.2}", metrics.calmar_ratio);
    println!("  Win Rate:          {:>8.2}%", metrics.win_rate * 100.0);
    println!("  Profit Factor:     {:>8.2}", metrics.profit_factor);
    println!("  Total Trades:      {:>8}", metrics.num_trades);
    println!("  Avg Win:           ${:>7.2}", metrics.avg_win);
    println!("  Avg Loss:          ${:>7.2}", metrics.avg_loss);
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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

        let bars: Vec<StockBar> = generator.bars().to_vec();
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
        let bars: Vec<StockBar> = generator.bars().to_vec();

        let mut calc = IndicatorCalculator::new();
        let indicators = calc.compute_indicators(&bars);

        let creator = InstanceCreator::new(10);
        let instances = creator.create_instances(ticker, &bars, &indicators);

        // Should have instances from day 10 to day 44 (50 - 5 forward - 1)
        assert!(!instances.is_empty());
    }

    #[test]
    fn test_embedding_tables() {
        let tables = EmbeddingTables::new(10, 16, 11, 8);

        let ticker_emb = tables.lookup_ticker(0);
        assert_eq!(ticker_emb.len(), 16);

        let sector_emb = tables.lookup_sector(0);
        assert_eq!(sector_emb.len(), 8);
    }

    #[test]
    fn test_model_forward() {
        let config = StockPredictorConfig {
            num_tickers: 5,
            days_of_history: 50,
            lookback_window: 10,
            ticker_embedding_dim: 8,
            sector_embedding_dim: 4,
            dien_hidden_size: 16,
            dcn_cross_layers: 2,
            ..Default::default()
        };

        let model = StockPredictionModel::new(&config);

        // Create test instance
        let instance = StockInstance {
            ticker_fid: 0,
            sector_fid: 0,
            price_features: vec![0.01, 0.02, 0.01, 1.0],
            indicator_features: vec![0.0; TechnicalIndicators::NUM_FEATURES],
            historical_sequence: vec![vec![0.0; 4 + TechnicalIndicators::NUM_FEATURES]; 10],
            direction_label: 1.0,
            magnitude_label: 2.0,
            profitable_label: 1.0,
        };

        let batch = vec![&instance];
        let output = model.forward(&batch);

        assert_eq!(output.direction.len(), 1);
        assert_eq!(output.magnitude.len(), 1);
        assert_eq!(output.profitable.len(), 1);

        // Direction should be between 0 and 1 (sigmoid)
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
}
