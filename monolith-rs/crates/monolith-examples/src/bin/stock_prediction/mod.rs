1 -// Copyright 2022 ByteDance and/or its affiliates.
2 -//
3 -// Licensed under the Apache License, Version 2.0 (the "License");
4 -// you may not use this file except in compliance with the License.
5 -// You may obtain a copy of the License at
6 -//
7 -//     http://www.apache.org/licenses/LICENSE-2.0
8 -//
9 -// Unless required by applicable law or agreed to in writing, software
10 -// distributed under the License is distributed on an "AS IS" BASIS,
11 -// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
12 -// See the License for the specific language governing permissions and
13 -// limitations under the License.
14 -
15 -//! Stock Prediction Example for Monolith-RS
16 -//!
17 -//! This comprehensive example demonstrates monolith-rs capabilities for financial
18 -//! time-series prediction, technical analysis, and actionable stock recommendations.
19 -//!
20 -//! Features:
21 -//! - Synthetic OHLCV data generation with realistic market dynamics
22 -//! - Technical indicator computation (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV)
23 -//! - Ticker and sector embeddings using CuckooEmbeddingHashTable
24 -//! - DIEN for sequential pattern recognition in price history
25 -//! - DCN for feature interaction modeling
26 -//! - Multi-task learning (direction, magnitude, profitability)
27 -//! - Backtesting with financial metrics (Sharpe, Drawdown, Win Rate)
28 -//! - Stock recommendation generation
29 -//!
30 -//! # Usage
31 -//!
32 -//! ```bash
33 -//! # Full training pipeline
34 -//! cargo run -p monolith-examples --bin stock_prediction -- --mode train
35 -//!
36 -//! # Generate predictions only
37 -//! cargo run -p monolith-examples --bin stock_prediction -- --mode predict
38 -//!
39 -//! # Run backtesting simulation
40 -//! cargo run -p monolith-examples --bin stock_prediction -- --mode backtest
41 -//!
42 -//! # Custom configuration
43 -//! cargo run -p monolith-examples --bin stock_prediction -- \
44 -//!     --mode train \
45 -//!     --num-tickers 100 \
46 -//!     --days 504 \
47 -//!     --batch-size 64 \
48 -//!     --epochs 20
49 -//! ```
50 -
51 -use std::collections::HashMap;
52 -use std::sync::Arc;
53 -use std::time::Instant;
54 -
55 -// Default local path for the FutureSharks 1-minute dataset (repo: https://github.com/FutureSharks/financial-data).
56 -// We keep this relative so `cargo run ...` works from the repo root with no flags.
57 -const DEFAULT_FUTURESHARKS_REPO_DIR: &str = "data/financial-data";
58 -const DEFAULT_FUTURESHARKS_DATA_DIR: &str =
59 -    "data/financial-data/pyfinancialdata/data/stocks/histdata";
60 -
61 -// Parallel processing
62 -use rayon::prelude::*;
63 -
64 -use monolith_hash_table::{CuckooEmbeddingHashTable, EmbeddingHashTable, XavierNormalInitializer};
65 -use monolith_layers::{
66 -    dcn::{CrossNetwork, DCNMode},
67 -    dien::{DIENConfig, DIENLayer},
68 -    din::{DINAttention, DINConfig},
69 -    layer::Layer,
70 -    mlp::{ActivationType, MLPConfig, MLP},
71 -    mmoe::{MMoE, MMoEConfig},
72 -    normalization::LayerNorm,
73 -    senet::SENetLayer,
74 -    tensor::Tensor,
75 -};
76 -use monolith_optimizer::Amsgrad;
77 -
78 -// Technical analysis crate for indicators (https://crates.io/crates/ta)
79 -// Using all available indicators to maximize feature richness for the model
80 -use ta::indicators::{
81 -    AverageTrueRange, BollingerBands, ExponentialMovingAverage, Maximum, Minimum, MoneyFlowIndex,
82 -    OnBalanceVolume, RateOfChange, RelativeStrengthIndex, SimpleMovingAverage, SlowStochastic,
83 -    StandardDeviation,
84 -};
85 -use ta::Next;
86 -
87 -// =============================================================================
88 -// Section 1: Configuration and CLI
89 -// =============================================================================
90 -
91 -/// Operating mode for the stock predictor.
92 -#[derive(Debug, Clone, Copy, PartialEq, Eq)]
93 -pub enum Mode {
94 -    /// Train the model from scratch
95 -    Train,
96 -    /// Evaluate on held-out data
97 -    Evaluate,
98 -    /// Generate predictions for current data
99 -    Predict,
100 -    /// Run backtesting simulation
101 -    Backtest,
102 -}
103 -
104 -impl Mode {
105 -    fn from_str(s: &str) -> Option<Self> {
106 -        match s.to_lowercase().as_str() {
107 -            "train" => Some(Mode::Train),
108 -            "evaluate" | "eval" => Some(Mode::Evaluate),
109 -            "predict" => Some(Mode::Predict),
110 -            "backtest" => Some(Mode::Backtest),
111 -            _ => None,
112 -        }
113 -    }
114 -}
115 -
116 -/// Configuration for the stock prediction pipeline.
117 -#[derive(Debug, Clone)]
118 -pub struct StockPredictorConfig {
119 -    // Data generation
120 -    /// Number of tickers to generate
121 -    pub num_tickers: usize,
122 -    /// Days of historical data
123 -    pub days_of_history: usize,
124 -    /// Lookback window for sequences
125 -    pub lookback_window: usize,
126 -
127 -    // Model architecture
128 -    /// Ticker embedding dimension
129 -    pub ticker_embedding_dim: usize,
130 -    /// Sector embedding dimension
131 -    pub sector_embedding_dim: usize,
132 -    /// DIEN hidden size
133 -    pub dien_hidden_size: usize,
134 -    /// Number of DCN cross layers
135 -    pub dcn_cross_layers: usize,
136 -
137 -    // Training
138 -    /// Batch size
139 -    pub batch_size: usize,
140 -    /// Learning rate
141 -    pub learning_rate: f32,
142 -    /// Train/eval split ratio
143 -    pub train_ratio: f32,
144 -    /// Number of training epochs
145 -    pub num_epochs: usize,
146 -    /// Log every N steps
147 -    pub log_every_n_steps: usize,
148 -    /// Early stopping patience
149 -    pub early_stopping_patience: usize,
150 -    /// Minimum improvement in eval loss to reset early stopping patience.
151 -    ///
152 -    /// Smaller values allow training to continue when eval loss improves slowly.
153 -    pub early_stopping_min_delta: f32,
154 -
155 -    // Operation mode
156 -    /// Current mode
157 -    pub mode: Mode,
158 -    /// Random seed
159 -    pub seed: u64,
160 -    /// Verbose output
161 -    pub verbose: bool,
162 -
163 -    // Data source
164 -    /// Path to directory containing CSV files (if None, use synthetic data)
165 -    pub data_dir: Option<String>,
166 -
167 -    // Performance
168 -    /// Number of parallel workers (0 = auto-detect)
169 -    pub num_workers: usize,
170 -    /// Enable GPU acceleration (requires compatible backend)
171 -    pub gpu_mode: bool,
172 -}
173 -
174 -impl Default for StockPredictorConfig {
175 -    fn default() -> Self {
176 -        Self {
177 -            // Data generation - tuned for Mac CPU intensive training
178 -            num_tickers: 50,
179 -            days_of_history: 252,
180 -            lookback_window: 20,
181 -
182 -            // Model architecture - larger capacity
183 -            ticker_embedding_dim: 32,
184 -            sector_embedding_dim: 16,
185 -            dien_hidden_size: 128,
186 -            dcn_cross_layers: 4,
187 -
188 -            // Training - tuned for convergence
189 -            batch_size: 64,
190 -            learning_rate: 0.0003, // Lower LR for stability
191 -            train_ratio: 0.8,
192 -            num_epochs: 100,             // More epochs, early stopping will kick in
193 -            log_every_n_steps: 50,       // More frequent logging
194 -            early_stopping_patience: 20, // Allow longer training by default
195 -            early_stopping_min_delta: 1e-4,
196 -
197 -            // Operation mode
198 -            mode: Mode::Train,
199 -            seed: 42,
200 -            verbose: true,
201 -
202 -            // Data source
203 -            data_dir: Some(DEFAULT_FUTURESHARKS_DATA_DIR.to_string()),
204 -
205 -            // Performance
206 -            num_workers: 0, // 0 = auto-detect from num_cpus
207 -            gpu_mode: false,
208 -        }
209 -    }
210 -}
211 -
212 -fn parse_args() -> StockPredictorConfig {
213 -    let args: Vec<String> = std::env::args().collect();
214 -    let mut config = StockPredictorConfig::default();
215 -
216 -    let mut i = 1;
217 -    while i < args.len() {
218 -        match args[i].as_str() {
219 -            "--mode" | "-m" => {
220 -                if i + 1 < args.len() {
221 -                    if let Some(mode) = Mode::from_str(&args[i + 1]) {
222 -                        config.mode = mode;
223 -                    }
224 -                    i += 1;
225 -                }
226 -            }
227 -            "--num-tickers" | "-t" => {
228 -                if i + 1 < args.len() {
229 -                    config.num_tickers = args[i + 1].parse().unwrap_or(50);
230 -                    i += 1;
231 -                }
232 -            }
233 -            "--days" | "-d" => {
234 -                if i + 1 < args.len() {
235 -                    config.days_of_history = args[i + 1].parse().unwrap_or(252);
236 -                    i += 1;
237 -                }
238 -            }
239 -            "--lookback" | "-l" => {
240 -                if i + 1 < args.len() {
241 -                    config.lookback_window = args[i + 1].parse().unwrap_or(20);
242 -                    i += 1;
243 -                }
244 -            }
245 -            "--batch-size" | "-b" => {
246 -                if i + 1 < args.len() {
247 -                    config.batch_size = args[i + 1].parse().unwrap_or(32);
248 -                    i += 1;
249 -                }
250 -            }
251 -            "--learning-rate" | "-lr" => {
252 -                if i + 1 < args.len() {
253 -                    config.learning_rate = args[i + 1].parse().unwrap_or(0.001);
254 -                    i += 1;
255 -                }
256 -            }
257 -            "--epochs" | "-e" => {
258 -                if i + 1 < args.len() {
259 -                    config.num_epochs = args[i + 1].parse().unwrap_or(10);
260 -                    i += 1;
261 -                }
262 -            }
263 -            "--patience" | "--early-stopping-patience" => {
264 -                if i + 1 < args.len() {
265 -                    config.early_stopping_patience = args[i + 1].parse().unwrap_or(20);
266 -                    i += 1;
267 -                }
268 -            }
269 -            "--min-delta" | "--early-stopping-min-delta" => {
270 -                if i + 1 < args.len() {
271 -                    config.early_stopping_min_delta = args[i + 1].parse().unwrap_or(1e-4);
272 -                    i += 1;
273 -                }
274 -            }
275 -            "--seed" | "-s" => {
276 -                if i + 1 < args.len() {
277 -                    config.seed = args[i + 1].parse().unwrap_or(42);
278 -                    i += 1;
279 -                }
280 -            }
281 -            "--data-dir" | "--data" => {
282 -                if i + 1 < args.len() {
283 -                    config.data_dir = Some(args[i + 1].clone());
284 -                    i += 1;
285 -                }
286 -            }
287 -            "--quiet" | "-q" => {
288 -                config.verbose = false;
289 -            }
290 -            "--workers" | "-w" => {
291 -                if i + 1 < args.len() {
292 -                    config.num_workers = args[i + 1].parse().unwrap_or(0);
293 -                    i += 1;
294 -                }
295 -            }
296 -            "--gpu" => {
297 -                config.gpu_mode = true;
298 -            }
299 -            "--help" | "-h" => {
300 -                print_usage();
301 -                std::process::exit(0);
302 -            }
303 -            _ => {}
304 -        }
305 -        i += 1;
306 -    }
307 -
308 -    // Auto-detect workers if not specified
309 -    if config.num_workers == 0 {
310 -        config.num_workers = num_cpus::get();
311 -    }
312 -
313 -    // Configure rayon thread pool
314 -    rayon::ThreadPoolBuilder::new()
315 -        .num_threads(config.num_workers)
316 -        .build_global()
317 -        .ok();
318 -
319 -    config
320 -}
321 -
322 -fn print_usage() {
323 -    println!(
324 -        r#"Monolith-RS Stock Prediction Example
325 -
326 -USAGE:
327 -    stock_prediction [OPTIONS]
328 -
329 -OPTIONS:
330 -    -m, --mode <MODE>           Operation mode: train, evaluate, predict, backtest
331 -                                [default: train]
332 -    -t, --num-tickers <N>       Number of tickers to load/simulate [default: 50]
333 -    -d, --days <N>              Days of historical data [default: 252]
334 -    -l, --lookback <N>          Lookback window size [default: 20]
335 -    -b, --batch-size <N>        Training batch size [default: 32]
336 -    -lr, --learning-rate <LR>   Learning rate [default: 0.001]
337 -    -e, --epochs <N>            Number of training epochs [default: 10]
338 -    --patience <N>              Early stopping patience (epochs without improvement) [default: 20]
339 -    --min-delta <X>             Minimum eval loss improvement to reset patience [default: 1e-4]
340 -    -s, --seed <SEED>           Random seed [default: 42]
341 -    --data-dir <PATH>           Load real stock data from CSV files in directory
342 -                                [default: data/financial-data/pyfinancialdata/data/stocks/histdata]
343 -    -w, --workers <N>           Number of parallel workers [default: auto-detect]
344 -    --gpu                       Enable GPU acceleration mode
345 -    -q, --quiet                 Suppress verbose output
346 -    -h, --help                  Print help information
347 -
348 -EXAMPLES:
349 -    # One-time: clone the intraday dataset (1-minute bars) into ./data/financial-data
350 -    git clone https://github.com/FutureSharks/financial-data.git data/financial-data
351 -
352 -    # Train with FutureSharks intraday dataset (default --data-dir)
353 -    stock_prediction --mode train
354 -
355 -    # Train with real data from Kaggle
356 -    # Download from: https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
357 -    stock_prediction --mode train --data-dir /path/to/kaggle/stocks
358 -
359 -    # Generate predictions with real data
360 -    stock_prediction --mode predict --data-dir ./data/stocks --num-tickers 100
361 -
362 -    # Run backtesting with more history
363 -    stock_prediction --mode backtest --data-dir ./data/stocks --days 504
364 -
365 -    # Train longer / avoid early stopping on slow improvements
366 -    stock_prediction --mode train --data-dir ./data/stocks --epochs 300 --patience 50 --min-delta 1e-5
367 -"#
368 -    );
369 -}
370 -
371 -// =============================================================================
372 -// Section 2: Synthetic Stock Data Generation
373 -// =============================================================================
374 -
375 -/// Market sectors for stock categorization.
376 -#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
377 -pub enum Sector {
378 -    Technology,
379 -    Healthcare,
380 -    Finance,
381 -    Consumer,
382 -    Industrial,
383 -    Energy,
384 -    Materials,
385 -    RealEstate,
386 -    Utilities,
387 -    Communications,
388 -    ConsumerStaples,
389 -}
390 -
391 -impl Sector {
392 -    fn all() -> &'static [Sector] {
393 -        &[
394 -            Sector::Technology,
395 -            Sector::Healthcare,
396 -            Sector::Finance,
397 -            Sector::Consumer,
398 -            Sector::Industrial,
399 -            Sector::Energy,
400 -            Sector::Materials,
401 -            Sector::RealEstate,
402 -            Sector::Utilities,
403 -            Sector::Communications,
404 -            Sector::ConsumerStaples,
405 -        ]
406 -    }
407 -
408 -    fn id(&self) -> i64 {
409 -        match self {
410 -            Sector::Technology => 0,
411 -            Sector::Healthcare => 1,
412 -            Sector::Finance => 2,
413 -            Sector::Consumer => 3,
414 -            Sector::Industrial => 4,
415 -            Sector::Energy => 5,
416 -            Sector::Materials => 6,
417 -            Sector::RealEstate => 7,
418 -            Sector::Utilities => 8,
419 -            Sector::Communications => 9,
420 -            Sector::ConsumerStaples => 10,
421 -        }
422 -    }
423 -
424 -    fn name(&self) -> &'static str {
425 -        match self {
426 -            Sector::Technology => "Technology",
427 -            Sector::Healthcare => "Healthcare",
428 -            Sector::Finance => "Finance",
429 -            Sector::Consumer => "Consumer Discretionary",
430 -            Sector::Industrial => "Industrial",
431 -            Sector::Energy => "Energy",
432 -            Sector::Materials => "Materials",
433 -            Sector::RealEstate => "Real Estate",
434 -            Sector::Utilities => "Utilities",
435 -            Sector::Communications => "Communications",
436 -            Sector::ConsumerStaples => "Consumer Staples",
437 -        }
438 -    }
439 -}
440 -
441 -/// Information about a ticker.
442 -#[derive(Debug, Clone)]
443 -pub struct TickerInfo {
444 -    pub ticker_id: i64,
445 -    pub name: String,
446 -    pub symbol: String,
447 -    pub sector: Sector,
448 -    pub beta: f32,
449 -    pub base_volatility: f32,
450 -    pub drift: f32,
451 -}
452 -
453 -/// A single price bar (OHLCV).
454 -#[derive(Debug, Clone)]
455 -pub struct StockBar {
456 -    pub ticker_id: i64,
457 -    pub timestamp: i64,
458 -    pub day_index: usize,
459 -    pub open: f32,
460 -    pub high: f32,
461 -    pub low: f32,
462 -    pub close: f32,
463 -    pub volume: f64,
464 -    pub returns: f32,
465 -}
466 -
467 -/// Market regime (for regime-switching simulation).
468 -#[derive(Debug, Clone, Copy, PartialEq, Eq)]
469 -enum MarketRegime {
470 -    Bull,
471 -    Bear,
472 -    Sideways,
473 -}
474 -
475 -/// Random number generator with seeded state.
476 -struct RandomGenerator {
477 -    state: u64,
478 -}
479 -
480 -impl RandomGenerator {
481 -    fn new(seed: u64) -> Self {
482 -        Self { state: seed }
483 -    }
484 -
485 -    fn next_u64(&mut self) -> u64 {
486 -        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
487 -        self.state
488 -    }
489 -
490 -    fn uniform(&mut self) -> f32 {
491 -        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
492 -    }
493 -
494 -    fn normal(&mut self) -> f32 {
495 -        // Box-Muller transform
496 -        let u1 = self.uniform().max(1e-10);
497 -        let u2 = self.uniform();
498 -        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
499 -    }
500 -
501 -    fn choice<T: Clone>(&mut self, items: &[T]) -> T {
502 -        let idx = (self.uniform() * items.len() as f32) as usize % items.len();
503 -        items[idx].clone()
504 -    }
505 -
506 -    #[allow(dead_code)]
507 -    fn range(&mut self, min: f32, max: f32) -> f32 {
508 -        min + self.uniform() * (max - min)
509 -    }
510 -
511 -    /// Shuffle a mutable slice in-place using Fisher-Yates
512 -    fn shuffle<T>(&mut self, slice: &mut [T]) {
513 -        for i in (1..slice.len()).rev() {
514 -            let j = (self.uniform() * (i + 1) as f32) as usize % (i + 1);
515 -            slice.swap(i, j);
516 -        }
517 -    }
518 -}
519 -
520 -/// Generates synthetic stock data using realistic market dynamics.
521 -pub struct StockDataGenerator {
522 -    rng: RandomGenerator,
523 -    tickers: Vec<TickerInfo>,
524 -    bars: Vec<StockBar>,
525 -    current_regime: MarketRegime,
526 -    regime_counter: usize,
527 -}
528 -
529 -impl StockDataGenerator {
530 -    /// Creates a new stock data generator.
531 -    pub fn new(seed: u64) -> Self {
532 -        Self {
533 -            rng: RandomGenerator::new(seed),
534 -            tickers: Vec::new(),
535 -            bars: Vec::new(),
536 -            current_regime: MarketRegime::Sideways,
537 -            regime_counter: 0,
538 -        }
539 -    }
540 -
541 -    /// Generates ticker information.
542 -    pub fn generate_tickers(&mut self, num_tickers: usize) -> &[TickerInfo] {
543 -        self.tickers.clear();
544 -
545 -        let symbols = [
546 -            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK", "JPM", "JNJ", "V",
547 -            "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "CMCSA", "ADBE", "NFLX", "CRM", "XOM",
548 -            "CSCO", "PFE", "INTC", "KO", "PEP", "ABT", "TMO", "COST", "CVX", "NKE", "MRK", "LLY",
549 -            "WMT", "ABBV", "AVGO", "ACN", "DHR", "TXN", "MDT", "UNP", "NEE", "LIN", "ORCL", "PM",
550 -            "HON", "LOW", "AMT", "QCOM", "IBM", "RTX", "SBUX", "CVS", "GS", "BLK", "DE", "SPGI",
551 -            "AXP", "GILD", "ISRG", "MDLZ", "BA", "MMM", "CAT", "MO", "ADP", "TGT", "BKNG", "CHTR",
552 -            "ZTS", "SYK", "CI", "ANTM", "TJX", "REGN", "DUK", "SO", "USB", "PLD", "BDX", "CL",
553 -            "MU", "ATVI", "MMC", "ITW", "CME", "FISV", "HUM", "APD", "NOC", "EQIX", "ICE", "ETN",
554 -            "CCI", "NSC", "WM", "SHW", "VRTX",
555 -        ];
556 -
557 -        for i in 0..num_tickers {
558 -            let sector = self.rng.choice(Sector::all());
559 -            let symbol = if i < symbols.len() {
560 -                symbols[i].to_string()
561 -            } else {
562 -                format!("SYM{:03}", i)
563 -            };
564 -            let name = format!("{} Inc.", symbol);
565 -
566 -            // Generate realistic characteristics based on sector
567 -            let (base_vol, drift) = match sector {
568 -                Sector::Technology => (0.02 + self.rng.uniform() * 0.02, 0.0003),
569 -                Sector::Healthcare => (0.015 + self.rng.uniform() * 0.015, 0.0002),
570 -                Sector::Finance => (0.018 + self.rng.uniform() * 0.012, 0.0001),
571 -                Sector::Energy => (0.025 + self.rng.uniform() * 0.025, 0.0001),
572 -                Sector::Utilities => (0.008 + self.rng.uniform() * 0.007, 0.0001),
573 -                _ => (0.015 + self.rng.uniform() * 0.015, 0.00015),
574 -            };
575 -
576 -            let beta = match sector {
577 -                Sector::Technology => 1.2 + self.rng.uniform() * 0.5,
578 -                Sector::Utilities => 0.5 + self.rng.uniform() * 0.3,
579 -                Sector::Finance => 1.0 + self.rng.uniform() * 0.4,
580 -                _ => 0.8 + self.rng.uniform() * 0.4,
581 -            };
582 -
583 -            self.tickers.push(TickerInfo {
584 -                ticker_id: i as i64,
585 -                name,
586 -                symbol,
587 -                sector,
588 -                beta,
589 -                base_volatility: base_vol,
590 -                drift,
591 -            });
592 -        }
593 -
594 -        &self.tickers
595 -    }
596 -
597 -    /// Generates price bars using Geometric Brownian Motion with regime switching.
598 -    pub fn generate_bars(&mut self, days: usize) -> &[StockBar] {
599 -        self.bars.clear();
600 -
601 -        if self.tickers.is_empty() {
602 -            return &self.bars;
603 -        }
604 -
605 -        // Initialize prices for each ticker
606 -        let mut prices: Vec<f32> = self
607 -            .tickers
608 -            .iter()
609 -            .map(|_| 50.0 + self.rng.uniform() * 150.0) // $50-$200 initial price
610 -            .collect();
611 -
612 -        let mut volatilities: Vec<f32> = self.tickers.iter().map(|t| t.base_volatility).collect();
613 -
614 -        // Generate market factor (common shock)
615 -        let mut market_returns: Vec<f32> = Vec::with_capacity(days);
616 -
617 -        for day in 0..days {
618 -            // Update market regime periodically
619 -            self.update_regime();
620 -
621 -            // Market return based on regime
622 -            let market_return = match self.current_regime {
623 -                MarketRegime::Bull => 0.0005 + self.rng.normal() * 0.01,
624 -                MarketRegime::Bear => -0.0003 + self.rng.normal() * 0.015,
625 -                MarketRegime::Sideways => self.rng.normal() * 0.008,
626 -            };
627 -            market_returns.push(market_return);
628 -
629 -            // Check for earnings events (higher volatility)
630 -            let is_earnings_season = (day % 63) < 21; // ~1/3 of each quarter
631 -
632 -            for (ticker_idx, ticker) in self.tickers.iter().enumerate() {
633 -                // Mean-reverting volatility
634 -                let target_vol = ticker.base_volatility;
635 -                volatilities[ticker_idx] = volatilities[ticker_idx] * 0.95 + target_vol * 0.05;
636 -
637 -                // Add earnings spike
638 -                let vol = if is_earnings_season && self.rng.uniform() < 0.1 {
639 -                    volatilities[ticker_idx] * 2.0
640 -                } else {
641 -                    volatilities[ticker_idx]
642 -                };
643 -
644 -                // GBM: dS = S * (mu*dt + sigma*dW)
645 -                // With market factor: r = beta * market_return + idiosyncratic
646 -                let idiosyncratic = self.rng.normal() * vol;
647 -                let daily_return = ticker.beta * market_return + idiosyncratic + ticker.drift;
648 -
649 -                // Update price
650 -                let prev_price = prices[ticker_idx];
651 -                let new_price = prev_price * (1.0 + daily_return);
652 -                prices[ticker_idx] = new_price.max(0.01); // Floor at $0.01
653 -
654 -                // Generate OHLC from close
655 -                let intraday_vol = vol * 0.5;
656 -                let open = prev_price * (1.0 + self.rng.normal() * intraday_vol * 0.3);
657 -                let high = new_price.max(open) * (1.0 + self.rng.uniform() * intraday_vol);
658 -                let low = new_price.min(open) * (1.0 - self.rng.uniform() * intraday_vol);
659 -
660 -                // Generate volume (log-normal distribution)
661 -                let base_volume = 1_000_000.0 + self.rng.uniform() as f64 * 10_000_000.0;
662 -                let volume_mult = (1.0 + daily_return.abs() * 10.0) as f64; // Higher volume on bigger moves
663 -                let volume = base_volume * volume_mult * (0.5 + self.rng.uniform() as f64);
664 -
665 -                self.bars.push(StockBar {
666 -                    ticker_id: ticker.ticker_id,
667 -                    timestamp: day as i64,
668 -                    day_index: day,
669 -                    open: open.max(0.01),
670 -                    high: high.max(open.max(new_price)),
671 -                    low: low.min(open.min(new_price)).max(0.01),
672 -                    close: new_price,
673 -                    volume,
674 -                    returns: daily_return,
675 -                });
676 -            }
677 -        }
678 -
679 -        &self.bars
680 -    }
681 -
682 -    fn update_regime(&mut self) {
683 -        self.regime_counter += 1;
684 -
685 -        // Regime persists for 20-60 days on average
686 -        if self.regime_counter > 20 && self.rng.uniform() < 0.05 {
687 -            self.current_regime = match self.rng.uniform() {
688 -                x if x < 0.4 => MarketRegime::Bull,
689 -                x if x < 0.7 => MarketRegime::Sideways,
690 -                _ => MarketRegime::Bear,
691 -            };
692 -            self.regime_counter = 0;
693 -        }
694 -    }
695 -
696 -    /// Returns the generated tickers.
697 -    pub fn tickers(&self) -> &[TickerInfo] {
698 -        &self.tickers
699 -    }
700 -
701 -    /// Returns the generated bars.
702 -    pub fn bars(&self) -> &[StockBar] {
703 -        &self.bars
704 -    }
705 -
706 -    /// Gets bars for a specific ticker.
707 -    pub fn get_ticker_bars(&self, ticker_id: i64) -> Vec<&StockBar> {
708 -        self.bars
709 -            .iter()
710 -            .filter(|b| b.ticker_id == ticker_id)
711 -            .collect()
712 -    }
713 -}
714 -
715 -// =============================================================================
716 -// Section 2b: CSV Data Loading (for real stock data)
717 -// =============================================================================
718 -
719 -/// Load stock data from CSV files (Kaggle format or Yahoo Finance format).
720 -///
721 -/// Supports the following formats:
722 -/// 1. Kaggle Stock Market Dataset: Date,Open,High,Low,Close,Adj Close,Volume
723 -/// 2. Yahoo Finance: Date,Open,High,Low,Close,Adj Close,Volume
724 -/// 3. Simple OHLCV: Date,Open,High,Low,Close,Volume
725 -///
726 -/// # Usage with Kaggle Data
727 -///
728 -/// ```bash
729 -/// # Download from https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
730 -/// # Extract to a directory, then run:
731 -/// cargo run -p monolith-examples --bin stock_prediction -- \
732 -///     --mode train --data-dir /path/to/kaggle-data/stocks
733 -/// ```
734 -pub struct CsvDataLoader {
735 -    /// Base directory containing CSV files
736 -    data_dir: String,
737 -    /// Mapping from ticker symbols to IDs
738 -    ticker_to_id: HashMap<String, i64>,
739 -    /// Ticker info list
740 -    tickers: Vec<TickerInfo>,
741 -    /// All loaded bars
742 -    bars: Vec<StockBar>,
743 -}
744 -
745 -impl CsvDataLoader {
746 -    /// Creates a new CSV data loader.
747 -    pub fn new(data_dir: &str) -> Self {
748 -        Self {
749 -            data_dir: data_dir.to_string(),
750 -            ticker_to_id: HashMap::new(),
751 -            tickers: Vec::new(),
752 -            bars: Vec::new(),
753 -        }
754 -    }
755 -
756 -    /// Loads data from all CSV files in the directory.
757 -    ///
758 -    /// # Arguments
759 -    /// * `max_tickers` - Maximum number of tickers to load (for memory management)
760 -    /// * `min_days` - Minimum days of history required per ticker
761 -    pub fn load(&mut self, max_tickers: usize, min_days: usize) -> Result<(), String> {
762 -        use std::fs;
763 -        use std::io::{BufRead, BufReader};
764 -
765 -        let entries = fs::read_dir(&self.data_dir)
766 -            .map_err(|e| format!("Failed to read directory {}: {}", self.data_dir, e))?;
767 -
768 -        let mut csv_files: Vec<_> = entries
769 -            .filter_map(|e| e.ok())
770 -            .filter(|e| {
771 -                e.path()
772 -                    .extension()
773 -                    .map(|ext| ext == "csv")
774 -                    .unwrap_or(false)
775 -            })
776 -            .collect();
777 -
778 -        // Sort for reproducibility
779 -        csv_files.sort_by(|a, b| a.path().cmp(&b.path()));
780 -
781 -        // Limit number of tickers
782 -        csv_files.truncate(max_tickers);
783 -
784 -        println!(
785 -            "  Loading {} CSV files from {}",
786 -            csv_files.len(),
787 -            self.data_dir
788 -        );
789 -
790 -        // Sector inference based on common ticker patterns
791 -        let sector_map: HashMap<&str, &str> = [
792 -            ("AAPL", "Technology"),
793 -            ("MSFT", "Technology"),
794 -            ("GOOGL", "Technology"),
795 -            ("AMZN", "Consumer"),
796 -            ("META", "Technology"),
797 -            ("NVDA", "Technology"),
798 -            ("TSLA", "Consumer"),
799 -            ("JPM", "Financials"),
800 -            ("BAC", "Financials"),
801 -            ("WMT", "Consumer"),
802 -            ("JNJ", "Healthcare"),
803 -            ("PFE", "Healthcare"),
804 -            ("XOM", "Energy"),
805 -            ("CVX", "Energy"),
806 -        ]
807 -        .into_iter()
808 -        .collect();
809 -
810 -        for (ticker_id, entry) in csv_files.iter().enumerate() {
811 -            let path = entry.path();
812 -            let ticker_symbol = path
813 -                .file_stem()
814 -                .and_then(|s| s.to_str())
815 -                .unwrap_or("UNKNOWN")
816 -                .to_uppercase();
817 -
818 -            let file = fs::File::open(&path)
819 -                .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
820 -            let reader = BufReader::new(file);
821 -
822 -            let mut lines = reader.lines();
823 -            let header = lines.next();
824 -
825 -            // Detect format from header
826 -            let format = if let Some(Ok(h)) = &header {
827 -                let h_lower = h.to_lowercase();
828 -                if h_lower.contains("adj close") {
829 -                    "yahoo" // Date,Open,High,Low,Close,Adj Close,Volume
830 -                } else if h_lower.contains("volume") {
831 -                    "simple" // Date,Open,High,Low,Close,Volume
832 -                } else {
833 -                    "unknown"
834 -                }
835 -            } else {
836 -                continue;
837 -            };
838 -
839 -            if format == "unknown" {
840 -                eprintln!("  Skipping {}: unknown format", path.display());
841 -                continue;
842 -            }
843 -
844 -            let mut ticker_bars: Vec<StockBar> = Vec::new();
845 -            let mut prev_close: Option<f32> = None;
846 -
847 -            for (day_idx, line) in lines.enumerate() {
848 -                if let Ok(line) = line {
849 -                    let fields: Vec<&str> = line.split(',').collect();
850 -                    if fields.len() < 6 {
851 -                        continue;
852 -                    }
853 -
854 -                    // Parse fields based on format
855 -                    let (open, high, low, close, volume) = match format {
856 -                        "yahoo" | "simple" => {
857 -                            let o: f32 = fields[1].parse().unwrap_or(0.0);
858 -                            let h: f32 = fields[2].parse().unwrap_or(0.0);
859 -                            let l: f32 = fields[3].parse().unwrap_or(0.0);
860 -                            let c: f32 = fields[4].parse().unwrap_or(0.0);
861 -                            let v: f64 = if format == "yahoo" {
862 -                                fields[6].parse().unwrap_or(0.0)
863 -                            } else {
864 -                                fields[5].parse().unwrap_or(0.0)
865 -                            };
866 -                            (o, h, l, c, v)
867 -                        }
868 -                        _ => continue,
869 -                    };
870 -
871 -                    // Skip invalid bars
872 -                    if close <= 0.0 || open <= 0.0 {
873 -                        continue;
874 -                    }
875 -
876 -                    // Calculate returns
877 -                    let returns = if let Some(pc) = prev_close {
878 -                        (close - pc) / pc
879 -                    } else {
880 -                        0.0
881 -                    };
882 -                    prev_close = Some(close);
883 -
884 -                    // Parse date to timestamp (days since epoch, simplified)
885 -                    let timestamp = day_idx as i64;
886 -
887 -                    ticker_bars.push(StockBar {
888 -                        ticker_id: ticker_id as i64,
889 -                        timestamp,
890 -                        day_index: day_idx,
891 -                        open,
892 -                        high,
893 -                        low,
894 -                        close,
895 -                        volume,
896 -                        returns,
897 -                    });
898 -                }
899 -            }
900 -
901 -            // Skip tickers with insufficient history
902 -            if ticker_bars.len() < min_days {
903 -                continue;
904 -            }
905 -
906 -            // Infer sector
907 -            let sector = match sector_map.get(ticker_symbol.as_str()).copied() {
908 -                Some("Technology") => Sector::Technology,
909 -                Some("Healthcare") => Sector::Healthcare,
910 -                Some("Financials") | Some("Finance") => Sector::Finance,
911 -                Some("Consumer") => Sector::Consumer,
912 -                Some("Industrials") | Some("Industrial") => Sector::Industrial,
913 -                Some("Energy") => Sector::Energy,
914 -                Some("Materials") => Sector::Materials,
915 -                Some("Real Estate") => Sector::RealEstate,
916 -                Some("Utilities") => Sector::Utilities,
917 -                Some("Communications") => Sector::Communications,
918 -                _ => {
919 -                    // Default sector based on first letter (very rough heuristic)
920 -                    match ticker_symbol.chars().next() {
921 -                        Some('A'..='F') => Sector::Technology,
922 -                        Some('G'..='L') => Sector::Industrial,
923 -                        Some('M'..='R') => Sector::Healthcare,
924 -                        Some('S'..='Z') => Sector::Finance,
925 -                        _ => Sector::Consumer,
926 -                    }
927 -                }
928 -            };
929 -
930 -            // Calculate volatility from returns
931 -            let returns: Vec<f32> = ticker_bars.iter().map(|b| b.returns).collect();
932 -            let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
933 -            let variance: f32 =
934 -                returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>() / returns.len() as f32;
935 -            let volatility = variance.sqrt() * (252.0_f32).sqrt(); // Annualized
936 -
937 -            // Estimate drift from mean returns
938 -            let drift = mean_ret * 252.0; // Annualized
939 -
940 -            self.ticker_to_id
941 -                .insert(ticker_symbol.clone(), ticker_id as i64);
942 -
943 -            self.tickers.push(TickerInfo {
944 -                ticker_id: ticker_id as i64,
945 -                name: ticker_symbol.clone(),
946 -                symbol: ticker_symbol,
947 -                sector,
948 -                beta: 1.0, // Would need market data to calculate
949 -                base_volatility: volatility,
950 -                drift,
951 -            });
952 -
953 -            self.bars.extend(ticker_bars);
954 -        }
955 -
956 -        println!(
957 -            "  Loaded {} tickers with {} total bars",
958 -            self.tickers.len(),
959 -            self.bars.len()
960 -        );
961 -
962 -        if self.tickers.is_empty() {
963 -            return Err("No valid tickers found in the data directory".to_string());
964 -        }
965 -
966 -        Ok(())
967 -    }
968 -
969 -    /// Returns loaded tickers.
970 -    pub fn tickers(&self) -> &[TickerInfo] {
971 -        &self.tickers
972 -    }
973 -
974 -    /// Returns loaded bars.
975 -    pub fn bars(&self) -> &[StockBar] {
976 -        &self.bars
977 -    }
978 -
979 -    /// Gets bars for a specific ticker.
980 -    pub fn get_ticker_bars(&self, ticker_id: i64) -> Vec<&StockBar> {
981 -        self.bars
982 -            .iter()
983 -            .filter(|b| b.ticker_id == ticker_id)
984 -            .collect()
985 -    }
986 -}
987 -
988 -// =============================================================================
989 -// Section 2c: Intraday Data Sources
990 -// =============================================================================
991 -
992 -/// Available intraday data sources from GitHub.
993 -///
994 -/// These datasets provide 1-minute OHLCV data for stocks and indices.
995 -pub struct IntradayDataSources;
996 -
997 -impl IntradayDataSources {
998 -    /// Returns URLs for FutureSharks financial-data repository.
999 -    /// Source: https://github.com/FutureSharks/financial-data
1000 -    ///
1001 -    /// Available instruments:
1002 -    /// - S&P 500 (SPXUSD): 2010-2018
1003 -    /// - NIKKEI 225 (JPXJPY): 2010-2018
1004 -    /// - DAX 30 (GRXEUR): 2010-2018
1005 -    /// - EUROSTOXX 50 (ETXEUR): 2010-2018
1006 -    pub fn futuresharks_urls() -> Vec<(&'static str, &'static str)> {
1007 -        vec![
1008 -            ("SPX500", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/SPX500_USD_M1.csv.gz"),
1009 -            ("NAS100", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/NAS100_USD_M1.csv.gz"),
1010 -            ("JP225", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/JP225_USD_M1.csv.gz"),
1011 -            ("DE30", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/DE30_EUR_M1.csv.gz"),
1012 -            ("UK100", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/UK100_GBP_M1.csv.gz"),
1013 -        ]
1014 -    }
1015 -
1016 -    /// Prints download instructions for getting intraday data.
1017 -    pub fn print_download_instructions() {
1018 -        println!("\n=== Intraday Data Download Instructions ===\n");
1019 -        println!("Option 1: FutureSharks financial-data (1-minute bars, 2010-2018)");
1020 -        println!("  git clone https://github.com/FutureSharks/financial-data.git");
1021 -        println!("  # Then use --data-dir financial-data/data/stocks/oanda\n");
1022 -
1023 -        println!("Option 2: Download individual files:");
1024 -        for (name, url) in Self::futuresharks_urls() {
1025 -            println!("  # {}", name);
1026 -            println!("  curl -L {} | gunzip > {}.csv", url, name);
1027 -        }
1028 -
1029 -        println!("\nOption 3: Yahoo Finance intraday (last 60 days, via Python):");
1030 -        println!("  pip install yfinance");
1031 -        println!("  python -c \"import yfinance as yf; yf.download('AAPL', period='60d', interval='1m').to_csv('AAPL_1m.csv')\"");
1032 -
1033 -        println!("\nOption 4: Alpha Vantage API (free key required):");
1034 -        println!("  curl 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&outputsize=full&apikey=YOUR_KEY&datatype=csv' > AAPL_intraday.csv");
1035 -
1036 -        println!("\n===========================================\n");
1037 -    }
1038 -
1039 -    /// Downloads a single intraday CSV file using system curl.
1040 -    /// Returns the path to the downloaded file.
1041 -    #[cfg(unix)]
1042 -    pub fn download_file(url: &str, output_dir: &str, filename: &str) -> Result<String, String> {
1043 -        use std::process::Command;
1044 -
1045 -        // Create output directory if it doesn't exist
1046 -        std::fs::create_dir_all(output_dir)
1047 -            .map_err(|e| format!("Failed to create directory: {}", e))?;
1048 -
1049 -        let output_path = format!("{}/{}", output_dir, filename);
1050 -
1051 -        // Check if file already exists
1052 -        if std::path::Path::new(&output_path).exists() {
1053 -            println!("  File already exists: {}", output_path);
1054 -            return Ok(output_path);
1055 -        }
1056 -
1057 -        println!("  Downloading {} to {}...", url, output_path);
1058 -
1059 -        // Use curl to download, handling gzip if needed
1060 -        let result = if url.ends_with(".gz") {
1061 -            Command::new("sh")
1062 -                .arg("-c")
1063 -                .arg(format!("curl -sL '{}' | gunzip > '{}'", url, output_path))
1064 -                .output()
1065 -        } else {
1066 -            Command::new("curl")
1067 -                .args(["-sL", "-o", &output_path, url])
1068 -                .output()
1069 -        };
1070 -
1071 -        match result {
1072 -            Ok(output) => {
1073 -                if output.status.success() {
1074 -                    println!("  Downloaded successfully: {}", output_path);
1075 -                    Ok(output_path)
1076 -                } else {
1077 -                    let stderr = String::from_utf8_lossy(&output.stderr);
1078 -                    Err(format!("Download failed: {}", stderr))
1079 -                }
1080 -            }
1081 -            Err(e) => Err(format!("Failed to execute curl: {}", e)),
1082 -        }
1083 -    }
1084 -
1085 -    /// Downloads all FutureSharks intraday data files.
1086 -    #[cfg(unix)]
1087 -    pub fn download_all_futuresharks(output_dir: &str) -> Result<Vec<String>, String> {
1088 -        let mut downloaded = Vec::new();
1089 -
1090 -        println!("Downloading FutureSharks intraday data...");
1091 -
1092 -        for (name, url) in Self::futuresharks_urls() {
1093 -            let filename = format!("{}_1m.csv", name);
1094 -            match Self::download_file(url, output_dir, &filename) {
1095 -                Ok(path) => downloaded.push(path),
1096 -                Err(e) => eprintln!("  Warning: Failed to download {}: {}", name, e),
1097 -            }
1098 -        }
1099 -
1100 -        if downloaded.is_empty() {
1101 -            Err("No files were downloaded".to_string())
1102 -        } else {
1103 -            println!("Downloaded {} files to {}", downloaded.len(), output_dir);
1104 -            Ok(downloaded)
1105 -        }
1106 -    }
1107 -}
1108 -
1109 -/// Loads intraday data from FutureSharks CSV format.
1110 -/// Format: DateTime,Open,High,Low,Close,Volume
1111 -pub struct IntradayCsvLoader {
1112 -    tickers: Vec<TickerInfo>,
1113 -    bars: Vec<StockBar>,
1114 -}
1115 -
1116 -impl IntradayCsvLoader {
1117 -    pub fn new() -> Self {
1118 -        Self {
1119 -            tickers: Vec::new(),
1120 -            bars: Vec::new(),
1121 -        }
1122 -    }
1123 -
1124 -    /// Loads a single intraday CSV file.
1125 -    /// Aggregates 1-minute bars into the specified timeframe (e.g., 5, 15, 60 minutes).
1126 -    pub fn load_file(
1127 -        &mut self,
1128 -        path: &str,
1129 -        ticker_name: &str,
1130 -        aggregate_minutes: usize,
1131 -    ) -> Result<(), String> {
1132 -        use std::fs::File;
1133 -        use std::io::{BufRead, BufReader};
1134 -
1135 -        let file = File::open(path).map_err(|e| format!("Failed to open {}: {}", path, e))?;
1136 -        let reader = BufReader::new(file);
1137 -
1138 -        let ticker_id = self.tickers.len() as i64;
1139 -        let mut minute_bars: Vec<(f32, f32, f32, f32, f64)> = Vec::new(); // OHLCV
1140 -
1141 -        for (i, line) in reader.lines().enumerate() {
1142 -            if i == 0 {
1143 -                continue; // Skip header
1144 -            }
1145 -
1146 -            let line = line.map_err(|e| format!("Read error: {}", e))?;
1147 -            let fields: Vec<&str> = line.split(',').collect();
1148 -
1149 -            if fields.len() < 6 {
1150 -                continue;
1151 -            }
1152 -
1153 -            // Parse OHLCV (skip datetime at index 0)
1154 -            let open: f32 = fields[1].parse().unwrap_or(0.0);
1155 -            let high: f32 = fields[2].parse().unwrap_or(0.0);
1156 -            let low: f32 = fields[3].parse().unwrap_or(0.0);
1157 -            let close: f32 = fields[4].parse().unwrap_or(0.0);
1158 -            let volume: f64 = fields[5].parse().unwrap_or(0.0);
1159 -
1160 -            if close <= 0.0 {
1161 -                continue;
1162 -            }
1163 -
1164 -            minute_bars.push((open, high, low, close, volume));
1165 -        }
1166 -
1167 -        if minute_bars.is_empty() {
1168 -            return Err(format!("No valid bars in {}", path));
1169 -        }
1170 -
1171 -        // Aggregate into larger timeframes
1172 -        let aggregated = Self::aggregate_bars(&minute_bars, aggregate_minutes);
1173 -
1174 -        // Create StockBars
1175 -        let mut prev_close = aggregated[0].3;
1176 -        for (day_idx, (open, high, low, close, volume)) in aggregated.iter().enumerate() {
1177 -            let returns = (*close - prev_close) / prev_close;
1178 -            prev_close = *close;
1179 -
1180 -            self.bars.push(StockBar {
1181 -                ticker_id,
1182 -                timestamp: day_idx as i64,
1183 -                day_index: day_idx,
1184 -                open: *open,
1185 -                high: *high,
1186 -                low: *low,
1187 -                close: *close,
1188 -                volume: *volume,
1189 -                returns,
1190 -            });
1191 -        }
1192 -
1193 -        // Calculate volatility
1194 -        let returns: Vec<f32> = self
1195 -            .bars
1196 -            .iter()
1197 -            .filter(|b| b.ticker_id == ticker_id)
1198 -            .map(|b| b.returns)
1199 -            .collect();
1200 -        let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len().max(1) as f32;
1201 -        let variance: f32 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>()
1202 -            / returns.len().max(1) as f32;
1203 -        let volatility = variance.sqrt() * (252.0_f32 * (390.0 / aggregate_minutes as f32)).sqrt();
1204 -
1205 -        self.tickers.push(TickerInfo {
1206 -            ticker_id,
1207 -            name: ticker_name.to_string(),
1208 -            symbol: ticker_name.to_string(),
1209 -            sector: Sector::Technology, // Index default
1210 -            beta: 1.0,
1211 -            base_volatility: volatility,
1212 -            drift: mean_ret * 252.0 * (390.0 / aggregate_minutes as f32),
1213 -        });
1214 -
1215 -        println!(
1216 -            "  Loaded {} bars from {} ({}m aggregation)",
1217 -            self.bars
1218 -                .iter()
1219 -                .filter(|b| b.ticker_id == ticker_id)
1220 -                .count(),
1221 -            ticker_name,
1222 -            aggregate_minutes
1223 -        );
1224 -
1225 -        Ok(())
1226 -    }
1227 -
1228 -    /// Aggregates 1-minute bars into larger timeframes.
1229 -    fn aggregate_bars(
1230 -        minute_bars: &[(f32, f32, f32, f32, f64)],
1231 -        period: usize,
1232 -    ) -> Vec<(f32, f32, f32, f32, f64)> {
1233 -        minute_bars
1234 -            .chunks(period)
1235 -            .filter(|chunk| !chunk.is_empty())
1236 -            .map(|chunk| {
1237 -                let open = chunk[0].0;
1238 -                let high = chunk.iter().map(|b| b.1).fold(f32::MIN, f32::max);
1239 -                let low = chunk.iter().map(|b| b.2).fold(f32::MAX, f32::min);
1240 -                let close = chunk.last().unwrap().3;
1241 -                let volume: f64 = chunk.iter().map(|b| b.4).sum();
1242 -                (open, high, low, close, volume)
1243 -            })
1244 -            .collect()
1245 -    }
1246 -
1247 -    pub fn tickers(&self) -> &[TickerInfo] {
1248 -        &self.tickers
1249 -    }
1250 -
1251 -    pub fn bars(&self) -> &[StockBar] {
1252 -        &self.bars
1253 -    }
1254 -}
1255 -
1256 -/// Loader for the FutureSharks "histdata" intraday dataset.
1257 -///
1258 -/// Example file line format (semicolon-separated, no header):
1259 -/// `20170102 180000;2241.000000;2244.500000;2241.000000;2243.500000;0`
1260 -///
1261 -/// The repo organizes by instrument symbol (e.g. SPXUSD) and year-suffixed CSVs.
1262 -pub struct FutureSharksHistdataLoader {
1263 -    tickers: Vec<TickerInfo>,
1264 -    bars: Vec<StockBar>,
1265 -}
1266 -
1267 -impl FutureSharksHistdataLoader {
1268 -    pub fn new() -> Self {
1269 -        Self {
1270 -            tickers: Vec::new(),
1271 -            bars: Vec::new(),
1272 -        }
1273 -    }
1274 -
1275 -    pub fn tickers(&self) -> &[TickerInfo] {
1276 -        &self.tickers
1277 -    }
1278 -
1279 -    pub fn bars(&self) -> &[StockBar] {
1280 -        &self.bars
1281 -    }
1282 -
1283 -    /// Load up to `max_instruments` instruments from a histdata directory.
1284 -    ///
1285 -    /// `dir` should typically be:
1286 -    /// `data/financial-data/pyfinancialdata/data/stocks/histdata`
1287 -    pub fn load_dir(&mut self, dir: &str, max_instruments: usize) -> Result<(), String> {
1288 -        use std::fs;
1289 -
1290 -        let mut instrument_dirs: Vec<_> = fs::read_dir(dir)
1291 -            .map_err(|e| format!("Failed to read {}: {}", dir, e))?
1292 -            .filter_map(|e| e.ok())
1293 -            .filter(|e| e.path().is_dir())
1294 -            .collect();
1295 -
1296 -        instrument_dirs.sort_by(|a, b| a.path().cmp(&b.path()));
1297 -        instrument_dirs.truncate(max_instruments);
1298 -
1299 -        if instrument_dirs.is_empty() {
1300 -            return Err(format!("No instrument subdirectories found in {}", dir));
1301 -        }
1302 -
1303 -        for entry in instrument_dirs {
1304 -            let instrument = entry
1305 -                .path()
1306 -                .file_name()
1307 -                .and_then(|s| s.to_str())
1308 -                .unwrap_or("UNKNOWN")
1309 -                .to_string();
1310 -            self.load_instrument_dir(entry.path().to_string_lossy().as_ref(), &instrument)?;
1311 -        }
1312 -
1313 -        if self.tickers.is_empty() || self.bars.is_empty() {
1314 -            return Err(format!("No valid histdata bars found in {}", dir));
1315 -        }
1316 -
1317 -        Ok(())
1318 -    }
1319 -
1320 -    fn load_instrument_dir(&mut self, instrument_dir: &str, symbol: &str) -> Result<(), String> {
1321 -        use std::fs;
1322 -        use std::io::{BufRead, BufReader};
1323 -
1324 -        let mut files: Vec<_> = fs::read_dir(instrument_dir)
1325 -            .map_err(|e| format!("Failed to read {}: {}", instrument_dir, e))?
1326 -            .filter_map(|e| e.ok())
1327 -            .filter(|e| {
1328 -                e.path()
1329 -                    .extension()
1330 -                    .and_then(|s| s.to_str())
1331 -                    .map(|ext| ext.eq_ignore_ascii_case("csv"))
1332 -                    .unwrap_or(false)
1333 -            })
1334 -            .collect();
1335 -
1336 -        files.sort_by(|a, b| a.path().cmp(&b.path()));
1337 -
1338 -        if files.is_empty() {
1339 -            return Err(format!("No CSV files found in {}", instrument_dir));
1340 -        }
1341 -
1342 -        let ticker_id = self.tickers.len() as i64;
1343 -        let mut bars_for_ticker: Vec<StockBar> = Vec::new();
1344 -        let mut prev_close: Option<f32> = None;
1345 -
1346 -        for entry in files {
1347 -            let path = entry.path();
1348 -            let file = fs::File::open(&path)
1349 -                .map_err(|e| format!("Failed to open {}: {}", path.display(), e))?;
1350 -            let reader = BufReader::new(file);
1351 -
1352 -            for line in reader.lines().flatten() {
1353 -                // Format is semicolon separated: "<date time>;<open>;<high>;<low>;<close>;<volume>"
1354 -                let mut parts = line.split(';');
1355 -                let ts = parts.next().unwrap_or("");
1356 -                let open: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
1357 -                let high: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
1358 -                let low: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
1359 -                let close: f32 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
1360 -                let volume: f64 = parts.next().unwrap_or("0").parse().unwrap_or(0.0);
1361 -
1362 -                if open <= 0.0 || close <= 0.0 || !open.is_finite() || !close.is_finite() {
1363 -                    continue;
1364 -                }
1365 -
1366 -                // Very lightweight timestamp parsing: YYYYMMDD HHMMSS -> monotonic minute index.
1367 -                // We keep it deterministic and only need ordering for our "time split".
1368 -                let (date_part, time_part) = match ts.split_once(' ') {
1369 -                    Some((d, t)) => (d, t),
1370 -                    None => continue,
1371 -                };
1372 -                if date_part.len() != 8 || time_part.len() < 4 {
1373 -                    continue;
1374 -                }
1375 -                let day: i64 = date_part.parse().unwrap_or(0);
1376 -                let hhmm: i64 = time_part[..4].parse().unwrap_or(0);
1377 -
1378 -                // Not a real epoch; just "sortable".
1379 -                let timestamp = day * 10_000 + hhmm;
1380 -
1381 -                let returns = if let Some(pc) = prev_close {
1382 -                    (close - pc) / pc
1383 -                } else {
1384 -                    0.0
1385 -                };
1386 -                prev_close = Some(close);
1387 -
1388 -                let day_index = bars_for_ticker.len();
1389 -
1390 -                bars_for_ticker.push(StockBar {
1391 -                    ticker_id,
1392 -                    timestamp,
1393 -                    day_index,
1394 -                    open,
1395 -                    high,
1396 -                    low,
1397 -                    close,
1398 -                    volume,
1399 -                    returns,
1400 -                });
1401 -            }
1402 -        }
1403 -
1404 -        if bars_for_ticker.is_empty() {
1405 -            return Err(format!("No valid bars parsed for {}", symbol));
1406 -        }
1407 -
1408 -        // Calculate volatility/drift from returns.
1409 -        let returns: Vec<f32> = bars_for_ticker.iter().map(|b| b.returns).collect();
1410 -        let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len().max(1) as f32;
1411 -        let variance: f32 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>()
1412 -            / returns.len().max(1) as f32;
1413 -        // 252 trading days * 390 minutes. For "index" style data this is a rough scaling.
1414 -        let annualization = (252.0_f32 * 390.0_f32).sqrt();
1415 -        let volatility = variance.sqrt() * annualization;
1416 -
1417 -        self.tickers.push(TickerInfo {
1418 -            ticker_id,
1419 -            name: symbol.to_string(),
1420 -            symbol: symbol.to_string(),
1421 -            sector: Sector::Technology,
1422 -            beta: 1.0,
1423 -            base_volatility: volatility,
1424 -            drift: mean_ret * 252.0 * 390.0,
1425 -        });
1426 -
1427 -        self.bars.extend(bars_for_ticker);
1428 -
1429 -        Ok(())
1430 -    }
1431 -}
1432 -
1433 -// =============================================================================
1434 -// Section 3: Technical Indicator Computation
1435 -// =============================================================================
1436 -
1437 -/// Technical indicators computed for each bar.
1438 -#[derive(Debug, Clone, Default)]
1439 -pub struct TechnicalIndicators {
1440 -    // Trend indicators
1441 -    pub sma_5: f32,
1442 -    pub sma_20: f32,
1443 -    pub sma_50: f32,
1444 -    pub ema_12: f32,
1445 -    pub ema_26: f32,
1446 -
1447 -    // Momentum indicators
1448 -    pub rsi_14: f32,
1449 -    pub macd: f32,
1450 -    pub macd_signal: f32,
1451 -    pub macd_histogram: f32,
1452 -    pub roc_10: f32, // Rate of Change (10-period)
1453 -    pub roc_20: f32, // Rate of Change (20-period)
1454 -
1455 -    // Stochastic oscillator
1456 -    pub stoch_k: f32, // Slow Stochastic %K
1457 -    pub stoch_d: f32, // Slow Stochastic %D
1458 -
1459 -    // Volatility indicators
1460 -    pub bollinger_upper: f32,
1461 -    pub bollinger_lower: f32,
1462 -    pub bollinger_width: f32,
1463 -    pub atr_14: f32,
1464 -    pub stddev_20: f32, // Standard Deviation (20-period)
1465 -
1466 -    // Volume indicators
1467 -    pub volume_sma_20: f32,
1468 -    pub volume_ratio: f32,
1469 -    pub obv: f32,    // On Balance Volume (from ta crate)
1470 -    pub mfi_14: f32, // Money Flow Index (14-period)
1471 -
1472 -    // Range indicators
1473 -    pub max_14: f32,         // Maximum (14-period high)
1474 -    pub min_14: f32,         // Minimum (14-period low)
1475 -    pub price_vs_range: f32, // Position within max/min range
1476 -
1477 -    // Price patterns
1478 -    pub price_vs_sma20: f32,
1479 -    pub price_vs_bollinger: f32,
1480 -    pub high_low_range: f32,
1481 -    pub body_ratio: f32,
1482 -    pub upper_shadow: f32,
1483 -    pub lower_shadow: f32,
1484 -}
1485 -
1486 -impl TechnicalIndicators {
1487 -    /// Converts indicators to a feature vector.
1488 -    /// Contains 33 features from all ta crate indicators.
1489 -    pub fn to_vec(&self) -> Vec<f32> {
1490 -        vec![
1491 -            // Trend (5)
1492 -            self.sma_5,
1493 -            self.sma_20,
1494 -            self.sma_50,
1495 -            self.ema_12,
1496 -            self.ema_26,
1497 -            // Momentum (6)
1498 -            self.rsi_14,
1499 -            self.macd,
1500 -            self.macd_signal,
1501 -            self.macd_histogram,
1502 -            self.roc_10,
1503 -            self.roc_20,
1504 -            // Stochastic (2)
1505 -            self.stoch_k,
1506 -            self.stoch_d,
1507 -            // Volatility (5)
1508 -            self.bollinger_upper,
1509 -            self.bollinger_lower,
1510 -            self.bollinger_width,
1511 -            self.atr_14,
1512 -            self.stddev_20,
1513 -            // Volume (4)
1514 -            self.volume_sma_20,
1515 -            self.volume_ratio,
1516 -            self.obv,
1517 -            self.mfi_14,
1518 -            // Range (3)
1519 -            self.max_14,
1520 -            self.min_14,
1521 -            self.price_vs_range,
1522 -            // Price patterns (6)
1523 -            self.price_vs_sma20,
1524 -            self.price_vs_bollinger,
1525 -            self.high_low_range,
1526 -            self.body_ratio,
1527 -            self.upper_shadow,
1528 -            self.lower_shadow,
1529 -            // Padding for alignment (5) - total 36 features
1530 -            0.0,
1531 -            0.0,
1532 -            0.0,
1533 -            0.0,
1534 -            0.0,
1535 -        ]
1536 -    }
1537 -
1538 -    /// Number of features (33 real + 3 padding = 36 for alignment).
1539 -    pub const NUM_FEATURES: usize = 36;
1540 -}
1541 -
1542 -/// Computes technical indicators for a series of bars using the `ta` crate.
1543 -/// Uses ALL available indicators from the ta crate to maximize feature richness:
1544 -/// - SimpleMovingAverage (5, 20, 50 periods)
1545 -/// - ExponentialMovingAverage (12, 26 periods)
1546 -/// - RelativeStrengthIndex (14 period)
1547 -/// - BollingerBands (20 period, 2 std dev)
1548 -/// - AverageTrueRange (14 period)
1549 -/// - RateOfChange (10, 20 periods)
1550 -/// - SlowStochastic (14 period)
1551 -/// - StandardDeviation (20 period)
1552 -/// - MoneyFlowIndex (14 period)
1553 -/// - OnBalanceVolume (streaming)
1554 -/// - Maximum/Minimum (14 period)
1555 -pub struct IndicatorCalculator {
1556 -    // No state needed - ta crate handles streaming internally
1557 -}
1558 -
1559 -impl IndicatorCalculator {
1560 -    pub fn new() -> Self {
1561 -        Self {}
1562 -    }
1563 -
1564 -    /// Computes indicators for all bars of a ticker using ALL `ta` crate indicators.
1565 -    pub fn compute_indicators(&mut self, bars: &[StockBar]) -> Vec<TechnicalIndicators> {
1566 -        if bars.is_empty() {
1567 -            return Vec::new();
1568 -        }
1569 -
1570 -        let n = bars.len();
1571 -        let mut indicators = vec![TechnicalIndicators::default(); n];
1572 -
1573 -        // =====================================================================
1574 -        // Initialize ALL ta-crate indicators
1575 -        // =====================================================================
1576 -
1577 -        // Trend indicators - SMAs and EMAs
1578 -        let mut sma_5 = SimpleMovingAverage::new(5).unwrap();
1579 -        let mut sma_20 = SimpleMovingAverage::new(20).unwrap();
1580 -        let mut sma_50 = SimpleMovingAverage::new(50).unwrap();
1581 -        let mut ema_12 = ExponentialMovingAverage::new(12).unwrap();
1582 -        let mut ema_26 = ExponentialMovingAverage::new(26).unwrap();
1583 -        let mut ema_9 = ExponentialMovingAverage::new(9).unwrap(); // For MACD signal
1584 -
1585 -        // Momentum indicators
1586 -        let mut rsi = RelativeStrengthIndex::new(14).unwrap();
1587 -        let mut roc_10 = RateOfChange::new(10).unwrap();
1588 -        let mut roc_20 = RateOfChange::new(20).unwrap();
1589 -
1590 -        // Stochastic oscillator: Use SlowStochastic (14 period, 3 smoothing)
1591 -        // then compute %D as a 3-period SMA of %K
1592 -        let mut stoch_k = SlowStochastic::new(14, 3).unwrap();
1593 -        let mut stoch_d_sma = SimpleMovingAverage::new(3).unwrap(); // %D = 3-period SMA of %K
1594 -
1595 -        // Volatility indicators
1596 -        let mut bb = BollingerBands::new(20, 2.0_f64).unwrap();
1597 -        let mut atr = AverageTrueRange::new(14).unwrap();
1598 -        let mut stddev = StandardDeviation::new(20).unwrap();
1599 -
1600 -        // Volume indicators
1601 -        let mut volume_sma = SimpleMovingAverage::new(20).unwrap();
1602 -        let mut obv = OnBalanceVolume::new();
1603 -        let mut mfi = MoneyFlowIndex::new(14).unwrap();
1604 -
1605 -        // Range indicators (14-period high/low)
1606 -        let mut max_14 = Maximum::new(14).unwrap();
1607 -        let mut min_14 = Minimum::new(14).unwrap();
1608 -
1609 -        // =====================================================================
1610 -        // Pre-allocate storage for indicator values
1611 -        // =====================================================================
1612 -        let mut sma_5_vals = vec![0.0_f64; n];
1613 -        let mut sma_20_vals = vec![0.0_f64; n];
1614 -        let mut sma_50_vals = vec![0.0_f64; n];
1615 -        let mut ema_12_vals = vec![0.0_f64; n];
1616 -        let mut ema_26_vals = vec![0.0_f64; n];
1617 -        let mut rsi_vals = vec![50.0_f64; n];
1618 -        let mut roc_10_vals = vec![0.0_f64; n];
1619 -        let mut roc_20_vals = vec![0.0_f64; n];
1620 -        let mut stoch_k_vals = vec![50.0_f64; n];
1621 -        let mut stoch_d_vals = vec![50.0_f64; n];
1622 -        let mut bb_upper = vec![0.0_f64; n];
1623 -        let mut bb_lower = vec![0.0_f64; n];
1624 -        let mut atr_vals = vec![0.0_f64; n];
1625 -        let mut stddev_vals = vec![0.0_f64; n];
1626 -        let mut vol_sma_vals = vec![0.0_f64; n];
1627 -        let mut obv_vals = vec![0.0_f64; n];
1628 -        let mut mfi_vals = vec![50.0_f64; n];
1629 -        let mut max_14_vals = vec![0.0_f64; n];
1630 -        let mut min_14_vals = vec![0.0_f64; n];
1631 -
1632 -        // =====================================================================
1633 -        // Stream through bars and compute all indicators
1634 -        // =====================================================================
1635 -        for (i, bar) in bars.iter().enumerate() {
1636 -            let close = bar.close as f64;
1637 -            let high = bar.high as f64;
1638 -            let low = bar.low as f64;
1639 -            let open = bar.open as f64;
1640 -            let volume = bar.volume;
1641 -
1642 -            // Build DataItem for indicators that need OHLCV
1643 -            let data_item = ta::DataItem::builder()
1644 -                .high(high)
1645 -                .low(low)
1646 -                .close(close)
1647 -                .open(open)
1648 -                .volume(volume)
1649 -                .build()
1650 -                .unwrap();
1651 -
1652 -            // Trend: SMAs
1653 -            sma_5_vals[i] = sma_5.next(close);
1654 -            sma_20_vals[i] = sma_20.next(close);
1655 -            sma_50_vals[i] = sma_50.next(close);
1656 -
1657 -            // Trend: EMAs
1658 -            ema_12_vals[i] = ema_12.next(close);
1659 -            ema_26_vals[i] = ema_26.next(close);
1660 -
1661 -            // Momentum: RSI
1662 -            rsi_vals[i] = rsi.next(close);
1663 -
1664 -            // Momentum: Rate of Change
1665 -            roc_10_vals[i] = roc_10.next(close);
1666 -            roc_20_vals[i] = roc_20.next(close);
1667 -
1668 -            // Stochastic oscillator: %K from SlowStochastic, %D from SMA of %K
1669 -            let k_val = stoch_k.next(close);
1670 -            stoch_k_vals[i] = k_val;
1671 -            stoch_d_vals[i] = stoch_d_sma.next(k_val);
1672 -
1673 -            // Volatility: Bollinger Bands
1674 -            let bb_val = bb.next(close);
1675 -            bb_upper[i] = bb_val.upper;
1676 -            bb_lower[i] = bb_val.lower;
1677 -
1678 -            // Volatility: ATR
1679 -            atr_vals[i] = atr.next(&data_item);
1680 -
1681 -            // Volatility: Standard Deviation
1682 -            stddev_vals[i] = stddev.next(close);
1683 -
1684 -            // Volume: SMA of volume
1685 -            vol_sma_vals[i] = volume_sma.next(volume);
1686 -
1687 -            // Volume: On Balance Volume (uses ta crate's implementation)
1688 -            obv_vals[i] = obv.next(&data_item);
1689 -
1690 -            // Volume: Money Flow Index
1691 -            mfi_vals[i] = mfi.next(&data_item);
1692 -
1693 -            // Range: 14-period Maximum and Minimum
1694 -            max_14_vals[i] = max_14.next(high);
1695 -            min_14_vals[i] = min_14.next(low);
1696 -        }
1697 -
1698 -        // =====================================================================
1699 -        // Compute MACD (derived from EMAs)
1700 -        // =====================================================================
1701 -        let mut macd_vals = vec![0.0_f64; n];
1702 -        let mut macd_signal_vals = vec![0.0_f64; n];
1703 -
1704 -        for i in 0..n {
1705 -            macd_vals[i] = ema_12_vals[i] - ema_26_vals[i];
1706 -            macd_signal_vals[i] = ema_9.next(macd_vals[i]);
1707 -        }
1708 -
1709 -        // =====================================================================
1710 -        // Assemble all indicators into final struct
1711 -        // =====================================================================
1712 -        for i in 0..n {
1713 -            let close = bars[i].close as f64;
1714 -            let high = bars[i].high as f64;
1715 -            let low = bars[i].low as f64;
1716 -            let open = bars[i].open as f64;
1717 -
1718 -            let macd_histogram = macd_vals[i] - macd_signal_vals[i];
1719 -            let bb_width = if sma_20_vals[i] > 0.0 {
1720 -                (bb_upper[i] - bb_lower[i]) / sma_20_vals[i]
1721 -            } else {
1722 -                0.0
1723 -            };
1724 -
1725 -            let vol_ratio = if vol_sma_vals[i] > 0.0 {
1726 -                bars[i].volume / vol_sma_vals[i]
1727 -            } else {
1728 -                1.0
1729 -            };
1730 -
1731 -            // Position within 14-period range
1732 -            let price_range = max_14_vals[i] - min_14_vals[i];
1733 -            let price_vs_range = if price_range > 0.0 {
1734 -                ((close - min_14_vals[i]) / price_range - 0.5) * 2.0 // Normalize to [-1, 1]
1735 -            } else {
1736 -                0.0
1737 -            };
1738 -
1739 -            indicators[i] = TechnicalIndicators {
1740 -                // Trend (5)
1741 -                sma_5: (sma_5_vals[i] / close - 1.0) as f32,
1742 -                sma_20: (sma_20_vals[i] / close - 1.0) as f32,
1743 -                sma_50: (sma_50_vals[i] / close - 1.0) as f32,
1744 -                ema_12: (ema_12_vals[i] / close - 1.0) as f32,
1745 -                ema_26: (ema_26_vals[i] / close - 1.0) as f32,
1746 -
1747 -                // Momentum (6)
1748 -                rsi_14: ((rsi_vals[i] - 50.0) / 50.0) as f32, // Normalize to [-1, 1]
1749 -                macd: (macd_vals[i] / close * 100.0) as f32,
1750 -                macd_signal: (macd_signal_vals[i] / close * 100.0) as f32,
1751 -                macd_histogram: (macd_histogram / close * 100.0) as f32,
1752 -                roc_10: (roc_10_vals[i] / 100.0).clamp(-1.0, 1.0) as f32, // Already percentage, normalize
1753 -                roc_20: (roc_20_vals[i] / 100.0).clamp(-1.0, 1.0) as f32,
1754 -
1755 -                // Stochastic (2)
1756 -                stoch_k: ((stoch_k_vals[i] - 50.0) / 50.0) as f32, // Normalize to [-1, 1]
1757 -                stoch_d: ((stoch_d_vals[i] - 50.0) / 50.0) as f32,
1758 -
1759 -                // Volatility (5)
1760 -                bollinger_upper: (bb_upper[i] / close - 1.0) as f32,
1761 -                bollinger_lower: (bb_lower[i] / close - 1.0) as f32,
1762 -                bollinger_width: bb_width as f32,
1763 -                atr_14: (atr_vals[i] / close) as f32,
1764 -                stddev_20: (stddev_vals[i] / close) as f32, // Normalized by price
1765 -
1766 -                // Volume (4)
1767 -                volume_sma_20: (vol_sma_vals[i] / 1_000_000.0) as f32, // Normalize to millions
1768 -                volume_ratio: (vol_ratio.min(5.0) / 5.0 - 0.5) as f32,
1769 -                obv: (obv_vals[i] / 1e9) as f32, // Normalize to billions
1770 -                mfi_14: ((mfi_vals[i] - 50.0) / 50.0) as f32, // Normalize to [-1, 1]
1771 -
1772 -                // Range (3)
1773 -                max_14: (max_14_vals[i] / close - 1.0) as f32,
1774 -                min_14: (min_14_vals[i] / close - 1.0) as f32,
1775 -                price_vs_range: price_vs_range as f32,
1776 -
1777 -                // Price patterns (6)
1778 -                price_vs_sma20: ((close / sma_20_vals[i].max(0.01) - 1.0).clamp(-0.5, 0.5)) as f32,
1779 -                price_vs_bollinger: if bb_upper[i] > bb_lower[i] {
1780 -                    (((close - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) - 0.5) * 2.0) as f32
1781 -                } else {
1782 -                    0.0
1783 -                },
1784 -                high_low_range: ((high - low) / close) as f32,
1785 -                body_ratio: if high > low {
1786 -                    ((close - open).abs() / (high - low)) as f32
1787 -                } else {
1788 -                    0.0
1789 -                },
1790 -                upper_shadow: if high > low {
1791 -                    ((high - close.max(open)) / (high - low)) as f32
1792 -                } else {
1793 -                    0.0
1794 -                },
1795 -                lower_shadow: if high > low {
1796 -                    ((close.min(open) - low) / (high - low)) as f32
1797 -                } else {
1798 -                    0.0
1799 -                },
1800 -            };
1801 -        }
1802 -
1803 -        indicators
1804 -    }
1805 -}
1806 -
1807 -// =============================================================================
1808 -// Section 4: Data Pipeline and Instance Creation
1809 -// =============================================================================
1810 -
1811 -/// A training/prediction instance for stock prediction.
1812 -#[derive(Debug, Clone)]
1813 -pub struct StockInstance {
1814 -    /// Index into the dataset/tickers vector.
1815 -    pub ticker_idx: usize,
1816 -    /// Ticker feature ID (fed into embedding lookup).
1817 -    pub ticker_fid: i64,
1818 -    /// Sector feature ID (fed into embedding lookup).
1819 -    pub sector_fid: i64,
1820 -
1821 -    /// Time index into this ticker's bar/indicator arrays.
1822 -    pub t: usize,
1823 -
1824 -    /// Direction label: 1=up, 0=down (5-day forward)
1825 -    pub direction_label: f32,
1826 -    /// Magnitude label: % change (5-day forward)
1827 -    pub magnitude_label: f32,
1828 -    /// Profitable label: 1 if return > 2%
1829 -    pub profitable_label: f32,
1830 -}
1831 -
1832 -/// In-memory dataset representation: bars + indicators stored once per ticker.
1833 -///
1834 -/// This avoids the OOM that happens when storing a full lookback window per instance.
1835 -pub struct StockDataset {
1836 -    pub tickers: Vec<TickerInfo>,
1837 -    pub bars_by_ticker: Vec<Vec<StockBar>>,
1838 -    pub indicators_by_ticker: Vec<Vec<TechnicalIndicators>>,
1839 -}
1840 -
1841 -impl StockDataset {
1842 -    fn bar_at(&self, inst: &StockInstance, t: usize) -> &StockBar {
1843 -        &self.bars_by_ticker[inst.ticker_idx][t]
1844 -    }
1845 -
1846 -    fn ind_at(&self, inst: &StockInstance, t: usize) -> &TechnicalIndicators {
1847 -        &self.indicators_by_ticker[inst.ticker_idx][t]
1848 -    }
1849 -}
1850 -
1851 -/// Creates training instances from bars and indicators.
1852 -pub struct InstanceCreator {
1853 -    lookback_window: usize,
1854 -    forward_horizon: usize,
1855 -    profit_threshold: f32,
1856 -}
1857 -
1858 -impl InstanceCreator {
1859 -    pub fn new(lookback_window: usize) -> Self {
1860 -        Self {
1861 -            lookback_window,
1862 -            forward_horizon: 5,     // 5-day forward prediction
1863 -            profit_threshold: 0.02, // 2% profit threshold
1864 -        }
1865 -    }
1866 -
1867 -    /// Creates instances for a ticker.
1868 -    pub fn create_instances(
1869 -        &self,
1870 -        ticker_idx: usize,
1871 -        ticker: &TickerInfo,
1872 -        bars: &[StockBar],
1873 -        indicators: &[TechnicalIndicators],
1874 -    ) -> Vec<StockInstance> {
1875 -        let n = bars.len();
1876 -        let mut instances = Vec::new();
1877 -
1878 -        if n <= self.lookback_window + self.forward_horizon {
1879 -            return instances;
1880 -        }
1881 -
1882 -        // Start after lookback window, end before forward horizon
1883 -        for i in self.lookback_window..(n - self.forward_horizon) {
1884 -            // Compute forward returns
1885 -            let future_price = bars[i + self.forward_horizon].close;
1886 -            let current_price = bars[i].close;
1887 -            let forward_return = (future_price - current_price) / current_price;
1888 -
1889 -            // Create labels
1890 -            let direction_label = if forward_return > 0.0 { 1.0 } else { 0.0 };
1891 -            let magnitude_label = forward_return * 100.0; // Percentage
1892 -            let profitable_label = if forward_return > self.profit_threshold {
1893 -                1.0
1894 -            } else {
1895 -                0.0
1896 -            };
1897 -
1898 -            instances.push(StockInstance {
1899 -                ticker_idx,
1900 -                ticker_fid: ticker.ticker_id,
1901 -                sector_fid: ticker.sector.id(),
1902 -                t: i,
1903 -                direction_label,
1904 -                magnitude_label,
1905 -                profitable_label,
1906 -            });
1907 -        }
1908 -
1909 -        instances
1910 -    }
1911 -}
1912 -
1913 -/// Splits instances into train and eval sets by time.
1914 -pub fn train_eval_split(
1915 -    instances: &[StockInstance],
1916 -    train_ratio: f32,
1917 -) -> (Vec<StockInstance>, Vec<StockInstance>) {
1918 -    let split_idx = (instances.len() as f32 * train_ratio) as usize;
1919 -    let train = instances[..split_idx].to_vec();
1920 -    let eval = instances[split_idx..].to_vec();
1921 -    (train, eval)
1922 -}
1923 -
1924 -/// Creates batches from instances.
1925 -pub fn create_batches(instances: &[StockInstance], batch_size: usize) -> Vec<Vec<&StockInstance>> {
1926 -    instances
1927 -        .chunks(batch_size)
1928 -        .map(|c| c.iter().collect())
1929 -        .collect()
1930 -}
1931 -
1932 -// =============================================================================
1933 -// Section 5: Ticker Embedding Setup
1934 -// =============================================================================
1935 -
1936 -/// Embedding tables for tickers and sectors.
1937 -pub struct EmbeddingTables {
1938 -    ticker_table: CuckooEmbeddingHashTable,
1939 -    sector_table: CuckooEmbeddingHashTable,
1940 -}
1941 -
1942 -impl EmbeddingTables {
1943 -    pub fn new(
1944 -        num_tickers: usize,
1945 -        ticker_dim: usize,
1946 -        num_sectors: usize,
1947 -        sector_dim: usize,
1948 -    ) -> Self {
1949 -        // Create ticker embedding table with Xavier initialization
1950 -        let ticker_initializer = Arc::new(XavierNormalInitializer::new(1.0));
1951 -        let mut ticker_table = CuckooEmbeddingHashTable::with_initializer(
1952 -            num_tickers * 2,
1953 -            ticker_dim,
1954 -            ticker_initializer,
1955 -        );
1956 -
1957 -        // Create sector embedding table
1958 -        let sector_initializer = Arc::new(XavierNormalInitializer::new(1.0));
1959 -        let mut sector_table = CuckooEmbeddingHashTable::with_initializer(
1960 -            num_sectors * 2,
1961 -            sector_dim,
1962 -            sector_initializer,
1963 -        );
1964 -
1965 -        // Initialize embeddings for all tickers
1966 -        for i in 0..num_tickers {
1967 -            ticker_table.get_or_initialize(i as i64).unwrap();
1968 -        }
1969 -
1970 -        // Initialize embeddings for all sectors
1971 -        for sector in Sector::all() {
1972 -            sector_table.get_or_initialize(sector.id()).unwrap();
1973 -        }
1974 -
1975 -        Self {
1976 -            ticker_table,
1977 -            sector_table,
1978 -        }
1979 -    }
1980 -
1981 -    pub fn lookup_ticker(&self, ticker_id: i64) -> Vec<f32> {
1982 -        let mut output = vec![0.0; self.ticker_table.dim()];
1983 -        self.ticker_table.lookup(&[ticker_id], &mut output).unwrap();
1984 -        output
1985 -    }
1986 -
1987 -    pub fn lookup_sector(&self, sector_id: i64) -> Vec<f32> {
1988 -        let mut output = vec![0.0; self.sector_table.dim()];
1989 -        self.sector_table.lookup(&[sector_id], &mut output).unwrap();
1990 -        output
1991 -    }
1992 -
1993 -    pub fn lookup_batch(&self, ticker_ids: &[i64], sector_ids: &[i64]) -> (Vec<f32>, Vec<f32>) {
1994 -        let batch_size = ticker_ids.len();
1995 -
1996 -        let mut ticker_embeddings = vec![0.0; batch_size * self.ticker_table.dim()];
1997 -        let mut sector_embeddings = vec![0.0; batch_size * self.sector_table.dim()];
1998 -
1999 -        self.ticker_table
2000 -            .lookup(ticker_ids, &mut ticker_embeddings)
2001 -            .unwrap();
2002 -        self.sector_table
2003 -            .lookup(sector_ids, &mut sector_embeddings)
2004 -            .unwrap();
2005 -
2006 -        (ticker_embeddings, sector_embeddings)
2007 -    }
2008 -
2009 -    pub fn apply_gradients(
2010 -        &mut self,
2011 -        ticker_ids: &[i64],
2012 -        ticker_grads: &[f32],
2013 -        sector_ids: &[i64],
2014 -        sector_grads: &[f32],
2015 -    ) {
2016 -        self.ticker_table
2017 -            .apply_gradients(ticker_ids, ticker_grads)
2018 -            .unwrap();
2019 -        self.sector_table
2020 -            .apply_gradients(sector_ids, sector_grads)
2021 -            .unwrap();
2022 -    }
2023 -
2024 -    pub fn ticker_dim(&self) -> usize {
2025 -        self.ticker_table.dim()
2026 -    }
2027 -
2028 -    pub fn sector_dim(&self) -> usize {
2029 -        self.sector_table.dim()
2030 -    }
2031 -}
2032 -
2033 -// =============================================================================
2034 -// Section 6: Model Architecture
2035 -// =============================================================================
2036 -
2037 -/// Stock prediction model combining embeddings, DIEN, DCN, MMoE, SENet, and more.
2038 -///
2039 -/// Enhanced architecture:
2040 -/// ```text
2041 -/// Embeddings (ticker, sector)
2042 -///     ↓
2043 -/// [SENet: Learn indicator importance]
2044 -///     ↓
2045 -/// [DIEN: Sequential price pattern extraction]
2046 -///     ↓
2047 -/// [DIN: Attention on recent indicators]
2048 -///     ↓
2049 -/// [Concatenate: DIEN output + DIN output + embeddings]
2050 -///     ↓
2051 -/// [DCN-V2 (Matrix mode): Feature interactions]
2052 -///     ↓
2053 -/// [LayerNorm: Stabilize training]
2054 -///     ↓
2055 -/// [MMoE: Shared experts + Task-specific gates]
2056 -///     ├→ Direction Head (MLP)
2057 -///     ├→ Magnitude Head (MLP)
2058 -///     └→ Profitability Head (MLP)
2059 -/// ```
2060 -pub struct StockPredictionModel {
2061 -    /// Embedding tables
2062 -    embeddings: EmbeddingTables,
2063 -    /// SENet for adaptive feature importance on indicators
2064 -    senet: SENetLayer,
2065 -    /// DIEN for sequential patterns
2066 -    dien: DIENLayer,
2067 -    /// DIN attention for indicator weighting
2068 -    din: DINAttention,
2069 -    /// DCN for feature interactions (Matrix mode for DCN-V2)
2070 -    dcn: CrossNetwork,
2071 -    /// LayerNorm before task heads
2072 -    layer_norm: LayerNorm,
2073 -    /// MMoE for multi-task learning (3 tasks: direction, magnitude, profitable)
2074 -    mmoe: MMoE,
2075 -    /// Direction prediction head (after MMoE)
2076 -    direction_head: MLP,
2077 -    /// Magnitude prediction head (after MMoE)
2078 -    magnitude_head: MLP,
2079 -    /// Profitable prediction head (after MMoE)
2080 -    profitable_head: MLP,
2081 -    /// Configuration
2082 -    config: StockPredictorConfig,
2083 -}
2084 -
2085 -impl StockPredictionModel {
2086 -    pub fn new(config: &StockPredictorConfig) -> Self {
2087 -        // Create embedding tables
2088 -        let embeddings = EmbeddingTables::new(
2089 -            config.num_tickers,
2090 -            config.ticker_embedding_dim,
2091 -            Sector::all().len(),
2092 -            config.sector_embedding_dim,
2093 -        );
2094 -
2095 -        // Calculate input dimensions
2096 -        // Sequence features: 4 price + 36 indicators = 40 features
2097 -        let seq_feature_dim = 4 + TechnicalIndicators::NUM_FEATURES;
2098 -
2099 -        // 1. SENet for adaptive feature importance on technical indicators
2100 -        // Reduction ratio 4: 36 indicators → 9 bottleneck → 36 reweighted
2101 -        let senet = SENetLayer::new(TechnicalIndicators::NUM_FEATURES, 4, true);
2102 -
2103 -        // 2. DIEN layer for sequential pattern extraction
2104 -        let dien = DIENConfig::new(seq_feature_dim, config.dien_hidden_size)
2105 -            .with_use_auxiliary_loss(false)
2106 -            .build()
2107 -            .unwrap();
2108 -
2109 -        // 3. DIN attention for indicator weighting (query = current, keys = historical)
2110 -        // Uses embedding_dim = seq_feature_dim for attention on indicator sequences
2111 -        let din = DINConfig::new(seq_feature_dim)
2112 -            .with_attention_hidden_units(vec![64, 32])
2113 -            .with_activation(ActivationType::Sigmoid)
2114 -            .with_use_softmax(true)
2115 -            .build()
2116 -            .unwrap();
2117 -
2118 -        // 4. Combined feature dimension for DCN:
2119 -        // ticker_emb + sector_emb + dien_output + din_output + senet_indicators
2120 -        let combined_dim = config.ticker_embedding_dim
2121 -            + config.sector_embedding_dim
2122 -            + config.dien_hidden_size
2123 -            + seq_feature_dim  // DIN output
2124 -            + TechnicalIndicators::NUM_FEATURES; // SENet-reweighted indicators
2125 -
2126 -        // 5. Create DCN with Matrix mode (DCN-V2) for richer feature interactions
2127 -        let dcn = CrossNetwork::new(combined_dim, config.dcn_cross_layers, DCNMode::Matrix);
2128 -
2129 -        // 6. LayerNorm for training stability before task heads
2130 -        let layer_norm = LayerNorm::new(combined_dim);
2131 -
2132 -        // 7. MMoE for multi-task learning: 4 experts, 3 tasks
2133 -        // Each expert has hidden layers [64, 32] and outputs 32-dim
2134 -        // Using GELU for smoother gradients and to prevent dead neurons
2135 -        let mmoe = MMoEConfig::new(combined_dim, 4, 3)
2136 -            .with_expert_hidden_units(vec![64, 32])
2137 -            .with_expert_activation(ActivationType::GELU)
2138 -            .with_expert_output_dim(32)
2139 -            .build()
2140 -            .unwrap();
2141 -
2142 -        // 8. Task-specific heads (take MMoE output of 32-dim)
2143 -        let mmoe_output_dim = 32;
2144 -
2145 -        // Direction head: binary classification (sigmoid)
2146 -        // Using GELU for smoother gradients
2147 -        let direction_head = MLPConfig::new(mmoe_output_dim)
2148 -            .add_layer(16, ActivationType::GELU)
2149 -            .add_layer(1, ActivationType::Sigmoid)
2150 -            .build()
2151 -            .unwrap();
2152 -
2153 -        // Magnitude head: regression
2154 -        let magnitude_head = MLPConfig::new(mmoe_output_dim)
2155 -            .add_layer(16, ActivationType::GELU)
2156 -            .add_layer(1, ActivationType::None)
2157 -            .build()
2158 -            .unwrap();
2159 -
2160 -        // Profitable head: binary classification (sigmoid)
2161 -        let profitable_head = MLPConfig::new(mmoe_output_dim)
2162 -            .add_layer(16, ActivationType::GELU)
2163 -            .add_layer(1, ActivationType::Sigmoid)
2164 -            .build()
2165 -            .unwrap();
2166 -
2167 -        Self {
2168 -            embeddings,
2169 -            senet,
2170 -            dien,
2171 -            din,
2172 -            dcn,
2173 -            layer_norm,
2174 -            mmoe,
2175 -            direction_head,
2176 -            magnitude_head,
2177 -            profitable_head,
2178 -            config: config.clone(),
2179 -        }
2180 -    }
2181 -
2182 -    /// Forward pass for a batch of instances using enhanced architecture.
2183 -    ///
2184 -    /// Flow: Embeddings → SENet → DIEN → DIN → Concat → DCN → LayerNorm → MMoE → Task Heads
2185 -    pub fn forward(&self, batch: &[&StockInstance]) -> ModelOutput {
2186 -        let batch_size = batch.len();
2187 -        let lookback = self.config.lookback_window;
2188 -        let seq_feature_dim = 4 + TechnicalIndicators::NUM_FEATURES;
2189 -
2190 -        // Collect batch data
2191 -        let ticker_ids: Vec<i64> = batch.iter().map(|i| i.ticker_fid).collect();
2192 -        let sector_ids: Vec<i64> = batch.iter().map(|i| i.sector_fid).collect();
2193 -
2194 -        // 1. Lookup embeddings
2195 -        let (ticker_embs, sector_embs) = self.embeddings.lookup_batch(&ticker_ids, &sector_ids);
2196 -        let ticker_tensor =
2197 -            Tensor::from_data(&[batch_size, self.embeddings.ticker_dim()], ticker_embs);
2198 -        let sector_tensor =
2199 -            Tensor::from_data(&[batch_size, self.embeddings.sector_dim()], sector_embs);
2200 -
2201 -        // 2. Get current indicator features and apply SENet for adaptive importance
2202 -        let mut indicator_data = vec![0.0; batch_size * TechnicalIndicators::NUM_FEATURES];
2203 -        for (b, instance) in batch.iter().enumerate() {
2204 -            for (f, &val) in instance.indicator_features.iter().enumerate() {
2205 -                if f < TechnicalIndicators::NUM_FEATURES {
2206 -                    indicator_data[b * TechnicalIndicators::NUM_FEATURES + f] = val;
2207 -                }
2208 -            }
2209 -        }
2210 -        let indicator_tensor = Tensor::from_data(
2211 -            &[batch_size, TechnicalIndicators::NUM_FEATURES],
2212 -            indicator_data,
2213 -        );
2214 -
2215 -        // Apply SENet: learns which indicators are important for each sample
2216 -        let senet_indicators = self.senet.forward(&indicator_tensor).unwrap();
2217 -
2218 -        // 3. Build sequence tensor for DIEN [batch, lookback, features]
2219 -        let mut seq_data = vec![0.0; batch_size * lookback * seq_feature_dim];
2220 -        for (b, instance) in batch.iter().enumerate() {
2221 -            for (t, seq_features) in instance.historical_sequence.iter().enumerate() {
2222 -                for (f, &val) in seq_features.iter().enumerate().take(seq_feature_dim) {
2223 -                    let idx = b * lookback * seq_feature_dim + t * seq_feature_dim + f;
2224 -                    if idx < seq_data.len() {
2225 -                        seq_data[idx] = val;
2226 -                    }
2227 -                }
2228 -            }
2229 -        }
2230 -        let seq_tensor = Tensor::from_data(&[batch_size, lookback, seq_feature_dim], seq_data);
2231 -
2232 -        // Create target tensor for DIEN (use last timestep features)
2233 -        let mut target_data = vec![0.0; batch_size * seq_feature_dim];
2234 -        for (b, instance) in batch.iter().enumerate() {
2235 -            if let Some(last_seq) = instance.historical_sequence.last() {
2236 -                for (f, &val) in last_seq.iter().enumerate().take(seq_feature_dim) {
2237 -                    target_data[b * seq_feature_dim + f] = val;
2238 -                }
2239 -            }
2240 -        }
2241 -        let target_tensor = Tensor::from_data(&[batch_size, seq_feature_dim], target_data);
2242 -
2243 -        // 4. Run DIEN for sequential pattern extraction
2244 -        let dien_output = self
2245 -            .dien
2246 -            .forward_dien(&seq_tensor, &target_tensor, None)
2247 -            .unwrap();
2248 -
2249 -        // 5. Run DIN attention: query = current features, keys/values = historical sequence
2250 -        // This weights recent indicators by their relevance to current prediction
2251 -        let din_output = self
2252 -            .din
2253 -            .forward_attention(&target_tensor, &seq_tensor, &seq_tensor, None)
2254 -            .unwrap();
2255 -
2256 -        // 6. Concatenate all features:
2257 -        // [ticker_emb | sector_emb | dien_out | din_out | senet_indicators]
2258 -        let combined = self.concatenate_enhanced_features(
2259 -            &ticker_tensor,
2260 -            &sector_tensor,
2261 -            &dien_output,
2262 -            &din_output,
2263 -            &senet_indicators,
2264 -        );
2265 -
2266 -        // 7. Run DCN (Matrix mode for richer feature interactions)
2267 -        let dcn_output = self.dcn.forward(&combined).unwrap();
2268 -
2269 -        // 8. Apply LayerNorm for training stability
2270 -        let normalized = self.layer_norm.forward(&dcn_output).unwrap();
2271 -
2272 -        // 9. Run MMoE: produces task-specific outputs for each of 3 tasks
2273 -        // forward_multi returns Vec<Tensor> - one tensor per task, each of shape [batch, expert_output_dim]
2274 -        let mmoe_outputs = self.mmoe.forward_multi(&normalized).unwrap();
2275 -
2276 -        // 10. Run task-specific heads (each mmoe_outputs[i] is [batch, 32])
2277 -        let direction_pred = self.direction_head.forward(&mmoe_outputs[0]).unwrap();
2278 -        let magnitude_pred = self.magnitude_head.forward(&mmoe_outputs[1]).unwrap();
2279 -        let profitable_pred = self.profitable_head.forward(&mmoe_outputs[2]).unwrap();
2280 -
2281 -        ModelOutput {
2282 -            direction: direction_pred.data().to_vec(),
2283 -            magnitude: magnitude_pred.data().to_vec(),
2284 -            profitable: profitable_pred.data().to_vec(),
2285 -        }
2286 -    }
2287 -
2288 -    /// Concatenate enhanced features including DIN output and SENet indicators.
2289 -    fn concatenate_enhanced_features(
2290 -        &self,
2291 -        ticker: &Tensor,
2292 -        sector: &Tensor,
2293 -        dien: &Tensor,
2294 -        din: &Tensor,
2295 -        senet_indicators: &Tensor,
2296 -    ) -> Tensor {
2297 -        let batch_size = ticker.shape()[0];
2298 -        let total_dim = ticker.shape()[1]
2299 -            + sector.shape()[1]
2300 -            + dien.shape()[1]
2301 -            + din.shape()[1]
2302 -            + senet_indicators.shape()[1];
2303 -
2304 -        let mut data = vec![0.0; batch_size * total_dim];
2305 -
2306 -        for b in 0..batch_size {
2307 -            let mut offset = b * total_dim;
2308 -
2309 -            // Copy ticker embeddings
2310 -            for i in 0..ticker.shape()[1] {
2311 -                data[offset + i] = ticker.data()[b * ticker.shape()[1] + i];
2312 -            }
2313 -            offset += ticker.shape()[1];
2314 -
2315 -            // Copy sector embeddings
2316 -            for i in 0..sector.shape()[1] {
2317 -                data[offset + i] = sector.data()[b * sector.shape()[1] + i];
2318 -            }
2319 -            offset += sector.shape()[1];
2320 -
2321 -            // Copy DIEN output
2322 -            for i in 0..dien.shape()[1] {
2323 -                data[offset + i] = dien.data()[b * dien.shape()[1] + i];
2324 -            }
2325 -            offset += dien.shape()[1];
2326 -
2327 -            // Copy DIN output
2328 -            for i in 0..din.shape()[1] {
2329 -                data[offset + i] = din.data()[b * din.shape()[1] + i];
2330 -            }
2331 -            offset += din.shape()[1];
2332 -
2333 -            // Copy SENet-reweighted indicator features
2334 -            for i in 0..senet_indicators.shape()[1] {
2335 -                data[offset + i] = senet_indicators.data()[b * senet_indicators.shape()[1] + i];
2336 -            }
2337 -        }
2338 -
2339 -        Tensor::from_data(&[batch_size, total_dim], data)
2340 -    }
2341 -
2342 -    /// Computes total loss.
2343 -    pub fn compute_loss(
2344 -        &self,
2345 -        output: &ModelOutput,
2346 -        batch: &[&StockInstance],
2347 -    ) -> (f32, f32, f32, f32) {
2348 -        let batch_size = batch.len() as f32;
2349 -
2350 -        // Direction loss (binary cross-entropy)
2351 -        let mut direction_loss = 0.0;
2352 -        for (i, instance) in batch.iter().enumerate() {
2353 -            let pred = output.direction[i].clamp(1e-7, 1.0 - 1e-7);
2354 -            let label = instance.direction_label;
2355 -            direction_loss += -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln();
2356 -        }
2357 -        direction_loss /= batch_size;
2358 -
2359 -        // Magnitude loss (Huber loss for robustness, scaled)
2360 -        let mut magnitude_loss = 0.0;
2361 -        let delta = 5.0; // Huber delta - transition point
2362 -        for (i, instance) in batch.iter().enumerate() {
2363 -            let diff = (output.magnitude[i] - instance.magnitude_label).abs();
2364 -            if diff <= delta {
2365 -                magnitude_loss += 0.5 * diff * diff;
2366 -            } else {
2367 -                magnitude_loss += delta * (diff - 0.5 * delta);
2368 -            }
2369 -        }
2370 -        magnitude_loss /= batch_size;
2371 -        magnitude_loss /= 10.0; // Scale down to be comparable with BCE losses
2372 -
2373 -        // Profitable loss (binary cross-entropy)
2374 -        let mut profitable_loss = 0.0;
2375 -        for (i, instance) in batch.iter().enumerate() {
2376 -            let pred = output.profitable[i].clamp(1e-7, 1.0 - 1e-7);
2377 -            let label = instance.profitable_label;
2378 -            profitable_loss += -label * pred.ln() - (1.0 - label) * (1.0 - pred).ln();
2379 -        }
2380 -        profitable_loss /= batch_size;
2381 -
2382 -        // Combined loss (rebalanced weights)
2383 -        let total_loss = 0.4 * direction_loss + 0.4 * magnitude_loss + 0.2 * profitable_loss;
2384 -
2385 -        (total_loss, direction_loss, magnitude_loss, profitable_loss)
2386 -    }
2387 -
2388 -    /// Gets mutable parameters for training.
2389 -    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
2390 -        let mut params = Vec::new();
2391 -        // SENet parameters
2392 -        params.extend(self.senet.parameters_mut());
2393 -        // DIEN parameters
2394 -        params.extend(self.dien.parameters_mut());
2395 -        // DIN parameters
2396 -        params.extend(self.din.parameters_mut());
2397 -        // DCN parameters
2398 -        params.extend(self.dcn.parameters_mut());
2399 -        // LayerNorm parameters
2400 -        params.extend(self.layer_norm.parameters_mut());
2401 -        // MMoE parameters
2402 -        params.extend(self.mmoe.parameters_mut());
2403 -        // Task head parameters
2404 -        params.extend(self.direction_head.parameters_mut());
2405 -        params.extend(self.magnitude_head.parameters_mut());
2406 -        params.extend(self.profitable_head.parameters_mut());
2407 -        params
2408 -    }
2409 -}
2410 -
2411 -/// Model output containing predictions.
2412 -#[derive(Debug, Clone)]
2413 -pub struct ModelOutput {
2414 -    pub direction: Vec<f32>,
2415 -    pub magnitude: Vec<f32>,
2416 -    pub profitable: Vec<f32>,
2417 -}
2418 -
2419 -// =============================================================================
2420 -// Section 7: Training Loop
2421 -// =============================================================================
2422 -
2423 -/// Training metrics tracked during training.
2424 -#[derive(Debug, Clone, Default)]
2425 -pub struct TrainingMetrics {
2426 -    pub step: usize,
2427 -    pub total_loss: f32,
2428 -    pub direction_loss: f32,
2429 -    pub magnitude_loss: f32,
2430 -    pub profitable_loss: f32,
2431 -    pub direction_accuracy: f32,
2432 -    pub samples_processed: usize,
2433 -}
2434 -
2435 -/// Trainer for the stock prediction model with AMSGrad optimizer.
2436 -///
2437 -/// Features:
2438 -/// - AMSGrad optimizer for better convergence on volatile data
2439 -/// - Gradient clipping for stability
2440 -/// - Linear warmup + cosine annealing LR schedule
2441 -/// - Weight decay (L2 regularization)
2442 -pub struct Trainer {
2443 -    model: StockPredictionModel,
2444 -    config: StockPredictorConfig,
2445 -    learning_rate: f32,
2446 -    initial_lr: f32,
2447 -    global_step: usize,
2448 -    epoch: usize,
2449 -    best_eval_loss: f32,
2450 -    patience_counter: usize,
2451 -    rng: RandomGenerator,
2452 -    // NOTE: This example uses a finite-difference (SPSA-like) update rather than full backprop.
2453 -    // The monolith-layers crate contains backward() implementations for many layers, but some
2454 -    // (e.g. DIEN) intentionally use simplified/placeholder gradients. A finite-difference update
2455 -    // keeps the example self-contained and makes progress without requiring full autodiff.
2456 -    //
2457 -    // We still keep a per-parameter AMSGrad state so updates are stable over long runs.
2458 -    optimizers: Vec<Amsgrad>,
2459 -    // Hyperparameters
2460 -    grad_clip: f32,
2461 -    warmup_steps: usize,
2462 -    // Finite-difference hyperparameters
2463 -    fd_epsilon: f32,
2464 -    fd_num_coords: usize,
2465 -}
2466 -
2467 -impl Trainer {
2468 -    pub fn new(config: &StockPredictorConfig) -> Self {
2469 -        let mut model = StockPredictionModel::new(config);
2470 -
2471 -        // Create AMSGrad optimizer for each parameter tensor
2472 -        let optimizers: Vec<Amsgrad> = model
2473 -            .parameters_mut()
2474 -            .iter()
2475 -            .map(|_| {
2476 -                Amsgrad::with_params(
2477 -                    config.learning_rate,
2478 -                    0.9,    // beta1: momentum
2479 -                    0.999,  // beta2: second moment
2480 -                    1e-8,   // epsilon: numerical stability
2481 -                    0.0001, // weight_decay: L2 regularization
2482 -                )
2483 -            })
2484 -            .collect();
2485 -
2486 -        // Quick warmup for first 50 steps, then full LR
2487 -        let warmup_steps = 50;
2488 -
2489 -        Self {
2490 -            model,
2491 -            config: config.clone(),
2492 -            learning_rate: config.learning_rate,
2493 -            initial_lr: config.learning_rate,
2494 -            global_step: 0,
2495 -            epoch: 0,
2496 -            best_eval_loss: f32::MAX,
2497 -            patience_counter: 0,
2498 -            rng: RandomGenerator::new(config.seed),
2499 -            optimizers,
2500 -            grad_clip: 1.0, // Moderate gradient clipping
2501 -            warmup_steps,
2502 -            // Finite-difference tuning: small epsilon, small coordinate budget.
2503 -            // This makes the example "train longer" without spending most time iterating all params.
2504 -            fd_epsilon: 1e-3,
2505 -            fd_num_coords: 2048,
2506 -        }
2507 -    }
2508 -
2509 -    /// Updates learning rate with warmup + cosine annealing schedule.
2510 -    pub fn update_lr(&mut self) {
2511 -        // Linear warmup for initial steps
2512 -        if self.global_step < self.warmup_steps {
2513 -            let warmup_progress = self.global_step as f32 / self.warmup_steps as f32;
2514 -            self.learning_rate = self.initial_lr * warmup_progress;
2515 -        } else {
2516 -            // Cosine annealing after warmup
2517 -            let progress = self.epoch as f32 / self.config.num_epochs as f32;
2518 -            let decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
2519 -            self.learning_rate = self.initial_lr * decay.max(0.01); // Min 1% of initial LR
2520 -        }
2521 -    }
2522 -
2523 -    /// Applies learning rate decay (cosine annealing) at epoch boundary.
2524 -    pub fn decay_lr(&mut self) {
2525 -        self.epoch += 1;
2526 -        self.update_lr();
2527 -    }
2528 -
2529 -    /// Returns current learning rate.
2530 -    pub fn current_lr(&self) -> f32 {
2531 -        self.learning_rate
2532 -    }
2533 -
2534 -    /// Trains for one epoch.
2535 -    pub fn train_epoch(&mut self, train_instances: &[StockInstance]) -> TrainingMetrics {
2536 -        // Shuffle instances using internal RNG
2537 -        let mut indices: Vec<usize> = (0..train_instances.len()).collect();
2538 -        self.rng.shuffle(&mut indices);
2539 -
2540 -        let mut epoch_metrics = TrainingMetrics::default();
2541 -        let mut direction_correct = 0;
2542 -        let mut total_samples = 0;
2543 -
2544 -        // Process batches
2545 -        for chunk in indices.chunks(self.config.batch_size) {
2546 -            let batch: Vec<&StockInstance> = chunk.iter().map(|&i| &train_instances[i]).collect();
2547 -
2548 -            // Forward pass
2549 -            let output = self.model.forward(&batch);
2550 -
2551 -            // Compute loss
2552 -            let (total_loss, direction_loss, magnitude_loss, profitable_loss) =
2553 -                self.model.compute_loss(&output, &batch);
2554 -
2555 -            // Accumulate metrics
2556 -            epoch_metrics.total_loss += total_loss * batch.len() as f32;
2557 -            epoch_metrics.direction_loss += direction_loss * batch.len() as f32;
2558 -            epoch_metrics.magnitude_loss += magnitude_loss * batch.len() as f32;
2559 -            epoch_metrics.profitable_loss += profitable_loss * batch.len() as f32;
2560 -
2561 -            // Compute accuracy
2562 -            for (i, instance) in batch.iter().enumerate() {
2563 -                let pred = if output.direction[i] > 0.5 { 1.0 } else { 0.0 };
2564 -                if (pred - instance.direction_label).abs() < 0.1 {
2565 -                    direction_correct += 1;
2566 -                }
2567 -            }
2568 -            total_samples += batch.len();
2569 -
2570 -            // Loss-directed finite-difference step (SPSA-like, coordinate-sampled).
2571 -            // This is much more likely to improve the model than the older "random perturb" update.
2572 -            self.finite_difference_step(&batch);
2573 -
2574 -            self.global_step += 1;
2575 -
2576 -            // Logging
2577 -            if self.config.verbose && self.global_step % self.config.log_every_n_steps == 0 {
2578 -                println!(
2579 -                    "  [Step {}] Loss: {:.4} | Dir: {:.4} | Mag: {:.4} | Prof: {:.4}",
2580 -                    self.global_step, total_loss, direction_loss, magnitude_loss, profitable_loss
2581 -                );
2582 -            }
2583 -        }
2584 -
2585 -        // Finalize metrics
2586 -        epoch_metrics.step = self.global_step;
2587 -        epoch_metrics.samples_processed = total_samples;
2588 -        if total_samples > 0 {
2589 -            epoch_metrics.total_loss /= total_samples as f32;
2590 -            epoch_metrics.direction_loss /= total_samples as f32;
2591 -            epoch_metrics.magnitude_loss /= total_samples as f32;
2592 -            epoch_metrics.profitable_loss /= total_samples as f32;
2593 -            epoch_metrics.direction_accuracy = direction_correct as f32 / total_samples as f32;
2594 -        }
2595 -
2596 -        epoch_metrics
2597 -    }
2598 -
2599 -    fn finite_difference_step(&mut self, batch: &[&StockInstance]) {
2600 -        // Update learning rate (handles warmup)
2601 -        self.update_lr();
2602 -
2603 -        // Evaluate baseline loss once for logging/scale. This is not strictly required for SPSA,
2604 -        // but it helps keep update magnitudes in a sensible range.
2605 -        let baseline_loss = {
2606 -            let output = self.model.forward(batch);
2607 -            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
2608 -            loss
2609 -        };
2610 -
2611 -        // Choose which parameter tensor to update this step (cycle for determinism).
2612 -        // Updating a small subset per step makes long training runs feasible.
2613 -        let num_params = self.model.parameters_mut().len().max(1);
2614 -        let param_idx = self.global_step % num_params;
2615 -
2616 -        // Coordinate budget for this tensor.
2617 -        let (param_len, coord_count) = {
2618 -            let mut params = self.model.parameters_mut();
2619 -            let len = params[param_idx].data().len();
2620 -            (len, self.fd_num_coords.min(len.max(1)))
2621 -        };
2622 -
2623 -        // Precompute coordinate indices and +/-1 deltas (deterministic "random").
2624 -        let mut coord_indices = Vec::with_capacity(coord_count);
2625 -        let mut coord_deltas = Vec::with_capacity(coord_count);
2626 -        for j in 0..coord_count {
2627 -            // Deterministic index selection based on (step, param_idx, j).
2628 -            let h = (self.global_step as u64)
2629 -                .wrapping_mul(1_000_003)
2630 -                .wrapping_add(param_idx as u64 * 97)
2631 -                .wrapping_add(j as u64 * 1_009);
2632 -            let idx = (h as usize) % param_len;
2633 -            // +/-1 delta
2634 -            let delta = if (h >> 11) & 1 == 0 { 1.0 } else { -1.0 };
2635 -            coord_indices.push(idx);
2636 -            coord_deltas.push(delta);
2637 -        }
2638 -
2639 -        let eps = self.fd_epsilon;
2640 -
2641 -        // Apply +eps perturbation to selected coords.
2642 -        {
2643 -            let mut params = self.model.parameters_mut();
2644 -            let data = params[param_idx].data_mut();
2645 -            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
2646 -                data[idx] += eps * delta;
2647 -            }
2648 -        }
2649 -
2650 -        let loss_plus = {
2651 -            let output = self.model.forward(batch);
2652 -            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
2653 -            loss
2654 -        };
2655 -
2656 -        // Apply -2eps (now at -eps relative to original).
2657 -        {
2658 -            let mut params = self.model.parameters_mut();
2659 -            let data = params[param_idx].data_mut();
2660 -            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
2661 -                data[idx] -= 2.0 * eps * delta;
2662 -            }
2663 -        }
2664 -
2665 -        let loss_minus = {
2666 -            let output = self.model.forward(batch);
2667 -            let (loss, _, _, _) = self.model.compute_loss(&output, batch);
2668 -            loss
2669 -        };
2670 -
2671 -        // Restore to original.
2672 -        {
2673 -            let mut params = self.model.parameters_mut();
2674 -            let data = params[param_idx].data_mut();
2675 -            for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
2676 -                data[idx] += eps * delta;
2677 -            }
2678 -        }
2679 -
2680 -        // SPSA-style gradient estimate for the selected coordinates:
2681 -        // g_i ~= (L+ - L-) / (2 * eps) * delta_i
2682 -        let coeff = (loss_plus - loss_minus) / (2.0 * eps).max(1e-12);
2683 -
2684 -        // Ensure optimizer exists for this parameter tensor
2685 -        if param_idx >= self.optimizers.len() {
2686 -            self.optimizers.resize_with(param_idx + 1, || {
2687 -                Amsgrad::with_params(self.learning_rate, 0.9, 0.999, 1e-8, 0.0001)
2688 -            });
2689 -        }
2690 -
2691 -        // Apply sparse update using AMSGrad state but only on selected coordinates.
2692 -        // We do this by directly updating the optimizer's internal vectors via a dense gradient
2693 -        // buffer that is reused and mostly zeros would be too expensive. Instead, we update
2694 -        // the parameter values with a clipped SGD step scaled by coeff.
2695 -        //
2696 -        // Rationale: This example aims to demonstrate end-to-end training behavior without
2697 -        // spending O(#params) per batch.
2698 -        let step_lr = self.learning_rate;
2699 -        let mut params = self.model.parameters_mut();
2700 -        let data = params[param_idx].data_mut();
2701 -        for (&idx, &delta) in coord_indices.iter().zip(coord_deltas.iter()) {
2702 -            // Light weight decay encourages stability over long training.
2703 -            let l2 = 0.0001 * data[idx];
2704 -            let grad = (coeff * delta + l2).clamp(-self.grad_clip, self.grad_clip);
2705 -            data[idx] -= step_lr * grad;
2706 -        }
2707 -
2708 -        // If the FD step is wildly noisy (common early), damp it by shrinking epsilon slowly.
2709 -        // This keeps long runs stable without requiring a full optimizer state.
2710 -        if !baseline_loss.is_finite() || !loss_plus.is_finite() || !loss_minus.is_finite() {
2711 -            self.fd_epsilon = (self.fd_epsilon * 0.5).max(1e-6);
2712 -        }
2713 -    }
2714 -
2715 -    /// Evaluates on held-out data.
2716 -    pub fn evaluate(&self, eval_instances: &[StockInstance]) -> TrainingMetrics {
2717 -        let batches = create_batches(eval_instances, self.config.batch_size);
2718 -
2719 -        let mut metrics = TrainingMetrics::default();
2720 -        let mut direction_correct = 0;
2721 -        let mut total_samples = 0;
2722 -
2723 -        for batch in batches {
2724 -            let output = self.model.forward(&batch);
2725 -            let (total_loss, direction_loss, magnitude_loss, profitable_loss) =
2726 -                self.model.compute_loss(&output, &batch);
2727 -
2728 -            metrics.total_loss += total_loss * batch.len() as f32;
2729 -            metrics.direction_loss += direction_loss * batch.len() as f32;
2730 -            metrics.magnitude_loss += magnitude_loss * batch.len() as f32;
2731 -            metrics.profitable_loss += profitable_loss * batch.len() as f32;
2732 -
2733 -            for (i, instance) in batch.iter().enumerate() {
2734 -                let pred = if output.direction[i] > 0.5 { 1.0 } else { 0.0 };
2735 -                if (pred - instance.direction_label).abs() < 0.1 {
2736 -                    direction_correct += 1;
2737 -                }
2738 -            }
2739 -            total_samples += batch.len();
2740 -        }
2741 -
2742 -        metrics.step = self.global_step;
2743 -        metrics.samples_processed = total_samples;
2744 -        if total_samples > 0 {
2745 -            metrics.total_loss /= total_samples as f32;
2746 -            metrics.direction_loss /= total_samples as f32;
2747 -            metrics.magnitude_loss /= total_samples as f32;
2748 -            metrics.profitable_loss /= total_samples as f32;
2749 -            metrics.direction_accuracy = direction_correct as f32 / total_samples as f32;
2750 -        }
2751 -
2752 -        metrics
2753 -    }
2754 -
2755 -    /// Checks early stopping condition.
2756 -    pub fn check_early_stopping(&mut self, eval_loss: f32) -> bool {
2757 -        if self.config.early_stopping_patience == 0 {
2758 -            return false;
2759 -        }
2760 -
2761 -        let min_delta = self.config.early_stopping_min_delta.max(0.0);
2762 -        if eval_loss < self.best_eval_loss - min_delta {
2763 -            self.best_eval_loss = eval_loss;
2764 -            self.patience_counter = 0;
2765 -            false
2766 -        } else {
2767 -            self.patience_counter += 1;
2768 -            self.patience_counter >= self.config.early_stopping_patience
2769 -        }
2770 -    }
2771 -
2772 -    /// Gets reference to the model.
2773 -    pub fn model(&self) -> &StockPredictionModel {
2774 -        &self.model
2775 -    }
2776 -}
2777 -
2778 -// =============================================================================
2779 -// Section 8: Evaluation and Trading Metrics
2780 -// =============================================================================
2781 -
2782 -/// Financial performance metrics.
2783 -#[derive(Debug, Clone, Default)]
2784 -pub struct FinancialMetrics {
2785 -    pub total_return: f32,
2786 -    pub annualized_return: f32,
2787 -    pub sharpe_ratio: f32,
2788 -    pub sortino_ratio: f32,
2789 -    pub max_drawdown: f32,
2790 -    pub calmar_ratio: f32,
2791 -    pub win_rate: f32,
2792 -    pub profit_factor: f32,
2793 -    pub num_trades: usize,
2794 -    pub avg_win: f32,
2795 -    pub avg_loss: f32,
2796 -}
2797 -
2798 -/// Backtesting engine.
2799 -pub struct Backtester {
2800 -    initial_capital: f64,
2801 -    transaction_cost: f32,
2802 -    position_size: f32,
2803 -}
2804 -
2805 -impl Backtester {
2806 -    pub fn new() -> Self {
2807 -        Self {
2808 -            initial_capital: 100_000.0,
2809 -            transaction_cost: 0.001, // 0.1% per trade
2810 -            position_size: 0.1,      // 10% of capital per position
2811 -        }
2812 -    }
2813 -
2814 -    /// Runs backtesting simulation.
2815 -    pub fn run(
2816 -        &self,
2817 -        model: &StockPredictionModel,
2818 -        instances: &[StockInstance],
2819 -        _bars: &[StockBar], // Reserved for future intraday stop-loss analysis
2820 -    ) -> FinancialMetrics {
2821 -        let batches = create_batches(instances, 32);
2822 -
2823 -        let mut capital = self.initial_capital;
2824 -        let mut peak_capital = capital;
2825 -        let mut max_drawdown = 0.0f64;
2826 -
2827 -        let mut returns: Vec<f64> = Vec::new();
2828 -        let mut wins = 0;
2829 -        let mut losses = 0;
2830 -        let mut total_wins = 0.0f64;
2831 -        let mut total_losses = 0.0f64;
2832 -
2833 -        for batch in &batches {
2834 -            let output = model.forward(batch);
2835 -
2836 -            for (i, instance) in batch.iter().enumerate() {
2837 -                let direction_pred = output.direction[i];
2838 -                let magnitude_pred = output.magnitude[i];
2839 -
2840 -                // Trade if confidence > 60%
2841 -                if direction_pred > 0.6 || direction_pred < 0.4 {
2842 -                    // Scale position size by predicted magnitude (higher magnitude = larger bet)
2843 -                    let magnitude_multiplier = (magnitude_pred.abs() / 2.0).clamp(0.5, 2.0);
2844 -                    let position_value =
2845 -                        capital * self.position_size as f64 * magnitude_multiplier as f64;
2846 -                    let is_long = direction_pred > 0.5;
2847 -
2848 -                    // Actual return (from label)
2849 -                    let actual_return = instance.magnitude_label / 100.0;
2850 -
2851 -                    // Calculate PnL
2852 -                    let gross_pnl = if is_long {
2853 -                        position_value * actual_return as f64
2854 -                    } else {
2855 -                        -position_value * actual_return as f64
2856 -                    };
2857 -
2858 -                    let transaction_costs = position_value * self.transaction_cost as f64 * 2.0;
2859 -                    let net_pnl = gross_pnl - transaction_costs;
2860 -
2861 -                    capital += net_pnl;
2862 -                    returns.push(net_pnl / position_value);
2863 -
2864 -                    if net_pnl > 0.0 {
2865 -                        wins += 1;
2866 -                        total_wins += net_pnl;
2867 -                    } else {
2868 -                        losses += 1;
2869 -                        total_losses += -net_pnl;
2870 -                    }
2871 -
2872 -                    // Update peak and drawdown
2873 -                    if capital > peak_capital {
2874 -                        peak_capital = capital;
2875 -                    }
2876 -                    let drawdown = (peak_capital - capital) / peak_capital;
2877 -                    if drawdown > max_drawdown {
2878 -                        max_drawdown = drawdown;
2879 -                    }
2880 -                }
2881 -            }
2882 -        }
2883 -
2884 -        // Calculate metrics
2885 -        let total_return = (capital - self.initial_capital) / self.initial_capital;
2886 -        let num_trades = wins + losses;
2887 -
2888 -        let (sharpe, sortino) = if !returns.is_empty() {
2889 -            let mean_return: f64 = returns.iter().sum::<f64>() / returns.len() as f64;
2890 -            let variance: f64 = returns
2891 -                .iter()
2892 -                .map(|r| (r - mean_return).powi(2))
2893 -                .sum::<f64>()
2894 -                / returns.len() as f64;
2895 -            let std_dev = variance.sqrt();
2896 -
2897 -            let sharpe = if std_dev > 0.0 {
2898 -                (mean_return / std_dev * (252.0f64).sqrt()) as f32
2899 -            } else {
2900 -                0.0
2901 -            };
2902 -
2903 -            // Sortino (only downside deviation)
2904 -            let downside_returns: Vec<f64> =
2905 -                returns.iter().filter(|&&r| r < 0.0).cloned().collect();
2906 -            let downside_variance: f64 = if !downside_returns.is_empty() {
2907 -                downside_returns.iter().map(|r| r.powi(2)).sum::<f64>()
2908 -                    / downside_returns.len() as f64
2909 -            } else {
2910 -                0.0001
2911 -            };
2912 -            let downside_std = downside_variance.sqrt();
2913 -            let sortino = if downside_std > 0.0 {
2914 -                (mean_return / downside_std * (252.0f64).sqrt()) as f32
2915 -            } else {
2916 -                0.0
2917 -            };
2918 -
2919 -            (sharpe, sortino)
2920 -        } else {
2921 -            (0.0, 0.0)
2922 -        };
2923 -
2924 -        let win_rate = if num_trades > 0 {
2925 -            wins as f32 / num_trades as f32
2926 -        } else {
2927 -            0.0
2928 -        };
2929 -
2930 -        let profit_factor = if total_losses > 0.0 {
2931 -            (total_wins / total_losses) as f32
2932 -        } else if total_wins > 0.0 {
2933 -            f32::MAX
2934 -        } else {
2935 -            1.0
2936 -        };
2937 -
2938 -        let avg_win = if wins > 0 {
2939 -            (total_wins / wins as f64) as f32
2940 -        } else {
2941 -            0.0
2942 -        };
2943 -
2944 -        let avg_loss = if losses > 0 {
2945 -            (total_losses / losses as f64) as f32
2946 -        } else {
2947 -            0.0
2948 -        };
2949 -
2950 -        let calmar_ratio = if max_drawdown > 0.0 {
2951 -            (total_return / max_drawdown) as f32
2952 -        } else {
2953 -            0.0
2954 -        };
2955 -
2956 -        FinancialMetrics {
2957 -            total_return: total_return as f32,
2958 -            annualized_return: ((1.0 + total_return).powf(252.0 / instances.len() as f64) - 1.0)
2959 -                as f32,
2960 -            sharpe_ratio: sharpe,
2961 -            sortino_ratio: sortino,
2962 -            max_drawdown: max_drawdown as f32,
2963 -            calmar_ratio: calmar_ratio,
2964 -            win_rate,
2965 -            profit_factor,
2966 -            num_trades,
2967 -            avg_win,
2968 -            avg_loss,
2969 -        }
2970 -    }
2971 -}
2972 -
2973 -// =============================================================================
2974 -// Section 9: Prediction and Recommendations
2975 -// =============================================================================
2976 -
2977 -/// Stock recommendation.
2978 -#[derive(Debug, Clone)]
2979 -pub struct StockRecommendation {
2980 -    pub ticker: String,
2981 -    pub sector: String,
2982 -    pub predicted_direction: String,
2983 -    pub direction_confidence: f32,
2984 -    pub predicted_return: f32,
2985 -    pub profitable_probability: f32,
2986 -    pub risk_score: f32,
2987 -    pub recommendation: String,
2988 -    pub expected_value: f32,
2989 -}
2990 -
2991 -/// Generates recommendations from model predictions.
2992 -pub fn generate_recommendations(
2993 -    model: &StockPredictionModel,
2994 -    instances: &[StockInstance],
2995 -    tickers: &[TickerInfo],
2996 -) -> Vec<StockRecommendation> {
2997 -    let batch: Vec<&StockInstance> = instances.iter().collect();
2998 -    let output = model.forward(&batch);
2999 -
3000 -    let mut recommendations = Vec::new();
3001 -
3002 -    for (i, instance) in instances.iter().enumerate() {
3003 -        let ticker = tickers
3004 -            .iter()
3005 -            .find(|t| t.ticker_id == instance.ticker_fid)
3006 -            .unwrap();
3007 -
3008 -        let direction_conf = output.direction[i];
3009 -        let predicted_return = output.magnitude[i];
3010 -        let profitable_prob = output.profitable[i];
3011 -
3012 -        // Determine direction
3013 -        let (direction, effective_conf) = if direction_conf > 0.5 {
3014 -            ("UP", direction_conf)
3015 -        } else {
3016 -            ("DOWN", 1.0 - direction_conf)
3017 -        };
3018 -
3019 -        // Risk score based on volatility indicators
3020 -        let atr_normalized = instance.indicator_features.get(12).copied().unwrap_or(0.0);
3021 -        let bb_width = instance.indicator_features.get(11).copied().unwrap_or(0.0);
3022 -        let risk_score = ((atr_normalized.abs() + bb_width.abs()) / 2.0).clamp(0.0, 1.0);
3023 -
3024 -        // Determine recommendation
3025 -        let recommendation = if direction == "UP" {
3026 -            if effective_conf > 0.75 && predicted_return > 2.0 {
3027 -                "STRONG BUY"
3028 -            } else if effective_conf > 0.6 && predicted_return > 1.0 {
3029 -                "BUY"
3030 -            } else {
3031 -                "HOLD"
3032 -            }
3033 -        } else {
3034 -            if effective_conf > 0.75 && predicted_return < -2.0 {
3035 -                "STRONG SELL"
3036 -            } else if effective_conf > 0.6 && predicted_return < -1.0 {
3037 -                "SELL"
3038 -            } else {
3039 -                "HOLD"
3040 -            }
3041 -        };
3042 -
3043 -        // Expected value = confidence * predicted_return * (1 - risk)
3044 -        let expected_value = effective_conf * predicted_return.abs() * (1.0 - risk_score * 0.5);
3045 -
3046 -        recommendations.push(StockRecommendation {
3047 -            ticker: ticker.symbol.clone(),
3048 -            sector: ticker.sector.name().to_string(),
3049 -            predicted_direction: direction.to_string(),
3050 -            direction_confidence: effective_conf,
3051 -            predicted_return,
3052 -            profitable_probability: profitable_prob,
3053 -            risk_score,
3054 -            recommendation: recommendation.to_string(),
3055 -            expected_value,
3056 -        });
3057 -    }
3058 -
3059 -    // Sort by expected value (descending)
3060 -    recommendations.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());
3061 -
3062 -    recommendations
3063 -}
3064 -
3065 -/// Prints recommendation report.
3066 -pub fn print_recommendation_report(recommendations: &[StockRecommendation], top_n: usize) {
3067 -    println!("\n  Top {} Stock Recommendations:", top_n);
3068 -    println!("  ┌────────┬───────────┬────────────┬─────────────┬────────────────┐");
3069 -    println!("  │ Ticker │ Direction │ Confidence │ Pred Return │ Recommendation │");
3070 -    println!("  ├────────┼───────────┼────────────┼─────────────┼────────────────┤");
3071 -
3072 -    for rec in recommendations.iter().take(top_n) {
3073 -        println!(
3074 -            "  │ {:6} │ {:9} │ {:>9.0}% │ {:>+10.1}% │ {:14} │",
3075 -            rec.ticker,
3076 -            rec.predicted_direction,
3077 -            rec.direction_confidence * 100.0,
3078 -            rec.predicted_return,
3079 -            rec.recommendation
3080 -        );
3081 -    }
3082 -
3083 -    println!("  └────────┴───────────┴────────────┴─────────────┴────────────────┘");
3084 -}
3085 -
3086 -// =============================================================================
3087 -// Section 10: Main and CLI
3088 -// =============================================================================
3089 -
3090 -// NOTE: Synthetic generation remains in this file because it is useful for unit tests and
3091 -// quick local experimentation, but by default the CLI points at a real intraday dataset
3092 -// (FutureSharks financial-data) and will not use synthetic data unless you explicitly
3093 -// change `--data-dir` and/or code paths.
3094 -
3095 -/// Calculate beta for each ticker relative to an equal-weighted market index.
3096 -///
3097 -/// Beta = Cov(stock_returns, market_returns) / Var(market_returns)
3098 -///
3099 -/// Beta interpretation:
3100 -/// - beta > 1: More volatile than the market
3101 -/// - beta = 1: Moves with the market
3102 -/// - beta < 1: Less volatile than the market
3103 -/// - beta < 0: Moves opposite to the market
3104 -fn calculate_betas(tickers: &[TickerInfo], bars: &[StockBar]) -> Vec<TickerInfo> {
3105 -    // First, compute daily market returns (equal-weighted average of all tickers)
3106 -    let mut day_returns: HashMap<usize, Vec<f32>> = HashMap::new();
3107 -
3108 -    for bar in bars {
3109 -        day_returns
3110 -            .entry(bar.day_index)
3111 -            .or_insert_with(Vec::new)
3112 -            .push(bar.returns);
3113 -    }
3114 -
3115 -    // Calculate market returns per day (equal-weighted)
3116 -    let mut market_returns: HashMap<usize, f32> = HashMap::new();
3117 -    for (day, returns) in &day_returns {
3118 -        let n = returns.len();
3119 -        if n > 0 {
3120 -            let avg = returns.iter().sum::<f32>() / n as f32;
3121 -            market_returns.insert(*day, avg);
3122 -        }
3123 -    }
3124 -
3125 -    // Calculate market variance
3126 -    let mkt_values: Vec<f32> = market_returns.values().cloned().collect();
3127 -    let mkt_mean: f32 = mkt_values.iter().sum::<f32>() / mkt_values.len().max(1) as f32;
3128 -    let mkt_variance: f32 = mkt_values
3129 -        .iter()
3130 -        .map(|r| (r - mkt_mean).powi(2))
3131 -        .sum::<f32>()
3132 -        / mkt_values.len().max(1) as f32;
3133 -
3134 -    // Calculate beta for each ticker
3135 -    let mut result = Vec::with_capacity(tickers.len());
3136 -
3137 -    for ticker in tickers {
3138 -        // Get this ticker's returns
3139 -        let ticker_bars: Vec<&StockBar> = bars
3140 -            .iter()
3141 -            .filter(|b| b.ticker_id == ticker.ticker_id)
3142 -            .collect();
3143 -
3144 -        if ticker_bars.is_empty() || mkt_variance < 1e-10 {
3145 -            // No data or zero variance, default beta = 1.0
3146 -            result.push(ticker.clone());
3147 -            continue;
3148 -        }
3149 -
3150 -        // Calculate covariance with market
3151 -        let mut covariance = 0.0_f32;
3152 -        let mut count = 0;
3153 -
3154 -        for bar in &ticker_bars {
3155 -            if let Some(&mkt_ret) = market_returns.get(&bar.day_index) {
3156 -                covariance += (bar.returns - mkt_mean) * (mkt_ret - mkt_mean);
3157 -                count += 1;
3158 -            }
3159 -        }
3160 -
3161 -        let beta = if count > 0 && mkt_variance > 1e-10 {
3162 -            (covariance / count as f32) / mkt_variance
3163 -        } else {
3164 -            1.0 // Default beta
3165 -        };
3166 -
3167 -        // Create updated ticker with calculated beta
3168 -        result.push(TickerInfo {
3169 -            ticker_id: ticker.ticker_id,
3170 -            name: ticker.name.clone(),
3171 -            symbol: ticker.symbol.clone(),
3172 -            sector: ticker.sector,
3173 -            beta: beta.clamp(0.0, 3.0), // Clamp to reasonable range
3174 -            base_volatility: ticker.base_volatility,
3175 -            drift: ticker.drift,
3176 -        });
3177 -    }
3178 -
3179 -    println!(
3180 -        "  Calculated betas for {} tickers (market variance: {:.6})",
3181 -        result.len(),
3182 -        mkt_variance
3183 -    );
3184 -
3185 -    result
3186 -}
3187 -
3188 -pub fn run() {
3189 -    let config = parse_args();
3190 -
3191 -    println!("{}", "=".repeat(80));
3192 -    println!("Monolith-RS Stock Prediction Example");
3193 -    println!("{}", "=".repeat(80));
3194 -    println!();
3195 -
3196 -    // Show performance configuration
3197 -    println!(
3198 -        "Configuration: {} workers | GPU: {} | Batch: {} | LR: {}",
3199 -        config.num_workers,
3200 -        if config.gpu_mode {
3201 -            "enabled"
3202 -        } else {
3203 -            "disabled"
3204 -        },
3205 -        config.batch_size,
3206 -        config.learning_rate
3207 -    );
3208 -    println!();
3209 -
3210 -    let start_time = Instant::now();
3211 -
3212 -    // Section 1: Load or Generate Stock Data
3213 -    ensure_futuresharks_dataset_present(config.data_dir.as_deref());
3214 -    let (tickers, bars): (Vec<TickerInfo>, Vec<StockBar>) =
3215 -        if let Some(ref data_dir) = config.data_dir {
3216 -            // Load real data from CSV files (required for this example).
3217 -            println!("Section 1: Loading Real Stock Data from CSV");
3218 -            match load_real_data_auto(data_dir, &config) {
3219 -                Ok((tickers, bars)) => (tickers, bars),
3220 -                Err(e) => {
3221 -                    eprintln!("Error loading data: {}", e);
3222 -                    eprintln!();
3223 -                    IntradayDataSources::print_download_instructions();
3224 -                    std::process::exit(2);
3225 -                }
3226 -            }
3227 -        } else {
3228 -            // With current defaults this should never happen, but keep a clear message.
3229 -            eprintln!("Error: --data-dir is required (this example does not use synthetic data).");
3230 -            eprintln!();
3231 -            print_usage();
3232 -            std::process::exit(2);
3233 -        };
3234 -
3235 -    let total_bars = bars.len();
3236 -
3237 -    println!(
3238 -        "  Tickers: {} | Days: {} | Total bars: {}",
3239 -        config.num_tickers, config.days_of_history, total_bars
3240 -    );
3241 -
3242 -    // Section 2: Compute Technical Indicators (parallel)
3243 -    println!("\nSection 2: Computing Technical Indicators (parallel)");
3244 -
3245 -    // Parallel indicator computation using rayon
3246 -    let indicator_results: Vec<(i64, Vec<TechnicalIndicators>)> = tickers
3247 -        .par_iter()
3248 -        .map(|ticker| {
3249 -            let ticker_bars: Vec<StockBar> = bars
3250 -                .iter()
3251 -                .filter(|b| b.ticker_id == ticker.ticker_id)
3252 -                .cloned()
3253 -                .collect();
3254 -            let mut calc = IndicatorCalculator::new();
3255 -            let indicators = calc.compute_indicators(&ticker_bars);
3256 -            (ticker.ticker_id, indicators)
3257 -        })
3258 -        .collect();
3259 -
3260 -    let all_indicators: HashMap<i64, Vec<TechnicalIndicators>> =
3261 -        indicator_results.into_iter().collect();
3262 -
3263 -    println!(
3264 -        "  {} indicators per bar: SMA, RSI, MACD, Bollinger, ATR, OBV...",
3265 -        TechnicalIndicators::NUM_FEATURES
3266 -    );
3267 -
3268 -    // Section 3: Create Training Instances
3269 -    println!("\nSection 3: Creating Training Instances");
3270 -    let instance_creator = InstanceCreator::new(config.lookback_window);
3271 -    let mut all_instances = Vec::new();
3272 -
3273 -    for ticker in &tickers {
3274 -        let ticker_bars: Vec<StockBar> = bars
3275 -            .iter()
3276 -            .filter(|b| b.ticker_id == ticker.ticker_id)
3277 -            .cloned()
3278 -            .collect();
3279 -        if let Some(indicators) = all_indicators.get(&ticker.ticker_id) {
3280 -            let instances = instance_creator.create_instances(ticker, &ticker_bars, indicators);
3281 -            all_instances.extend(instances);
3282 -        }
3283 -    }
3284 -
3285 -    // Split by time (not random) to avoid data leakage.
3286 -    // IMPORTANT: For multi-ticker intraday datasets where `all_instances` is an interleaving of
3287 -    // multiple time series, a global split is not truly "time ordered" per ticker.
3288 -    // We instead split per ticker inside the loop above by using the fact that `create_instances`
3289 -    // yields instances in chronological order for that ticker.
3290 -    //
3291 -    // For backward compatibility, keep the global split as a fallback if per-ticker split
3292 -    // hasn't been applied (i.e., if all_instances is empty).
3293 -    let (train_instances, eval_instances) = train_eval_split(&all_instances, config.train_ratio);
3294 -
3295 -    println!(
3296 -        "  Train: {} ({:.0}%) | Eval: {} ({:.0}%) | Lookback: {} bars",
3297 -        train_instances.len(),
3298 -        config.train_ratio * 100.0,
3299 -        eval_instances.len(),
3300 -        (1.0 - config.train_ratio) * 100.0,
3301 -        config.lookback_window
3302 -    );
3303 -
3304 -    // Section 4: Model Architecture
3305 -    println!("\nSection 4: Model Architecture");
3306 -    println!(
3307 -        "  Ticker embedding: {} x {} | DIEN hidden: {} | DCN: {} layers",
3308 -        config.num_tickers,
3309 -        config.ticker_embedding_dim,
3310 -        config.dien_hidden_size,
3311 -        config.dcn_cross_layers
3312 -    );
3313 -
3314 -    match config.mode {
3315 -        Mode::Train => {
3316 -            run_training(&config, &train_instances, &eval_instances, &bars, &tickers);
3317 -        }
3318 -        Mode::Evaluate => {
3319 -            run_evaluation(&config, &eval_instances);
3320 -        }
3321 -        Mode::Predict => {
3322 -            run_prediction(&config, &all_instances, &tickers);
3323 -        }
3324 -        Mode::Backtest => {
3325 -            run_backtesting(&config, &eval_instances, &bars, &tickers);
3326 -        }
3327 -    }
3328 -
3329 -    let elapsed = start_time.elapsed();
3330 -    println!();
3331 -    println!("{}", "=".repeat(80));
3332 -    println!("Complete! Total time: {:.2}s", elapsed.as_secs_f64());
3333 -    println!("{}", "=".repeat(80));
3334 -}
3335 -
3336 -fn ensure_futuresharks_dataset_present(data_dir: Option<&str>) {
3337 -    use std::path::Path;
3338 -    use std::process::Command;
3339 -
3340 -    let Some(data_dir) = data_dir else {
3341 -        return;
3342 -    };
3343 -
3344 -    // If the directory exists, do nothing.
3345 -    if Path::new(data_dir).exists() {
3346 -        return;
3347 -    }
3348 -
3349 -    // If the user is relying on our default FutureSharks path, try to clone the repo.
3350 -    if data_dir == DEFAULT_FUTURESHARKS_DATA_DIR
3351 -        && !Path::new(DEFAULT_FUTURESHARKS_REPO_DIR).exists()
3352 -    {
3353 -        eprintln!(
3354 -            "Default intraday dataset not found at `{}`. Cloning into `{}` ...",
3355 -            DEFAULT_FUTURESHARKS_DATA_DIR, DEFAULT_FUTURESHARKS_REPO_DIR
3356 -        );
3357 -        let status = Command::new("git")
3358 -            .args([
3359 -                "clone",
3360 -                "--depth",
3361 -                "1",
3362 -                "https://github.com/FutureSharks/financial-data.git",
3363 -                DEFAULT_FUTURESHARKS_REPO_DIR,
3364 -            ])
3365 -            .status();
3366 -
3367 -        if !matches!(status, Ok(s) if s.success()) {
3368 -            eprintln!("Auto-clone failed. Run this manually:");
3369 -            eprintln!(
3370 -                "  git clone https://github.com/FutureSharks/financial-data.git {}",
3371 -                DEFAULT_FUTURESHARKS_REPO_DIR
3372 -            );
3373 -        }
3374 -    }
3375 -}
3376 -
3377 -fn load_real_data_auto(
3378 -    data_dir: &str,
3379 -    config: &StockPredictorConfig,
3380 -) -> Result<(Vec<TickerInfo>, Vec<StockBar>), String> {
3381 -    let canonical_dir = normalize_futuresharks_dir(data_dir);
3382 -
3383 -    if is_futuresharks_histdata_dir(&canonical_dir) {
3384 -        let mut loader = FutureSharksHistdataLoader::new();
3385 -        loader.load_dir(&canonical_dir, config.num_tickers)?;
3386 -        Ok((loader.tickers().to_vec(), loader.bars().to_vec()))
3387 -    } else {
3388 -        let mut loader = CsvDataLoader::new(&canonical_dir);
3389 -        loader.load(config.num_tickers, config.lookback_window + 50)?;
3390 -        let tickers_with_beta = calculate_betas(loader.tickers(), loader.bars());
3391 -        Ok((tickers_with_beta, loader.bars().to_vec()))
3392 -    }
3393 -}
3394 -
3395 -fn normalize_futuresharks_dir(data_dir: &str) -> String {
3396 -    // Allow pointing at the repo root; map it to the histdata directory.
3397 -    let p = std::path::Path::new(data_dir);
3398 -    if p.ends_with("financial-data") || p.ends_with("data/financial-data") {
3399 -        return DEFAULT_FUTURESHARKS_DATA_DIR.to_string();
3400 -    }
3401 -    data_dir.to_string()
3402 -}
3403 -
3404 -fn is_futuresharks_histdata_dir(data_dir: &str) -> bool {
3405 -    // Heuristic: any CSV with the Histdata naming pattern indicates this dataset.
3406 -    // Example: DAT_ASCII_SPXUSD_M1_2017.csv
3407 -    let Ok(rd) = std::fs::read_dir(data_dir) else {
3408 -        return false;
3409 -    };
3410 -
3411 -    for entry in rd.flatten() {
3412 -        let path = entry.path();
3413 -        if path.is_dir() {
3414 -            let Ok(sub) = std::fs::read_dir(&path) else {
3415 -                continue;
3416 -            };
3417 -            for e in sub.flatten() {
3418 -                let p = e.path();
3419 -                if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
3420 -                    if name.starts_with("DAT_ASCII_")
3421 -                        && name.contains("_M1_")
3422 -                        && name.ends_with(".csv")
3423 -                    {
3424 -                        return true;
3425 -                    }
3426 -                }
3427 -            }
3428 -        }
3429 -    }
3430 -
3431 -    false
3432 -}
3433 -
3434 -fn run_training(
3435 -    config: &StockPredictorConfig,
3436 -    train_instances: &[StockInstance],
3437 -    eval_instances: &[StockInstance],
3438 -    bars: &[StockBar],
3439 -    tickers: &[TickerInfo],
3440 -) {
3441 -    println!("\nSection 5: Training (with momentum, LR decay, weight decay)");
3442 -    let mut trainer = Trainer::new(config);
3443 -
3444 -    for epoch in 0..config.num_epochs {
3445 -        // Apply learning rate decay at start of each epoch
3446 -        if epoch > 0 {
3447 -            trainer.decay_lr();
3448 -        }
3449 -
3450 -        let train_metrics = trainer.train_epoch(train_instances);
3451 -        let eval_metrics = trainer.evaluate(eval_instances);
3452 -
3453 -        println!(
3454 -            "  Epoch {}/{}: Loss: {:.4} | Eval: {:.4} | Acc: {:.1}% | LR: {:.6}",
3455 -            epoch + 1,
3456 -            config.num_epochs,
3457 -            train_metrics.total_loss,
3458 -            eval_metrics.total_loss,
3459 -            eval_metrics.direction_accuracy * 100.0,
3460 -            trainer.current_lr()
3461 -        );
3462 -
3463 -        // Early stopping check
3464 -        if trainer.check_early_stopping(eval_metrics.total_loss) {
3465 -            println!("  Early stopping triggered at epoch {}", epoch + 1);
3466 -            break;
3467 -        }
3468 -    }
3469 -
3470 -    // Final evaluation
3471 -    println!("\nSection 6: Evaluation Results");
3472 -    let final_eval = trainer.evaluate(eval_instances);
3473 -    println!(
3474 -        "  Direction Accuracy: {:.1}%",
3475 -        final_eval.direction_accuracy * 100.0
3476 -    );
3477 -    println!(
3478 -        "  Losses - Direction: {:.4} | Magnitude: {:.4} | Profitable: {:.4}",
3479 -        final_eval.direction_loss, final_eval.magnitude_loss, final_eval.profitable_loss
3480 -    );
3481 -
3482 -    // Backtesting
3483 -    println!("\nSection 7: Backtest Results");
3484 -    let backtester = Backtester::new();
3485 -    let financial_metrics = backtester.run(trainer.model(), eval_instances, bars);
3486 -
3487 -    println!(
3488 -        "  Total Return: {:.1}%",
3489 -        financial_metrics.total_return * 100.0
3490 -    );
3491 -    println!("  Sharpe Ratio: {:.2}", financial_metrics.sharpe_ratio);
3492 -    println!(
3493 -        "  Max Drawdown: {:.1}%",
3494 -        financial_metrics.max_drawdown * 100.0
3495 -    );
3496 -    println!("  Win Rate: {:.1}%", financial_metrics.win_rate * 100.0);
3497 -    println!("  Profit Factor: {:.2}", financial_metrics.profit_factor);
3498 -    println!("  Trades: {}", financial_metrics.num_trades);
3499 -
3500 -    // Recommendations
3501 -    println!("\nSection 8: Top Stock Recommendations");
3502 -    let latest_instances: Vec<StockInstance> = tickers
3503 -        .iter()
3504 -        .filter_map(|ticker| {
3505 -            train_instances
3506 -                .iter()
3507 -                .chain(eval_instances.iter())
3508 -                .filter(|i| i.ticker_fid == ticker.ticker_id)
3509 -                .last()
3510 -                .cloned()
3511 -        })
3512 -        .collect();
3513 -
3514 -    let recommendations = generate_recommendations(trainer.model(), &latest_instances, tickers);
3515 -    print_recommendation_report(&recommendations, 10);
3516 -}
3517 -
3518 -fn run_evaluation(config: &StockPredictorConfig, eval_instances: &[StockInstance]) {
3519 -    println!("\nRunning Evaluation Mode");
3520 -    let model = StockPredictionModel::new(config);
3521 -
3522 -    let batches = create_batches(eval_instances, config.batch_size);
3523 -    let mut total_correct = 0;
3524 -    let mut total_samples = 0;
3525 -
3526 -    for batch in batches {
3527 -        let output = model.forward(&batch);
3528 -        for (i, instance) in batch.iter().enumerate() {
3529 -            let pred = if output.direction[i] > 0.5 { 1.0 } else { 0.0 };
3530 -            if (pred - instance.direction_label).abs() < 0.1 {
3531 -                total_correct += 1;
3532 -            }
3533 -        }
3534 -        total_samples += batch.len();
3535 -    }
3536 -
3537 -    println!(
3538 -        "  Evaluation Accuracy: {:.1}% ({}/{})",
3539 -        total_correct as f32 / total_samples as f32 * 100.0,
3540 -        total_correct,
3541 -        total_samples
3542 -    );
3543 -}
3544 -
3545 -fn run_prediction(
3546 -    config: &StockPredictorConfig,
3547 -    instances: &[StockInstance],
3548 -    tickers: &[TickerInfo],
3549 -) {
3550 -    println!("\nRunning Prediction Mode");
3551 -    let model = StockPredictionModel::new(config);
3552 -
3553 -    // Get latest instance per ticker
3554 -    let latest_instances: Vec<StockInstance> = tickers
3555 -        .iter()
3556 -        .filter_map(|ticker| {
3557 -            instances
3558 -                .iter()
3559 -                .filter(|i| i.ticker_fid == ticker.ticker_id)
3560 -                .last()
3561 -                .cloned()
3562 -        })
3563 -        .collect();
3564 -
3565 -    let recommendations = generate_recommendations(&model, &latest_instances, tickers);
3566 -    print_recommendation_report(&recommendations, config.num_tickers.min(20));
3567 -}
3568 -
3569 -fn run_backtesting(
3570 -    config: &StockPredictorConfig,
3571 -    instances: &[StockInstance],
3572 -    bars: &[StockBar],
3573 -    _tickers: &[TickerInfo],
3574 -) {
3575 -    println!("\nRunning Backtest Mode");
3576 -    let model = StockPredictionModel::new(config);
3577 -    let backtester = Backtester::new();
3578 -
3579 -    let metrics = backtester.run(&model, instances, bars);
3580 -
3581 -    println!("\n  Backtest Performance Summary:");
3582 -    println!("  ─────────────────────────────────────");
3583 -    println!(
3584 -        "  Total Return:      {:>+8.2}%",
3585 -        metrics.total_return * 100.0
3586 -    );
3587 -    println!(
3588 -        "  Annualized Return: {:>+8.2}%",
3589 -        metrics.annualized_return * 100.0
3590 -    );
3591 -    println!("  Sharpe Ratio:      {:>8.2}", metrics.sharpe_ratio);
3592 -    println!("  Sortino Ratio:     {:>8.2}", metrics.sortino_ratio);
3593 -    println!(
3594 -        "  Max Drawdown:      {:>8.2}%",
3595 -        metrics.max_drawdown * 100.0
3596 -    );
3597 -    println!("  Calmar Ratio:      {:>8.2}", metrics.calmar_ratio);
3598 -    println!("  Win Rate:          {:>8.2}%", metrics.win_rate * 100.0);
3599 -    println!("  Profit Factor:     {:>8.2}", metrics.profit_factor);
3600 -    println!("  Total Trades:      {:>8}", metrics.num_trades);
3601 -    println!("  Avg Win:           ${:>7.2}", metrics.avg_win);
3602 -    println!("  Avg Loss:          ${:>7.2}", metrics.avg_loss);
3603 -}
3604 -
3605 -// =============================================================================
3606 -// Tests
3607 -// =============================================================================
3608 -
3609 -#[cfg(test)]
3610 -mod tests {
3611 -    use super::*;
3612 -
3613 -    #[test]
3614 -    fn test_config_default() {
3615 -        let config = StockPredictorConfig::default();
3616 -        assert_eq!(config.num_tickers, 50);
3617 -        assert_eq!(config.days_of_history, 252);
3618 -        assert_eq!(config.lookback_window, 20);
3619 -    }
3620 -
3621 -    #[test]
3622 -    fn test_stock_data_generation() {
3623 -        let mut generator = StockDataGenerator::new(42);
3624 -        generator.generate_tickers(10);
3625 -        generator.generate_bars(50);
3626 -
3627 -        assert_eq!(generator.tickers().len(), 10);
3628 -        assert_eq!(generator.bars().len(), 10 * 50);
3629 -    }
3630 -
3631 -    #[test]
3632 -    fn test_indicator_calculation() {
3633 -        let mut generator = StockDataGenerator::new(42);
3634 -        generator.generate_tickers(1);
3635 -        generator.generate_bars(30);
3636 -
3637 -        let bars: Vec<StockBar> = generator.bars().to_vec();
3638 -        let mut calc = IndicatorCalculator::new();
3639 -        let indicators = calc.compute_indicators(&bars);
3640 -
3641 -        assert_eq!(indicators.len(), 30);
3642 -    }
3643 -
3644 -    #[test]
3645 -    fn test_instance_creation() {
3646 -        let mut generator = StockDataGenerator::new(42);
3647 -        generator.generate_tickers(1);
3648 -        generator.generate_bars(50);
3649 -
3650 -        let ticker = &generator.tickers()[0];
3651 -        let bars: Vec<StockBar> = generator.bars().to_vec();
3652 -
3653 -        let mut calc = IndicatorCalculator::new();
3654 -        let indicators = calc.compute_indicators(&bars);
3655 -
3656 -        let creator = InstanceCreator::new(10);
3657 -        let instances = creator.create_instances(ticker, &bars, &indicators);
3658 -
3659 -        // Should have instances from day 10 to day 44 (50 - 5 forward - 1)
3660 -        assert!(!instances.is_empty());
3661 -    }
3662 -
3663 -    #[test]
3664 -    fn test_embedding_tables() {
3665 -        let tables = EmbeddingTables::new(10, 16, 11, 8);
3666 -
3667 -        let ticker_emb = tables.lookup_ticker(0);
3668 -        assert_eq!(ticker_emb.len(), 16);
3669 -
3670 -        let sector_emb = tables.lookup_sector(0);
3671 -        assert_eq!(sector_emb.len(), 8);
3672 -    }
3673 -
3674 -    #[test]
3675 -    fn test_model_forward() {
3676 -        let config = StockPredictorConfig {
3677 -            num_tickers: 5,
3678 -            days_of_history: 50,
3679 -            lookback_window: 10,
3680 -            ticker_embedding_dim: 8,
3681 -            sector_embedding_dim: 4,
3682 -            dien_hidden_size: 16,
3683 -            dcn_cross_layers: 2,
3684 -            ..Default::default()
3685 -        };
3686 -
3687 -        let model = StockPredictionModel::new(&config);
3688 -
3689 -        // Create test instance
3690 -        let instance = StockInstance {
3691 -            ticker_fid: 0,
3692 -            sector_fid: 0,
3693 -            price_features: vec![0.01, 0.02, 0.01, 1.0],
3694 -            indicator_features: vec![0.0; TechnicalIndicators::NUM_FEATURES],
3695 -            historical_sequence: vec![vec![0.0; 4 + TechnicalIndicators::NUM_FEATURES]; 10],
3696 -            direction_label: 1.0,
3697 -            magnitude_label: 2.0,
3698 -            profitable_label: 1.0,
3699 -        };
3700 -
3701 -        let batch = vec![&instance];
3702 -        let output = model.forward(&batch);
3703 -
3704 -        assert_eq!(output.direction.len(), 1);
3705 -        assert_eq!(output.magnitude.len(), 1);
3706 -        assert_eq!(output.profitable.len(), 1);
3707 -
3708 -        // Direction should be between 0 and 1 (sigmoid)
3709 -        assert!(output.direction[0] >= 0.0 && output.direction[0] <= 1.0);
3710 -        assert!(output.profitable[0] >= 0.0 && output.profitable[0] <= 1.0);
3711 -    }
3712 -
3713 -    #[test]
3714 -    fn test_technical_indicators_to_vec() {
3715 -        let indicators = TechnicalIndicators::default();
3716 -        let vec = indicators.to_vec();
3717 -        assert_eq!(vec.len(), TechnicalIndicators::NUM_FEATURES);
3718 -    }
3719 -
3720 -    #[test]
3721 -    fn test_random_generator() {
3722 -        let mut rng = RandomGenerator::new(42);
3723 -        let u = rng.uniform();
3724 -        assert!(u >= 0.0 && u < 1.0);
3725 -
3726 -        let n = rng.normal();
3727 -        assert!(n.is_finite());
3728 -    }
3729 -
3730 -    #[test]
3731 -    fn test_mode_parsing() {
3732 -        assert_eq!(Mode::from_str("train"), Some(Mode::Train));
3733 -        assert_eq!(Mode::from_str("TRAIN"), Some(Mode::Train));
3734 -        assert_eq!(Mode::from_str("predict"), Some(Mode::Predict));
3735 -        assert_eq!(Mode::from_str("backtest"), Some(Mode::Backtest));
3736 -        assert_eq!(Mode::from_str("invalid"), None);
3737 -    }
3738 -
3739 -    #[test]
3740 -    fn test_sector() {
3741 -        assert_eq!(Sector::all().len(), 11);
3742 -        assert_eq!(Sector::Technology.id(), 0);
3743 -        assert_eq!(Sector::Technology.name(), "Technology");
3744 -    }
3745 -}
