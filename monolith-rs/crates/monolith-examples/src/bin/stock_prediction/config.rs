use num_cpus;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Train,
    Evaluate,
    Predict,
    Backtest,
}

impl Mode {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "train" => Some(Mode::Train),
            "evaluate" | "eval" => Some(Mode::Evaluate),
            "predict" => Some(Mode::Predict),
            "backtest" => Some(Mode::Backtest),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StockPredictorConfig {
    pub num_tickers: usize,
    pub days_of_history: usize,
    pub lookback_window: usize,
    /// Additional pooled lookback horizons (in bars) to provide multi-horizon context.
    /// `lookback_window` is automatically set to `max(lookbacks)` when parsing args.
    pub lookbacks: Vec<usize>,

    pub intraday_file: Option<String>,
    pub intraday_ticker: Option<String>,
    pub intraday_aggregate: usize,

    pub ticker_embedding_dim: usize,
    pub sector_embedding_dim: usize,
    pub dien_hidden_size: usize,
    pub dcn_cross_layers: usize,

    pub batch_size: usize,
    pub learning_rate: f32,
    pub train_ratio: f32,
    pub num_epochs: usize,
    pub log_every_n_steps: usize,
    pub early_stopping_patience: usize,
    pub early_stopping_min_delta: f32,

    pub mode: Mode,
    pub seed: u64,
    pub verbose: bool,

    pub data_dir: Option<String>,

    pub num_workers: usize,
    pub gpu_mode: bool,

    // Data volume controls
    pub max_bars_per_ticker: usize,
    pub bar_stride: usize,

    // Feature timeframes in minutes (must include 1)
    pub timeframes: Vec<usize>,
}

/// Default location for local intraday datasets checked into/outside this repo.
///
/// In this repo we keep datasets out of `data/` (which is gitignored) and instead
/// use `examples/stock_prediction/data/` for the stock prediction example.
pub const DEFAULT_STOCK_PREDICTION_DATA_DIR: &str = "examples/stock_prediction/data";

impl Default for StockPredictorConfig {
    fn default() -> Self {
        Self {
            num_tickers: 50,
            days_of_history: 252,
            lookback_window: 20,
            lookbacks: vec![20],

            intraday_file: None,
            intraday_ticker: None,
            intraday_aggregate: 5,

            ticker_embedding_dim: 128,
            sector_embedding_dim: 64,
            dien_hidden_size: 0,
            dcn_cross_layers: 4,

            batch_size: 256,
            learning_rate: 0.0003,
            train_ratio: 0.8,
            num_epochs: 100,
            log_every_n_steps: 50,
            early_stopping_patience: 20,
            early_stopping_min_delta: 1e-4,

            mode: Mode::Train,
            seed: 42,
            verbose: true,

            data_dir: Some(DEFAULT_STOCK_PREDICTION_DATA_DIR.to_string()),

            num_workers: 0,
            gpu_mode: cfg!(any(feature = "metal", feature = "cuda")),

            max_bars_per_ticker: 0,
            bar_stride: 5,

            timeframes: vec![1, 5, 15, 30, 60],
        }
    }
}

pub fn parse_args() -> StockPredictorConfig {
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
                    let v = args[i + 1].parse().unwrap_or(120);
                    config.lookbacks = vec![v.max(1)];
                    config.lookback_window = v.max(1);
                    i += 1;
                }
            }
            "--lookbacks" => {
                if i + 1 < args.len() {
                    let parsed: Vec<usize> = args[i + 1]
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .filter(|&v| v > 0)
                        .collect();
                    if !parsed.is_empty() {
                        config.lookbacks = parsed;
                    }
                    i += 1;
                }
            }
            "--intraday-file" => {
                if i + 1 < args.len() {
                    config.intraday_file = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--intraday-ticker" => {
                if i + 1 < args.len() {
                    config.intraday_ticker = Some(args[i + 1].clone());
                    i += 1;
                }
            }
            "--intraday-agg" | "--intraday-aggregate" => {
                if i + 1 < args.len() {
                    config.intraday_aggregate = args[i + 1].parse().unwrap_or(5);
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
            "--max-bars" => {
                if i + 1 < args.len() {
                    config.max_bars_per_ticker = args[i + 1].parse().unwrap_or(50_000);
                    i += 1;
                }
            }
            "--bar-stride" => {
                if i + 1 < args.len() {
                    config.bar_stride = args[i + 1].parse().unwrap_or(5);
                    i += 1;
                }
            }
            "--timeframes" => {
                if i + 1 < args.len() {
                    let parsed: Vec<usize> = args[i + 1]
                        .split(',')
                        .filter_map(|s| s.trim().parse::<usize>().ok())
                        .filter(|&v| v > 0)
                        .collect();
                    if !parsed.is_empty() {
                        config.timeframes = parsed;
                    }
                    i += 1;
                }
            }
            "--help" | "-h" => {
                print_usage();
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    if !config.timeframes.contains(&1) {
        config.timeframes.push(1);
    }
    config.timeframes.sort_unstable();
    config.timeframes.dedup();

    if config.lookbacks.is_empty() {
        config.lookbacks = vec![config.lookback_window.max(1)];
    }
    config.lookbacks.iter_mut().for_each(|v| {
        if *v == 0 {
            *v = 1;
        }
    });
    config.lookbacks.sort_unstable();
    config.lookbacks.dedup();
    config.lookback_window = *config.lookbacks.last().unwrap_or(&config.lookback_window).max(&1);

    if config.num_workers == 0 {
        config.num_workers = num_cpus::get();
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(config.num_workers)
        .build_global()
        .ok();

    if config.gpu_mode && !cfg!(any(feature = "metal", feature = "cuda")) {
        eprintln!("Warning: --gpu requested but metal/cuda features are not enabled.");
        eprintln!("         Rebuild with: --features metal (macOS) or --features cuda (Linux/Windows)");
        config.gpu_mode = false;
    }

    config
}

pub fn print_usage() {
    println!(
        r#"Monolith-RS Stock Prediction Example

USAGE:
    stock_prediction [OPTIONS]

OPTIONS:
    -m, --mode <MODE>           Operation mode: train, evaluate, predict, backtest
                                [default: train]
    -t, --num-tickers <N>       Number of tickers to load/simulate [default: 50]
    -d, --days <N>              Days of historical data [default: 252]
    -l, --lookback <N>          Lookback window size (sets --lookbacks to this single value) [default: 120]
    --lookbacks <LIST>          Comma-separated pooled lookbacks (bars) [default: 20,60,120]
    --intraday-file <PATH>      Load a single intraday CSV (supported formats)
    --intraday-ticker <SYMBOL>  Override ticker symbol for --intraday-file
    --intraday-agg <MIN>        Aggregate intraday minutes before feature calc [default: 5]
    -b, --batch-size <N>        Training batch size [default: 32]
    -lr, --learning-rate <LR>   Learning rate [default: 0.001]
    -e, --epochs <N>            Number of training epochs [default: 10]
    --patience <N>              Early stopping patience (epochs without improvement) [default: 20]
    --min-delta <X>             Minimum eval loss improvement to reset patience [default: 1e-4]
    -s, --seed <SEED>           Random seed [default: 42]
    --data-dir <PATH>           Load real stock data from CSV files in directory
                                [default: examples/stock_prediction/data]
    -w, --workers <N>           Number of parallel workers [default: auto-detect]
    --gpu                       Enable GPU acceleration mode (requires metal/cuda feature)
    --max-bars <N>              Max bars per ticker after load [default: 0=all]
    --bar-stride <N>            Downsample bars by stride [default: 5]
    --timeframes <LIST>         Comma-separated minutes (pre-stride) [default: 1,5,15,30,60]
    -q, --quiet                 Suppress verbose output
    -h, --help                  Print help information
"#
    );
}
