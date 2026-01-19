use std::collections::HashMap;

use super::config::{DEFAULT_FUTURESHARKS_DATA_DIR, DEFAULT_FUTURESHARKS_REPO_DIR};

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
    pub fn all() -> &'static [Sector] {
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

    pub fn id(&self) -> i64 {
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

    pub fn name(&self) -> &'static str {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

#[derive(Clone)]
pub struct RandomGenerator {
    state: u64,
}

impl RandomGenerator {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 33) as f32 / (1u64 << 31) as f32
    }

    pub fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-10);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    pub fn choice<T: Clone>(&mut self, items: &[T]) -> T {
        let idx = (self.uniform() * items.len() as f32) as usize % items.len();
        items[idx].clone()
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.uniform() * (i + 1) as f32) as usize % (i + 1);
            slice.swap(i, j);
        }
    }
}

pub struct StockDataGenerator {
    rng: RandomGenerator,
    tickers: Vec<TickerInfo>,
    bars: Vec<StockBar>,
    current_regime: MarketRegime,
    regime_counter: usize,
}

impl StockDataGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RandomGenerator::new(seed),
            tickers: Vec::new(),
            bars: Vec::new(),
            current_regime: MarketRegime::Sideways,
            regime_counter: 0,
        }
    }

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

    pub fn generate_bars(&mut self, days: usize) -> &[StockBar] {
        self.bars.clear();

        if self.tickers.is_empty() {
            return &self.bars;
        }

        let mut prices: Vec<f32> = self
            .tickers
            .iter()
            .map(|_| 50.0 + self.rng.uniform() * 150.0)
            .collect();

        let mut volatilities: Vec<f32> = self.tickers.iter().map(|t| t.base_volatility).collect();
        let mut market_returns: Vec<f32> = Vec::with_capacity(days);

        for day in 0..days {
            self.update_regime();

            let market_return = match self.current_regime {
                MarketRegime::Bull => 0.0005 + self.rng.normal() * 0.01,
                MarketRegime::Bear => -0.0003 + self.rng.normal() * 0.015,
                MarketRegime::Sideways => self.rng.normal() * 0.008,
            };
            market_returns.push(market_return);

            let is_earnings_season = (day % 63) < 21;

            for (ticker_idx, ticker) in self.tickers.iter().enumerate() {
                let target_vol = ticker.base_volatility;
                volatilities[ticker_idx] = volatilities[ticker_idx] * 0.95 + target_vol * 0.05;

                let vol = if is_earnings_season && self.rng.uniform() < 0.1 {
                    volatilities[ticker_idx] * 2.0
                } else {
                    volatilities[ticker_idx]
                };

                let idiosyncratic = self.rng.normal() * vol;
                let daily_return = ticker.beta * market_return + idiosyncratic + ticker.drift;

                let prev_price = prices[ticker_idx];
                let new_price = prev_price * (1.0 + daily_return);
                prices[ticker_idx] = new_price.max(0.01);

                let intraday_vol = vol * 0.5;
                let open = prev_price * (1.0 + self.rng.normal() * intraday_vol * 0.3);
                let high = new_price.max(open) * (1.0 + self.rng.uniform() * intraday_vol);
                let low = new_price.min(open) * (1.0 - self.rng.uniform() * intraday_vol);

                let base_volume = 1_000_000.0 + self.rng.uniform() as f64 * 10_000_000.0;
                let volume_mult = (1.0 + daily_return.abs() * 10.0) as f64;
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

        if self.regime_counter > 20 && self.rng.uniform() < 0.05 {
            self.current_regime = match self.rng.uniform() {
                x if x < 0.4 => MarketRegime::Bull,
                x if x < 0.7 => MarketRegime::Sideways,
                _ => MarketRegime::Bear,
            };
            self.regime_counter = 0;
        }
    }

    #[allow(dead_code)]
    pub fn tickers(&self) -> &[TickerInfo] {
        &self.tickers
    }

    #[allow(dead_code)]
    pub fn bars(&self) -> &[StockBar] {
        &self.bars
    }
}

pub struct CsvDataLoader {
    data_dir: String,
    ticker_to_id: HashMap<String, i64>,
    tickers: Vec<TickerInfo>,
    bars: Vec<StockBar>,
}

impl CsvDataLoader {
    pub fn new(data_dir: &str) -> Self {
        Self {
            data_dir: data_dir.to_string(),
            ticker_to_id: HashMap::new(),
            tickers: Vec::new(),
            bars: Vec::new(),
        }
    }

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

        csv_files.sort_by_key(|a| a.path());
        csv_files.truncate(max_tickers);

        println!(
            "  Loading {} CSV files from {}",
            csv_files.len(),
            self.data_dir
        );

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

            let format = if let Some(Ok(h)) = &header {
                let h_lower = h.to_lowercase();
                if h_lower.contains("adj close") {
                    "yahoo"
                } else if h_lower.contains("volume") {
                    "simple"
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

                    if close <= 0.0 || open <= 0.0 {
                        continue;
                    }

                    let returns = if let Some(pc) = prev_close {
                        (close - pc) / pc
                    } else {
                        0.0
                    };
                    prev_close = Some(close);

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

            if ticker_bars.len() < min_days {
                continue;
            }

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
                _ => match ticker_symbol.chars().next() {
                    Some('A'..='F') => Sector::Technology,
                    Some('G'..='L') => Sector::Industrial,
                    Some('M'..='R') => Sector::Healthcare,
                    Some('S'..='Z') => Sector::Finance,
                    _ => Sector::Consumer,
                },
            };

            let returns: Vec<f32> = ticker_bars.iter().map(|b| b.returns).collect();
            let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
            let variance: f32 =
                returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>() / returns.len() as f32;
            let volatility = variance.sqrt() * (252.0_f32).sqrt();

            let drift = mean_ret * 252.0;

            self.ticker_to_id
                .insert(ticker_symbol.clone(), ticker_id as i64);

            self.tickers.push(TickerInfo {
                ticker_id: ticker_id as i64,
                name: ticker_symbol.clone(),
                symbol: ticker_symbol,
                sector,
                beta: 1.0,
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

    pub fn tickers(&self) -> &[TickerInfo] {
        &self.tickers
    }

    pub fn bars(&self) -> &[StockBar] {
        &self.bars
    }
}

pub struct IntradayDataSources;

impl IntradayDataSources {
    pub fn futuresharks_urls() -> Vec<(&'static str, &'static str)> {
        vec![
            ("SPX500", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/SPX500_USD_M1.csv.gz"),
            ("NAS100", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/NAS100_USD_M1.csv.gz"),
            ("JP225", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/JP225_USD_M1.csv.gz"),
            ("DE30", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/DE30_EUR_M1.csv.gz"),
            ("UK100", "https://raw.githubusercontent.com/FutureSharks/financial-data/master/data/stocks/oanda/UK100_GBP_M1.csv.gz"),
        ]
    }

    pub fn print_download_instructions() {
        println!("\n=== Intraday Data Download Instructions ===\n");
        println!("Option 1: FutureSharks financial-data (1-minute bars, 2010-2018)");
        println!("  git clone https://github.com/FutureSharks/financial-data.git");
        println!("  # Default path is data/financial-data/data/stocks/oanda\n");

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
}

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
        let mut minute_bars: Vec<(f32, f32, f32, f32, f64)> = Vec::new();

        for (i, line) in reader.lines().enumerate() {
            if i == 0 {
                continue;
            }

            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let fields: Vec<&str> = line.split(',').collect();

            if fields.len() < 6 {
                continue;
            }

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

        let aggregated = Self::aggregate_bars(&minute_bars, aggregate_minutes);

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
            sector: Sector::Technology,
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

    pub fn load_dir(&mut self, dir: &str, max_instruments: usize) -> Result<(), String> {
        use std::fs;

        let mut instrument_dirs: Vec<_> = fs::read_dir(dir)
            .map_err(|e| format!("Failed to read {}: {}", dir, e))?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .collect();

        instrument_dirs.sort_by_key(|a| a.path());
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

        files.sort_by_key(|a| a.path());

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

            for line in reader.lines().map_while(Result::ok) {
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

                let (date_part, time_part) = match ts.split_once(' ') {
                    Some((d, t)) => (d, t),
                    None => continue,
                };
                if date_part.len() != 8 || time_part.len() < 4 {
                    continue;
                }
                let day: i64 = date_part.parse().unwrap_or(0);
                let hhmm: i64 = time_part[..4].parse().unwrap_or(0);

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

        let returns: Vec<f32> = bars_for_ticker.iter().map(|b| b.returns).collect();
        let mean_ret: f32 = returns.iter().sum::<f32>() / returns.len().max(1) as f32;
        let variance: f32 = returns.iter().map(|r| (r - mean_ret).powi(2)).sum::<f32>()
            / returns.len().max(1) as f32;
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

pub fn ensure_futuresharks_dataset_present(data_dir: Option<&str>) {
    use std::path::Path;
    use std::process::Command;

    let Some(data_dir) = data_dir else {
        return;
    };

    if Path::new(data_dir).exists() {
        return;
    }

    if Path::new(DEFAULT_FUTURESHARKS_REPO_DIR).exists() {
        return;
    }

    if data_dir == DEFAULT_FUTURESHARKS_DATA_DIR
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

pub fn load_real_data_auto(
    data_dir: &str,
    num_tickers: usize,
    lookback_window: usize,
) -> Result<(Vec<TickerInfo>, Vec<StockBar>), String> {
    let canonical_dir = normalize_futuresharks_dir(data_dir);

    if is_futuresharks_histdata_dir(&canonical_dir) {
        let mut loader = FutureSharksHistdataLoader::new();
        loader.load_dir(&canonical_dir, num_tickers)?;
        Ok((loader.tickers().to_vec(), loader.bars().to_vec()))
    } else {
        let mut loader = CsvDataLoader::new(&canonical_dir);
        loader.load(num_tickers, lookback_window + 50)?;
        let tickers_with_beta = calculate_betas(loader.tickers(), loader.bars());
        Ok((tickers_with_beta, loader.bars().to_vec()))
    }
}

pub fn aggregate_bars(bars: &[StockBar], timeframe_minutes: usize) -> Vec<StockBar> {
    if timeframe_minutes <= 1 {
        return bars.to_vec();
    }

    bars.chunks(timeframe_minutes)
        .filter(|chunk| !chunk.is_empty())
        .enumerate()
        .map(|(idx, chunk)| {
            let open = chunk.first().unwrap().open;
            let close = chunk.last().unwrap().close;
            let high = chunk.iter().map(|b| b.high).fold(f32::MIN, f32::max);
            let low = chunk.iter().map(|b| b.low).fold(f32::MAX, f32::min);
            let volume: f64 = chunk.iter().map(|b| b.volume).sum();
            let returns = if open.abs() > 1e-9 {
                (close - open) / open
            } else {
                0.0
            };

            StockBar {
                ticker_id: chunk.first().unwrap().ticker_id,
                timestamp: chunk.last().unwrap().timestamp,
                day_index: idx,
                open,
                high,
                low,
                close,
                volume,
                returns,
            }
        })
        .collect()
}

fn normalize_futuresharks_dir(data_dir: &str) -> String {
    let p = std::path::Path::new(data_dir);
    if p.ends_with("financial-data") || p.ends_with("data/financial-data") {
        return DEFAULT_FUTURESHARKS_DATA_DIR.to_string();
    }
    data_dir.to_string()
}

fn is_futuresharks_histdata_dir(data_dir: &str) -> bool {
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

fn calculate_betas(tickers: &[TickerInfo], bars: &[StockBar]) -> Vec<TickerInfo> {
    let mut day_returns: HashMap<usize, Vec<f32>> = HashMap::new();

    for bar in bars {
        day_returns
            .entry(bar.day_index)
            .or_default()
            .push(bar.returns);
    }

    let mut market_returns: HashMap<usize, f32> = HashMap::new();
    for (day, returns) in &day_returns {
        let n = returns.len();
        if n > 0 {
            let avg = returns.iter().sum::<f32>() / n as f32;
            market_returns.insert(*day, avg);
        }
    }

    let mkt_values: Vec<f32> = market_returns.values().cloned().collect();
    let mkt_mean: f32 = mkt_values.iter().sum::<f32>() / mkt_values.len().max(1) as f32;
    let mkt_variance: f32 = mkt_values
        .iter()
        .map(|r| (r - mkt_mean).powi(2))
        .sum::<f32>()
        / mkt_values.len().max(1) as f32;

    let mut result = Vec::with_capacity(tickers.len());

    for ticker in tickers {
        let ticker_bars: Vec<&StockBar> = bars
            .iter()
            .filter(|b| b.ticker_id == ticker.ticker_id)
            .collect();

        if ticker_bars.is_empty() || mkt_variance < 1e-10 {
            result.push(ticker.clone());
            continue;
        }

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
            1.0
        };

        result.push(TickerInfo {
            ticker_id: ticker.ticker_id,
            name: ticker.name.clone(),
            symbol: ticker.symbol.clone(),
            sector: ticker.sector,
            beta: beta.clamp(0.0, 3.0),
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
