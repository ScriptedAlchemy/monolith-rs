use std::collections::HashMap;

use super::config::DEFAULT_STOCK_PREDICTION_DATA_DIR;

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

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.uniform() * (i + 1) as f32) as usize % (i + 1);
            slice.swap(i, j);
        }
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
    pub fn print_download_instructions() {
        println!("\n=== Intraday Data Download Instructions ===\n");
        println!(
            "This example expects intraday CSVs in:\n  {}\n",
            DEFAULT_STOCK_PREDICTION_DATA_DIR
        );
        println!("Supported CSV format (no header):");
        println!("  SYMBOL,YYYY-MM-DD,HH:MM:SS,OPEN,HIGH,LOW,CLOSE,VOLUME\n");
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
            let line = line.map_err(|e| format!("Read error: {}", e))?;
            let fields: Vec<&str> = line.split(',').collect();

            // The supported intraday formats in this repo:
            // - Flat 6-column: `ts,open,high,low,close,volume` with a header row
            // - This dataset (no header): `symbol,date,time,open,high,low,close,volume`
            //
            // For intraday aggregation, we only need OHLCV, so we ignore symbol/date/time.

            // Skip obvious header rows.
            if i == 0 {
                let lower = line.to_lowercase();
                if lower.contains("open") && lower.contains("close") {
                    continue;
                }
            }

            if fields.len() < 6 {
                continue;
            }

            let (open_idx, high_idx, low_idx, close_idx, vol_idx) = match fields.len() {
                6 => (1, 2, 3, 4, 5),
                // symbol,date,time,open,high,low,close,volume
                n if n >= 8 => (3, 4, 5, 6, 7),
                _ => continue,
            };

            let open: f32 = fields[open_idx].parse().unwrap_or(0.0);
            let high: f32 = fields[high_idx].parse().unwrap_or(0.0);
            let low: f32 = fields[low_idx].parse().unwrap_or(0.0);
            let close: f32 = fields[close_idx].parse().unwrap_or(0.0);
            let volume: f64 = fields[vol_idx].parse().unwrap_or(0.0);

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

    /// Loads a directory of intraday CSVs.
    ///
    /// Supports:
    /// - "flat" layout: `<dir>/<TICKER>.csv`
    /// - nested layout (like the user's dataset): `<dir>/**/<TICKER>.csv`
    ///
    /// CSV format expected (no header):
    /// `SYMBOL,YYYY-MM-DD,HH:MM:SS,OPEN,HIGH,LOW,CLOSE,VOLUME`
    pub fn load_dir_recursive(
        &mut self,
        dir: &str,
        max_tickers: usize,
        aggregate_minutes: usize,
    ) -> Result<(), String> {
        use std::collections::HashSet;
        use std::fs;
        use std::path::Path;

        let root = Path::new(dir);
        if !root.exists() {
            return Err(format!("Directory does not exist: {}", dir));
        }

        let mut csv_paths: Vec<String> = Vec::new();
        let mut seen: HashSet<String> = HashSet::new();

        // DFS over the directory tree collecting .csv files.
        let mut stack: Vec<std::path::PathBuf> = vec![root.to_path_buf()];
        while let Some(p) = stack.pop() {
            let rd = match fs::read_dir(&p) {
                Ok(rd) => rd,
                Err(_) => continue,
            };

            for entry in rd.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if path
                    .extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("csv"))
                    .unwrap_or(false)
                {
                    // De-dup by ticker symbol inferred from filename.
                    let sym = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("")
                        .to_uppercase();
                    if sym.is_empty() {
                        continue;
                    }
                    if seen.insert(sym) {
                        csv_paths.push(path.to_string_lossy().to_string());
                    }
                }
            }
        }

        csv_paths.sort();
        csv_paths.truncate(max_tickers);

        if csv_paths.is_empty() {
            return Err(format!("No CSV files found under {}", dir));
        }

        println!(
            "  Loading {} intraday CSV files from {} (recursive)",
            csv_paths.len(),
            dir
        );

        for path in csv_paths {
            let sym = std::path::Path::new(&path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("INTRADAY")
                .to_uppercase();
            self.load_file(&path, &sym, aggregate_minutes)?;
        }

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

// NOTE: This example is focused on local intraday CSV datasets under
// `examples/stock_prediction/data/`.

pub fn ensure_dataset_present(data_dir: Option<&str>) {
    use std::path::Path;

    let Some(data_dir) = data_dir else {
        return;
    };

    if Path::new(data_dir).exists() {
        return;
    }
    eprintln!(
        "Dataset directory not found: `{}`. Put intraday CSVs under `{}` (or pass --data-dir).",
        data_dir, DEFAULT_STOCK_PREDICTION_DATA_DIR
    );
}

pub fn load_real_data_auto(
    data_dir: &str,
    num_tickers: usize,
    lookback_window: usize,
) -> Result<(Vec<TickerInfo>, Vec<StockBar>), String> {
    let canonical_dir = data_dir.to_string();

    // Prefer intraday recursive loader for the local dataset layout.
    let mut intraday = IntradayCsvLoader::new();
    intraday.load_dir_recursive(&canonical_dir, num_tickers, 1)?;
    let tickers = intraday.tickers().to_vec();
    let bars = intraday.bars().to_vec();

    // Enforce a minimum bar count similar to the daily loader's lookback requirements.
    let min_required = lookback_window + 50;
    let mut bars_by_tid: HashMap<i64, usize> = HashMap::new();
    for b in &bars {
        *bars_by_tid.entry(b.ticker_id).or_insert(0) += 1;
    }
    let eligible: std::collections::HashSet<i64> = bars_by_tid
        .iter()
        .filter_map(|(&tid, &n)| if n >= min_required { Some(tid) } else { None })
        .collect();

    let mut filtered_tickers: Vec<TickerInfo> = Vec::new();
    let mut id_remap: HashMap<i64, i64> = HashMap::new();
    for t in &tickers {
        if eligible.contains(&t.ticker_id) {
            let new_id = filtered_tickers.len() as i64;
            id_remap.insert(t.ticker_id, new_id);
            let mut nt = t.clone();
            nt.ticker_id = new_id;
            filtered_tickers.push(nt);
        }
    }

    let mut filtered_bars: Vec<StockBar> = Vec::new();
    for b in bars {
        if let Some(&new_id) = id_remap.get(&b.ticker_id) {
            let mut nb = b;
            nb.ticker_id = new_id;
            filtered_bars.push(nb);
        }
    }

    if filtered_tickers.is_empty() {
        return Err(format!(
            "No tickers had at least {} bars under {}",
            min_required, canonical_dir
        ));
    }

    let tickers_with_beta = calculate_betas(&filtered_tickers, &filtered_bars);
    Ok((tickers_with_beta, filtered_bars))
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

// NOTE: We default to local intraday CSVs placed under `examples/stock_prediction/data/`.

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
