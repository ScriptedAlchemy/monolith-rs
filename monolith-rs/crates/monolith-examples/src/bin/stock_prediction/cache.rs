//! Indicator caching for stock prediction.
//!
//! Caches computed technical indicators to disk to avoid recomputation on subsequent runs
//! with the same data configuration.

use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::config::StockPredictorConfig;
use super::data::StockBar;
use super::indicators::TechnicalIndicators;

const CACHE_VERSION: u32 = 1;
const CACHE_DIR: &str = "target/stock_prediction_cache";

/// Cache key that uniquely identifies a dataset configuration.
#[derive(Debug, Clone, Hash, Serialize, Deserialize)]
struct CacheKey {
    version: u32,
    data_dir: String,
    intraday_file: Option<String>,
    num_tickers: usize,
    bar_stride: usize,
    max_bars_per_ticker: usize,
    timeframes: Vec<usize>,
    data_hash: u64,
}

/// Cached indicator data.
#[derive(Serialize, Deserialize)]
struct CachedIndicators {
    key: CacheKey,
    /// indicators_by_ticker[ticker_idx][timeframe_idx][bar_idx]
    indicators: Vec<Vec<Vec<TechnicalIndicators>>>,
}

/// Computes a hash of the raw bar data to detect changes.
fn compute_data_hash(bars_by_ticker: &[Vec<StockBar>]) -> u64 {
    let mut hasher = DefaultHasher::new();

    // Hash number of tickers
    bars_by_ticker.len().hash(&mut hasher);

    // For each ticker, hash the number of bars and some sample data
    for ticker_bars in bars_by_ticker {
        ticker_bars.len().hash(&mut hasher);

        // Hash first, middle, and last bars for quick change detection
        if !ticker_bars.is_empty() {
            hash_bar(&ticker_bars[0], &mut hasher);
            hash_bar(&ticker_bars[ticker_bars.len() / 2], &mut hasher);
            hash_bar(&ticker_bars[ticker_bars.len() - 1], &mut hasher);
        }
    }

    hasher.finish()
}

fn hash_bar(bar: &StockBar, hasher: &mut impl Hasher) {
    bar.ticker_id.hash(hasher);
    bar.timestamp.hash(hasher);
    bar.open.to_bits().hash(hasher);
    bar.close.to_bits().hash(hasher);
    bar.high.to_bits().hash(hasher);
    bar.low.to_bits().hash(hasher);
}

fn cache_key_hash(key: &CacheKey) -> u64 {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    hasher.finish()
}

fn cache_path(key: &CacheKey) -> PathBuf {
    let hash = cache_key_hash(key);
    Path::new(CACHE_DIR).join(format!("indicators_{:016x}.bin", hash))
}

/// Creates a cache key from the current configuration and data.
fn create_cache_key(config: &StockPredictorConfig, bars_by_ticker: &[Vec<StockBar>]) -> CacheKey {
    CacheKey {
        version: CACHE_VERSION,
        data_dir: config.data_dir.clone().unwrap_or_default(),
        intraday_file: config.intraday_file.clone(),
        num_tickers: config.num_tickers,
        bar_stride: config.bar_stride,
        max_bars_per_ticker: config.max_bars_per_ticker,
        timeframes: config.timeframes.clone(),
        data_hash: compute_data_hash(bars_by_ticker),
    }
}

/// Attempts to load cached indicators.
/// Returns None if cache doesn't exist or is invalid.
pub fn load_cached_indicators(
    config: &StockPredictorConfig,
    bars_by_ticker: &[Vec<StockBar>],
) -> Option<Vec<Vec<Vec<TechnicalIndicators>>>> {
    let key = create_cache_key(config, bars_by_ticker);
    let path = cache_path(&key);

    if !path.exists() {
        return None;
    }

    let file = File::open(&path).ok()?;
    let reader = BufReader::new(file);

    let cached: CachedIndicators = bincode::deserialize_from(reader).ok()?;

    // Verify the key matches
    if cached.key.version != key.version
        || cached.key.data_hash != key.data_hash
        || cached.key.timeframes != key.timeframes
        || cached.key.bar_stride != key.bar_stride
    {
        // Cache is stale, remove it
        let _ = fs::remove_file(&path);
        return None;
    }

    // Verify dimensions match
    if cached.indicators.len() != bars_by_ticker.len() {
        let _ = fs::remove_file(&path);
        return None;
    }

    Some(cached.indicators)
}

/// Saves computed indicators to cache.
pub fn save_indicators_to_cache(
    config: &StockPredictorConfig,
    bars_by_ticker: &[Vec<StockBar>],
    indicators: &[Vec<Vec<TechnicalIndicators>>],
) -> Result<PathBuf, String> {
    let key = create_cache_key(config, bars_by_ticker);
    let path = cache_path(&key);

    // Ensure cache directory exists
    fs::create_dir_all(CACHE_DIR)
        .map_err(|e| format!("Failed to create cache directory: {}", e))?;

    let cached = CachedIndicators {
        key,
        indicators: indicators.to_vec(),
    };

    let file = File::create(&path)
        .map_err(|e| format!("Failed to create cache file: {}", e))?;
    let writer = BufWriter::new(file);

    bincode::serialize_into(writer, &cached)
        .map_err(|e| format!("Failed to serialize indicators: {}", e))?;

    Ok(path)
}

/// Clears all cached indicator files.
#[allow(dead_code)]
pub fn clear_cache() -> Result<usize, String> {
    let cache_dir = Path::new(CACHE_DIR);
    if !cache_dir.exists() {
        return Ok(0);
    }

    let mut count = 0;
    for entry in fs::read_dir(cache_dir).map_err(|e| format!("Failed to read cache dir: {}", e))? {
        if let Ok(entry) = entry {
            let path = entry.path();
            if path.extension().map(|e| e == "bin").unwrap_or(false) {
                if fs::remove_file(&path).is_ok() {
                    count += 1;
                }
            }
        }
    }

    Ok(count)
}

/// Returns the size of the cache directory in bytes.
#[allow(dead_code)]
pub fn cache_size() -> u64 {
    let cache_dir = Path::new(CACHE_DIR);
    if !cache_dir.exists() {
        return 0;
    }

    let mut size = 0;
    if let Ok(entries) = fs::read_dir(cache_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                size += metadata.len();
            }
        }
    }

    size
}
