use std::time::Instant;

use rayon::prelude::*;

use super::backtest::Backtester;
use super::cache::{load_cached_indicators, save_indicators_to_cache};
use super::config::{parse_args, print_usage, Mode, StockPredictorConfig};
use super::data::IntradayCsvLoader;
use super::data::{
    aggregate_bars, ensure_dataset_present, load_real_data_auto, StockBar, TickerInfo,
};
use super::indicators::{IndicatorCalculator, TechnicalIndicators};
use super::instances::{
    create_batches, create_instances_parallel, train_eval_split_time_by_ticker, FeatureIndex,
    InstanceCreator, StockInstance,
};
use super::model::StockPredictionModel;
use super::report::{generate_recommendations, print_recommendation_report};
use super::trainer::Trainer;
use monolith_layers::tensor::set_gpu_enabled;

pub fn run() {
    let config = parse_args();
    set_gpu_enabled(config.gpu_mode);

    println!("{}", "=".repeat(80));
    println!("Monolith-RS Stock Prediction Example");
    println!("{}", "=".repeat(80));
    println!();

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
    println!(
        "Features: timeframes={} | lookbacks={} (seq={}) | stride={} | max-bars={}",
        config
            .timeframes
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
        config
            .lookbacks
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(","),
        config.lookback_window,
        config.bar_stride,
        config.max_bars_per_ticker
    );
    println!();

    let start_time = Instant::now();

    let (tickers, bars): (Vec<TickerInfo>, Vec<StockBar>) =
        if let Some(ref path) = config.intraday_file {
            println!("Section 1: Loading Intraday CSV");
            let mut loader = IntradayCsvLoader::new();
            let ticker_name = config.intraday_ticker.as_deref().unwrap_or("INTRADAY");
            match loader.load_file(path, ticker_name, config.intraday_aggregate.max(1)) {
                Ok(()) => (loader.tickers().to_vec(), loader.bars().to_vec()),
                Err(e) => {
                    eprintln!("Error loading intraday file: {}", e);
                    std::process::exit(2);
                }
            }
        } else if let Some(ref data_dir) = config.data_dir {
            ensure_dataset_present(Some(data_dir));
            println!("Section 1: Loading Real Stock Data from CSV");
            match load_real_data_auto(data_dir, config.num_tickers, config.lookback_window) {
                Ok((tickers, bars)) => (tickers, bars),
                Err(e) => {
                    eprintln!("Error loading data: {}", e);
                    eprintln!();
                    super::data::IntradayDataSources::print_download_instructions();
                    std::process::exit(2);
                }
            }
        } else {
            eprintln!("Error: Provide --data-dir or --intraday-file.");
            eprintln!();
            print_usage();
            std::process::exit(2);
        };

    let total_bars = bars.len();
    println!(
        "  Tickers: {} | Days: {} | Total bars: {}",
        tickers.len(),
        config.days_of_history,
        total_bars
    );

    let mut bars_by_ticker: Vec<Vec<StockBar>> = vec![Vec::new(); tickers.len()];
    let ticker_index: std::collections::HashMap<i64, usize> = tickers
        .iter()
        .enumerate()
        .map(|(i, t)| (t.ticker_id, i))
        .collect();
    for bar in bars {
        if let Some(&idx) = ticker_index.get(&bar.ticker_id) {
            bars_by_ticker[idx].push(bar);
        }
    }
    for ticker_bars in &mut bars_by_ticker {
        apply_stride_and_limit(
            ticker_bars,
            config.bar_stride.max(1),
            config.max_bars_per_ticker,
        );
    }
    let total_bars_after: usize = bars_by_ticker.iter().map(|b| b.len()).sum();
    println!(
        "  Bars after stride/limit: {} (stride {}, max {})",
        total_bars_after, config.bar_stride, config.max_bars_per_ticker
    );

    println!("\nSection 2: Computing Technical Indicators (parallel)");
    let stride = config.bar_stride.max(1);
    let timeframes_minutes = normalize_timeframes(&config.timeframes);
    let timeframes_bars = normalize_timeframes(
        &timeframes_minutes
            .iter()
            .map(|&tf| tf.div_ceil(stride).max(1))
            .collect::<Vec<_>>(),
    );

    // Try to load from cache first
    let (indicators_by_ticker, from_cache) =
        if let Some(cached) = load_cached_indicators(&config, &bars_by_ticker) {
            println!("  Loaded indicators from cache");
            (cached, true)
        } else {
            let indicators: Vec<Vec<Vec<TechnicalIndicators>>> = bars_by_ticker
                .par_iter()
                .map(|_ticker_bars| {
                    timeframes_bars
                        .iter()
                        .map(|&tf| {
                            let mut calc = IndicatorCalculator::new();
                            if tf == 1 {
                                calc.compute_indicators(_ticker_bars)
                            } else {
                                let aggregated = aggregate_bars(_ticker_bars, tf);
                                calc.compute_indicators(&aggregated)
                            }
                        })
                        .collect()
                })
                .collect();

            // Save to cache for next run
            match save_indicators_to_cache(&config, &bars_by_ticker, &indicators) {
                Ok(path) => println!("  Saved indicators to cache: {}", path.display()),
                Err(e) => eprintln!("  Warning: Failed to save cache: {}", e),
            }

            (indicators, false)
        };

    println!(
        "  {} indicators per bar x {} timeframes (requested {}m, effective {} bars, stride {}){}",
        TechnicalIndicators::NUM_FEATURES,
        timeframes_bars.len(),
        timeframes_minutes
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        timeframes_bars
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(","),
        stride,
        if from_cache { " [cached]" } else { "" }
    );

    let feature_index = FeatureIndex::new(
        tickers.clone(),
        bars_by_ticker,
        indicators_by_ticker,
        timeframes_bars,
    );

    println!("\nSection 3: Creating Training Instances");
    let instance_creator = InstanceCreator::new(config.lookback_window);
    let all_instances = create_instances_parallel(&feature_index, &instance_creator);

    let (train_instances, eval_instances) =
        train_eval_split_time_by_ticker(&all_instances, config.train_ratio);

    // Show coverage by ticker so it's obvious we're not accidentally evaluating on unseen tickers.
    let mut train_tickers = std::collections::HashSet::new();
    let mut eval_tickers = std::collections::HashSet::new();
    for i in &train_instances {
        train_tickers.insert(i.ticker_idx);
    }
    for i in &eval_instances {
        eval_tickers.insert(i.ticker_idx);
    }
    let overlap = train_tickers.intersection(&eval_tickers).count();

    println!(
        "  Train: {} ({:.0}%) | Eval: {} ({:.0}%) | Lookback: {} bars | Tickers train/eval/overlap: {}/{}/{}",
        train_instances.len(),
        config.train_ratio * 100.0,
        eval_instances.len(),
        (1.0 - config.train_ratio) * 100.0,
        config.lookback_window,
        train_tickers.len(),
        eval_tickers.len(),
        overlap
    );

    let seq_feature_dim = 4 + feature_index.indicator_dim();
    let dien_hidden = if config.dien_hidden_size == 0 {
        seq_feature_dim
    } else {
        config.dien_hidden_size
    };

    println!("\nSection 4: Model Architecture");
    println!(
        "  Ticker embedding: {} x {} | DIEN hidden: {} | DCN: {} layers",
        config.num_tickers, config.ticker_embedding_dim, dien_hidden, config.dcn_cross_layers
    );

    match config.mode {
        Mode::Train => {
            run_training(
                &config,
                &train_instances,
                &eval_instances,
                &feature_index,
                &tickers,
            );
        }
        Mode::Evaluate => {
            run_evaluation(&config, &eval_instances, &feature_index);
        }
        Mode::Predict => {
            run_prediction(&config, &all_instances, &feature_index, &tickers);
        }
        Mode::Backtest => {
            run_backtesting(&config, &eval_instances, &feature_index);
        }
    }

    let elapsed = start_time.elapsed();
    println!();
    println!("{}", "=".repeat(80));
    println!("Complete! Total time: {:.2}s", elapsed.as_secs_f64());
    println!("{}", "=".repeat(80));
}

fn normalize_timeframes(timeframes: &[usize]) -> Vec<usize> {
    let mut tfs: Vec<usize> = timeframes.iter().copied().filter(|&t| t > 0).collect();
    if !tfs.contains(&1) {
        tfs.push(1);
    }
    tfs.sort_unstable();
    tfs.dedup();
    tfs
}

fn apply_stride_and_limit(bars: &mut Vec<StockBar>, stride: usize, max_bars: usize) {
    if stride > 1 {
        *bars = bars.iter().step_by(stride).cloned().collect();
    }
    if max_bars > 0 && bars.len() > max_bars {
        let start = bars.len() - max_bars;
        *bars = bars[start..].to_vec();
    }
}

fn run_training(
    config: &StockPredictorConfig,
    train_instances: &[StockInstance],
    eval_instances: &[StockInstance],
    features: &FeatureIndex,
    tickers: &[TickerInfo],
) {
    println!("\nSection 5: Training (with momentum, LR decay, weight decay)");
    let mut trainer = Trainer::new(config, features.indicator_dim());

    for epoch in 0..config.num_epochs {
        if epoch > 0 {
            trainer.decay_lr();
        }

        trainer.reset_device_metrics();
        let train_metrics = trainer.train_epoch(train_instances, features);
        let eval_metrics = trainer.evaluate(eval_instances, features);

        let dm = trainer.device_metrics();
        println!(
            "  Epoch {}/{}: Loss: {:.4} | Eval: {:.4} | Acc: {:.1}% | LR: {:.6}",
            epoch + 1,
            config.num_epochs,
            train_metrics.total_loss,
            eval_metrics.total_loss,
            eval_metrics.direction_accuracy * 100.0,
            trainer.current_lr()
        );
        println!(
            "    Timing: Forward {:.1}s | Backward {:.1}s | Loss {:.1}s | Data {:.2}s | Total {:.1}s",
            dm.forward_time.as_secs_f32(),
            dm.backward_time.as_secs_f32(),
            dm.loss_time.as_secs_f32(),
            dm.data_prep_time.as_secs_f32(),
            dm.total_time.as_secs_f32()
        );
        println!(
            "    Throughput: {:.0} samples/sec | GPU activity: {:.1}%",
            dm.throughput(),
            dm.gpu_activity_pct()
        );

        if trainer.check_early_stopping(eval_metrics.total_loss) {
            println!("  Early stopping triggered at epoch {}", epoch + 1);
            break;
        }
    }

    println!("\nSection 6: Evaluation Results");
    let final_eval = trainer.evaluate(eval_instances, features);
    println!(
        "  Direction Accuracy: {:.1}%",
        final_eval.direction_accuracy * 100.0
    );
    println!(
        "  Losses - Direction: {:.4} | Magnitude: {:.4} | Profitable: {:.4}",
        final_eval.direction_loss, final_eval.magnitude_loss, final_eval.profitable_loss
    );

    println!("\nSection 7: Backtest Results");
    let backtester = Backtester::new();
    let financial_metrics = backtester.run(trainer.model(), eval_instances, features);

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

    println!("\nSection 8: Top Stock Recommendations");
    let mut latest_instances = Vec::new();
    for ticker in tickers {
        let mut last = None;
        for inst in train_instances.iter().chain(eval_instances.iter()) {
            if inst.ticker_fid == ticker.ticker_id {
                last = Some(inst.clone());
            }
        }
        if let Some(instance) = last {
            latest_instances.push(instance);
        }
    }

    let recommendations =
        generate_recommendations(trainer.model(), &latest_instances, features, tickers);
    print_recommendation_report(&recommendations, 10);
}

fn run_evaluation(
    config: &StockPredictorConfig,
    eval_instances: &[StockInstance],
    features: &FeatureIndex,
) {
    println!("\nRunning Evaluation Mode");
    let model = StockPredictionModel::new(config, features.indicator_dim());

    let batches = create_batches(eval_instances, config.batch_size);
    let mut total_correct = 0;
    let mut total_samples = 0;

    for batch in batches {
        let output = model.forward(&batch, features);
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
    features: &FeatureIndex,
    tickers: &[TickerInfo],
) {
    println!("\nRunning Prediction Mode");
    let model = StockPredictionModel::new(config, features.indicator_dim());

    let mut latest_instances = Vec::new();
    for ticker in tickers {
        let mut last = None;
        for inst in instances {
            if inst.ticker_fid == ticker.ticker_id {
                last = Some(inst.clone());
            }
        }
        if let Some(instance) = last {
            latest_instances.push(instance);
        }
    }

    let recommendations = generate_recommendations(&model, &latest_instances, features, tickers);
    print_recommendation_report(&recommendations, config.num_tickers.min(20));
}

fn run_backtesting(
    config: &StockPredictorConfig,
    instances: &[StockInstance],
    features: &FeatureIndex,
) {
    println!("\nRunning Backtest Mode");
    let model = StockPredictionModel::new(config, features.indicator_dim());
    let backtester = Backtester::new();

    let metrics = backtester.run(&model, instances, features);

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
