use super::instances::{FeatureIndex, StockInstance};
use super::model::StockPredictionModel;

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

pub struct Backtester {
    initial_capital: f64,
    transaction_cost: f32,
    position_size: f32,
}

impl Backtester {
    pub fn new() -> Self {
        Self {
            initial_capital: 100_000.0,
            transaction_cost: 0.001,
            position_size: 0.1,
        }
    }

    pub fn run(
        &self,
        model: &StockPredictionModel,
        instances: &[StockInstance],
        features: &FeatureIndex,
    ) -> FinancialMetrics {
        // Backtests must be chronological; otherwise metrics are meaningless.
        let mut ordered: Vec<&StockInstance> = instances.iter().collect();
        ordered.sort_by_key(|i| (i.t, i.ticker_idx));

        let batches: Vec<Vec<&StockInstance>> = ordered.chunks(32).map(|c| c.to_vec()).collect();

        let mut capital = self.initial_capital;
        let mut peak_capital = capital;
        let mut max_drawdown = 0.0f64;

        let mut returns: Vec<f64> = Vec::new();
        let mut wins = 0;
        let mut losses = 0;
        let mut total_wins = 0.0f64;
        let mut total_losses = 0.0f64;

        for batch in &batches {
            let output = model.forward(batch, features);

            for (i, instance) in batch.iter().enumerate() {
                let direction_pred = output.direction[i];
                let magnitude_pred = output.magnitude[i];
                let profitable_pred = output.profitable[i];

                // Confidence in [0,1] where 0.0 ~ neutral, 1.0 ~ very confident
                let edge = ((direction_pred - 0.5).abs() * 2.0).clamp(0.0, 1.0);

                // Risk estimate from indicators (0..1)
                let indicators = features.indicator_at(instance, instance.t, 0);
                let risk_score = indicators
                    .map(|ind| {
                        // atr_14 is already normalized by close in our indicators implementation
                        (ind.atr_14.abs() + ind.bollinger_width.abs()) * 0.5
                    })
                    .unwrap_or(0.0)
                    .clamp(0.0, 1.0);

                // Only trade when direction is confident AND "profitable" head agrees
                if edge > 0.2 && profitable_pred > 0.55 && risk_score < 0.85 {
                    // Size down in high risk regimes; size up with confidence.
                    let risk_mult = (1.0 - risk_score).clamp(0.1, 1.0);
                    let magnitude_multiplier = (magnitude_pred.abs() / 2.0).clamp(0.5, 2.0);
                    let position_value = capital
                        * self.position_size as f64
                        * edge as f64
                        * risk_mult as f64
                        * magnitude_multiplier as f64;
                    let is_long = direction_pred > 0.5;

                    let actual_return = instance.magnitude_label / 100.0;

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
            calmar_ratio,
            win_rate,
            profit_factor,
            num_trades,
            avg_win,
            avg_loss,
        }
    }
}
