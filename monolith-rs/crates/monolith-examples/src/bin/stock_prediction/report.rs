use super::data::TickerInfo;
use super::instances::{FeatureIndex, StockInstance};
use super::model::StockPredictionModel;

#[derive(Debug, Clone)]
pub struct StockRecommendation {
    pub ticker: String,
    pub sector: String,
    pub predicted_direction: String,
    pub direction_confidence: f32,
    #[allow(dead_code)]
    pub predicted_return: f32,
    pub profitable_probability: f32,
    pub risk_score: f32,
    pub recommendation: String,
    pub expected_value: f32,
}

pub fn generate_recommendations(
    model: &StockPredictionModel,
    instances: &[StockInstance],
    features: &FeatureIndex,
    tickers: &[TickerInfo],
) -> Vec<StockRecommendation> {
    let batch: Vec<&StockInstance> = instances.iter().collect();
    let output = model.forward(&batch, features);

    let mut recommendations = Vec::new();

    for (i, instance) in instances.iter().enumerate() {
        let ticker = tickers
            .iter()
            .find(|t| t.ticker_id == instance.ticker_fid)
            .unwrap();

        let direction_conf = output.direction[i];
        let predicted_return = output.magnitude[i];
        let profitable_prob = output.profitable[i];

        let (direction, effective_conf) = if direction_conf > 0.5 {
            ("UP", direction_conf)
        } else {
            ("DOWN", 1.0 - direction_conf)
        };

        let indicators = features.indicator_at(instance, instance.t, 0);
        let bar = features.bar_at(instance, instance.t);
        let close = bar.close.max(1e-6);
        let atr_normalized = indicators.map(|ind| ind.atr_14 / close).unwrap_or(0.0);
        let bb_width = indicators.map(|ind| ind.bollinger_width).unwrap_or(0.0);
        let risk_score = ((atr_normalized.abs() + bb_width.abs()) / 2.0).clamp(0.0, 1.0);

        let recommendation = if direction == "UP" {
            if effective_conf > 0.75 && predicted_return > 2.0 {
                "STRONG BUY"
            } else if effective_conf > 0.6 && predicted_return > 1.0 {
                "BUY"
            } else {
                "HOLD"
            }
        } else if effective_conf > 0.75 && predicted_return < -2.0 {
            "STRONG SELL"
        } else if effective_conf > 0.6 && predicted_return < -1.0 {
            "SELL"
        } else {
            "HOLD"
        };

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

    recommendations.sort_by(|a, b| b.expected_value.partial_cmp(&a.expected_value).unwrap());

    recommendations
}

pub fn print_recommendation_report(recommendations: &[StockRecommendation], top_n: usize) {
    println!("\n  Top {} Stock Recommendations:", top_n);
    println!("  ┌────────┬────────────┬───────────┬────────────┬────────────┬────────┬─────────────┬────────────────┐");
    println!("  │ Ticker │ Sector     │ Direction │ Confidence │ Profit Prob│ Risk % │ Exp Return  │ Recommendation │");
    println!("  ├────────┼────────────┼───────────┼────────────┼────────────┼────────┼─────────────┼────────────────┤");

    for rec in recommendations.iter().take(top_n) {
        println!(
            "  │ {:6} │ {:10} │ {:9} │ {:>9.0}% │ {:>10.0}% │ {:>6.0}% │ {:>+10.2} │ {:14} │",
            rec.ticker,
            rec.sector,
            rec.predicted_direction,
            rec.direction_confidence * 100.0,
            rec.profitable_probability * 100.0,
            rec.risk_score * 100.0,
            rec.expected_value,
            rec.recommendation
        );
    }

    println!("  └────────┴────────────┴───────────┴────────────┴────────────┴────────┴─────────────┴────────────────┘");
}
