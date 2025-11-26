//! Privacy level handlers

use crate::error::{PrivacyError, Result};
use crate::models::*;
use crate::state::AppState;
use axum::{
    extract::{Extension, Path, Query},
    Json,
};
use serde::Deserialize;
use std::sync::Arc;

/// List all privacy levels
pub async fn list_levels() -> Json<Vec<PrivacyLevel>> {
    Json(get_all_privacy_levels())
}

/// Get details for a specific privacy level
pub async fn get_level_details(Path(level): Path<i16>) -> Result<Json<PrivacyLevel>> {
    let levels = get_all_privacy_levels();

    levels
        .into_iter()
        .find(|l| l.level == level)
        .map(Json)
        .ok_or_else(|| PrivacyError::InvalidLevel(level))
}

/// Recommend a privacy level
pub async fn recommend_level(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<RecommendationRequest>,
) -> Result<Json<RecommendationResponse>> {
    if !state.config.privacy.enable_recommendations {
        return Ok(Json(RecommendationResponse {
            recommended_level: state.config.privacy.default_level,
            confidence: 1.0,
            factors: vec![RecommendationFactor {
                name: "default_policy".to_string(),
                impact: "high".to_string(),
                weight: 1.0,
                reason: "Recommendations disabled, using default level".to_string(),
            }],
            alternatives: vec![],
            cost_analysis: calculate_cost_analysis(&req.chain_id),
        }));
    }

    // Calculate recommendation based on various factors
    let mut factors: Vec<RecommendationFactor> = Vec::new();
    let mut score = 0.0;

    // Factor 1: Amount-based scoring
    let amount_score = calculate_amount_score(req.amount);
    factors.push(RecommendationFactor {
        name: "transaction_amount".to_string(),
        impact: if amount_score > 4.0 { "high" } else { "medium" },
        weight: 0.35,
        reason: format!(
            "Amount {} suggests privacy level {}",
            req.amount,
            amount_score.round() as i16
        ),
    });
    score += amount_score * 0.35;

    // Factor 2: User preference
    if let Some(pref) = req.privacy_preference {
        let pref_score = (pref as f64 / 10.0) * 5.0;
        factors.push(RecommendationFactor {
            name: "user_preference".to_string(),
            impact: "high".to_string(),
            weight: 0.30,
            reason: format!("User preference {} suggests level {}", pref, pref_score.round() as i16),
        });
        score += pref_score * 0.30;
    } else {
        score += 4.0 * 0.30; // Default to level 4 weight
    }

    // Factor 3: Chain-specific considerations
    let chain_score = calculate_chain_score(&req.chain_id);
    factors.push(RecommendationFactor {
        name: "chain_characteristics".to_string(),
        impact: "medium".to_string(),
        weight: 0.20,
        reason: format!(
            "Chain {} has {} privacy support",
            req.chain_id,
            if chain_score > 3.0 { "good" } else { "limited" }
        ),
    });
    score += chain_score * 0.20;

    // Factor 4: Transaction type
    let type_score = req
        .transaction_type
        .as_ref()
        .map(|t| calculate_type_score(t))
        .unwrap_or(4.0);
    factors.push(RecommendationFactor {
        name: "transaction_type".to_string(),
        impact: "low".to_string(),
        weight: 0.15,
        reason: format!(
            "Transaction type suggests level {}",
            type_score.round() as i16
        ),
    });
    score += type_score * 0.15;

    let recommended_level = score.round().clamp(0.0, 5.0) as i16;
    let confidence = calculate_confidence(&factors);

    // Generate alternatives
    let alternatives = generate_alternatives(recommended_level, &factors);

    Ok(Json(RecommendationResponse {
        recommended_level,
        confidence,
        factors,
        alternatives,
        cost_analysis: calculate_cost_analysis(&req.chain_id),
    }))
}

/// Validate a privacy level for a transaction
pub async fn validate_level(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<ValidationRequest>,
) -> Result<Json<ValidationResponse>> {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Validate level range
    if req.privacy_level < 0 || req.privacy_level > 5 {
        errors.push(format!(
            "Privacy level {} is out of range (0-5)",
            req.privacy_level
        ));
        return Ok(Json(ValidationResponse {
            valid: false,
            errors,
            warnings,
            suggested_level: Some(state.config.privacy.default_level),
        }));
    }

    // Chain-specific validations
    match req.chain_id.to_lowercase().as_str() {
        "bitcoin" if req.privacy_level > 3 => {
            errors.push("Bitcoin does not support privacy levels > 3".to_string());
        }
        "ethereum" if req.privacy_level == 5 && req.amount > 1_000_000_000_000 => {
            warnings.push("Level 5 with high amounts may have elevated gas costs".to_string());
        }
        _ => {}
    }

    // Amount-based warnings
    if req.amount > 100_000_000_000_000 && req.privacy_level < 4 {
        warnings.push(
            "Large transaction amounts typically benefit from higher privacy".to_string(),
        );
    }

    if req.privacy_level == 0 {
        warnings.push("Level 0 provides no privacy protection".to_string());
    }

    let valid = errors.is_empty();
    let suggested_level = if !valid {
        Some(state.config.privacy.default_level)
    } else {
        None
    };

    Ok(Json(ValidationResponse {
        valid,
        errors,
        warnings,
        suggested_level,
    }))
}

/// Cost query parameters
#[derive(Debug, Deserialize)]
pub struct CostQuery {
    pub level: i16,
    pub chain_id: String,
}

/// Calculate privacy cost
pub async fn calculate_cost(Query(query): Query<CostQuery>) -> Result<Json<LevelCost>> {
    if query.level < 0 || query.level > 5 {
        return Err(PrivacyError::InvalidLevel(query.level));
    }

    let base_gas: u64 = 21000;
    let proof_gas: u64 = match query.level {
        0 => 0,
        1 | 2 => 50000,
        3 => 75000,
        4 => 150000,
        5 => 250000,
        _ => 100000,
    };

    // Chain-specific gas price multiplier (approximate)
    let gas_price_gwei = match query.chain_id.to_lowercase().as_str() {
        "ethereum" => 30.0,
        "polygon" => 50.0,
        "arbitrum" => 0.1,
        "optimism" => 0.01,
        "base" => 0.01,
        _ => 20.0,
    };

    let total_gas = base_gas + proof_gas;
    let fee_eth = (total_gas as f64) * gas_price_gwei * 1e-9;
    let eth_price_usd = 2000.0; // Placeholder
    let fee_usd = fee_eth * eth_price_usd;

    let proof_time_ms = match query.level {
        0 => 0,
        1 | 2 => 500,
        3 => 1000,
        4 => 2500,
        5 => 5000,
        _ => 1500,
    };

    Ok(Json(LevelCost {
        level: query.level,
        gas: total_gas,
        fee_usd,
        proof_time_ms,
    }))
}

// Helper functions

fn calculate_amount_score(amount: i64) -> f64 {
    if amount > 1_000_000_000_000_000 {
        5.0 // Maximum for very large amounts
    } else if amount > 100_000_000_000_000 {
        4.5
    } else if amount > 10_000_000_000_000 {
        4.0
    } else if amount > 1_000_000_000_000 {
        3.5
    } else if amount > 100_000_000_000 {
        3.0
    } else {
        4.0 // Default to full privacy for normal amounts
    }
}

fn calculate_chain_score(chain_id: &str) -> f64 {
    match chain_id.to_lowercase().as_str() {
        "ethereum" | "polygon" => 5.0,
        "arbitrum" | "optimism" | "base" => 4.5,
        "bsc" | "avalanche" => 4.0,
        "bitcoin" => 2.0, // Limited ZK support
        _ => 3.5,
    }
}

fn calculate_type_score(tx_type: &str) -> f64 {
    match tx_type.to_lowercase().as_str() {
        "salary" | "payment" => 4.0,
        "donation" => 3.0,
        "exchange" | "swap" => 2.0,
        "investment" => 4.5,
        "personal" => 5.0,
        _ => 4.0,
    }
}

fn calculate_confidence(factors: &[RecommendationFactor]) -> f64 {
    // Higher confidence when factors agree
    let total_weight: f64 = factors.iter().map(|f| f.weight).sum();
    let normalized = if total_weight > 0.0 {
        0.7 + (total_weight * 0.3)
    } else {
        0.5
    };
    normalized.clamp(0.0, 1.0)
}

fn generate_alternatives(
    recommended: i16,
    _factors: &[RecommendationFactor],
) -> Vec<AlternativeRecommendation> {
    let mut alternatives = Vec::new();

    if recommended > 0 {
        alternatives.push(AlternativeRecommendation {
            level: recommended - 1,
            reason: "Lower cost, slightly reduced privacy".to_string(),
            score: 0.7,
        });
    }

    if recommended < 5 {
        alternatives.push(AlternativeRecommendation {
            level: recommended + 1,
            reason: "Enhanced privacy, slightly higher cost".to_string(),
            score: 0.8,
        });
    }

    if recommended != 4 {
        alternatives.push(AlternativeRecommendation {
            level: 4,
            reason: "Full privacy - recommended for most transactions".to_string(),
            score: 0.85,
        });
    }

    alternatives
}

fn calculate_cost_analysis(chain_id: &str) -> CostAnalysis {
    let levels: Vec<LevelCost> = (0..=5)
        .map(|level| {
            let base_gas: u64 = 21000;
            let proof_gas: u64 = match level {
                0 => 0,
                1 | 2 => 50000,
                3 => 75000,
                4 => 150000,
                5 => 250000,
                _ => 100000,
            };

            let gas_price_gwei = match chain_id.to_lowercase().as_str() {
                "ethereum" => 30.0,
                "polygon" => 50.0,
                "arbitrum" => 0.1,
                _ => 20.0,
            };

            let total_gas = base_gas + proof_gas;
            let fee_usd = (total_gas as f64) * gas_price_gwei * 1e-9 * 2000.0;

            LevelCost {
                level,
                gas: total_gas,
                fee_usd,
                proof_time_ms: match level {
                    0 => 0,
                    1 | 2 => 500,
                    3 => 1000,
                    4 => 2500,
                    5 => 5000,
                    _ => 1500,
                },
            }
        })
        .collect();

    CostAnalysis {
        by_level: levels,
        optimal_balance: 4, // Level 4 is usually the best balance
    }
}
