//! Risk assessment handlers

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::Result;
use crate::handlers::metrics::RISK_ASSESSMENTS_TOTAL;
use crate::models::*;
use crate::state::AppState;

/// Perform risk assessment
pub async fn assess_risk(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<RiskAssessmentRequest>,
) -> Result<(StatusCode, Json<RiskAssessment>)> {
    RISK_ASSESSMENTS_TOTAL.inc();
    
    // Calculate risk factors
    let factors = calculate_risk_factors(&request);
    
    // Calculate overall score
    let total_weight: f64 = factors.iter().map(|f| f.weight).sum();
    let weighted_score: f64 = factors.iter()
        .map(|f| f.score * f.weight)
        .sum();
    
    let risk_score = if total_weight > 0.0 {
        weighted_score / total_weight
    } else {
        0.0
    };
    
    let assessment = RiskAssessment {
        entity_id: request.entity_id,
        risk_score,
        risk_level: RiskLevel::from_score(risk_score),
        factors,
        recommendations: generate_risk_recommendations(risk_score),
        assessed_at: Utc::now(),
        valid_until: Utc::now() + chrono::Duration::days(90),
    };
    
    Ok((StatusCode::OK, Json(assessment)))
}

/// Get risk score for entity
pub async fn get_risk_score(
    State(_state): State<Arc<AppState>>,
    Path(entity_id): Path<Uuid>,
) -> Result<Json<RiskAssessment>> {
    // In production, this would query cached/stored assessments
    Ok(Json(RiskAssessment {
        entity_id,
        risk_score: 0.25,
        risk_level: RiskLevel::Medium,
        factors: vec![],
        recommendations: vec![],
        assessed_at: Utc::now(),
        valid_until: Utc::now() + chrono::Duration::days(90),
    }))
}

/// List available risk factors
pub async fn list_risk_factors() -> Json<Vec<RiskFactorDefinition>> {
    Json(vec![
        RiskFactorDefinition {
            name: "transaction_volume".to_string(),
            category: "behavioral".to_string(),
            description: "Transaction volume analysis".to_string(),
            weight: 0.15,
        },
        RiskFactorDefinition {
            name: "geographic_risk".to_string(),
            category: "geographic".to_string(),
            description: "Risk based on transaction geography".to_string(),
            weight: 0.20,
        },
        RiskFactorDefinition {
            name: "counterparty_risk".to_string(),
            category: "relationship".to_string(),
            description: "Risk from transaction counterparties".to_string(),
            weight: 0.20,
        },
        RiskFactorDefinition {
            name: "pattern_anomaly".to_string(),
            category: "behavioral".to_string(),
            description: "Unusual transaction patterns".to_string(),
            weight: 0.25,
        },
        RiskFactorDefinition {
            name: "kyc_status".to_string(),
            category: "identity".to_string(),
            description: "KYC verification status".to_string(),
            weight: 0.20,
        },
    ])
}

/// Get current risk thresholds
pub async fn get_thresholds(
    State(state): State<Arc<AppState>>,
) -> Json<RiskThresholds> {
    let engine = state.rule_engine.read().await;
    let thresholds = engine.get_thresholds("DEFAULT");
    Json(thresholds)
}

/// Update risk thresholds
pub async fn update_thresholds(
    State(state): State<Arc<AppState>>,
    Json(thresholds): Json<RiskThresholds>,
) -> Result<(StatusCode, Json<RiskThresholds>)> {
    let mut engine = state.rule_engine.write().await;
    engine.update_thresholds(thresholds.clone());
    Ok((StatusCode::OK, Json(thresholds)))
}

/// Risk factor definition (for listing)
#[derive(Debug, serde::Serialize)]
pub struct RiskFactorDefinition {
    pub name: String,
    pub category: String,
    pub description: String,
    pub weight: f64,
}

/// Calculate risk factors for entity
fn calculate_risk_factors(request: &RiskAssessmentRequest) -> Vec<RiskFactor> {
    let mut factors = Vec::new();
    
    // Geographic risk
    factors.push(RiskFactor {
        name: "geographic_risk".to_string(),
        category: "geographic".to_string(),
        weight: 0.20,
        score: calculate_geographic_risk(&request.jurisdiction),
        description: format!("Geographic risk for {}", request.jurisdiction),
    });
    
    // Entity type risk
    factors.push(RiskFactor {
        name: "entity_type_risk".to_string(),
        category: "identity".to_string(),
        weight: 0.15,
        score: calculate_entity_type_risk(&request.entity_type),
        description: format!("Risk based on entity type: {}", request.entity_type),
    });
    
    // Add more factors based on context
    if let Some(tx_volume) = request.context.get("transaction_volume") {
        if let Ok(volume) = tx_volume.parse::<u64>() {
            factors.push(RiskFactor {
                name: "transaction_volume".to_string(),
                category: "behavioral".to_string(),
                weight: 0.15,
                score: calculate_volume_risk(volume),
                description: format!("Transaction volume: {}", volume),
            });
        }
    }
    
    factors
}

/// Calculate geographic risk score
fn calculate_geographic_risk(jurisdiction: &str) -> f64 {
    match jurisdiction {
        "US" | "EU" | "GB" | "CA" | "AU" => 0.1,  // Low risk
        "APAC" => 0.25,                            // Medium risk
        "FATF" => 0.15,                            // Low-medium
        _ => 0.5,                                  // Unknown - higher risk
    }
}

/// Calculate entity type risk
fn calculate_entity_type_risk(entity_type: &str) -> f64 {
    match entity_type {
        "individual" => 0.2,
        "business" => 0.3,
        "financial_institution" => 0.15,
        "government" => 0.1,
        "exchange" => 0.4,
        "mixer" => 0.9,
        _ => 0.5,
    }
}

/// Calculate volume-based risk
fn calculate_volume_risk(volume: u64) -> f64 {
    match volume {
        0..=10_000 => 0.1,
        10_001..=100_000 => 0.2,
        100_001..=1_000_000 => 0.4,
        _ => 0.6,
    }
}

/// Generate recommendations based on risk score
fn generate_risk_recommendations(risk_score: f64) -> Vec<String> {
    let mut recommendations = Vec::new();
    
    if risk_score < 0.25 {
        recommendations.push("Standard monitoring sufficient".to_string());
    } else if risk_score < 0.5 {
        recommendations.push("Enhanced monitoring recommended".to_string());
        recommendations.push("Periodic review of transaction patterns".to_string());
    } else if risk_score < 0.75 {
        recommendations.push("Enhanced due diligence required".to_string());
        recommendations.push("Frequent transaction monitoring".to_string());
        recommendations.push("Senior review for large transactions".to_string());
    } else {
        recommendations.push("Immediate compliance review required".to_string());
        recommendations.push("Transaction blocking may be necessary".to_string());
        recommendations.push("Consider SAR filing".to_string());
    }
    
    recommendations
}
