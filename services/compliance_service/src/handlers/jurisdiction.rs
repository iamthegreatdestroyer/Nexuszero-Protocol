//! Jurisdiction management handlers

use axum::{
    extract::{Path, State},
    Json,
};
use std::sync::Arc;

use crate::config::JurisdictionConfig;
use crate::error::{ComplianceError, Result};
use crate::models::*;
use crate::state::AppState;

/// List all supported jurisdictions
pub async fn list_jurisdictions(
    State(state): State<Arc<AppState>>,
) -> Json<Vec<JurisdictionInfo>> {
    let enabled = &state.config.enabled_jurisdictions;
    
    let jurisdictions: Vec<JurisdictionInfo> = enabled
        .iter()
        .map(|code| {
            let config = match code.as_str() {
                "FATF" => JurisdictionConfig::fatf(),
                "EU" => JurisdictionConfig::eu(),
                "US" => JurisdictionConfig::us(),
                "APAC" => JurisdictionConfig::apac(),
                _ => JurisdictionConfig {
                    code: code.clone(),
                    name: code.clone(),
                    reporting_threshold: 10000,
                    kyc_required: true,
                    travel_rule_enabled: true,
                    sar_required: true,
                    data_retention_years: 5,
                },
            };
            
            JurisdictionInfo {
                code: config.code.clone(),
                name: config.name,
                region: get_region(&config.code),
                reporting_threshold: config.reporting_threshold,
                currency: get_currency(&config.code),
                kyc_required: config.kyc_required,
                travel_rule_enabled: config.travel_rule_enabled,
                travel_rule_threshold: 3000,
                sar_required: config.sar_required,
                data_retention_years: config.data_retention_years,
                active_rules: 0, // Would be populated from DB
            }
        })
        .collect();
    
    Json(jurisdictions)
}

/// Get specific jurisdiction details
pub async fn get_jurisdiction(
    State(state): State<Arc<AppState>>,
    Path(code): Path<String>,
) -> Result<Json<JurisdictionInfo>> {
    if !state.config.enabled_jurisdictions.contains(&code) {
        return Err(ComplianceError::UnsupportedJurisdiction(code));
    }
    
    let config = match code.as_str() {
        "FATF" => JurisdictionConfig::fatf(),
        "EU" => JurisdictionConfig::eu(),
        "US" => JurisdictionConfig::us(),
        "APAC" => JurisdictionConfig::apac(),
        _ => return Err(ComplianceError::UnsupportedJurisdiction(code)),
    };
    
    Ok(Json(JurisdictionInfo {
        code: config.code.clone(),
        name: config.name,
        region: get_region(&config.code),
        reporting_threshold: config.reporting_threshold,
        currency: get_currency(&config.code),
        kyc_required: config.kyc_required,
        travel_rule_enabled: config.travel_rule_enabled,
        travel_rule_threshold: 3000,
        sar_required: config.sar_required,
        data_retention_years: config.data_retention_years,
        active_rules: 0,
    }))
}

/// Get rules for jurisdiction
pub async fn get_rules(
    State(state): State<Arc<AppState>>,
    Path(code): Path<String>,
) -> Result<Json<Vec<ComplianceRule>>> {
    if !state.config.enabled_jurisdictions.contains(&code) {
        return Err(ComplianceError::UnsupportedJurisdiction(code));
    }
    
    let engine = state.rule_engine.read().await;
    let rules = engine.get_rules(&code);
    
    Ok(Json(rules))
}

/// Get thresholds for jurisdiction
pub async fn get_thresholds(
    State(state): State<Arc<AppState>>,
    Path(code): Path<String>,
) -> Result<Json<RiskThresholds>> {
    if !state.config.enabled_jurisdictions.contains(&code) {
        return Err(ComplianceError::UnsupportedJurisdiction(code));
    }
    
    let engine = state.rule_engine.read().await;
    let thresholds = engine.get_thresholds(&code);
    
    Ok(Json(thresholds))
}

/// Get region for jurisdiction code
fn get_region(code: &str) -> String {
    match code {
        "US" | "CA" => "Americas".to_string(),
        "EU" | "GB" | "CH" => "Europe".to_string(),
        "APAC" | "SG" | "HK" | "JP" | "AU" => "Asia-Pacific".to_string(),
        "FATF" => "Global".to_string(),
        _ => "Other".to_string(),
    }
}

/// Get currency for jurisdiction
fn get_currency(code: &str) -> String {
    match code {
        "US" => "USD".to_string(),
        "EU" => "EUR".to_string(),
        "GB" => "GBP".to_string(),
        "CH" => "CHF".to_string(),
        "JP" => "JPY".to_string(),
        "AU" => "AUD".to_string(),
        "CA" => "CAD".to_string(),
        _ => "USD".to_string(),
    }
}
