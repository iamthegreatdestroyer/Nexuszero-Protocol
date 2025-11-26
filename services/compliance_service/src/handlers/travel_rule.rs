//! Travel Rule handlers

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{ComplianceError, Result};
use crate::handlers::metrics::TRAVEL_RULE_TRANSFERS;
use crate::models::*;
use crate::state::AppState;

/// Submit originator information
pub async fn submit_originator(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TravelRuleOriginator>,
) -> Result<(StatusCode, Json<TravelRuleSubmissionResult>)> {
    // Validate required fields based on threshold
    let validation = validate_originator(&request, state.config.travel_rule_threshold);
    
    if !validation.is_valid {
        TRAVEL_RULE_TRANSFERS.with_label_values(&["rejected"]).inc();
        return Ok((StatusCode::BAD_REQUEST, Json(TravelRuleSubmissionResult {
            transfer_id: request.transfer_id,
            accepted: false,
            missing_fields: validation.missing_fields,
            submitted_at: Utc::now(),
        })));
    }
    
    TRAVEL_RULE_TRANSFERS.with_label_values(&["accepted"]).inc();
    
    Ok((StatusCode::CREATED, Json(TravelRuleSubmissionResult {
        transfer_id: request.transfer_id,
        accepted: true,
        missing_fields: vec![],
        submitted_at: Utc::now(),
    })))
}

/// Submit beneficiary information
pub async fn submit_beneficiary(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TravelRuleBeneficiary>,
) -> Result<(StatusCode, Json<TravelRuleSubmissionResult>)> {
    let validation = validate_beneficiary(&request);
    
    if !validation.is_valid {
        return Ok((StatusCode::BAD_REQUEST, Json(TravelRuleSubmissionResult {
            transfer_id: request.transfer_id,
            accepted: false,
            missing_fields: validation.missing_fields,
            submitted_at: Utc::now(),
        })));
    }
    
    Ok((StatusCode::CREATED, Json(TravelRuleSubmissionResult {
        transfer_id: request.transfer_id,
        accepted: true,
        missing_fields: vec![],
        submitted_at: Utc::now(),
    })))
}

/// Verify transfer compliance
pub async fn verify_transfer(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<TravelRuleVerifyRequest>,
) -> Result<Json<TravelRuleVerification>> {
    let mut missing_fields = Vec::new();
    let mut warnings = Vec::new();
    
    // Validate originator data
    if request.originator_name.is_none() {
        missing_fields.push("originator_name".to_string());
    }
    if request.originator_account.is_none() {
        missing_fields.push("originator_account".to_string());
    }
    
    // Validate beneficiary data
    if request.beneficiary_name.is_none() {
        missing_fields.push("beneficiary_name".to_string());
    }
    if request.beneficiary_account.is_none() {
        missing_fields.push("beneficiary_account".to_string());
    }
    
    // Check VASP registration
    if let Some(ref vasp_id) = request.originator_vasp_id {
        if !is_registered_vasp(vasp_id) {
            warnings.push(format!("Originator VASP {} not registered", vasp_id));
        }
    }
    
    let is_compliant = missing_fields.is_empty();
    
    Ok(Json(TravelRuleVerification {
        transfer_id: request.transfer_id,
        is_compliant,
        missing_fields,
        warnings,
        verified_at: Utc::now(),
    }))
}

/// Get VASP information
pub async fn get_vasp_info(
    State(_state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<VaspInfo>> {
    // In production, this would query VASP registry
    if id == "unknown" {
        return Err(ComplianceError::EntityNotFound(format!("VASP {}", id)));
    }
    
    Ok(Json(VaspInfo {
        vasp_id: id.clone(),
        name: format!("VASP {}", id),
        jurisdiction: "US".to_string(),
        registration_number: Some(format!("REG-{}", id)),
        is_registered: true,
        travel_rule_capable: true,
        protocols_supported: vec!["TRISA".to_string(), "TRP".to_string()],
        endpoint_url: Some(format!("https://vasp-{}.example.com/travel-rule", id)),
    }))
}

/// Travel Rule submission result
#[derive(Debug, serde::Serialize)]
pub struct TravelRuleSubmissionResult {
    pub transfer_id: Uuid,
    pub accepted: bool,
    pub missing_fields: Vec<String>,
    pub submitted_at: chrono::DateTime<Utc>,
}

/// Travel Rule verification request
#[derive(Debug, serde::Deserialize)]
pub struct TravelRuleVerifyRequest {
    pub transfer_id: Uuid,
    pub amount: Option<u64>,
    pub currency: Option<String>,
    pub originator_vasp_id: Option<String>,
    pub originator_name: Option<String>,
    pub originator_account: Option<String>,
    pub beneficiary_vasp_id: Option<String>,
    pub beneficiary_name: Option<String>,
    pub beneficiary_account: Option<String>,
}

/// Validation result
struct ValidationResult {
    is_valid: bool,
    missing_fields: Vec<String>,
}

/// Validate originator data
fn validate_originator(request: &TravelRuleOriginator, _threshold: u64) -> ValidationResult {
    let mut missing = Vec::new();
    
    if request.originator_name.is_empty() {
        missing.push("originator_name".to_string());
    }
    if request.originator_account.is_empty() {
        missing.push("originator_account".to_string());
    }
    if request.originator_vasp_id.is_empty() {
        missing.push("originator_vasp_id".to_string());
    }
    if request.originator_location.is_empty() {
        missing.push("originator_location".to_string());
    }
    
    ValidationResult {
        is_valid: missing.is_empty(),
        missing_fields: missing,
    }
}

/// Validate beneficiary data
fn validate_beneficiary(request: &TravelRuleBeneficiary) -> ValidationResult {
    let mut missing = Vec::new();
    
    if request.beneficiary_name.is_empty() {
        missing.push("beneficiary_name".to_string());
    }
    if request.beneficiary_account.is_empty() {
        missing.push("beneficiary_account".to_string());
    }
    if request.beneficiary_vasp_id.is_empty() {
        missing.push("beneficiary_vasp_id".to_string());
    }
    
    ValidationResult {
        is_valid: missing.is_empty(),
        missing_fields: missing,
    }
}

/// Check if VASP is registered
fn is_registered_vasp(vasp_id: &str) -> bool {
    // In production, this would check against VASP registry
    !vasp_id.starts_with("unregistered")
}
