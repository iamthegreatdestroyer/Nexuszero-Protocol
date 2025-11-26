//! KYC/AML handlers

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::Result;
use crate::handlers::metrics::{AML_SCREENINGS, KYC_VERIFICATIONS, WATCHLIST_MATCHES};
use crate::models::*;
use crate::state::AppState;

/// Verify identity (KYC)
pub async fn verify_identity(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<KycVerificationRequest>,
) -> Result<(StatusCode, Json<KycVerificationResult>)> {
    // In production, this would call external KYC provider
    let result = perform_kyc_verification(&request).await;
    
    let status_str = format!("{:?}", result.status).to_lowercase();
    KYC_VERIFICATIONS.with_label_values(&[&status_str]).inc();
    
    Ok((StatusCode::OK, Json(result)))
}

/// Get KYC status for entity
pub async fn get_kyc_status(
    State(_state): State<Arc<AppState>>,
    Path(entity_id): Path<Uuid>,
) -> Result<Json<KycVerificationResult>> {
    // In production, this would query stored KYC results
    Ok(Json(KycVerificationResult {
        entity_id,
        status: KycStatus::Verified,
        verification_level: 2,
        checks_passed: vec!["identity".to_string(), "address".to_string()],
        checks_failed: vec![],
        requires_manual_review: false,
        verified_at: Some(Utc::now()),
        expires_at: Some(Utc::now() + chrono::Duration::days(365)),
    }))
}

/// Perform AML screening
pub async fn aml_screening(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<AmlScreeningRequest>,
) -> Result<(StatusCode, Json<AmlScreeningResult>)> {
    // In production, this would check against watchlists
    let result = perform_aml_screening(&request).await;
    
    let status = if result.matches.is_empty() { "clear" } else { "match" };
    AML_SCREENINGS.with_label_values(&[status]).inc();
    
    if !result.matches.is_empty() {
        WATCHLIST_MATCHES.inc();
    }
    
    Ok((StatusCode::OK, Json(result)))
}

/// Check against watchlists
pub async fn watchlist_check(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<AmlScreeningRequest>,
) -> Result<Json<Vec<WatchlistMatch>>> {
    let result = perform_aml_screening(&request).await;
    Ok(Json(result.matches))
}

/// Perform KYC verification (simulated)
async fn perform_kyc_verification(request: &KycVerificationRequest) -> KycVerificationResult {
    let mut checks_passed = Vec::new();
    let mut checks_failed = Vec::new();
    
    // Simulate ID verification
    if !request.id_number.is_empty() {
        checks_passed.push("id_document".to_string());
    } else {
        checks_failed.push("id_document".to_string());
    }
    
    // Simulate name verification
    if !request.full_name.is_empty() {
        checks_passed.push("name_match".to_string());
    } else {
        checks_failed.push("name_match".to_string());
    }
    
    // Simulate address verification
    if request.address.is_some() {
        checks_passed.push("address".to_string());
    }
    
    // Simulate DOB verification
    if request.date_of_birth.is_some() {
        checks_passed.push("date_of_birth".to_string());
    }
    
    let status = if checks_failed.is_empty() {
        KycStatus::Verified
    } else if checks_passed.len() > checks_failed.len() {
        KycStatus::Pending
    } else {
        KycStatus::Failed
    };
    
    let verification_level = match checks_passed.len() {
        0..=1 => 0,
        2 => 1,
        3 => 2,
        _ => 3,
    };
    
    KycVerificationResult {
        entity_id: request.entity_id,
        status: status.clone(),
        verification_level,
        checks_passed,
        checks_failed,
        requires_manual_review: verification_level < 2,
        verified_at: if status == KycStatus::Verified {
            Some(Utc::now())
        } else {
            None
        },
        expires_at: if status == KycStatus::Verified {
            Some(Utc::now() + chrono::Duration::days(365))
        } else {
            None
        },
    }
}

/// Perform AML screening (simulated)
async fn perform_aml_screening(request: &AmlScreeningRequest) -> AmlScreeningResult {
    let mut matches = Vec::new();
    let mut risk_indicators = Vec::new();
    
    // Simulate watchlist checks
    let name_lower = request.name.to_lowercase();
    
    // Check against simulated OFAC list
    let ofac_names = ["test blocked", "blocked entity", "sanctioned person"];
    for blocked in &ofac_names {
        if name_lower.contains(blocked) {
            matches.push(WatchlistMatch {
                list_name: "OFAC SDN".to_string(),
                list_type: "sanctions".to_string(),
                match_score: 0.95,
                matched_name: blocked.to_string(),
                matched_fields: vec!["name".to_string()],
                source_url: Some("https://sanctionssearch.ofac.treas.gov/".to_string()),
            });
            risk_indicators.push("OFAC sanctions match".to_string());
        }
    }
    
    // Check PEP (Politically Exposed Persons)
    let pep_indicators = ["minister", "president", "governor", "senator"];
    for indicator in &pep_indicators {
        if name_lower.contains(indicator) {
            matches.push(WatchlistMatch {
                list_name: "PEP Database".to_string(),
                list_type: "pep".to_string(),
                match_score: 0.7,
                matched_name: request.name.clone(),
                matched_fields: vec!["title".to_string()],
                source_url: None,
            });
            risk_indicators.push("Potential PEP".to_string());
            break;
        }
    }
    
    let overall_status = if matches.is_empty() {
        "clear".to_string()
    } else if matches.iter().any(|m| m.match_score > 0.9) {
        "match".to_string()
    } else {
        "potential_match".to_string()
    };
    
    AmlScreeningResult {
        entity_id: request.entity_id,
        screened_at: Utc::now(),
        matches,
        risk_indicators,
        overall_status,
    }
}
