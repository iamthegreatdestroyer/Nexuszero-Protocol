//! SAR (Suspicious Activity Report) handlers

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{ComplianceError, Result};
use crate::handlers::metrics::{SARS_CREATED, SARS_SUBMITTED};
use crate::models::*;
use crate::state::AppState;

/// Create new SAR
pub async fn create_sar(
    State(_state): State<Arc<AppState>>,
    Json(request): Json<CreateSarRequest>,
) -> Result<(StatusCode, Json<SuspiciousActivityReport>)> {
    let sar_id = Uuid::new_v4();
    let reference = generate_sar_reference(&request.jurisdiction);
    
    let sar = SuspiciousActivityReport {
        id: sar_id,
        reference_number: reference,
        status: SarStatus::Draft,
        subject_id: request.subject_id,
        subject_type: request.subject_type,
        activity_type: request.activity_type,
        amount_involved: request.amount_involved,
        currency: request.currency,
        jurisdiction: request.jurisdiction.clone(),
        description: request.description,
        transaction_ids: request.transaction_ids,
        evidence: vec![],
        created_at: Utc::now(),
        updated_at: Utc::now(),
        submitted_at: None,
        created_by: Uuid::nil(), // Would come from auth context
        reviewed_by: None,
    };
    
    SARS_CREATED.with_label_values(&[&request.jurisdiction]).inc();
    
    Ok((StatusCode::CREATED, Json(sar)))
}

#[derive(Debug, Deserialize)]
pub struct SarListParams {
    pub status: Option<String>,
    pub jurisdiction: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

/// List SARs with filters
pub async fn list_sars(
    State(_state): State<Arc<AppState>>,
    Query(params): Query<SarListParams>,
) -> Result<Json<Vec<SuspiciousActivityReport>>> {
    // In production, this would query the database
    Ok(Json(vec![]))
}

/// Get SAR by ID
pub async fn get_sar(
    State(_state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SuspiciousActivityReport>> {
    // In production, this would query the database
    Err(ComplianceError::SarNotFound(id.to_string()))
}

/// Update SAR request
#[derive(Debug, Deserialize)]
pub struct UpdateSarRequest {
    pub description: Option<String>,
    pub evidence: Option<Vec<SarEvidence>>,
    pub transaction_ids: Option<Vec<Uuid>>,
}

/// Update SAR
pub async fn update_sar(
    State(_state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<UpdateSarRequest>,
) -> Result<Json<SuspiciousActivityReport>> {
    // In production, this would update in database
    Err(ComplianceError::SarNotFound(id.to_string()))
}

/// Submit SAR to regulatory authority
pub async fn submit_sar(
    State(_state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SarSubmissionResult>> {
    // Validate SAR is complete
    // Submit to regulatory endpoint
    // Update status
    
    let result = SarSubmissionResult {
        sar_id: id,
        submitted: true,
        submission_id: format!("SUB-{}", Uuid::new_v4()),
        submitted_at: Utc::now(),
        confirmation_number: Some(generate_confirmation_number()),
        warnings: vec![],
    };
    
    SARS_SUBMITTED.with_label_values(&["UNKNOWN"]).inc();
    
    Ok(Json(result))
}

/// SAR submission result
#[derive(Debug, serde::Serialize)]
pub struct SarSubmissionResult {
    pub sar_id: Uuid,
    pub submitted: bool,
    pub submission_id: String,
    pub submitted_at: chrono::DateTime<Utc>,
    pub confirmation_number: Option<String>,
    pub warnings: Vec<String>,
}

/// Generate SAR reference number
fn generate_sar_reference(jurisdiction: &str) -> String {
    let year = Utc::now().format("%Y");
    let random = &Uuid::new_v4().to_string()[..8];
    format!("SAR-{}-{}-{}", jurisdiction, year, random.to_uppercase())
}

/// Generate confirmation number
fn generate_confirmation_number() -> String {
    let random = &Uuid::new_v4().to_string()[..12];
    format!("CONF-{}", random.to_uppercase())
}
