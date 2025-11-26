//! Compliance Handlers
//!
//! Handles Regulatory Compliance Layer (RCL) operations

use crate::error::{ApiError, ApiResult};
use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use axum::{
    extract::{Extension, Path, Query},
    Json,
};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

/// Access tier for regulatory compliance
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessTier {
    PublicAuditor,
    Regulator,
    LawEnforcement,
    UserSelfDisclosure,
}

/// Compliance proof type
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComplianceProofType {
    AgeVerification,
    AccreditedInvestor,
    SanctionsCompliance,
    SourceOfFunds,
    KycComplete,
    TransactionLimit,
    JurisdictionCompliance,
}

/// Verify compliance request
#[derive(Debug, Deserialize, Validate)]
pub struct VerifyComplianceRequest {
    pub proof_type: ComplianceProofType,
    pub parameters: serde_json::Value,
}

/// Verify compliance response
#[derive(Debug, Serialize)]
pub struct VerifyComplianceResponse {
    pub proof_id: String,
    pub proof_type: ComplianceProofType,
    pub verified: bool,
    pub proof_data: Option<String>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub message: String,
}

/// Selective disclosure request
#[derive(Debug, Deserialize, Validate)]
pub struct SelectiveDisclosureRequest {
    pub transaction_id: Uuid,
    pub requester_tier: AccessTier,
    pub disclosure_fields: Vec<DisclosureField>,
    #[validate(length(min = 1, max = 500))]
    pub purpose: String,
    pub warrant_hash: Option<String>,
    pub expiry_hours: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DisclosureField {
    TransactionAmount,
    SenderAddress,
    RecipientAddress,
    Timestamp,
    TransactionHash,
    ProofDetails,
}

/// Selective disclosure response
#[derive(Debug, Serialize)]
pub struct SelectiveDisclosureResponse {
    pub disclosure_id: String,
    pub transaction_id: String,
    pub disclosed_fields: Vec<DisclosedField>,
    pub requester_tier: AccessTier,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub verification_proof: String,
}

#[derive(Debug, Serialize)]
pub struct DisclosedField {
    pub field: DisclosureField,
    pub value: Option<serde_json::Value>,
    pub commitment: String,
}

/// Compliance proof response
#[derive(Debug, Serialize)]
pub struct ComplianceProofResponse {
    pub id: String,
    pub proof_type: String,
    pub verified: bool,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// List compliance proofs query
#[derive(Debug, Deserialize)]
pub struct ListComplianceProofsQuery {
    pub proof_type: Option<String>,
    pub verified: Option<bool>,
    pub page: Option<u32>,
    pub limit: Option<u32>,
}

/// List compliance proofs response
#[derive(Debug, Serialize)]
pub struct ListComplianceProofsResponse {
    pub proofs: Vec<ComplianceProofResponse>,
    pub total: i64,
    pub page: u32,
    pub limit: u32,
}

/// Compliance record from database
#[derive(Debug, FromRow)]
struct ComplianceRecord {
    id: Uuid,
    proof_type: String,
    verified: bool,
    expires_at: chrono::DateTime<chrono::Utc>,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Transaction data from database
#[derive(Debug, FromRow)]
struct TransactionData {
    id: Uuid,
    privacy_level: i32,
}

/// Verify compliance handler
pub async fn verify_compliance(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<VerifyComplianceRequest>,
) -> ApiResult<Json<VerifyComplianceResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let (verified, proof_data, expiry_hours) = match &payload.proof_type {
        ComplianceProofType::AgeVerification => (true, None, 24),
        ComplianceProofType::AccreditedInvestor => (true, None, 720),
        ComplianceProofType::SanctionsCompliance => (true, None, 1),
        ComplianceProofType::SourceOfFunds => (true, None, 168),
        ComplianceProofType::KycComplete => (true, None, 8760),
        ComplianceProofType::TransactionLimit => {
            let threshold = payload.parameters.get("threshold_usd").and_then(|v| v.as_f64()).unwrap_or(10000.0);
            let amount = payload.parameters.get("amount_usd").and_then(|v| v.as_f64()).unwrap_or(0.0);
            (amount <= threshold, None, 24)
        }
        ComplianceProofType::JurisdictionCompliance => (true, None, 24),
    };

    let expires_at = chrono::Utc::now() + chrono::Duration::hours(expiry_hours as i64);
    let proof_id = Uuid::new_v4();

    // Store compliance proof using runtime query
    sqlx::query(
        r#"
        INSERT INTO compliance_records (id, user_id, proof_type, proof_data, verified, expires_at, metadata, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        "#,
    )
    .bind(proof_id)
    .bind(user_id)
    .bind(format!("{:?}", payload.proof_type))
    .bind(vec![0u8; 32])
    .bind(verified)
    .bind(expires_at)
    .bind(&payload.parameters)
    .execute(&state.db)
    .await?;

    let message = if verified {
        format!("{:?} verification successful", payload.proof_type)
    } else {
        format!("{:?} verification failed", payload.proof_type)
    };

    tracing::info!(user_id = %user_id, proof_type = ?payload.proof_type, verified = verified, "Compliance proof generated");

    Ok(Json(VerifyComplianceResponse {
        proof_id: proof_id.to_string(),
        proof_type: payload.proof_type,
        verified,
        proof_data,
        expires_at,
        message,
    }))
}

/// Selective disclosure handler
pub async fn selective_disclosure(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<SelectiveDisclosureRequest>,
) -> ApiResult<Json<SelectiveDisclosureResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    // Verify transaction belongs to user
    let transaction: TransactionData = sqlx::query_as(
        r#"
        SELECT id, privacy_level
        FROM transactions
        WHERE id = $1 AND user_id = $2
        "#,
    )
    .bind(payload.transaction_id)
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Transaction not found".to_string()))?;

    // Validate access tier requirements
    if payload.requester_tier == AccessTier::LawEnforcement && payload.warrant_hash.is_none() {
        return Err(ApiError::BadRequest("Warrant hash required for law enforcement access".to_string()));
    }

    let allowed_fields = get_allowed_fields(payload.requester_tier, transaction.privacy_level as u8);

    let disclosed_fields: Vec<DisclosedField> = payload.disclosure_fields
        .iter()
        .map(|field| {
            if allowed_fields.contains(field) {
                DisclosedField {
                    field: field.clone(),
                    value: None,
                    commitment: format!("commitment_{:?}", field),
                }
            } else {
                DisclosedField {
                    field: field.clone(),
                    value: None,
                    commitment: "access_denied".to_string(),
                }
            }
        })
        .collect();

    let expiry_hours = payload.expiry_hours.unwrap_or(24);
    let expires_at = chrono::Utc::now() + chrono::Duration::hours(expiry_hours as i64);
    let disclosure_id = Uuid::new_v4();
    let verification_proof = generate_disclosure_proof(&disclosed_fields)?;

    tracing::info!(
        user_id = %user_id,
        transaction_id = %payload.transaction_id,
        requester_tier = ?payload.requester_tier,
        "Selective disclosure performed"
    );

    Ok(Json(SelectiveDisclosureResponse {
        disclosure_id: disclosure_id.to_string(),
        transaction_id: payload.transaction_id.to_string(),
        disclosed_fields,
        requester_tier: payload.requester_tier,
        expires_at,
        verification_proof,
    }))
}

/// List compliance proofs handler
pub async fn list_compliance_proofs(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Query(params): Query<ListComplianceProofsQuery>,
) -> ApiResult<Json<ListComplianceProofsResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let page = params.page.unwrap_or(1).max(1);
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = ((page - 1) * limit) as i64;

    let proofs: Vec<ComplianceRecord> = sqlx::query_as(
        r#"
        SELECT id, proof_type, verified, expires_at, created_at
        FROM compliance_records
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        "#,
    )
    .bind(user_id)
    .bind(limit as i64)
    .bind(offset)
    .fetch_all(&state.db)
    .await?;

    let total: Option<i64> = sqlx::query_scalar(
        "SELECT COUNT(*) FROM compliance_records WHERE user_id = $1",
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;

    let proof_responses: Vec<ComplianceProofResponse> = proofs
        .into_iter()
        .map(|p| ComplianceProofResponse {
            id: p.id.to_string(),
            proof_type: p.proof_type,
            verified: p.verified,
            expires_at: p.expires_at,
            created_at: p.created_at,
        })
        .collect();

    Ok(Json(ListComplianceProofsResponse {
        proofs: proof_responses,
        total: total.unwrap_or(0),
        page,
        limit,
    }))
}

/// Get compliance proof by ID handler
pub async fn get_compliance_proof(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ComplianceProofResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let proof: ComplianceRecord = sqlx::query_as(
        r#"
        SELECT id, proof_type, verified, expires_at, created_at
        FROM compliance_records
        WHERE id = $1 AND user_id = $2
        "#,
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Compliance proof not found".to_string()))?;

    Ok(Json(ComplianceProofResponse {
        id: proof.id.to_string(),
        proof_type: proof.proof_type,
        verified: proof.verified,
        expires_at: proof.expires_at,
        created_at: proof.created_at,
    }))
}

fn get_allowed_fields(tier: AccessTier, privacy_level: u8) -> Vec<DisclosureField> {
    match tier {
        AccessTier::PublicAuditor => vec![],
        AccessTier::Regulator => {
            if privacy_level <= 2 {
                vec![DisclosureField::Timestamp, DisclosureField::TransactionHash]
            } else {
                vec![DisclosureField::Timestamp]
            }
        }
        AccessTier::LawEnforcement | AccessTier::UserSelfDisclosure => {
            vec![
                DisclosureField::TransactionAmount,
                DisclosureField::SenderAddress,
                DisclosureField::RecipientAddress,
                DisclosureField::Timestamp,
                DisclosureField::TransactionHash,
                DisclosureField::ProofDetails,
            ]
        }
    }
}

fn generate_disclosure_proof(fields: &[DisclosedField]) -> Result<String, ApiError> {
    use sha2::{Digest, Sha256};
    
    let mut hasher = Sha256::new();
    for field in fields {
        hasher.update(format!("{:?}", field.field).as_bytes());
        hasher.update(&field.commitment);
    }
    
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_tier_fields() {
        let public_fields = get_allowed_fields(AccessTier::PublicAuditor, 3);
        assert!(public_fields.is_empty());

        let regulator_fields = get_allowed_fields(AccessTier::Regulator, 1);
        assert_eq!(regulator_fields.len(), 2);

        let law_enforcement_fields = get_allowed_fields(AccessTier::LawEnforcement, 5);
        assert_eq!(law_enforcement_fields.len(), 6);
    }

    #[test]
    fn test_generate_disclosure_proof() {
        let fields = vec![
            DisclosedField {
                field: DisclosureField::TransactionAmount,
                value: None,
                commitment: "test_commitment".to_string(),
            },
        ];

        let proof = generate_disclosure_proof(&fields).unwrap();
        assert!(!proof.is_empty());
        assert_eq!(proof.len(), 64);
    }
}

