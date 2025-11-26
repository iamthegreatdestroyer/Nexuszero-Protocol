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
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

/// Access tier for regulatory compliance
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessTier {
    /// Tier 1: Public auditors - aggregate statistics only
    PublicAuditor,
    /// Tier 2: Regulators - transaction patterns, no amounts
    Regulator,
    /// Tier 3: Law enforcement - full transaction details with warrant
    LawEnforcement,
    /// Tier 4: User self-disclosure - voluntary full disclosure
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
    /// Type of compliance proof
    pub proof_type: ComplianceProofType,

    /// Additional parameters based on proof type
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
    /// Transaction ID to disclose
    pub transaction_id: Uuid,

    /// Requester access tier
    pub requester_tier: AccessTier,

    /// Fields to disclose
    pub disclosure_fields: Vec<DisclosureField>,

    /// Purpose of disclosure
    #[validate(length(min = 1, max = 500))]
    pub purpose: String,

    /// Warrant hash (required for LawEnforcement tier)
    pub warrant_hash: Option<String>,

    /// Disclosure expiry
    pub expiry_hours: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

/// Verify compliance handler
pub async fn verify_compliance(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<VerifyComplianceRequest>,
) -> ApiResult<Json<VerifyComplianceResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    // Generate compliance proof based on type
    let (verified, proof_data, expiry_hours) = match &payload.proof_type {
        ComplianceProofType::AgeVerification => {
            let min_age = payload.parameters
                .get("minimum_age")
                .and_then(|v| v.as_u64())
                .unwrap_or(18) as u8;
            
            // In production, would verify against encrypted user data
            let verified = verify_age_requirement(&state, &user_id, min_age).await?;
            (verified, None, 24)
        }
        ComplianceProofType::AccreditedInvestor => {
            let jurisdiction = payload.parameters
                .get("jurisdiction")
                .and_then(|v| v.as_str())
                .unwrap_or("US");
            
            let verified = verify_accredited_investor(&state, &user_id, jurisdiction).await?;
            (verified, None, 720) // 30 days
        }
        ComplianceProofType::SanctionsCompliance => {
            let verified = verify_sanctions_compliance(&state, &user_id).await?;
            (verified, None, 1) // 1 hour - must be re-verified frequently
        }
        ComplianceProofType::SourceOfFunds => {
            let category = payload.parameters
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("employment");
            
            let verified = verify_source_of_funds(&state, &user_id, category).await?;
            (verified, None, 168) // 7 days
        }
        ComplianceProofType::KycComplete => {
            let provider = payload.parameters
                .get("provider")
                .and_then(|v| v.as_str())
                .unwrap_or("internal");
            
            let verified = verify_kyc_complete(&state, &user_id, provider).await?;
            (verified, None, 8760) // 1 year
        }
        ComplianceProofType::TransactionLimit => {
            let threshold_usd = payload.parameters
                .get("threshold_usd")
                .and_then(|v| v.as_f64())
                .unwrap_or(10000.0);
            let amount_usd = payload.parameters
                .get("amount_usd")
                .and_then(|v| v.as_f64())
                .ok_or(ApiError::BadRequest("amount_usd required".to_string()))?;
            
            let verified = amount_usd <= threshold_usd;
            (verified, None, 24)
        }
        ComplianceProofType::JurisdictionCompliance => {
            let jurisdiction = payload.parameters
                .get("jurisdiction")
                .and_then(|v| v.as_str())
                .ok_or(ApiError::BadRequest("jurisdiction required".to_string()))?;
            
            let verified = verify_jurisdiction_compliance(&state, &user_id, jurisdiction).await?;
            (verified, None, 24)
        }
    };

    let expires_at = chrono::Utc::now() + chrono::Duration::hours(expiry_hours as i64);
    let proof_id = Uuid::new_v4();

    // Store compliance proof
    sqlx::query!(
        r#"
        INSERT INTO compliance_records (id, user_id, proof_type, proof_data, verified, expires_at, metadata, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        "#,
        proof_id,
        user_id,
        format!("{:?}", payload.proof_type),
        &vec![0u8; 32], // Placeholder proof data
        verified,
        expires_at,
        payload.parameters
    )
    .execute(&state.db)
    .await?;

    let message = if verified {
        format!("{:?} verification successful", payload.proof_type)
    } else {
        format!("{:?} verification failed", payload.proof_type)
    };

    tracing::info!(
        user_id = %user_id,
        proof_type = ?payload.proof_type,
        verified = verified,
        "Compliance proof generated"
    );

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
    let transaction = sqlx::query!(
        r#"
        SELECT id, sender_commitment, recipient_commitment, amount_commitment, privacy_level
        FROM transactions
        WHERE id = $1 AND user_id = $2
        "#,
        payload.transaction_id,
        user_id
    )
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Transaction not found".to_string()))?;

    // Validate access tier requirements
    match payload.requester_tier {
        AccessTier::LawEnforcement => {
            if payload.warrant_hash.is_none() {
                return Err(ApiError::BadRequest(
                    "Warrant hash required for law enforcement access".to_string(),
                ));
            }
        }
        _ => {}
    }

    // Determine which fields can be disclosed based on tier and privacy level
    let allowed_fields = get_allowed_fields(payload.requester_tier, transaction.privacy_level as u8);

    let mut disclosed_fields = Vec::new();
    for field in payload.disclosure_fields {
        if allowed_fields.contains(&field) {
            let (value, commitment) = disclose_field(
                &state,
                &transaction,
                &field,
                payload.requester_tier,
            ).await?;

            disclosed_fields.push(DisclosedField {
                field,
                value,
                commitment,
            });
        } else {
            // Field not allowed for this tier
            disclosed_fields.push(DisclosedField {
                field,
                value: None,
                commitment: "access_denied".to_string(),
            });
        }
    }

    let expiry_hours = payload.expiry_hours.unwrap_or(24);
    let expires_at = chrono::Utc::now() + chrono::Duration::hours(expiry_hours as i64);
    let disclosure_id = Uuid::new_v4();

    // Generate verification proof
    let verification_proof = generate_disclosure_proof(&disclosed_fields)?;

    tracing::info!(
        user_id = %user_id,
        transaction_id = %payload.transaction_id,
        requester_tier = ?payload.requester_tier,
        fields_disclosed = disclosed_fields.len(),
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

    let proofs = sqlx::query_as!(
        ComplianceRecord,
        r#"
        SELECT id, proof_type, verified, expires_at, created_at
        FROM compliance_records
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        "#,
        user_id,
        limit as i64,
        offset
    )
    .fetch_all(&state.db)
    .await?;

    let total = sqlx::query_scalar!(
        "SELECT COUNT(*) FROM compliance_records WHERE user_id = $1",
        user_id
    )
    .fetch_one(&state.db)
    .await?
    .unwrap_or(0);

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
        total,
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

    let proof = sqlx::query_as!(
        ComplianceRecord,
        r#"
        SELECT id, proof_type, verified, expires_at, created_at
        FROM compliance_records
        WHERE id = $1 AND user_id = $2
        "#,
        id,
        user_id
    )
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

// Helper types and functions

#[derive(Debug)]
struct ComplianceRecord {
    id: Uuid,
    proof_type: String,
    verified: bool,
    expires_at: chrono::DateTime<chrono::Utc>,
    created_at: chrono::DateTime<chrono::Utc>,
}

struct TransactionData {
    id: Uuid,
    sender_commitment: Vec<u8>,
    recipient_commitment: Vec<u8>,
    amount_commitment: Option<Vec<u8>>,
    privacy_level: i32,
}

fn get_allowed_fields(tier: AccessTier, privacy_level: u8) -> Vec<DisclosureField> {
    match tier {
        AccessTier::PublicAuditor => {
            // Only aggregate data, no individual fields
            vec![]
        }
        AccessTier::Regulator => {
            if privacy_level <= 2 {
                vec![
                    DisclosureField::Timestamp,
                    DisclosureField::TransactionHash,
                ]
            } else {
                vec![DisclosureField::Timestamp]
            }
        }
        AccessTier::LawEnforcement => {
            // Full access with warrant
            vec![
                DisclosureField::TransactionAmount,
                DisclosureField::SenderAddress,
                DisclosureField::RecipientAddress,
                DisclosureField::Timestamp,
                DisclosureField::TransactionHash,
                DisclosureField::ProofDetails,
            ]
        }
        AccessTier::UserSelfDisclosure => {
            // User can disclose everything
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

async fn disclose_field(
    _state: &AppState,
    transaction: &sqlx::postgres::PgRow,
    field: &DisclosureField,
    _tier: AccessTier,
) -> Result<(Option<serde_json::Value>, String), ApiError> {
    // In production, would decrypt/reveal based on tier permissions
    // For now, return commitments
    let commitment = format!("commitment_{:?}", field);
    Ok((None, commitment))
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

async fn verify_age_requirement(
    _state: &AppState,
    _user_id: &Uuid,
    _min_age: u8,
) -> Result<bool, ApiError> {
    // In production, would verify against encrypted user data
    Ok(true)
}

async fn verify_accredited_investor(
    _state: &AppState,
    _user_id: &Uuid,
    _jurisdiction: &str,
) -> Result<bool, ApiError> {
    // In production, would verify against encrypted financial data
    Ok(true)
}

async fn verify_sanctions_compliance(
    _state: &AppState,
    _user_id: &Uuid,
) -> Result<bool, ApiError> {
    // In production, would check against sanctions lists using ZK proofs
    Ok(true)
}

async fn verify_source_of_funds(
    _state: &AppState,
    _user_id: &Uuid,
    _category: &str,
) -> Result<bool, ApiError> {
    // In production, would verify source of funds documentation
    Ok(true)
}

async fn verify_kyc_complete(
    _state: &AppState,
    _user_id: &Uuid,
    _provider: &str,
) -> Result<bool, ApiError> {
    // In production, would verify KYC status with provider
    Ok(true)
}

async fn verify_jurisdiction_compliance(
    _state: &AppState,
    _user_id: &Uuid,
    _jurisdiction: &str,
) -> Result<bool, ApiError> {
    // In production, would verify jurisdiction-specific requirements
    Ok(true)
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

        let user_fields = get_allowed_fields(AccessTier::UserSelfDisclosure, 5);
        assert_eq!(user_fields.len(), 6);
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
        assert_eq!(proof.len(), 64); // SHA-256 hex
    }
}
