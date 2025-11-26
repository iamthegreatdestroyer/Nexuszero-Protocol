//! Selective disclosure handlers

use crate::error::{PrivacyError, Result};
use crate::handlers::metrics::*;
use crate::models::*;
use crate::state::AppState;
use axum::{
    extract::{Extension, Path},
    Json,
};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use uuid::Uuid;

/// Valid disclosure fields
const VALID_FIELDS: &[&str] = &["sender", "recipient", "amount", "memo", "chain_id", "asset_id"];

/// Create a selective disclosure
pub async fn create_disclosure(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<DisclosureRequest>,
) -> Result<Json<DisclosureResponse>> {
    // Validate fields
    for field in &req.fields {
        if !VALID_FIELDS.contains(&field.as_str()) {
            return Err(PrivacyError::InvalidDisclosureFields(format!(
                "Invalid field: {}. Valid fields: {:?}",
                field, VALID_FIELDS
            )));
        }
    }

    if req.fields.is_empty() {
        return Err(PrivacyError::InvalidDisclosureFields(
            "At least one field must be disclosed".to_string(),
        ));
    }

    // Generate disclosure proof
    let disclosure_id = Uuid::now_v7();
    let proof = generate_disclosure_proof(
        disclosure_id,
        req.transaction_id,
        &req.fields,
        &req.recipient_id,
    );

    // Store in database
    sqlx::query(
        r#"
        INSERT INTO selective_disclosures (id, transaction_id, recipient_id, fields, purpose, proof, expires_at, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
        "#,
    )
    .bind(disclosure_id)
    .bind(req.transaction_id)
    .bind(&req.recipient_id)
    .bind(&req.fields)
    .bind(&req.purpose)
    .bind(&proof)
    .bind(req.expires_at)
    .execute(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    // Update metrics
    DISCLOSURES_CREATED.inc();

    tracing::info!(
        disclosure_id = %disclosure_id,
        transaction_id = %req.transaction_id,
        fields = ?req.fields,
        recipient = %req.recipient_id,
        "Selective disclosure created"
    );

    Ok(Json(DisclosureResponse {
        disclosure_id,
        transaction_id: req.transaction_id,
        fields: req.fields,
        proof,
        recipient_id: req.recipient_id,
        purpose: req.purpose,
        expires_at: req.expires_at,
        created_at: chrono::Utc::now(),
    }))
}

/// Get disclosure by ID
pub async fn get_disclosure(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<DisclosureResponse>> {
    let disclosure: Option<(
        Uuid,
        Uuid,
        String,
        Vec<String>,
        String,
        String,
        Option<chrono::DateTime<chrono::Utc>>,
        Option<chrono::DateTime<chrono::Utc>>,
        chrono::DateTime<chrono::Utc>,
    )> = sqlx::query_as(
        r#"
        SELECT id, transaction_id, recipient_id, fields, purpose, proof, expires_at, revoked_at, created_at
        FROM selective_disclosures
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    match disclosure {
        Some((
            disclosure_id,
            transaction_id,
            recipient_id,
            fields,
            purpose,
            proof,
            expires_at,
            revoked_at,
            created_at,
        )) => {
            // Check if revoked
            if revoked_at.is_some() {
                return Err(PrivacyError::DisclosureRevoked(id.to_string()));
            }

            // Check if expired
            if let Some(exp) = expires_at {
                if exp < chrono::Utc::now() {
                    return Err(PrivacyError::DisclosureExpired(id.to_string()));
                }
            }

            Ok(Json(DisclosureResponse {
                disclosure_id,
                transaction_id,
                fields,
                proof,
                recipient_id,
                purpose,
                expires_at,
                created_at,
            }))
        }
        None => Err(PrivacyError::DisclosureNotFound(id.to_string())),
    }
}

/// Verify a disclosure
pub async fn verify_disclosure(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<VerifyDisclosureRequest>,
) -> Result<Json<VerifyDisclosureResponse>> {
    // Get disclosure
    let disclosure: Option<(
        String,
        Uuid,
        Vec<String>,
        String,
        Option<chrono::DateTime<chrono::Utc>>,
        Option<chrono::DateTime<chrono::Utc>>,
    )> = sqlx::query_as(
        r#"
        SELECT proof, transaction_id, fields, recipient_id, expires_at, revoked_at
        FROM selective_disclosures
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    match disclosure {
        Some((stored_proof, transaction_id, fields, recipient_id, expires_at, revoked_at)) => {
            let mut errors: Vec<String> = Vec::new();

            // Check revocation
            if revoked_at.is_some() {
                errors.push("Disclosure has been revoked".to_string());
            }

            // Check expiration
            if let Some(exp) = expires_at {
                if exp < chrono::Utc::now() {
                    errors.push("Disclosure has expired".to_string());
                }
            }

            // Verify proof matches
            let proof_valid = req.proof == stored_proof;
            if !proof_valid {
                errors.push("Proof mismatch".to_string());
            }

            // Verify recipient matches
            if req.verifier_id != recipient_id {
                errors.push("Verifier is not the authorized recipient".to_string());
            }

            let valid = errors.is_empty();

            // Update metrics
            DISCLOSURES_VERIFIED.inc();

            Ok(Json(VerifyDisclosureResponse {
                valid,
                disclosure_id: id,
                transaction_id,
                disclosed_fields: if valid { fields } else { vec![] },
                errors,
                verified_at: chrono::Utc::now(),
            }))
        }
        None => Err(PrivacyError::DisclosureNotFound(id.to_string())),
    }
}

/// Revoke a disclosure
pub async fn revoke_disclosure(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>> {
    let result = sqlx::query(
        r#"
        UPDATE selective_disclosures
        SET revoked_at = NOW()
        WHERE id = $1 AND revoked_at IS NULL
        "#,
    )
    .bind(id)
    .execute(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    if result.rows_affected() > 0 {
        tracing::info!(disclosure_id = %id, "Disclosure revoked");
        Ok(Json(serde_json::json!({
            "disclosure_id": id,
            "revoked": true,
            "revoked_at": chrono::Utc::now(),
        })))
    } else {
        Err(PrivacyError::DisclosureNotFound(id.to_string()))
    }
}

/// Verify disclosure request
#[derive(Debug, serde::Deserialize)]
pub struct VerifyDisclosureRequest {
    /// Proof to verify
    pub proof: String,

    /// ID of verifier
    pub verifier_id: String,
}

/// Verify disclosure response
#[derive(Debug, serde::Serialize)]
pub struct VerifyDisclosureResponse {
    /// Is valid
    pub valid: bool,

    /// Disclosure ID
    pub disclosure_id: Uuid,

    /// Transaction ID
    pub transaction_id: Uuid,

    /// Disclosed fields (only if valid)
    pub disclosed_fields: Vec<String>,

    /// Errors if invalid
    pub errors: Vec<String>,

    /// Verification timestamp
    pub verified_at: chrono::DateTime<chrono::Utc>,
}

/// Generate a disclosure proof (placeholder - would use actual ZK derivation)
fn generate_disclosure_proof(
    disclosure_id: Uuid,
    transaction_id: Uuid,
    fields: &[String],
    recipient_id: &str,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"disclosure_v1");
    hasher.update(disclosure_id.as_bytes());
    hasher.update(transaction_id.as_bytes());
    for field in fields {
        hasher.update(field.as_bytes());
    }
    hasher.update(recipient_id.as_bytes());
    hasher.update(chrono::Utc::now().timestamp().to_le_bytes());

    let hash = hasher.finalize();
    format!("disclosure_{}", hex::encode(hash))
}
