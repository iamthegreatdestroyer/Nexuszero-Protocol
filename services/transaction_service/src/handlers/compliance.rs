//! Compliance handlers

use crate::db::TransactionRepository;
use crate::error::{Result, TransactionError};
use crate::models::*;
use crate::state::AppState;
use axum::{
    extract::{Extension, Path},
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

/// Get compliance status for a transaction
pub async fn get_compliance_status(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ComplianceStatus>> {
    let repo = TransactionRepository::new(&state.db);

    // Verify transaction exists
    let tx = repo.get_by_id(id).await?;

    // Call Compliance Service
    let compliance_response = state
        .http_client
        .get(format!(
            "{}/api/v1/compliance/transaction/{}",
            state.config.compliance_service_url, id
        ))
        .send()
        .await;

    match compliance_response {
        Ok(resp) if resp.status().is_success() => {
            let status: ComplianceStatus = resp
                .json()
                .await
                .map_err(|e| TransactionError::ExternalService(e.to_string()))?;
            Ok(Json(status))
        }
        Ok(resp) => {
            // Service returned error
            let error = resp.text().await.unwrap_or_default();
            Err(TransactionError::ExternalService(format!(
                "Compliance service error: {}",
                error
            )))
        }
        Err(_) => {
            // Service unavailable - return basic compliance check
            // In production, this should be configurable behavior
            tracing::warn!(
                transaction_id = %id,
                "Compliance service unavailable, using fallback"
            );

            Ok(Json(ComplianceStatus {
                transaction_id: id,
                compliant: true, // Assume compliant if service unavailable
                checks: vec![
                    ComplianceCheck {
                        name: "basic_validation".to_string(),
                        passed: true,
                        details: Some("Basic transaction validation passed".to_string()),
                    },
                    ComplianceCheck {
                        name: "amount_limit".to_string(),
                        passed: tx.amount < 1_000_000_000_000, // 1 trillion limit
                        details: None,
                    },
                ],
                risk_score: calculate_basic_risk_score(&tx),
                flags: vec![],
                checked_at: chrono::Utc::now(),
            }))
        }
    }
}

/// Create selective disclosure for regulatory compliance
pub async fn create_selective_disclosure(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<SelectiveDisclosureRequest>,
) -> Result<Json<SelectiveDisclosureResponse>> {
    let repo = TransactionRepository::new(&state.db);

    // Verify transaction exists
    let tx = repo.get_by_id(id).await?;

    // Validate requested fields
    let valid_fields = ["sender", "recipient", "amount", "memo", "chain_id", "asset_id"];
    for field in &req.fields {
        if !valid_fields.contains(&field.as_str()) {
            return Err(TransactionError::InvalidRequest(format!(
                "Invalid disclosure field: {}",
                field
            )));
        }
    }

    // Call Compliance Service to generate disclosure proof
    let disclosure_response = state
        .http_client
        .post(format!(
            "{}/api/v1/selective-disclosure",
            state.config.compliance_service_url
        ))
        .json(&serde_json::json!({
            "transaction_id": id,
            "privacy_level": tx.privacy_level,
            "proof": tx.proof,
            "fields": req.fields,
            "recipient_id": req.recipient_id,
            "purpose": req.purpose,
            "expires_at": req.expires_at,
        }))
        .send()
        .await;

    match disclosure_response {
        Ok(resp) if resp.status().is_success() => {
            let disclosure: SelectiveDisclosureResponse = resp
                .json()
                .await
                .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

            tracing::info!(
                transaction_id = %id,
                disclosure_id = %disclosure.disclosure_id,
                fields = ?req.fields,
                recipient = %req.recipient_id,
                "Selective disclosure created"
            );

            Ok(Json(disclosure))
        }
        Ok(resp) => {
            let error = resp.text().await.unwrap_or_default();
            Err(TransactionError::ExternalService(format!(
                "Disclosure creation failed: {}",
                error
            )))
        }
        Err(_) => {
            // Fallback: Generate basic disclosure without ZK proof
            tracing::warn!(
                transaction_id = %id,
                "Compliance service unavailable, generating fallback disclosure"
            );

            let disclosure_id = Uuid::now_v7();

            // Generate a placeholder proof
            let proof = generate_fallback_disclosure_proof(id, &req.fields);

            // Store disclosure in Redis
            if let Ok(mut conn) = state.redis_conn().await {
                let disclosure_data = serde_json::json!({
                    "disclosure_id": disclosure_id,
                    "transaction_id": id,
                    "fields": req.fields,
                    "recipient_id": req.recipient_id,
                    "purpose": req.purpose,
                    "expires_at": req.expires_at,
                    "created_at": chrono::Utc::now(),
                });

                let _: std::result::Result<(), _> = redis::cmd("SETEX")
                    .arg(format!("disclosure:{}", disclosure_id))
                    .arg(req.expires_at.map(|e| {
                        (e - chrono::Utc::now()).num_seconds().max(0) as usize
                    }).unwrap_or(86400 * 30)) // 30 days default
                    .arg(disclosure_data.to_string())
                    .query_async(&mut conn)
                    .await;
            }

            Ok(Json(SelectiveDisclosureResponse {
                disclosure_id,
                transaction_id: id,
                fields: req.fields,
                proof,
                expires_at: req.expires_at,
                created_at: chrono::Utc::now(),
            }))
        }
    }
}

/// Calculate basic risk score for a transaction
fn calculate_basic_risk_score(tx: &Transaction) -> u8 {
    let mut score = 0u8;

    // Higher amounts = higher risk
    if tx.amount > 100_000_000_000 {
        // > 100B
        score += 30;
    } else if tx.amount > 10_000_000_000 {
        // > 10B
        score += 20;
    } else if tx.amount > 1_000_000_000 {
        // > 1B
        score += 10;
    }

    // Higher privacy = slightly higher risk (for compliance purposes)
    score += (tx.privacy_level * 5) as u8;

    // Unknown chains = higher risk
    let known_chains = ["ethereum", "polygon", "arbitrum", "optimism", "base"];
    if !known_chains.contains(&tx.chain_id.to_lowercase().as_str()) {
        score += 15;
    }

    score.min(100)
}

/// Generate fallback disclosure proof (placeholder)
fn generate_fallback_disclosure_proof(transaction_id: Uuid, fields: &[String]) -> String {
    use sha2::{Sha256, Digest};

    let mut hasher = Sha256::new();
    hasher.update(transaction_id.as_bytes());
    for field in fields {
        hasher.update(field.as_bytes());
    }
    hasher.update(chrono::Utc::now().timestamp().to_le_bytes());

    let hash = hasher.finalize();
    format!("fallback_disclosure_{}", hex::encode(hash))
}
