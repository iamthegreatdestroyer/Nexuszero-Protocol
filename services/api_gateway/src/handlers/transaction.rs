//! Transaction Handlers
//!
//! Handles privacy-preserving transaction operations

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

/// Privacy level enum (0-5)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PrivacyLevel {
    Transparent = 0,
    Pseudonymous = 1,
    Confidential = 2,
    Private = 3,
    Anonymous = 4,
    Sovereign = 5,
}

impl From<u8> for PrivacyLevel {
    fn from(level: u8) -> Self {
        match level {
            0 => PrivacyLevel::Transparent,
            1 => PrivacyLevel::Pseudonymous,
            2 => PrivacyLevel::Confidential,
            3 => PrivacyLevel::Private,
            4 => PrivacyLevel::Anonymous,
            5 => PrivacyLevel::Sovereign,
            _ => PrivacyLevel::Private, // Default
        }
    }
}

impl From<PrivacyLevel> for u8 {
    fn from(level: PrivacyLevel) -> Self {
        level as u8
    }
}

/// Transaction status
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransactionStatus {
    Created,
    PrivacySelected,
    ProofGenerating,
    ProofGenerated,
    Submitted,
    Confirmed,
    Failed,
}

/// Create transaction request
#[derive(Debug, Deserialize, Validate)]
pub struct CreateTransactionRequest {
    /// Recipient address/commitment
    #[validate(length(min = 1, max = 255))]
    pub recipient: String,

    /// Transaction amount (in smallest unit)
    pub amount: u64,

    /// Target blockchain
    #[validate(length(min = 1, max = 50))]
    pub chain: String,

    /// Requested privacy level
    pub privacy_level: Option<u8>,

    /// Additional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Transaction response
#[derive(Debug, Serialize)]
pub struct TransactionResponse {
    pub id: String,
    pub sender: String,
    pub recipient: String,
    pub amount_encrypted: bool,
    pub privacy_level: PrivacyLevel,
    pub status: TransactionStatus,
    pub chain: String,
    pub proof_id: Option<String>,
    pub chain_tx_hash: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Transaction list query parameters
#[derive(Debug, Deserialize)]
pub struct ListTransactionsQuery {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub status: Option<String>,
    pub chain: Option<String>,
    pub privacy_level: Option<u8>,
}

/// Paginated transaction list response
#[derive(Debug, Serialize)]
pub struct TransactionListResponse {
    pub transactions: Vec<TransactionResponse>,
    pub total: i64,
    pub page: u32,
    pub limit: u32,
    pub has_more: bool,
}

/// Proof response
#[derive(Debug, Serialize)]
pub struct ProofResponse {
    pub proof_id: String,
    pub transaction_id: String,
    pub proof_type: String,
    pub privacy_level: PrivacyLevel,
    pub verified: bool,
    pub generation_time_ms: Option<i64>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Transaction status response
#[derive(Debug, Serialize)]
pub struct StatusResponse {
    pub transaction_id: String,
    pub status: TransactionStatus,
    pub chain_tx_hash: Option<String>,
    pub block_number: Option<i64>,
    pub confirmations: Option<i64>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// List transactions handler
pub async fn list_transactions(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Query(params): Query<ListTransactionsQuery>,
) -> ApiResult<Json<TransactionListResponse>> {
    let page = params.page.unwrap_or(1).max(1);
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = ((page - 1) * limit) as i64;

    // Build query
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let transactions = sqlx::query_as::<_, TransactionRecord>(
        r#"
        SELECT 
            id, user_id, sender_commitment, recipient_commitment,
            privacy_level, status, chain, proof_id, chain_tx_hash,
            created_at, updated_at
        FROM transactions
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        "#
    )
    .bind(user_id)
    .bind(limit as i64)
    .bind(offset)
    .fetch_all(&state.db)
    .await?;

    let total: Option<i64> = sqlx::query_scalar(
        "SELECT COUNT(*) FROM transactions WHERE user_id = $1"
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;
    let total = total.unwrap_or(0);

    let transaction_responses: Vec<TransactionResponse> = transactions
        .into_iter()
        .map(|t| TransactionResponse {
            id: t.id.to_string(),
            sender: hex::encode(&t.sender_commitment),
            recipient: hex::encode(&t.recipient_commitment),
            amount_encrypted: t.privacy_level >= 2,
            privacy_level: PrivacyLevel::from(t.privacy_level as u8),
            status: parse_status(&t.status),
            chain: t.chain,
            proof_id: t.proof_id.map(|id| id.to_string()),
            chain_tx_hash: t.chain_tx_hash.map(|h| hex::encode(&h)),
            created_at: t.created_at,
            updated_at: t.updated_at,
        })
        .collect();

    Ok(Json(TransactionListResponse {
        transactions: transaction_responses,
        total,
        page,
        limit,
        has_more: (page * limit) < total as u32,
    }))
}

/// Create transaction handler
pub async fn create_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<CreateTransactionRequest>,
) -> ApiResult<Json<TransactionResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    // Determine privacy level
    let privacy_level = payload.privacy_level.unwrap_or(
        user.claims.custom.default_privacy_level.unwrap_or(3),
    );

    if privacy_level > 5 {
        return Err(ApiError::UnsupportedPrivacyLevel(privacy_level));
    }

    // Create sender and recipient commitments
    // In production, these would be proper cryptographic commitments
    let sender_commitment = create_commitment(&user.user_id);
    let recipient_commitment = create_commitment(&payload.recipient);

    let tx_id = Uuid::new_v4();
    let now = chrono::Utc::now();

    // Insert transaction
    sqlx::query(
        r#"
        INSERT INTO transactions (
            id, user_id, sender_commitment, recipient_commitment,
            amount_commitment, privacy_level, status, chain, metadata,
            created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $10)
        "#
    )
    .bind(tx_id)
    .bind(user_id)
    .bind(&sender_commitment)
    .bind(&recipient_commitment)
    .bind(&payload.amount.to_le_bytes().to_vec())
    .bind(privacy_level as i32)
    .bind("created")
    .bind(&payload.chain)
    .bind(payload.metadata.clone().unwrap_or(serde_json::json!({})))
    .bind(now)
    .execute(&state.db)
    .await?;

    // Request proof generation from privacy service
    if privacy_level > 0 {
        let _ = request_proof_generation(&state, &tx_id, privacy_level).await;
    }

    tracing::info!(
        transaction_id = %tx_id,
        user_id = %user_id,
        privacy_level = privacy_level,
        chain = %payload.chain,
        "Transaction created"
    );

    // Record metrics
    crate::handlers::metrics::record_transaction(privacy_level);

    Ok(Json(TransactionResponse {
        id: tx_id.to_string(),
        sender: hex::encode(&sender_commitment),
        recipient: hex::encode(&recipient_commitment),
        amount_encrypted: privacy_level >= 2,
        privacy_level: PrivacyLevel::from(privacy_level),
        status: TransactionStatus::Created,
        chain: payload.chain,
        proof_id: None,
        chain_tx_hash: None,
        created_at: now,
        updated_at: now,
    }))
}

/// Get transaction by ID handler
pub async fn get_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<TransactionResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let transaction = sqlx::query_as::<_, TransactionRecord>(
        r#"
        SELECT 
            id, user_id, sender_commitment, recipient_commitment,
            privacy_level, status, chain, proof_id, chain_tx_hash,
            created_at, updated_at
        FROM transactions
        WHERE id = $1 AND user_id = $2
        "#
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Transaction not found".to_string()))?;

    Ok(Json(TransactionResponse {
        id: transaction.id.to_string(),
        sender: hex::encode(&transaction.sender_commitment),
        recipient: hex::encode(&transaction.recipient_commitment),
        amount_encrypted: transaction.privacy_level >= 2,
        privacy_level: PrivacyLevel::from(transaction.privacy_level as u8),
        status: parse_status(&transaction.status),
        chain: transaction.chain,
        proof_id: transaction.proof_id.map(|id| id.to_string()),
        chain_tx_hash: transaction.chain_tx_hash.map(|h| hex::encode(&h)),
        created_at: transaction.created_at,
        updated_at: transaction.updated_at,
    }))
}

/// Get proof for transaction handler
pub async fn get_proof(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<ProofResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    // Verify transaction belongs to user
    let transaction: Option<Option<Uuid>> = sqlx::query_scalar(
        "SELECT proof_id FROM transactions WHERE id = $1 AND user_id = $2"
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Transaction not found".to_string()))?;

    let proof_id = transaction
        .ok_or(ApiError::NotFound("Proof not generated yet".to_string()))?;

    let proof = sqlx::query_as::<_, ProofRecord>(
        r#"
        SELECT id, transaction_id, proof_type, privacy_level, verified, generation_time_ms, created_at
        FROM proofs
        WHERE id = $1
        "#
    )
    .bind(proof_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Proof not found".to_string()))?;

    Ok(Json(ProofResponse {
        proof_id: proof.id.to_string(),
        transaction_id: proof.transaction_id.to_string(),
        proof_type: proof.proof_type,
        privacy_level: PrivacyLevel::from(proof.privacy_level as u8),
        verified: proof.verified,
        generation_time_ms: proof.generation_time_ms.map(|t| t as i64),
        created_at: proof.created_at,
    }))
}

/// Get transaction status handler
pub async fn get_status(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<StatusResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let transaction = sqlx::query_as::<_, TransactionStatusRecord>(
        r#"
        SELECT id, status, chain_tx_hash, updated_at
        FROM transactions
        WHERE id = $1 AND user_id = $2
        "#
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Transaction not found".to_string()))?;

    // If submitted, check chain for confirmations
    let (block_number, confirmations) = if transaction.status == "confirmed" {
        // Query chain connector for confirmation details
        (Some(12345i64), Some(6i64)) // Placeholder
    } else {
        (None, None)
    };

    Ok(Json(StatusResponse {
        transaction_id: transaction.id.to_string(),
        status: parse_status(&transaction.status),
        chain_tx_hash: transaction.chain_tx_hash.map(|h| hex::encode(&h)),
        block_number,
        confirmations,
        updated_at: transaction.updated_at,
    }))
}

// Helper types and functions

#[derive(Debug, sqlx::FromRow)]
struct TransactionRecord {
    id: Uuid,
    user_id: Uuid,
    sender_commitment: Vec<u8>,
    recipient_commitment: Vec<u8>,
    privacy_level: i32,
    status: String,
    chain: String,
    proof_id: Option<Uuid>,
    chain_tx_hash: Option<Vec<u8>>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, sqlx::FromRow)]
struct ProofRecord {
    id: Uuid,
    transaction_id: Uuid,
    proof_type: String,
    privacy_level: i32,
    verified: bool,
    generation_time_ms: Option<i32>,
    created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, sqlx::FromRow)]
struct TransactionStatusRecord {
    id: Uuid,
    status: String,
    chain_tx_hash: Option<Vec<u8>>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

fn parse_status(status: &str) -> TransactionStatus {
    match status {
        "created" => TransactionStatus::Created,
        "privacy_selected" => TransactionStatus::PrivacySelected,
        "proof_generating" => TransactionStatus::ProofGenerating,
        "proof_generated" => TransactionStatus::ProofGenerated,
        "submitted" => TransactionStatus::Submitted,
        "confirmed" => TransactionStatus::Confirmed,
        "failed" => TransactionStatus::Failed,
        _ => TransactionStatus::Created,
    }
}

fn create_commitment(data: &str) -> Vec<u8> {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hasher.finalize().to_vec()
}

async fn request_proof_generation(
    state: &AppState,
    tx_id: &Uuid,
    privacy_level: u8,
) -> Result<(), ApiError> {
    // Request proof generation from privacy service
    let url = format!(
        "{}/internal/proofs/generate",
        state.config.services.privacy_service
    );

    let response = state
        .http_client
        .post(&url)
        .json(&serde_json::json!({
            "transaction_id": tx_id.to_string(),
            "privacy_level": privacy_level
        }))
        .send()
        .await;

    match response {
        Ok(resp) if resp.status().is_success() => {
            tracing::debug!(transaction_id = %tx_id, "Proof generation requested");
            Ok(())
        }
        Ok(resp) => {
            tracing::warn!(
                transaction_id = %tx_id,
                status = %resp.status(),
                "Proof generation request failed"
            );
            Ok(()) // Don't fail the transaction creation
        }
        Err(e) => {
            tracing::warn!(
                transaction_id = %tx_id,
                error = %e,
                "Failed to request proof generation"
            );
            Ok(()) // Don't fail the transaction creation
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_level_conversion() {
        assert_eq!(PrivacyLevel::from(0u8), PrivacyLevel::Transparent);
        assert_eq!(PrivacyLevel::from(3u8), PrivacyLevel::Private);
        assert_eq!(PrivacyLevel::from(5u8), PrivacyLevel::Sovereign);
        assert_eq!(PrivacyLevel::from(99u8), PrivacyLevel::Private); // Default
    }

    #[test]
    fn test_create_commitment() {
        let commitment1 = create_commitment("user1");
        let commitment2 = create_commitment("user2");
        let commitment1_again = create_commitment("user1");

        assert_eq!(commitment1.len(), 32); // SHA-256
        assert_ne!(commitment1, commitment2);
        assert_eq!(commitment1, commitment1_again);
    }

    #[test]
    fn test_parse_status() {
        assert!(matches!(parse_status("created"), TransactionStatus::Created));
        assert!(matches!(parse_status("confirmed"), TransactionStatus::Confirmed));
        assert!(matches!(parse_status("unknown"), TransactionStatus::Created));
    }
}

