//! Batch transaction handlers

use crate::db::TransactionRepository;
use crate::error::{Result, TransactionError};
use crate::models::*;
use crate::state::AppState;
use crate::handlers::metrics::*;
use axum::{
    extract::{Extension, Path},
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

/// Temporary user ID extraction
fn get_user_id() -> Uuid {
    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
}

/// Batch status response
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchStatusResponse {
    pub batch_id: Uuid,
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
    pub pending: usize,
    pub status: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Create a batch of transactions
pub async fn create_batch(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<BatchTransactionRequest>,
) -> Result<Json<BatchTransactionResponse>> {
    let user_id = get_user_id();
    let repo = TransactionRepository::new(&state.db);

    // Validate batch size
    if req.transactions.is_empty() {
        return Err(TransactionError::InvalidRequest(
            "Batch must contain at least one transaction".to_string(),
        ));
    }

    if req.transactions.len() > state.config.max_batch_size {
        return Err(TransactionError::InvalidRequest(format!(
            "Batch size {} exceeds maximum {}",
            req.transactions.len(),
            state.config.max_batch_size
        )));
    }

    let batch_id = Uuid::now_v7();
    let mut results: Vec<BatchTransactionResult> = Vec::new();
    let mut created = 0;
    let mut failed = 0;

    // Process each transaction
    for (index, tx_req) in req.transactions.iter().enumerate() {
        match repo.create(user_id, tx_req).await {
            Ok(tx) => {
                // Add to proof queue if requested
                if req.generate_proofs {
                    let mut queue = state.proof_queue.write().await;
                    queue.enqueue(tx.id);
                }

                results.push(BatchTransactionResult {
                    index,
                    success: true,
                    transaction_id: Some(tx.id),
                    error: None,
                });
                created += 1;

                TRANSACTIONS_CREATED.inc();
                TRANSACTIONS_BY_PRIVACY_LEVEL
                    .with_label_values(&[&tx.privacy_level.to_string()])
                    .inc();
            }
            Err(e) => {
                if req.atomic {
                    // Rollback all created transactions
                    // In a real implementation, this would use a database transaction
                    return Err(TransactionError::BatchFailed(format!(
                        "Atomic batch failed at index {}: {}",
                        index, e
                    )));
                }

                results.push(BatchTransactionResult {
                    index,
                    success: false,
                    transaction_id: None,
                    error: Some(e.to_string()),
                });
                failed += 1;

                ERRORS_TOTAL.with_label_values(&["batch_create"]).inc();
            }
        }
    }

    // Store batch info in Redis
    if let Ok(mut conn) = state.redis_conn().await {
        let batch_info = serde_json::json!({
            "batch_id": batch_id,
            "total": req.transactions.len(),
            "created": created,
            "failed": failed,
            "generate_proofs": req.generate_proofs,
            "created_at": chrono::Utc::now(),
        });

        let _: std::result::Result<(), _> = redis::cmd("SETEX")
            .arg(format!("batch:{}", batch_id))
            .arg(86400) // 24 hour TTL
            .arg(batch_info.to_string())
            .query_async(&mut conn)
            .await;
    }

    // Update metrics
    BATCH_OPERATIONS.inc();
    BATCH_SIZE.observe(req.transactions.len() as f64);

    if req.generate_proofs {
        let queue = state.proof_queue.read().await;
        PROOF_QUEUE_LENGTH.set(queue.queue_length() as f64);
        PROOFS_REQUESTED.inc_by(created as f64);
    }

    tracing::info!(
        batch_id = %batch_id,
        total = req.transactions.len(),
        created = created,
        failed = failed,
        "Batch created"
    );

    Ok(Json(BatchTransactionResponse {
        batch_id,
        total: req.transactions.len(),
        created,
        failed,
        results,
    }))
}

/// Get batch status
pub async fn get_batch_status(
    Extension(state): Extension<Arc<AppState>>,
    Path(batch_id): Path<Uuid>,
) -> Result<Json<BatchStatusResponse>> {
    // Get batch info from Redis
    let mut conn = state.redis_conn().await.map_err(|e| {
        TransactionError::Internal(format!("Redis connection failed: {}", e))
    })?;

    let batch_info: Option<String> = redis::cmd("GET")
        .arg(format!("batch:{}", batch_id))
        .query_async(&mut conn)
        .await
        .unwrap_or(None);

    match batch_info {
        Some(info) => {
            let data: serde_json::Value = serde_json::from_str(&info)
                .map_err(|e| TransactionError::Internal(e.to_string()))?;

            Ok(Json(BatchStatusResponse {
                batch_id,
                total: data["total"].as_u64().unwrap_or(0) as usize,
                completed: data["created"].as_u64().unwrap_or(0) as usize,
                failed: data["failed"].as_u64().unwrap_or(0) as usize,
                pending: 0, // TODO: Track pending
                status: "completed".to_string(),
                created_at: chrono::DateTime::parse_from_rfc3339(
                    data["created_at"].as_str().unwrap_or("1970-01-01T00:00:00Z"),
                )
                .unwrap_or_default()
                .with_timezone(&chrono::Utc),
            }))
        }
        None => Err(TransactionError::NotFound(format!(
            "Batch {} not found",
            batch_id
        ))),
    }
}
