//! Transaction CRUD handlers

use crate::db::TransactionRepository;
use crate::error::{Result, TransactionError};
use crate::models::*;
use crate::state::AppState;
use crate::handlers::metrics::*;
use axum::{
    extract::{Extension, Path, Query},
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

/// Temporary user ID extraction (should come from auth middleware)
fn get_user_id() -> Uuid {
    // In production, this would come from JWT claims
    Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap()
}

/// Create a new transaction
pub async fn create_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<CreateTransactionRequest>,
) -> Result<Json<CreateTransactionResponse>> {
    let user_id = get_user_id();
    let repo = TransactionRepository::new(&state.db);

    // Validate amount
    if req.amount <= 0 {
        return Err(TransactionError::InvalidRequest(
            "Amount must be positive".to_string(),
        ));
    }

    // Create transaction
    let tx = repo.create(user_id, &req).await?;

    // Update metrics
    TRANSACTIONS_CREATED.inc();
    TRANSACTIONS_BY_STATUS
        .with_label_values(&["pending"])
        .inc();
    TRANSACTIONS_BY_PRIVACY_LEVEL
        .with_label_values(&[&tx.privacy_level.to_string()])
        .inc();

    // Auto-generate proof if requested
    let proof_id = if req.auto_generate_proof {
        let mut queue = state.proof_queue.write().await;
        queue.enqueue(tx.id);
        PROOFS_REQUESTED.inc();
        PROOF_QUEUE_LENGTH.set(queue.queue_length() as f64);
        Some(Uuid::now_v7())
    } else {
        None
    };

    tracing::info!(
        transaction_id = %tx.id,
        privacy_level = tx.privacy_level,
        "Transaction created"
    );

    Ok(Json(CreateTransactionResponse {
        id: tx.id,
        status: tx.status,
        privacy_level: tx.privacy_level,
        proof_id,
        created_at: tx.created_at,
    }))
}

/// List transactions
pub async fn list_transactions(
    Extension(state): Extension<Arc<AppState>>,
    Query(query): Query<ListTransactionsQuery>,
) -> Result<Json<TransactionListResponse>> {
    let user_id = get_user_id();
    let repo = TransactionRepository::new(&state.db);

    let response = repo.list(user_id, &query).await?;

    Ok(Json(response))
}

/// Get a specific transaction
pub async fn get_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Transaction>> {
    let user_id = get_user_id();
    let repo = TransactionRepository::new(&state.db);

    let tx = repo.get_by_id_for_user(id, user_id).await?;

    Ok(Json(tx))
}

/// Update a transaction
pub async fn update_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<UpdateTransactionRequest>,
) -> Result<Json<Transaction>> {
    let user_id = get_user_id();
    let repo = TransactionRepository::new(&state.db);

    let tx = repo.update(id, user_id, &req).await?;

    tracing::info!(transaction_id = %id, "Transaction updated");

    Ok(Json(tx))
}

/// Cancel a transaction
pub async fn cancel_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Transaction>> {
    let user_id = get_user_id();
    let repo = TransactionRepository::new(&state.db);

    let tx = repo.cancel(id, user_id).await?;

    TRANSACTIONS_BY_STATUS
        .with_label_values(&["cancelled"])
        .inc();

    tracing::info!(transaction_id = %id, "Transaction cancelled");

    Ok(Json(tx))
}

/// Submit transaction to blockchain
pub async fn submit_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Transaction>> {
    let repo = TransactionRepository::new(&state.db);

    // Verify proof is ready
    let tx = repo.get_by_id(id).await?;

    if tx.proof.is_none() {
        return Err(TransactionError::ProofNotReady(id.to_string()));
    }

    // Update status to submitted
    let tx = repo.update_status(id, TransactionStatus::Submitted).await?;

    TRANSACTIONS_BY_STATUS
        .with_label_values(&["submitted"])
        .inc();

    // TODO: Actually submit to blockchain via chain adapter

    tracing::info!(transaction_id = %id, "Transaction submitted");

    Ok(Json(tx))
}

/// Finalize a confirmed transaction
pub async fn finalize_transaction(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<Transaction>> {
    let repo = TransactionRepository::new(&state.db);

    let tx = repo.get_by_id(id).await?;

    if tx.status != TransactionStatus::Confirmed {
        return Err(TransactionError::InvalidStatusTransition {
            from: tx.status,
            to: TransactionStatus::Finalized,
        });
    }

    let tx = repo.update_status(id, TransactionStatus::Finalized).await?;

    TRANSACTIONS_BY_STATUS
        .with_label_values(&["finalized"])
        .inc();

    tracing::info!(transaction_id = %id, "Transaction finalized");

    Ok(Json(tx))
}

/// Get proof for a transaction
pub async fn get_proof(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ProofResponse>> {
    let repo = TransactionRepository::new(&state.db);

    let tx = repo.get_by_id(id).await?;

    let has_proof = tx.proof.is_some();
    let status = if has_proof {
        ProofStatus::Completed
    } else if tx.status == TransactionStatus::ProofGenerating {
        ProofStatus::Generating
    } else if tx.status == TransactionStatus::Failed {
        ProofStatus::Failed
    } else {
        ProofStatus::Queued
    };

    let completed_at = if has_proof {
        Some(tx.updated_at)
    } else {
        None
    };

    Ok(Json(ProofResponse {
        proof_id: tx.proof_id.unwrap_or_else(Uuid::nil),
        transaction_id: tx.id,
        status,
        proof: tx.proof,
        verification_key: None, // TODO: Get from proof service
        public_inputs: None,
        generation_time_ms: None,
        error: tx.error_message,
        created_at: tx.created_at,
        completed_at,
    }))
}

/// Request proof generation for a transaction
pub async fn request_proof(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<ProofRequest>,
) -> Result<Json<ProofResponse>> {
    let repo = TransactionRepository::new(&state.db);

    let tx = repo.get_by_id(id).await?;

    // Only allow proof request for pending transactions
    if tx.status != TransactionStatus::Pending {
        return Err(TransactionError::InvalidRequest(
            "Proof can only be requested for pending transactions".to_string(),
        ));
    }

    // Add to proof queue
    let proof_id = Uuid::now_v7();
    {
        let mut queue = state.proof_queue.write().await;
        queue.enqueue(id);
        PROOFS_REQUESTED.inc();
        PROOF_QUEUE_LENGTH.set(queue.queue_length() as f64);
    }

    tracing::info!(
        transaction_id = %id,
        proof_id = %proof_id,
        priority = ?req.priority,
        "Proof generation requested"
    );

    Ok(Json(ProofResponse {
        proof_id,
        transaction_id: id,
        status: ProofStatus::Queued,
        proof: None,
        verification_key: None,
        public_inputs: None,
        generation_time_ms: None,
        error: None,
        created_at: chrono::Utc::now(),
        completed_at: None,
    }))
}
