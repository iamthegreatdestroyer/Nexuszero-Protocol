//! Privacy level handlers

use crate::db::TransactionRepository;
use crate::error::{Result, TransactionError};
use crate::models::*;
use crate::state::AppState;
use crate::handlers::metrics::*;
use axum::{
    extract::{Extension, Path},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;

/// Privacy level info response
#[derive(Debug, Serialize, Deserialize)]
pub struct PrivacyLevelInfo {
    pub transaction_id: Uuid,
    pub current_level: i16,
    pub level_name: String,
    pub description: String,
    pub shielded_fields: Vec<String>,
    pub can_morph: bool,
    pub available_levels: Vec<AvailableLevel>,
}

/// Available privacy level
#[derive(Debug, Serialize, Deserialize)]
pub struct AvailableLevel {
    pub level: i16,
    pub name: String,
    pub requires_new_proof: bool,
}

/// Get privacy level for a transaction
pub async fn get_privacy_level(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<PrivacyLevelInfo>> {
    let repo = TransactionRepository::new(&state.db);
    let tx = repo.get_by_id(id).await?;

    let (level_name, description, shielded_fields) = get_level_info(tx.privacy_level);

    let can_morph = matches!(
        tx.status,
        TransactionStatus::Pending | TransactionStatus::ProofReady
    );

    let available_levels: Vec<AvailableLevel> = (0..=5)
        .map(|level| {
            let (name, _, _) = get_level_info(level);
            AvailableLevel {
                level,
                name,
                requires_new_proof: level != tx.privacy_level && tx.proof.is_some(),
            }
        })
        .collect();

    Ok(Json(PrivacyLevelInfo {
        transaction_id: id,
        current_level: tx.privacy_level,
        level_name,
        description,
        shielded_fields,
        can_morph,
        available_levels,
    }))
}

/// Update privacy level for a transaction
pub async fn update_privacy_level(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<PrivacyMorphRequest>,
) -> Result<Json<PrivacyMorphResponse>> {
    let repo = TransactionRepository::new(&state.db);

    let tx = repo.get_by_id(id).await?;
    let previous_level = tx.privacy_level;

    // Check if we can morph
    if !matches!(
        tx.status,
        TransactionStatus::Pending | TransactionStatus::ProofReady
    ) {
        return Err(TransactionError::PrivacyMorphFailed(
            "Cannot change privacy level in current state".to_string(),
        ));
    }

    // Validate target level
    if req.target_level < 0 || req.target_level > 5 {
        return Err(TransactionError::InvalidPrivacyLevel(req.target_level));
    }

    // Check if lowering privacy level without force
    if req.target_level < previous_level && !req.force {
        return Err(TransactionError::PrivacyMorphFailed(
            "Cannot lower privacy level without force flag".to_string(),
        ));
    }

    // Perform the update
    let updated = repo.update_privacy_level(id, req.target_level).await?;

    // Track in metrics
    PRIVACY_MORPHS
        .with_label_values(&[
            &previous_level.to_string(),
            &req.target_level.to_string(),
        ])
        .inc();

    // Determine if new proof needed
    let proof_id = if previous_level != req.target_level && tx.proof.is_some() {
        // Enqueue for new proof generation
        let mut queue = state.proof_queue.write().await;
        queue.enqueue(id);
        PROOFS_REQUESTED.inc();
        PROOF_QUEUE_LENGTH.set(queue.queue_length() as f64);
        Some(Uuid::now_v7())
    } else {
        None
    };

    tracing::info!(
        transaction_id = %id,
        previous_level = previous_level,
        new_level = req.target_level,
        reason = ?req.reason,
        "Privacy level morphed"
    );

    Ok(Json(PrivacyMorphResponse {
        transaction_id: id,
        previous_level,
        new_level: updated.privacy_level,
        proof_id,
        morphed_at: chrono::Utc::now(),
    }))
}

/// Morph privacy level (alias for update with more context)
pub async fn morph_privacy(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(req): Json<PrivacyMorphRequest>,
) -> Result<Json<PrivacyMorphResponse>> {
    update_privacy_level(Extension(state), Path(id), Json(req)).await
}

/// Get human-readable info for a privacy level
fn get_level_info(level: i16) -> (String, String, Vec<String>) {
    match level {
        0 => (
            "Transparent".to_string(),
            "All transaction data is publicly visible".to_string(),
            vec![],
        ),
        1 => (
            "Sender-Shielded".to_string(),
            "Sender address is hidden, recipient and amount visible".to_string(),
            vec!["sender".to_string()],
        ),
        2 => (
            "Recipient-Shielded".to_string(),
            "Recipient address is hidden, sender and amount visible".to_string(),
            vec!["recipient".to_string()],
        ),
        3 => (
            "Amount-Shielded".to_string(),
            "Transaction amount is hidden, parties visible".to_string(),
            vec!["amount".to_string()],
        ),
        4 => (
            "Full Privacy".to_string(),
            "Sender, recipient, and amount hidden, only memo visible".to_string(),
            vec![
                "sender".to_string(),
                "recipient".to_string(),
                "amount".to_string(),
            ],
        ),
        5 => (
            "Maximum".to_string(),
            "All transaction data is fully shielded".to_string(),
            vec![
                "sender".to_string(),
                "recipient".to_string(),
                "amount".to_string(),
                "memo".to_string(),
                "metadata".to_string(),
            ],
        ),
        _ => (
            "Unknown".to_string(),
            "Unknown privacy level".to_string(),
            vec![],
        ),
    }
}
