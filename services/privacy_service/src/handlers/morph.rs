//! Privacy morph handlers

use crate::error::{PrivacyError, Result};
use crate::handlers::metrics::*;
use crate::models::*;
use crate::state::AppState;
use axum::{
    extract::{Extension, Path},
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

/// Morph transaction privacy level
pub async fn morph_privacy(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<MorphRequest>,
) -> Result<Json<MorphResponse>> {
    // Validate levels
    if req.current_level < 0 || req.current_level > 5 {
        return Err(PrivacyError::InvalidLevel(req.current_level));
    }
    if req.target_level < 0 || req.target_level > 5 {
        return Err(PrivacyError::InvalidLevel(req.target_level));
    }

    // Check if downgrade is allowed
    if req.target_level < req.current_level {
        if !state.config.privacy.allow_downgrade {
            return Err(PrivacyError::MorphNotAllowed(
                "Privacy level downgrade is not allowed".to_string(),
            ));
        }
        if state.config.privacy.require_force_for_downgrade && !req.force {
            return Err(PrivacyError::MorphNotAllowed(
                "Force flag required for privacy level downgrade".to_string(),
            ));
        }
    }

    // Determine if new proof is needed
    let requires_new_proof = req.target_level != req.current_level && req.current_proof.is_some();

    let morph_id = Uuid::now_v7();
    let mut new_proof = None;

    // If new proof required, queue generation
    if requires_new_proof {
        // For now, we'll simulate immediate proof generation
        // In production, this would queue the proof and return immediately
        new_proof = Some(generate_morph_proof(
            req.transaction_id,
            req.target_level,
        ));
    }

    // Record morph in database
    let _ = sqlx::query(
        r#"
        INSERT INTO privacy_morphs (id, transaction_id, previous_level, new_level, reason, morphed_at)
        VALUES ($1, $2, $3, $4, $5, NOW())
        "#,
    )
    .bind(morph_id)
    .bind(req.transaction_id)
    .bind(req.current_level)
    .bind(req.target_level)
    .bind(&req.reason)
    .execute(&state.db)
    .await;

    // Update metrics
    PRIVACY_MORPHS
        .with_label_values(&[
            &req.current_level.to_string(),
            &req.target_level.to_string(),
        ])
        .inc();

    tracing::info!(
        morph_id = %morph_id,
        transaction_id = %req.transaction_id,
        from = req.current_level,
        to = req.target_level,
        "Privacy level morphed"
    );

    Ok(Json(MorphResponse {
        morph_id,
        transaction_id: req.transaction_id,
        previous_level: req.current_level,
        new_level: req.target_level,
        new_proof,
        requires_new_proof,
        morphed_at: chrono::Utc::now(),
    }))
}

/// Get morph status
pub async fn get_morph_status(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<MorphResponse>> {
    let morph: Option<(Uuid, Uuid, i16, i16, Option<String>, chrono::DateTime<chrono::Utc>)> =
        sqlx::query_as(
            r#"
            SELECT id, transaction_id, previous_level, new_level, reason, morphed_at
            FROM privacy_morphs
            WHERE id = $1
            "#,
        )
        .bind(id)
        .fetch_optional(&state.db)
        .await
        .map_err(PrivacyError::Database)?;

    match morph {
        Some((morph_id, transaction_id, previous_level, new_level, _reason, morphed_at)) => {
            Ok(Json(MorphResponse {
                morph_id,
                transaction_id,
                previous_level,
                new_level,
                new_proof: None, // Would fetch from proof table
                requires_new_proof: previous_level != new_level,
                morphed_at,
            }))
        }
        None => Err(PrivacyError::Internal(format!("Morph {} not found", id))),
    }
}

/// Estimate morph operation
pub async fn estimate_morph(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<MorphEstimateRequest>,
) -> Result<Json<MorphEstimateResponse>> {
    let mut warnings: Vec<String> = Vec::new();

    // Validate levels
    if req.from_level < 0 || req.from_level > 5 || req.to_level < 0 || req.to_level > 5 {
        return Ok(Json(MorphEstimateResponse {
            can_morph: false,
            requires_new_proof: false,
            estimated_gas: 0,
            estimated_fee_usd: 0.0,
            estimated_time_ms: 0,
            warnings: vec!["Invalid privacy level".to_string()],
        }));
    }

    // Check downgrade
    let can_morph = if req.to_level < req.from_level {
        if !state.config.privacy.allow_downgrade {
            warnings.push("Privacy downgrade is disabled".to_string());
            false
        } else {
            warnings.push("Downgrade requires force flag".to_string());
            true
        }
    } else {
        true
    };

    // Calculate costs
    let requires_new_proof = req.from_level != req.to_level && req.has_proof;

    let proof_gas = if requires_new_proof {
        match req.to_level {
            0 => 0,
            1 | 2 => 50000,
            3 => 75000,
            4 => 150000,
            5 => 250000,
            _ => 100000,
        }
    } else {
        0
    };

    let base_gas = 21000u64;
    let total_gas = base_gas + proof_gas;
    let fee_usd = (total_gas as f64) * 30.0 * 1e-9 * 2000.0; // Assuming ETH

    let estimated_time_ms = if requires_new_proof {
        match req.to_level {
            0 => 0,
            1 | 2 => 500,
            3 => 1000,
            4 => 2500,
            5 => 5000,
            _ => 1500,
        }
    } else {
        100 // Just state update
    };

    Ok(Json(MorphEstimateResponse {
        can_morph,
        requires_new_proof,
        estimated_gas: total_gas,
        estimated_fee_usd: fee_usd,
        estimated_time_ms,
        warnings,
    }))
}

/// Generate proof for morph operation (placeholder)
fn generate_morph_proof(transaction_id: Uuid, target_level: i16) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(b"morph_proof_v1");
    hasher.update(transaction_id.as_bytes());
    hasher.update(target_level.to_le_bytes());
    hasher.update(chrono::Utc::now().timestamp().to_le_bytes());

    let hash = hasher.finalize();
    format!("morph_proof_{}", hex::encode(hash))
}
