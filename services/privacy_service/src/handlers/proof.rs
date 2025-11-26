//! Proof generation and verification handlers

use crate::error::{PrivacyError, Result};
use crate::handlers::metrics::*;
use crate::models::*;
use crate::state::{AppState, ProofJob};
use axum::{
    extract::{Extension, Path},
    Json,
};
use std::sync::Arc;
use uuid::Uuid;

/// Generate a ZK proof
pub async fn generate_proof(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<ProofGenerationRequest>,
) -> Result<Json<ProofGenerationResponse>> {
    // Validate privacy level
    if req.privacy_level < 0 || req.privacy_level > 5 {
        return Err(PrivacyError::InvalidLevel(req.privacy_level));
    }

    // Level 0 doesn't need a proof
    if req.privacy_level == 0 {
        return Ok(Json(ProofGenerationResponse {
            proof_id: Uuid::now_v7(),
            transaction_id: req.transaction_id,
            status: ProofStatus::Completed,
            proof: Some("no_proof_required".to_string()),
            verification_key: None,
            public_inputs: Some(vec![]),
            generation_time_ms: Some(0),
            error: None,
            created_at: chrono::Utc::now(),
            completed_at: Some(chrono::Utc::now()),
            queue_position: None,
            estimated_wait_ms: None,
        }));
    }

    // Create job
    let job = ProofJob::from_request(&req);
    let proof_id = job.proof_id;

    // Check queue capacity
    let queue_stats = state.get_queue_stats().await;
    if queue_stats.total_queued >= 1000 {
        return Err(PrivacyError::QueueFull);
    }

    // Store job in database
    let _ = sqlx::query(
        r#"
        INSERT INTO proof_jobs (id, transaction_id, sender, recipient, amount, privacy_level, priority, callback_url, status, created_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'queued', NOW())
        "#,
    )
    .bind(job.proof_id)
    .bind(job.transaction_id)
    .bind(&job.sender)
    .bind(&job.recipient)
    .bind(job.amount)
    .bind(job.privacy_level)
    .bind(format!("{:?}", job.priority).to_lowercase())
    .bind(&job.callback_url)
    .execute(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    // Add to queue
    let queue_position = {
        let mut queue = state.proof_queue.lock().await;
        queue.enqueue(job);
        PROOF_QUEUE_LENGTH.set(queue.len() as f64);
        queue.position(proof_id)
    };

    // Estimate wait time
    let estimated_wait_ms = queue_position.map(|pos| {
        let base_time = match req.privacy_level {
            1 | 2 => 500,
            3 => 1000,
            4 => 2500,
            5 => 5000,
            _ => 1500,
        };
        (pos as u64 + 1) * base_time / state.config.proof.max_concurrent as u64
    });

    tracing::info!(
        proof_id = %proof_id,
        transaction_id = %req.transaction_id,
        privacy_level = req.privacy_level,
        priority = ?req.priority,
        queue_position = ?queue_position,
        "Proof generation queued"
    );

    Ok(Json(ProofGenerationResponse {
        proof_id,
        transaction_id: req.transaction_id,
        status: ProofStatus::Queued,
        proof: None,
        verification_key: None,
        public_inputs: None,
        generation_time_ms: None,
        error: None,
        created_at: chrono::Utc::now(),
        completed_at: None,
        queue_position,
        estimated_wait_ms,
    }))
}

/// Get proof by ID
pub async fn get_proof(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<ProofGenerationResponse>> {
    // Check active proofs first
    {
        let active = state.active_proofs.read().await;
        if let Some(job) = active.get(&id) {
            return Ok(Json(ProofGenerationResponse {
                proof_id: job.proof_id,
                transaction_id: job.transaction_id,
                status: job.status,
                proof: None,
                verification_key: None,
                public_inputs: None,
                generation_time_ms: None,
                error: None,
                created_at: job.created_at,
                completed_at: None,
                queue_position: None,
                estimated_wait_ms: None,
            }));
        }
    }

    // Check database
    let job: Option<(
        Uuid,
        Uuid,
        String,
        Option<String>,
        Option<String>,
        Option<i64>,
        chrono::DateTime<chrono::Utc>,
        Option<chrono::DateTime<chrono::Utc>>,
    )> = sqlx::query_as(
        r#"
        SELECT id, transaction_id, status, proof, error, 
               EXTRACT(EPOCH FROM (completed_at - created_at)) * 1000 as generation_time_ms,
               created_at, completed_at
        FROM proof_jobs
        WHERE id = $1
        "#,
    )
    .bind(id)
    .fetch_optional(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    match job {
        Some((
            proof_id,
            transaction_id,
            status_str,
            proof,
            error,
            generation_time_ms,
            created_at,
            completed_at,
        )) => {
            let status = match status_str.as_str() {
                "queued" => ProofStatus::Queued,
                "generating" => ProofStatus::Generating,
                "completed" => ProofStatus::Completed,
                "failed" => ProofStatus::Failed,
                "cancelled" => ProofStatus::Cancelled,
                _ => ProofStatus::Queued,
            };

            Ok(Json(ProofGenerationResponse {
                proof_id,
                transaction_id,
                status,
                proof,
                verification_key: None, // TODO: Store/retrieve verification key
                public_inputs: None,
                generation_time_ms: generation_time_ms.map(|ms| ms as u64),
                error,
                created_at,
                completed_at,
                queue_position: None,
                estimated_wait_ms: None,
            }))
        }
        None => Err(PrivacyError::ProofNotFound(id.to_string())),
    }
}

/// Get proof status
pub async fn get_proof_status(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>> {
    // Check active proofs
    {
        let active = state.active_proofs.read().await;
        if let Some(job) = active.get(&id) {
            return Ok(Json(serde_json::json!({
                "proof_id": job.proof_id,
                "status": format!("{:?}", job.status).to_lowercase(),
                "created_at": job.created_at,
            })));
        }
    }

    // Check queue
    {
        let queue = state.proof_queue.lock().await;
        if let Some(position) = queue.position(id) {
            return Ok(Json(serde_json::json!({
                "proof_id": id,
                "status": "queued",
                "queue_position": position,
            })));
        }
    }

    // Check database
    let status: Option<(String,)> =
        sqlx::query_as("SELECT status FROM proof_jobs WHERE id = $1")
            .bind(id)
            .fetch_optional(&state.db)
            .await
            .map_err(PrivacyError::Database)?;

    match status {
        Some((s,)) => Ok(Json(serde_json::json!({
            "proof_id": id,
            "status": s,
        }))),
        None => Err(PrivacyError::ProofNotFound(id.to_string())),
    }
}

/// Cancel proof generation
pub async fn cancel_proof(
    Extension(state): Extension<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<serde_json::Value>> {
    // Update database
    let result = sqlx::query(
        r#"
        UPDATE proof_jobs 
        SET status = 'cancelled', completed_at = NOW()
        WHERE id = $1 AND status IN ('queued', 'generating')
        "#,
    )
    .bind(id)
    .execute(&state.db)
    .await
    .map_err(PrivacyError::Database)?;

    if result.rows_affected() > 0 {
        tracing::info!(proof_id = %id, "Proof generation cancelled");
        Ok(Json(serde_json::json!({
            "proof_id": id,
            "status": "cancelled",
        })))
    } else {
        Err(PrivacyError::ProofNotFound(id.to_string()))
    }
}

/// Verify a proof
pub async fn verify_proof(
    Json(req): Json<ProofVerificationRequest>,
) -> Result<Json<ProofVerificationResponse>> {
    let start = std::time::Instant::now();

    // Placeholder verification - in production, this would use actual ZK verification
    let valid = req.proof.starts_with("proof_v1_") || req.proof.starts_with("morph_proof_");

    let verification_time_ms = start.elapsed().as_millis() as u64;

    // Update metrics
    PROOFS_VERIFIED.inc();
    PROOF_VERIFICATION_DURATION.observe(verification_time_ms as f64 / 1000.0);

    Ok(Json(ProofVerificationResponse {
        valid,
        verification_time_ms,
        error: if valid {
            None
        } else {
            Some("Invalid proof format".to_string())
        },
    }))
}

/// Batch proof generation
pub async fn batch_generate(
    Extension(state): Extension<Arc<AppState>>,
    Json(req): Json<BatchProofRequest>,
) -> Result<Json<BatchProofResponse>> {
    if req.proofs.len() > state.config.proof.max_batch_size {
        return Err(PrivacyError::Internal(format!(
            "Batch size {} exceeds maximum {}",
            req.proofs.len(),
            state.config.proof.max_batch_size
        )));
    }

    let batch_id = Uuid::now_v7();
    let mut results: Vec<BatchProofResult> = Vec::new();
    let mut queued = 0;
    let mut failed = 0;

    for (index, proof_req) in req.proofs.iter().enumerate() {
        // Validate
        if proof_req.privacy_level < 0 || proof_req.privacy_level > 5 {
            if req.fail_fast {
                return Err(PrivacyError::InvalidLevel(proof_req.privacy_level));
            }
            results.push(BatchProofResult {
                index,
                proof_id: None,
                status: ProofStatus::Failed,
                error: Some(format!("Invalid privacy level: {}", proof_req.privacy_level)),
            });
            failed += 1;
            continue;
        }

        // Create job
        let job = ProofJob::from_request(proof_req);
        let proof_id = job.proof_id;

        // Store in database
        let db_result = sqlx::query(
            r#"
            INSERT INTO proof_jobs (id, transaction_id, sender, recipient, amount, privacy_level, priority, callback_url, status, batch_id, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'queued', $9, NOW())
            "#,
        )
        .bind(job.proof_id)
        .bind(job.transaction_id)
        .bind(&job.sender)
        .bind(&job.recipient)
        .bind(job.amount)
        .bind(job.privacy_level)
        .bind(format!("{:?}", job.priority).to_lowercase())
        .bind(&job.callback_url)
        .bind(batch_id)
        .execute(&state.db)
        .await;

        match db_result {
            Ok(_) => {
                // Add to queue
                {
                    let mut queue = state.proof_queue.lock().await;
                    queue.enqueue(job);
                }

                results.push(BatchProofResult {
                    index,
                    proof_id: Some(proof_id),
                    status: ProofStatus::Queued,
                    error: None,
                });
                queued += 1;
            }
            Err(e) => {
                if req.fail_fast {
                    return Err(PrivacyError::Database(e));
                }
                results.push(BatchProofResult {
                    index,
                    proof_id: None,
                    status: ProofStatus::Failed,
                    error: Some(e.to_string()),
                });
                failed += 1;
            }
        }
    }

    // Update queue metric
    {
        let queue = state.proof_queue.lock().await;
        PROOF_QUEUE_LENGTH.set(queue.len() as f64);
    }

    tracing::info!(
        batch_id = %batch_id,
        total = req.proofs.len(),
        queued = queued,
        failed = failed,
        "Batch proof generation queued"
    );

    Ok(Json(BatchProofResponse {
        batch_id,
        total: req.proofs.len(),
        queued,
        failed,
        results,
    }))
}
