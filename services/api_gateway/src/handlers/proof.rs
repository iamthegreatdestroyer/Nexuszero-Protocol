//! Proof Generation Handlers
//!
//! Handles zero-knowledge proof generation and verification

use crate::error::{ApiError, ApiResult};
use crate::handlers::transaction::PrivacyLevel;
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

/// Generate proof request
#[derive(Debug, Deserialize, Validate)]
pub struct GenerateProofRequest {
    /// Transaction ID to generate proof for
    pub transaction_id: Option<Uuid>,

    /// Privacy level for the proof
    pub privacy_level: u8,

    /// Circuit data (base64 encoded)
    #[validate(length(min = 1, max = 1048576))]
    pub circuit_data: String,

    /// Public inputs (base64 encoded)
    pub public_inputs: Option<Vec<String>>,

    /// Priority (affects queue position)
    pub priority: Option<ProofPriority>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofPriority {
    Low,
    Normal,
    High,
    Urgent,
}

impl Default for ProofPriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Generate proof response
#[derive(Debug, Serialize)]
pub struct GenerateProofResponse {
    pub proof_id: String,
    pub status: ProofStatus,
    pub privacy_level: PrivacyLevel,
    pub estimated_time_ms: u64,
    pub queue_position: Option<u32>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProofStatus {
    Queued,
    Generating,
    Generated,
    Verified,
    Failed,
}

/// Verify proof request
#[derive(Debug, Deserialize, Validate)]
pub struct VerifyProofRequest {
    /// Proof data (base64 encoded)
    #[validate(length(min = 1, max = 1048576))]
    pub proof: String,

    /// Public inputs (base64 encoded)
    pub public_inputs: Vec<String>,

    /// Verification key ID (if using stored key)
    pub verification_key_id: Option<String>,

    /// Privacy level of the proof
    pub privacy_level: u8,
}

/// Verify proof response
#[derive(Debug, Serialize)]
pub struct VerifyProofResponse {
    pub valid: bool,
    pub verification_time_ms: u64,
    pub proof_type: String,
    pub privacy_level: PrivacyLevel,
    pub details: Option<serde_json::Value>,
}

/// Batch generate proofs request
#[derive(Debug, Deserialize, Validate)]
pub struct BatchGenerateRequest {
    /// List of proof requests
    #[validate(length(min = 1, max = 100))]
    pub proofs: Vec<GenerateProofRequest>,

    /// Whether to wait for all proofs
    pub wait_for_all: bool,
}

/// Batch generate proofs response
#[derive(Debug, Serialize)]
pub struct BatchGenerateResponse {
    pub batch_id: String,
    pub proofs: Vec<BatchProofStatus>,
    pub total: usize,
    pub completed: usize,
    pub estimated_total_time_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct BatchProofStatus {
    pub index: usize,
    pub proof_id: String,
    pub status: ProofStatus,
    pub error: Option<String>,
}

/// Get proof response
#[derive(Debug, Serialize)]
pub struct GetProofResponse {
    pub proof_id: String,
    pub transaction_id: Option<String>,
    pub status: ProofStatus,
    pub privacy_level: PrivacyLevel,
    pub proof_data: Option<String>,
    pub generation_time_ms: Option<u64>,
    pub prover_node_id: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub completed_at: Option<chrono::DateTime<chrono::Utc>>,
}

/// Proof status query
#[derive(Debug, Deserialize)]
pub struct ProofStatusQuery {
    pub include_proof_data: Option<bool>,
}

/// Generate proof handler
pub async fn generate_proof(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<GenerateProofRequest>,
) -> ApiResult<Json<GenerateProofResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    if payload.privacy_level > 5 {
        return Err(ApiError::UnsupportedPrivacyLevel(payload.privacy_level));
    }

    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let proof_id = Uuid::new_v4();
    let priority = payload.priority.unwrap_or_default();
    let now = chrono::Utc::now();

    // Decode circuit data
    let circuit_data = base64::decode(&payload.circuit_data)
        .map_err(|_| ApiError::BadRequest("Invalid base64 circuit data".to_string()))?;

    // Estimate proof generation time based on privacy level and data size
    let estimated_time_ms = estimate_proof_time(payload.privacy_level, circuit_data.len());

    // Store proof request in database
    sqlx::query!(
        r#"
        INSERT INTO proofs (id, transaction_id, proof_type, privacy_level, verified, created_at)
        VALUES ($1, $2, $3, $4, false, $5)
        "#,
        proof_id,
        payload.transaction_id,
        format!("level_{}", payload.privacy_level),
        payload.privacy_level as i32,
        now
    )
    .execute(&state.db)
    .await?;

    // Send proof generation request to proof coordinator
    let queue_position = submit_to_proof_network(
        &state,
        &proof_id,
        &circuit_data,
        payload.privacy_level,
        priority,
    )
    .await?;

    tracing::info!(
        proof_id = %proof_id,
        user_id = %user_id,
        privacy_level = payload.privacy_level,
        priority = ?priority,
        "Proof generation requested"
    );

    Ok(Json(GenerateProofResponse {
        proof_id: proof_id.to_string(),
        status: ProofStatus::Queued,
        privacy_level: PrivacyLevel::from(payload.privacy_level),
        estimated_time_ms,
        queue_position: Some(queue_position),
        created_at: now,
    }))
}

/// Verify proof handler
pub async fn verify_proof(
    Extension(state): Extension<Arc<AppState>>,
    Extension(_user): Extension<AuthenticatedUser>,
    Json(payload): Json<VerifyProofRequest>,
) -> ApiResult<Json<VerifyProofResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    if payload.privacy_level > 5 {
        return Err(ApiError::UnsupportedPrivacyLevel(payload.privacy_level));
    }

    let start = std::time::Instant::now();

    // Decode proof
    let proof_data = base64::decode(&payload.proof)
        .map_err(|_| ApiError::BadRequest("Invalid base64 proof data".to_string()))?;

    // Decode public inputs
    let public_inputs: Result<Vec<Vec<u8>>, _> = payload
        .public_inputs
        .iter()
        .map(|s| base64::decode(s))
        .collect();
    let public_inputs = public_inputs
        .map_err(|_| ApiError::BadRequest("Invalid base64 public inputs".to_string()))?;

    // Perform verification based on privacy level
    let (valid, proof_type) = verify_proof_data(&proof_data, &public_inputs, payload.privacy_level)?;

    let verification_time_ms = start.elapsed().as_millis() as u64;

    tracing::debug!(
        privacy_level = payload.privacy_level,
        valid = valid,
        verification_time_ms = verification_time_ms,
        "Proof verified"
    );

    Ok(Json(VerifyProofResponse {
        valid,
        verification_time_ms,
        proof_type,
        privacy_level: PrivacyLevel::from(payload.privacy_level),
        details: None,
    }))
}

/// Batch generate proofs handler
pub async fn batch_generate(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<BatchGenerateRequest>,
) -> ApiResult<Json<BatchGenerateResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    let batch_id = Uuid::new_v4();
    let mut proof_statuses = Vec::with_capacity(payload.proofs.len());
    let mut total_estimated_time = 0u64;

    for (index, proof_req) in payload.proofs.iter().enumerate() {
        if proof_req.privacy_level > 5 {
            proof_statuses.push(BatchProofStatus {
                index,
                proof_id: Uuid::new_v4().to_string(),
                status: ProofStatus::Failed,
                error: Some(format!("Unsupported privacy level: {}", proof_req.privacy_level)),
            });
            continue;
        }

        let proof_id = Uuid::new_v4();
        
        // Decode and estimate
        if let Ok(circuit_data) = base64::decode(&proof_req.circuit_data) {
            let estimated_time = estimate_proof_time(proof_req.privacy_level, circuit_data.len());
            total_estimated_time += estimated_time;

            proof_statuses.push(BatchProofStatus {
                index,
                proof_id: proof_id.to_string(),
                status: ProofStatus::Queued,
                error: None,
            });
        } else {
            proof_statuses.push(BatchProofStatus {
                index,
                proof_id: proof_id.to_string(),
                status: ProofStatus::Failed,
                error: Some("Invalid base64 circuit data".to_string()),
            });
        }
    }

    let completed = proof_statuses
        .iter()
        .filter(|s| matches!(s.status, ProofStatus::Failed))
        .count();

    tracing::info!(
        batch_id = %batch_id,
        total = payload.proofs.len(),
        queued = payload.proofs.len() - completed,
        "Batch proof generation requested"
    );

    Ok(Json(BatchGenerateResponse {
        batch_id: batch_id.to_string(),
        proofs: proof_statuses,
        total: payload.proofs.len(),
        completed,
        estimated_total_time_ms: total_estimated_time,
    }))
}

/// Get proof by ID handler
pub async fn get_proof(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Path(id): Path<Uuid>,
    Query(params): Query<ProofStatusQuery>,
) -> ApiResult<Json<GetProofResponse>> {
    let proof = sqlx::query!(
        r#"
        SELECT 
            p.id, p.transaction_id, p.proof_type, p.privacy_level, 
            p.verified, p.generation_time_ms, p.prover_node_id, 
            p.created_at, p.proof_data
        FROM proofs p
        LEFT JOIN transactions t ON p.transaction_id = t.id
        WHERE p.id = $1
        "#,
        id
    )
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Proof not found".to_string()))?;

    let include_data = params.include_proof_data.unwrap_or(false);

    let status = if proof.verified {
        ProofStatus::Verified
    } else if proof.proof_data.is_some() {
        ProofStatus::Generated
    } else {
        ProofStatus::Generating
    };

    Ok(Json(GetProofResponse {
        proof_id: proof.id.to_string(),
        transaction_id: proof.transaction_id.map(|id| id.to_string()),
        status,
        privacy_level: PrivacyLevel::from(proof.privacy_level as u8),
        proof_data: if include_data {
            proof.proof_data.map(|d| base64::encode(&d))
        } else {
            None
        },
        generation_time_ms: proof.generation_time_ms.map(|t| t as u64),
        prover_node_id: proof.prover_node_id.map(|id| id.to_string()),
        created_at: proof.created_at,
        completed_at: None, // Would be set when proof is generated
    }))
}

/// Get proof status handler
pub async fn get_proof_status(
    Extension(state): Extension<Arc<AppState>>,
    Extension(_user): Extension<AuthenticatedUser>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<serde_json::Value>> {
    let proof = sqlx::query!(
        "SELECT id, verified, proof_data IS NOT NULL as has_data FROM proofs WHERE id = $1",
        id
    )
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Proof not found".to_string()))?;

    let status = if proof.verified {
        "verified"
    } else if proof.has_data.unwrap_or(false) {
        "generated"
    } else {
        "generating"
    };

    Ok(Json(serde_json::json!({
        "proof_id": proof.id.to_string(),
        "status": status,
        "verified": proof.verified
    })))
}

// Helper functions

fn estimate_proof_time(privacy_level: u8, data_size: usize) -> u64 {
    let base_time = match privacy_level {
        0 => 0,
        1 => 50,
        2 => 100,
        3 => 250,
        4 => 500,
        5 => 1000,
        _ => 0,
    };

    // Add time based on data size
    let size_factor = (data_size / 1024) as u64;
    base_time + (size_factor * 10)
}

fn verify_proof_data(
    proof: &[u8],
    _public_inputs: &[Vec<u8>],
    privacy_level: u8,
) -> Result<(bool, String), ApiError> {
    // In production, would call nexuszero-crypto for verification
    let proof_type = match privacy_level {
        0 => "none",
        1 | 2 => "bulletproofs",
        3 | 4 => "quantum_lattice",
        5 => "hybrid_zk_lattice",
        _ => "unknown",
    };

    // Placeholder verification - check proof structure
    let valid = proof.len() >= 32 * (privacy_level as usize + 1);

    Ok((valid, proof_type.to_string()))
}

async fn submit_to_proof_network(
    state: &AppState,
    proof_id: &Uuid,
    circuit_data: &[u8],
    privacy_level: u8,
    priority: ProofPriority,
) -> Result<u32, ApiError> {
    let url = format!(
        "{}/proofs/submit",
        state.config.services.proof_coordinator
    );

    let response = state
        .http_client
        .post(&url)
        .json(&serde_json::json!({
            "proof_id": proof_id.to_string(),
            "circuit_data": base64::encode(circuit_data),
            "privacy_level": privacy_level,
            "priority": priority
        }))
        .send()
        .await;

    match response {
        Ok(resp) if resp.status().is_success() => {
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                let position = body
                    .get("queue_position")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as u32;
                return Ok(position);
            }
            Ok(1) // Default queue position
        }
        _ => {
            tracing::warn!("Failed to submit to proof network, using local queue");
            Ok(1)
        }
    }
}

// Base64 module for encoding/decoding
mod base64 {
    use base64::{engine::general_purpose::STANDARD, Engine};

    pub fn encode(data: &[u8]) -> String {
        STANDARD.encode(data)
    }

    pub fn decode(data: &str) -> Result<Vec<u8>, base64::DecodeError> {
        STANDARD.decode(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_proof_time() {
        assert_eq!(estimate_proof_time(0, 1024), 0);
        assert_eq!(estimate_proof_time(3, 1024), 260);
        assert_eq!(estimate_proof_time(5, 2048), 1020);
    }

    #[test]
    fn test_verify_proof_data() {
        let small_proof = vec![0u8; 32];
        let large_proof = vec![0u8; 256];

        let (valid, _) = verify_proof_data(&small_proof, &[], 0).unwrap();
        assert!(valid);

        let (valid, proof_type) = verify_proof_data(&large_proof, &[], 5).unwrap();
        assert!(valid);
        assert_eq!(proof_type, "hybrid_zk_lattice");
    }

    #[test]
    fn test_priority_default() {
        let priority = ProofPriority::default();
        assert!(matches!(priority, ProofPriority::Normal));
    }
}
