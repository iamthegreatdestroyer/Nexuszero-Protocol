//! Privacy Handlers
//!
//! Handles Adaptive Privacy Morphing (APM) operations

use crate::error::{ApiError, ApiResult};
use crate::handlers::transaction::PrivacyLevel;
use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use axum::{extract::Extension, Json};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

/// Privacy level details
#[derive(Debug, Clone, Serialize)]
pub struct PrivacyLevelInfo {
    pub level: u8,
    pub name: String,
    pub description: String,
    pub features: Vec<String>,
    pub proof_type: String,
    pub estimated_proof_time_ms: u64,
    pub estimated_gas_cost: u64,
    pub security_bits: u32,
    pub quantum_resistant: bool,
}

/// Privacy recommendation request
#[derive(Debug, Deserialize, Validate)]
pub struct RecommendRequest {
    /// Transaction value in USD
    pub value_usd: f64,

    /// Whether compliance is required
    #[serde(default)]
    pub requires_compliance: bool,

    /// Target chain
    #[serde(default = "default_chain")]
    pub chain: String,

    /// Counterparty known flag
    #[serde(default)]
    pub counterparty_known: bool,

    /// Jurisdiction
    pub jurisdiction: Option<String>,

    /// User risk score (0.0 - 1.0)
    pub risk_score: Option<f64>,
}

fn default_chain() -> String {
    "ethereum".to_string()
}

/// Privacy recommendation response
#[derive(Debug, Serialize)]
pub struct RecommendResponse {
    pub recommended_level: u8,
    pub level_info: PrivacyLevelInfo,
    pub reasons: Vec<String>,
    pub alternative_levels: Vec<AlternativeLevel>,
}

#[derive(Debug, Serialize)]
pub struct AlternativeLevel {
    pub level: u8,
    pub reason: String,
}

/// Privacy morph request
#[derive(Debug, Deserialize, Validate)]
pub struct MorphRequest {
    /// Transaction ID to morph
    pub transaction_id: Uuid,

    /// Target privacy level
    pub target_level: u8,
}

/// Privacy morph response
#[derive(Debug, Serialize)]
pub struct MorphResponse {
    pub transaction_id: String,
    pub previous_level: u8,
    pub new_level: u8,
    pub morphing_steps: u32,
    pub new_proof_id: Option<String>,
    pub status: String,
}

/// Cost estimation request
#[derive(Debug, Deserialize, Validate)]
pub struct EstimateCostRequest {
    pub privacy_level: u8,
    pub chain: String,
    pub data_size_bytes: Option<u64>,
}

/// Cost estimation response
#[derive(Debug, Serialize)]
pub struct EstimateCostResponse {
    pub privacy_level: u8,
    pub chain: String,
    pub estimated_proof_time_ms: u64,
    pub estimated_gas_cost: u64,
    pub estimated_gas_cost_usd: f64,
    pub proof_size_bytes: u64,
}

/// Database record for transaction privacy queries
#[derive(Debug, sqlx::FromRow)]
struct TransactionPrivacyRecord {
    id: Uuid,
    privacy_level: i32,
    status: String,
}

/// List all privacy levels handler
pub async fn list_privacy_levels() -> Json<Vec<PrivacyLevelInfo>> {
    Json(get_all_privacy_levels())
}

/// Recommend privacy level handler
pub async fn recommend_level(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<RecommendRequest>,
) -> ApiResult<Json<RecommendResponse>> {
    let mut recommended_level = 3u8; // Default to Private
    let mut reasons = Vec::new();
    let mut alternatives = Vec::new();

    // Value-based recommendation
    if payload.value_usd > 100_000.0 {
        recommended_level = 5;
        reasons.push("High-value transaction (>$100K) - Sovereign privacy recommended".to_string());
    } else if payload.value_usd > 10_000.0 {
        recommended_level = 4;
        reasons.push("Significant value transaction (>$10K) - Anonymous privacy recommended".to_string());
    } else if payload.value_usd > 1_000.0 {
        recommended_level = 3;
        reasons.push("Standard transaction value - Private privacy recommended".to_string());
    } else {
        recommended_level = 2;
        reasons.push("Lower value transaction - Confidential privacy sufficient".to_string());
        alternatives.push(AlternativeLevel {
            level: 3,
            reason: "Upgrade to Private for additional unlinkability".to_string(),
        });
    }

    // Compliance considerations
    if payload.requires_compliance {
        recommended_level = recommended_level.min(3);
        reasons.push("Compliance required - Maximum privacy level capped at Private".to_string());
    }

    // Counterparty considerations
    if !payload.counterparty_known {
        recommended_level = recommended_level.max(3);
        reasons.push("Unknown counterparty - Minimum Private level recommended".to_string());
    }

    // Risk score considerations
    if let Some(risk_score) = payload.risk_score {
        if risk_score > 0.7 {
            recommended_level = recommended_level.min(2);
            reasons.push("Elevated risk score - Privacy level limited".to_string());
        }
    }

    // Jurisdiction considerations
    if let Some(jurisdiction) = &payload.jurisdiction {
        match jurisdiction.as_str() {
            "US" | "EU" => {
                if recommended_level > 4 {
                    alternatives.push(AlternativeLevel {
                        level: recommended_level,
                        reason: format!("Level {} available but may require additional compliance in {}", recommended_level, jurisdiction),
                    });
                }
            }
            "CH" | "SG" => {
                // More privacy-friendly jurisdictions
            }
            _ => {}
        }
    }

    // User preference override
    if let Some(default_level) = user.claims.custom.default_privacy_level {
        if default_level != recommended_level {
            alternatives.push(AlternativeLevel {
                level: default_level,
                reason: format!("Your default preference is Level {}", default_level),
            });
        }
    }

    let level_info = get_privacy_level_info(recommended_level);

    Ok(Json(RecommendResponse {
        recommended_level,
        level_info,
        reasons,
        alternative_levels: alternatives,
    }))
}

/// Morph privacy level handler
pub async fn morph_privacy(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<MorphRequest>,
) -> ApiResult<Json<MorphResponse>> {
    if payload.target_level > 5 {
        return Err(ApiError::UnsupportedPrivacyLevel(payload.target_level));
    }

    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    // Get current transaction
    let transaction = sqlx::query_as::<_, TransactionPrivacyRecord>(
        r#"
        SELECT id, privacy_level, status
        FROM transactions
        WHERE id = $1 AND user_id = $2
        "#
    )
    .bind(payload.transaction_id)
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Transaction not found".to_string()))?;

    let current_level = transaction.privacy_level as u8;

    // Validate state for morphing
    if transaction.status != "created" && transaction.status != "privacy_selected" {
        return Err(ApiError::BadRequest(
            "Transaction cannot be morphed in current state".to_string(),
        ));
    }

    // Calculate morphing steps
    let morphing_steps = if payload.target_level > current_level {
        // Increasing privacy - single step with new proof
        1
    } else {
        // Decreasing privacy - must do incrementally
        (current_level - payload.target_level) as u32
    };

    // Update transaction privacy level
    sqlx::query(
        r#"
        UPDATE transactions
        SET privacy_level = $1, status = 'privacy_selected', updated_at = NOW()
        WHERE id = $2
        "#
    )
    .bind(payload.target_level as i32)
    .bind(payload.transaction_id)
    .execute(&state.db)
    .await?;

    // Request new proof generation
    let new_proof_id = if payload.target_level > 0 {
        request_proof_generation(&state, &payload.transaction_id, payload.target_level).await?
    } else {
        None
    };

    tracing::info!(
        transaction_id = %payload.transaction_id,
        from_level = current_level,
        to_level = payload.target_level,
        "Privacy level morphed"
    );

    Ok(Json(MorphResponse {
        transaction_id: payload.transaction_id.to_string(),
        previous_level: current_level,
        new_level: payload.target_level,
        morphing_steps,
        new_proof_id: new_proof_id.map(|id| id.to_string()),
        status: "morphed".to_string(),
    }))
}

/// Estimate cost handler
pub async fn estimate_cost(
    Extension(_state): Extension<Arc<AppState>>,
    Json(payload): Json<EstimateCostRequest>,
) -> ApiResult<Json<EstimateCostResponse>> {
    if payload.privacy_level > 5 {
        return Err(ApiError::UnsupportedPrivacyLevel(payload.privacy_level));
    }

    let data_size = payload.data_size_bytes.unwrap_or(256);

    // Estimate proof time based on privacy level
    let estimated_proof_time_ms = match payload.privacy_level {
        0 => 0,
        1 => 50,
        2 => 100,
        3 => 250 + (data_size / 1024) * 10,
        4 => 500 + (data_size / 1024) * 20,
        5 => 1000 + (data_size / 1024) * 50,
        _ => 0,
    };

    // Estimate gas cost based on chain and privacy level
    let base_gas = match payload.chain.as_str() {
        "ethereum" => 21000,
        "polygon" => 21000,
        "arbitrum" => 15000,
        "optimism" => 15000,
        "solana" => 5000,
        _ => 21000,
    };

    let privacy_gas = match payload.privacy_level {
        0 => 0,
        1 => 30000,
        2 => 80000,
        3 => 180000,
        4 => 330000,
        5 => 480000,
        _ => 0,
    };

    let proof_size_gas = (data_size / 32) * 16; // calldata cost

    let estimated_gas_cost = base_gas + privacy_gas + proof_size_gas;

    // Get gas price (placeholder - would query chain)
    let gas_price_gwei = match payload.chain.as_str() {
        "ethereum" => 30.0,
        "polygon" => 50.0,
        "arbitrum" => 0.1,
        "optimism" => 0.01,
        _ => 30.0,
    };

    // Calculate USD cost (placeholder prices)
    let native_price_usd = match payload.chain.as_str() {
        "ethereum" => 3000.0,
        "polygon" => 0.50,
        "arbitrum" => 3000.0,
        "optimism" => 3000.0,
        _ => 3000.0,
    };

    let gas_cost_native = (estimated_gas_cost as f64 * gas_price_gwei) / 1e9;
    let gas_cost_usd = gas_cost_native * native_price_usd;

    // Estimate proof size
    let proof_size_bytes = match payload.privacy_level {
        0 => 0,
        1 => 128,
        2 => 256,
        3 => 512,
        4 => 1024,
        5 => 2048,
        _ => 0,
    };

    Ok(Json(EstimateCostResponse {
        privacy_level: payload.privacy_level,
        chain: payload.chain,
        estimated_proof_time_ms,
        estimated_gas_cost,
        estimated_gas_cost_usd: gas_cost_usd,
        proof_size_bytes,
    }))
}

// Helper functions

fn get_all_privacy_levels() -> Vec<PrivacyLevelInfo> {
    vec![
        PrivacyLevelInfo {
            level: 0,
            name: "Transparent".to_string(),
            description: "Public blockchain parity - all data visible".to_string(),
            features: vec![
                "Full transaction visibility".to_string(),
                "Maximum compliance".to_string(),
                "Lowest cost".to_string(),
            ],
            proof_type: "none".to_string(),
            estimated_proof_time_ms: 0,
            estimated_gas_cost: 21000,
            security_bits: 0,
            quantum_resistant: false,
        },
        PrivacyLevelInfo {
            level: 1,
            name: "Pseudonymous".to_string(),
            description: "Address obfuscation - amounts visible".to_string(),
            features: vec![
                "Address mixing".to_string(),
                "Decoy outputs".to_string(),
                "Basic unlinkability".to_string(),
            ],
            proof_type: "bulletproofs".to_string(),
            estimated_proof_time_ms: 50,
            estimated_gas_cost: 50000,
            security_bits: 80,
            quantum_resistant: false,
        },
        PrivacyLevelInfo {
            level: 2,
            name: "Confidential".to_string(),
            description: "Encrypted amounts - addresses visible".to_string(),
            features: vec![
                "Amount encryption".to_string(),
                "Range proofs".to_string(),
                "Pedersen commitments".to_string(),
            ],
            proof_type: "bulletproofs".to_string(),
            estimated_proof_time_ms: 100,
            estimated_gas_cost: 100000,
            security_bits: 128,
            quantum_resistant: false,
        },
        PrivacyLevelInfo {
            level: 3,
            name: "Private".to_string(),
            description: "Full transaction privacy".to_string(),
            features: vec![
                "Encrypted amounts".to_string(),
                "Address obfuscation".to_string(),
                "Ring signatures".to_string(),
                "Quantum-resistant lattice proofs".to_string(),
            ],
            proof_type: "quantum_lattice".to_string(),
            estimated_proof_time_ms: 250,
            estimated_gas_cost: 200000,
            security_bits: 192,
            quantum_resistant: true,
        },
        PrivacyLevelInfo {
            level: 4,
            name: "Anonymous".to_string(),
            description: "Unlinkable transactions".to_string(),
            features: vec![
                "Full encryption".to_string(),
                "Large anonymity set".to_string(),
                "Transaction unlinkability".to_string(),
                "Stealth addresses".to_string(),
                "Quantum-resistant".to_string(),
            ],
            proof_type: "quantum_lattice".to_string(),
            estimated_proof_time_ms: 500,
            estimated_gas_cost: 350000,
            security_bits: 256,
            quantum_resistant: true,
        },
        PrivacyLevelInfo {
            level: 5,
            name: "Sovereign".to_string(),
            description: "Maximum privacy - ZK everything".to_string(),
            features: vec![
                "Maximum anonymity set".to_string(),
                "Full ZK-SNARK proofs".to_string(),
                "Lattice-based hybrid".to_string(),
                "Post-quantum secure".to_string(),
                "Zero metadata leakage".to_string(),
            ],
            proof_type: "hybrid_zk_lattice".to_string(),
            estimated_proof_time_ms: 1000,
            estimated_gas_cost: 500000,
            security_bits: 256,
            quantum_resistant: true,
        },
    ]
}

fn get_privacy_level_info(level: u8) -> PrivacyLevelInfo {
    get_all_privacy_levels()
        .into_iter()
        .find(|l| l.level == level)
        .unwrap_or_else(|| get_all_privacy_levels()[3].clone()) // Default to Private
}

async fn request_proof_generation(
    state: &AppState,
    tx_id: &Uuid,
    privacy_level: u8,
) -> Result<Option<Uuid>, ApiError> {
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
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                if let Some(proof_id) = body.get("proof_id").and_then(|v| v.as_str()) {
                    return Ok(Uuid::parse_str(proof_id).ok());
                }
            }
            Ok(None)
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_levels() {
        let levels = get_all_privacy_levels();
        assert_eq!(levels.len(), 6);
        
        // Verify ordering
        for (i, level) in levels.iter().enumerate() {
            assert_eq!(level.level, i as u8);
        }

        // Verify quantum resistance
        assert!(!levels[0].quantum_resistant); // Transparent
        assert!(!levels[1].quantum_resistant); // Pseudonymous
        assert!(!levels[2].quantum_resistant); // Confidential
        assert!(levels[3].quantum_resistant);  // Private
        assert!(levels[4].quantum_resistant);  // Anonymous
        assert!(levels[5].quantum_resistant);  // Sovereign
    }

    #[test]
    fn test_get_privacy_level_info() {
        let level_3 = get_privacy_level_info(3);
        assert_eq!(level_3.name, "Private");
        assert!(level_3.quantum_resistant);

        let level_invalid = get_privacy_level_info(99);
        assert_eq!(level_invalid.level, 3); // Default to Private
    }
}

