//! Proof generation service integration

use crate::config::Config;
use crate::error::{Result, TransactionError};
use crate::models::{ProofPriority, ProofStatus};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Proof service client
pub struct ProofServiceClient {
    client: reqwest::Client,
    base_url: String,
}

/// Proof generation request
#[derive(Debug, Serialize)]
pub struct GenerateProofRequest {
    pub transaction_id: Uuid,
    pub sender: String,
    pub recipient: String,
    pub amount: i64,
    pub privacy_level: i16,
    pub priority: ProofPriority,
    pub callback_url: Option<String>,
}

/// Proof generation response
#[derive(Debug, Deserialize)]
pub struct GenerateProofResponse {
    pub proof_id: Uuid,
    pub status: ProofStatus,
    pub estimated_time_ms: Option<u64>,
    pub queue_position: Option<u32>,
}

/// Completed proof data
#[derive(Debug, Deserialize)]
pub struct CompletedProof {
    pub proof_id: Uuid,
    pub transaction_id: Uuid,
    pub proof: String,
    pub verification_key: String,
    pub public_inputs: Vec<String>,
    pub generation_time_ms: u64,
    pub circuit_type: String,
}

impl ProofServiceClient {
    /// Create new proof service client
    pub fn new(config: &Config) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: config.privacy_service_url.clone(), // Proof is handled by privacy service
        }
    }

    /// Request proof generation
    pub async fn generate(
        &self,
        request: GenerateProofRequest,
    ) -> Result<GenerateProofResponse> {
        let response = self
            .client
            .post(format!("{}/api/v1/proof/generate", self.base_url))
            .json(&request)
            .send()
            .await
            .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

        if response.status().is_success() {
            response
                .json()
                .await
                .map_err(|e| TransactionError::ExternalService(e.to_string()))
        } else {
            let error = response.text().await.unwrap_or_default();
            Err(TransactionError::ProofGenerationFailed(error))
        }
    }

    /// Get proof status
    pub async fn get_status(&self, proof_id: Uuid) -> Result<ProofStatus> {
        let response = self
            .client
            .get(format!("{}/api/v1/proof/{}/status", self.base_url, proof_id))
            .send()
            .await
            .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

        if response.status().is_success() {
            let data: serde_json::Value = response.json().await.unwrap_or_default();
            let status_str = data["status"].as_str().unwrap_or("queued");

            Ok(match status_str {
                "queued" => ProofStatus::Queued,
                "generating" => ProofStatus::Generating,
                "completed" => ProofStatus::Completed,
                "failed" => ProofStatus::Failed,
                "cancelled" => ProofStatus::Cancelled,
                _ => ProofStatus::Queued,
            })
        } else {
            Err(TransactionError::ExternalService(
                "Failed to get proof status".to_string(),
            ))
        }
    }

    /// Get completed proof
    pub async fn get_proof(&self, proof_id: Uuid) -> Result<CompletedProof> {
        let response = self
            .client
            .get(format!("{}/api/v1/proof/{}", self.base_url, proof_id))
            .send()
            .await
            .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

        if response.status().is_success() {
            response
                .json()
                .await
                .map_err(|e| TransactionError::ExternalService(e.to_string()))
        } else if response.status().as_u16() == 404 {
            Err(TransactionError::NotFound(proof_id.to_string()))
        } else {
            let error = response.text().await.unwrap_or_default();
            Err(TransactionError::ExternalService(error))
        }
    }

    /// Cancel proof generation
    pub async fn cancel(&self, proof_id: Uuid) -> Result<()> {
        let response = self
            .client
            .post(format!("{}/api/v1/proof/{}/cancel", self.base_url, proof_id))
            .send()
            .await
            .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

        if response.status().is_success() {
            Ok(())
        } else {
            let error = response.text().await.unwrap_or_default();
            Err(TransactionError::ExternalService(error))
        }
    }

    /// Verify a proof
    pub async fn verify(
        &self,
        proof: &str,
        verification_key: &str,
        public_inputs: &[String],
    ) -> Result<bool> {
        let response = self
            .client
            .post(format!("{}/api/v1/proof/verify", self.base_url))
            .json(&serde_json::json!({
                "proof": proof,
                "verification_key": verification_key,
                "public_inputs": public_inputs,
            }))
            .send()
            .await
            .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

        if response.status().is_success() {
            let data: serde_json::Value = response.json().await.unwrap_or_default();
            Ok(data["valid"].as_bool().unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    /// Batch proof generation
    pub async fn generate_batch(
        &self,
        requests: Vec<GenerateProofRequest>,
    ) -> Result<Vec<GenerateProofResponse>> {
        let response = self
            .client
            .post(format!("{}/api/v1/proof/batch", self.base_url))
            .json(&serde_json::json!({
                "proofs": requests,
            }))
            .send()
            .await
            .map_err(|e| TransactionError::ExternalService(e.to_string()))?;

        if response.status().is_success() {
            response
                .json()
                .await
                .map_err(|e| TransactionError::ExternalService(e.to_string()))
        } else {
            let error = response.text().await.unwrap_or_default();
            Err(TransactionError::ProofGenerationFailed(format!(
                "Batch proof generation failed: {}",
                error
            )))
        }
    }
}

/// Estimate proof generation time based on privacy level
pub fn estimate_generation_time_ms(privacy_level: i16, priority: ProofPriority) -> u64 {
    let base_time = match privacy_level {
        0 => 0,      // No proof needed
        1 => 1000,   // 1 second
        2 => 1000,   // 1 second
        3 => 2000,   // 2 seconds
        4 => 5000,   // 5 seconds
        5 => 10000,  // 10 seconds
        _ => 3000,
    };

    let multiplier = match priority {
        ProofPriority::High => 1.0,
        ProofPriority::Normal => 1.5,
        ProofPriority::Low => 3.0,
    };

    (base_time as f64 * multiplier) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_generation_time() {
        assert_eq!(estimate_generation_time_ms(0, ProofPriority::Normal), 0);
        assert_eq!(estimate_generation_time_ms(5, ProofPriority::High), 10000);
        assert!(estimate_generation_time_ms(5, ProofPriority::Low) > estimate_generation_time_ms(5, ProofPriority::High));
    }
}
