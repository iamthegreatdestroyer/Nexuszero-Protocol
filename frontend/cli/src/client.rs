//! NexusZero API Client

use reqwest::{Client, Response, StatusCode};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::time::Duration;

use crate::error::{ApiError, CliError, CliResult};

/// Health check response
#[derive(Debug, Deserialize)]
pub struct HealthResponse {
    pub healthy: bool,
    pub version: String,
    pub network: String,
    pub prover_nodes: u32,
    pub pending_proofs: u32,
    pub supported_chains: Vec<String>,
}

/// Privacy shield request
#[derive(Debug, Serialize)]
pub struct ShieldRequest {
    pub amount: String,
    pub token: String,
    pub recipient_address: Option<String>,
}

/// Privacy shield response
#[derive(Debug, Deserialize)]
pub struct ShieldResponse {
    pub tx_hash: String,
    pub commitment: String,
    pub note: String,
    pub status: String,
}

/// Privacy unshield request
#[derive(Debug, Serialize)]
pub struct UnshieldRequest {
    pub note: String,
    pub recipient: String,
    pub amount: String,
}

/// Privacy unshield response
#[derive(Debug, Deserialize)]
pub struct UnshieldResponse {
    pub tx_hash: String,
    pub nullifier: String,
    pub status: String,
}

/// Private transfer request
#[derive(Debug, Serialize)]
pub struct TransferRequest {
    pub input_notes: Vec<String>,
    pub outputs: Vec<TransferOutput>,
}

#[derive(Debug, Serialize)]
pub struct TransferOutput {
    pub amount: String,
    pub recipient: String,
}

/// Transfer response
#[derive(Debug, Deserialize)]
pub struct TransferResponse {
    pub tx_hash: String,
    pub output_notes: Vec<String>,
    pub status: String,
}

/// Proof generation request
#[derive(Debug, Serialize)]
pub struct ProofRequest {
    pub proof_type: String,
    pub public_inputs: serde_json::Value,
    pub private_inputs: serde_json::Value,
}

/// Proof response
#[derive(Debug, Serialize, Deserialize)]
pub struct ProofResponse {
    pub proof_id: String,
    pub proof: String,
    pub public_signals: Vec<String>,
    pub verification_key: String,
    pub status: String,
}

/// Bridge transfer request
#[derive(Debug, Serialize)]
pub struct BridgeRequest {
    pub source_chain: String,
    pub dest_chain: String,
    pub amount: String,
    pub token: String,
    pub recipient: String,
    pub preserve_privacy: bool,
}

/// Bridge transfer response
#[derive(Debug, Deserialize)]
pub struct BridgeResponse {
    pub transfer_id: String,
    pub source_tx_hash: Option<String>,
    pub dest_tx_hash: Option<String>,
    pub status: String,
    pub confirmations: u32,
    pub required_confirmations: u32,
}

/// Compliance attestation request
#[derive(Debug, Serialize)]
pub struct AttestationRequest {
    pub attestation_type: String,
    pub jurisdiction: String,
    pub data_hash: String,
}

/// Attestation response
#[derive(Debug, Deserialize)]
pub struct AttestationResponse {
    pub attestation_id: String,
    pub commitment: String,
    pub proof: String,
    pub valid_until: String,
    pub status: String,
}

/// API Client for NexusZero
pub struct NexusZeroClient {
    client: Client,
    base_url: String,
}

impl NexusZeroClient {
    /// Create a new API client
    pub fn new(base_url: &str) -> CliResult<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(CliError::Network)?;

        Ok(Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
        })
    }

    /// Perform health check
    pub async fn health_check(&self) -> CliResult<HealthResponse> {
        self.get("/health").await
    }

    // Privacy operations
    pub async fn shield(&self, request: ShieldRequest) -> CliResult<ShieldResponse> {
        self.post("/api/v1/privacy/shield", &request).await
    }

    pub async fn unshield(&self, request: UnshieldRequest) -> CliResult<UnshieldResponse> {
        self.post("/api/v1/privacy/unshield", &request).await
    }

    pub async fn transfer(&self, request: TransferRequest) -> CliResult<TransferResponse> {
        self.post("/api/v1/privacy/transfer", &request).await
    }

    pub async fn get_balance(&self, commitment: &str) -> CliResult<serde_json::Value> {
        self.get(&format!("/api/v1/privacy/balance/{}", commitment)).await
    }

    // Proof operations
    pub async fn generate_proof(&self, request: ProofRequest) -> CliResult<ProofResponse> {
        self.post("/api/v1/proof/generate", &request).await
    }

    pub async fn verify_proof(&self, proof_id: &str) -> CliResult<serde_json::Value> {
        self.get(&format!("/api/v1/proof/verify/{}", proof_id)).await
    }

    pub async fn get_proof_status(&self, proof_id: &str) -> CliResult<ProofResponse> {
        self.get(&format!("/api/v1/proof/{}", proof_id)).await
    }

    // Bridge operations
    pub async fn bridge_transfer(&self, request: BridgeRequest) -> CliResult<BridgeResponse> {
        self.post("/api/v1/bridge/transfer", &request).await
    }

    pub async fn bridge_status(&self, transfer_id: &str) -> CliResult<BridgeResponse> {
        self.get(&format!("/api/v1/bridge/status/{}", transfer_id)).await
    }

    pub async fn list_supported_chains(&self) -> CliResult<Vec<String>> {
        self.get("/api/v1/bridge/chains").await
    }

    // Compliance operations
    pub async fn create_attestation(&self, request: AttestationRequest) -> CliResult<AttestationResponse> {
        self.post("/api/v1/compliance/attestation", &request).await
    }

    pub async fn verify_attestation(&self, attestation_id: &str) -> CliResult<serde_json::Value> {
        self.get(&format!("/api/v1/compliance/verify/{}", attestation_id)).await
    }

    // HTTP helpers
    async fn get<T: DeserializeOwned>(&self, path: &str) -> CliResult<T> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client.get(&url).send().await?;
        self.handle_response(response).await
    }

    async fn post<T: DeserializeOwned, B: Serialize>(&self, path: &str, body: &B) -> CliResult<T> {
        let url = format!("{}{}", self.base_url, path);
        let response = self.client.post(&url).json(body).send().await?;
        self.handle_response(response).await
    }

    async fn handle_response<T: DeserializeOwned>(&self, response: Response) -> CliResult<T> {
        let status = response.status();

        match status {
            StatusCode::OK | StatusCode::CREATED => {
                response.json().await.map_err(|e| {
                    CliError::Api(ApiError::InvalidResponse(e.to_string()))
                })
            }
            StatusCode::UNAUTHORIZED => {
                Err(CliError::AuthRequired)
            }
            StatusCode::NOT_FOUND => {
                Err(CliError::Api(ApiError::NotFound("Resource not found".to_string())))
            }
            StatusCode::TOO_MANY_REQUESTS => {
                let retry_after = response
                    .headers()
                    .get("retry-after")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(60);
                Err(CliError::Api(ApiError::RateLimited { retry_after }))
            }
            _ => {
                let message = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
                Err(CliError::Api(ApiError::RequestFailed {
                    status: status.as_u16(),
                    message,
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = NexusZeroClient::new("http://localhost:8080");
        assert!(client.is_ok());
    }

    #[test]
    fn test_client_url_normalization() {
        let client = NexusZeroClient::new("http://localhost:8080/").unwrap();
        assert_eq!(client.base_url, "http://localhost:8080");
    }
}
