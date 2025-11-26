//! NexusZero client for interacting with the protocol

use crate::error::{NexusZeroError, Result};
use crate::privacy::{PrivacyEngine, PrivacyLevel};
use crate::types::{ProofResult, Transaction, TransactionRequest};

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Configuration for the NexusZero client
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub endpoint: String,
    pub api_key: Option<String>,
    pub timeout_ms: u64,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.nexuszero.io".to_string(),
            api_key: None,
            timeout_ms: 30000,
        }
    }
}

/// NexusZero Protocol client
#[derive(Debug, Clone)]
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct NexusZeroClient {
    #[cfg_attr(feature = "wasm", wasm_bindgen(skip))]
    config: ClientConfig,
    #[cfg_attr(feature = "wasm", wasm_bindgen(skip))]
    privacy_engine: PrivacyEngine,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl NexusZeroClient {
    /// Create a new client with default configuration
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        Self {
            config: ClientConfig::default(),
            privacy_engine: PrivacyEngine::new(),
        }
    }

    /// Get the configured endpoint
    pub fn endpoint(&self) -> String {
        self.config.endpoint.clone()
    }
}

impl NexusZeroClient {
    /// Create client with custom configuration
    pub fn with_config(config: ClientConfig) -> Self {
        Self {
            config,
            privacy_engine: PrivacyEngine::new(),
        }
    }

    /// Create a new transaction
    pub fn create_transaction(&self, request: TransactionRequest) -> Result<Transaction> {
        // Validate privacy level
        let _ = PrivacyLevel::from_u8(request.privacy_level)?;

        Ok(Transaction::from_request(&request))
    }

    /// Generate a zero-knowledge proof for a transaction
    pub fn generate_proof(&self, transaction: &Transaction) -> Result<ProofResult> {
        let level = PrivacyLevel::from_u8(transaction.privacy_level)?;
        let params = self.privacy_engine.get_parameters(transaction.privacy_level)?;

        // Simulate proof generation time
        let generation_time_ms = level.estimated_proof_time_ms() as u64;

        // Generate mock proof (real implementation would use actual ZK)
        let proof_data = format!(
            "{{\"strategy\":\"{}\",\"security_bits\":{}}}",
            params.proof_strategy, params.security_bits
        );
        let proof_hash = hex::encode(format!("proof_{}_{}", transaction.id, transaction.privacy_level));

        Ok(ProofResult {
            proof_hash,
            proof_data,
            verification_key: format!("vk_{}", transaction.id),
            generation_time_ms,
            privacy_level: transaction.privacy_level,
        })
    }

    /// Verify a zero-knowledge proof
    pub fn verify_proof(&self, proof: &ProofResult) -> Result<bool> {
        // Validate the privacy level
        let _ = PrivacyLevel::from_u8(proof.privacy_level)?;

        // In real implementation, this would verify the actual proof
        // For SDK, we validate structure
        if proof.proof_hash.is_empty() {
            return Err(NexusZeroError::ProofVerificationError(
                "Empty proof hash".to_string(),
            ));
        }

        Ok(true)
    }

    /// Get the privacy engine
    pub fn privacy_engine(&self) -> &PrivacyEngine {
        &self.privacy_engine
    }
}

impl Default for NexusZeroClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_creation() {
        let client = NexusZeroClient::new();
        assert_eq!(client.endpoint(), "https://api.nexuszero.io");
    }

    #[test]
    fn test_create_transaction() {
        let client = NexusZeroClient::new();
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            3,
        );
        let tx = client.create_transaction(request).unwrap();
        assert_eq!(tx.status, "pending");
        assert_eq!(tx.privacy_level, 3);
    }

    #[test]
    fn test_generate_and_verify_proof() {
        let client = NexusZeroClient::new();
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            4,
        );
        let tx = client.create_transaction(request).unwrap();
        let proof = client.generate_proof(&tx).unwrap();

        assert!(!proof.proof_hash.is_empty());
        assert!(client.verify_proof(&proof).unwrap());
    }

    #[test]
    fn test_invalid_privacy_level() {
        let client = NexusZeroClient::new();
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            10, // Invalid level
        );
        assert!(client.create_transaction(request).is_err());
    }
}
