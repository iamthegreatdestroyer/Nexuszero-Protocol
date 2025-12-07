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

    // ========================================================================
    // PRODUCTION HARDENING TESTS - Sprint 1.1 Phase 1.4
    // ========================================================================

    #[test]
    fn test_client_default_trait() {
        let client = NexusZeroClient::default();
        assert_eq!(client.endpoint(), "https://api.nexuszero.io");
    }

    #[test]
    fn test_client_with_custom_config() {
        let config = ClientConfig {
            endpoint: "https://custom.api.example.com".to_string(),
            api_key: Some("test_api_key_12345".to_string()),
            timeout_ms: 60000,
        };
        let client = NexusZeroClient::with_config(config);
        assert_eq!(client.endpoint(), "https://custom.api.example.com");
    }

    #[test]
    fn test_all_valid_privacy_levels() {
        let client = NexusZeroClient::new();
        
        for level in 0..=5 {
            let request = TransactionRequest::new(
                "0xsender".to_string(),
                "0xrecipient".to_string(),
                "1000".to_string(),
                level,
            );
            let result = client.create_transaction(request);
            assert!(result.is_ok(), "Privacy level {} should be valid", level);
        }
    }

    #[test]
    fn test_all_invalid_privacy_levels() {
        let client = NexusZeroClient::new();
        
        for level in 6..=255 {
            let request = TransactionRequest::new(
                "0xsender".to_string(),
                "0xrecipient".to_string(),
                "1000".to_string(),
                level,
            );
            let result = client.create_transaction(request);
            assert!(result.is_err(), "Privacy level {} should be invalid", level);
        }
    }

    #[test]
    fn test_proof_generation_all_levels() {
        let client = NexusZeroClient::new();
        
        for level in 0..=5 {
            let request = TransactionRequest::new(
                "0xsender".to_string(),
                "0xrecipient".to_string(),
                "1000".to_string(),
                level,
            );
            let tx = client.create_transaction(request).unwrap();
            let proof = client.generate_proof(&tx);
            assert!(proof.is_ok(), "Proof generation for level {} should work", level);
            
            let proof = proof.unwrap();
            assert_eq!(proof.privacy_level, level);
            assert!(!proof.proof_hash.is_empty());
            assert!(!proof.proof_data.is_empty());
            assert!(!proof.verification_key.is_empty());
        }
    }

    #[test]
    fn test_proof_verification_structure() {
        let client = NexusZeroClient::new();
        
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            3,
        );
        let tx = client.create_transaction(request).unwrap();
        let proof = client.generate_proof(&tx).unwrap();
        
        // Verify the proof structure
        assert!(proof.proof_data.contains("strategy"));
        assert!(proof.proof_data.contains("security_bits"));
    }

    #[test]
    fn test_empty_proof_hash_verification_fails() {
        let client = NexusZeroClient::new();
        
        let empty_proof = ProofResult {
            proof_hash: "".to_string(),
            proof_data: "{}".to_string(),
            verification_key: "vk_test".to_string(),
            generation_time_ms: 100,
            privacy_level: 3,
        };
        
        let result = client.verify_proof(&empty_proof);
        assert!(result.is_err(), "Empty proof hash should fail verification");
    }

    #[test]
    fn test_transaction_uniqueness() {
        let client = NexusZeroClient::new();
        
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            3,
        );
        
        // Create multiple transactions from the same request
        let tx1 = client.create_transaction(request.clone()).unwrap();
        let tx2 = client.create_transaction(request.clone()).unwrap();
        
        // Each transaction should have a unique ID
        assert_ne!(tx1.id, tx2.id, "Transactions should have unique IDs");
    }

    #[test]
    fn test_transaction_timestamps() {
        let client = NexusZeroClient::new();
        
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            3,
        );
        
        let tx = client.create_transaction(request).unwrap();
        
        // Timestamp should be non-zero and reasonable
        assert!(tx.timestamp > 0, "Timestamp should be positive");
        // Should be within last hour (3600 seconds)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        assert!(now - tx.timestamp < 3600, "Timestamp should be recent");
    }

    #[test]
    fn test_proof_generation_times_increase_with_level() {
        let client = NexusZeroClient::new();
        
        let mut prev_time: u64 = 0;
        
        for level in 1..=5 {
            let request = TransactionRequest::new(
                "0xsender".to_string(),
                "0xrecipient".to_string(),
                "1000".to_string(),
                level,
            );
            let tx = client.create_transaction(request).unwrap();
            let proof = client.generate_proof(&tx).unwrap();
            
            // Higher privacy levels should take more time
            assert!(proof.generation_time_ms >= prev_time,
                "Level {} time ({}) should be >= level {} time ({})",
                level, proof.generation_time_ms, level - 1, prev_time);
            prev_time = proof.generation_time_ms;
        }
    }

    #[test]
    fn test_client_config_defaults() {
        let config = ClientConfig::default();
        
        assert_eq!(config.endpoint, "https://api.nexuszero.io");
        assert!(config.api_key.is_none());
        assert_eq!(config.timeout_ms, 30000);
    }

    #[test]
    fn test_privacy_engine_access() {
        let client = NexusZeroClient::new();
        let engine = client.privacy_engine();
        
        // Should be able to get parameters for all valid levels
        for level in 0..=5 {
            let result = engine.get_parameters(level);
            assert!(result.is_ok(), "Should get parameters for level {}", level);
        }
    }

    #[test]
    fn test_transaction_preserves_request_data() {
        let client = NexusZeroClient::new();
        
        let from = "0xsender_address_12345".to_string();
        let to = "0xrecipient_address_67890".to_string();
        let amount = "999999999".to_string();
        let privacy_level = 4u8;
        
        let request = TransactionRequest::new(
            from.clone(),
            to.clone(),
            amount.clone(),
            privacy_level,
        );
        
        let tx = client.create_transaction(request).unwrap();
        
        assert_eq!(tx.from, from);
        assert_eq!(tx.to, to);
        assert_eq!(tx.amount, amount);
        assert_eq!(tx.privacy_level, privacy_level);
    }

    #[test]
    fn test_concurrent_client_usage() {
        use std::sync::Arc;
        use std::thread;
        
        let client = Arc::new(NexusZeroClient::new());
        let mut handles = vec![];
        
        for i in 0..4 {
            let client_clone = Arc::clone(&client);
            let handle = thread::spawn(move || {
                let request = TransactionRequest::new(
                    format!("0xsender_{}", i),
                    format!("0xrecipient_{}", i),
                    format!("{}", i * 1000),
                    (i % 6) as u8,
                );
                let tx = client_clone.create_transaction(request).unwrap();
                let proof = client_clone.generate_proof(&tx).unwrap();
                client_clone.verify_proof(&proof).unwrap()
            });
            handles.push(handle);
        }
        
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result, "Concurrent proof should verify");
        }
    }

    #[test]
    fn test_transaction_serialization_roundtrip() {
        let client = NexusZeroClient::new();
        
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            3,
        );
        let tx = client.create_transaction(request).unwrap();
        
        // Serialize
        let serialized = serde_json::to_string(&tx).expect("Serialization should work");
        
        // Deserialize
        let deserialized: Transaction = serde_json::from_str(&serialized)
            .expect("Deserialization should work");
        
        assert_eq!(tx.id, deserialized.id);
        assert_eq!(tx.from, deserialized.from);
        assert_eq!(tx.to, deserialized.to);
        assert_eq!(tx.amount, deserialized.amount);
        assert_eq!(tx.privacy_level, deserialized.privacy_level);
    }

    #[test]
    fn test_proof_serialization_roundtrip() {
        let client = NexusZeroClient::new();
        
        let request = TransactionRequest::new(
            "0xsender".to_string(),
            "0xrecipient".to_string(),
            "1000".to_string(),
            4,
        );
        let tx = client.create_transaction(request).unwrap();
        let proof = client.generate_proof(&tx).unwrap();
        
        // Serialize
        let serialized = serde_json::to_string(&proof).expect("Serialization should work");
        
        // Deserialize
        let deserialized: ProofResult = serde_json::from_str(&serialized)
            .expect("Deserialization should work");
        
        assert_eq!(proof.proof_hash, deserialized.proof_hash);
        assert_eq!(proof.privacy_level, deserialized.privacy_level);
    }
}
