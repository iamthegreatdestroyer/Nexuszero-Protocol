//! Core types for NexusZero SDK

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Transaction request for creating a new transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct TransactionRequest {
    pub from: String,
    pub to: String,
    pub amount: String,
    pub privacy_level: u8,
    pub memo: Option<String>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl TransactionRequest {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(from: String, to: String, amount: String, privacy_level: u8) -> Self {
        Self {
            from,
            to,
            amount,
            privacy_level,
            memo: None,
        }
    }

    pub fn with_memo(mut self, memo: String) -> Self {
        self.memo = Some(memo);
        self
    }
}

/// A transaction in the NexusZero protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct Transaction {
    pub id: String,
    pub from: String,
    pub to: String,
    pub amount: String,
    pub privacy_level: u8,
    pub status: String,
    pub proof_hash: Option<String>,
    pub timestamp: u64,
}

impl Transaction {
    /// Create a new transaction from a request
    pub fn from_request(request: &TransactionRequest) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from: request.from.clone(),
            to: request.to.clone(),
            amount: request.amount.clone(),
            privacy_level: request.privacy_level,
            status: "pending".to_string(),
            proof_hash: None,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
        }
    }
}

/// Result of proof generation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct ProofResult {
    pub proof_hash: String,
    pub proof_data: String,
    pub verification_key: String,
    pub generation_time_ms: u64,
    pub privacy_level: u8,
}

/// Cross-chain bridge request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct BridgeRequest {
    pub source_chain: String,
    pub target_chain: String,
    pub amount: String,
    pub recipient: String,
    pub privacy_level: u8,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl BridgeRequest {
    #[cfg_attr(feature = "wasm", wasm_bindgen(constructor))]
    pub fn new(
        source_chain: String,
        target_chain: String,
        amount: String,
        recipient: String,
        privacy_level: u8,
    ) -> Self {
        Self {
            source_chain,
            target_chain,
            amount,
            recipient,
            privacy_level,
        }
    }
}

/// Bridge transfer result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct BridgeResult {
    pub transfer_id: String,
    pub source_tx_hash: String,
    pub target_tx_hash: Option<String>,
    pub status: String,
    pub privacy_proof: String,
}

/// Compliance proof for regulatory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "wasm", wasm_bindgen(getter_with_clone))]
pub struct ComplianceProof {
    pub proof_id: String,
    pub proof_type: String,
    pub proof_data: String,
    pub issuer: String,
    pub timestamp: u64,
    pub expires_at: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transaction_request() {
        let req = TransactionRequest::new(
            "0xabc".to_string(),
            "0xdef".to_string(),
            "1000".to_string(),
            3,
        );
        assert_eq!(req.from, "0xabc");
        assert_eq!(req.privacy_level, 3);
    }

    #[test]
    fn test_transaction_from_request() {
        let req = TransactionRequest::new(
            "0xabc".to_string(),
            "0xdef".to_string(),
            "1000".to_string(),
            4,
        );
        let tx = Transaction::from_request(&req);
        assert!(!tx.id.is_empty());
        assert_eq!(tx.status, "pending");
    }
}
