//! Common types for chain connectors

use serde::{Deserialize, Serialize};
use std::fmt;

/// Supported blockchain identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChainId {
    /// Ethereum Mainnet
    Ethereum,
    /// Bitcoin Mainnet
    Bitcoin,
    /// Solana Mainnet
    Solana,
    /// Polygon PoS
    Polygon,
    /// Cosmos Hub
    Cosmos,
    /// Arbitrum One
    Arbitrum,
    /// Optimism
    Optimism,
    /// Base
    Base,
    /// Avalanche C-Chain
    Avalanche,
    /// BNB Smart Chain
    BnbChain,
    /// Custom chain with numeric ID
    Custom(u64),
}

impl ChainId {
    /// Get the EVM chain ID if applicable
    pub fn evm_chain_id(&self) -> Option<u64> {
        match self {
            ChainId::Ethereum => Some(1),
            ChainId::Polygon => Some(137),
            ChainId::Arbitrum => Some(42161),
            ChainId::Optimism => Some(10),
            ChainId::Base => Some(8453),
            ChainId::Avalanche => Some(43114),
            ChainId::BnbChain => Some(56),
            ChainId::Custom(id) => Some(*id),
            _ => None,
        }
    }

    /// Check if this is an EVM-compatible chain
    pub fn is_evm(&self) -> bool {
        self.evm_chain_id().is_some()
    }

    /// Get the native currency symbol
    pub fn native_symbol(&self) -> &'static str {
        match self {
            ChainId::Ethereum => "ETH",
            ChainId::Bitcoin => "BTC",
            ChainId::Solana => "SOL",
            ChainId::Polygon => "MATIC",
            ChainId::Cosmos => "ATOM",
            ChainId::Arbitrum => "ETH",
            ChainId::Optimism => "ETH",
            ChainId::Base => "ETH",
            ChainId::Avalanche => "AVAX",
            ChainId::BnbChain => "BNB",
            ChainId::Custom(_) => "NATIVE",
        }
    }

    /// Get the number of decimal places for the native currency
    pub fn decimals(&self) -> u8 {
        match self {
            ChainId::Bitcoin => 8,
            ChainId::Solana => 9,
            _ => 18, // Most EVM chains use 18 decimals
        }
    }
}

impl fmt::Display for ChainId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChainId::Custom(id) => write!(f, "custom:{}", id),
            _ => write!(f, "{:?}", self).map(|_| ()).and_then(|_| {
                write!(f, "{}", format!("{:?}", self).to_lowercase())
            }),
        }
    }
}

/// Proof metadata for on-chain submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Privacy level (0-5)
    pub privacy_level: u8,
    /// Type of proof (groth16, plonk, bulletproofs, etc.)
    pub proof_type: String,
    /// Unix timestamp of proof generation
    pub timestamp: u64,
    /// Sender commitment hash
    pub sender_commitment: [u8; 32],
    /// Recipient commitment hash
    pub recipient_commitment: [u8; 32],
    /// Optional nullifier for double-spend prevention
    pub nullifier: Option<[u8; 32]>,
    /// Additional metadata
    #[serde(default)]
    pub extra: serde_json::Value,
}

impl ProofMetadata {
    /// Create new proof metadata
    pub fn new(
        privacy_level: u8,
        proof_type: impl Into<String>,
        sender_commitment: [u8; 32],
        recipient_commitment: [u8; 32],
    ) -> Self {
        Self {
            privacy_level,
            proof_type: proof_type.into(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            sender_commitment,
            recipient_commitment,
            nullifier: None,
            extra: serde_json::Value::Null,
        }
    }

    /// Set nullifier
    pub fn with_nullifier(mut self, nullifier: [u8; 32]) -> Self {
        self.nullifier = Some(nullifier);
        self
    }
}

/// Transaction receipt from chain submission
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    /// Transaction hash
    pub tx_hash: [u8; 32],
    /// Block number where transaction was included
    pub block_number: u64,
    /// Block hash
    pub block_hash: Option<[u8; 32]>,
    /// Gas/compute units used
    pub gas_used: u64,
    /// Transaction status (true = success)
    pub status: bool,
    /// Event logs emitted
    pub logs: Vec<EventLog>,
    /// Effective gas price (for EVM chains)
    pub effective_gas_price: Option<u128>,
    /// Transaction index in block
    pub transaction_index: u32,
}

impl TransactionReceipt {
    /// Get transaction hash as hex string
    pub fn tx_hash_hex(&self) -> String {
        hex::encode(self.tx_hash)
    }
}

/// Event log from transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventLog {
    /// Contract/program address that emitted the event
    pub address: Vec<u8>,
    /// Event topics (first topic is usually event signature)
    pub topics: Vec<[u8; 32]>,
    /// Event data payload
    pub data: Vec<u8>,
    /// Log index in the transaction
    pub log_index: u32,
}

/// Transaction status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TransactionStatus {
    /// Transaction is pending in mempool
    Pending,
    /// Transaction is confirmed on-chain
    Confirmed,
    /// Transaction failed/reverted
    Failed,
    /// Transaction was dropped from mempool
    Dropped,
    /// Status unknown
    Unknown,
}

impl TransactionStatus {
    /// Check if transaction is finalized (success or failure)
    pub fn is_final(&self) -> bool {
        matches!(self, TransactionStatus::Confirmed | TransactionStatus::Failed | TransactionStatus::Dropped)
    }
}

/// Chain operation types for fee estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChainOperation {
    /// Submit a privacy proof
    SubmitProof {
        /// Size of the proof in bytes
        proof_size: usize,
        /// Privacy level affects verification cost
        privacy_level: u8,
    },
    /// Verify an existing proof
    VerifyProof {
        /// Proof ID to verify
        proof_id: [u8; 32],
    },
    /// Native token transfer
    Transfer {
        /// Amount in smallest units
        amount: u128,
        /// Recipient address
        recipient: Vec<u8>,
    },
    /// Initiate cross-chain bridge transfer
    BridgeInitiate {
        /// Target chain
        target_chain: ChainId,
        /// Amount to bridge
        amount: u128,
    },
    /// Complete cross-chain bridge transfer
    BridgeComplete {
        /// Transfer ID
        transfer_id: [u8; 32],
    },
    /// Contract deployment
    Deploy {
        /// Bytecode size
        bytecode_size: usize,
    },
    /// Generic contract call
    ContractCall {
        /// Estimated calldata size
        calldata_size: usize,
    },
}

/// Fee estimation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeEstimate {
    /// Gas/compute units required
    pub gas_units: u64,
    /// Gas price in gwei (for EVM) or equivalent
    pub gas_price: f64,
    /// Priority fee (for EIP-1559)
    pub priority_fee: Option<f64>,
    /// Total fee in native currency
    pub total_fee_native: f64,
    /// Total fee in USD (if price available)
    pub total_fee_usd: Option<f64>,
    /// Confidence level (low, medium, high)
    pub confidence: FeeConfidence,
}

/// Fee estimation confidence level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeeConfidence {
    /// Low confidence - might be underestimated
    Low,
    /// Medium confidence - reasonable estimate
    Medium,
    /// High confidence - likely accurate
    High,
}

/// Address type for cross-chain compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainAddress {
    /// Raw address bytes
    pub bytes: Vec<u8>,
    /// Chain this address belongs to
    pub chain: ChainId,
    /// Human-readable format (if available)
    pub display: Option<String>,
}

impl ChainAddress {
    /// Create new chain address
    pub fn new(bytes: Vec<u8>, chain: ChainId) -> Self {
        Self {
            bytes,
            chain,
            display: None,
        }
    }

    /// Create with display string
    pub fn with_display(mut self, display: impl Into<String>) -> Self {
        self.display = Some(display.into());
        self
    }

    /// Get hex representation
    pub fn to_hex(&self) -> String {
        hex::encode(&self.bytes)
    }
}

/// Block information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockInfo {
    /// Block number/height
    pub number: u64,
    /// Block hash
    pub hash: [u8; 32],
    /// Parent block hash
    pub parent_hash: [u8; 32],
    /// Block timestamp
    pub timestamp: u64,
    /// Number of transactions in block
    pub transaction_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_id_evm() {
        assert_eq!(ChainId::Ethereum.evm_chain_id(), Some(1));
        assert_eq!(ChainId::Polygon.evm_chain_id(), Some(137));
        assert_eq!(ChainId::Bitcoin.evm_chain_id(), None);
        assert_eq!(ChainId::Solana.evm_chain_id(), None);
    }

    #[test]
    fn test_chain_id_is_evm() {
        assert!(ChainId::Ethereum.is_evm());
        assert!(ChainId::Arbitrum.is_evm());
        assert!(!ChainId::Bitcoin.is_evm());
        assert!(!ChainId::Solana.is_evm());
    }

    #[test]
    fn test_transaction_status_final() {
        assert!(!TransactionStatus::Pending.is_final());
        assert!(TransactionStatus::Confirmed.is_final());
        assert!(TransactionStatus::Failed.is_final());
    }

    #[test]
    fn test_proof_metadata_creation() {
        let meta = ProofMetadata::new(3, "groth16", [0u8; 32], [1u8; 32]);
        assert_eq!(meta.privacy_level, 3);
        assert_eq!(meta.proof_type, "groth16");
        assert!(meta.nullifier.is_none());
    }

    // ===== HARDENING TESTS =====

    #[test]
    fn test_chain_id_all_evm_chains() {
        // Test all EVM chains
        let evm_chains = [
            (ChainId::Ethereum, 1u64),
            (ChainId::Polygon, 137),
            (ChainId::Arbitrum, 42161),
            (ChainId::Optimism, 10),
            (ChainId::Base, 8453),
            (ChainId::Avalanche, 43114),
            (ChainId::BnbChain, 56),
        ];
        
        for (chain, expected_id) in evm_chains {
            assert!(chain.is_evm(), "Chain {:?} should be EVM", chain);
            assert_eq!(chain.evm_chain_id(), Some(expected_id), "Chain {:?} has wrong ID", chain);
        }
    }

    #[test]
    fn test_chain_id_non_evm_chains() {
        let non_evm_chains = [ChainId::Bitcoin, ChainId::Solana, ChainId::Cosmos];
        
        for chain in non_evm_chains {
            assert!(!chain.is_evm(), "Chain {:?} should not be EVM", chain);
            assert!(chain.evm_chain_id().is_none(), "Chain {:?} should have no EVM ID", chain);
        }
    }

    #[test]
    fn test_chain_id_custom() {
        let custom_chain = ChainId::Custom(12345);
        assert!(custom_chain.is_evm());
        assert_eq!(custom_chain.evm_chain_id(), Some(12345));
    }

    #[test]
    fn test_chain_id_native_symbols() {
        assert_eq!(ChainId::Ethereum.native_symbol(), "ETH");
        assert_eq!(ChainId::Bitcoin.native_symbol(), "BTC");
        assert_eq!(ChainId::Solana.native_symbol(), "SOL");
        assert_eq!(ChainId::Polygon.native_symbol(), "MATIC");
        assert_eq!(ChainId::Cosmos.native_symbol(), "ATOM");
        assert_eq!(ChainId::Avalanche.native_symbol(), "AVAX");
        assert_eq!(ChainId::BnbChain.native_symbol(), "BNB");
        assert_eq!(ChainId::Custom(999).native_symbol(), "NATIVE");
    }

    #[test]
    fn test_chain_id_decimals() {
        assert_eq!(ChainId::Bitcoin.decimals(), 8);
        assert_eq!(ChainId::Solana.decimals(), 9);
        assert_eq!(ChainId::Ethereum.decimals(), 18);
        assert_eq!(ChainId::Polygon.decimals(), 18);
        assert_eq!(ChainId::Cosmos.decimals(), 18);
    }

    #[test]
    fn test_proof_metadata_with_nullifier() {
        let meta = ProofMetadata::new(5, "plonk", [0u8; 32], [1u8; 32])
            .with_nullifier([42u8; 32]);
        
        assert_eq!(meta.privacy_level, 5);
        assert_eq!(meta.proof_type, "plonk");
        assert!(meta.nullifier.is_some());
        assert_eq!(meta.nullifier.unwrap(), [42u8; 32]);
    }

    #[test]
    fn test_proof_metadata_timestamp_auto_set() {
        let before = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let meta = ProofMetadata::new(1, "bulletproofs", [0u8; 32], [1u8; 32]);
        
        let after = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        assert!(meta.timestamp >= before && meta.timestamp <= after);
    }

    #[test]
    fn test_proof_metadata_all_privacy_levels() {
        for level in 0..=5 {
            let meta = ProofMetadata::new(level, "groth16", [0u8; 32], [1u8; 32]);
            assert_eq!(meta.privacy_level, level);
        }
    }

    #[test]
    fn test_transaction_receipt_tx_hash_hex() {
        let receipt = TransactionReceipt {
            tx_hash: [0xab; 32],
            block_number: 12345,
            block_hash: Some([0xcd; 32]),
            gas_used: 21000,
            status: true,
            logs: vec![],
            effective_gas_price: Some(1_000_000_000),
            transaction_index: 5,
        };
        
        let hex = receipt.tx_hash_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_transaction_status_all_variants() {
        let statuses = [
            TransactionStatus::Pending,
            TransactionStatus::Confirmed,
            TransactionStatus::Failed,
            TransactionStatus::Dropped,
            TransactionStatus::Unknown,
        ];
        
        // Pending and Unknown are not final
        assert!(!TransactionStatus::Pending.is_final());
        assert!(!TransactionStatus::Unknown.is_final());
        
        // Confirmed, Failed, Dropped are final
        assert!(TransactionStatus::Confirmed.is_final());
        assert!(TransactionStatus::Failed.is_final());
        assert!(TransactionStatus::Dropped.is_final());
    }

    #[test]
    fn test_chain_operation_submit_proof() {
        let op = ChainOperation::SubmitProof {
            proof_size: 1024,
            privacy_level: 3,
        };
        
        match op {
            ChainOperation::SubmitProof { proof_size, privacy_level } => {
                assert_eq!(proof_size, 1024);
                assert_eq!(privacy_level, 3);
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_chain_operation_all_variants() {
        let ops = vec![
            ChainOperation::SubmitProof { proof_size: 100, privacy_level: 1 },
            ChainOperation::VerifyProof { proof_id: [0u8; 32] },
            ChainOperation::Transfer { amount: 1000, recipient: vec![1, 2, 3] },
            ChainOperation::BridgeInitiate { target_chain: ChainId::Polygon, amount: 5000 },
            ChainOperation::BridgeComplete { transfer_id: [1u8; 32] },
            ChainOperation::Deploy { bytecode_size: 2048 },
            ChainOperation::ContractCall { calldata_size: 256 },
        ];
        
        assert_eq!(ops.len(), 7);
    }

    #[test]
    fn test_fee_estimate_confidence_levels() {
        let estimate = FeeEstimate {
            gas_units: 21000,
            gas_price: 20.0,
            priority_fee: Some(2.0),
            total_fee_native: 0.00042,
            total_fee_usd: Some(1.05),
            confidence: FeeConfidence::High,
        };
        
        assert_eq!(estimate.confidence, FeeConfidence::High);
        assert!(estimate.priority_fee.is_some());
        assert!(estimate.total_fee_usd.is_some());
    }

    #[test]
    fn test_chain_address_creation() {
        let addr = ChainAddress::new(vec![0x12, 0x34, 0x56], ChainId::Ethereum);
        assert_eq!(addr.bytes, vec![0x12, 0x34, 0x56]);
        assert_eq!(addr.chain, ChainId::Ethereum);
        assert!(addr.display.is_none());
    }

    #[test]
    fn test_chain_address_with_display() {
        let addr = ChainAddress::new(vec![0x12, 0x34], ChainId::Bitcoin)
            .with_display("bc1qtest...");
        
        assert!(addr.display.is_some());
        assert_eq!(addr.display.unwrap(), "bc1qtest...");
    }

    #[test]
    fn test_chain_address_to_hex() {
        let addr = ChainAddress::new(vec![0xde, 0xad, 0xbe, 0xef], ChainId::Ethereum);
        assert_eq!(addr.to_hex(), "deadbeef");
    }

    #[test]
    fn test_block_info_creation() {
        let block = BlockInfo {
            number: 1_000_000,
            hash: [0xaa; 32],
            parent_hash: [0xbb; 32],
            timestamp: 1609459200,
            transaction_count: 150,
        };
        
        assert_eq!(block.number, 1_000_000);
        assert_ne!(block.hash, block.parent_hash);
        assert!(block.transaction_count > 0);
    }

    #[test]
    fn test_event_log_structure() {
        let log = EventLog {
            address: vec![0x12; 20],
            topics: vec![[0xab; 32], [0xcd; 32]],
            data: vec![1, 2, 3, 4],
            log_index: 0,
        };
        
        assert_eq!(log.address.len(), 20);
        assert_eq!(log.topics.len(), 2);
        assert_eq!(log.data.len(), 4);
    }

    #[test]
    fn test_chain_id_serde_roundtrip() {
        let chains = [
            ChainId::Ethereum,
            ChainId::Bitcoin,
            ChainId::Custom(42),
        ];
        
        for chain in chains {
            let json = serde_json::to_string(&chain).unwrap();
            let parsed: ChainId = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, chain);
        }
    }

    #[test]
    fn test_proof_metadata_serde_roundtrip() {
        let meta = ProofMetadata::new(3, "groth16", [1u8; 32], [2u8; 32])
            .with_nullifier([3u8; 32]);
        
        let json = serde_json::to_string(&meta).unwrap();
        let parsed: ProofMetadata = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.privacy_level, meta.privacy_level);
        assert_eq!(parsed.proof_type, meta.proof_type);
        assert_eq!(parsed.nullifier, meta.nullifier);
    }

    #[test]
    fn test_transaction_receipt_serde_roundtrip() {
        let receipt = TransactionReceipt {
            tx_hash: [0x11; 32],
            block_number: 999,
            block_hash: Some([0x22; 32]),
            gas_used: 50000,
            status: true,
            logs: vec![],
            effective_gas_price: Some(10_000_000_000),
            transaction_index: 10,
        };
        
        let json = serde_json::to_string(&receipt).unwrap();
        let parsed: TransactionReceipt = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.tx_hash, receipt.tx_hash);
        assert_eq!(parsed.status, receipt.status);
    }

    #[test]
    fn test_fee_confidence_serde() {
        for conf in [FeeConfidence::Low, FeeConfidence::Medium, FeeConfidence::High] {
            let json = serde_json::to_string(&conf).unwrap();
            let parsed: FeeConfidence = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, conf);
        }
    }

    #[test]
    fn test_transaction_status_serde() {
        let statuses = [
            TransactionStatus::Pending,
            TransactionStatus::Confirmed,
            TransactionStatus::Failed,
            TransactionStatus::Dropped,
            TransactionStatus::Unknown,
        ];
        
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let parsed: TransactionStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, status);
        }
    }
}
