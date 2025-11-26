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
}
