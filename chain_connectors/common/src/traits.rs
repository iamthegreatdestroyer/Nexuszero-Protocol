//! Chain connector trait definition

use async_trait::async_trait;
use crate::error::ChainError;
use crate::types::*;
use crate::events::{EventFilter, EventStream};

/// Unified trait for all blockchain connectors
///
/// This trait provides a consistent interface for interacting with different
/// blockchains, abstracting away chain-specific details while exposing
/// NexusZero-specific functionality.
#[async_trait]
pub trait ChainConnector: Send + Sync {
    /// Get the chain identifier
    fn chain_id(&self) -> ChainId;

    /// Get the human-readable chain name
    fn chain_name(&self) -> &str;

    /// Check if the connector is connected and healthy
    async fn is_healthy(&self) -> bool;

    /// Get current block number/height
    async fn get_block_number(&self) -> Result<u64, ChainError>;

    /// Get block information by number
    async fn get_block(&self, block_number: u64) -> Result<BlockInfo, ChainError>;

    /// Submit a privacy proof to the chain
    ///
    /// This submits the ZK proof and associated metadata to the on-chain
    /// verifier contract for permanent storage and verification.
    async fn submit_proof(
        &self,
        proof: &[u8],
        metadata: &ProofMetadata,
    ) -> Result<TransactionReceipt, ChainError>;

    /// Verify a proof that was previously submitted on-chain
    ///
    /// Returns true if the proof is valid and verified on-chain.
    async fn verify_proof(&self, proof_id: &[u8; 32]) -> Result<bool, ChainError>;

    /// Get details of a submitted proof
    async fn get_proof_details(
        &self,
        proof_id: &[u8; 32],
    ) -> Result<Option<OnChainProof>, ChainError>;

    /// Subscribe to blockchain events
    ///
    /// Returns a stream of events matching the provided filter.
    async fn subscribe_events(&self, filter: EventFilter) -> Result<EventStream, ChainError>;

    /// Get transaction status by hash
    async fn get_transaction_status(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<TransactionStatus, ChainError>;

    /// Get full transaction receipt
    async fn get_transaction_receipt(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<Option<TransactionReceipt>, ChainError>;

    /// Estimate fee for an operation
    async fn estimate_fee(&self, operation: ChainOperation) -> Result<FeeEstimate, ChainError>;

    /// Get native token balance for an address
    async fn get_balance(&self, address: &[u8]) -> Result<u128, ChainError>;

    /// Get the verifier contract address
    fn verifier_address(&self) -> Option<&[u8]>;

    /// Get the bridge contract address
    fn bridge_address(&self) -> Option<&[u8]>;

    /// Wait for transaction confirmation with timeout
    async fn wait_for_confirmation(
        &self,
        tx_hash: &[u8; 32],
        confirmations: u32,
        timeout_secs: u64,
    ) -> Result<TransactionReceipt, ChainError> {
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(timeout_secs);

        loop {
            if start.elapsed() > timeout {
                return Err(ChainError::TransactionTimeout(timeout_secs));
            }

            match self.get_transaction_receipt(tx_hash).await? {
                Some(receipt) if receipt.status => {
                    // Check confirmations
                    let current_block = self.get_block_number().await?;
                    let tx_block = receipt.block_number;
                    
                    if current_block >= tx_block + confirmations as u64 {
                        return Ok(receipt);
                    }
                }
                Some(receipt) if !receipt.status => {
                    return Err(ChainError::TransactionFailed(
                        format!("Transaction reverted at block {}", receipt.block_number)
                    ));
                }
                _ => {}
            }

            // Wait before next poll
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }
}

/// On-chain proof data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OnChainProof {
    /// Proof ID (hash)
    pub id: [u8; 32],
    /// Privacy level
    pub privacy_level: u8,
    /// Proof type
    pub proof_type: String,
    /// Submission timestamp
    pub timestamp: u64,
    /// Whether the proof is verified
    pub verified: bool,
    /// Submitter address
    pub submitter: Vec<u8>,
    /// Block number of submission
    pub block_number: u64,
}

/// Extension trait for bridge operations
#[async_trait]
pub trait BridgeConnector: ChainConnector {
    /// Initiate a cross-chain transfer
    async fn initiate_bridge_transfer(
        &self,
        target_chain: ChainId,
        proof: &[u8],
        recipient: &[u8],
        amount: u128,
    ) -> Result<BridgeTransfer, ChainError>;

    /// Complete a bridge transfer (on target chain)
    async fn complete_bridge_transfer(
        &self,
        transfer_id: &[u8; 32],
        relayer_proof: &[u8],
    ) -> Result<TransactionReceipt, ChainError>;

    /// Get bridge transfer status
    async fn get_bridge_transfer_status(
        &self,
        transfer_id: &[u8; 32],
    ) -> Result<BridgeTransferStatus, ChainError>;

    /// Get supported target chains for bridging
    fn supported_bridge_targets(&self) -> Vec<ChainId>;
}

/// Bridge transfer data
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BridgeTransfer {
    /// Transfer ID
    pub id: [u8; 32],
    /// Source chain
    pub source_chain: ChainId,
    /// Target chain
    pub target_chain: ChainId,
    /// Amount being transferred
    pub amount: u128,
    /// Recipient on target chain
    pub recipient: Vec<u8>,
    /// Proof data
    pub proof_id: [u8; 32],
    /// Source transaction hash
    pub source_tx_hash: [u8; 32],
    /// Status
    pub status: BridgeTransferStatus,
}

/// Bridge transfer status
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BridgeTransferStatus {
    /// Transfer initiated on source chain
    Initiated,
    /// Transfer is being relayed
    Relaying,
    /// Transfer is ready to complete on target chain
    ReadyToComplete,
    /// Transfer completed successfully
    Completed,
    /// Transfer failed
    Failed,
    /// Transfer expired
    Expired,
}

/// Extension trait for HTLC (Hashed Time-Locked Contract) operations
#[async_trait]
pub trait HtlcConnector: ChainConnector {
    /// Create a new HTLC
    async fn create_htlc(
        &self,
        recipient: &[u8],
        hash_lock: &[u8; 32],
        timeout_blocks: u64,
        amount: u128,
    ) -> Result<HtlcInfo, ChainError>;

    /// Redeem an HTLC with the preimage
    async fn redeem_htlc(
        &self,
        htlc_id: &[u8; 32],
        preimage: &[u8; 32],
    ) -> Result<TransactionReceipt, ChainError>;

    /// Refund an expired HTLC
    async fn refund_htlc(&self, htlc_id: &[u8; 32]) -> Result<TransactionReceipt, ChainError>;

    /// Get HTLC information
    async fn get_htlc(&self, htlc_id: &[u8; 32]) -> Result<Option<HtlcInfo>, ChainError>;
}

/// HTLC information
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HtlcInfo {
    /// HTLC ID
    pub id: [u8; 32],
    /// Sender address
    pub sender: Vec<u8>,
    /// Recipient address
    pub recipient: Vec<u8>,
    /// Hash lock
    pub hash_lock: [u8; 32],
    /// Timeout block number
    pub timeout_block: u64,
    /// Amount locked
    pub amount: u128,
    /// Current status
    pub status: HtlcStatus,
}

/// HTLC status
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HtlcStatus {
    /// HTLC is active and can be redeemed
    Active,
    /// HTLC was redeemed
    Redeemed,
    /// HTLC was refunded
    Refunded,
    /// HTLC expired
    Expired,
}
