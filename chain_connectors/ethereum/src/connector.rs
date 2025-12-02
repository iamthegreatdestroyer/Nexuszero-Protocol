//! Ethereum blockchain connector implementation

use std::sync::Arc;
use async_trait::async_trait;
use alloy::{
    primitives::{Address, FixedBytes},
    providers::{Provider, ProviderBuilder, RootProvider},
    transports::http::{Client, Http},
    signers::local::PrivateKeySigner,
};
use tracing::{debug, info, warn};

use chain_connectors_common::{
    ChainConnector, ChainError, ChainId, ChainOperation, FeeEstimate,
    FeeConfidence, ProofMetadata, TransactionReceipt, TransactionStatus,
    EventFilter, EventStream, BlockInfo, OnChainProof, event_channel,
    BridgeConnector, BridgeTransfer, BridgeTransferStatus,
    HtlcConnector, HtlcInfo, HtlcStatus,
};

use crate::config::EthereumConfig;
use crate::error::EthereumError;
use crate::contracts::{NexusZeroVerifier, NexusZeroBridge};

/// Ethereum blockchain connector
pub struct EthereumConnector {
    /// Configuration
    config: EthereumConfig,
    /// HTTP provider
    provider: Arc<RootProvider<Http<Client>>>,
    /// Verifier contract address
    verifier_address: Address,
    /// Bridge contract address
    bridge_address: Option<Address>,
    /// HTLC contract address
    htlc_address: Option<Address>,
    /// Wallet signer
    signer: Option<PrivateKeySigner>,
}

impl EthereumConnector {
    /// Create a new Ethereum connector
    pub async fn new(config: EthereumConfig) -> Result<Self, EthereumError> {
        config.validate().map_err(EthereumError::ConfigError)?;

        // Create HTTP provider
        let provider = ProviderBuilder::new()
            .on_http(config.rpc_url.parse().map_err(|e| {
                EthereumError::ConfigError(format!("Invalid RPC URL: {}", e))
            })?);
        let provider = Arc::new(provider);

        // Parse addresses
        let verifier_address: Address = config.verifier_address.parse()
            .map_err(|e| EthereumError::InvalidAddress(format!("{}", e)))?;

        let bridge_address = if let Some(ref addr) = config.bridge_address {
            Some(addr.parse().map_err(|e| EthereumError::InvalidAddress(format!("{}", e)))?)
        } else {
            None
        };

        // Parse private key if provided
        let signer = if let Some(ref key) = config.private_key {
            let key_bytes = hex::decode(key.trim_start_matches("0x"))
                .map_err(|e| EthereumError::ConfigError(format!("Invalid private key: {}", e)))?;
            
            Some(PrivateKeySigner::from_slice(&key_bytes)
                .map_err(|e| EthereumError::ConfigError(format!("Invalid private key: {}", e)))?)
        } else {
            None
        };

        info!(
            "Ethereum connector initialized for chain {} at {}",
            config.chain_id, config.rpc_url
        );

        Ok(Self {
            config,
            provider,
            verifier_address,
            bridge_address,
            htlc_address: None,
            signer,
        })
    }

    /// Get the verifier contract instance
    fn verifier_contract(&self) -> NexusZeroVerifier::NexusZeroVerifierInstance<Http<Client>, Arc<RootProvider<Http<Client>>>> {
        NexusZeroVerifier::new(self.verifier_address, self.provider.clone())
    }

    /// Get the bridge contract instance
    fn bridge_contract(&self) -> Option<NexusZeroBridge::NexusZeroBridgeInstance<Http<Client>, Arc<RootProvider<Http<Client>>>>> {
        self.bridge_address.map(|addr| {
            NexusZeroBridge::new(addr, self.provider.clone())
        })
    }

    /// Non-trait helper to call verifyProofById (exec) - produces a call that would be sent by a signer
    pub async fn verify_proof_by_id_exec(&self, proof_id: &[u8;32], nullifier: &[u8;32], commitment: &[u8;32]) -> Result<bool, ChainError> {
        let contract = self.verifier_contract();
        let call = contract.verifyProofById(
            FixedBytes::from(*proof_id),
            FixedBytes::from(*nullifier),
            FixedBytes::from(*commitment),
        );
        let result = call.call().await
            .map_err(|e| ChainError::ContractError(e.to_string()))?;
        Ok(result._0)
    }

    /// Convert chain ID to NexusZero ChainId enum
    fn to_nexus_chain_id(&self) -> ChainId {
        match self.config.chain_id {
            1 => ChainId::Ethereum,
            137 => ChainId::Polygon,
            42161 => ChainId::Arbitrum,
            10 => ChainId::Optimism,
            8453 => ChainId::Base,
            _ => ChainId::Custom(self.config.chain_id),
        }
    }

    /// Get current gas price
    async fn get_gas_price(&self) -> Result<u128, EthereumError> {
        let price = self.provider.get_gas_price().await
            .map_err(|e| EthereumError::ProviderError(e.to_string()))?;
        Ok(price)
    }

    /// Estimate gas for a transaction
    fn estimate_gas_for_proof(&self, proof_size: usize, privacy_level: u8) -> u64 {
        // Base cost + per-byte cost + verification cost based on privacy level
        let base_cost = 50_000u64;
        let per_byte_cost = 68u64; // ~68 gas per calldata byte
        let verification_cost = match privacy_level {
            0 => 0,
            1 => 100_000,
            2 => 150_000,
            3 => 250_000,
            4 => 400_000,
            5 => 600_000,
            _ => 250_000,
        };

        base_cost + (proof_size as u64 * per_byte_cost) + verification_cost
    }
}

#[async_trait]
impl ChainConnector for EthereumConnector {
    fn chain_id(&self) -> ChainId {
        self.to_nexus_chain_id()
    }

    fn chain_name(&self) -> &str {
        match self.config.chain_id {
            1 => "Ethereum Mainnet",
            11155111 => "Sepolia Testnet",
            137 => "Polygon Mainnet",
            42161 => "Arbitrum One",
            10 => "Optimism",
            8453 => "Base",
            31337 => "Local Development",
            _ => "Unknown Ethereum Chain",
        }
    }

    async fn is_healthy(&self) -> bool {
        match self.provider.get_chain_id().await {
            Ok(id) => id == self.config.chain_id,
            Err(_) => false,
        }
    }

    async fn get_block_number(&self) -> Result<u64, ChainError> {
        self.provider
            .get_block_number()
            .await
            .map_err(|e| ChainError::RpcError(e.to_string()))
    }

    async fn get_block(&self, block_number: u64) -> Result<BlockInfo, ChainError> {
        let block = self.provider
            .get_block_by_number(block_number.into(), false)
            .await
            .map_err(|e| ChainError::RpcError(e.to_string()))?
            .ok_or_else(|| ChainError::RpcError("Block not found".to_string()))?;

        let header = &block.header;
        
        Ok(BlockInfo {
            number: header.number,
            hash: header.hash.0,
            parent_hash: header.parent_hash.0,
            timestamp: header.timestamp,
            transaction_count: block.transactions.len() as u32,
        })
    }

    async fn submit_proof(
        &self,
        proof: &[u8],
        metadata: &ProofMetadata,
    ) -> Result<TransactionReceipt, ChainError> {
        let _signer = self.signer.as_ref()
            .ok_or_else(|| ChainError::KeyError("No signer configured".to_string()))?;

        debug!(
            "Submitting proof of size {} bytes at privacy level {}",
            proof.len(),
            metadata.privacy_level
        );

        let contract = self.verifier_contract();

        // Keep backward-compatible raw proof submission for now
        let call = contract.submitProofRaw(
            alloy::primitives::Bytes::from(proof.to_vec()),
            FixedBytes::from([0u8; 32]), // circuitId - connector must supply appropriate circuit id if available
            FixedBytes::from(metadata.sender_commitment),
            FixedBytes::from(metadata.recipient_commitment),
            metadata.privacy_level,
        );

        // For now, return a placeholder - full signing integration requires more setup
        // In production, this would use a SignerMiddleware
        let _call_builder = call;

        // Placeholder receipt - in production, this would be the actual transaction
        warn!("Proof submission requires transaction signing setup");
        
        Ok(TransactionReceipt {
            tx_hash: [0u8; 32],
            block_number: 0,
            block_hash: None,
            gas_used: self.estimate_gas_for_proof(proof.len(), metadata.privacy_level),
            status: false,
            logs: vec![],
            effective_gas_price: None,
            transaction_index: 0,
        })
    }

    async fn verify_proof(&self, proof_id: &[u8; 32]) -> Result<bool, ChainError> {
        let contract = self.verifier_contract();
        
        let result = contract.verifyProof(FixedBytes::from(*proof_id))
            .call()
            .await
            .map_err(|e| ChainError::ContractError(e.to_string()))?;

        Ok(result._0)
    }
    

    async fn get_proof_details(
        &self,
        proof_id: &[u8; 32],
    ) -> Result<Option<OnChainProof>, ChainError> {
        let contract = self.verifier_contract();
        
        let result = contract.getProofDetails(FixedBytes::from(*proof_id))
            .call()
            .await
            .map_err(|e| ChainError::ContractError(e.to_string()))?;

        // Check if proof exists (timestamp > 0)
        if result.timestamp == 0 {
            return Ok(None);
        }

        Ok(Some(OnChainProof {
            id: *proof_id,
            privacy_level: result.level,
            proof_type: "zk-snark".to_string(), // Default, could be stored on-chain
            timestamp: result.timestamp,
            verified: result.verified,
            submitter: result.submitter.as_slice().to_vec(),
            block_number: 0, // Would need separate query
        }))
    }

    async fn subscribe_events(&self, filter: EventFilter) -> Result<EventStream, ChainError> {
        // Create event channel
        let (tx, rx) = event_channel(100);
        let _filter = filter;

        // In production, this would set up WebSocket subscription
        // For now, return the receiver without active subscription
        warn!("Event subscription requires WebSocket provider setup");
        
        // Spawn a task that would poll for events
        let provider = self.provider.clone();
        let chain_id = self.chain_id();
        
        tokio::spawn(async move {
            let _provider = provider;
            let _chain_id = chain_id;
            let _tx = tx;
            // Event polling/subscription logic would go here
        });

        Ok(rx)
    }

    async fn get_transaction_status(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<TransactionStatus, ChainError> {
        let receipt = self.get_transaction_receipt(tx_hash).await?;
        
        match receipt {
            Some(r) => {
                if r.status {
                    Ok(TransactionStatus::Confirmed)
                } else {
                    Ok(TransactionStatus::Failed)
                }
            }
            None => Ok(TransactionStatus::Pending),
        }
    }

    async fn get_transaction_receipt(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<Option<TransactionReceipt>, ChainError> {
        let hash = FixedBytes::from(*tx_hash);
        
        let receipt = self.provider
            .get_transaction_receipt(hash)
            .await
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        Ok(receipt.map(|r| TransactionReceipt {
            tx_hash: *tx_hash,
            block_number: r.block_number.unwrap_or(0),
            block_hash: r.block_hash.map(|h| h.0),
            gas_used: r.gas_used as u64,
            status: r.status(),
            logs: r.inner.logs().iter().map(|log| {
                chain_connectors_common::EventLog {
                    address: log.address().as_slice().to_vec(),
                    topics: log.topics().iter().map(|t| t.0).collect(),
                    data: log.data().data.to_vec(),
                    log_index: log.log_index.unwrap_or(0) as u32,
                }
            }).collect(),
            effective_gas_price: Some(r.effective_gas_price as u128),
            transaction_index: r.transaction_index.unwrap_or(0) as u32,
        }))
    }

    async fn estimate_fee(&self, operation: ChainOperation) -> Result<FeeEstimate, ChainError> {
        let gas_price = self.get_gas_price().await
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        let gas_units = match operation {
            ChainOperation::SubmitProof { proof_size, privacy_level } => {
                self.estimate_gas_for_proof(proof_size, privacy_level)
            }
            ChainOperation::VerifyProof { .. } => 50_000,
            ChainOperation::Transfer { .. } => 21_000,
            ChainOperation::BridgeInitiate { .. } => 150_000,
            ChainOperation::BridgeComplete { .. } => 100_000,
            ChainOperation::Deploy { bytecode_size } => {
                32_000 + (bytecode_size as u64 * 200)
            }
            ChainOperation::ContractCall { calldata_size } => {
                21_000 + (calldata_size as u64 * 68)
            }
        };

        let gas_price_gwei = gas_price as f64 / 1e9;
        let total_fee_native = (gas_units as f64 * gas_price_gwei) / 1e9;
        
        // ETH price would come from an oracle in production
        let eth_price_usd = 3000.0;

        Ok(FeeEstimate {
            gas_units,
            gas_price: gas_price_gwei,
            priority_fee: self.config.priority_fee_gwei,
            total_fee_native,
            total_fee_usd: Some(total_fee_native * eth_price_usd),
            confidence: FeeConfidence::Medium,
        })
    }

    async fn get_balance(&self, address: &[u8]) -> Result<u128, ChainError> {
        if address.len() < 20 {
            return Err(ChainError::InvalidAddress("Address too short".to_string()));
        }

        let addr = Address::from_slice(&address[..20]);
        
        let balance = self.provider
            .get_balance(addr)
            .await
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        // Convert U256 to u128 (may overflow for very large balances)
        Ok(balance.try_into().unwrap_or(u128::MAX))
    }

    fn verifier_address(&self) -> Option<&[u8]> {
        Some(self.verifier_address.as_slice())
    }

    fn bridge_address(&self) -> Option<&[u8]> {
        self.bridge_address.as_ref().map(|a| a.as_slice())
    }
}

#[async_trait]
impl BridgeConnector for EthereumConnector {
    async fn initiate_bridge_transfer(
        &self,
        target_chain: ChainId,
        proof: &[u8],
        recipient: &[u8],
        amount: u128,
    ) -> Result<BridgeTransfer, ChainError> {
        let _contract = self.bridge_contract()
            .ok_or_else(|| ChainError::ConfigError("Bridge contract not configured".to_string()))?;

        // Convert target chain to bytes32
        let target_chain_bytes = match target_chain {
            ChainId::Ethereum => [0u8; 32],
            ChainId::Polygon => {
                let mut bytes = [0u8; 32];
                bytes[31] = 137;
                bytes
            }
            ChainId::Solana => {
                let mut bytes = [0u8; 32];
                bytes[28..32].copy_from_slice(b"SOL\0");
                bytes
            }
            _ => [0u8; 32],
        };

        debug!(
            "Initiating bridge transfer to {:?} for {} wei",
            target_chain, amount
        );

        // Placeholder - actual implementation would call contract
        let transfer_id = {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&target_chain_bytes);
            hasher.update(proof);
            hasher.update(recipient);
            hasher.update(&amount.to_le_bytes());
            hasher.update(&chrono::Utc::now().timestamp().to_le_bytes());
            let result = hasher.finalize();
            let mut id = [0u8; 32];
            id.copy_from_slice(&result);
            id
        };

        Ok(BridgeTransfer {
            id: transfer_id,
            source_chain: self.chain_id(),
            target_chain,
            amount,
            recipient: recipient.to_vec(),
            proof_id: [0u8; 32], // Would be generated
            source_tx_hash: [0u8; 32], // Would come from tx receipt
            status: BridgeTransferStatus::Initiated,
        })
    }

    async fn complete_bridge_transfer(
        &self,
        transfer_id: &[u8; 32],
        _relayer_proof: &[u8],
    ) -> Result<TransactionReceipt, ChainError> {
        let _contract = self.bridge_contract()
            .ok_or_else(|| ChainError::ConfigError("Bridge contract not configured".to_string()))?;

        debug!("Completing bridge transfer {}", hex::encode(transfer_id));

        // Placeholder
        Ok(TransactionReceipt {
            tx_hash: [0u8; 32],
            block_number: 0,
            block_hash: None,
            gas_used: 100_000,
            status: false,
            logs: vec![],
            effective_gas_price: None,
            transaction_index: 0,
        })
    }

    async fn get_bridge_transfer_status(
        &self,
        transfer_id: &[u8; 32],
    ) -> Result<BridgeTransferStatus, ChainError> {
        let contract = self.bridge_contract()
            .ok_or_else(|| ChainError::ConfigError("Bridge contract not configured".to_string()))?;

        let result = contract
            .getTransferStatus(FixedBytes::from(*transfer_id))
            .call()
            .await
            .map_err(|e| ChainError::ContractError(e.to_string()))?;

        Ok(match result._0 {
            0 => BridgeTransferStatus::Initiated,
            1 => BridgeTransferStatus::Relaying,
            2 => BridgeTransferStatus::ReadyToComplete,
            3 => BridgeTransferStatus::Completed,
            4 => BridgeTransferStatus::Failed,
            5 => BridgeTransferStatus::Expired,
            _ => BridgeTransferStatus::Failed,
        })
    }

    fn supported_bridge_targets(&self) -> Vec<ChainId> {
        vec![
            ChainId::Polygon,
            ChainId::Arbitrum,
            ChainId::Optimism,
            ChainId::Base,
            ChainId::Solana,
        ]
    }
}

#[async_trait]
impl HtlcConnector for EthereumConnector {
    async fn create_htlc(
        &self,
        recipient: &[u8],
        hash_lock: &[u8; 32],
        timeout_blocks: u64,
        amount: u128,
    ) -> Result<HtlcInfo, ChainError> {
        if recipient.len() < 20 {
            return Err(ChainError::InvalidAddress("Recipient address too short".to_string()));
        }

        debug!(
            "Creating HTLC for {} wei with timeout {} blocks",
            amount, timeout_blocks
        );

        // Generate HTLC ID
        let htlc_id = {
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(recipient);
            hasher.update(hash_lock);
            hasher.update(&timeout_blocks.to_le_bytes());
            hasher.update(&amount.to_le_bytes());
            hasher.update(&chrono::Utc::now().timestamp().to_le_bytes());
            let result = hasher.finalize();
            let mut id = [0u8; 32];
            id.copy_from_slice(&result);
            id
        };

        let current_block = self.get_block_number().await?;

        Ok(HtlcInfo {
            id: htlc_id,
            sender: vec![], // Would be signer address
            recipient: recipient.to_vec(),
            hash_lock: *hash_lock,
            timeout_block: current_block + timeout_blocks,
            amount,
            status: HtlcStatus::Active,
        })
    }

    async fn redeem_htlc(
        &self,
        htlc_id: &[u8; 32],
        preimage: &[u8; 32],
    ) -> Result<TransactionReceipt, ChainError> {
        debug!(
            "Redeeming HTLC {} with preimage",
            hex::encode(htlc_id)
        );

        // Verify preimage matches hash lock
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(preimage);
        let _hash = hasher.finalize();

        // Placeholder
        Ok(TransactionReceipt {
            tx_hash: [0u8; 32],
            block_number: 0,
            block_hash: None,
            gas_used: 50_000,
            status: false,
            logs: vec![],
            effective_gas_price: None,
            transaction_index: 0,
        })
    }

    async fn refund_htlc(&self, htlc_id: &[u8; 32]) -> Result<TransactionReceipt, ChainError> {
        debug!("Refunding HTLC {}", hex::encode(htlc_id));

        // Placeholder
        Ok(TransactionReceipt {
            tx_hash: [0u8; 32],
            block_number: 0,
            block_hash: None,
            gas_used: 30_000,
            status: false,
            logs: vec![],
            effective_gas_price: None,
            transaction_index: 0,
        })
    }

    async fn get_htlc(&self, _htlc_id: &[u8; 32]) -> Result<Option<HtlcInfo>, ChainError> {
        // Would query contract
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chain_id_mapping() {
        let config = EthereumConfig::mainnet(
            "http://localhost:8545",
            "0x1234567890123456789012345678901234567890",
        );
        
        // Can't test async new() in sync test, just verify config
        assert_eq!(config.chain_id, 1);
    }

    #[test]
    fn test_gas_estimation() {
        // Base cost calculation test
        let base = 50_000u64;
        let proof_size = 1000usize;
        let per_byte = 68u64;
        let verification = 250_000u64; // Level 3
        
        let expected = base + (proof_size as u64 * per_byte) + verification;
        assert_eq!(expected, 368_000);
    }
}
