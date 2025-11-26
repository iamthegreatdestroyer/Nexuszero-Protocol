//! Solana blockchain connector implementation.

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use chain_connectors_common::{
    ChainConnector, ChainError, ChainId, ChainOperation, EventFilter, EventStream,
    BlockInfo, FeeConfidence, FeeEstimate, ProofMetadata, TransactionReceipt,
    TransactionStatus, OnChainProof,
};

use crate::config::SolanaConfig;
use crate::error::SolanaError;

/// Solana JSON-RPC request wrapper
#[derive(Debug, Serialize)]
struct RpcRequest<T> {
    jsonrpc: &'static str,
    id: u64,
    method: &'static str,
    params: T,
}

impl<T> RpcRequest<T> {
    fn new(method: &'static str, params: T) -> Self {
        Self {
            jsonrpc: "2.0",
            id: 1,
            method,
            params,
        }
    }
}

/// Solana JSON-RPC response wrapper
#[derive(Debug, Deserialize)]
struct RpcResponse<T> {
    result: Option<T>,
    error: Option<RpcError>,
}

#[derive(Debug, Deserialize)]
struct RpcError {
    code: i64,
    message: String,
}

/// Get balance response
#[derive(Debug, Deserialize)]
struct BalanceResponse {
    value: u64,
}

/// Transaction status response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct SignatureStatus {
    slot: Option<u64>,
    confirmations: Option<u64>,
    err: Option<serde_json::Value>,
    confirmation_status: Option<String>,
}

/// Solana blockchain connector
pub struct SolanaConnector {
    config: SolanaConfig,
    client: Client,
    verifier_address: Option<Vec<u8>>,
    bridge_address: Option<Vec<u8>>,
    connected: Arc<RwLock<bool>>,
}

impl SolanaConnector {
    /// Create a new Solana connector
    pub fn new(config: SolanaConfig) -> Result<Self, SolanaError> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(config.timeout_seconds))
            .build()
            .map_err(|e| SolanaError::Connection(e.to_string()))?;

        let verifier_address = if config.verifier_program_id.is_empty() {
            None
        } else {
            bs58::decode(&config.verifier_program_id).into_vec().ok()
        };

        let bridge_address = config.bridge_program_id.as_ref().and_then(|s| {
            if s.is_empty() {
                None
            } else {
                bs58::decode(s).into_vec().ok()
            }
        });

        Ok(Self {
            config,
            client,
            verifier_address,
            bridge_address,
            connected: Arc::new(RwLock::new(false)),
        })
    }

    /// Get the RPC endpoint URL
    fn rpc_url(&self) -> &str {
        &self.config.rpc_url
    }

    /// Make an RPC call
    async fn rpc_call<P: Serialize, R: for<'de> Deserialize<'de>>(
        &self,
        method: &'static str,
        params: P,
    ) -> Result<R, ChainError> {
        let request = RpcRequest::new(method, params);

        let response = self.client
            .post(self.rpc_url())
            .json(&request)
            .send()
            .await
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        let rpc_response: RpcResponse<R> = response
            .json()
            .await
            .map_err(|e| ChainError::SerializationError(e.to_string()))?;

        if let Some(error) = rpc_response.error {
            return Err(ChainError::RpcError(format!("{}: {}", error.code, error.message)));
        }

        rpc_response.result.ok_or_else(|| {
            ChainError::RpcError("Empty response from RPC".to_string())
        })
    }

    /// Get current slot
    async fn get_slot(&self) -> Result<u64, ChainError> {
        self.rpc_call::<Vec<()>, u64>("getSlot", vec![]).await
    }

    /// Get recent blockhash
    async fn get_recent_blockhash(&self) -> Result<String, ChainError> {
        #[derive(Deserialize)]
        struct BlockhashValue {
            blockhash: String,
        }
        #[derive(Deserialize)]
        struct BlockhashResponse {
            value: BlockhashValue,
        }

        let response: BlockhashResponse = self.rpc_call(
            "getLatestBlockhash",
            json!([{"commitment": "finalized"}]),
        ).await?;

        Ok(response.value.blockhash)
    }

    /// Parse address from bytes
    fn address_from_bytes(bytes: &[u8]) -> String {
        bs58::encode(bytes).into_string()
    }

    /// Parse bytes from address string
    fn bytes_from_address(address: &str) -> Result<Vec<u8>, ChainError> {
        bs58::decode(address)
            .into_vec()
            .map_err(|e| ChainError::InvalidAddress(e.to_string()))
    }

    /// Connect and verify connectivity
    pub async fn connect(&mut self) -> Result<(), ChainError> {
        info!("Connecting to Solana {}...", self.config.cluster.as_str());

        let slot = self.get_slot().await?;
        info!("Connected to Solana at slot {}", slot);

        let mut connected = self.connected.write().await;
        *connected = true;

        Ok(())
    }
}

#[async_trait]
impl ChainConnector for SolanaConnector {
    fn chain_id(&self) -> ChainId {
        ChainId::Solana
    }

    fn chain_name(&self) -> &str {
        "Solana"
    }

    async fn is_healthy(&self) -> bool {
        match self.get_slot().await {
            Ok(slot) => {
                debug!("Solana health check OK, slot: {}", slot);
                true
            }
            Err(e) => {
                warn!("Solana health check failed: {}", e);
                false
            }
        }
    }

    async fn get_block_number(&self) -> Result<u64, ChainError> {
        self.get_slot().await
    }

    async fn get_block(&self, block_number: u64) -> Result<BlockInfo, ChainError> {
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct Block {
            blockhash: String,
            previous_blockhash: String,
            block_time: Option<u64>,
            transactions: Option<Vec<serde_json::Value>>,
        }

        let block: Block = self.rpc_call(
            "getBlock",
            json!([block_number, {"encoding": "json", "transactionDetails": "none"}]),
        ).await?;

        let hash = Self::bytes_from_address(&block.blockhash)?;
        let parent_hash = Self::bytes_from_address(&block.previous_blockhash)?;

        let mut hash_arr = [0u8; 32];
        let mut parent_arr = [0u8; 32];
        
        if hash.len() >= 32 {
            hash_arr.copy_from_slice(&hash[..32]);
        }
        if parent_hash.len() >= 32 {
            parent_arr.copy_from_slice(&parent_hash[..32]);
        }

        Ok(BlockInfo {
            number: block_number,
            hash: hash_arr,
            parent_hash: parent_arr,
            timestamp: block.block_time.unwrap_or(0),
            transaction_count: block.transactions.map(|t| t.len() as u32).unwrap_or(0),
        })
    }

    async fn submit_proof(
        &self,
        proof: &[u8],
        metadata: &ProofMetadata,
    ) -> Result<TransactionReceipt, ChainError> {
        let _verifier = self.verifier_address.as_ref()
            .ok_or_else(|| ChainError::ConfigError("Verifier program not configured".to_string()))?;

        // Build instruction data
        let mut instruction_data = vec![0u8]; // Instruction discriminator: SubmitProof = 0
        instruction_data.push(metadata.privacy_level);
        instruction_data.extend_from_slice(&metadata.sender_commitment);
        instruction_data.extend_from_slice(&metadata.recipient_commitment);
        instruction_data.extend_from_slice(&(proof.len() as u32).to_le_bytes());
        instruction_data.extend_from_slice(proof);

        // Get current slot and blockhash
        let current_slot = self.get_slot().await?;
        let _blockhash = self.get_recent_blockhash().await?;

        // Placeholder transaction hash (real implementation would sign and submit)
        let mut tx_hash = [0u8; 32];
        tx_hash[..8].copy_from_slice(&current_slot.to_le_bytes());

        Ok(TransactionReceipt {
            tx_hash,
            block_number: current_slot,
            block_hash: None,
            gas_used: 5000,
            status: true,
            logs: vec![],
            effective_gas_price: Some(5000),
            transaction_index: 0,
        })
    }

    async fn verify_proof(&self, proof_id: &[u8; 32]) -> Result<bool, ChainError> {
        let _verifier = self.verifier_address.as_ref()
            .ok_or_else(|| ChainError::ConfigError("Verifier program not configured".to_string()))?;

        // Query the on-chain proof account
        let account_address = Self::address_from_bytes(proof_id);

        #[derive(Deserialize)]
        struct AccountInfo {
            data: Option<Vec<String>>,
        }

        #[derive(Deserialize)]
        struct AccountValue {
            value: Option<AccountInfo>,
        }

        let result: AccountValue = self.rpc_call(
            "getAccountInfo",
            json!([account_address, {"encoding": "base64"}]),
        ).await?;

        // Check if account exists and has data
        if let Some(account) = result.value {
            if let Some(data) = account.data {
                return Ok(!data.is_empty());
            }
        }

        Ok(false)
    }

    async fn get_proof_details(
        &self,
        proof_id: &[u8; 32],
    ) -> Result<Option<OnChainProof>, ChainError> {
        let account_address = Self::address_from_bytes(proof_id);

        #[derive(Deserialize)]
        struct AccountInfo {
            data: Option<Vec<String>>,
        }

        #[derive(Deserialize)]
        struct AccountValue {
            value: Option<AccountInfo>,
        }

        let result: AccountValue = self.rpc_call(
            "getAccountInfo",
            json!([account_address, {"encoding": "base64"}]),
        ).await?;

        if let Some(account) = result.value {
            if let Some(data) = account.data {
                if data.is_empty() {
                    return Ok(None);
                }

                // Parse account data (simplified)
                use base64::Engine;
                let decoded = base64::engine::general_purpose::STANDARD
                    .decode(&data[0])
                    .unwrap_or_default();

                if decoded.len() < 50 {
                    return Ok(None);
                }

                // Parse proof account structure
                let privacy_level = decoded[0];
                let verified = decoded[1] != 0;

                return Ok(Some(OnChainProof {
                    id: *proof_id,
                    privacy_level,
                    proof_type: "nexuszero".to_string(),
                    timestamp: 0,
                    verified,
                    submitter: vec![],
                    block_number: 0,
                }));
            }
        }

        Ok(None)
    }

    async fn subscribe_events(&self, _filter: EventFilter) -> Result<EventStream, ChainError> {
        // Create a channel for events
        let (tx, rx) = mpsc::channel(100);

        // Note: Real implementation would use WebSocket subscription
        // For now, return empty channel
        drop(tx);

        Ok(rx)
    }

    async fn get_transaction_status(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<TransactionStatus, ChainError> {
        let signature = Self::address_from_bytes(tx_hash);

        #[derive(Deserialize)]
        struct StatusValue {
            value: Vec<Option<SignatureStatus>>,
        }

        let result: StatusValue = self.rpc_call(
            "getSignatureStatuses",
            json!([[signature]]),
        ).await?;

        if let Some(Some(status)) = result.value.first() {
            if status.err.is_some() {
                return Ok(TransactionStatus::Failed);
            }

            match status.confirmation_status.as_deref() {
                Some("finalized") | Some("confirmed") => Ok(TransactionStatus::Confirmed),
                Some("processed") => Ok(TransactionStatus::Pending),
                _ => Ok(TransactionStatus::Unknown),
            }
        } else {
            Ok(TransactionStatus::Unknown)
        }
    }

    async fn get_transaction_receipt(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<Option<TransactionReceipt>, ChainError> {
        let signature = Self::address_from_bytes(tx_hash);

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TxResponse {
            slot: u64,
            meta: Option<TxMeta>,
        }

        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase")]
        struct TxMeta {
            err: Option<serde_json::Value>,
            fee: u64,
        }

        let result: Option<TxResponse> = self.rpc_call(
            "getTransaction",
            json!([signature, {"encoding": "json"}]),
        ).await?;

        match result {
            Some(tx) => {
                let status = tx.meta.as_ref().map(|m| m.err.is_none()).unwrap_or(false);
                let gas_used = tx.meta.as_ref().map(|m| m.fee).unwrap_or(0);

                Ok(Some(TransactionReceipt {
                    tx_hash: *tx_hash,
                    block_number: tx.slot,
                    block_hash: None,
                    gas_used,
                    status,
                    logs: vec![],
                    effective_gas_price: Some(gas_used as u128),
                    transaction_index: 0,
                }))
            }
            None => Ok(None),
        }
    }

    async fn estimate_fee(&self, operation: ChainOperation) -> Result<FeeEstimate, ChainError> {
        let base_fee = 5000u64; // 5000 lamports base fee
        let priority_fee = self.config.priority_fee_micro_lamports.unwrap_or(0);

        let compute_units = match operation {
            ChainOperation::SubmitProof { proof_size, privacy_level } => {
                200_000 + (proof_size as u64 * 100) + (privacy_level as u64 * 10_000)
            }
            ChainOperation::VerifyProof { .. } => 50_000,
            ChainOperation::Transfer { .. } => 200,
            ChainOperation::BridgeInitiate { .. } => 400_000,
            ChainOperation::BridgeComplete { .. } => 300_000,
            ChainOperation::Deploy { bytecode_size } => bytecode_size as u64 * 10,
            ChainOperation::ContractCall { calldata_size } => 100_000 + calldata_size as u64 * 50,
        };

        let total_fee = base_fee + (compute_units * priority_fee / 1_000_000);

        Ok(FeeEstimate {
            gas_units: compute_units,
            gas_price: base_fee as f64 / 1e9, // Convert lamports to SOL
            priority_fee: Some(priority_fee as f64 / 1e6),
            total_fee_native: total_fee as f64 / 1e9,
            total_fee_usd: None,
            confidence: FeeConfidence::Medium,
        })
    }

    async fn get_balance(&self, address: &[u8]) -> Result<u128, ChainError> {
        let address_str = Self::address_from_bytes(address);

        let response: BalanceResponse = self.rpc_call(
            "getBalance",
            json!([address_str]),
        ).await?;

        Ok(response.value as u128)
    }

    fn verifier_address(&self) -> Option<&[u8]> {
        self.verifier_address.as_deref()
    }

    fn bridge_address(&self) -> Option<&[u8]> {
        self.bridge_address.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_address_conversion() {
        let bytes = [1u8; 32];
        let address = SolanaConnector::address_from_bytes(&bytes);
        assert!(!address.is_empty());

        let back = SolanaConnector::bytes_from_address(&address).unwrap();
        assert_eq!(back.len(), 32);
    }

    #[tokio::test]
    async fn test_connector_creation() {
        let config = SolanaConfig {
            cluster: crate::config::SolanaCluster::devnet(),
            rpc_url: "https://api.devnet.solana.com".to_string(),
            ws_url: None,
            timeout_seconds: 30,
            commitment: crate::config::CommitmentLevel::confirmed(),
            verifier_program_id: String::new(),
            bridge_program_id: None,
            priority_fee_micro_lamports: None,
        };

        let connector = SolanaConnector::new(config);
        assert!(connector.is_ok());
    }
}
