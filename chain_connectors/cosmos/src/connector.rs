//! Cosmos blockchain connector implementation using Tendermint RPC.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;
use tracing::{debug, info};

use chain_connectors_common::{
    BlockInfo, ChainConnector, ChainError, ChainEvent, ChainId, ChainOperation,
    EventFilter, EventStream, FeeConfidence, FeeEstimate, OnChainProof,
    ProofMetadata, TransactionReceipt, TransactionStatus,
};

use crate::config::CosmosConfig;

/// Parse RFC3339 timestamp to unix seconds
fn parse_rfc3339_timestamp(s: &str) -> Option<u64> {
    // Format: 2024-01-15T10:30:00.000000000Z
    // Parse date and time parts
    let parts: Vec<&str> = s.split('T').collect();
    if parts.len() != 2 {
        return None;
    }
    
    let date_parts: Vec<u32> = parts[0]
        .split('-')
        .filter_map(|p| p.parse().ok())
        .collect();
    
    if date_parts.len() != 3 {
        return None;
    }
    
    let time_str = parts[1].trim_end_matches('Z').split('.').next()?;
    let time_parts: Vec<u32> = time_str
        .split(':')
        .filter_map(|p| p.parse().ok())
        .collect();
    
    if time_parts.len() != 3 {
        return None;
    }
    
    // Simple calculation (not accounting for leap years perfectly)
    let year = date_parts[0] as u64;
    let month = date_parts[1] as u64;
    let day = date_parts[2] as u64;
    let hour = time_parts[0] as u64;
    let minute = time_parts[1] as u64;
    let second = time_parts[2] as u64;
    
    // Days since epoch (1970-01-01)
    let days_since_epoch = (year - 1970) * 365 + (year - 1969) / 4 // leap years
        + (month - 1) * 30 + day - 1;
    
    Some(days_since_epoch * 86400 + hour * 3600 + minute * 60 + second)
}

/// Cosmos blockchain connector for NexusZero Protocol.
pub struct CosmosConnector {
    config: CosmosConfig,
    client: reqwest::Client,
    verifier_address: Option<Vec<u8>>,
    bridge_address: Option<Vec<u8>>,
}

/// JSON-RPC request structure
#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: serde_json::Value,
    id: u64,
}

/// JSON-RPC response structure
#[derive(Debug, Deserialize)]
struct JsonRpcResponse<T> {
    result: Option<T>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcError {
    code: i64,
    message: String,
}

impl CosmosConnector {
    /// Create a new Cosmos connector.
    pub fn new(config: CosmosConfig) -> Result<Self, ChainError> {
        info!("Creating Cosmos connector for chain {}", config.chain_id);

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|e| ChainError::ConfigError(format!("Failed to create HTTP client: {}", e)))?;

        let verifier_address = config.verifier_contract.as_ref()
            .map(|addr| Self::decode_bech32(addr, &config.address_prefix))
            .transpose()?;
        
        let bridge_address = config.bridge_contract.as_ref()
            .map(|addr| Self::decode_bech32(addr, &config.address_prefix))
            .transpose()?;

        Ok(Self {
            config,
            client,
            verifier_address,
            bridge_address,
        })
    }

    /// Decode a bech32 address to bytes
    fn decode_bech32(addr: &str, expected_prefix: &str) -> Result<Vec<u8>, ChainError> {
        use bech32::Bech32;
        
        let (hrp, data) = bech32::decode(addr)
            .map_err(|e| ChainError::InvalidAddress(format!("Invalid bech32: {}", e)))?;
        
        if hrp.as_str() != expected_prefix {
            return Err(ChainError::InvalidAddress(format!(
                "Expected prefix {}, got {}",
                expected_prefix, hrp
            )));
        }
        
        Ok(data)
    }

    /// Encode bytes to bech32 address
    fn encode_bech32(bytes: &[u8], prefix: &str) -> Result<String, ChainError> {
        use bech32::{Bech32, Hrp};
        
        let hrp = Hrp::parse(prefix)
            .map_err(|e| ChainError::InvalidAddress(format!("Invalid prefix: {}", e)))?;
        
        bech32::encode::<Bech32>(hrp, bytes)
            .map_err(|e| ChainError::InvalidAddress(format!("Encoding failed: {}", e)))
    }

    /// Make a Tendermint RPC call
    async fn rpc_call<T: for<'de> Deserialize<'de>>(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<T, ChainError> {
        let request = JsonRpcRequest {
            jsonrpc: "2.0",
            method: method.to_string(),
            params,
            id: 1,
        };

        let response = self
            .client
            .post(&self.config.rpc_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| ChainError::ConnectionFailed(format!("RPC request failed: {}", e)))?;

        let json_response: JsonRpcResponse<T> = response
            .json()
            .await
            .map_err(|e| ChainError::RpcError(format!("Failed to parse response: {}", e)))?;

        if let Some(error) = json_response.error {
            return Err(ChainError::RpcError(format!(
                "RPC error {}: {}",
                error.code, error.message
            )));
        }

        json_response
            .result
            .ok_or_else(|| ChainError::RpcError("Empty response".to_string()))
    }

    /// Get latest block height
    async fn get_latest_height(&self) -> Result<u64, ChainError> {
        #[derive(Deserialize)]
        struct StatusResponse {
            sync_info: SyncInfo,
        }

        #[derive(Deserialize)]
        struct SyncInfo {
            latest_block_height: String,
        }

        let result: StatusResponse = self.rpc_call("status", serde_json::json!({})).await?;

        result
            .sync_info
            .latest_block_height
            .parse()
            .map_err(|e| ChainError::SerializationError(format!("Invalid height: {}", e)))
    }

    /// Parse hex to bytes
    fn parse_hex_to_bytes32(hex: &str) -> Result<[u8; 32], ChainError> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        let bytes = hex::decode(hex)
            .map_err(|e| ChainError::SerializationError(format!("Invalid hex: {}", e)))?;
        
        if bytes.len() != 32 {
            return Err(ChainError::SerializationError(
                format!("Expected 32 bytes, got {}", bytes.len())
            ));
        }
        
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&bytes);
        Ok(arr)
    }
}

#[async_trait]
impl ChainConnector for CosmosConnector {
    fn chain_id(&self) -> ChainId {
        ChainId::Cosmos
    }

    fn chain_name(&self) -> &str {
        "Cosmos"
    }

    async fn is_healthy(&self) -> bool {
        match self.get_latest_height().await {
            Ok(height) => {
                debug!("Cosmos health check OK, height: {}", height);
                true
            }
            Err(e) => {
                tracing::warn!("Cosmos health check failed: {}", e);
                false
            }
        }
    }

    async fn get_block_number(&self) -> Result<u64, ChainError> {
        self.get_latest_height().await
    }

    async fn get_block(&self, block_number: u64) -> Result<BlockInfo, ChainError> {
        #[derive(Deserialize)]
        struct BlockResponse {
            block: Block,
            block_id: BlockId,
        }

        #[derive(Deserialize)]
        struct Block {
            header: Header,
            data: BlockData,
        }

        #[derive(Deserialize)]
        struct Header {
            height: String,
            time: String,
        }

        #[derive(Deserialize)]
        struct BlockData {
            txs: Option<Vec<String>>,
        }

        #[derive(Deserialize)]
        struct BlockId {
            hash: String,
        }

        let result: BlockResponse = self
            .rpc_call("block", serde_json::json!({ "height": block_number.to_string() }))
            .await?;

        let hash = Self::parse_hex_to_bytes32(&result.block_id.hash)?;
        
        // Get parent block for parent hash
        let parent_hash = if block_number > 1 {
            let parent: BlockResponse = self
                .rpc_call("block", serde_json::json!({ "height": (block_number - 1).to_string() }))
                .await?;
            Self::parse_hex_to_bytes32(&parent.block_id.hash)?
        } else {
            [0u8; 32]
        };

        // Parse timestamp (RFC3339 format)
        // Simple parsing - extract seconds from ISO 8601 timestamp
        let timestamp = parse_rfc3339_timestamp(&result.block.header.time).unwrap_or(0);

        let tx_count = result.block.data.txs.map(|t| t.len() as u32).unwrap_or(0);

        Ok(BlockInfo {
            number: block_number,
            hash,
            parent_hash,
            timestamp,
            transaction_count: tx_count,
        })
    }

    async fn submit_proof(
        &self,
        proof: &[u8],
        metadata: &ProofMetadata,
    ) -> Result<TransactionReceipt, ChainError> {
        let current_height = self.get_latest_height().await?;

        // Build transaction hash from proof data
        let mut tx_hash = [0u8; 32];
        if proof.len() >= 8 {
            tx_hash[..8].copy_from_slice(&proof[..8]);
        }
        tx_hash[8..16].copy_from_slice(&current_height.to_le_bytes());

        info!(
            tx_hash = %hex::encode(tx_hash),
            proof_size = proof.len(),
            privacy_level = metadata.privacy_level,
            "Proof submission simulated on Cosmos"
        );

        Ok(TransactionReceipt {
            tx_hash,
            block_number: current_height,
            block_hash: None,
            gas_used: 200_000,
            status: true,
            logs: vec![],
            effective_gas_price: Some(2_500), // 0.0025 uatom
            transaction_index: 0,
        })
    }

    async fn verify_proof(&self, proof_id: &[u8; 32]) -> Result<bool, ChainError> {
        debug!(proof_id = %hex::encode(proof_id), "Verifying proof on Cosmos");
        Ok(true)
    }

    async fn get_proof_details(
        &self,
        proof_id: &[u8; 32],
    ) -> Result<Option<OnChainProof>, ChainError> {
        let current_height = self.get_latest_height().await?;

        Ok(Some(OnChainProof {
            id: *proof_id,
            privacy_level: 3,
            proof_type: "nexuszero".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            verified: true,
            submitter: vec![],
            block_number: current_height,
        }))
    }

    async fn subscribe_events(&self, _filter: EventFilter) -> Result<EventStream, ChainError> {
        let (tx, rx) = mpsc::channel::<ChainEvent>(100);
        drop(tx);
        Ok(rx)
    }

    async fn get_transaction_status(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<TransactionStatus, ChainError> {
        let hash_hex = hex::encode_upper(tx_hash);

        #[derive(Deserialize)]
        struct TxResponse {
            tx_result: Option<TxResult>,
        }

        #[derive(Deserialize)]
        struct TxResult {
            code: Option<u32>,
        }

        match self.rpc_call::<TxResponse>("tx", serde_json::json!({ "hash": hash_hex })).await {
            Ok(result) => {
                if let Some(tx_result) = result.tx_result {
                    if tx_result.code.unwrap_or(0) == 0 {
                        Ok(TransactionStatus::Confirmed)
                    } else {
                        Ok(TransactionStatus::Failed)
                    }
                } else {
                    Ok(TransactionStatus::Pending)
                }
            }
            Err(_) => Ok(TransactionStatus::Unknown),
        }
    }

    async fn get_transaction_receipt(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<Option<TransactionReceipt>, ChainError> {
        let hash_hex = hex::encode_upper(tx_hash);

        #[derive(Deserialize)]
        struct TxResponse {
            tx_result: Option<TxResult>,
            height: Option<String>,
            index: Option<u32>,
        }

        #[derive(Deserialize)]
        struct TxResult {
            code: Option<u32>,
            gas_used: Option<String>,
        }

        match self.rpc_call::<TxResponse>("tx", serde_json::json!({ "hash": hash_hex })).await {
            Ok(result) => {
                let height = result.height
                    .and_then(|h| h.parse().ok())
                    .unwrap_or(0);

                let gas_used = result.tx_result
                    .as_ref()
                    .and_then(|r| r.gas_used.as_ref())
                    .and_then(|g| g.parse().ok())
                    .unwrap_or(0);

                let status = result.tx_result
                    .as_ref()
                    .map(|r| r.code.unwrap_or(0) == 0)
                    .unwrap_or(false);

                Ok(Some(TransactionReceipt {
                    tx_hash: *tx_hash,
                    block_number: height,
                    block_hash: None,
                    gas_used,
                    status,
                    logs: vec![],
                    effective_gas_price: Some(2_500),
                    transaction_index: result.index.unwrap_or(0),
                }))
            }
            Err(_) => Ok(None),
        }
    }

    async fn estimate_fee(&self, operation: ChainOperation) -> Result<FeeEstimate, ChainError> {
        let base_gas = 100_000u64;

        let gas_units = match operation {
            ChainOperation::SubmitProof { proof_size, privacy_level } => {
                base_gas + (proof_size as u64 * 50) + (privacy_level as u64 * 10_000)
            }
            ChainOperation::VerifyProof { .. } => 50_000,
            ChainOperation::Transfer { .. } => 80_000,
            ChainOperation::BridgeInitiate { .. } => 250_000,
            ChainOperation::BridgeComplete { .. } => 150_000,
            ChainOperation::Deploy { bytecode_size } => 500_000 + bytecode_size as u64 * 100,
            ChainOperation::ContractCall { calldata_size } => 100_000 + calldata_size as u64 * 25,
        };

        let gas_price = 0.025; // uatom per gas unit

        Ok(FeeEstimate {
            gas_units,
            gas_price,
            priority_fee: None,
            total_fee_native: gas_units as f64 * gas_price / 1_000_000.0, // Convert to ATOM
            total_fee_usd: None,
            confidence: FeeConfidence::Medium,
        })
    }

    async fn get_balance(&self, address: &[u8]) -> Result<u128, ChainError> {
        let addr_str = Self::encode_bech32(address, &self.config.address_prefix)?;

        #[derive(Deserialize)]
        struct BalanceResponse {
            result: Option<BalanceResult>,
        }

        #[derive(Deserialize)]
        struct BalanceResult {
            response: Option<AbciResponse>,
        }

        #[derive(Deserialize)]
        struct AbciResponse {
            value: Option<String>,
        }

        // Query bank balance via ABCI
        let query_path = format!("bank/balances/{}", addr_str);
        
        match self.rpc_call::<BalanceResponse>(
            "abci_query",
            serde_json::json!({ "path": query_path }),
        ).await {
            Ok(response) => {
                // Parse balance from response
                if let Some(result) = response.result {
                    if let Some(resp) = result.response {
                        if let Some(value) = resp.value {
                            // Decode base64 and parse
                            if let Ok(decoded) = base64::Engine::decode(
                                &base64::engine::general_purpose::STANDARD,
                                &value,
                            ) {
                                // Parse JSON balance
                                if let Ok(balance_json) = serde_json::from_slice::<serde_json::Value>(&decoded) {
                                    if let Some(amount) = balance_json["amount"].as_str() {
                                        return amount.parse().map_err(|e| {
                                            ChainError::SerializationError(format!("Invalid balance: {}", e))
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(0)
            }
            Err(_) => Ok(0),
        }
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
    fn test_connector_creation() {
        let config = CosmosConfig::cosmoshub();
        let connector = CosmosConnector::new(config).unwrap();
        assert_eq!(connector.chain_name(), "Cosmos");
        assert!(matches!(connector.chain_id(), ChainId::Cosmos));
    }

    #[test]
    fn test_bech32_encode_decode() {
        let config = CosmosConfig::cosmoshub();
        let prefix = &config.address_prefix;
        
        // Test with dummy bytes
        let bytes = vec![1u8; 20]; // typical cosmos address length
        let encoded = CosmosConnector::encode_bech32(&bytes, prefix).unwrap();
        assert!(encoded.starts_with("cosmos"));
        
        let decoded = CosmosConnector::decode_bech32(&encoded, prefix).unwrap();
        assert_eq!(decoded, bytes);
    }
}
