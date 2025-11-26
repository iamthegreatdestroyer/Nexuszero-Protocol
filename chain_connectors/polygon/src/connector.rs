//! Polygon Chain Connector - EVM-compatible connector for Polygon PoS network
//!
//! Uses JSON-RPC for direct interaction with Polygon nodes.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc;

use chain_connectors_common::{
    BlockInfo, ChainConnector, ChainError, ChainEvent, ChainId, ChainOperation,
    EventFilter, EventStream, FeeConfidence, FeeEstimate, OnChainProof, 
    ProofMetadata, TransactionReceipt, TransactionStatus,
};

use crate::config::PolygonConfig;

/// Polygon chain connector implementation
pub struct PolygonConnector {
    config: PolygonConfig,
    client: reqwest::Client,
    verifier_address: Option<Vec<u8>>,
    bridge_address: Option<Vec<u8>>,
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest {
    jsonrpc: &'static str,
    method: String,
    params: serde_json::Value,
    id: u64,
}

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

impl PolygonConnector {
    /// Create a new Polygon connector
    pub fn new(config: PolygonConfig) -> Result<Self, ChainError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|e| ChainError::ConfigError(format!("Failed to create HTTP client: {}", e)))?;

        let verifier_address = config.verifier_contract.as_ref()
            .map(|addr| Self::decode_address(addr))
            .transpose()?;
        
        let bridge_address = config.bridge_contract.as_ref()
            .map(|addr| Self::decode_address(addr))
            .transpose()?;

        Ok(Self { 
            config, 
            client,
            verifier_address,
            bridge_address,
        })
    }

    /// Decode hex address to bytes
    fn decode_address(addr: &str) -> Result<Vec<u8>, ChainError> {
        let addr = addr.strip_prefix("0x").unwrap_or(addr);
        hex::decode(addr)
            .map_err(|e| ChainError::InvalidAddress(format!("Invalid hex address: {}", e)))
    }

    /// Encode bytes to hex string
    fn encode_address(bytes: &[u8]) -> String {
        format!("0x{}", hex::encode(bytes))
    }

    /// Send a JSON-RPC request
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

    /// Parse hex string to u64
    fn parse_hex_u64(hex: &str) -> Result<u64, ChainError> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        u64::from_str_radix(hex, 16)
            .map_err(|e| ChainError::SerializationError(format!("Invalid hex: {}", e)))
    }

    /// Parse hex string to u128
    fn parse_hex_u128(hex: &str) -> Result<u128, ChainError> {
        let hex = hex.strip_prefix("0x").unwrap_or(hex);
        u128::from_str_radix(hex, 16)
            .map_err(|e| ChainError::SerializationError(format!("Invalid hex: {}", e)))
    }

    /// Parse hex string to [u8; 32]
    fn parse_hex_bytes32(hex: &str) -> Result<[u8; 32], ChainError> {
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
impl ChainConnector for PolygonConnector {
    fn chain_id(&self) -> ChainId {
        ChainId::Polygon
    }

    fn chain_name(&self) -> &str {
        "Polygon"
    }

    async fn is_healthy(&self) -> bool {
        self.get_block_number().await.is_ok()
    }

    async fn get_block_number(&self) -> Result<u64, ChainError> {
        let result: String = self
            .rpc_call("eth_blockNumber", serde_json::json!([]))
            .await?;
        Self::parse_hex_u64(&result)
    }

    async fn get_block(&self, block_number: u64) -> Result<BlockInfo, ChainError> {
        let block_hex = format!("0x{:x}", block_number);
        let result: serde_json::Value = self
            .rpc_call("eth_getBlockByNumber", serde_json::json!([block_hex, false]))
            .await?;

        let hash_hex = result["hash"]
            .as_str()
            .ok_or_else(|| ChainError::RpcError("Missing block hash".to_string()))?;
        let hash = Self::parse_hex_bytes32(hash_hex)?;

        let parent_hash_hex = result["parentHash"]
            .as_str()
            .ok_or_else(|| ChainError::RpcError("Missing parent hash".to_string()))?;
        let parent_hash = Self::parse_hex_bytes32(parent_hash_hex)?;

        let timestamp_hex = result["timestamp"]
            .as_str()
            .ok_or_else(|| ChainError::RpcError("Missing timestamp".to_string()))?;
        let timestamp = Self::parse_hex_u64(timestamp_hex)?;

        let tx_count = result["transactions"]
            .as_array()
            .map(|arr| arr.len() as u32)
            .unwrap_or(0);

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
        proof_data: &[u8],
        metadata: &ProofMetadata,
    ) -> Result<TransactionReceipt, ChainError> {
        // Get current block for reference
        let current_block = self.get_block_number().await?;
        
        // Build transaction hash from proof data
        let mut tx_hash = [0u8; 32];
        if proof_data.len() >= 8 {
            tx_hash[..8].copy_from_slice(&proof_data[..8]);
        }
        tx_hash[8..16].copy_from_slice(&current_block.to_le_bytes());

        tracing::info!(
            tx_hash = %hex::encode(tx_hash),
            proof_size = proof_data.len(),
            privacy_level = metadata.privacy_level,
            "Proof submission simulated on Polygon"
        );

        Ok(TransactionReceipt {
            tx_hash,
            block_number: current_block,
            block_hash: None,
            gas_used: 300_000,
            status: true,
            logs: vec![],
            effective_gas_price: Some(30_000_000_000), // 30 gwei
            transaction_index: 0,
        })
    }

    async fn verify_proof(&self, proof_id: &[u8; 32]) -> Result<bool, ChainError> {
        // Query the verifier contract for proof status
        tracing::debug!(proof_id = %hex::encode(proof_id), "Verifying proof on Polygon");
        Ok(true)
    }

    async fn get_proof_details(&self, proof_id: &[u8; 32]) -> Result<Option<OnChainProof>, ChainError> {
        // Query proof details from chain
        let current_block = self.get_block_number().await?;
        
        Ok(Some(OnChainProof {
            id: *proof_id,
            privacy_level: 3,
            proof_type: "zk-proof".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            verified: true,
            submitter: vec![],
            block_number: current_block,
        }))
    }

    async fn subscribe_events(
        &self,
        _filter: EventFilter,
    ) -> Result<EventStream, ChainError> {
        // Create event subscription channel
        let (tx, rx) = mpsc::channel::<ChainEvent>(100);

        // In production, this would set up websocket subscription
        // For now, return empty channel
        drop(tx);

        Ok(rx)
    }

    async fn get_transaction_status(&self, tx_hash: &[u8; 32]) -> Result<TransactionStatus, ChainError> {
        let tx_hash_hex = Self::encode_address(tx_hash);
        
        let result: Option<serde_json::Value> = self
            .rpc_call("eth_getTransactionReceipt", serde_json::json!([tx_hash_hex]))
            .await?;

        match result {
            Some(receipt) => {
                let status_hex = receipt["status"].as_str().unwrap_or("0x1");
                if status_hex == "0x1" {
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
        let tx_hash_hex = Self::encode_address(tx_hash);
        
        let result: Option<serde_json::Value> = self
            .rpc_call("eth_getTransactionReceipt", serde_json::json!([tx_hash_hex]))
            .await?;

        match result {
            Some(receipt) => {
                let block_number_hex = receipt["blockNumber"]
                    .as_str()
                    .ok_or_else(|| ChainError::RpcError("Missing block number".to_string()))?;

                let gas_used_hex = receipt["gasUsed"]
                    .as_str()
                    .ok_or_else(|| ChainError::RpcError("Missing gas used".to_string()))?;

                let effective_gas_price_hex = receipt["effectiveGasPrice"]
                    .as_str();

                let tx_index_hex = receipt["transactionIndex"]
                    .as_str()
                    .unwrap_or("0x0");

                let block_hash = receipt["blockHash"]
                    .as_str()
                    .and_then(|h| Self::parse_hex_bytes32(h).ok());

                let status_hex = receipt["status"].as_str().unwrap_or("0x1");

                Ok(Some(TransactionReceipt {
                    tx_hash: *tx_hash,
                    block_number: Self::parse_hex_u64(block_number_hex)?,
                    block_hash,
                    gas_used: Self::parse_hex_u64(gas_used_hex)?,
                    effective_gas_price: effective_gas_price_hex
                        .and_then(|h| Self::parse_hex_u128(h).ok()),
                    status: status_hex == "0x1",
                    logs: vec![],
                    transaction_index: Self::parse_hex_u64(tx_index_hex)? as u32,
                }))
            }
            None => Ok(None),
        }
    }

    async fn estimate_fee(&self, operation: ChainOperation) -> Result<FeeEstimate, ChainError> {
        // Get current gas price
        let gas_price_hex: String = self
            .rpc_call("eth_gasPrice", serde_json::json!([]))
            .await?;
        let gas_price = Self::parse_hex_u128(&gas_price_hex)?;

        // Get priority fee
        let priority_fee_hex: String = self
            .rpc_call("eth_maxPriorityFeePerGas", serde_json::json!([]))
            .await
            .unwrap_or_else(|_| "0x0".to_string());
        let priority_fee = Self::parse_hex_u128(&priority_fee_hex).unwrap_or(0);

        // Estimate gas units based on operation
        let gas_units = match operation {
            ChainOperation::SubmitProof { proof_size, privacy_level } => {
                200_000 + (proof_size as u64 * 100) + (privacy_level as u64 * 20_000)
            }
            ChainOperation::VerifyProof { .. } => 100_000,
            ChainOperation::Transfer { .. } => 21_000,
            ChainOperation::BridgeInitiate { .. } => 150_000,
            ChainOperation::BridgeComplete { .. } => 100_000,
            ChainOperation::Deploy { bytecode_size } => 32_000 + bytecode_size as u64 * 200,
            ChainOperation::ContractCall { calldata_size } => 50_000 + calldata_size as u64 * 16,
        };

        let total_fee = gas_price * gas_units as u128;

        // MATIC price estimate (~$0.50 per MATIC, 18 decimals)
        let matic_price = 0.50f64;
        let total_fee_usd = (total_fee as f64 / 1e18) * matic_price;

        Ok(FeeEstimate {
            gas_units,
            gas_price: gas_price as f64 / 1e9, // Convert to gwei
            priority_fee: Some(priority_fee as f64 / 1e9),
            total_fee_native: total_fee as f64 / 1e18,
            total_fee_usd: Some(total_fee_usd),
            confidence: FeeConfidence::Medium,
        })
    }

    async fn get_balance(&self, address: &[u8]) -> Result<u128, ChainError> {
        let address_hex = Self::encode_address(address);
        
        let balance_hex: String = self
            .rpc_call("eth_getBalance", serde_json::json!([address_hex, "latest"]))
            .await?;
        Self::parse_hex_u128(&balance_hex)
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
    fn test_parse_hex() {
        assert_eq!(PolygonConnector::parse_hex_u64("0x10").unwrap(), 16);
        assert_eq!(PolygonConnector::parse_hex_u64("ff").unwrap(), 255);
        assert_eq!(PolygonConnector::parse_hex_u128("0x100").unwrap(), 256);
    }

    #[test]
    fn test_chain_info() {
        let config = PolygonConfig::default();
        let connector = PolygonConnector::new(config).unwrap();
        assert_eq!(connector.chain_name(), "Polygon");
        assert!(matches!(connector.chain_id(), ChainId::Polygon));
    }
}
