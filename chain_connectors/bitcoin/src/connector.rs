//! Bitcoin blockchain connector implementation.

use async_trait::async_trait;
use bitcoin::hashes::Hash;
use bitcoin::{Address, Network, Txid};
use bitcoincore_rpc::{Auth, Client, RpcApi};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock};
use tracing::{debug, info, warn};

use chain_connectors_common::{
    BlockInfo, ChainConnector, ChainError, ChainId, ChainOperation, EventFilter,
    EventStream, FeeConfidence, FeeEstimate, OnChainProof, ProofMetadata,
    TransactionReceipt, TransactionStatus,
};

use crate::config::BitcoinConfig;
use crate::error::BitcoinError;

/// Bitcoin blockchain connector.
pub struct BitcoinConnector {
    config: BitcoinConfig,
    client: Arc<Client>,
    connected: Arc<RwLock<bool>>,
}

impl BitcoinConnector {
    /// Create a new Bitcoin connector.
    pub fn new(config: BitcoinConfig) -> Result<Self, BitcoinError> {
        let auth = if config.rpc_user.is_empty() {
            Auth::None
        } else {
            Auth::UserPass(config.rpc_user.clone(), config.rpc_password.clone())
        };

        let client = Client::new(&config.rpc_url, auth)
            .map_err(|e| BitcoinError::RpcError(e.to_string()))?;

        Ok(Self {
            config,
            client: Arc::new(client),
            connected: Arc::new(RwLock::new(false)),
        })
    }

    /// Connect to the Bitcoin node.
    pub async fn connect(&mut self) -> Result<(), ChainError> {
        info!("Connecting to Bitcoin node at {}...", self.config.rpc_url);

        // Test connection
        let info = self.client
            .get_blockchain_info()
            .map_err(|e| ChainError::ConnectionFailed(e.to_string()))?;

        info!("Connected to Bitcoin {} at height {}", info.chain, info.blocks);

        let mut connected = self.connected.write().await;
        *connected = true;

        Ok(())
    }

    /// Convert bytes to Txid.
    fn bytes_to_txid(bytes: &[u8; 32]) -> Txid {
        Txid::from_slice(bytes).unwrap_or_else(|_| {
            Txid::from_slice(&[0u8; 32]).unwrap()
        })
    }
}

#[async_trait]
impl ChainConnector for BitcoinConnector {
    fn chain_id(&self) -> ChainId {
        ChainId::Bitcoin
    }

    fn chain_name(&self) -> &str {
        "Bitcoin"
    }

    async fn is_healthy(&self) -> bool {
        match self.client.get_blockchain_info() {
            Ok(info) => {
                debug!("Bitcoin health check OK, height: {}", info.blocks);
                true
            }
            Err(e) => {
                warn!("Bitcoin health check failed: {}", e);
                false
            }
        }
    }

    async fn get_block_number(&self) -> Result<u64, ChainError> {
        let count = self.client
            .get_block_count()
            .map_err(|e| ChainError::RpcError(e.to_string()))?;
        Ok(count)
    }

    async fn get_block(&self, block_number: u64) -> Result<BlockInfo, ChainError> {
        let hash = self.client
            .get_block_hash(block_number)
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        let header = self.client
            .get_block_header(&hash)
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        let block = self.client
            .get_block(&hash)
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        Ok(BlockInfo {
            number: block_number,
            hash: *hash.as_byte_array(),
            parent_hash: *header.prev_blockhash.as_byte_array(),
            timestamp: header.time as u64,
            transaction_count: block.txdata.len() as u32,
        })
    }

    async fn submit_proof(
        &self,
        _proof: &[u8],
        _metadata: &ProofMetadata,
    ) -> Result<TransactionReceipt, ChainError> {
        // Bitcoin doesn't have native smart contracts
        // This would use OP_RETURN or a sidechain/Layer 2
        warn!("Bitcoin proof submission requires OP_RETURN or Layer 2");

        Err(ChainError::ChainNotSupported(
            "Bitcoin requires specialized proof embedding (OP_RETURN, Stacks, etc.)".to_string()
        ))
    }

    async fn verify_proof(&self, _proof_id: &[u8; 32]) -> Result<bool, ChainError> {
        // Would need to look up OP_RETURN data in transactions
        Err(ChainError::ChainNotSupported(
            "Bitcoin proof verification requires specialized implementation".to_string()
        ))
    }

    async fn get_proof_details(
        &self,
        _proof_id: &[u8; 32],
    ) -> Result<Option<OnChainProof>, ChainError> {
        // Would need specialized indexer
        Ok(None)
    }

    async fn subscribe_events(&self, _filter: EventFilter) -> Result<EventStream, ChainError> {
        let (tx, rx) = mpsc::channel(100);
        drop(tx);
        Ok(rx)
    }

    async fn get_transaction_status(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<TransactionStatus, ChainError> {
        let txid = Self::bytes_to_txid(tx_hash);

        match self.client.get_raw_transaction_info(&txid, None) {
            Ok(info) => {
                let confirmations = info.confirmations.unwrap_or(0);
                if confirmations >= self.config.confirmations {
                    Ok(TransactionStatus::Confirmed)
                } else if confirmations > 0 {
                    Ok(TransactionStatus::Pending)
                } else {
                    Ok(TransactionStatus::Unknown)
                }
            }
            Err(_) => Ok(TransactionStatus::Unknown),
        }
    }

    async fn get_transaction_receipt(
        &self,
        tx_hash: &[u8; 32],
    ) -> Result<Option<TransactionReceipt>, ChainError> {
        let txid = Self::bytes_to_txid(tx_hash);

        match self.client.get_raw_transaction_info(&txid, None) {
            Ok(info) => {
                let confirmations = info.confirmations.unwrap_or(0);
                let status = confirmations >= self.config.confirmations;
                let block_hash = info.blockhash.map(|h| *h.as_byte_array());

                Ok(Some(TransactionReceipt {
                    tx_hash: *tx_hash,
                    block_number: 0, // Bitcoin doesn't have block number in tx info
                    block_hash,
                    gas_used: info.vsize as u64,
                    status,
                    logs: vec![],
                    effective_gas_price: None,
                    transaction_index: 0,
                }))
            }
            Err(_) => Ok(None),
        }
    }

    async fn estimate_fee(&self, operation: ChainOperation) -> Result<FeeEstimate, ChainError> {
        // Get fee estimate for target blocks
        let target_blocks = 6; // ~1 hour
        let fee_rate = self.client
            .estimate_smart_fee(target_blocks, None)
            .map_err(|e| ChainError::RpcError(e.to_string()))?;

        let sat_per_vbyte = fee_rate.fee_rate
            .map(|r| r.to_sat() as f64 / 1000.0)
            .unwrap_or(10.0);

        let vsize = match operation {
            ChainOperation::SubmitProof { proof_size, .. } => {
                // OP_RETURN transaction estimate
                200 + proof_size as u64
            }
            ChainOperation::Transfer { .. } => 250, // Simple P2WPKH
            ChainOperation::BridgeInitiate { .. } => 400,
            ChainOperation::BridgeComplete { .. } => 350,
            _ => 300,
        };

        let total_fee_sats = (vsize as f64 * sat_per_vbyte) as u64;
        let total_fee_btc = total_fee_sats as f64 / 100_000_000.0;

        // BTC price would come from oracle
        let btc_price_usd = 65000.0;

        Ok(FeeEstimate {
            gas_units: vsize,
            gas_price: sat_per_vbyte,
            priority_fee: None,
            total_fee_native: total_fee_btc,
            total_fee_usd: Some(total_fee_btc * btc_price_usd),
            confidence: FeeConfidence::Medium,
        })
    }

    async fn get_balance(&self, address: &[u8]) -> Result<u128, ChainError> {
        // Convert bytes to address string
        let addr_str = String::from_utf8_lossy(address);
        let address = Address::from_str(&addr_str)
            .map_err(|e| ChainError::InvalidAddress(e.to_string()))?
            .assume_checked();

        // For a full node, we'd need to scan UTXOs
        // This is a simplified version that would work with electrum or similar
        warn!("Balance lookup requires UTXO scanning or external indexer");

        Ok(0)
    }

    fn verifier_address(&self) -> Option<&[u8]> {
        None // Bitcoin doesn't have contract addresses
    }

    fn bridge_address(&self) -> Option<&[u8]> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_txid_conversion() {
        let bytes = [1u8; 32];
        let txid = BitcoinConnector::bytes_to_txid(&bytes);
        assert!(!txid.to_string().is_empty());
    }
}
