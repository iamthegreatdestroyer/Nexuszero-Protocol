//! Bridge Service Application State
//! 
//! Shared state for the cross-chain bridge service.

use crate::config::BridgeConfig;
use redis::aio::ConnectionManager;
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;

/// Application state shared across handlers
#[derive(Clone)]
pub struct AppState {
    /// Database connection pool
    pub db: PgPool,
    
    /// Redis connection manager
    pub redis: ConnectionManager,
    
    /// Application configuration
    pub config: Arc<BridgeConfig>,
    
    /// Chain clients registry
    pub chain_clients: Arc<ChainClientRegistry>,
    
    /// HTLC manager
    pub htlc_manager: Arc<HtlcManager>,
    
    /// Fee calculator
    pub fee_calculator: Arc<FeeCalculator>,
    
    /// Transfer state cache
    pub transfer_cache: Arc<TransferCache>,
}

impl AppState {
    /// Create new application state
    pub fn new(
        db: PgPool,
        redis: ConnectionManager,
        config: BridgeConfig,
    ) -> Self {
        let config = Arc::new(config);
        
        Self {
            db: db.clone(),
            redis: redis.clone(),
            config: config.clone(),
            chain_clients: Arc::new(ChainClientRegistry::new(config.clone())),
            htlc_manager: Arc::new(HtlcManager::new(db.clone(), config.clone())),
            fee_calculator: Arc::new(FeeCalculator::new(config.clone())),
            transfer_cache: Arc::new(TransferCache::new()),
        }
    }
}

/// Registry of blockchain clients
pub struct ChainClientRegistry {
    /// Configuration
    config: Arc<BridgeConfig>,
    
    /// EVM clients
    evm_clients: RwLock<HashMap<String, EvmClient>>,
    
    /// Bitcoin client
    bitcoin_client: RwLock<Option<BitcoinClient>>,
    
    /// Solana client
    solana_client: RwLock<Option<SolanaClient>>,
}

impl ChainClientRegistry {
    /// Create new registry
    pub fn new(config: Arc<BridgeConfig>) -> Self {
        Self {
            config,
            evm_clients: RwLock::new(HashMap::new()),
            bitcoin_client: RwLock::new(None),
            solana_client: RwLock::new(None),
        }
    }
    
    /// Get or create EVM client for chain
    pub async fn get_evm_client(&self, chain_id: &str) -> Option<EvmClient> {
        let clients = self.evm_clients.read().await;
        clients.get(chain_id).cloned()
    }
    
    /// Initialize all configured chain clients
    pub async fn initialize(&self) -> Result<(), anyhow::Error> {
        for (chain_id, chain_config) in self.config.enabled_chains() {
            match chain_config.chain_type {
                crate::config::ChainType::Evm => {
                    let client = EvmClient::new(
                        chain_config.rpc_url.clone(),
                        chain_config.ws_url.clone(),
                        chain_id.clone(),
                    );
                    let mut clients = self.evm_clients.write().await;
                    clients.insert(chain_id.clone(), client);
                }
                crate::config::ChainType::Bitcoin => {
                    let client = BitcoinClient::new(chain_config.rpc_url.clone());
                    *self.bitcoin_client.write().await = Some(client);
                }
                crate::config::ChainType::Solana => {
                    let client = SolanaClient::new(chain_config.rpc_url.clone());
                    *self.solana_client.write().await = Some(client);
                }
                _ => {
                    tracing::warn!("Unsupported chain type for {}", chain_id);
                }
            }
        }
        
        Ok(())
    }
}

/// EVM blockchain client wrapper
#[derive(Clone)]
pub struct EvmClient {
    /// RPC URL
    pub rpc_url: String,
    
    /// WebSocket URL
    pub ws_url: Option<String>,
    
    /// Chain ID
    pub chain_id: String,
}

impl EvmClient {
    pub fn new(rpc_url: String, ws_url: Option<String>, chain_id: String) -> Self {
        Self {
            rpc_url,
            ws_url,
            chain_id,
        }
    }
    
    /// Get current block number
    pub async fn get_block_number(&self) -> Result<u64, anyhow::Error> {
        // Implementation using ethers
        Ok(0) // Placeholder
    }
    
    /// Get transaction receipt
    pub async fn get_transaction_receipt(
        &self,
        tx_hash: &str,
    ) -> Result<Option<TransactionReceipt>, anyhow::Error> {
        // Implementation using ethers
        Ok(None) // Placeholder
    }
}

/// Bitcoin client wrapper
#[derive(Clone)]
pub struct BitcoinClient {
    pub rpc_url: String,
}

impl BitcoinClient {
    pub fn new(rpc_url: String) -> Self {
        Self { rpc_url }
    }
}

/// Solana client wrapper
#[derive(Clone)]
pub struct SolanaClient {
    pub rpc_url: String,
}

impl SolanaClient {
    pub fn new(rpc_url: String) -> Self {
        Self { rpc_url }
    }
}

/// Transaction receipt structure
#[derive(Debug, Clone)]
pub struct TransactionReceipt {
    pub tx_hash: String,
    pub block_number: u64,
    pub status: bool,
    pub gas_used: u64,
}

/// HTLC Manager for atomic swaps
pub struct HtlcManager {
    /// Database pool
    db: PgPool,
    
    /// Configuration
    config: Arc<BridgeConfig>,
}

impl HtlcManager {
    pub fn new(db: PgPool, config: Arc<BridgeConfig>) -> Self {
        Self { db, config }
    }
    
    /// Generate a new secret for HTLC
    pub fn generate_secret(&self) -> (String, String) {
        use rand::Rng;
        use sha2::{Sha256, Digest};
        
        let mut rng = rand::thread_rng();
        let mut secret = vec![0u8; self.config.htlc.secret_length];
        rng.fill(&mut secret[..]);
        
        let secret_hex = hex::encode(&secret);
        
        let mut hasher = Sha256::new();
        hasher.update(&secret);
        let secret_hash = hex::encode(hasher.finalize());
        
        (secret_hex, secret_hash)
    }
    
    /// Calculate timelock expiry
    pub fn calculate_timelock(
        &self,
        timelock_secs: Option<u64>,
    ) -> chrono::DateTime<chrono::Utc> {
        let duration = timelock_secs.unwrap_or(self.config.htlc.default_timelock_secs);
        let duration = duration.clamp(
            self.config.htlc.min_timelock_secs,
            self.config.htlc.max_timelock_secs,
        );
        
        chrono::Utc::now() + chrono::Duration::seconds(duration as i64)
    }
}

/// Fee Calculator
pub struct FeeCalculator {
    /// Configuration
    config: Arc<BridgeConfig>,
}

impl FeeCalculator {
    pub fn new(config: Arc<BridgeConfig>) -> Self {
        Self { config }
    }
    
    /// Calculate bridge fee for a transfer
    pub fn calculate_bridge_fee(
        &self,
        source_chain: &str,
        amount_usd: f64,
    ) -> rust_decimal::Decimal {
        let fee = self.config.calculate_fee(source_chain, amount_usd);
        rust_decimal::Decimal::from_f64_retain(fee).unwrap_or_default()
    }
    
    /// Estimate gas fee for destination chain
    pub async fn estimate_gas_fee(
        &self,
        destination_chain: &str,
    ) -> rust_decimal::Decimal {
        // Get chain configuration
        if let Some(chain) = self.config.get_chain(destination_chain) {
            // Simplified gas estimation
            let base_gas = 100000u64;
            let gas_price = 50_000_000_000u64; // 50 gwei placeholder
            
            let fee_wei = base_gas * gas_price;
            let fee_eth = fee_wei as f64 / 1e18;
            
            rust_decimal::Decimal::from_f64_retain(fee_eth).unwrap_or_default()
        } else {
            rust_decimal::Decimal::ZERO
        }
    }
}

/// In-memory transfer state cache
pub struct TransferCache {
    /// Active transfers by ID
    transfers: RwLock<HashMap<String, CachedTransfer>>,
}

impl TransferCache {
    pub fn new() -> Self {
        Self {
            transfers: RwLock::new(HashMap::new()),
        }
    }
    
    /// Cache a transfer
    pub async fn set(&self, transfer_id: &str, transfer: CachedTransfer) {
        let mut cache = self.transfers.write().await;
        cache.insert(transfer_id.to_string(), transfer);
    }
    
    /// Get cached transfer
    pub async fn get(&self, transfer_id: &str) -> Option<CachedTransfer> {
        let cache = self.transfers.read().await;
        cache.get(transfer_id).cloned()
    }
    
    /// Remove from cache
    pub async fn remove(&self, transfer_id: &str) {
        let mut cache = self.transfers.write().await;
        cache.remove(transfer_id);
    }
    
    /// Clean expired entries
    pub async fn cleanup(&self) {
        let now = chrono::Utc::now();
        let mut cache = self.transfers.write().await;
        cache.retain(|_, v| v.expires_at > now);
    }
}

/// Cached transfer state
#[derive(Clone)]
pub struct CachedTransfer {
    pub transfer_id: String,
    pub status: crate::models::TransferStatus,
    pub source_chain: String,
    pub destination_chain: String,
    pub last_updated: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}
