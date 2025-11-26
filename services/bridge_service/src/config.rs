//! Bridge Service Configuration
//! 
//! Manages cross-chain bridge configuration for NexusZero Protocol.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main bridge service configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BridgeConfig {
    /// Service configuration
    pub service: ServiceConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Redis configuration
    pub redis: RedisConfig,
    
    /// Supported chains configuration
    pub chains: HashMap<String, ChainConfig>,
    
    /// HTLC configuration
    pub htlc: HtlcConfig,
    
    /// Fee configuration
    pub fees: FeeConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Relayer configuration
    pub relayer: RelayerConfig,
}

/// Service configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServiceConfig {
    /// Service name
    pub name: String,
    
    /// Host to bind to
    pub host: String,
    
    /// Port to listen on
    pub port: u16,
    
    /// Environment
    pub environment: String,
    
    /// Log level
    pub log_level: String,
    
    /// Enable metrics
    pub metrics_enabled: bool,
    
    /// Worker thread count
    pub worker_threads: usize,
}

/// Database configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL
    pub url: String,
    
    /// Maximum connection pool size
    pub max_connections: u32,
    
    /// Minimum connection pool size
    pub min_connections: u32,
    
    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
    
    /// Enable SQL logging
    pub log_statements: bool,
}

/// Redis configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,
    
    /// Connection pool size
    pub pool_size: u32,
    
    /// Key prefix for namespacing
    pub key_prefix: String,
}

/// Chain-specific configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChainConfig {
    /// Chain identifier
    pub chain_id: String,
    
    /// Chain type (EVM, Bitcoin, Solana, etc.)
    pub chain_type: ChainType,
    
    /// RPC endpoint URL
    pub rpc_url: String,
    
    /// WebSocket endpoint (optional)
    pub ws_url: Option<String>,
    
    /// Chain name for display
    pub name: String,
    
    /// Native token symbol
    pub native_token: String,
    
    /// Block confirmation count for finality
    pub confirmations_required: u32,
    
    /// Average block time in seconds
    pub block_time_secs: u64,
    
    /// Bridge contract address (for EVM chains)
    pub bridge_contract: Option<String>,
    
    /// HTLC contract address (for EVM chains)
    pub htlc_contract: Option<String>,
    
    /// Whether chain is enabled
    pub enabled: bool,
    
    /// Gas price multiplier
    pub gas_price_multiplier: f64,
    
    /// Maximum gas limit
    pub max_gas_limit: u64,
    
    /// Supported assets on this chain
    pub supported_assets: Vec<AssetConfig>,
}

/// Chain type enumeration
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ChainType {
    /// Ethereum Virtual Machine compatible chains
    Evm,
    /// Bitcoin and UTXO-based chains
    Bitcoin,
    /// Solana
    Solana,
    /// Cosmos SDK chains
    Cosmos,
    /// Polkadot/Substrate chains
    Substrate,
    /// Near Protocol
    Near,
    /// Custom chain type
    Custom(String),
}

/// Asset configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AssetConfig {
    /// Asset symbol
    pub symbol: String,
    
    /// Asset name
    pub name: String,
    
    /// Contract address (for tokens)
    pub contract_address: Option<String>,
    
    /// Decimal places
    pub decimals: u8,
    
    /// Minimum transfer amount
    pub min_amount: String,
    
    /// Maximum transfer amount
    pub max_amount: String,
    
    /// Whether asset is enabled for bridging
    pub enabled: bool,
}

/// HTLC (Hash Time-Locked Contract) configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HtlcConfig {
    /// Default timelock duration in seconds
    pub default_timelock_secs: u64,
    
    /// Minimum timelock duration
    pub min_timelock_secs: u64,
    
    /// Maximum timelock duration
    pub max_timelock_secs: u64,
    
    /// Hash algorithm for secret hashing
    pub hash_algorithm: String,
    
    /// Secret length in bytes
    pub secret_length: usize,
    
    /// Grace period for refunds after expiry
    pub refund_grace_period_secs: u64,
}

/// Fee configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FeeConfig {
    /// Base fee in basis points (1 = 0.01%)
    pub base_fee_bps: u32,
    
    /// Minimum fee in USD equivalent
    pub min_fee_usd: f64,
    
    /// Maximum fee in USD equivalent
    pub max_fee_usd: f64,
    
    /// Dynamic fee adjustment enabled
    pub dynamic_fees_enabled: bool,
    
    /// Fee recipient address
    pub fee_recipient: String,
    
    /// Chain-specific fee overrides
    pub chain_overrides: HashMap<String, ChainFeeConfig>,
}

/// Chain-specific fee configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChainFeeConfig {
    /// Additional fee in basis points for this chain
    pub additional_fee_bps: u32,
    
    /// Flat fee in native token
    pub flat_fee: String,
}

/// Security configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SecurityConfig {
    /// Maximum single transfer amount in USD
    pub max_transfer_usd: f64,
    
    /// Daily transfer limit per user in USD
    pub daily_limit_per_user_usd: f64,
    
    /// Rate limiting window in seconds
    pub rate_limit_window_secs: u64,
    
    /// Maximum transfers per window
    pub max_transfers_per_window: u32,
    
    /// Enable sanctions screening
    pub sanctions_screening_enabled: bool,
    
    /// Trusted relayer addresses
    pub trusted_relayers: Vec<String>,
    
    /// Pause bridge operations
    pub paused: bool,
}

/// Relayer configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RelayerConfig {
    /// Enable automatic relaying
    pub auto_relay_enabled: bool,
    
    /// Relayer private key (encrypted reference)
    pub private_key_ref: String,
    
    /// Minimum profit threshold for auto-relay
    pub min_profit_threshold_usd: f64,
    
    /// Maximum pending transactions per chain
    pub max_pending_per_chain: u32,
    
    /// Transaction timeout in seconds
    pub tx_timeout_secs: u64,
    
    /// Retry attempts for failed transactions
    pub retry_attempts: u32,
    
    /// Retry delay in seconds
    pub retry_delay_secs: u64,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        let mut chains = HashMap::new();
        
        // Default Ethereum configuration
        chains.insert(
            "ethereum".to_string(),
            ChainConfig {
                chain_id: "1".to_string(),
                chain_type: ChainType::Evm,
                rpc_url: "https://eth-mainnet.g.alchemy.com/v2/YOUR_KEY".to_string(),
                ws_url: Some("wss://eth-mainnet.g.alchemy.com/v2/YOUR_KEY".to_string()),
                name: "Ethereum".to_string(),
                native_token: "ETH".to_string(),
                confirmations_required: 12,
                block_time_secs: 12,
                bridge_contract: None,
                htlc_contract: None,
                enabled: true,
                gas_price_multiplier: 1.1,
                max_gas_limit: 500000,
                supported_assets: vec![],
            },
        );
        
        // Default Polygon configuration
        chains.insert(
            "polygon".to_string(),
            ChainConfig {
                chain_id: "137".to_string(),
                chain_type: ChainType::Evm,
                rpc_url: "https://polygon-rpc.com".to_string(),
                ws_url: None,
                name: "Polygon".to_string(),
                native_token: "MATIC".to_string(),
                confirmations_required: 128,
                block_time_secs: 2,
                bridge_contract: None,
                htlc_contract: None,
                enabled: true,
                gas_price_multiplier: 1.2,
                max_gas_limit: 500000,
                supported_assets: vec![],
            },
        );
        
        Self {
            service: ServiceConfig {
                name: "bridge-service".to_string(),
                host: "0.0.0.0".to_string(),
                port: 8084,
                environment: "development".to_string(),
                log_level: "info".to_string(),
                metrics_enabled: true,
                worker_threads: 4,
            },
            database: DatabaseConfig {
                url: "postgresql://nexuszero:nexuszero@localhost:5432/bridge_db".to_string(),
                max_connections: 10,
                min_connections: 2,
                connect_timeout_secs: 30,
                log_statements: false,
            },
            redis: RedisConfig {
                url: "redis://localhost:6379/3".to_string(),
                pool_size: 10,
                key_prefix: "bridge:".to_string(),
            },
            chains,
            htlc: HtlcConfig {
                default_timelock_secs: 3600,
                min_timelock_secs: 600,
                max_timelock_secs: 86400,
                hash_algorithm: "sha256".to_string(),
                secret_length: 32,
                refund_grace_period_secs: 300,
            },
            fees: FeeConfig {
                base_fee_bps: 30,
                min_fee_usd: 1.0,
                max_fee_usd: 1000.0,
                dynamic_fees_enabled: true,
                fee_recipient: "0x0000000000000000000000000000000000000000".to_string(),
                chain_overrides: HashMap::new(),
            },
            security: SecurityConfig {
                max_transfer_usd: 100000.0,
                daily_limit_per_user_usd: 500000.0,
                rate_limit_window_secs: 3600,
                max_transfers_per_window: 50,
                sanctions_screening_enabled: true,
                trusted_relayers: vec![],
                paused: false,
            },
            relayer: RelayerConfig {
                auto_relay_enabled: false,
                private_key_ref: "vault://bridge/relayer/private_key".to_string(),
                min_profit_threshold_usd: 0.10,
                max_pending_per_chain: 100,
                tx_timeout_secs: 300,
                retry_attempts: 3,
                retry_delay_secs: 30,
            },
        }
    }
}

impl BridgeConfig {
    /// Load configuration from environment
    pub fn from_env() -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::Environment::with_prefix("BRIDGE").separator("__"))
            .build()?;
        
        settings.try_deserialize()
    }
    
    /// Get chain configuration by chain identifier
    pub fn get_chain(&self, chain_id: &str) -> Option<&ChainConfig> {
        self.chains.get(chain_id)
    }
    
    /// Get all enabled chains
    pub fn enabled_chains(&self) -> Vec<(&String, &ChainConfig)> {
        self.chains
            .iter()
            .filter(|(_, config)| config.enabled)
            .collect()
    }
    
    /// Check if a chain pair is supported
    pub fn is_route_supported(&self, source: &str, destination: &str) -> bool {
        self.chains.get(source).map(|c| c.enabled).unwrap_or(false)
            && self.chains.get(destination).map(|c| c.enabled).unwrap_or(false)
    }
    
    /// Calculate fee for a transfer
    pub fn calculate_fee(&self, chain_id: &str, amount_usd: f64) -> f64 {
        let base_fee = amount_usd * (self.fees.base_fee_bps as f64 / 10000.0);
        
        let additional_fee = self.fees.chain_overrides
            .get(chain_id)
            .map(|c| amount_usd * (c.additional_fee_bps as f64 / 10000.0))
            .unwrap_or(0.0);
        
        let total_fee = base_fee + additional_fee;
        
        total_fee
            .max(self.fees.min_fee_usd)
            .min(self.fees.max_fee_usd)
    }
}
