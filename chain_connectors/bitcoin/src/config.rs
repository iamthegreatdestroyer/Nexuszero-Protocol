//! Bitcoin connector configuration

use serde::{Deserialize, Serialize};

/// Configuration for the Bitcoin connector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BitcoinConfig {
    /// Bitcoin Core RPC URL
    pub rpc_url: String,
    
    /// RPC username
    pub rpc_user: String,
    
    /// RPC password
    pub rpc_password: String,
    
    /// Network type (mainnet, testnet, signet, regtest)
    pub network: BitcoinNetwork,
    
    /// Wallet name (if using Bitcoin Core wallet)
    pub wallet_name: Option<String>,
    
    /// Number of confirmations to consider final
    pub confirmations: u32,
    
    /// Request timeout in seconds
    pub timeout_secs: u64,
    
    /// Use Taproot (P2TR) by default for new outputs
    pub use_taproot: bool,
    
    /// Minimum fee rate in sat/vB
    pub min_fee_rate: f64,
    
    /// Maximum fee rate in sat/vB (safety limit)
    pub max_fee_rate: f64,
    
    /// Enable RBF (Replace-By-Fee) for transactions
    pub enable_rbf: bool,
}

/// Bitcoin network types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BitcoinNetwork {
    /// Bitcoin Mainnet
    Mainnet,
    /// Bitcoin Testnet3
    Testnet,
    /// Bitcoin Signet
    Signet,
    /// Bitcoin Regtest (local development)
    Regtest,
}

impl BitcoinNetwork {
    /// Get the network as bitcoin crate Network type
    pub fn to_bitcoin_network(&self) -> bitcoin::Network {
        match self {
            BitcoinNetwork::Mainnet => bitcoin::Network::Bitcoin,
            BitcoinNetwork::Testnet => bitcoin::Network::Testnet,
            BitcoinNetwork::Signet => bitcoin::Network::Signet,
            BitcoinNetwork::Regtest => bitcoin::Network::Regtest,
        }
    }
    
    /// Get the default port for this network
    pub fn default_port(&self) -> u16 {
        match self {
            BitcoinNetwork::Mainnet => 8332,
            BitcoinNetwork::Testnet => 18332,
            BitcoinNetwork::Signet => 38332,
            BitcoinNetwork::Regtest => 18443,
        }
    }
}

impl Default for BitcoinConfig {
    fn default() -> Self {
        Self {
            rpc_url: "http://127.0.0.1:8332".to_string(),
            rpc_user: "bitcoin".to_string(),
            rpc_password: String::new(),
            network: BitcoinNetwork::Mainnet,
            wallet_name: None,
            confirmations: 6,
            timeout_secs: 60,
            use_taproot: true,
            min_fee_rate: 1.0,
            max_fee_rate: 500.0,
            enable_rbf: true,
        }
    }
}

impl BitcoinConfig {
    /// Create configuration for mainnet
    pub fn mainnet(rpc_url: impl Into<String>, user: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            rpc_user: user.into(),
            rpc_password: password.into(),
            network: BitcoinNetwork::Mainnet,
            ..Default::default()
        }
    }

    /// Create configuration for testnet
    pub fn testnet(rpc_url: impl Into<String>, user: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            rpc_user: user.into(),
            rpc_password: password.into(),
            network: BitcoinNetwork::Testnet,
            confirmations: 1,
            ..Default::default()
        }
    }

    /// Create configuration for regtest (local development)
    pub fn regtest(user: impl Into<String>, password: impl Into<String>) -> Self {
        Self {
            rpc_url: "http://127.0.0.1:18443".to_string(),
            rpc_user: user.into(),
            rpc_password: password.into(),
            network: BitcoinNetwork::Regtest,
            confirmations: 1,
            ..Default::default()
        }
    }

    /// Set wallet name
    pub fn with_wallet(mut self, name: impl Into<String>) -> Self {
        self.wallet_name = Some(name.into());
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.rpc_url.is_empty() {
            return Err("RPC URL is required".to_string());
        }
        if self.rpc_user.is_empty() {
            return Err("RPC user is required".to_string());
        }
        if self.min_fee_rate <= 0.0 {
            return Err("Minimum fee rate must be positive".to_string());
        }
        if self.max_fee_rate < self.min_fee_rate {
            return Err("Maximum fee rate must be >= minimum fee rate".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = BitcoinConfig::default();
        assert_eq!(config.network, BitcoinNetwork::Mainnet);
        assert!(config.use_taproot);
        assert_eq!(config.confirmations, 6);
    }

    #[test]
    fn test_network_ports() {
        assert_eq!(BitcoinNetwork::Mainnet.default_port(), 8332);
        assert_eq!(BitcoinNetwork::Testnet.default_port(), 18332);
        assert_eq!(BitcoinNetwork::Regtest.default_port(), 18443);
    }

    #[test]
    fn test_config_validation() {
        let config = BitcoinConfig::mainnet("http://localhost:8332", "user", "pass");
        assert!(config.validate().is_ok());

        let bad_config = BitcoinConfig {
            rpc_url: String::new(),
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }
}
