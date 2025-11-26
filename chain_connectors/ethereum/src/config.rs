//! Ethereum connector configuration

use serde::{Deserialize, Serialize};

/// Configuration for the Ethereum connector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EthereumConfig {
    /// HTTP RPC endpoint URL
    pub rpc_url: String,
    
    /// WebSocket RPC endpoint URL (for subscriptions)
    pub ws_url: Option<String>,
    
    /// Chain ID (1 = mainnet, 11155111 = sepolia, etc.)
    pub chain_id: u64,
    
    /// NexusZero Verifier contract address
    pub verifier_address: String,
    
    /// NexusZero Bridge contract address
    pub bridge_address: Option<String>,
    
    /// Private key for signing transactions (hex encoded, without 0x prefix)
    /// WARNING: In production, use a secure key management system
    pub private_key: Option<String>,
    
    /// Maximum gas price in gwei (for safety)
    pub max_gas_price_gwei: Option<f64>,
    
    /// Number of confirmations to wait for
    pub confirmations: u32,
    
    /// Request timeout in seconds
    pub timeout_secs: u64,
    
    /// Enable EIP-1559 transactions
    pub use_eip1559: bool,
    
    /// Priority fee (tip) in gwei for EIP-1559
    pub priority_fee_gwei: Option<f64>,
}

impl Default for EthereumConfig {
    fn default() -> Self {
        Self {
            rpc_url: "http://localhost:8545".to_string(),
            ws_url: None,
            chain_id: 1,
            verifier_address: String::new(),
            bridge_address: None,
            private_key: None,
            max_gas_price_gwei: Some(500.0),
            confirmations: 2,
            timeout_secs: 60,
            use_eip1559: true,
            priority_fee_gwei: Some(2.0),
        }
    }
}

impl EthereumConfig {
    /// Create a new configuration for Ethereum mainnet
    pub fn mainnet(rpc_url: impl Into<String>, verifier_address: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            chain_id: 1,
            verifier_address: verifier_address.into(),
            ..Default::default()
        }
    }

    /// Create a new configuration for Sepolia testnet
    pub fn sepolia(rpc_url: impl Into<String>, verifier_address: impl Into<String>) -> Self {
        Self {
            rpc_url: rpc_url.into(),
            chain_id: 11155111,
            verifier_address: verifier_address.into(),
            confirmations: 1,
            ..Default::default()
        }
    }

    /// Create a new configuration for local development
    pub fn local(verifier_address: impl Into<String>) -> Self {
        Self {
            rpc_url: "http://localhost:8545".to_string(),
            ws_url: Some("ws://localhost:8545".to_string()),
            chain_id: 31337,
            verifier_address: verifier_address.into(),
            confirmations: 1,
            ..Default::default()
        }
    }

    /// Set the WebSocket URL
    pub fn with_ws_url(mut self, url: impl Into<String>) -> Self {
        self.ws_url = Some(url.into());
        self
    }

    /// Set the bridge contract address
    pub fn with_bridge_address(mut self, address: impl Into<String>) -> Self {
        self.bridge_address = Some(address.into());
        self
    }

    /// Set the private key for signing
    pub fn with_private_key(mut self, key: impl Into<String>) -> Self {
        self.private_key = Some(key.into());
        self
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.rpc_url.is_empty() {
            return Err("RPC URL is required".to_string());
        }
        if self.verifier_address.is_empty() {
            return Err("Verifier address is required".to_string());
        }
        if !self.verifier_address.starts_with("0x") || self.verifier_address.len() != 42 {
            return Err("Invalid verifier address format".to_string());
        }
        if let Some(ref bridge) = self.bridge_address {
            if !bridge.starts_with("0x") || bridge.len() != 42 {
                return Err("Invalid bridge address format".to_string());
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = EthereumConfig::default();
        assert_eq!(config.chain_id, 1);
        assert!(config.use_eip1559);
    }

    #[test]
    fn test_mainnet_config() {
        let config = EthereumConfig::mainnet(
            "https://eth.example.com",
            "0x1234567890123456789012345678901234567890",
        );
        assert_eq!(config.chain_id, 1);
    }

    #[test]
    fn test_config_validation() {
        let config = EthereumConfig::mainnet(
            "https://eth.example.com",
            "0x1234567890123456789012345678901234567890",
        );
        assert!(config.validate().is_ok());

        let bad_config = EthereumConfig {
            verifier_address: "invalid".to_string(),
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());
    }
}
