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

    // ===== HARDENING TESTS =====

    #[test]
    fn test_default_config_values() {
        let config = EthereumConfig::default();
        
        assert_eq!(config.rpc_url, "http://localhost:8545");
        assert!(config.ws_url.is_none());
        assert_eq!(config.chain_id, 1);
        assert!(config.verifier_address.is_empty());
        assert!(config.bridge_address.is_none());
        assert!(config.private_key.is_none());
        assert_eq!(config.max_gas_price_gwei, Some(500.0));
        assert_eq!(config.confirmations, 2);
        assert_eq!(config.timeout_secs, 60);
        assert!(config.use_eip1559);
        assert_eq!(config.priority_fee_gwei, Some(2.0));
    }

    #[test]
    fn test_sepolia_config() {
        let config = EthereumConfig::sepolia(
            "https://sepolia.example.com",
            "0xabcdef0123456789012345678901234567890123",
        );
        
        assert_eq!(config.chain_id, 11155111);
        assert_eq!(config.confirmations, 1);
        assert_eq!(config.rpc_url, "https://sepolia.example.com");
    }

    #[test]
    fn test_local_config() {
        let config = EthereumConfig::local(
            "0x1234567890123456789012345678901234567890",
        );
        
        assert_eq!(config.chain_id, 31337);
        assert_eq!(config.rpc_url, "http://localhost:8545");
        assert_eq!(config.ws_url, Some("ws://localhost:8545".to_string()));
        assert_eq!(config.confirmations, 1);
    }

    #[test]
    fn test_with_ws_url() {
        let config = EthereumConfig::default()
            .with_ws_url("wss://mainnet.infura.io/ws/v3/key");
        
        assert_eq!(config.ws_url, Some("wss://mainnet.infura.io/ws/v3/key".to_string()));
    }

    #[test]
    fn test_with_bridge_address() {
        let config = EthereumConfig::default()
            .with_bridge_address("0xbridge0000000000000000000000000000000000");
        
        assert_eq!(
            config.bridge_address,
            Some("0xbridge0000000000000000000000000000000000".to_string())
        );
    }

    #[test]
    fn test_with_private_key() {
        let config = EthereumConfig::default()
            .with_private_key("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
        
        assert!(config.private_key.is_some());
    }

    #[test]
    fn test_validate_empty_rpc_url() {
        let config = EthereumConfig {
            rpc_url: String::new(),
            verifier_address: "0x1234567890123456789012345678901234567890".to_string(),
            ..Default::default()
        };
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("RPC URL"));
    }

    #[test]
    fn test_validate_empty_verifier_address() {
        let config = EthereumConfig {
            rpc_url: "http://localhost".to_string(),
            verifier_address: String::new(),
            ..Default::default()
        };
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Verifier address"));
    }

    #[test]
    fn test_validate_invalid_verifier_address_no_prefix() {
        let config = EthereumConfig {
            rpc_url: "http://localhost".to_string(),
            verifier_address: "1234567890123456789012345678901234567890".to_string(),
            ..Default::default()
        };
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid verifier address"));
    }

    #[test]
    fn test_validate_invalid_verifier_address_wrong_length() {
        let config = EthereumConfig {
            rpc_url: "http://localhost".to_string(),
            verifier_address: "0x12345".to_string(),
            ..Default::default()
        };
        
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_invalid_bridge_address() {
        let config = EthereumConfig {
            rpc_url: "http://localhost".to_string(),
            verifier_address: "0x1234567890123456789012345678901234567890".to_string(),
            bridge_address: Some("invalid".to_string()),
            ..Default::default()
        };
        
        let result = config.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid bridge address"));
    }

    #[test]
    fn test_validate_with_valid_bridge_address() {
        let config = EthereumConfig {
            rpc_url: "http://localhost".to_string(),
            verifier_address: "0x1234567890123456789012345678901234567890".to_string(),
            bridge_address: Some("0xabcdef0123456789012345678901234567890123".to_string()),
            ..Default::default()
        };
        
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = EthereumConfig::mainnet(
            "https://eth.example.com",
            "0x1234567890123456789012345678901234567890",
        )
        .with_ws_url("wss://eth.example.com")
        .with_bridge_address("0xabcdef0123456789012345678901234567890123");
        
        let json = serde_json::to_string(&config).unwrap();
        let parsed: EthereumConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.rpc_url, config.rpc_url);
        assert_eq!(parsed.chain_id, config.chain_id);
        assert_eq!(parsed.ws_url, config.ws_url);
        assert_eq!(parsed.bridge_address, config.bridge_address);
    }

    #[test]
    fn test_builder_pattern_chaining() {
        let config = EthereumConfig::default()
            .with_ws_url("ws://localhost:8546")
            .with_bridge_address("0x0000000000000000000000000000000000000001")
            .with_private_key("deadbeef");
        
        assert!(config.ws_url.is_some());
        assert!(config.bridge_address.is_some());
        assert!(config.private_key.is_some());
    }

    #[test]
    fn test_config_clone() {
        let config = EthereumConfig::mainnet(
            "https://eth.example.com",
            "0x1234567890123456789012345678901234567890",
        );
        let cloned = config.clone();
        
        assert_eq!(cloned.rpc_url, config.rpc_url);
        assert_eq!(cloned.chain_id, config.chain_id);
    }

    #[test]
    fn test_common_chain_ids() {
        // Mainnet
        let mainnet = EthereumConfig::mainnet(
            "http://localhost",
            "0x1234567890123456789012345678901234567890",
        );
        assert_eq!(mainnet.chain_id, 1);
        
        // Sepolia
        let sepolia = EthereumConfig::sepolia(
            "http://localhost",
            "0x1234567890123456789012345678901234567890",
        );
        assert_eq!(sepolia.chain_id, 11155111);
        
        // Local (Hardhat/Anvil)
        let local = EthereumConfig::local(
            "0x1234567890123456789012345678901234567890",
        );
        assert_eq!(local.chain_id, 31337);
    }

    #[test]
    fn test_validate_address_case_sensitivity() {
        // Lowercase should be valid
        let config_lower = EthereumConfig {
            rpc_url: "http://localhost".to_string(),
            verifier_address: "0xabcdef0123456789012345678901234567890123".to_string(),
            ..Default::default()
        };
        assert!(config_lower.validate().is_ok());
        
        // Uppercase should be valid
        let config_upper = EthereumConfig {
            rpc_url: "http://localhost".to_string(),
            verifier_address: "0xABCDEF0123456789012345678901234567890123".to_string(),
            ..Default::default()
        };
        assert!(config_upper.validate().is_ok());
    }
}
