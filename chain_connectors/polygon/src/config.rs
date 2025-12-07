//! Polygon connector configuration.

use serde::{Deserialize, Serialize};

/// Configuration for connecting to Polygon network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolygonConfig {
    /// RPC endpoint URL
    pub rpc_url: String,
    
    /// WebSocket endpoint for subscriptions
    pub ws_url: Option<String>,
    
    /// Chain ID (137 for mainnet, 80001 for Mumbai)
    pub chain_id: u64,
    
    /// NexusZero verifier contract address
    pub verifier_contract: Option<String>,
    
    /// NexusZero bridge contract address
    pub bridge_contract: Option<String>,
    
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    
    /// Maximum retries for failed requests
    pub max_retries: u32,
}

impl PolygonConfig {
    /// Create configuration for Polygon mainnet.
    pub fn mainnet() -> Self {
        Self {
            rpc_url: "https://polygon-rpc.com".to_string(),
            ws_url: Some("wss://polygon-rpc.com".to_string()),
            chain_id: 137,
            verifier_contract: None,
            bridge_contract: None,
            request_timeout_secs: 30,
            max_retries: 3,
        }
    }
    
    /// Create configuration for Mumbai testnet.
    pub fn mumbai() -> Self {
        Self {
            rpc_url: "https://rpc-mumbai.maticvigil.com".to_string(),
            ws_url: Some("wss://rpc-mumbai.maticvigil.com/ws".to_string()),
            chain_id: 80001,
            verifier_contract: None,
            bridge_contract: None,
            request_timeout_secs: 30,
            max_retries: 3,
        }
    }

    /// Set verifier contract address
    pub fn with_verifier_contract(mut self, address: impl Into<String>) -> Self {
        self.verifier_contract = Some(address.into());
        self
    }

    /// Set bridge contract address
    pub fn with_bridge_contract(mut self, address: impl Into<String>) -> Self {
        self.bridge_contract = Some(address.into());
        self
    }

    /// Set RPC URL
    pub fn with_rpc_url(mut self, url: impl Into<String>) -> Self {
        self.rpc_url = url.into();
        self
    }
}

impl Default for PolygonConfig {
    fn default() -> Self {
        Self::mainnet()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mainnet_config() {
        let config = PolygonConfig::mainnet();
        assert_eq!(config.chain_id, 137);
    }
    
    #[test]
    fn test_mumbai_config() {
        let config = PolygonConfig::mumbai();
        assert_eq!(config.chain_id, 80001);
    }

    // ===== HARDENING TESTS =====

    #[test]
    fn test_mainnet_config_values() {
        let config = PolygonConfig::mainnet();
        
        assert_eq!(config.rpc_url, "https://polygon-rpc.com");
        assert_eq!(config.ws_url, Some("wss://polygon-rpc.com".to_string()));
        assert_eq!(config.chain_id, 137);
        assert!(config.verifier_contract.is_none());
        assert!(config.bridge_contract.is_none());
        assert_eq!(config.request_timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_mumbai_config_values() {
        let config = PolygonConfig::mumbai();
        
        assert_eq!(config.rpc_url, "https://rpc-mumbai.maticvigil.com");
        assert!(config.ws_url.as_ref().unwrap().contains("mumbai"));
        assert_eq!(config.chain_id, 80001);
        assert_eq!(config.request_timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_default_is_mainnet() {
        let config = PolygonConfig::default();
        assert_eq!(config.chain_id, 137);
    }

    #[test]
    fn test_with_verifier_contract() {
        let config = PolygonConfig::mainnet()
            .with_verifier_contract("0x1234567890123456789012345678901234567890");
        
        assert_eq!(
            config.verifier_contract,
            Some("0x1234567890123456789012345678901234567890".to_string())
        );
    }

    #[test]
    fn test_with_bridge_contract() {
        let config = PolygonConfig::mainnet()
            .with_bridge_contract("0xabcdef0123456789012345678901234567890123");
        
        assert_eq!(
            config.bridge_contract,
            Some("0xabcdef0123456789012345678901234567890123".to_string())
        );
    }

    #[test]
    fn test_with_rpc_url() {
        let config = PolygonConfig::mainnet()
            .with_rpc_url("https://custom-rpc.example.com");
        
        assert_eq!(config.rpc_url, "https://custom-rpc.example.com");
    }

    #[test]
    fn test_builder_pattern_chaining() {
        let config = PolygonConfig::mainnet()
            .with_rpc_url("https://custom.rpc.com")
            .with_verifier_contract("0x1111111111111111111111111111111111111111")
            .with_bridge_contract("0x2222222222222222222222222222222222222222");
        
        assert_eq!(config.rpc_url, "https://custom.rpc.com");
        assert!(config.verifier_contract.is_some());
        assert!(config.bridge_contract.is_some());
    }

    #[test]
    fn test_config_clone() {
        let config = PolygonConfig::mainnet()
            .with_verifier_contract("0x1234567890123456789012345678901234567890");
        let cloned = config.clone();
        
        assert_eq!(cloned.chain_id, config.chain_id);
        assert_eq!(cloned.verifier_contract, config.verifier_contract);
    }

    #[test]
    fn test_config_debug() {
        let config = PolygonConfig::mainnet();
        let debug = format!("{:?}", config);
        
        assert!(debug.contains("PolygonConfig"));
        assert!(debug.contains("chain_id: 137"));
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = PolygonConfig::mainnet()
            .with_verifier_contract("0x1234567890123456789012345678901234567890")
            .with_bridge_contract("0xabcdef0123456789012345678901234567890123");
        
        let json = serde_json::to_string(&config).unwrap();
        let parsed: PolygonConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.chain_id, config.chain_id);
        assert_eq!(parsed.rpc_url, config.rpc_url);
        assert_eq!(parsed.verifier_contract, config.verifier_contract);
        assert_eq!(parsed.bridge_contract, config.bridge_contract);
    }

    #[test]
    fn test_chain_id_mainnet() {
        let config = PolygonConfig::mainnet();
        assert_eq!(config.chain_id, 137);
    }

    #[test]
    fn test_chain_id_mumbai() {
        let config = PolygonConfig::mumbai();
        assert_eq!(config.chain_id, 80001);
    }

    #[test]
    fn test_timeout_and_retries() {
        let config = PolygonConfig::mainnet();
        
        assert_eq!(config.request_timeout_secs, 30);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_ws_url_mainnet() {
        let config = PolygonConfig::mainnet();
        assert!(config.ws_url.is_some());
        assert!(config.ws_url.as_ref().unwrap().starts_with("wss://"));
    }

    #[test]
    fn test_ws_url_mumbai() {
        let config = PolygonConfig::mumbai();
        assert!(config.ws_url.is_some());
        assert!(config.ws_url.as_ref().unwrap().starts_with("wss://"));
    }

    #[test]
    fn test_with_string_types() {
        // Test with String
        let config1 = PolygonConfig::mainnet()
            .with_verifier_contract(String::from("0x1234567890123456789012345678901234567890"));
        
        // Test with &str
        let config2 = PolygonConfig::mainnet()
            .with_verifier_contract("0x1234567890123456789012345678901234567890");
        
        assert_eq!(config1.verifier_contract, config2.verifier_contract);
    }

    #[test]
    fn test_serde_with_optional_none() {
        let config = PolygonConfig::mainnet();
        
        let json = serde_json::to_string(&config).unwrap();
        let parsed: PolygonConfig = serde_json::from_str(&json).unwrap();
        
        assert!(parsed.verifier_contract.is_none());
        assert!(parsed.bridge_contract.is_none());
    }
}
