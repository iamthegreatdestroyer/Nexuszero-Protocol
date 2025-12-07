//! Cosmos connector configuration.

use serde::{Deserialize, Serialize};

/// Configuration for connecting to a Cosmos SDK chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosmosConfig {
    /// Chain ID (e.g., "cosmoshub-4", "osmosis-1")
    pub chain_id: String,
    
    /// Tendermint RPC endpoint
    pub rpc_url: String,
    
    /// WebSocket endpoint for subscriptions
    pub ws_url: Option<String>,
    
    /// Address prefix (e.g., "cosmos", "osmo")
    pub address_prefix: String,
    
    /// Native denomination (e.g., "uatom", "uosmo")
    pub denom: String,
    
    /// Display denomination (e.g., "ATOM", "OSMO")
    pub display_denom: String,
    
    /// Decimal places for native token
    pub decimals: u8,
    
    /// NexusZero contract address (CosmWasm)
    pub verifier_contract: Option<String>,
    
    /// Bridge contract address
    pub bridge_contract: Option<String>,
    
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
}

impl CosmosConfig {
    /// Create configuration for Cosmos Hub mainnet.
    pub fn cosmoshub() -> Self {
        Self {
            chain_id: "cosmoshub-4".to_string(),
            rpc_url: "https://rpc.cosmos.network:443".to_string(),
            ws_url: Some("wss://rpc.cosmos.network:443/websocket".to_string()),
            address_prefix: "cosmos".to_string(),
            denom: "uatom".to_string(),
            display_denom: "ATOM".to_string(),
            decimals: 6,
            verifier_contract: None,
            bridge_contract: None,
            request_timeout_secs: 30,
        }
    }
    
    /// Create configuration for Osmosis mainnet.
    pub fn osmosis() -> Self {
        Self {
            chain_id: "osmosis-1".to_string(),
            rpc_url: "https://rpc.osmosis.zone:443".to_string(),
            ws_url: Some("wss://rpc.osmosis.zone:443/websocket".to_string()),
            address_prefix: "osmo".to_string(),
            denom: "uosmo".to_string(),
            display_denom: "OSMO".to_string(),
            decimals: 6,
            verifier_contract: None,
            bridge_contract: None,
            request_timeout_secs: 30,
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

    /// Set custom RPC URL
    pub fn with_rpc_url(mut self, url: impl Into<String>) -> Self {
        self.rpc_url = url.into();
        self
    }
}

impl Default for CosmosConfig {
    fn default() -> Self {
        Self::cosmoshub()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosmoshub_config() {
        let config = CosmosConfig::cosmoshub();
        assert_eq!(config.chain_id, "cosmoshub-4");
        assert_eq!(config.address_prefix, "cosmos");
        assert_eq!(config.denom, "uatom");
    }

    #[test]
    fn test_osmosis_config() {
        let config = CosmosConfig::osmosis();
        assert_eq!(config.chain_id, "osmosis-1");
        assert_eq!(config.address_prefix, "osmo");
    }

    // ===== HARDENING TESTS =====

    #[test]
    fn test_cosmoshub_config_full() {
        let config = CosmosConfig::cosmoshub();
        
        assert_eq!(config.chain_id, "cosmoshub-4");
        assert_eq!(config.rpc_url, "https://rpc.cosmos.network:443");
        assert!(config.ws_url.is_some());
        assert_eq!(config.address_prefix, "cosmos");
        assert_eq!(config.denom, "uatom");
        assert_eq!(config.display_denom, "ATOM");
        assert_eq!(config.decimals, 6);
        assert!(config.verifier_contract.is_none());
        assert!(config.bridge_contract.is_none());
        assert_eq!(config.request_timeout_secs, 30);
    }

    #[test]
    fn test_osmosis_config_full() {
        let config = CosmosConfig::osmosis();
        
        assert_eq!(config.chain_id, "osmosis-1");
        assert_eq!(config.rpc_url, "https://rpc.osmosis.zone:443");
        assert!(config.ws_url.is_some());
        assert_eq!(config.address_prefix, "osmo");
        assert_eq!(config.denom, "uosmo");
        assert_eq!(config.display_denom, "OSMO");
        assert_eq!(config.decimals, 6);
        assert_eq!(config.request_timeout_secs, 30);
    }

    #[test]
    fn test_default_is_cosmoshub() {
        let config = CosmosConfig::default();
        assert_eq!(config.chain_id, "cosmoshub-4");
        assert_eq!(config.address_prefix, "cosmos");
    }

    #[test]
    fn test_with_verifier_contract() {
        let config = CosmosConfig::cosmoshub()
            .with_verifier_contract("cosmos1abc123def456...");
        
        assert_eq!(config.verifier_contract, Some("cosmos1abc123def456...".to_string()));
    }

    #[test]
    fn test_with_bridge_contract() {
        let config = CosmosConfig::cosmoshub()
            .with_bridge_contract("cosmos1xyz789...");
        
        assert_eq!(config.bridge_contract, Some("cosmos1xyz789...".to_string()));
    }

    #[test]
    fn test_with_rpc_url() {
        let config = CosmosConfig::cosmoshub()
            .with_rpc_url("https://custom-rpc.example.com");
        
        assert_eq!(config.rpc_url, "https://custom-rpc.example.com");
    }

    #[test]
    fn test_builder_pattern_chaining() {
        let config = CosmosConfig::cosmoshub()
            .with_rpc_url("https://custom.rpc.com")
            .with_verifier_contract("cosmos1verifier...")
            .with_bridge_contract("cosmos1bridge...");
        
        assert_eq!(config.rpc_url, "https://custom.rpc.com");
        assert!(config.verifier_contract.is_some());
        assert!(config.bridge_contract.is_some());
    }

    #[test]
    fn test_config_clone() {
        let config = CosmosConfig::cosmoshub()
            .with_verifier_contract("cosmos1verifier...");
        let cloned = config.clone();
        
        assert_eq!(cloned.chain_id, config.chain_id);
        assert_eq!(cloned.verifier_contract, config.verifier_contract);
    }

    #[test]
    fn test_config_debug() {
        let config = CosmosConfig::cosmoshub();
        let debug = format!("{:?}", config);
        
        assert!(debug.contains("CosmosConfig"));
        assert!(debug.contains("cosmoshub-4"));
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = CosmosConfig::cosmoshub()
            .with_verifier_contract("cosmos1abc...")
            .with_bridge_contract("cosmos1xyz...");
        
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CosmosConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.chain_id, config.chain_id);
        assert_eq!(parsed.rpc_url, config.rpc_url);
        assert_eq!(parsed.verifier_contract, config.verifier_contract);
        assert_eq!(parsed.bridge_contract, config.bridge_contract);
    }

    #[test]
    fn test_decimals_cosmoshub() {
        let config = CosmosConfig::cosmoshub();
        assert_eq!(config.decimals, 6);
    }

    #[test]
    fn test_decimals_osmosis() {
        let config = CosmosConfig::osmosis();
        assert_eq!(config.decimals, 6);
    }

    #[test]
    fn test_ws_url_cosmoshub() {
        let config = CosmosConfig::cosmoshub();
        assert!(config.ws_url.is_some());
        assert!(config.ws_url.as_ref().unwrap().starts_with("wss://"));
    }

    #[test]
    fn test_ws_url_osmosis() {
        let config = CosmosConfig::osmosis();
        assert!(config.ws_url.is_some());
        assert!(config.ws_url.as_ref().unwrap().contains("websocket"));
    }

    #[test]
    fn test_with_string_types() {
        // Test with String
        let config1 = CosmosConfig::cosmoshub()
            .with_verifier_contract(String::from("cosmos1verifier"));
        
        // Test with &str
        let config2 = CosmosConfig::cosmoshub()
            .with_verifier_contract("cosmos1verifier");
        
        assert_eq!(config1.verifier_contract, config2.verifier_contract);
    }

    #[test]
    fn test_serde_with_optional_none() {
        let config = CosmosConfig::cosmoshub();
        
        let json = serde_json::to_string(&config).unwrap();
        let parsed: CosmosConfig = serde_json::from_str(&json).unwrap();
        
        assert!(parsed.verifier_contract.is_none());
        assert!(parsed.bridge_contract.is_none());
    }

    #[test]
    fn test_display_denom_cosmoshub() {
        let config = CosmosConfig::cosmoshub();
        assert_eq!(config.display_denom, "ATOM");
    }

    #[test]
    fn test_display_denom_osmosis() {
        let config = CosmosConfig::osmosis();
        assert_eq!(config.display_denom, "OSMO");
    }

    #[test]
    fn test_request_timeout() {
        let config = CosmosConfig::cosmoshub();
        assert_eq!(config.request_timeout_secs, 30);
    }
}
