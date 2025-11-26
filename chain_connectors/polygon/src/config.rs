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
}
