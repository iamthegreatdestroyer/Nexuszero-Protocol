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
}
