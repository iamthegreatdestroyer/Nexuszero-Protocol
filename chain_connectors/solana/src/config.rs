//! Solana connector configuration.

use serde::{Deserialize, Serialize};

/// Solana cluster configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaCluster(String);

impl SolanaCluster {
    /// Mainnet-beta cluster.
    pub fn mainnet() -> Self {
        Self("mainnet-beta".to_string())
    }
    
    /// Testnet cluster.
    pub fn testnet() -> Self {
        Self("testnet".to_string())
    }
    
    /// Devnet cluster.
    pub fn devnet() -> Self {
        Self("devnet".to_string())
    }
    
    /// Get cluster name as string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SolanaCluster {
    fn default() -> Self {
        Self::devnet()
    }
}

/// Commitment level for Solana queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentLevel(String);

impl CommitmentLevel {
    pub fn processed() -> Self {
        Self("processed".to_string())
    }
    
    pub fn confirmed() -> Self {
        Self("confirmed".to_string())
    }
    
    pub fn finalized() -> Self {
        Self("finalized".to_string())
    }
    
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for CommitmentLevel {
    fn default() -> Self {
        Self::confirmed()
    }
}

/// Configuration for Solana connector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolanaConfig {
    /// Cluster to connect to
    pub cluster: SolanaCluster,
    
    /// RPC endpoint URL
    pub rpc_url: String,
    
    /// WebSocket endpoint URL
    pub ws_url: Option<String>,
    
    /// NexusZero verifier program ID
    pub verifier_program_id: String,
    
    /// Bridge program ID (optional)
    pub bridge_program_id: Option<String>,
    
    /// Commitment level for queries
    pub commitment: CommitmentLevel,
    
    /// Priority fee in micro-lamports
    pub priority_fee_micro_lamports: Option<u64>,
    
    /// Request timeout in seconds
    pub timeout_seconds: u64,
}

impl SolanaConfig {
    /// Create configuration for mainnet-beta.
    pub fn mainnet() -> Self {
        Self {
            cluster: SolanaCluster::mainnet(),
            rpc_url: "https://api.mainnet-beta.solana.com".to_string(),
            ws_url: Some("wss://api.mainnet-beta.solana.com".to_string()),
            verifier_program_id: String::new(),
            bridge_program_id: None,
            commitment: CommitmentLevel::finalized(),
            priority_fee_micro_lamports: Some(10_000),
            timeout_seconds: 60,
        }
    }
    
    /// Create configuration for devnet.
    pub fn devnet() -> Self {
        Self {
            cluster: SolanaCluster::devnet(),
            rpc_url: "https://api.devnet.solana.com".to_string(),
            ws_url: Some("wss://api.devnet.solana.com".to_string()),
            verifier_program_id: String::new(),
            bridge_program_id: None,
            commitment: CommitmentLevel::confirmed(),
            priority_fee_micro_lamports: None,
            timeout_seconds: 30,
        }
    }
    
    /// Create configuration for testnet.
    pub fn testnet() -> Self {
        Self {
            cluster: SolanaCluster::testnet(),
            rpc_url: "https://api.testnet.solana.com".to_string(),
            ws_url: Some("wss://api.testnet.solana.com".to_string()),
            verifier_program_id: String::new(),
            bridge_program_id: None,
            commitment: CommitmentLevel::confirmed(),
            priority_fee_micro_lamports: None,
            timeout_seconds: 30,
        }
    }
    
    /// Create configuration from RPC URL.
    pub fn from_rpc_url(rpc_url: &str, verifier_program_id: &str) -> Self {
        Self {
            cluster: SolanaCluster::default(),
            rpc_url: rpc_url.to_string(),
            ws_url: None,
            verifier_program_id: verifier_program_id.to_string(),
            bridge_program_id: None,
            commitment: CommitmentLevel::confirmed(),
            priority_fee_micro_lamports: None,
            timeout_seconds: 30,
        }
    }
    
    /// Set the verifier program ID.
    pub fn with_verifier_program(mut self, program_id: &str) -> Self {
        self.verifier_program_id = program_id.to_string();
        self
    }
    
    /// Set the bridge program ID.
    pub fn with_bridge_program(mut self, program_id: &str) -> Self {
        self.bridge_program_id = Some(program_id.to_string());
        self
    }
    
    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.rpc_url.is_empty() {
            return Err("RPC URL is required".to_string());
        }
        
        Ok(())
    }
}

impl Default for SolanaConfig {
    fn default() -> Self {
        Self::devnet()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_mainnet_config() {
        let config = SolanaConfig::mainnet();
        assert_eq!(config.cluster.as_str(), "mainnet-beta");
        assert!(config.rpc_url.contains("mainnet"));
    }
    
    #[test]
    fn test_devnet_config() {
        let config = SolanaConfig::devnet();
        assert_eq!(config.cluster.as_str(), "devnet");
    }
    
    #[test]
    fn test_validation() {
        let valid = SolanaConfig::devnet();
        assert!(valid.validate().is_ok());
        
        let mut invalid = SolanaConfig::devnet();
        invalid.rpc_url = String::new();
        assert!(invalid.validate().is_err());
    }
}
