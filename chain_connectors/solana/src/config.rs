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

    // ===== HARDENING TESTS =====

    #[test]
    fn test_cluster_mainnet() {
        let cluster = SolanaCluster::mainnet();
        assert_eq!(cluster.as_str(), "mainnet-beta");
    }

    #[test]
    fn test_cluster_testnet() {
        let cluster = SolanaCluster::testnet();
        assert_eq!(cluster.as_str(), "testnet");
    }

    #[test]
    fn test_cluster_devnet() {
        let cluster = SolanaCluster::devnet();
        assert_eq!(cluster.as_str(), "devnet");
    }

    #[test]
    fn test_cluster_default() {
        let cluster = SolanaCluster::default();
        assert_eq!(cluster.as_str(), "devnet");
    }

    #[test]
    fn test_commitment_processed() {
        let commitment = CommitmentLevel::processed();
        assert_eq!(commitment.as_str(), "processed");
    }

    #[test]
    fn test_commitment_confirmed() {
        let commitment = CommitmentLevel::confirmed();
        assert_eq!(commitment.as_str(), "confirmed");
    }

    #[test]
    fn test_commitment_finalized() {
        let commitment = CommitmentLevel::finalized();
        assert_eq!(commitment.as_str(), "finalized");
    }

    #[test]
    fn test_commitment_default() {
        let commitment = CommitmentLevel::default();
        assert_eq!(commitment.as_str(), "confirmed");
    }

    #[test]
    fn test_testnet_config() {
        let config = SolanaConfig::testnet();
        
        assert_eq!(config.cluster.as_str(), "testnet");
        assert!(config.rpc_url.contains("testnet"));
        assert!(config.ws_url.as_ref().unwrap().contains("testnet"));
        assert_eq!(config.commitment.as_str(), "confirmed");
        assert!(config.priority_fee_micro_lamports.is_none());
        assert_eq!(config.timeout_seconds, 30);
    }

    #[test]
    fn test_mainnet_config_values() {
        let config = SolanaConfig::mainnet();
        
        assert_eq!(config.cluster.as_str(), "mainnet-beta");
        assert_eq!(config.rpc_url, "https://api.mainnet-beta.solana.com");
        assert_eq!(config.ws_url, Some("wss://api.mainnet-beta.solana.com".to_string()));
        assert_eq!(config.commitment.as_str(), "finalized");
        assert_eq!(config.priority_fee_micro_lamports, Some(10_000));
        assert_eq!(config.timeout_seconds, 60);
    }

    #[test]
    fn test_devnet_config_values() {
        let config = SolanaConfig::devnet();
        
        assert_eq!(config.cluster.as_str(), "devnet");
        assert_eq!(config.rpc_url, "https://api.devnet.solana.com");
        assert!(config.priority_fee_micro_lamports.is_none());
        assert_eq!(config.timeout_seconds, 30);
    }

    #[test]
    fn test_from_rpc_url() {
        let config = SolanaConfig::from_rpc_url(
            "http://localhost:8899",
            "ProgramId11111111111111111111111111111111111"
        );
        
        assert_eq!(config.rpc_url, "http://localhost:8899");
        assert_eq!(config.verifier_program_id, "ProgramId11111111111111111111111111111111111");
        assert!(config.ws_url.is_none());
    }

    #[test]
    fn test_with_verifier_program() {
        let config = SolanaConfig::devnet()
            .with_verifier_program("VerifierProgram11111111111111111111111111111");
        
        assert_eq!(config.verifier_program_id, "VerifierProgram11111111111111111111111111111");
    }

    #[test]
    fn test_with_bridge_program() {
        let config = SolanaConfig::devnet()
            .with_bridge_program("BridgeProgram111111111111111111111111111111");
        
        assert_eq!(
            config.bridge_program_id,
            Some("BridgeProgram111111111111111111111111111111".to_string())
        );
    }

    #[test]
    fn test_builder_pattern() {
        let config = SolanaConfig::devnet()
            .with_verifier_program("VerifierProgram")
            .with_bridge_program("BridgeProgram");
        
        assert_eq!(config.verifier_program_id, "VerifierProgram");
        assert_eq!(config.bridge_program_id, Some("BridgeProgram".to_string()));
    }

    #[test]
    fn test_default_config() {
        let config = SolanaConfig::default();
        
        assert_eq!(config.cluster.as_str(), "devnet");
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = SolanaConfig::mainnet()
            .with_verifier_program("TestVerifier")
            .with_bridge_program("TestBridge");
        
        let json = serde_json::to_string(&config).unwrap();
        let parsed: SolanaConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.rpc_url, config.rpc_url);
        assert_eq!(parsed.verifier_program_id, config.verifier_program_id);
        assert_eq!(parsed.bridge_program_id, config.bridge_program_id);
    }

    #[test]
    fn test_config_clone() {
        let config = SolanaConfig::mainnet();
        let cloned = config.clone();
        
        assert_eq!(cloned.rpc_url, config.rpc_url);
        assert_eq!(cloned.cluster.as_str(), config.cluster.as_str());
    }

    #[test]
    fn test_cluster_clone() {
        let cluster = SolanaCluster::mainnet();
        let cloned = cluster.clone();
        
        assert_eq!(cloned.as_str(), cluster.as_str());
    }

    #[test]
    fn test_commitment_clone() {
        let commitment = CommitmentLevel::finalized();
        let cloned = commitment.clone();
        
        assert_eq!(cloned.as_str(), commitment.as_str());
    }

    #[test]
    fn test_validation_error_message() {
        let mut invalid = SolanaConfig::devnet();
        invalid.rpc_url = String::new();
        
        let result = invalid.validate();
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("RPC URL"));
    }

    #[test]
    fn test_cluster_serde() {
        let cluster = SolanaCluster::mainnet();
        let json = serde_json::to_string(&cluster).unwrap();
        let parsed: SolanaCluster = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.as_str(), cluster.as_str());
    }

    #[test]
    fn test_commitment_serde() {
        let commitment = CommitmentLevel::finalized();
        let json = serde_json::to_string(&commitment).unwrap();
        let parsed: CommitmentLevel = serde_json::from_str(&json).unwrap();
        
        assert_eq!(parsed.as_str(), commitment.as_str());
    }
}
