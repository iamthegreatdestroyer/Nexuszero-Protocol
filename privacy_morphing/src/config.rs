//! Privacy Morphing Configuration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::types::PrivacyLevel;

/// Configuration for the Privacy Morphing Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphingConfig {
    /// Default privacy level for new transactions
    pub default_privacy_level: PrivacyLevel,

    /// Minimum anonymity set size
    pub min_anonymity_set_size: usize,

    /// Maximum time to wait for anonymity set expansion (seconds)
    pub max_anonymity_wait_seconds: u64,

    /// Enable automatic privacy morphing
    pub auto_morphing_enabled: bool,

    /// Morphing interval in seconds
    pub morphing_interval_seconds: u64,

    /// Chain-specific configurations
    pub chain_configs: HashMap<String, ChainConfig>,

    /// Compliance configurations
    pub compliance: ComplianceConfig,

    /// Performance tuning
    pub performance: PerformanceConfig,
}

/// Chain-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainConfig {
    /// Chain ID
    pub chain_id: String,
    /// Maximum privacy level on this chain
    pub max_privacy_level: PrivacyLevel,
    /// Block time in seconds
    pub block_time_seconds: u64,
    /// Confirmation requirements
    pub required_confirmations: u32,
}

/// Compliance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Enable compliance integration
    pub enabled: bool,
    /// Maximum privacy level for regulated entities
    pub regulated_max_level: PrivacyLevel,
    /// Jurisdiction-specific overrides
    pub jurisdiction_overrides: HashMap<String, PrivacyLevel>,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Cache size for anonymity sets
    pub anonymity_cache_size: usize,
    /// Proof generation parallelism
    pub proof_parallelism: usize,
}

impl Default for MorphingConfig {
    fn default() -> Self {
        let mut chain_configs = HashMap::new();
        
        chain_configs.insert("ethereum".to_string(), ChainConfig {
            chain_id: "1".to_string(),
            max_privacy_level: PrivacyLevel::new(10),
            block_time_seconds: 12,
            required_confirmations: 12,
        });

        chain_configs.insert("polygon".to_string(), ChainConfig {
            chain_id: "137".to_string(),
            max_privacy_level: PrivacyLevel::new(10),
            block_time_seconds: 2,
            required_confirmations: 32,
        });

        chain_configs.insert("bitcoin".to_string(), ChainConfig {
            chain_id: "bitcoin".to_string(),
            max_privacy_level: PrivacyLevel::new(8),
            block_time_seconds: 600,
            required_confirmations: 6,
        });

        Self {
            default_privacy_level: PrivacyLevel::default(),
            min_anonymity_set_size: 10,
            max_anonymity_wait_seconds: 300,
            auto_morphing_enabled: true,
            morphing_interval_seconds: 3600,
            chain_configs,
            compliance: ComplianceConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        let mut jurisdiction_overrides = HashMap::new();
        jurisdiction_overrides.insert("FATF".to_string(), PrivacyLevel::new(5));

        Self {
            enabled: true,
            regulated_max_level: PrivacyLevel::new(7),
            jurisdiction_overrides,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            worker_threads: 4,
            anonymity_cache_size: 10000,
            proof_parallelism: 2,
        }
    }
}

impl MorphingConfig {
    /// Load configuration from file
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Get chain configuration
    pub fn get_chain_config(&self, chain: &str) -> Option<&ChainConfig> {
        self.chain_configs.get(chain)
    }

    /// Get privacy ceiling for a jurisdiction
    pub fn get_jurisdiction_ceiling(&self, jurisdiction: &str) -> PrivacyLevel {
        self.compliance.jurisdiction_overrides
            .get(jurisdiction)
            .copied()
            .unwrap_or(self.compliance.regulated_max_level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = MorphingConfig::default();
        assert_eq!(config.default_privacy_level.value(), 5);
        assert!(config.auto_morphing_enabled);
    }

    #[test]
    fn test_chain_config() {
        let config = MorphingConfig::default();
        let eth = config.get_chain_config("ethereum").unwrap();
        assert_eq!(eth.chain_id, "1");
        assert_eq!(eth.max_privacy_level.value(), 10);
    }

    #[test]
    fn test_jurisdiction_ceiling() {
        let config = MorphingConfig::default();
        assert_eq!(config.get_jurisdiction_ceiling("FATF").value(), 5);
        assert_eq!(config.get_jurisdiction_ceiling("unknown").value(), 7);
    }
}
