//! CLI Configuration Management

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::error::{CliError, CliResult};

/// CLI Configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Config {
    /// API endpoint URL
    pub api_url: Option<String>,

    /// Default network (mainnet, testnet, devnet)
    pub network: Option<String>,

    /// Default wallet path
    pub wallet_path: Option<PathBuf>,

    /// Default gas settings
    pub gas: Option<GasConfig>,

    /// Proof generation settings
    pub proof: Option<ProofConfig>,

    /// Bridge settings
    pub bridge: Option<BridgeConfig>,

    /// Output preferences
    pub output: Option<OutputConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GasConfig {
    /// Gas limit multiplier (1.0 = estimated)
    pub limit_multiplier: f64,
    /// Max gas price in gwei
    pub max_price_gwei: u64,
    /// Priority fee in gwei
    pub priority_fee_gwei: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofConfig {
    /// Preferred prover (local, network, auto)
    pub prover: String,
    /// Proof generation timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Default source chain
    pub default_source_chain: Option<String>,
    /// Default destination chain
    pub default_dest_chain: Option<String>,
    /// Confirmation requirements per chain
    pub confirmations: std::collections::HashMap<String, u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Default output format (text, json, yaml)
    pub format: String,
    /// Enable colored output
    pub colored: bool,
    /// Show progress indicators
    pub progress: bool,
}

impl Config {
    /// Load configuration from file or defaults
    pub fn load(path: Option<&str>) -> CliResult<Self> {
        let config_path = match path {
            Some(p) => PathBuf::from(p),
            None => Self::default_config_path()?,
        };

        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: Config = toml::from_str(&content)?;
            Ok(config)
        } else {
            Ok(Config::default())
        }
    }

    /// Save configuration to file
    pub fn save(&self, path: Option<&str>) -> CliResult<()> {
        let config_path = match path {
            Some(p) => PathBuf::from(p),
            None => Self::default_config_path()?,
        };

        // Create parent directories if needed
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)?;
        std::fs::write(&config_path, content)?;
        Ok(())
    }

    /// Get default configuration file path
    pub fn default_config_path() -> CliResult<PathBuf> {
        let config_dir = dirs::config_dir()
            .ok_or_else(|| CliError::Config("Could not find config directory".to_string()))?;
        Ok(config_dir.join("nexuszero").join("config.toml"))
    }

    /// Get default data directory
    pub fn default_data_dir() -> CliResult<PathBuf> {
        let data_dir = dirs::data_dir()
            .ok_or_else(|| CliError::Config("Could not find data directory".to_string()))?;
        Ok(data_dir.join("nexuszero"))
    }

    /// Get API URL with fallback
    pub fn get_api_url(&self) -> String {
        self.api_url.clone().unwrap_or_else(|| "http://localhost:8080".to_string())
    }

    /// Get network with fallback
    pub fn get_network(&self) -> String {
        self.network.clone().unwrap_or_else(|| "mainnet".to_string())
    }
}

impl Default for GasConfig {
    fn default() -> Self {
        Self {
            limit_multiplier: 1.2,
            max_price_gwei: 500,
            priority_fee_gwei: 2,
        }
    }
}

impl Default for ProofConfig {
    fn default() -> Self {
        Self {
            prover: "auto".to_string(),
            timeout_secs: 300,
            max_retries: 3,
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: "text".to_string(),
            colored: true,
            progress: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_default() {
        let config = Config::default();
        assert!(config.api_url.is_none());
        assert_eq!(config.get_network(), "mainnet");
    }

    #[test]
    fn test_config_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("config.toml");
        let path_str = path.to_str().unwrap();

        let config = Config {
            api_url: Some("http://test:8080".to_string()),
            network: Some("testnet".to_string()),
            ..Default::default()
        };

        config.save(Some(path_str)).unwrap();
        let loaded = Config::load(Some(path_str)).unwrap();

        assert_eq!(loaded.api_url, config.api_url);
        assert_eq!(loaded.network, config.network);
    }
}
