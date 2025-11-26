//! Configuration for Compliance Service

use serde::Deserialize;
use std::collections::HashSet;

/// Main configuration structure
#[derive(Debug, Clone, Deserialize)]
pub struct ComplianceConfig {
    /// Service port
    #[serde(default = "default_port")]
    pub port: u16,
    
    /// Database URL
    pub database_url: String,
    
    /// Redis URL for caching
    pub redis_url: String,
    
    /// Enabled jurisdictions
    #[serde(default = "default_jurisdictions")]
    pub enabled_jurisdictions: HashSet<String>,
    
    /// Default risk threshold (0.0 - 1.0)
    #[serde(default = "default_risk_threshold")]
    pub default_risk_threshold: f64,
    
    /// SAR auto-generation threshold
    #[serde(default = "default_sar_threshold")]
    pub sar_threshold: f64,
    
    /// Travel Rule threshold (USD equivalent)
    #[serde(default = "default_travel_rule_threshold")]
    pub travel_rule_threshold: u64,
    
    /// External AML service URL
    pub aml_service_url: Option<String>,
    
    /// External KYC service URL
    pub kyc_service_url: Option<String>,
    
    /// Watchlist update interval (seconds)
    #[serde(default = "default_watchlist_interval")]
    pub watchlist_update_interval: u64,
    
    /// Report generation workers
    #[serde(default = "default_report_workers")]
    pub report_workers: usize,
    
    /// Enable real-time monitoring
    #[serde(default = "default_true")]
    pub realtime_monitoring: bool,
    
    /// Maximum batch size for compliance checks
    #[serde(default = "default_batch_size")]
    pub max_batch_size: usize,
    
    /// Audit log retention days
    #[serde(default = "default_retention_days")]
    pub audit_retention_days: u32,
}

fn default_port() -> u16 {
    8083
}

fn default_jurisdictions() -> HashSet<String> {
    ["FATF", "EU", "US", "APAC"].iter().map(|s| s.to_string()).collect()
}

fn default_risk_threshold() -> f64 {
    0.7
}

fn default_sar_threshold() -> f64 {
    0.85
}

fn default_travel_rule_threshold() -> u64 {
    3000 // USD equivalent
}

fn default_watchlist_interval() -> u64 {
    3600 // 1 hour
}

fn default_report_workers() -> usize {
    4
}

fn default_true() -> bool {
    true
}

fn default_batch_size() -> usize {
    100
}

fn default_retention_days() -> u32 {
    2555 // ~7 years
}

impl ComplianceConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> anyhow::Result<Self> {
        let config = config::Config::builder()
            .add_source(config::Environment::with_prefix("COMPLIANCE"))
            .set_default("port", 8083)?
            .set_default("default_risk_threshold", 0.7)?
            .set_default("sar_threshold", 0.85)?
            .set_default("travel_rule_threshold", 3000i64)?
            .set_default("watchlist_update_interval", 3600i64)?
            .set_default("report_workers", 4i64)?
            .set_default("realtime_monitoring", true)?
            .set_default("max_batch_size", 100i64)?
            .set_default("audit_retention_days", 2555i64)?
            .build()?;
        
        Ok(config.try_deserialize()?)
    }
    
    /// Check if jurisdiction is enabled
    pub fn is_jurisdiction_enabled(&self, code: &str) -> bool {
        self.enabled_jurisdictions.contains(code)
    }
}

/// Jurisdiction-specific configuration
#[derive(Debug, Clone)]
pub struct JurisdictionConfig {
    pub code: String,
    pub name: String,
    pub reporting_threshold: u64,
    pub kyc_required: bool,
    pub travel_rule_enabled: bool,
    pub sar_required: bool,
    pub data_retention_years: u32,
}

impl JurisdictionConfig {
    /// Get FATF jurisdiction config
    pub fn fatf() -> Self {
        Self {
            code: "FATF".to_string(),
            name: "Financial Action Task Force".to_string(),
            reporting_threshold: 10000,
            kyc_required: true,
            travel_rule_enabled: true,
            sar_required: true,
            data_retention_years: 5,
        }
    }
    
    /// Get EU jurisdiction config (AMLD6)
    pub fn eu() -> Self {
        Self {
            code: "EU".to_string(),
            name: "European Union (AMLD6)".to_string(),
            reporting_threshold: 10000, // EUR
            kyc_required: true,
            travel_rule_enabled: true,
            sar_required: true,
            data_retention_years: 5,
        }
    }
    
    /// Get US jurisdiction config (FinCEN)
    pub fn us() -> Self {
        Self {
            code: "US".to_string(),
            name: "United States (FinCEN)".to_string(),
            reporting_threshold: 10000, // USD
            kyc_required: true,
            travel_rule_enabled: true,
            sar_required: true,
            data_retention_years: 5,
        }
    }
    
    /// Get APAC jurisdiction config
    pub fn apac() -> Self {
        Self {
            code: "APAC".to_string(),
            name: "Asia-Pacific".to_string(),
            reporting_threshold: 15000, // USD equivalent
            kyc_required: true,
            travel_rule_enabled: true,
            sar_required: true,
            data_retention_years: 7,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_jurisdictions() {
        let jurisdictions = default_jurisdictions();
        assert!(jurisdictions.contains("FATF"));
        assert!(jurisdictions.contains("EU"));
        assert!(jurisdictions.contains("US"));
        assert!(jurisdictions.contains("APAC"));
    }

    #[test]
    fn test_jurisdiction_configs() {
        let fatf = JurisdictionConfig::fatf();
        assert_eq!(fatf.code, "FATF");
        assert!(fatf.travel_rule_enabled);
        
        let eu = JurisdictionConfig::eu();
        assert_eq!(eu.data_retention_years, 5);
    }
}
