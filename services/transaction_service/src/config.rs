//! Configuration management for Transaction Service

use serde::Deserialize;
use std::env;

/// Transaction Service configuration
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Server host
    pub host: String,

    /// Server port
    pub port: u16,

    /// Database configuration
    pub database: DatabaseConfig,

    /// Redis configuration
    pub redis: RedisConfig,

    /// Privacy Service URL
    pub privacy_service_url: String,

    /// Compliance Service URL
    pub compliance_service_url: String,

    /// Proof generation timeout in seconds
    pub proof_timeout_secs: u64,

    /// Maximum transactions per batch
    pub max_batch_size: usize,

    /// Transaction retention days
    pub retention_days: u32,
}

/// Database configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    /// Database connection URL
    pub url: String,

    /// Maximum connections in pool
    pub max_connections: u32,

    /// Minimum connections in pool
    pub min_connections: u32,

    /// Connection timeout in seconds
    pub connect_timeout_secs: u64,
}

/// Redis configuration
#[derive(Debug, Clone, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    pub url: String,

    /// Cache TTL in seconds
    pub cache_ttl_secs: u64,
}

impl Config {
    /// Load configuration from environment variables
    pub fn load() -> anyhow::Result<Self> {
        Ok(Self {
            host: env::var("TRANSACTION_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("TRANSACTION_PORT")
                .unwrap_or_else(|_| "8081".to_string())
                .parse()?,
            database: DatabaseConfig {
                url: env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "postgres://localhost/nexuszero".to_string()),
                max_connections: env::var("DATABASE_MAX_CONNECTIONS")
                    .unwrap_or_else(|_| "20".to_string())
                    .parse()?,
                min_connections: env::var("DATABASE_MIN_CONNECTIONS")
                    .unwrap_or_else(|_| "5".to_string())
                    .parse()?,
                connect_timeout_secs: env::var("DATABASE_CONNECT_TIMEOUT")
                    .unwrap_or_else(|_| "30".to_string())
                    .parse()?,
            },
            redis: RedisConfig {
                url: env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string()),
                cache_ttl_secs: env::var("REDIS_CACHE_TTL")
                    .unwrap_or_else(|_| "3600".to_string())
                    .parse()?,
            },
            privacy_service_url: env::var("PRIVACY_SERVICE_URL")
                .unwrap_or_else(|_| "http://localhost:8082".to_string()),
            compliance_service_url: env::var("COMPLIANCE_SERVICE_URL")
                .unwrap_or_else(|_| "http://localhost:8083".to_string()),
            proof_timeout_secs: env::var("PROOF_TIMEOUT_SECS")
                .unwrap_or_else(|_| "120".to_string())
                .parse()?,
            max_batch_size: env::var("MAX_BATCH_SIZE")
                .unwrap_or_else(|_| "100".to_string())
                .parse()?,
            retention_days: env::var("RETENTION_DAYS")
                .unwrap_or_else(|_| "365".to_string())
                .parse()?,
        })
    }
}
