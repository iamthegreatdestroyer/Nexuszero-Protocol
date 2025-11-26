//! Privacy Service configuration

use serde::Deserialize;
use std::env;

/// Privacy Service configuration
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

    /// Proof generation configuration
    pub proof: ProofConfig,

    /// Privacy engine configuration
    pub privacy: PrivacyEngineConfig,
}

/// Database configuration
#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub connect_timeout_secs: u64,
}

/// Redis configuration
#[derive(Debug, Clone, Deserialize)]
pub struct RedisConfig {
    pub url: String,
    pub cache_ttl_secs: u64,
}

/// Proof generation configuration
#[derive(Debug, Clone, Deserialize)]
pub struct ProofConfig {
    /// Maximum concurrent proof generations
    pub max_concurrent: usize,

    /// Proof timeout in seconds
    pub timeout_secs: u64,

    /// Maximum batch size for batch proof generation
    pub max_batch_size: usize,

    /// Use GPU acceleration if available
    pub use_gpu: bool,

    /// Circuit parameters cache directory
    pub params_cache_dir: String,
}

/// Privacy engine configuration
#[derive(Debug, Clone, Deserialize)]
pub struct PrivacyEngineConfig {
    /// Default privacy level
    pub default_level: i16,

    /// Allow privacy level downgrade
    pub allow_downgrade: bool,

    /// Require force flag for downgrade
    pub require_force_for_downgrade: bool,

    /// Enable privacy recommendations
    pub enable_recommendations: bool,
}

impl Config {
    /// Load configuration from environment variables
    pub fn load() -> anyhow::Result<Self> {
        Ok(Self {
            host: env::var("PRIVACY_HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
            port: env::var("PRIVACY_PORT")
                .unwrap_or_else(|_| "8082".to_string())
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
            proof: ProofConfig {
                max_concurrent: env::var("PROOF_MAX_CONCURRENT")
                    .unwrap_or_else(|_| "4".to_string())
                    .parse()?,
                timeout_secs: env::var("PROOF_TIMEOUT_SECS")
                    .unwrap_or_else(|_| "120".to_string())
                    .parse()?,
                max_batch_size: env::var("PROOF_MAX_BATCH_SIZE")
                    .unwrap_or_else(|_| "50".to_string())
                    .parse()?,
                use_gpu: env::var("PROOF_USE_GPU")
                    .unwrap_or_else(|_| "false".to_string())
                    .parse()?,
                params_cache_dir: env::var("PROOF_PARAMS_CACHE_DIR")
                    .unwrap_or_else(|_| "./params".to_string()),
            },
            privacy: PrivacyEngineConfig {
                default_level: env::var("PRIVACY_DEFAULT_LEVEL")
                    .unwrap_or_else(|_| "4".to_string())
                    .parse()?,
                allow_downgrade: env::var("PRIVACY_ALLOW_DOWNGRADE")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()?,
                require_force_for_downgrade: env::var("PRIVACY_REQUIRE_FORCE")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()?,
                enable_recommendations: env::var("PRIVACY_ENABLE_RECOMMENDATIONS")
                    .unwrap_or_else(|_| "true".to_string())
                    .parse()?,
            },
        })
    }
}
