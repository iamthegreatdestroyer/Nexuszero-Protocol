//! Configuration management for API Gateway
//!
//! Supports loading from:
//! - Environment variables
//! - Config files (TOML, YAML)
//! - Defaults for development

use serde::Deserialize;
use std::env;

/// Main configuration structure
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// Server host
    #[serde(default = "default_host")]
    pub host: String,

    /// Server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// Metrics server port
    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    /// Environment (development, staging, production)
    #[serde(default = "default_environment")]
    pub environment: String,

    /// Database configuration
    #[serde(default)]
    pub database: DatabaseConfig,

    /// Redis configuration
    #[serde(default)]
    pub redis: RedisConfig,

    /// JWT configuration
    #[serde(default)]
    pub jwt: JwtConfig,

    /// Rate limiting configuration
    #[serde(default)]
    pub rate_limit: RateLimitConfig,

    /// Service URLs
    #[serde(default)]
    pub services: ServicesConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatabaseConfig {
    /// PostgreSQL connection URL
    #[serde(default = "default_database_url")]
    pub url: String,

    /// Maximum connections in pool
    #[serde(default = "default_max_connections")]
    pub max_connections: u32,

    /// Minimum connections in pool
    #[serde(default = "default_min_connections")]
    pub min_connections: u32,

    /// Connection timeout in seconds
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RedisConfig {
    /// Redis connection URL
    #[serde(default = "default_redis_url")]
    pub url: String,

    /// Connection pool size
    #[serde(default = "default_redis_pool_size")]
    pub pool_size: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct JwtConfig {
    /// JWT secret key
    #[serde(default = "default_jwt_secret")]
    pub secret: String,

    /// Access token expiry in seconds
    #[serde(default = "default_access_token_expiry")]
    pub access_token_expiry_secs: u64,

    /// Refresh token expiry in seconds
    #[serde(default = "default_refresh_token_expiry")]
    pub refresh_token_expiry_secs: u64,

    /// JWT issuer
    #[serde(default = "default_jwt_issuer")]
    pub issuer: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second
    #[serde(default = "default_requests_per_second")]
    pub requests_per_second: u32,

    /// Burst size
    #[serde(default = "default_burst_size")]
    pub burst_size: u32,

    /// Enable rate limiting
    #[serde(default = "default_rate_limit_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ServicesConfig {
    /// Transaction service URL
    #[serde(default = "default_transaction_service_url")]
    pub transaction_service: String,

    /// Privacy service URL
    #[serde(default = "default_privacy_service_url")]
    pub privacy_service: String,

    /// Compliance service URL
    #[serde(default = "default_compliance_service_url")]
    pub compliance_service: String,

    /// Bridge service URL
    #[serde(default = "default_bridge_service_url")]
    pub bridge_service: String,

    /// Proof coordinator URL
    #[serde(default = "default_proof_coordinator_url")]
    pub proof_coordinator: String,
}

// Default value functions
fn default_host() -> String {
    env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string())
}

fn default_port() -> u16 {
    env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080)
}

fn default_metrics_port() -> u16 {
    env::var("METRICS_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(9090)
}

fn default_environment() -> String {
    env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string())
}

fn default_database_url() -> String {
    env::var("DATABASE_URL")
        .unwrap_or_else(|_| "postgres://nexuszero:nexuszero@localhost:5432/nexuszero".to_string())
}

fn default_max_connections() -> u32 {
    env::var("DATABASE_MAX_CONNECTIONS")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(10)
}

fn default_min_connections() -> u32 {
    env::var("DATABASE_MIN_CONNECTIONS")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(2)
}

fn default_connect_timeout() -> u64 {
    env::var("DATABASE_CONNECT_TIMEOUT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(30)
}

fn default_redis_url() -> String {
    env::var("REDIS_URL").unwrap_or_else(|_| "redis://localhost:6379".to_string())
}

fn default_redis_pool_size() -> u32 {
    env::var("REDIS_POOL_SIZE")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(10)
}

fn default_jwt_secret() -> String {
    env::var("JWT_SECRET").unwrap_or_else(|_| {
        if cfg!(debug_assertions) {
            "DEVELOPMENT_SECRET_KEY_DO_NOT_USE_IN_PRODUCTION_32BYTES!".to_string()
        } else {
            panic!("JWT_SECRET must be set in production")
        }
    })
}

fn default_access_token_expiry() -> u64 {
    env::var("JWT_ACCESS_TOKEN_EXPIRY")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3600) // 1 hour
}

fn default_refresh_token_expiry() -> u64 {
    env::var("JWT_REFRESH_TOKEN_EXPIRY")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(604800) // 7 days
}

fn default_jwt_issuer() -> String {
    env::var("JWT_ISSUER").unwrap_or_else(|_| "nexuszero-api".to_string())
}

fn default_requests_per_second() -> u32 {
    env::var("RATE_LIMIT_RPS")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(100)
}

fn default_burst_size() -> u32 {
    env::var("RATE_LIMIT_BURST")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(200)
}

fn default_rate_limit_enabled() -> bool {
    env::var("RATE_LIMIT_ENABLED")
        .ok()
        .map(|v| v == "true" || v == "1")
        .unwrap_or(true)
}

fn default_transaction_service_url() -> String {
    env::var("TRANSACTION_SERVICE_URL")
        .unwrap_or_else(|_| "http://localhost:8081".to_string())
}

fn default_privacy_service_url() -> String {
    env::var("PRIVACY_SERVICE_URL").unwrap_or_else(|_| "http://localhost:8082".to_string())
}

fn default_compliance_service_url() -> String {
    env::var("COMPLIANCE_SERVICE_URL").unwrap_or_else(|_| "http://localhost:8083".to_string())
}

fn default_bridge_service_url() -> String {
    env::var("BRIDGE_SERVICE_URL").unwrap_or_else(|_| "http://localhost:8084".to_string())
}

fn default_proof_coordinator_url() -> String {
    env::var("PROOF_COORDINATOR_URL").unwrap_or_else(|_| "http://localhost:8085".to_string())
}

// Default implementations
impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            url: default_database_url(),
            max_connections: default_max_connections(),
            min_connections: default_min_connections(),
            connect_timeout_secs: default_connect_timeout(),
        }
    }
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: default_redis_url(),
            pool_size: default_redis_pool_size(),
        }
    }
}

impl Default for JwtConfig {
    fn default() -> Self {
        Self {
            secret: default_jwt_secret(),
            access_token_expiry_secs: default_access_token_expiry(),
            refresh_token_expiry_secs: default_refresh_token_expiry(),
            issuer: default_jwt_issuer(),
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_second: default_requests_per_second(),
            burst_size: default_burst_size(),
            enabled: default_rate_limit_enabled(),
        }
    }
}

impl Default for ServicesConfig {
    fn default() -> Self {
        Self {
            transaction_service: default_transaction_service_url(),
            privacy_service: default_privacy_service_url(),
            compliance_service: default_compliance_service_url(),
            bridge_service: default_bridge_service_url(),
            proof_coordinator: default_proof_coordinator_url(),
        }
    }
}

impl Config {
    /// Load configuration from environment and config files
    pub fn load() -> anyhow::Result<Self> {
        // Try to load .env file if it exists
        let _ = dotenvy::dotenv();

        // Build configuration
        let config = config::Config::builder()
            // Add defaults
            .set_default("host", default_host())?
            .set_default("port", default_port())?
            .set_default("metrics_port", default_metrics_port())?
            .set_default("environment", default_environment())?
            // Try to load config file
            .add_source(config::File::with_name("config/gateway").required(false))
            .add_source(config::File::with_name("config/gateway.local").required(false))
            // Override with environment variables
            .add_source(
                config::Environment::with_prefix("NEXUSZERO")
                    .separator("__")
                    .try_parsing(true),
            )
            .build()?;

        let config: Config = config.try_deserialize().unwrap_or_else(|_| Config {
            host: default_host(),
            port: default_port(),
            metrics_port: default_metrics_port(),
            environment: default_environment(),
            database: DatabaseConfig::default(),
            redis: RedisConfig::default(),
            jwt: JwtConfig::default(),
            rate_limit: RateLimitConfig::default(),
            services: ServicesConfig::default(),
        });

        Ok(config)
    }

    /// Check if running in production
    pub fn is_production(&self) -> bool {
        self.environment == "production"
    }

    /// Check if running in development
    pub fn is_development(&self) -> bool {
        self.environment == "development"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = Config {
            host: default_host(),
            port: default_port(),
            metrics_port: default_metrics_port(),
            environment: default_environment(),
            database: DatabaseConfig::default(),
            redis: RedisConfig::default(),
            jwt: JwtConfig::default(),
            rate_limit: RateLimitConfig::default(),
            services: ServicesConfig::default(),
        };

        assert_eq!(config.port, 8080);
        assert_eq!(config.metrics_port, 9090);
        assert_eq!(config.environment, "development");
        assert!(config.is_development());
        assert!(!config.is_production());
    }

    #[test]
    fn test_jwt_config_defaults() {
        let jwt = JwtConfig::default();
        assert_eq!(jwt.access_token_expiry_secs, 3600);
        assert_eq!(jwt.refresh_token_expiry_secs, 604800);
        assert_eq!(jwt.issuer, "nexuszero-api");
    }

    #[test]
    fn test_rate_limit_config_defaults() {
        let rate_limit = RateLimitConfig::default();
        assert_eq!(rate_limit.requests_per_second, 100);
        assert_eq!(rate_limit.burst_size, 200);
        assert!(rate_limit.enabled);
    }
}
