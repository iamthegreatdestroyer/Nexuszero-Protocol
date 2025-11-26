//! Application state shared across all handlers
//!
//! Contains database pools, service clients, and shared resources

use crate::config::Config;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Shared application state
pub struct AppState {
    /// PostgreSQL connection pool
    pub db: PgPool,

    /// Redis client
    pub redis: redis::Client,

    /// Configuration
    pub config: Config,

    /// HTTP client for service-to-service communication
    pub http_client: reqwest::Client,

    /// Rate limiter state (IP -> request count)
    pub rate_limiter: Arc<RwLock<RateLimiterState>>,

    /// Active WebSocket connections count
    pub ws_connections: Arc<std::sync::atomic::AtomicU64>,
}

/// Rate limiter state
pub struct RateLimiterState {
    /// Request counts per IP
    pub requests: std::collections::HashMap<String, RequestCounter>,
}

/// Request counter for rate limiting
pub struct RequestCounter {
    pub count: u32,
    pub window_start: std::time::Instant,
}

impl AppState {
    /// Create new application state
    pub async fn new(config: &Config) -> anyhow::Result<Self> {
        tracing::info!("Initializing application state...");

        // Initialize PostgreSQL connection pool
        tracing::info!("Connecting to PostgreSQL...");
        let db = PgPoolOptions::new()
            .max_connections(config.database.max_connections)
            .min_connections(config.database.min_connections)
            .acquire_timeout(Duration::from_secs(config.database.connect_timeout_secs))
            .connect(&config.database.url)
            .await?;

        // Run a test query
        sqlx::query("SELECT 1").execute(&db).await?;
        tracing::info!("PostgreSQL connection established");

        // Initialize Redis client
        tracing::info!("Connecting to Redis...");
        let redis = redis::Client::open(config.redis.url.clone())?;

        // Test Redis connection
        let mut conn = redis.get_multiplexed_async_connection().await?;
        redis::cmd("PING")
            .query_async::<_, String>(&mut conn)
            .await?;
        tracing::info!("Redis connection established");

        // Initialize HTTP client for service communication
        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .connect_timeout(Duration::from_secs(10))
            .pool_max_idle_per_host(10)
            .build()?;

        // Initialize rate limiter state
        let rate_limiter = Arc::new(RwLock::new(RateLimiterState {
            requests: std::collections::HashMap::new(),
        }));

        // Start rate limiter cleanup task
        let rate_limiter_clone = rate_limiter.clone();
        tokio::spawn(async move {
            Self::rate_limiter_cleanup(rate_limiter_clone).await;
        });

        tracing::info!("Application state initialized successfully");

        Ok(Self {
            db,
            redis,
            config: config.clone(),
            http_client,
            rate_limiter,
            ws_connections: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Get a Redis connection
    pub async fn redis_conn(
        &self,
    ) -> anyhow::Result<redis::aio::MultiplexedConnection> {
        Ok(self.redis.get_multiplexed_async_connection().await?)
    }

    /// Increment WebSocket connection count
    pub fn increment_ws_connections(&self) {
        self.ws_connections
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Decrement WebSocket connection count
    pub fn decrement_ws_connections(&self) {
        self.ws_connections
            .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// Get current WebSocket connection count
    pub fn get_ws_connection_count(&self) -> u64 {
        self.ws_connections
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Background task to clean up old rate limiter entries
    async fn rate_limiter_cleanup(rate_limiter: Arc<RwLock<RateLimiterState>>) {
        let cleanup_interval = Duration::from_secs(60);
        let entry_ttl = Duration::from_secs(120);

        loop {
            tokio::time::sleep(cleanup_interval).await;

            let now = std::time::Instant::now();
            let mut state = rate_limiter.write().await;

            state.requests.retain(|_ip, counter| {
                now.duration_since(counter.window_start) < entry_ttl
            });

            tracing::debug!(
                "Rate limiter cleanup: {} entries remaining",
                state.requests.len()
            );
        }
    }

    /// Check database health
    pub async fn check_db_health(&self) -> bool {
        sqlx::query("SELECT 1")
            .execute(&self.db)
            .await
            .is_ok()
    }

    /// Check Redis health
    pub async fn check_redis_health(&self) -> bool {
        if let Ok(mut conn) = self.redis.get_multiplexed_async_connection().await {
            redis::cmd("PING")
                .query_async::<_, String>(&mut conn)
                .await
                .is_ok()
        } else {
            false
        }
    }

    /// Check all service health
    pub async fn check_services_health(&self) -> ServicesHealth {
        let transaction = self
            .http_client
            .get(format!("{}/health", self.config.services.transaction_service))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        let privacy = self
            .http_client
            .get(format!("{}/health", self.config.services.privacy_service))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        let compliance = self
            .http_client
            .get(format!("{}/health", self.config.services.compliance_service))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        let bridge = self
            .http_client
            .get(format!("{}/health", self.config.services.bridge_service))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false);

        ServicesHealth {
            transaction,
            privacy,
            compliance,
            bridge,
        }
    }
}

/// Service health status
#[derive(Debug, Clone, serde::Serialize)]
pub struct ServicesHealth {
    pub transaction: bool,
    pub privacy: bool,
    pub compliance: bool,
    pub bridge: bool,
}

impl ServicesHealth {
    /// Check if all services are healthy
    pub fn all_healthy(&self) -> bool {
        self.transaction && self.privacy && self.compliance && self.bridge
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_services_health() {
        let health = ServicesHealth {
            transaction: true,
            privacy: true,
            compliance: true,
            bridge: true,
        };
        assert!(health.all_healthy());

        let health = ServicesHealth {
            transaction: true,
            privacy: false,
            compliance: true,
            bridge: true,
        };
        assert!(!health.all_healthy());
    }
}
