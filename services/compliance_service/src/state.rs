//! Application state for Compliance Service

use crate::config::ComplianceConfig;
use crate::error::Result;
use crate::rules::RuleEngine;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Application state shared across handlers
pub struct AppState {
    /// Configuration
    pub config: ComplianceConfig,
    /// Database pool
    pub db: PgPool,
    /// Redis connection
    pub redis: redis::aio::ConnectionManager,
    /// Rule engine
    pub rule_engine: Arc<RwLock<RuleEngine>>,
    /// Service start time
    pub started_at: chrono::DateTime<chrono::Utc>,
}

impl AppState {
    /// Create new application state
    pub async fn new(config: ComplianceConfig) -> Result<Self> {
        // Connect to PostgreSQL
        let db = PgPoolOptions::new()
            .max_connections(20)
            .connect(&config.database_url)
            .await?;

        // Connect to Redis
        let redis_client = redis::Client::open(config.redis_url.clone())?;
        let redis = redis::aio::ConnectionManager::new(redis_client).await?;

        // Initialize rule engine
        let rule_engine = Arc::new(RwLock::new(RuleEngine::new(&config)));

        Ok(Self {
            config,
            db,
            redis,
            rule_engine,
            started_at: chrono::Utc::now(),
        })
    }

    /// Start background worker tasks
    pub fn start_background_workers(&self) {
        // Watchlist update worker
        let config = self.config.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(
                    config.watchlist_update_interval,
                ))
                .await;
                tracing::info!("Refreshing watchlist data...");
                // TODO: Implement watchlist refresh
            }
        });

        // Report generation worker
        let _config = self.config.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(3600)).await;
                tracing::info!("Processing scheduled reports...");
                // TODO: Implement scheduled report generation
            }
        });

        // Audit log cleanup worker
        let retention_days = self.config.audit_retention_days;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(86400)).await;
                tracing::info!("Cleaning up old audit logs (retention: {} days)...", retention_days);
                // TODO: Implement audit log cleanup
            }
        });

        tracing::info!("Background workers started");
    }

    /// Get service uptime in seconds
    pub fn uptime_seconds(&self) -> i64 {
        (chrono::Utc::now() - self.started_at).num_seconds()
    }

    /// Check if service is healthy
    pub async fn is_healthy(&self) -> bool {
        // Check database connectivity
        if sqlx::query("SELECT 1")
            .fetch_one(&self.db)
            .await
            .is_err()
        {
            return false;
        }

        // Check Redis connectivity
        let mut redis = self.redis.clone();
        if redis::cmd("PING")
            .query_async::<String>(&mut redis)
            .await
            .is_err()
        {
            return false;
        }

        true
    }
}
