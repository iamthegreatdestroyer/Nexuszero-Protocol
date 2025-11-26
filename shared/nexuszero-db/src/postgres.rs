//! PostgreSQL utilities

use sqlx::postgres::{PgPoolOptions, PgPool};
use std::time::Duration;
use crate::error::DbError;

/// PostgreSQL connection configuration
#[derive(Debug, Clone)]
pub struct PostgresConfig {
    pub url: String,
    pub max_connections: u32,
    pub min_connections: u32,
    pub acquire_timeout: Duration,
    pub idle_timeout: Duration,
    pub max_lifetime: Duration,
}

impl Default for PostgresConfig {
    fn default() -> Self {
        Self {
            url: "postgres://postgres:postgres@localhost:5432/nexuszero".to_string(),
            max_connections: 10,
            min_connections: 2,
            acquire_timeout: Duration::from_secs(30),
            idle_timeout: Duration::from_secs(600),
            max_lifetime: Duration::from_secs(1800),
        }
    }
}

impl PostgresConfig {
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("DATABASE_URL")
                .unwrap_or_else(|_| Self::default().url),
            max_connections: std::env::var("DB_MAX_CONNECTIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(10),
            min_connections: std::env::var("DB_MIN_CONNECTIONS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2),
            ..Default::default()
        }
    }
}

/// Create a PostgreSQL connection pool
pub async fn create_pg_pool(config: &PostgresConfig) -> Result<PgPool, DbError> {
    PgPoolOptions::new()
        .max_connections(config.max_connections)
        .min_connections(config.min_connections)
        .acquire_timeout(config.acquire_timeout)
        .idle_timeout(Some(config.idle_timeout))
        .max_lifetime(Some(config.max_lifetime))
        .connect(&config.url)
        .await
        .map_err(DbError::from)
}

/// Health check for PostgreSQL
pub async fn pg_health_check(pool: &PgPool) -> Result<(), DbError> {
    sqlx::query("SELECT 1")
        .fetch_one(pool)
        .await
        .map(|_| ())
        .map_err(DbError::from)
}

/// Run migrations
/// Note: Each service should run its own migrations using sqlx::migrate!()
/// This function is provided as a helper that services can use with their own migration paths
pub async fn run_migrations_from_path(pool: &PgPool, path: &str) -> Result<(), DbError> {
    sqlx::migrate::Migrator::new(std::path::Path::new(path))
        .await
        .map_err(|e| DbError::Migration(e.to_string()))?
        .run(pool)
        .await
        .map_err(|e| DbError::Migration(e.to_string()))
}

/// Transaction wrapper
pub struct Transaction<'a> {
    tx: sqlx::Transaction<'a, sqlx::Postgres>,
}

impl<'a> Transaction<'a> {
    pub async fn begin(pool: &'a PgPool) -> Result<Self, DbError> {
        let tx = pool.begin().await?;
        Ok(Self { tx })
    }

    pub async fn commit(self) -> Result<(), DbError> {
        self.tx.commit().await.map_err(DbError::from)
    }

    pub async fn rollback(self) -> Result<(), DbError> {
        self.tx.rollback().await.map_err(DbError::from)
    }

    pub fn inner(&mut self) -> &mut sqlx::Transaction<'a, sqlx::Postgres> {
        &mut self.tx
    }
}
