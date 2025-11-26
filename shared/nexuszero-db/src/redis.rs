//! Redis utilities

use redis::{Client, aio::ConnectionManager, AsyncCommands};
use std::time::Duration;
use crate::error::DbError;
use serde::{Serialize, de::DeserializeOwned};

/// Redis connection configuration
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub default_ttl: Duration,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://localhost:6379".to_string(),
            default_ttl: Duration::from_secs(3600),
        }
    }
}

impl RedisConfig {
    pub fn from_env() -> Self {
        Self {
            url: std::env::var("REDIS_URL")
                .unwrap_or_else(|_| Self::default().url),
            default_ttl: Duration::from_secs(
                std::env::var("REDIS_DEFAULT_TTL")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(3600)
            ),
        }
    }
}

/// Create a Redis connection manager
pub async fn create_redis_connection(config: &RedisConfig) -> Result<ConnectionManager, DbError> {
    let client = Client::open(config.url.as_str())
        .map_err(|e| DbError::Redis(e.to_string()))?;
    
    ConnectionManager::new(client)
        .await
        .map_err(|e| DbError::Redis(e.to_string()))
}

/// Redis cache wrapper
#[derive(Clone)]
pub struct RedisCache {
    conn: ConnectionManager,
    default_ttl: Duration,
}

impl RedisCache {
    pub fn new(conn: ConnectionManager, default_ttl: Duration) -> Self {
        Self { conn, default_ttl }
    }

    /// Get a value by key
    pub async fn get<T: DeserializeOwned>(&self, key: &str) -> Result<Option<T>, DbError> {
        let mut conn = self.conn.clone();
        let value: Option<String> = conn.get(key).await
            .map_err(|e| DbError::Redis(e.to_string()))?;
        
        match value {
            Some(v) => {
                let parsed = serde_json::from_str(&v)
                    .map_err(|e| DbError::Serialization(e.to_string()))?;
                Ok(Some(parsed))
            }
            None => Ok(None),
        }
    }

    /// Set a value with default TTL
    pub async fn set<T: Serialize>(&self, key: &str, value: &T) -> Result<(), DbError> {
        self.set_with_ttl(key, value, self.default_ttl).await
    }

    /// Set a value with custom TTL
    pub async fn set_with_ttl<T: Serialize>(
        &self,
        key: &str,
        value: &T,
        ttl: Duration,
    ) -> Result<(), DbError> {
        let mut conn = self.conn.clone();
        let serialized = serde_json::to_string(value)
            .map_err(|e| DbError::Serialization(e.to_string()))?;
        
        conn.set_ex(key, serialized, ttl.as_secs())
            .await
            .map_err(|e| DbError::Redis(e.to_string()))
    }

    /// Delete a key
    pub async fn delete(&self, key: &str) -> Result<(), DbError> {
        let mut conn = self.conn.clone();
        conn.del(key)
            .await
            .map_err(|e| DbError::Redis(e.to_string()))
    }

    /// Check if key exists
    pub async fn exists(&self, key: &str) -> Result<bool, DbError> {
        let mut conn = self.conn.clone();
        conn.exists(key)
            .await
            .map_err(|e| DbError::Redis(e.to_string()))
    }

    /// Increment a counter
    pub async fn incr(&self, key: &str) -> Result<i64, DbError> {
        let mut conn = self.conn.clone();
        conn.incr(key, 1)
            .await
            .map_err(|e| DbError::Redis(e.to_string()))
    }

    /// Set TTL on existing key
    pub async fn expire(&self, key: &str, ttl: Duration) -> Result<(), DbError> {
        let mut conn = self.conn.clone();
        conn.expire(key, ttl.as_secs() as i64)
            .await
            .map_err(|e| DbError::Redis(e.to_string()))
    }

    /// Health check
    pub async fn ping(&self) -> Result<(), DbError> {
        let mut conn = self.conn.clone();
        redis::cmd("PING")
            .query_async::<String>(&mut conn)
            .await
            .map(|_| ())
            .map_err(|e| DbError::Redis(e.to_string()))
    }
}
