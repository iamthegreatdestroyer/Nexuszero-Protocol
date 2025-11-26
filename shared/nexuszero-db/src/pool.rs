//! Connection pool management

use sqlx::postgres::PgPool;
use crate::redis::RedisCache;
use std::sync::Arc;

/// Combined database pool holder
#[derive(Clone)]
pub struct DbPools {
    pub postgres: PgPool,
    pub redis: RedisCache,
}

impl DbPools {
    pub fn new(postgres: PgPool, redis: RedisCache) -> Self {
        Self { postgres, redis }
    }
}

/// Shared database state for services
pub type SharedDbPools = Arc<DbPools>;

/// Create shared database pools
pub fn create_shared_pools(postgres: PgPool, redis: RedisCache) -> SharedDbPools {
    Arc::new(DbPools::new(postgres, redis))
}
