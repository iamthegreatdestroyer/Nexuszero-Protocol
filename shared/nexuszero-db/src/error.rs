//! Database errors

use thiserror::Error;

/// Database operation errors
#[derive(Debug, Error)]
pub enum DbError {
    #[error("Connection failed: {0}")]
    Connection(String),

    #[error("Query failed: {0}")]
    Query(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Constraint violation: {0}")]
    Constraint(String),

    #[error("Transaction failed: {0}")]
    Transaction(String),

    #[error("Migration failed: {0}")]
    Migration(String),

    #[error("Redis error: {0}")]
    Redis(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Pool exhausted")]
    PoolExhausted,

    #[error("Timeout")]
    Timeout,
}

impl From<sqlx::Error> for DbError {
    fn from(err: sqlx::Error) -> Self {
        match err {
            sqlx::Error::RowNotFound => DbError::NotFound("Row not found".into()),
            sqlx::Error::PoolTimedOut => DbError::Timeout,
            sqlx::Error::PoolClosed => DbError::PoolExhausted,
            sqlx::Error::Database(ref e) => {
                if let Some(constraint) = e.constraint() {
                    DbError::Constraint(constraint.to_string())
                } else {
                    DbError::Query(err.to_string())
                }
            }
            _ => DbError::Query(err.to_string()),
        }
    }
}
