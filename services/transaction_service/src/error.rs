//! Transaction Service error types

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Transaction service errors
#[derive(Debug, Error)]
pub enum TransactionError {
    /// Transaction not found
    #[error("Transaction not found: {0}")]
    NotFound(String),

    /// Invalid request
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Invalid privacy level
    #[error("Invalid privacy level: {0}")]
    InvalidPrivacyLevel(i16),

    /// Invalid status transition
    #[error("Invalid status transition from {from:?} to {to:?}")]
    InvalidStatusTransition {
        from: super::TransactionStatus,
        to: super::TransactionStatus,
    },

    /// Proof generation failed
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    /// Proof not ready
    #[error("Proof not ready for transaction: {0}")]
    ProofNotReady(String),

    /// Privacy morph failed
    #[error("Privacy morph failed: {0}")]
    PrivacyMorphFailed(String),

    /// Compliance check failed
    #[error("Compliance check failed: {0}")]
    ComplianceCheckFailed(String),

    /// Batch operation failed
    #[error("Batch operation failed: {0}")]
    BatchFailed(String),

    /// External service error
    #[error("External service error: {0}")]
    ExternalService(String),

    /// Database error
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    /// Redis error
    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for TransactionError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match &self {
            TransactionError::NotFound(msg) => {
                (StatusCode::NOT_FOUND, "TRANSACTION_NOT_FOUND", msg.clone())
            }
            TransactionError::InvalidRequest(msg) => {
                (StatusCode::BAD_REQUEST, "INVALID_REQUEST", msg.clone())
            }
            TransactionError::InvalidPrivacyLevel(level) => (
                StatusCode::BAD_REQUEST,
                "INVALID_PRIVACY_LEVEL",
                format!("Privacy level {} is not valid (must be 0-5)", level),
            ),
            TransactionError::InvalidStatusTransition { from, to } => (
                StatusCode::CONFLICT,
                "INVALID_STATUS_TRANSITION",
                format!("Cannot transition from {:?} to {:?}", from, to),
            ),
            TransactionError::ProofGenerationFailed(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "PROOF_GENERATION_FAILED",
                msg.clone(),
            ),
            TransactionError::ProofNotReady(msg) => (
                StatusCode::CONFLICT,
                "PROOF_NOT_READY",
                msg.clone(),
            ),
            TransactionError::PrivacyMorphFailed(msg) => (
                StatusCode::BAD_REQUEST,
                "PRIVACY_MORPH_FAILED",
                msg.clone(),
            ),
            TransactionError::ComplianceCheckFailed(msg) => (
                StatusCode::FORBIDDEN,
                "COMPLIANCE_CHECK_FAILED",
                msg.clone(),
            ),
            TransactionError::BatchFailed(msg) => (
                StatusCode::BAD_REQUEST,
                "BATCH_FAILED",
                msg.clone(),
            ),
            TransactionError::ExternalService(msg) => (
                StatusCode::BAD_GATEWAY,
                "EXTERNAL_SERVICE_ERROR",
                msg.clone(),
            ),
            TransactionError::Database(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "DATABASE_ERROR",
                e.to_string(),
            ),
            TransactionError::Redis(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "REDIS_ERROR",
                e.to_string(),
            ),
            TransactionError::Serialization(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "SERIALIZATION_ERROR",
                e.to_string(),
            ),
            TransactionError::Internal(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                msg.clone(),
            ),
        };

        let body = Json(json!({
            "error": {
                "code": error_code,
                "message": message,
            }
        }));

        (status, body).into_response()
    }
}

/// Result type alias for TransactionError
pub type Result<T> = std::result::Result<T, TransactionError>;
