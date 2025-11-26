//! Privacy Service error types

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Privacy service errors
#[derive(Debug, Error)]
pub enum PrivacyError {
    /// Invalid privacy level
    #[error("Invalid privacy level: {0}")]
    InvalidLevel(i16),

    /// Privacy morph not allowed
    #[error("Privacy morph not allowed: {0}")]
    MorphNotAllowed(String),

    /// Proof generation failed
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    /// Proof verification failed
    #[error("Proof verification failed: {0}")]
    ProofVerificationFailed(String),

    /// Proof not found
    #[error("Proof not found: {0}")]
    ProofNotFound(String),

    /// Disclosure not found
    #[error("Disclosure not found: {0}")]
    DisclosureNotFound(String),

    /// Disclosure expired
    #[error("Disclosure expired: {0}")]
    DisclosureExpired(String),

    /// Disclosure revoked
    #[error("Disclosure revoked: {0}")]
    DisclosureRevoked(String),

    /// Invalid disclosure fields
    #[error("Invalid disclosure fields: {0}")]
    InvalidDisclosureFields(String),

    /// Queue full
    #[error("Proof generation queue is full")]
    QueueFull,

    /// Rate limited
    #[error("Rate limited")]
    RateLimited,

    /// Job not found
    #[error("Job not found: {0}")]
    JobNotFound(String),

    /// Verification key not found
    #[error("Verification key not found: {0}")]
    VerificationKeyNotFound(String),

    /// Proof too large
    #[error("Proof size {0} exceeds maximum allowed")]
    ProofTooLarge(usize),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Timeout
    #[error("Operation timed out")]
    Timeout,

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

impl IntoResponse for PrivacyError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match &self {
            PrivacyError::InvalidLevel(level) => (
                StatusCode::BAD_REQUEST,
                "INVALID_PRIVACY_LEVEL",
                format!("Privacy level {} is not valid (must be 0-5)", level),
            ),
            PrivacyError::MorphNotAllowed(msg) => (
                StatusCode::FORBIDDEN,
                "MORPH_NOT_ALLOWED",
                msg.clone(),
            ),
            PrivacyError::ProofGenerationFailed(msg) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "PROOF_GENERATION_FAILED",
                msg.clone(),
            ),
            PrivacyError::ProofVerificationFailed(msg) => (
                StatusCode::BAD_REQUEST,
                "PROOF_VERIFICATION_FAILED",
                msg.clone(),
            ),
            PrivacyError::ProofNotFound(id) => (
                StatusCode::NOT_FOUND,
                "PROOF_NOT_FOUND",
                format!("Proof {} not found", id),
            ),
            PrivacyError::DisclosureNotFound(id) => (
                StatusCode::NOT_FOUND,
                "DISCLOSURE_NOT_FOUND",
                format!("Disclosure {} not found", id),
            ),
            PrivacyError::DisclosureExpired(id) => (
                StatusCode::GONE,
                "DISCLOSURE_EXPIRED",
                format!("Disclosure {} has expired", id),
            ),
            PrivacyError::DisclosureRevoked(id) => (
                StatusCode::GONE,
                "DISCLOSURE_REVOKED",
                format!("Disclosure {} has been revoked", id),
            ),
            PrivacyError::InvalidDisclosureFields(msg) => (
                StatusCode::BAD_REQUEST,
                "INVALID_DISCLOSURE_FIELDS",
                msg.clone(),
            ),
            PrivacyError::QueueFull => (
                StatusCode::SERVICE_UNAVAILABLE,
                "QUEUE_FULL",
                "Proof generation queue is at capacity".to_string(),
            ),
            PrivacyError::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "RATE_LIMITED",
                "Too many requests, please try again later".to_string(),
            ),
            PrivacyError::JobNotFound(id) => (
                StatusCode::NOT_FOUND,
                "JOB_NOT_FOUND",
                format!("Job {} not found", id),
            ),
            PrivacyError::VerificationKeyNotFound(id) => (
                StatusCode::NOT_FOUND,
                "VERIFICATION_KEY_NOT_FOUND",
                format!("Verification key {} not found", id),
            ),
            PrivacyError::ProofTooLarge(size) => (
                StatusCode::PAYLOAD_TOO_LARGE,
                "PROOF_TOO_LARGE",
                format!("Proof size {} exceeds maximum allowed", size),
            ),
            PrivacyError::InvalidInput(msg) => (
                StatusCode::BAD_REQUEST,
                "INVALID_INPUT",
                msg.clone(),
            ),
            PrivacyError::Timeout => (
                StatusCode::GATEWAY_TIMEOUT,
                "TIMEOUT",
                "Operation timed out".to_string(),
            ),
            PrivacyError::Database(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "DATABASE_ERROR",
                e.to_string(),
            ),
            PrivacyError::Redis(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "REDIS_ERROR",
                e.to_string(),
            ),
            PrivacyError::Serialization(e) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "SERIALIZATION_ERROR",
                e.to_string(),
            ),
            PrivacyError::Internal(msg) => (
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

/// Result type alias
pub type Result<T> = std::result::Result<T, PrivacyError>;
