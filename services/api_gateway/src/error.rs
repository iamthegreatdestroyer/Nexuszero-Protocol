//! Error types and handling for the API Gateway
//!
//! Provides structured error responses with appropriate HTTP status codes

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;

/// API Error type
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Authentication required")]
    Unauthorized,

    #[error("Invalid credentials")]
    InvalidCredentials,

    #[error("Token expired")]
    TokenExpired,

    #[error("Invalid token")]
    InvalidToken,

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Bad request: {0}")]
    BadRequest(String),

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Internal server error")]
    InternalError,

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("External service error: {0}")]
    ExternalServiceError(String),

    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),

    #[error("Privacy level not supported: {0}")]
    UnsupportedPrivacyLevel(u8),

    #[error("Chain not supported: {0}")]
    UnsupportedChain(String),

    #[error("Insufficient funds")]
    InsufficientFunds,

    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    #[error("Compliance check failed: {0}")]
    ComplianceFailed(String),
}

/// Error response body
#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorDetails,
}

#[derive(Debug, Serialize)]
pub struct ErrorDetails {
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<serde_json::Value>,
}

impl ApiError {
    /// Get the error code
    pub fn code(&self) -> &'static str {
        match self {
            ApiError::Unauthorized => "UNAUTHORIZED",
            ApiError::InvalidCredentials => "INVALID_CREDENTIALS",
            ApiError::TokenExpired => "TOKEN_EXPIRED",
            ApiError::InvalidToken => "INVALID_TOKEN",
            ApiError::Forbidden(_) => "FORBIDDEN",
            ApiError::NotFound(_) => "NOT_FOUND",
            ApiError::BadRequest(_) => "BAD_REQUEST",
            ApiError::ValidationError(_) => "VALIDATION_ERROR",
            ApiError::RateLimitExceeded => "RATE_LIMIT_EXCEEDED",
            ApiError::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
            ApiError::InternalError => "INTERNAL_ERROR",
            ApiError::DatabaseError(_) => "DATABASE_ERROR",
            ApiError::ExternalServiceError(_) => "EXTERNAL_SERVICE_ERROR",
            ApiError::ProofGenerationFailed(_) => "PROOF_GENERATION_FAILED",
            ApiError::UnsupportedPrivacyLevel(_) => "UNSUPPORTED_PRIVACY_LEVEL",
            ApiError::UnsupportedChain(_) => "UNSUPPORTED_CHAIN",
            ApiError::InsufficientFunds => "INSUFFICIENT_FUNDS",
            ApiError::TransactionFailed(_) => "TRANSACTION_FAILED",
            ApiError::ComplianceFailed(_) => "COMPLIANCE_FAILED",
        }
    }

    /// Get the HTTP status code
    pub fn status_code(&self) -> StatusCode {
        match self {
            ApiError::Unauthorized
            | ApiError::InvalidCredentials
            | ApiError::TokenExpired
            | ApiError::InvalidToken => StatusCode::UNAUTHORIZED,
            ApiError::Forbidden(_) => StatusCode::FORBIDDEN,
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::BadRequest(_) | ApiError::ValidationError(_) => StatusCode::BAD_REQUEST,
            ApiError::RateLimitExceeded => StatusCode::TOO_MANY_REQUESTS,
            ApiError::ServiceUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
            ApiError::UnsupportedPrivacyLevel(_) | ApiError::UnsupportedChain(_) => {
                StatusCode::BAD_REQUEST
            }
            ApiError::InsufficientFunds => StatusCode::PAYMENT_REQUIRED,
            ApiError::ComplianceFailed(_) => StatusCode::FORBIDDEN,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_response = ErrorResponse {
            error: ErrorDetails {
                code: self.code().to_string(),
                message: self.to_string(),
                details: None,
            },
        };

        // Log errors
        match &self {
            ApiError::InternalError | ApiError::DatabaseError(_) => {
                tracing::error!("API Error: {:?}", self);
            }
            ApiError::ExternalServiceError(_) | ApiError::ServiceUnavailable(_) => {
                tracing::warn!("API Error: {:?}", self);
            }
            _ => {
                tracing::debug!("API Error: {:?}", self);
            }
        }

        (status, Json(error_response)).into_response()
    }
}

// Implement From traits for common error types
impl From<sqlx::Error> for ApiError {
    fn from(err: sqlx::Error) -> Self {
        tracing::error!("Database error: {:?}", err);
        match err {
            sqlx::Error::RowNotFound => ApiError::NotFound("Resource not found".to_string()),
            _ => ApiError::DatabaseError(err.to_string()),
        }
    }
}

impl From<redis::RedisError> for ApiError {
    fn from(err: redis::RedisError) -> Self {
        tracing::error!("Redis error: {:?}", err);
        ApiError::DatabaseError(format!("Redis error: {}", err))
    }
}

impl From<reqwest::Error> for ApiError {
    fn from(err: reqwest::Error) -> Self {
        tracing::error!("HTTP client error: {:?}", err);
        if err.is_timeout() {
            ApiError::ServiceUnavailable("Service timeout".to_string())
        } else if err.is_connect() {
            ApiError::ServiceUnavailable("Service connection failed".to_string())
        } else {
            ApiError::ExternalServiceError(err.to_string())
        }
    }
}

impl From<jsonwebtoken::errors::Error> for ApiError {
    fn from(err: jsonwebtoken::errors::Error) -> Self {
        use jsonwebtoken::errors::ErrorKind;
        match err.kind() {
            ErrorKind::ExpiredSignature => ApiError::TokenExpired,
            ErrorKind::InvalidToken | ErrorKind::InvalidSignature => ApiError::InvalidToken,
            _ => ApiError::InvalidToken,
        }
    }
}

impl From<serde_json::Error> for ApiError {
    fn from(err: serde_json::Error) -> Self {
        ApiError::BadRequest(format!("JSON parsing error: {}", err))
    }
}

impl From<validator::ValidationErrors> for ApiError {
    fn from(err: validator::ValidationErrors) -> Self {
        ApiError::ValidationError(format!("{}", err))
    }
}

/// Result type alias for API handlers
pub type ApiResult<T> = Result<T, ApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        assert_eq!(ApiError::Unauthorized.code(), "UNAUTHORIZED");
        assert_eq!(ApiError::NotFound("test".to_string()).code(), "NOT_FOUND");
        assert_eq!(ApiError::RateLimitExceeded.code(), "RATE_LIMIT_EXCEEDED");
    }

    #[test]
    fn test_status_codes() {
        assert_eq!(ApiError::Unauthorized.status_code(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            ApiError::NotFound("test".to_string()).status_code(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            ApiError::RateLimitExceeded.status_code(),
            StatusCode::TOO_MANY_REQUESTS
        );
        assert_eq!(
            ApiError::InternalError.status_code(),
            StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
