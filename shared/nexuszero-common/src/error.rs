//! Common error types

use thiserror::Error;
use serde::{Deserialize, Serialize};

/// Common application errors
#[derive(Debug, Error)]
pub enum AppError {
    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Forbidden: {0}")]
    Forbidden(String),

    #[error("Conflict: {0}")]
    Conflict(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("External service error: {0}")]
    ExternalService(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("Validation failed: {0}")]
    Validation(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl AppError {
    pub fn error_code(&self) -> &'static str {
        match self {
            AppError::NotFound(_) => "NOT_FOUND",
            AppError::InvalidInput(_) => "INVALID_INPUT",
            AppError::Unauthorized(_) => "UNAUTHORIZED",
            AppError::Forbidden(_) => "FORBIDDEN",
            AppError::Conflict(_) => "CONFLICT",
            AppError::RateLimited => "RATE_LIMITED",
            AppError::ServiceUnavailable(_) => "SERVICE_UNAVAILABLE",
            AppError::Internal(_) => "INTERNAL_ERROR",
            AppError::Database(_) => "DATABASE_ERROR",
            AppError::ExternalService(_) => "EXTERNAL_SERVICE_ERROR",
            AppError::Timeout(_) => "TIMEOUT",
            AppError::Validation(_) => "VALIDATION_ERROR",
            AppError::Serialization(_) => "SERIALIZATION_ERROR",
        }
    }

    pub fn status_code(&self) -> u16 {
        match self {
            AppError::NotFound(_) => 404,
            AppError::InvalidInput(_) => 400,
            AppError::Unauthorized(_) => 401,
            AppError::Forbidden(_) => 403,
            AppError::Conflict(_) => 409,
            AppError::RateLimited => 429,
            AppError::ServiceUnavailable(_) => 503,
            AppError::Internal(_) => 500,
            AppError::Database(_) => 500,
            AppError::ExternalService(_) => 502,
            AppError::Timeout(_) => 504,
            AppError::Validation(_) => 422,
            AppError::Serialization(_) => 400,
        }
    }
}

/// Result type alias for AppError
pub type AppResult<T> = Result<T, AppError>;

/// Error response format
#[derive(Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub code: String,
    pub message: String,
    pub request_id: Option<String>,
}

impl From<AppError> for ErrorResponse {
    fn from(err: AppError) -> Self {
        Self {
            code: err.error_code().to_string(),
            message: err.to_string(),
            request_id: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let err = AppError::NotFound("resource".to_string());
        assert_eq!(err.error_code(), "NOT_FOUND");
        assert_eq!(err.status_code(), 404);
    }
}
