//! Error types for Compliance Service

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

pub type Result<T> = std::result::Result<T, ComplianceError>;

#[derive(Debug, Error)]
pub enum ComplianceError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Redis error: {0}")]
    Redis(#[from] redis::RedisError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Transaction not found: {0}")]
    TransactionNotFound(String),

    #[error("Entity not found: {0}")]
    EntityNotFound(String),

    #[error("SAR not found: {0}")]
    SarNotFound(String),

    #[error("Rule not found: {0}")]
    RuleNotFound(String),

    #[error("Report not found: {0}")]
    ReportNotFound(String),

    #[error("Jurisdiction not supported: {0}")]
    UnsupportedJurisdiction(String),

    #[error("Compliance check failed: {0}")]
    ComplianceFailed(String),

    #[error("Risk assessment failed: {0}")]
    RiskAssessmentFailed(String),

    #[error("KYC verification failed: {0}")]
    KycFailed(String),

    #[error("AML screening failed: {0}")]
    AmlFailed(String),

    #[error("Watchlist match: {0}")]
    WatchlistMatch(String),

    #[error("Travel Rule violation: {0}")]
    TravelRuleViolation(String),

    #[error("Invalid rule configuration: {0}")]
    InvalidRule(String),

    #[error("Threshold exceeded: {0}")]
    ThresholdExceeded(String),

    #[error("Rate limited")]
    RateLimited,

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("External service error: {0}")]
    ExternalService(String),

    #[error("Authorization failed: {0}")]
    Unauthorized(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl IntoResponse for ComplianceError {
    fn into_response(self) -> Response {
        let (status, error_code, message) = match &self {
            ComplianceError::Database(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "DATABASE_ERROR",
                self.to_string(),
            ),
            ComplianceError::Redis(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "CACHE_ERROR",
                self.to_string(),
            ),
            ComplianceError::Serialization(_) => (
                StatusCode::BAD_REQUEST,
                "SERIALIZATION_ERROR",
                self.to_string(),
            ),
            ComplianceError::Config(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "CONFIG_ERROR",
                self.to_string(),
            ),
            ComplianceError::TransactionNotFound(_) => (
                StatusCode::NOT_FOUND,
                "TRANSACTION_NOT_FOUND",
                self.to_string(),
            ),
            ComplianceError::EntityNotFound(_) => (
                StatusCode::NOT_FOUND,
                "ENTITY_NOT_FOUND",
                self.to_string(),
            ),
            ComplianceError::SarNotFound(_) => (
                StatusCode::NOT_FOUND,
                "SAR_NOT_FOUND",
                self.to_string(),
            ),
            ComplianceError::RuleNotFound(_) => (
                StatusCode::NOT_FOUND,
                "RULE_NOT_FOUND",
                self.to_string(),
            ),
            ComplianceError::ReportNotFound(_) => (
                StatusCode::NOT_FOUND,
                "REPORT_NOT_FOUND",
                self.to_string(),
            ),
            ComplianceError::UnsupportedJurisdiction(_) => (
                StatusCode::BAD_REQUEST,
                "UNSUPPORTED_JURISDICTION",
                self.to_string(),
            ),
            ComplianceError::ComplianceFailed(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "COMPLIANCE_FAILED",
                self.to_string(),
            ),
            ComplianceError::RiskAssessmentFailed(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "RISK_ASSESSMENT_FAILED",
                self.to_string(),
            ),
            ComplianceError::KycFailed(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "KYC_FAILED",
                self.to_string(),
            ),
            ComplianceError::AmlFailed(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "AML_FAILED",
                self.to_string(),
            ),
            ComplianceError::WatchlistMatch(_) => (
                StatusCode::FORBIDDEN,
                "WATCHLIST_MATCH",
                self.to_string(),
            ),
            ComplianceError::TravelRuleViolation(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "TRAVEL_RULE_VIOLATION",
                self.to_string(),
            ),
            ComplianceError::InvalidRule(_) => (
                StatusCode::BAD_REQUEST,
                "INVALID_RULE",
                self.to_string(),
            ),
            ComplianceError::ThresholdExceeded(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "THRESHOLD_EXCEEDED",
                self.to_string(),
            ),
            ComplianceError::RateLimited => (
                StatusCode::TOO_MANY_REQUESTS,
                "RATE_LIMITED",
                "Too many requests".to_string(),
            ),
            ComplianceError::Validation(_) => (
                StatusCode::BAD_REQUEST,
                "VALIDATION_ERROR",
                self.to_string(),
            ),
            ComplianceError::ExternalService(_) => (
                StatusCode::BAD_GATEWAY,
                "EXTERNAL_SERVICE_ERROR",
                self.to_string(),
            ),
            ComplianceError::Unauthorized(_) => (
                StatusCode::UNAUTHORIZED,
                "UNAUTHORIZED",
                self.to_string(),
            ),
            ComplianceError::Internal(_) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                self.to_string(),
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
