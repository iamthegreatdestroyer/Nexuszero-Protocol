//! Privacy Morphing Error Types

use thiserror::Error;

/// Privacy morphing errors
#[derive(Error, Debug)]
pub enum MorphingError {
    #[error("Invalid privacy level: {0} (must be 1-10)")]
    InvalidPrivacyLevel(u8),

    #[error("Schedule error: {0}")]
    ScheduleError(String),

    #[error("Compliance conflict: {0}")]
    ComplianceConflict(String),

    #[error("Anonymity set too small: {size} (minimum: {minimum})")]
    AnonymitySetTooSmall { size: usize, minimum: usize },

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Calculation error: {0}")]
    CalculationError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("{0}")]
    Other(String),
}

/// Result type for morphing operations
pub type MorphingResult<T> = Result<T, MorphingError>;

impl From<String> for MorphingError {
    fn from(s: String) -> Self {
        MorphingError::Other(s)
    }
}

impl From<&str> for MorphingError {
    fn from(s: &str) -> Self {
        MorphingError::Other(s.to_string())
    }
}
