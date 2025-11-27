//! CLI Error Types and Result Definitions

use thiserror::Error;

/// CLI-specific errors
#[derive(Error, Debug)]
pub enum CliError {
    #[error("API error: {0}")]
    Api(#[from] ApiError),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Wallet error: {0}")]
    Wallet(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Proof generation failed: {0}")]
    ProofGeneration(String),

    #[error("Transaction failed: {0}")]
    Transaction(String),

    #[error("Bridge operation failed: {0}")]
    Bridge(String),

    #[error("Compliance check failed: {0}")]
    Compliance(String),

    #[error("Authentication required")]
    AuthRequired,

    #[error("Operation cancelled")]
    Cancelled,

    #[error("Timeout: {0}")]
    Timeout(String),

    #[error("{0}")]
    Other(String),
}

/// API-specific errors
#[derive(Error, Debug)]
pub enum ApiError {
    #[error("Request failed: {status} - {message}")]
    RequestFailed { status: u16, message: String },

    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    #[error("Authentication failed: {0}")]
    AuthFailed(String),

    #[error("Rate limited: retry after {retry_after}s")]
    RateLimited { retry_after: u64 },

    #[error("Resource not found: {0}")]
    NotFound(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

/// CLI Result type alias
pub type CliResult<T> = Result<T, CliError>;

impl From<String> for CliError {
    fn from(s: String) -> Self {
        CliError::Other(s)
    }
}

impl From<&str> for CliError {
    fn from(s: &str) -> Self {
        CliError::Other(s.to_string())
    }
}

impl From<toml::de::Error> for CliError {
    fn from(e: toml::de::Error) -> Self {
        CliError::Config(e.to_string())
    }
}

impl From<toml::ser::Error> for CliError {
    fn from(e: toml::ser::Error) -> Self {
        CliError::Config(e.to_string())
    }
}
