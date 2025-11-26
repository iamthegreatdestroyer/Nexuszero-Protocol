//! Error types for NexusZero SDK

use thiserror::Error;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Result type for NexusZero operations
pub type Result<T> = std::result::Result<T, NexusZeroError>;

/// Errors that can occur in NexusZero SDK operations
#[derive(Error, Debug, Clone)]
pub enum NexusZeroError {
    /// Invalid privacy level specified
    #[error("Invalid privacy level: {0}. Must be 0-5.")]
    InvalidPrivacyLevel(u8),

    /// Network request failed
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Proof generation failed
    #[error("Proof generation failed: {0}")]
    ProofGenerationError(String),

    /// Proof verification failed
    #[error("Proof verification failed: {0}")]
    ProofVerificationError(String),

    /// Transaction creation failed
    #[error("Transaction error: {0}")]
    TransactionError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Compliance requirement not met
    #[error("Compliance error: {0}")]
    ComplianceError(String),

    /// Bridge operation error
    #[error("Bridge error: {0}")]
    BridgeError(String),
}

#[cfg(feature = "wasm")]
impl From<NexusZeroError> for JsValue {
    fn from(err: NexusZeroError) -> JsValue {
        JsValue::from_str(&err.to_string())
    }
}

impl From<serde_json::Error> for NexusZeroError {
    fn from(err: serde_json::Error) -> Self {
        NexusZeroError::SerializationError(err.to_string())
    }
}
