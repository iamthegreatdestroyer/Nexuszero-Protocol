//! Error types for chain connectors

use thiserror::Error;

/// Unified error type for all chain connector operations
#[derive(Debug, Error)]
pub enum ChainError {
    /// Failed to establish connection to the chain
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),

    /// RPC call to the chain failed
    #[error("RPC error: {0}")]
    RpcError(String),

    /// Transaction submission failed
    #[error("Transaction failed: {0}")]
    TransactionFailed(String),

    /// Transaction was rejected by the network
    #[error("Transaction rejected: {0}")]
    TransactionRejected(String),

    /// Transaction timed out waiting for confirmation
    #[error("Transaction timeout after {0} seconds")]
    TransactionTimeout(u64),

    /// Proof verification failed on-chain
    #[error("Proof verification failed: {0}")]
    ProofVerificationFailed(String),

    /// Invalid proof format or data
    #[error("Invalid proof: {0}")]
    InvalidProof(String),

    /// Insufficient funds for the operation
    #[error("Insufficient funds: required {required}, available {available}")]
    InsufficientFunds { required: u128, available: u128 },

    /// Chain is not supported
    #[error("Chain not supported: {0}")]
    ChainNotSupported(String),

    /// Invalid address format
    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    /// Contract interaction failed
    #[error("Contract error: {0}")]
    ContractError(String),

    /// Event subscription failed
    #[error("Subscription failed: {0}")]
    SubscriptionFailed(String),

    /// Signing operation failed
    #[error("Signing failed: {0}")]
    SigningFailed(String),

    /// Key management error
    #[error("Key error: {0}")]
    KeyError(String),

    /// Rate limiting by the RPC provider
    #[error("Rate limited, retry after {0} ms")]
    RateLimited(u64),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Generic internal error
    #[error("Internal error: {0}")]
    InternalError(String),
}

impl ChainError {
    /// Check if the error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ChainError::RpcError(_)
                | ChainError::ConnectionFailed(_)
                | ChainError::TransactionTimeout(_)
                | ChainError::RateLimited(_)
        )
    }

    /// Get suggested retry delay in milliseconds
    pub fn retry_delay_ms(&self) -> Option<u64> {
        match self {
            ChainError::RateLimited(ms) => Some(*ms),
            ChainError::RpcError(_) => Some(1000),
            ChainError::ConnectionFailed(_) => Some(5000),
            ChainError::TransactionTimeout(_) => Some(2000),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for ChainError {
    fn from(err: serde_json::Error) -> Self {
        ChainError::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retryable_errors() {
        assert!(ChainError::RpcError("timeout".into()).is_retryable());
        assert!(ChainError::RateLimited(1000).is_retryable());
        assert!(!ChainError::InvalidAddress("bad".into()).is_retryable());
    }

    #[test]
    fn test_retry_delay() {
        assert_eq!(ChainError::RateLimited(5000).retry_delay_ms(), Some(5000));
        assert_eq!(ChainError::RpcError("err".into()).retry_delay_ms(), Some(1000));
        assert_eq!(ChainError::InvalidAddress("x".into()).retry_delay_ms(), None);
    }
}
