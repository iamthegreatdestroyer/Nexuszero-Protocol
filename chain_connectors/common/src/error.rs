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

    // ===== HARDENING TESTS =====

    #[test]
    fn test_all_retryable_error_types() {
        // Retryable errors
        assert!(ChainError::RpcError("test".into()).is_retryable());
        assert!(ChainError::ConnectionFailed("test".into()).is_retryable());
        assert!(ChainError::TransactionTimeout(60).is_retryable());
        assert!(ChainError::RateLimited(1000).is_retryable());
    }

    #[test]
    fn test_non_retryable_error_types() {
        // Non-retryable errors
        assert!(!ChainError::TransactionFailed("test".into()).is_retryable());
        assert!(!ChainError::TransactionRejected("test".into()).is_retryable());
        assert!(!ChainError::ProofVerificationFailed("test".into()).is_retryable());
        assert!(!ChainError::InvalidProof("test".into()).is_retryable());
        assert!(!ChainError::InsufficientFunds { required: 100, available: 50 }.is_retryable());
        assert!(!ChainError::ChainNotSupported("test".into()).is_retryable());
        assert!(!ChainError::InvalidAddress("test".into()).is_retryable());
        assert!(!ChainError::ContractError("test".into()).is_retryable());
        assert!(!ChainError::SubscriptionFailed("test".into()).is_retryable());
        assert!(!ChainError::SigningFailed("test".into()).is_retryable());
        assert!(!ChainError::KeyError("test".into()).is_retryable());
        assert!(!ChainError::ConfigError("test".into()).is_retryable());
        assert!(!ChainError::SerializationError("test".into()).is_retryable());
        assert!(!ChainError::InternalError("test".into()).is_retryable());
    }

    #[test]
    fn test_retry_delay_all_retryable() {
        assert_eq!(ChainError::RateLimited(5000).retry_delay_ms(), Some(5000));
        assert_eq!(ChainError::RateLimited(0).retry_delay_ms(), Some(0));
        assert_eq!(ChainError::RpcError("err".into()).retry_delay_ms(), Some(1000));
        assert_eq!(ChainError::ConnectionFailed("err".into()).retry_delay_ms(), Some(5000));
        assert_eq!(ChainError::TransactionTimeout(60).retry_delay_ms(), Some(2000));
    }

    #[test]
    fn test_retry_delay_none_for_non_retryable() {
        assert!(ChainError::InvalidAddress("x".into()).retry_delay_ms().is_none());
        assert!(ChainError::ContractError("x".into()).retry_delay_ms().is_none());
        assert!(ChainError::SigningFailed("x".into()).retry_delay_ms().is_none());
    }

    #[test]
    fn test_error_display_messages() {
        // Verify error messages are correctly formatted
        let err = ChainError::ConnectionFailed("network down".into());
        assert!(err.to_string().contains("Connection failed"));
        assert!(err.to_string().contains("network down"));
        
        let err = ChainError::TransactionTimeout(120);
        assert!(err.to_string().contains("120"));
        
        let err = ChainError::InsufficientFunds { required: 1000, available: 500 };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("500"));
    }

    #[test]
    fn test_error_from_serde_json() {
        let json = "{invalid json";
        let result: Result<serde_json::Value, _> = serde_json::from_str(json);
        let err: ChainError = result.unwrap_err().into();
        
        match err {
            ChainError::SerializationError(msg) => {
                assert!(!msg.is_empty());
            }
            _ => panic!("Expected SerializationError"),
        }
    }

    #[test]
    fn test_insufficient_funds_error_details() {
        let err = ChainError::InsufficientFunds { 
            required: 1_000_000_000_000_000_000u128, 
            available: 500_000_000_000_000_000u128 
        };
        
        let msg = err.to_string();
        assert!(msg.contains("1000000000000000000"));
        assert!(msg.contains("500000000000000000"));
    }

    #[test]
    fn test_rate_limited_with_various_delays() {
        for delay in [0, 100, 1000, 5000, 60000, u64::MAX] {
            let err = ChainError::RateLimited(delay);
            assert!(err.is_retryable());
            assert_eq!(err.retry_delay_ms(), Some(delay));
        }
    }

    #[test]
    fn test_error_debug_impl() {
        let errors = vec![
            ChainError::RpcError("rpc failed".into()),
            ChainError::InvalidProof("bad proof".into()),
            ChainError::ChainNotSupported("unknown chain".into()),
        ];
        
        for err in errors {
            // Debug should not panic
            let debug_str = format!("{:?}", err);
            assert!(!debug_str.is_empty());
        }
    }

    #[test]
    fn test_transaction_timeout_values() {
        for timeout in [1, 30, 60, 300, 3600] {
            let err = ChainError::TransactionTimeout(timeout);
            assert!(err.to_string().contains(&timeout.to_string()));
        }
    }

    #[test]
    fn test_all_error_variants_constructible() {
        // Ensure all error variants can be constructed
        let _errors = vec![
            ChainError::ConnectionFailed(String::new()),
            ChainError::RpcError(String::new()),
            ChainError::TransactionFailed(String::new()),
            ChainError::TransactionRejected(String::new()),
            ChainError::TransactionTimeout(0),
            ChainError::ProofVerificationFailed(String::new()),
            ChainError::InvalidProof(String::new()),
            ChainError::InsufficientFunds { required: 0, available: 0 },
            ChainError::ChainNotSupported(String::new()),
            ChainError::InvalidAddress(String::new()),
            ChainError::ContractError(String::new()),
            ChainError::SubscriptionFailed(String::new()),
            ChainError::SigningFailed(String::new()),
            ChainError::KeyError(String::new()),
            ChainError::RateLimited(0),
            ChainError::ConfigError(String::new()),
            ChainError::SerializationError(String::new()),
            ChainError::InternalError(String::new()),
        ];
    }
}
