//! Solana-specific error types.

use chain_connectors_common::ChainError;
use thiserror::Error;

/// Errors specific to Solana blockchain operations.
#[derive(Debug, Error)]
pub enum SolanaError {
    /// RPC connection error
    #[error("RPC error: {0}")]
    Rpc(String),
    
    /// Connection error
    #[error("Connection error: {0}")]
    Connection(String),
    
    /// Transaction error
    #[error("Transaction error: {0}")]
    Transaction(String),
    
    /// Invalid address format
    #[error("Invalid address: {0}")]
    InvalidAddress(String),
    
    /// Signing error
    #[error("Signing error: {0}")]
    Signing(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(String),
    
    /// Timeout
    #[error("Timeout")]
    Timeout,
    
    /// Program error
    #[error("Program error: {0}")]
    Program(String),
    
    /// Account not found
    #[error("Account not found: {0}")]
    AccountNotFound(String),
    
    /// Insufficient funds
    #[error("Insufficient funds: {0}")]
    InsufficientFunds(String),
    
    /// Internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<SolanaError> for ChainError {
    fn from(error: SolanaError) -> Self {
        match error {
            SolanaError::Rpc(msg) => ChainError::RpcError(msg),
            SolanaError::Connection(msg) => ChainError::ConnectionFailed(msg),
            SolanaError::Transaction(msg) => ChainError::TransactionFailed(msg),
            SolanaError::InvalidAddress(msg) => ChainError::InvalidAddress(msg),
            SolanaError::Signing(msg) => ChainError::SigningFailed(msg),
            SolanaError::Configuration(msg) => ChainError::ConfigError(msg),
            SolanaError::Serialization(msg) => ChainError::SerializationError(msg),
            SolanaError::Timeout => ChainError::TransactionTimeout(30),
            SolanaError::Program(msg) => ChainError::ContractError(msg),
            SolanaError::AccountNotFound(msg) => ChainError::InternalError(format!("Account not found: {}", msg)),
            SolanaError::InsufficientFunds(_) => {
                ChainError::InsufficientFunds { required: 0, available: 0 }
            }
            SolanaError::Internal(msg) => ChainError::InternalError(msg),
        }
    }
}

impl From<ChainError> for SolanaError {
    fn from(error: ChainError) -> Self {
        match error {
            ChainError::ConnectionFailed(msg) => SolanaError::Connection(msg),
            ChainError::RpcError(msg) => SolanaError::Rpc(msg),
            ChainError::TransactionFailed(msg) => SolanaError::Transaction(msg),
            ChainError::TransactionRejected(msg) => SolanaError::Transaction(msg),
            ChainError::TransactionTimeout(_) => SolanaError::Timeout,
            ChainError::ProofVerificationFailed(msg) => SolanaError::Program(msg),
            ChainError::InvalidProof(msg) => SolanaError::Program(msg),
            ChainError::InsufficientFunds { required, available } => {
                SolanaError::InsufficientFunds(format!("required: {}, available: {}", required, available))
            }
            ChainError::ChainNotSupported(msg) => SolanaError::Configuration(msg),
            ChainError::InvalidAddress(msg) => SolanaError::InvalidAddress(msg),
            ChainError::ContractError(msg) => SolanaError::Program(msg),
            ChainError::SubscriptionFailed(msg) => SolanaError::Internal(msg),
            ChainError::SigningFailed(msg) => SolanaError::Signing(msg),
            ChainError::KeyError(msg) => SolanaError::Signing(msg),
            ChainError::RateLimited(_) => SolanaError::Rpc("Rate limited".to_string()),
            ChainError::ConfigError(msg) => SolanaError::Configuration(msg),
            ChainError::SerializationError(msg) => SolanaError::Serialization(msg),
            ChainError::InternalError(msg) => SolanaError::Internal(msg),
        }
    }
}

/// Result type for Solana operations.
pub type SolanaResult<T> = Result<T, SolanaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_error() {
        let err = SolanaError::Rpc("Connection timeout".to_string());
        assert!(err.to_string().contains("RPC error"));
        assert!(err.to_string().contains("Connection timeout"));
    }

    #[test]
    fn test_connection_error() {
        let err = SolanaError::Connection("Failed to connect".to_string());
        assert!(err.to_string().contains("Connection error"));
    }

    #[test]
    fn test_transaction_error() {
        let err = SolanaError::Transaction("Signature verification failed".to_string());
        assert!(err.to_string().contains("Transaction error"));
    }

    #[test]
    fn test_invalid_address_error() {
        let err = SolanaError::InvalidAddress("Invalid pubkey".to_string());
        assert!(err.to_string().contains("Invalid address"));
    }

    #[test]
    fn test_signing_error() {
        let err = SolanaError::Signing("No signer available".to_string());
        assert!(err.to_string().contains("Signing error"));
    }

    #[test]
    fn test_configuration_error() {
        let err = SolanaError::Configuration("Missing RPC URL".to_string());
        assert!(err.to_string().contains("Configuration error"));
    }

    #[test]
    fn test_serialization_error() {
        let err = SolanaError::Serialization("Invalid data".to_string());
        assert!(err.to_string().contains("Serialization error"));
    }

    #[test]
    fn test_timeout_error() {
        let err = SolanaError::Timeout;
        assert!(err.to_string().contains("Timeout"));
    }

    #[test]
    fn test_program_error() {
        let err = SolanaError::Program("Program execution failed".to_string());
        assert!(err.to_string().contains("Program error"));
    }

    #[test]
    fn test_account_not_found_error() {
        let err = SolanaError::AccountNotFound("PDA account".to_string());
        assert!(err.to_string().contains("Account not found"));
    }

    #[test]
    fn test_insufficient_funds_error() {
        let err = SolanaError::InsufficientFunds("Need 1 SOL".to_string());
        assert!(err.to_string().contains("Insufficient funds"));
    }

    #[test]
    fn test_internal_error() {
        let err = SolanaError::Internal("Unexpected state".to_string());
        assert!(err.to_string().contains("Internal error"));
    }

    #[test]
    fn test_conversion_rpc_to_chain_error() {
        let err = SolanaError::Rpc("timeout".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::RpcError(_));
    }

    #[test]
    fn test_conversion_connection_to_chain_error() {
        let err = SolanaError::Connection("refused".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::ConnectionFailed(_));
    }

    #[test]
    fn test_conversion_transaction_to_chain_error() {
        let err = SolanaError::Transaction("failed".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::TransactionFailed(_));
    }

    #[test]
    fn test_conversion_invalid_address_to_chain_error() {
        let err = SolanaError::InvalidAddress("bad".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::InvalidAddress(_));
    }

    #[test]
    fn test_conversion_signing_to_chain_error() {
        let err = SolanaError::Signing("no key".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::SigningFailed(_));
    }

    #[test]
    fn test_conversion_config_to_chain_error() {
        let err = SolanaError::Configuration("invalid".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::ConfigError(_));
    }

    #[test]
    fn test_conversion_serialization_to_chain_error() {
        let err = SolanaError::Serialization("bad data".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::SerializationError(_));
    }

    #[test]
    fn test_conversion_timeout_to_chain_error() {
        let err = SolanaError::Timeout;
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::TransactionTimeout(_));
    }

    #[test]
    fn test_conversion_program_to_chain_error() {
        let err = SolanaError::Program("error".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::ContractError(_));
    }

    #[test]
    fn test_conversion_account_not_found_to_chain_error() {
        let err = SolanaError::AccountNotFound("acc".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::InternalError(_));
    }

    #[test]
    fn test_conversion_insufficient_funds_to_chain_error() {
        let err = SolanaError::InsufficientFunds("0 SOL".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::InsufficientFunds { .. });
    }

    #[test]
    fn test_conversion_internal_to_chain_error() {
        let err = SolanaError::Internal("unexpected".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::InternalError(_));
    }

    #[test]
    fn test_conversion_chain_error_to_solana_connection() {
        let chain_err = ChainError::ConnectionFailed("failed".to_string());
        let solana_err: SolanaError = chain_err.into();
        matches!(solana_err, SolanaError::Connection(_));
    }

    #[test]
    fn test_conversion_chain_error_to_solana_rpc() {
        let chain_err = ChainError::RpcError("error".to_string());
        let solana_err: SolanaError = chain_err.into();
        matches!(solana_err, SolanaError::Rpc(_));
    }

    #[test]
    fn test_conversion_chain_error_to_solana_transaction() {
        let chain_err = ChainError::TransactionFailed("tx failed".to_string());
        let solana_err: SolanaError = chain_err.into();
        matches!(solana_err, SolanaError::Transaction(_));
    }

    #[test]
    fn test_conversion_chain_error_to_solana_timeout() {
        let chain_err = ChainError::TransactionTimeout(30);
        let solana_err: SolanaError = chain_err.into();
        matches!(solana_err, SolanaError::Timeout);
    }

    #[test]
    fn test_conversion_chain_error_to_solana_insufficient_funds() {
        let chain_err = ChainError::InsufficientFunds { required: 100, available: 50 };
        let solana_err: SolanaError = chain_err.into();
        matches!(solana_err, SolanaError::InsufficientFunds(_));
    }

    #[test]
    fn test_error_debug() {
        let err = SolanaError::Rpc("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("Rpc"));
    }

    #[test]
    fn test_all_error_variants_display() {
        let errors: Vec<SolanaError> = vec![
            SolanaError::Rpc("test".to_string()),
            SolanaError::Connection("test".to_string()),
            SolanaError::Transaction("test".to_string()),
            SolanaError::InvalidAddress("test".to_string()),
            SolanaError::Signing("test".to_string()),
            SolanaError::Configuration("test".to_string()),
            SolanaError::Serialization("test".to_string()),
            SolanaError::Timeout,
            SolanaError::Program("test".to_string()),
            SolanaError::AccountNotFound("test".to_string()),
            SolanaError::InsufficientFunds("test".to_string()),
            SolanaError::Internal("test".to_string()),
        ];
        
        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn test_solana_result_type() {
        fn returns_ok() -> SolanaResult<u64> {
            Ok(42)
        }
        
        fn returns_err() -> SolanaResult<u64> {
            Err(SolanaError::Timeout)
        }
        
        assert!(returns_ok().is_ok());
        assert!(returns_err().is_err());
    }
}
