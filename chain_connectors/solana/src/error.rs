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
