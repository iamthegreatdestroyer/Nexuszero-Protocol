//! Ethereum-specific error types

use thiserror::Error;
use chain_connectors_common::ChainError;

/// Ethereum-specific errors
#[derive(Debug, Error)]
pub enum EthereumError {
    /// Provider connection error
    #[error("Provider error: {0}")]
    ProviderError(String),

    /// Contract interaction error
    #[error("Contract error: {0}")]
    ContractError(String),

    /// Transaction signing error
    #[error("Signing error: {0}")]
    SigningError(String),

    /// Invalid address format
    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    /// Gas estimation failed
    #[error("Gas estimation failed: {0}")]
    GasEstimationFailed(String),

    /// Nonce management error
    #[error("Nonce error: {0}")]
    NonceError(String),

    /// No wallet configured
    #[error("No wallet configured for signing")]
    NoWallet,

    /// ABI encoding/decoding error
    #[error("ABI error: {0}")]
    AbiError(String),

    /// WebSocket subscription error
    #[error("Subscription error: {0}")]
    SubscriptionError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

impl From<EthereumError> for ChainError {
    fn from(err: EthereumError) -> Self {
        match err {
            EthereumError::ProviderError(msg) => ChainError::RpcError(msg),
            EthereumError::ContractError(msg) => ChainError::ContractError(msg),
            EthereumError::SigningError(msg) => ChainError::SigningFailed(msg),
            EthereumError::InvalidAddress(msg) => ChainError::InvalidAddress(msg),
            EthereumError::GasEstimationFailed(msg) => ChainError::RpcError(msg),
            EthereumError::NonceError(msg) => ChainError::TransactionFailed(msg),
            EthereumError::NoWallet => ChainError::KeyError("No wallet configured".to_string()),
            EthereumError::AbiError(msg) => ChainError::SerializationError(msg),
            EthereumError::SubscriptionError(msg) => ChainError::SubscriptionFailed(msg),
            EthereumError::ConfigError(msg) => ChainError::ConfigError(msg),
        }
    }
}
