//! Bitcoin-specific error types

use thiserror::Error;
use chain_connectors_common::ChainError;

/// Bitcoin-specific errors
#[derive(Debug, Error)]
pub enum BitcoinError {
    /// RPC connection error
    #[error("RPC error: {0}")]
    RpcError(String),

    /// Transaction building error
    #[error("Transaction build error: {0}")]
    TransactionBuildError(String),

    /// PSBT error
    #[error("PSBT error: {0}")]
    PsbtError(String),

    /// Signing error
    #[error("Signing error: {0}")]
    SigningError(String),

    /// Invalid address
    #[error("Invalid address: {0}")]
    InvalidAddress(String),

    /// Insufficient funds
    #[error("Insufficient funds: need {required} sat, have {available} sat")]
    InsufficientFunds { required: u64, available: u64 },

    /// UTXO selection error
    #[error("UTXO selection error: {0}")]
    UtxoSelectionError(String),

    /// Fee estimation error
    #[error("Fee estimation error: {0}")]
    FeeEstimationError(String),

    /// Script error
    #[error("Script error: {0}")]
    ScriptError(String),

    /// Taproot error
    #[error("Taproot error: {0}")]
    TaprootError(String),

    /// Network mismatch
    #[error("Network mismatch: expected {expected}, got {actual}")]
    NetworkMismatch { expected: String, actual: String },

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Wallet not found
    #[error("Wallet not found: {0}")]
    WalletNotFound(String),
}

impl From<BitcoinError> for ChainError {
    fn from(err: BitcoinError) -> Self {
        match err {
            BitcoinError::RpcError(msg) => ChainError::RpcError(msg),
            BitcoinError::TransactionBuildError(msg) => ChainError::TransactionFailed(msg),
            BitcoinError::PsbtError(msg) => ChainError::TransactionFailed(msg),
            BitcoinError::SigningError(msg) => ChainError::SigningFailed(msg),
            BitcoinError::InvalidAddress(msg) => ChainError::InvalidAddress(msg),
            BitcoinError::InsufficientFunds { required, available } => {
                ChainError::InsufficientFunds {
                    required: required as u128,
                    available: available as u128,
                }
            }
            BitcoinError::UtxoSelectionError(msg) => ChainError::TransactionFailed(msg),
            BitcoinError::FeeEstimationError(msg) => ChainError::RpcError(msg),
            BitcoinError::ScriptError(msg) => ChainError::TransactionFailed(msg),
            BitcoinError::TaprootError(msg) => ChainError::TransactionFailed(msg),
            BitcoinError::NetworkMismatch { expected, actual } => {
                ChainError::ConfigError(format!("Network mismatch: {} vs {}", expected, actual))
            }
            BitcoinError::ConfigError(msg) => ChainError::ConfigError(msg),
            BitcoinError::WalletNotFound(msg) => ChainError::ConfigError(msg),
        }
    }
}

impl From<bitcoincore_rpc::Error> for BitcoinError {
    fn from(err: bitcoincore_rpc::Error) -> Self {
        BitcoinError::RpcError(err.to_string())
    }
}

impl From<bitcoin::address::ParseError> for BitcoinError {
    fn from(err: bitcoin::address::ParseError) -> Self {
        BitcoinError::InvalidAddress(err.to_string())
    }
}

impl From<bitcoin::psbt::Error> for BitcoinError {
    fn from(err: bitcoin::psbt::Error) -> Self {
        BitcoinError::PsbtError(err.to_string())
    }
}
