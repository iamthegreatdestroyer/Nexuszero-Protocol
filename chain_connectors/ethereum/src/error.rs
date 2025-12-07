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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_error() {
        let err = EthereumError::ProviderError("Connection refused".to_string());
        assert!(err.to_string().contains("Provider error"));
        assert!(err.to_string().contains("Connection refused"));
    }

    #[test]
    fn test_contract_error() {
        let err = EthereumError::ContractError("Execution reverted".to_string());
        assert!(err.to_string().contains("Contract error"));
    }

    #[test]
    fn test_signing_error() {
        let err = EthereumError::SigningError("Invalid key".to_string());
        assert!(err.to_string().contains("Signing error"));
    }

    #[test]
    fn test_invalid_address() {
        let err = EthereumError::InvalidAddress("0xinvalid".to_string());
        assert!(err.to_string().contains("Invalid address"));
    }

    #[test]
    fn test_gas_estimation_failed() {
        let err = EthereumError::GasEstimationFailed("Out of gas".to_string());
        assert!(err.to_string().contains("Gas estimation failed"));
    }

    #[test]
    fn test_nonce_error() {
        let err = EthereumError::NonceError("Nonce too low".to_string());
        assert!(err.to_string().contains("Nonce error"));
    }

    #[test]
    fn test_no_wallet() {
        let err = EthereumError::NoWallet;
        assert!(err.to_string().contains("No wallet"));
    }

    #[test]
    fn test_abi_error() {
        let err = EthereumError::AbiError("Decode failed".to_string());
        assert!(err.to_string().contains("ABI error"));
    }

    #[test]
    fn test_subscription_error() {
        let err = EthereumError::SubscriptionError("WebSocket closed".to_string());
        assert!(err.to_string().contains("Subscription error"));
    }

    #[test]
    fn test_config_error() {
        let err = EthereumError::ConfigError("Invalid chain ID".to_string());
        assert!(err.to_string().contains("Configuration error"));
    }

    #[test]
    fn test_conversion_provider_error() {
        let err = EthereumError::ProviderError("timeout".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::RpcError(_));
    }

    #[test]
    fn test_conversion_contract_error() {
        let err = EthereumError::ContractError("revert".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::ContractError(_));
    }

    #[test]
    fn test_conversion_signing_error() {
        let err = EthereumError::SigningError("invalid signature".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::SigningFailed(_));
    }

    #[test]
    fn test_conversion_invalid_address() {
        let err = EthereumError::InvalidAddress("bad address".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::InvalidAddress(_));
    }

    #[test]
    fn test_conversion_gas_estimation() {
        let err = EthereumError::GasEstimationFailed("insufficient gas".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::RpcError(_));
    }

    #[test]
    fn test_conversion_nonce_error() {
        let err = EthereumError::NonceError("nonce gap".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::TransactionFailed(_));
    }

    #[test]
    fn test_conversion_no_wallet() {
        let err = EthereumError::NoWallet;
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::KeyError(_));
    }

    #[test]
    fn test_conversion_abi_error() {
        let err = EthereumError::AbiError("invalid encoding".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::SerializationError(_));
    }

    #[test]
    fn test_conversion_subscription_error() {
        let err = EthereumError::SubscriptionError("disconnected".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::SubscriptionFailed(_));
    }

    #[test]
    fn test_conversion_config_error() {
        let err = EthereumError::ConfigError("invalid config".to_string());
        let chain_err: ChainError = err.into();
        matches!(chain_err, ChainError::ConfigError(_));
    }

    #[test]
    fn test_all_error_variants() {
        let errors: Vec<EthereumError> = vec![
            EthereumError::ProviderError("test".to_string()),
            EthereumError::ContractError("test".to_string()),
            EthereumError::SigningError("test".to_string()),
            EthereumError::InvalidAddress("test".to_string()),
            EthereumError::GasEstimationFailed("test".to_string()),
            EthereumError::NonceError("test".to_string()),
            EthereumError::NoWallet,
            EthereumError::AbiError("test".to_string()),
            EthereumError::SubscriptionError("test".to_string()),
            EthereumError::ConfigError("test".to_string()),
        ];
        
        // All errors should have Display
        for err in errors {
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn test_error_debug() {
        let err = EthereumError::ProviderError("test".to_string());
        let debug = format!("{:?}", err);
        assert!(debug.contains("ProviderError"));
    }
}
