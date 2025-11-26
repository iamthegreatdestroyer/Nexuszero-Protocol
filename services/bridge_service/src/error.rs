//! Bridge Service Error Types
//! 
//! Comprehensive error handling for cross-chain bridge operations.

use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// Bridge service error types
#[derive(Debug, Error)]
pub enum BridgeError {
    // ========================================================================
    // CHAIN ERRORS
    // ========================================================================
    
    /// Unsupported chain
    #[error("Unsupported chain: {0}")]
    UnsupportedChain(String),
    
    /// Chain not available
    #[error("Chain not available: {chain}. Reason: {reason}")]
    ChainUnavailable { chain: String, reason: String },
    
    /// Chain client error
    #[error("Chain client error: {0}")]
    ChainClientError(String),
    
    /// RPC error
    #[error("RPC error on {chain}: {message}")]
    RpcError { chain: String, message: String },
    
    // ========================================================================
    // TRANSFER ERRORS
    // ========================================================================
    
    /// Transfer not found
    #[error("Transfer not found: {0}")]
    TransferNotFound(String),
    
    /// Transfer already exists
    #[error("Transfer already exists: {0}")]
    TransferAlreadyExists(String),
    
    /// Transfer in invalid state
    #[error("Transfer {transfer_id} in invalid state: expected {expected}, got {actual}")]
    InvalidTransferState {
        transfer_id: String,
        expected: String,
        actual: String,
    },
    
    /// Transfer expired
    #[error("Transfer expired: {0}")]
    TransferExpired(String),
    
    /// Transfer failed
    #[error("Transfer failed: {0}")]
    TransferFailed(String),
    
    // ========================================================================
    // HTLC ERRORS
    // ========================================================================
    
    /// HTLC not found
    #[error("HTLC not found: {0}")]
    HtlcNotFound(String),
    
    /// HTLC creation failed
    #[error("HTLC creation failed: {0}")]
    HtlcCreationFailed(String),
    
    /// HTLC already claimed
    #[error("HTLC already claimed: {0}")]
    HtlcAlreadyClaimed(String),
    
    /// HTLC already refunded
    #[error("HTLC already refunded: {0}")]
    HtlcAlreadyRefunded(String),
    
    /// HTLC not expired (cannot refund)
    #[error("HTLC not expired yet: {0}")]
    HtlcNotExpired(String),
    
    /// Invalid secret
    #[error("Invalid secret for HTLC: hash mismatch")]
    InvalidSecret,
    
    /// Secret already revealed
    #[error("Secret already revealed for HTLC: {0}")]
    SecretAlreadyRevealed(String),
    
    // ========================================================================
    // ROUTE ERRORS
    // ========================================================================
    
    /// Route not found
    #[error("No route found from {source_chain} to {destination} for {asset}")]
    RouteNotFound {
        source_chain: String,
        destination: String,
        asset: String,
    },
    
    /// Route disabled
    #[error("Route disabled: {source_chain} -> {destination}")]
    RouteDisabled { source_chain: String, destination: String },
    
    /// Insufficient liquidity
    #[error("Insufficient liquidity: available {available}, required {required}")]
    InsufficientLiquidity { available: String, required: String },
    
    // ========================================================================
    // AMOUNT ERRORS
    // ========================================================================
    
    /// Amount too small
    #[error("Amount too small: minimum is {minimum}, got {amount}")]
    AmountTooSmall { minimum: String, amount: String },
    
    /// Amount too large
    #[error("Amount too large: maximum is {maximum}, got {amount}")]
    AmountTooLarge { maximum: String, amount: String },
    
    /// Invalid amount
    #[error("Invalid amount: {0}")]
    InvalidAmount(String),
    
    // ========================================================================
    // ASSET ERRORS
    // ========================================================================
    
    /// Asset not supported
    #[error("Asset not supported: {asset} on chain {chain}")]
    AssetNotSupported { asset: String, chain: String },
    
    /// Asset paused
    #[error("Asset temporarily paused: {0}")]
    AssetPaused(String),
    
    // ========================================================================
    // ADDRESS ERRORS
    // ========================================================================
    
    /// Invalid address
    #[error("Invalid address for chain {chain}: {address}")]
    InvalidAddress { chain: String, address: String },
    
    /// Sanctioned address
    #[error("Address is sanctioned and cannot be used")]
    SanctionedAddress,
    
    // ========================================================================
    // RATE LIMIT ERRORS
    // ========================================================================
    
    /// Rate limited
    #[error("Rate limited: {message}")]
    RateLimited { message: String },
    
    /// Daily limit exceeded
    #[error("Daily limit exceeded: limit is {limit}, used {used}")]
    DailyLimitExceeded { limit: String, used: String },
    
    // ========================================================================
    // VALIDATION ERRORS
    // ========================================================================
    
    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    /// Invalid parameter
    #[error("Invalid parameter {param}: {message}")]
    InvalidParameter { param: String, message: String },
    
    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(String),
    
    // ========================================================================
    // QUOTE ERRORS
    // ========================================================================
    
    /// Quote expired
    #[error("Quote expired: {0}")]
    QuoteExpired(String),
    
    /// Quote not found
    #[error("Quote not found: {0}")]
    QuoteNotFound(String),
    
    /// Price changed
    #[error("Price changed beyond slippage tolerance")]
    PriceChanged,
    
    // ========================================================================
    // RELAYER ERRORS
    // ========================================================================
    
    /// No relayer available
    #[error("No relayer available for chain: {0}")]
    NoRelayerAvailable(String),
    
    /// Relayer error
    #[error("Relayer error: {0}")]
    RelayerError(String),
    
    // ========================================================================
    // INFRASTRUCTURE ERRORS
    // ========================================================================
    
    /// Database error
    #[error("Database error: {0}")]
    DatabaseError(#[from] sqlx::Error),
    
    /// Redis error
    #[error("Cache error: {0}")]
    CacheError(#[from] redis::RedisError),
    
    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),
    
    /// Service unavailable
    #[error("Service temporarily unavailable")]
    ServiceUnavailable,
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
    
    // ========================================================================
    // TRANSACTION ERRORS
    // ========================================================================
    
    /// Transaction failed
    #[error("Transaction failed on {chain}: {reason}")]
    TransactionFailed { chain: String, reason: String },
    
    /// Transaction timeout
    #[error("Transaction timeout on {chain}: {tx_hash}")]
    TransactionTimeout { chain: String, tx_hash: String },
    
    /// Transaction reverted
    #[error("Transaction reverted: {0}")]
    TransactionReverted(String),
    
    /// Insufficient gas
    #[error("Insufficient gas for transaction")]
    InsufficientGas,
    
    // ========================================================================
    // AUTHENTICATION ERRORS
    // ========================================================================
    
    /// Unauthorized
    #[error("Unauthorized: {0}")]
    Unauthorized(String),
    
    /// Forbidden
    #[error("Forbidden: {0}")]
    Forbidden(String),
}

impl BridgeError {
    /// Get HTTP status code for this error
    pub fn status_code(&self) -> StatusCode {
        match self {
            // 400 Bad Request
            Self::ValidationError(_)
            | Self::InvalidParameter { .. }
            | Self::MissingField(_)
            | Self::InvalidAmount(_)
            | Self::AmountTooSmall { .. }
            | Self::AmountTooLarge { .. }
            | Self::InvalidAddress { .. }
            | Self::InvalidSecret => StatusCode::BAD_REQUEST,
            
            // 401 Unauthorized
            Self::Unauthorized(_) => StatusCode::UNAUTHORIZED,
            
            // 403 Forbidden
            Self::Forbidden(_)
            | Self::SanctionedAddress => StatusCode::FORBIDDEN,
            
            // 404 Not Found
            Self::TransferNotFound(_)
            | Self::HtlcNotFound(_)
            | Self::RouteNotFound { .. }
            | Self::QuoteNotFound(_) => StatusCode::NOT_FOUND,
            
            // 409 Conflict
            Self::TransferAlreadyExists(_)
            | Self::InvalidTransferState { .. }
            | Self::HtlcAlreadyClaimed(_)
            | Self::HtlcAlreadyRefunded(_)
            | Self::SecretAlreadyRevealed(_) => StatusCode::CONFLICT,
            
            // 410 Gone
            Self::TransferExpired(_)
            | Self::QuoteExpired(_)
            | Self::HtlcNotExpired(_) => StatusCode::GONE,
            
            // 422 Unprocessable Entity
            Self::RouteDisabled { .. }
            | Self::AssetNotSupported { .. }
            | Self::AssetPaused(_)
            | Self::UnsupportedChain(_)
            | Self::PriceChanged => StatusCode::UNPROCESSABLE_ENTITY,
            
            // 429 Too Many Requests
            Self::RateLimited { .. }
            | Self::DailyLimitExceeded { .. } => StatusCode::TOO_MANY_REQUESTS,
            
            // 500 Internal Server Error
            Self::DatabaseError(_)
            | Self::CacheError(_)
            | Self::InternalError(_)
            | Self::ConfigError(_)
            | Self::ChainClientError(_) => StatusCode::INTERNAL_SERVER_ERROR,
            
            // 502 Bad Gateway
            Self::RpcError { .. }
            | Self::TransactionFailed { .. }
            | Self::RelayerError(_) => StatusCode::BAD_GATEWAY,
            
            // 503 Service Unavailable
            Self::ChainUnavailable { .. }
            | Self::ServiceUnavailable
            | Self::NoRelayerAvailable(_)
            | Self::InsufficientLiquidity { .. } => StatusCode::SERVICE_UNAVAILABLE,
            
            // 504 Gateway Timeout
            Self::TransactionTimeout { .. } => StatusCode::GATEWAY_TIMEOUT,
            
            // Default to 500
            Self::TransferFailed(_)
            | Self::HtlcCreationFailed(_)
            | Self::TransactionReverted(_)
            | Self::InsufficientGas => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
    
    /// Get error code for this error
    pub fn error_code(&self) -> &'static str {
        match self {
            Self::UnsupportedChain(_) => "UNSUPPORTED_CHAIN",
            Self::ChainUnavailable { .. } => "CHAIN_UNAVAILABLE",
            Self::ChainClientError(_) => "CHAIN_CLIENT_ERROR",
            Self::RpcError { .. } => "RPC_ERROR",
            Self::TransferNotFound(_) => "TRANSFER_NOT_FOUND",
            Self::TransferAlreadyExists(_) => "TRANSFER_EXISTS",
            Self::InvalidTransferState { .. } => "INVALID_TRANSFER_STATE",
            Self::TransferExpired(_) => "TRANSFER_EXPIRED",
            Self::TransferFailed(_) => "TRANSFER_FAILED",
            Self::HtlcNotFound(_) => "HTLC_NOT_FOUND",
            Self::HtlcCreationFailed(_) => "HTLC_CREATION_FAILED",
            Self::HtlcAlreadyClaimed(_) => "HTLC_ALREADY_CLAIMED",
            Self::HtlcAlreadyRefunded(_) => "HTLC_ALREADY_REFUNDED",
            Self::HtlcNotExpired(_) => "HTLC_NOT_EXPIRED",
            Self::InvalidSecret => "INVALID_SECRET",
            Self::SecretAlreadyRevealed(_) => "SECRET_REVEALED",
            Self::RouteNotFound { .. } => "ROUTE_NOT_FOUND",
            Self::RouteDisabled { .. } => "ROUTE_DISABLED",
            Self::InsufficientLiquidity { .. } => "INSUFFICIENT_LIQUIDITY",
            Self::AmountTooSmall { .. } => "AMOUNT_TOO_SMALL",
            Self::AmountTooLarge { .. } => "AMOUNT_TOO_LARGE",
            Self::InvalidAmount(_) => "INVALID_AMOUNT",
            Self::AssetNotSupported { .. } => "ASSET_NOT_SUPPORTED",
            Self::AssetPaused(_) => "ASSET_PAUSED",
            Self::InvalidAddress { .. } => "INVALID_ADDRESS",
            Self::SanctionedAddress => "SANCTIONED_ADDRESS",
            Self::RateLimited { .. } => "RATE_LIMITED",
            Self::DailyLimitExceeded { .. } => "DAILY_LIMIT_EXCEEDED",
            Self::ValidationError(_) => "VALIDATION_ERROR",
            Self::InvalidParameter { .. } => "INVALID_PARAMETER",
            Self::MissingField(_) => "MISSING_FIELD",
            Self::QuoteExpired(_) => "QUOTE_EXPIRED",
            Self::QuoteNotFound(_) => "QUOTE_NOT_FOUND",
            Self::PriceChanged => "PRICE_CHANGED",
            Self::NoRelayerAvailable(_) => "NO_RELAYER",
            Self::RelayerError(_) => "RELAYER_ERROR",
            Self::DatabaseError(_) => "DATABASE_ERROR",
            Self::CacheError(_) => "CACHE_ERROR",
            Self::InternalError(_) => "INTERNAL_ERROR",
            Self::ServiceUnavailable => "SERVICE_UNAVAILABLE",
            Self::ConfigError(_) => "CONFIG_ERROR",
            Self::TransactionFailed { .. } => "TRANSACTION_FAILED",
            Self::TransactionTimeout { .. } => "TRANSACTION_TIMEOUT",
            Self::TransactionReverted(_) => "TRANSACTION_REVERTED",
            Self::InsufficientGas => "INSUFFICIENT_GAS",
            Self::Unauthorized(_) => "UNAUTHORIZED",
            Self::Forbidden(_) => "FORBIDDEN",
        }
    }
}

impl IntoResponse for BridgeError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let error_code = self.error_code();
        let message = self.to_string();
        
        let body = json!({
            "success": false,
            "error": {
                "code": error_code,
                "message": message
            },
            "timestamp": chrono::Utc::now().to_rfc3339()
        });
        
        (status, Json(body)).into_response()
    }
}

/// Result type alias for bridge operations
pub type BridgeResult<T> = Result<T, BridgeError>;

// Implement From for common error types
impl From<anyhow::Error> for BridgeError {
    fn from(err: anyhow::Error) -> Self {
        BridgeError::InternalError(err.to_string())
    }
}

impl From<validator::ValidationErrors> for BridgeError {
    fn from(err: validator::ValidationErrors) -> Self {
        BridgeError::ValidationError(err.to_string())
    }
}
