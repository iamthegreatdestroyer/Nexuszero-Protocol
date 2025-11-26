//! Bridge Service Data Models
//! 
//! Core data structures for cross-chain bridging operations.

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use uuid::Uuid;
use validator::Validate;

// ============================================================================
// BRIDGE TRANSFER MODELS
// ============================================================================

/// Bridge transfer status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "transfer_status", rename_all = "snake_case")]
pub enum TransferStatus {
    /// Transfer initiated, awaiting source chain deposit
    Pending,
    /// Deposit confirmed on source chain
    SourceConfirmed,
    /// HTLC created on source chain
    HtlcCreated,
    /// HTLC matched on destination chain
    HtlcMatched,
    /// Secret revealed, claiming in progress
    Claiming,
    /// Transfer completed successfully
    Completed,
    /// Transfer refunded due to timeout
    Refunded,
    /// Transfer failed with error
    Failed,
    /// Transfer cancelled by user
    Cancelled,
    /// Transfer expired before completion
    Expired,
}

/// Bridge transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "transfer_direction", rename_all = "snake_case")]
pub enum TransferDirection {
    /// User is locking on source, receiving on destination
    Lock,
    /// User is burning wrapped tokens, receiving on destination
    Burn,
    /// Atomic swap between chains
    AtomicSwap,
}

/// Bridge transfer record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BridgeTransfer {
    pub id: Uuid,
    
    /// Unique transfer identifier
    pub transfer_id: String,
    
    /// User initiating the transfer
    pub user_id: Uuid,
    
    // Source chain info
    pub source_chain: String,
    pub source_address: String,
    pub source_tx_hash: Option<String>,
    pub source_block_number: Option<i64>,
    pub source_confirmations: i32,
    
    // Destination chain info
    pub destination_chain: String,
    pub destination_address: String,
    pub destination_tx_hash: Option<String>,
    pub destination_block_number: Option<i64>,
    
    // Asset and amount
    pub asset_symbol: String,
    pub asset_address: Option<String>,
    pub amount: Decimal,
    pub amount_usd: Option<Decimal>,
    
    // Fees
    pub bridge_fee: Decimal,
    pub gas_fee_source: Option<Decimal>,
    pub gas_fee_destination: Option<Decimal>,
    
    // HTLC details
    pub htlc_id: Option<Uuid>,
    pub secret_hash: Option<String>,
    pub timelock_expiry: Option<DateTime<Utc>>,
    
    // Status
    pub status: TransferStatus,
    pub direction: TransferDirection,
    pub error_message: Option<String>,
    pub retry_count: i32,
    
    // Timestamps
    pub initiated_at: DateTime<Utc>,
    pub source_confirmed_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Request to initiate a bridge transfer
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct InitiateTransferRequest {
    /// Source chain identifier
    #[validate(length(min = 1, max = 50))]
    pub source_chain: String,
    
    /// Destination chain identifier
    #[validate(length(min = 1, max = 50))]
    pub destination_chain: String,
    
    /// Asset symbol to transfer
    #[validate(length(min = 1, max = 20))]
    pub asset_symbol: String,
    
    /// Amount to transfer
    pub amount: Decimal,
    
    /// Destination address
    #[validate(length(min = 1, max = 255))]
    pub destination_address: String,
    
    /// Optional: slippage tolerance in basis points
    pub slippage_tolerance_bps: Option<u32>,
}

/// Response from initiating a transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitiateTransferResponse {
    /// Transfer ID
    pub transfer_id: String,
    
    /// Deposit address on source chain
    pub deposit_address: String,
    
    /// Amount to deposit (including fees)
    pub deposit_amount: Decimal,
    
    /// Bridge fee
    pub bridge_fee: Decimal,
    
    /// Estimated gas fee on destination
    pub estimated_gas_fee: Decimal,
    
    /// Amount to receive on destination
    pub receive_amount: Decimal,
    
    /// Transfer expiry time
    pub expires_at: DateTime<Utc>,
    
    /// Instructions for user
    pub instructions: String,
}

// ============================================================================
// HTLC (Hash Time-Locked Contract) MODELS
// ============================================================================

/// HTLC status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "htlc_status", rename_all = "snake_case")]
pub enum HtlcStatus {
    /// HTLC pending creation
    Pending,
    /// HTLC created and locked
    Locked,
    /// HTLC claimed with secret
    Claimed,
    /// HTLC refunded after timeout
    Refunded,
    /// HTLC expired (can be refunded)
    Expired,
}

/// Hash Time-Locked Contract record
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Htlc {
    pub id: Uuid,
    
    /// Associated transfer ID
    pub transfer_id: Uuid,
    
    /// Chain where this HTLC exists
    pub chain: String,
    
    /// On-chain contract address
    pub contract_address: String,
    
    /// HTLC ID on chain (if applicable)
    pub onchain_htlc_id: Option<String>,
    
    // Participants
    pub sender_address: String,
    pub receiver_address: String,
    
    // Asset and amount
    pub asset: String,
    pub amount: Decimal,
    
    // Hashlock
    pub secret_hash: String,
    pub secret: Option<String>,
    
    // Timelock
    pub timelock: DateTime<Utc>,
    pub timelock_block: Option<i64>,
    
    // Transaction hashes
    pub lock_tx_hash: Option<String>,
    pub claim_tx_hash: Option<String>,
    pub refund_tx_hash: Option<String>,
    
    // Status
    pub status: HtlcStatus,
    
    // Timestamps
    pub created_at: DateTime<Utc>,
    pub locked_at: Option<DateTime<Utc>>,
    pub claimed_at: Option<DateTime<Utc>>,
    pub refunded_at: Option<DateTime<Utc>>,
    pub updated_at: DateTime<Utc>,
}

/// Request to create an HTLC
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct CreateHtlcRequest {
    /// Transfer ID this HTLC belongs to
    pub transfer_id: Uuid,
    
    /// Chain to create HTLC on
    #[validate(length(min = 1, max = 50))]
    pub chain: String,
    
    /// Receiver address
    #[validate(length(min = 1, max = 255))]
    pub receiver_address: String,
    
    /// Asset symbol
    #[validate(length(min = 1, max = 20))]
    pub asset: String,
    
    /// Amount to lock
    pub amount: Decimal,
    
    /// Timelock duration in seconds
    pub timelock_seconds: u64,
}

/// HTLC creation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateHtlcResponse {
    pub htlc_id: Uuid,
    pub secret_hash: String,
    pub timelock: DateTime<Utc>,
    pub contract_address: String,
    pub status: HtlcStatus,
}

// ============================================================================
// CHAIN MODELS
// ============================================================================

/// Supported chain information
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SupportedChain {
    pub id: Uuid,
    pub chain_id: String,
    pub name: String,
    pub chain_type: String,
    pub native_token: String,
    pub confirmations_required: i32,
    pub block_time_secs: i32,
    pub bridge_contract: Option<String>,
    pub htlc_contract: Option<String>,
    pub is_enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Chain status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainStatus {
    pub chain_id: String,
    pub name: String,
    pub is_healthy: bool,
    pub current_block: u64,
    pub last_block_time: DateTime<Utc>,
    pub pending_transfers: u32,
    pub gas_price: Option<String>,
    pub native_balance: Option<String>,
}

/// Supported asset on a chain
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct SupportedAsset {
    pub id: Uuid,
    pub chain_id: String,
    pub symbol: String,
    pub name: String,
    pub contract_address: Option<String>,
    pub decimals: i16,
    pub min_amount: Decimal,
    pub max_amount: Decimal,
    pub is_enabled: bool,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// ROUTE MODELS
// ============================================================================

/// Bridge route between two chains
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct BridgeRoute {
    pub id: Uuid,
    pub source_chain: String,
    pub destination_chain: String,
    pub asset_symbol: String,
    pub is_enabled: bool,
    pub fee_bps: i32,
    pub min_amount: Decimal,
    pub max_amount: Decimal,
    pub estimated_time_secs: i32,
    pub liquidity_available: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Quote request for a bridge transfer
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct QuoteRequest {
    /// Source chain
    #[validate(length(min = 1, max = 50))]
    pub source_chain: String,
    
    /// Destination chain
    #[validate(length(min = 1, max = 50))]
    pub destination_chain: String,
    
    /// Asset symbol
    #[validate(length(min = 1, max = 20))]
    pub asset_symbol: String,
    
    /// Amount to bridge
    pub amount: Decimal,
}

/// Quote response for a bridge transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuoteResponse {
    /// Requested quote
    pub source_chain: String,
    pub destination_chain: String,
    pub asset_symbol: String,
    pub input_amount: Decimal,
    
    /// Fees breakdown
    pub bridge_fee: Decimal,
    pub bridge_fee_usd: Decimal,
    pub gas_fee_estimate: Decimal,
    pub gas_fee_usd: Decimal,
    pub total_fees: Decimal,
    pub total_fees_usd: Decimal,
    
    /// Output
    pub output_amount: Decimal,
    pub output_amount_usd: Decimal,
    
    /// Route info
    pub estimated_time_secs: u32,
    pub route_available: bool,
    pub liquidity_available: Decimal,
    
    /// Quote validity
    pub valid_until: DateTime<Utc>,
    pub quote_id: String,
}

// ============================================================================
// LIQUIDITY MODELS
// ============================================================================

/// Liquidity pool for a chain/asset pair
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct LiquidityPool {
    pub id: Uuid,
    pub chain: String,
    pub asset_symbol: String,
    pub total_liquidity: Decimal,
    pub available_liquidity: Decimal,
    pub locked_liquidity: Decimal,
    pub utilization_rate: Decimal,
    pub apy: Decimal,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Liquidity provider position
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct LiquidityPosition {
    pub id: Uuid,
    pub user_id: Uuid,
    pub pool_id: Uuid,
    pub chain: String,
    pub asset_symbol: String,
    pub amount: Decimal,
    pub share_percentage: Decimal,
    pub rewards_earned: Decimal,
    pub rewards_claimed: Decimal,
    pub deposited_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// ============================================================================
// RELAYER MODELS
// ============================================================================

/// Relayer status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, sqlx::Type)]
#[sqlx(type_name = "relayer_status", rename_all = "snake_case")]
pub enum RelayerStatus {
    Active,
    Paused,
    Deactivated,
}

/// Bridge relayer
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Relayer {
    pub id: Uuid,
    pub address: String,
    pub chains: Vec<String>,
    pub status: RelayerStatus,
    pub stake_amount: Decimal,
    pub successful_relays: i64,
    pub failed_relays: i64,
    pub total_volume_usd: Decimal,
    pub reputation_score: Decimal,
    pub last_active_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Relay task for a transfer
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct RelayTask {
    pub id: Uuid,
    pub transfer_id: Uuid,
    pub relayer_id: Option<Uuid>,
    pub task_type: String,
    pub chain: String,
    pub status: String,
    pub tx_hash: Option<String>,
    pub gas_used: Option<Decimal>,
    pub error_message: Option<String>,
    pub attempts: i32,
    pub scheduled_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
}

// ============================================================================
// API RESPONSE WRAPPERS
// ============================================================================

/// API success response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: T,
    pub timestamp: DateTime<Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data,
            timestamp: Utc::now(),
        }
    }
}

/// API error response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    pub success: bool,
    pub error: String,
    pub error_code: String,
    pub timestamp: DateTime<Utc>,
}

impl ApiError {
    pub fn new(error: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            success: false,
            error: error.into(),
            error_code: code.into(),
            timestamp: Utc::now(),
        }
    }
}

/// Paginated response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    pub items: Vec<T>,
    pub total: i64,
    pub page: i32,
    pub page_size: i32,
    pub total_pages: i32,
}

impl<T> PaginatedResponse<T> {
    pub fn new(items: Vec<T>, total: i64, page: i32, page_size: i32) -> Self {
        let total_pages = ((total as f64) / (page_size as f64)).ceil() as i32;
        Self {
            items,
            total,
            page,
            page_size,
            total_pages,
        }
    }
}
