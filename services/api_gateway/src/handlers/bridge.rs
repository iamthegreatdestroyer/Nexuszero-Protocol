//! Bridge Handlers
//!
//! Handles cross-chain bridge operations

use crate::error::{ApiError, ApiResult};
use crate::middleware::auth::AuthenticatedUser;
use crate::state::AppState;
use axum::{
    extract::{Extension, Path, Query},
    Json,
};
use serde::{Deserialize, Serialize};
use sqlx::FromRow;
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

/// Supported blockchain networks
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Chain {
    Ethereum,
    Polygon,
    Arbitrum,
    Optimism,
    Solana,
    Bitcoin,
    Cosmos,
}

impl std::fmt::Display for Chain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Chain::Ethereum => write!(f, "ethereum"),
            Chain::Polygon => write!(f, "polygon"),
            Chain::Arbitrum => write!(f, "arbitrum"),
            Chain::Optimism => write!(f, "optimism"),
            Chain::Solana => write!(f, "solana"),
            Chain::Bitcoin => write!(f, "bitcoin"),
            Chain::Cosmos => write!(f, "cosmos"),
        }
    }
}

/// Bridge quote request
#[derive(Debug, Deserialize, Validate)]
pub struct QuoteRequest {
    /// Source chain
    pub source_chain: Chain,

    /// Target chain
    pub target_chain: Chain,

    /// Amount in smallest unit
    pub amount: u64,

    /// Token symbol or address
    #[validate(length(min = 1, max = 100))]
    pub token: String,

    /// Requested privacy level for bridge
    pub privacy_level: Option<u8>,
}

/// Bridge quote response
#[derive(Debug, Serialize)]
pub struct QuoteResponse {
    pub quote_id: String,
    pub source_chain: Chain,
    pub target_chain: Chain,
    pub input_amount: u64,
    pub output_amount: u64,
    pub fee_amount: u64,
    pub fee_percentage: f64,
    pub privacy_level: u8,
    pub estimated_time_seconds: u64,
    pub expires_at: chrono::DateTime<chrono::Utc>,
    pub route: Vec<RouteStep>,
}

#[derive(Debug, Serialize)]
pub struct RouteStep {
    pub chain: Chain,
    pub action: String,
    pub estimated_time_seconds: u64,
}

/// Initiate transfer request
#[derive(Debug, Deserialize, Validate)]
pub struct InitiateTransferRequest {
    /// Quote ID from previous quote request
    pub quote_id: Uuid,

    /// Recipient address on target chain
    #[validate(length(min = 1, max = 255))]
    pub recipient_address: String,

    /// Privacy level
    pub privacy_level: Option<u8>,

    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
}

/// Transfer response
#[derive(Debug, Serialize)]
pub struct TransferResponse {
    pub transfer_id: String,
    pub quote_id: String,
    pub source_chain: Chain,
    pub target_chain: Chain,
    pub status: TransferStatus,
    pub source_tx_hash: Option<String>,
    pub target_tx_hash: Option<String>,
    pub privacy_level: u8,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransferStatus {
    Initiated,
    SourcePending,
    SourceConfirmed,
    BridgeLocked,
    BridgeProcessing,
    TargetPending,
    TargetConfirmed,
    Completed,
    Failed,
    Refunding,
    Refunded,
}

/// Transfer history query
#[derive(Debug, Deserialize)]
pub struct TransferHistoryQuery {
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub status: Option<String>,
    pub source_chain: Option<Chain>,
    pub target_chain: Option<Chain>,
}

/// Transfer history response
#[derive(Debug, Serialize)]
pub struct TransferHistoryResponse {
    pub transfers: Vec<TransferSummary>,
    pub total: i64,
    pub page: u32,
    pub limit: u32,
}

#[derive(Debug, Serialize)]
pub struct TransferSummary {
    pub transfer_id: String,
    pub source_chain: Chain,
    pub target_chain: Chain,
    pub amount: u64,
    pub status: TransferStatus,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Database record for bridge transfer
#[derive(Debug, FromRow)]
struct BridgeTransferRecord {
    id: Uuid,
    quote_id: Uuid,
    source_chain: String,
    target_chain: String,
    status: String,
    source_tx_hash: Option<Vec<u8>>,
    target_tx_hash: Option<Vec<u8>>,
    privacy_level: i32,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Database record for transfer summary
#[derive(Debug, FromRow)]
struct TransferSummaryRecord {
    id: Uuid,
    source_chain: String,
    target_chain: String,
    input_amount: i64,
    status: String,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Supported chains response
#[derive(Debug, Serialize)]
pub struct SupportedChainsResponse {
    pub chains: Vec<ChainInfo>,
    pub pairs: Vec<ChainPair>,
}

#[derive(Debug, Serialize)]
pub struct ChainInfo {
    pub chain: Chain,
    pub name: String,
    pub native_token: String,
    pub confirmations_required: u32,
    pub average_block_time_ms: u64,
    pub privacy_support: bool,
    pub max_privacy_level: u8,
}

#[derive(Debug, Serialize)]
pub struct ChainPair {
    pub source: Chain,
    pub target: Chain,
    pub min_amount: u64,
    pub max_amount: u64,
    pub fee_percentage: f64,
    pub estimated_time_seconds: u64,
}

/// Get bridge quote handler
pub async fn get_quote(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<QuoteRequest>,
) -> ApiResult<Json<QuoteResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    // Check if pair is supported
    if payload.source_chain == payload.target_chain {
        return Err(ApiError::BadRequest(
            "Source and target chains must be different".to_string(),
        ));
    }

    if !is_pair_supported(&payload.source_chain, &payload.target_chain) {
        return Err(ApiError::UnsupportedChain(format!(
            "{} -> {}",
            payload.source_chain, payload.target_chain
        )));
    }

    let privacy_level = payload.privacy_level.unwrap_or(3);
    if privacy_level > 5 {
        return Err(ApiError::UnsupportedPrivacyLevel(privacy_level));
    }

    // Calculate fees and amounts
    let fee_percentage = get_fee_percentage(&payload.source_chain, &payload.target_chain, privacy_level);
    let fee_amount = ((payload.amount as f64) * fee_percentage / 100.0) as u64;
    let output_amount = payload.amount - fee_amount;

    // Estimate time
    let estimated_time = estimate_transfer_time(&payload.source_chain, &payload.target_chain);

    // Generate route
    let route = generate_route(&payload.source_chain, &payload.target_chain);

    let quote_id = Uuid::new_v4();
    let expires_at = chrono::Utc::now() + chrono::Duration::minutes(5);

    // Store quote in Redis for quick retrieval
    if let Ok(mut conn) = state.redis_conn().await {
        let quote_data = serde_json::json!({
            "source_chain": payload.source_chain,
            "target_chain": payload.target_chain,
            "input_amount": payload.amount,
            "output_amount": output_amount,
            "fee_amount": fee_amount,
            "privacy_level": privacy_level,
            "token": payload.token,
            "expires_at": expires_at.to_rfc3339()
        });

        let _: Result<(), _> = redis::cmd("SETEX")
            .arg(format!("bridge:quote:{}", quote_id))
            .arg(300) // 5 minutes
            .arg(quote_data.to_string())
            .query_async(&mut conn)
            .await;
    }

    tracing::info!(
        quote_id = %quote_id,
        source = %payload.source_chain,
        target = %payload.target_chain,
        amount = payload.amount,
        "Bridge quote generated"
    );

    Ok(Json(QuoteResponse {
        quote_id: quote_id.to_string(),
        source_chain: payload.source_chain,
        target_chain: payload.target_chain,
        input_amount: payload.amount,
        output_amount,
        fee_amount,
        fee_percentage,
        privacy_level,
        estimated_time_seconds: estimated_time,
        expires_at,
        route,
    }))
}

/// Initiate bridge transfer handler
pub async fn initiate_transfer(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Json(payload): Json<InitiateTransferRequest>,
) -> ApiResult<Json<TransferResponse>> {
    // Validate input
    payload.validate().map_err(|e| ApiError::ValidationError(e.to_string()))?;

    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    // Retrieve and validate quote from Redis
    let mut conn = state.redis_conn().await?;
    let quote_key = format!("bridge:quote:{}", payload.quote_id);
    let quote_data: Option<String> = redis::cmd("GET")
        .arg(&quote_key)
        .query_async(&mut conn)
        .await?;

    let quote: serde_json::Value = quote_data
        .ok_or(ApiError::NotFound("Quote not found or expired".to_string()))?
        .parse()
        .map_err(|_| ApiError::InternalError)?;

    // Extract quote details
    let source_chain: Chain = serde_json::from_value(quote["source_chain"].clone())
        .map_err(|_| ApiError::InternalError)?;
    let target_chain: Chain = serde_json::from_value(quote["target_chain"].clone())
        .map_err(|_| ApiError::InternalError)?;
    let input_amount = quote["input_amount"]
        .as_u64()
        .ok_or(ApiError::InternalError)?;
    let privacy_level = quote["privacy_level"]
        .as_u64()
        .ok_or(ApiError::InternalError)? as u8;

    let transfer_id = Uuid::new_v4();
    let now = chrono::Utc::now();

    // Create bridge transfer record
    sqlx::query(
        r#"
        INSERT INTO bridge_transfers (
            id, user_id, quote_id, source_chain, target_chain,
            input_amount, recipient_address, privacy_level, status,
            metadata, created_at, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 'initiated', $9, $10, $10)
        "#
    )
    .bind(transfer_id)
    .bind(user_id)
    .bind(payload.quote_id)
    .bind(source_chain.to_string())
    .bind(target_chain.to_string())
    .bind(input_amount as i64)
    .bind(&payload.recipient_address)
    .bind(privacy_level as i32)
    .bind(payload.metadata.clone().unwrap_or(serde_json::json!({})))
    .bind(now)
    .execute(&state.db)
    .await?;

    // Request bridge initiation from bridge service
    let _ = initiate_bridge_transfer(&state, &transfer_id, &source_chain, &target_chain).await;

    // Delete used quote
    let _: Result<(), _> = redis::cmd("DEL")
        .arg(&quote_key)
        .query_async(&mut conn)
        .await;

    // Record metrics
    crate::handlers::metrics::record_bridge_transfer(
        &source_chain.to_string(),
        &target_chain.to_string(),
        false,
    );

    tracing::info!(
        transfer_id = %transfer_id,
        source = %source_chain,
        target = %target_chain,
        amount = input_amount,
        "Bridge transfer initiated"
    );

    Ok(Json(TransferResponse {
        transfer_id: transfer_id.to_string(),
        quote_id: payload.quote_id.to_string(),
        source_chain,
        target_chain,
        status: TransferStatus::Initiated,
        source_tx_hash: None,
        target_tx_hash: None,
        privacy_level,
        created_at: now,
    }))
}

/// Get transfer status handler
pub async fn get_status(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Path(id): Path<Uuid>,
) -> ApiResult<Json<TransferResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let transfer: BridgeTransferRecord = sqlx::query_as(
        r#"
        SELECT 
            id, quote_id, source_chain, target_chain, status,
            source_tx_hash, target_tx_hash, privacy_level, created_at
        FROM bridge_transfers
        WHERE id = $1 AND user_id = $2
        "#
    )
    .bind(id)
    .bind(user_id)
    .fetch_optional(&state.db)
    .await?
    .ok_or(ApiError::NotFound("Transfer not found".to_string()))?;

    Ok(Json(TransferResponse {
        transfer_id: transfer.id.to_string(),
        quote_id: transfer.quote_id.to_string(),
        source_chain: parse_chain(&transfer.source_chain),
        target_chain: parse_chain(&transfer.target_chain),
        status: parse_transfer_status(&transfer.status),
        source_tx_hash: transfer.source_tx_hash.map(|h| hex::encode(&h)),
        target_tx_hash: transfer.target_tx_hash.map(|h| hex::encode(&h)),
        privacy_level: transfer.privacy_level as u8,
        created_at: transfer.created_at,
    }))
}

/// Get transfer history handler
pub async fn get_history(
    Extension(state): Extension<Arc<AppState>>,
    Extension(user): Extension<AuthenticatedUser>,
    Query(params): Query<TransferHistoryQuery>,
) -> ApiResult<Json<TransferHistoryResponse>> {
    let user_id = Uuid::parse_str(&user.user_id)
        .map_err(|_| ApiError::BadRequest("Invalid user ID".to_string()))?;

    let page = params.page.unwrap_or(1).max(1);
    let limit = params.limit.unwrap_or(20).min(100);
    let offset = ((page - 1) * limit) as i64;

    let transfers: Vec<TransferSummaryRecord> = sqlx::query_as(
        r#"
        SELECT id, source_chain, target_chain, input_amount, status, created_at
        FROM bridge_transfers
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT $2 OFFSET $3
        "#
    )
    .bind(user_id)
    .bind(limit as i64)
    .bind(offset)
    .fetch_all(&state.db)
    .await?;

    let total: Option<i64> = sqlx::query_scalar(
        "SELECT COUNT(*) FROM bridge_transfers WHERE user_id = $1"
    )
    .bind(user_id)
    .fetch_one(&state.db)
    .await?;

    let transfer_summaries: Vec<TransferSummary> = transfers
        .into_iter()
        .map(|t| TransferSummary {
            transfer_id: t.id.to_string(),
            source_chain: parse_chain(&t.source_chain),
            target_chain: parse_chain(&t.target_chain),
            amount: t.input_amount as u64,
            status: parse_transfer_status(&t.status),
            created_at: t.created_at,
        })
        .collect();

    Ok(Json(TransferHistoryResponse {
        transfers: transfer_summaries,
        total: total.unwrap_or(0),
        page,
        limit,
    }))
}

/// Get supported chains handler
pub async fn get_supported_chains() -> Json<SupportedChainsResponse> {
    Json(SupportedChainsResponse {
        chains: vec![
            ChainInfo {
                chain: Chain::Ethereum,
                name: "Ethereum Mainnet".to_string(),
                native_token: "ETH".to_string(),
                confirmations_required: 12,
                average_block_time_ms: 12000,
                privacy_support: true,
                max_privacy_level: 5,
            },
            ChainInfo {
                chain: Chain::Polygon,
                name: "Polygon".to_string(),
                native_token: "MATIC".to_string(),
                confirmations_required: 128,
                average_block_time_ms: 2000,
                privacy_support: true,
                max_privacy_level: 5,
            },
            ChainInfo {
                chain: Chain::Arbitrum,
                name: "Arbitrum One".to_string(),
                native_token: "ETH".to_string(),
                confirmations_required: 1,
                average_block_time_ms: 250,
                privacy_support: true,
                max_privacy_level: 4,
            },
            ChainInfo {
                chain: Chain::Optimism,
                name: "Optimism".to_string(),
                native_token: "ETH".to_string(),
                confirmations_required: 1,
                average_block_time_ms: 2000,
                privacy_support: true,
                max_privacy_level: 4,
            },
            ChainInfo {
                chain: Chain::Solana,
                name: "Solana".to_string(),
                native_token: "SOL".to_string(),
                confirmations_required: 32,
                average_block_time_ms: 400,
                privacy_support: true,
                max_privacy_level: 3,
            },
            ChainInfo {
                chain: Chain::Bitcoin,
                name: "Bitcoin".to_string(),
                native_token: "BTC".to_string(),
                confirmations_required: 6,
                average_block_time_ms: 600000,
                privacy_support: true,
                max_privacy_level: 2,
            },
            ChainInfo {
                chain: Chain::Cosmos,
                name: "Cosmos Hub".to_string(),
                native_token: "ATOM".to_string(),
                confirmations_required: 1,
                average_block_time_ms: 7000,
                privacy_support: true,
                max_privacy_level: 3,
            },
        ],
        pairs: vec![
            ChainPair {
                source: Chain::Ethereum,
                target: Chain::Polygon,
                min_amount: 1_000_000, // 0.001 ETH in wei (10^6)
                max_amount: 10_000_000_000_000_000_000, // 10 ETH in wei (10^19, fits in u64)
                fee_percentage: 0.1,
                estimated_time_seconds: 900,
            },
            ChainPair {
                source: Chain::Polygon,
                target: Chain::Ethereum,
                min_amount: 1_000_000_000_000_000_000, // 1 MATIC
                max_amount: 10_000_000_000_000_000_000, // 10 MATIC in wei
                fee_percentage: 0.1,
                estimated_time_seconds: 1800,
            },
            ChainPair {
                source: Chain::Ethereum,
                target: Chain::Arbitrum,
                min_amount: 1_000_000,
                max_amount: 10_000_000_000_000_000_000, // 10 ETH in wei
                fee_percentage: 0.05,
                estimated_time_seconds: 600,
            },
            ChainPair {
                source: Chain::Ethereum,
                target: Chain::Solana,
                min_amount: 10_000_000,
                max_amount: 10_000_000_000_000_000_000, // 10 ETH in wei
                fee_percentage: 0.2,
                estimated_time_seconds: 1200,
            },
        ],
    })
}

// Helper functions

fn is_pair_supported(source: &Chain, target: &Chain) -> bool {
    let supported_pairs = vec![
        (Chain::Ethereum, Chain::Polygon),
        (Chain::Polygon, Chain::Ethereum),
        (Chain::Ethereum, Chain::Arbitrum),
        (Chain::Arbitrum, Chain::Ethereum),
        (Chain::Ethereum, Chain::Optimism),
        (Chain::Optimism, Chain::Ethereum),
        (Chain::Ethereum, Chain::Solana),
        (Chain::Solana, Chain::Ethereum),
        (Chain::Polygon, Chain::Arbitrum),
        (Chain::Arbitrum, Chain::Polygon),
    ];

    supported_pairs.contains(&(source.clone(), target.clone()))
}

fn get_fee_percentage(source: &Chain, target: &Chain, privacy_level: u8) -> f64 {
    let base_fee = match (source, target) {
        (Chain::Ethereum, Chain::Polygon) | (Chain::Polygon, Chain::Ethereum) => 0.1,
        (Chain::Ethereum, Chain::Arbitrum) | (Chain::Arbitrum, Chain::Ethereum) => 0.05,
        (Chain::Ethereum, Chain::Optimism) | (Chain::Optimism, Chain::Ethereum) => 0.05,
        (Chain::Ethereum, Chain::Solana) | (Chain::Solana, Chain::Ethereum) => 0.2,
        _ => 0.15,
    };

    // Privacy level increases fees
    let privacy_multiplier = 1.0 + (privacy_level as f64 * 0.02);
    base_fee * privacy_multiplier
}

fn estimate_transfer_time(source: &Chain, target: &Chain) -> u64 {
    match (source, target) {
        (Chain::Ethereum, Chain::Polygon) => 900,
        (Chain::Polygon, Chain::Ethereum) => 1800,
        (Chain::Ethereum, Chain::Arbitrum) => 600,
        (Chain::Arbitrum, Chain::Ethereum) => 1200,
        (Chain::Ethereum, Chain::Solana) => 1200,
        (Chain::Solana, Chain::Ethereum) => 2400,
        _ => 3600,
    }
}

fn generate_route(source: &Chain, target: &Chain) -> Vec<RouteStep> {
    vec![
        RouteStep {
            chain: source.clone(),
            action: "Lock funds in bridge contract".to_string(),
            estimated_time_seconds: 60,
        },
        RouteStep {
            chain: source.clone(),
            action: "Wait for confirmations".to_string(),
            estimated_time_seconds: estimate_transfer_time(source, target) / 2,
        },
        RouteStep {
            chain: target.clone(),
            action: "Release funds on target chain".to_string(),
            estimated_time_seconds: estimate_transfer_time(source, target) / 2,
        },
    ]
}

fn parse_chain(s: &str) -> Chain {
    match s.to_lowercase().as_str() {
        "ethereum" => Chain::Ethereum,
        "polygon" => Chain::Polygon,
        "arbitrum" => Chain::Arbitrum,
        "optimism" => Chain::Optimism,
        "solana" => Chain::Solana,
        "bitcoin" => Chain::Bitcoin,
        "cosmos" => Chain::Cosmos,
        _ => Chain::Ethereum,
    }
}

fn parse_transfer_status(s: &str) -> TransferStatus {
    match s.to_lowercase().as_str() {
        "initiated" => TransferStatus::Initiated,
        "source_pending" => TransferStatus::SourcePending,
        "source_confirmed" => TransferStatus::SourceConfirmed,
        "bridge_locked" => TransferStatus::BridgeLocked,
        "bridge_processing" => TransferStatus::BridgeProcessing,
        "target_pending" => TransferStatus::TargetPending,
        "target_confirmed" => TransferStatus::TargetConfirmed,
        "completed" => TransferStatus::Completed,
        "failed" => TransferStatus::Failed,
        "refunding" => TransferStatus::Refunding,
        "refunded" => TransferStatus::Refunded,
        _ => TransferStatus::Initiated,
    }
}

async fn initiate_bridge_transfer(
    state: &AppState,
    transfer_id: &Uuid,
    source: &Chain,
    target: &Chain,
) -> Result<(), ApiError> {
    let url = format!(
        "{}/transfers/initiate",
        state.config.services.bridge_service
    );

    let _ = state
        .http_client
        .post(&url)
        .json(&serde_json::json!({
            "transfer_id": transfer_id.to_string(),
            "source_chain": source,
            "target_chain": target
        }))
        .send()
        .await;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_pair_supported() {
        assert!(is_pair_supported(&Chain::Ethereum, &Chain::Polygon));
        assert!(is_pair_supported(&Chain::Polygon, &Chain::Ethereum));
        assert!(!is_pair_supported(&Chain::Bitcoin, &Chain::Cosmos));
    }

    #[test]
    fn test_get_fee_percentage() {
        let base_fee = get_fee_percentage(&Chain::Ethereum, &Chain::Polygon, 0);
        assert!((base_fee - 0.1).abs() < 0.001);

        let privacy_fee = get_fee_percentage(&Chain::Ethereum, &Chain::Polygon, 5);
        assert!(privacy_fee > base_fee);
    }

    #[test]
    fn test_parse_chain() {
        assert_eq!(parse_chain("ethereum"), Chain::Ethereum);
        assert_eq!(parse_chain("POLYGON"), Chain::Polygon);
        assert_eq!(parse_chain("unknown"), Chain::Ethereum);
    }

    #[test]
    fn test_parse_transfer_status() {
        assert!(matches!(
            parse_transfer_status("initiated"),
            TransferStatus::Initiated
        ));
        assert!(matches!(
            parse_transfer_status("completed"),
            TransferStatus::Completed
        ));
    }
}

