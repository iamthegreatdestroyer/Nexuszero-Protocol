//! Liquidity handlers

use axum::{
    extract::{Path, Query, State},
    Json,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

use crate::db::BridgeRepository;
use crate::error::{BridgeError, BridgeResult};
use crate::models::*;
use crate::state::AppState;

/// Get liquidity pools
pub async fn list_pools(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<Vec<PoolInfo>>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    // Get pools for all enabled chains
    let mut pools = Vec::new();
    
    for (chain_id, _) in state.config.enabled_chains() {
        let assets = repo.get_chain_assets(chain_id).await?;
        
        for asset in assets {
            if let Some(pool) = repo.get_liquidity_pool(chain_id, &asset.symbol).await? {
                pools.push(PoolInfo {
                    id: pool.id.to_string(),
                    chain: pool.chain,
                    asset_symbol: pool.asset_symbol,
                    total_liquidity: pool.total_liquidity.to_string(),
                    available_liquidity: pool.available_liquidity.to_string(),
                    locked_liquidity: pool.locked_liquidity.to_string(),
                    utilization_rate: pool.utilization_rate.to_string(),
                    apy: pool.apy.to_string(),
                });
            }
        }
    }
    
    Ok(Json(ApiResponse::success(pools)))
}

/// Pool information
#[derive(Debug, Serialize)]
pub struct PoolInfo {
    pub id: String,
    pub chain: String,
    pub asset_symbol: String,
    pub total_liquidity: String,
    pub available_liquidity: String,
    pub locked_liquidity: String,
    pub utilization_rate: String,
    pub apy: String,
}

/// Get pool details
pub async fn get_pool(
    State(state): State<AppState>,
    Path((chain, asset)): Path<(String, String)>,
) -> BridgeResult<Json<ApiResponse<PoolDetails>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let pool = repo
        .get_liquidity_pool(&chain, &asset)
        .await?
        .ok_or(BridgeError::InternalError(format!(
            "Pool not found: {}/{}",
            chain, asset
        )))?;
    
    let details = PoolDetails {
        id: pool.id.to_string(),
        chain: pool.chain,
        asset_symbol: pool.asset_symbol,
        total_liquidity: pool.total_liquidity,
        available_liquidity: pool.available_liquidity,
        locked_liquidity: pool.locked_liquidity,
        utilization_rate: pool.utilization_rate,
        apy: pool.apy,
        updated_at: pool.updated_at,
    };
    
    Ok(Json(ApiResponse::success(details)))
}

/// Detailed pool information
#[derive(Debug, Serialize)]
pub struct PoolDetails {
    pub id: String,
    pub chain: String,
    pub asset_symbol: String,
    pub total_liquidity: Decimal,
    pub available_liquidity: Decimal,
    pub locked_liquidity: Decimal,
    pub utilization_rate: Decimal,
    pub apy: Decimal,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Deposit liquidity request
#[derive(Debug, Deserialize, Validate)]
pub struct DepositLiquidityRequest {
    #[validate(length(min = 1, max = 50))]
    pub chain: String,
    
    #[validate(length(min = 1, max = 20))]
    pub asset_symbol: String,
    
    pub amount: Decimal,
}

/// Deposit liquidity
pub async fn deposit_liquidity(
    State(_state): State<AppState>,
    Json(request): Json<DepositLiquidityRequest>,
) -> BridgeResult<Json<ApiResponse<DepositResponse>>> {
    request.validate().map_err(BridgeError::from)?;
    
    // TODO: In production, this would:
    // 1. Generate deposit address
    // 2. Wait for on-chain confirmation
    // 3. Credit user's LP position
    
    let response = DepositResponse {
        deposit_id: Uuid::new_v4().to_string(),
        chain: request.chain,
        asset_symbol: request.asset_symbol,
        amount: request.amount,
        deposit_address: format!("0x{:040x}", rand::random::<u64>()),
        status: "pending".to_string(),
        expires_at: chrono::Utc::now() + chrono::Duration::hours(1),
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Deposit response
#[derive(Debug, Serialize)]
pub struct DepositResponse {
    pub deposit_id: String,
    pub chain: String,
    pub asset_symbol: String,
    pub amount: Decimal,
    pub deposit_address: String,
    pub status: String,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// Withdraw liquidity request
#[derive(Debug, Deserialize, Validate)]
pub struct WithdrawLiquidityRequest {
    #[validate(length(min = 1, max = 50))]
    pub chain: String,
    
    #[validate(length(min = 1, max = 20))]
    pub asset_symbol: String,
    
    pub amount: Decimal,
    
    #[validate(length(min = 1, max = 255))]
    pub destination_address: String,
}

/// Withdraw liquidity
pub async fn withdraw_liquidity(
    State(_state): State<AppState>,
    Json(request): Json<WithdrawLiquidityRequest>,
) -> BridgeResult<Json<ApiResponse<WithdrawResponse>>> {
    request.validate().map_err(BridgeError::from)?;
    
    // TODO: In production, this would:
    // 1. Check user's LP position
    // 2. Queue withdrawal
    // 3. Process on-chain transfer
    
    let response = WithdrawResponse {
        withdrawal_id: Uuid::new_v4().to_string(),
        chain: request.chain,
        asset_symbol: request.asset_symbol,
        amount: request.amount,
        destination_address: request.destination_address,
        status: "pending".to_string(),
        estimated_completion: chrono::Utc::now() + chrono::Duration::minutes(30),
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Withdraw response
#[derive(Debug, Serialize)]
pub struct WithdrawResponse {
    pub withdrawal_id: String,
    pub chain: String,
    pub asset_symbol: String,
    pub amount: Decimal,
    pub destination_address: String,
    pub status: String,
    pub estimated_completion: chrono::DateTime<chrono::Utc>,
}

/// Get user's LP positions
#[derive(Debug, Deserialize)]
pub struct PositionsQuery {
    #[serde(default)]
    pub chain: Option<String>,
}

pub async fn get_positions(
    State(_state): State<AppState>,
    Query(query): Query<PositionsQuery>,
) -> BridgeResult<Json<ApiResponse<Vec<PositionInfo>>>> {
    // TODO: Get from database based on user auth
    
    let positions = vec![
        PositionInfo {
            id: Uuid::new_v4().to_string(),
            chain: "ethereum".to_string(),
            asset_symbol: "USDC".to_string(),
            amount: Decimal::from(10000),
            share_percentage: Decimal::from_str_exact("0.05").unwrap(),
            rewards_earned: Decimal::from(50),
            rewards_claimed: Decimal::from(25),
            deposited_at: chrono::Utc::now() - chrono::Duration::days(30),
            current_value: Decimal::from(10050),
        },
    ];
    
    let filtered: Vec<PositionInfo> = if let Some(chain) = query.chain {
        positions.into_iter().filter(|p| p.chain == chain).collect()
    } else {
        positions
    };
    
    Ok(Json(ApiResponse::success(filtered)))
}

/// Position information
#[derive(Debug, Serialize)]
pub struct PositionInfo {
    pub id: String,
    pub chain: String,
    pub asset_symbol: String,
    pub amount: Decimal,
    pub share_percentage: Decimal,
    pub rewards_earned: Decimal,
    pub rewards_claimed: Decimal,
    pub deposited_at: chrono::DateTime<chrono::Utc>,
    pub current_value: Decimal,
}

/// Claim rewards request
#[derive(Debug, Deserialize, Validate)]
pub struct ClaimRewardsRequest {
    pub position_id: Uuid,
    
    #[validate(length(min = 1, max = 255))]
    pub destination_address: String,
}

/// Claim rewards
pub async fn claim_rewards(
    State(_state): State<AppState>,
    Json(request): Json<ClaimRewardsRequest>,
) -> BridgeResult<Json<ApiResponse<ClaimResponse>>> {
    request.validate().map_err(BridgeError::from)?;
    
    // TODO: Process reward claim
    
    let response = ClaimResponse {
        claim_id: Uuid::new_v4().to_string(),
        position_id: request.position_id.to_string(),
        amount: Decimal::from(25),
        destination_address: request.destination_address,
        status: "pending".to_string(),
        tx_hash: None,
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Claim response
#[derive(Debug, Serialize)]
pub struct ClaimResponse {
    pub claim_id: String,
    pub position_id: String,
    pub amount: Decimal,
    pub destination_address: String,
    pub status: String,
    pub tx_hash: Option<String>,
}

use rust_decimal::prelude::FromStr;

impl Decimal {
    fn from_str_exact(s: &str) -> Result<Self, rust_decimal::Error> {
        Decimal::from_str(s)
    }
}
