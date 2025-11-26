//! Chain handlers

use axum::{
    extract::{Path, State},
    Json,
};
use serde::Serialize;

use crate::db::BridgeRepository;
use crate::error::{BridgeError, BridgeResult};
use crate::models::*;
use crate::state::AppState;

/// Get all supported chains
pub async fn list_chains(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<Vec<ChainInfo>>>> {
    let repo = BridgeRepository::new(state.db.clone());
    let chains = repo.get_all_chains().await?;
    
    let chain_infos: Vec<ChainInfo> = chains
        .into_iter()
        .map(|c| ChainInfo {
            chain_id: c.chain_id,
            name: c.name,
            chain_type: c.chain_type,
            native_token: c.native_token,
            confirmations_required: c.confirmations_required,
            block_time_secs: c.block_time_secs,
            bridge_contract: c.bridge_contract,
            is_enabled: c.is_enabled,
        })
        .collect();
    
    Ok(Json(ApiResponse::success(chain_infos)))
}

/// Chain information for API response
#[derive(Debug, Serialize)]
pub struct ChainInfo {
    pub chain_id: String,
    pub name: String,
    pub chain_type: String,
    pub native_token: String,
    pub confirmations_required: i32,
    pub block_time_secs: i32,
    pub bridge_contract: Option<String>,
    pub is_enabled: bool,
}

/// Get chain details
pub async fn get_chain(
    State(state): State<AppState>,
    Path(chain_id): Path<String>,
) -> BridgeResult<Json<ApiResponse<ChainDetails>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let chain = repo
        .get_chain(&chain_id)
        .await?
        .ok_or(BridgeError::UnsupportedChain(chain_id.clone()))?;
    
    let assets = repo.get_chain_assets(&chain_id).await?;
    
    let asset_infos: Vec<AssetInfo> = assets
        .into_iter()
        .map(|a| AssetInfo {
            symbol: a.symbol,
            name: a.name,
            contract_address: a.contract_address,
            decimals: a.decimals,
            min_amount: a.min_amount.to_string(),
            max_amount: a.max_amount.to_string(),
            is_enabled: a.is_enabled,
        })
        .collect();
    
    let details = ChainDetails {
        chain_id: chain.chain_id,
        name: chain.name,
        chain_type: chain.chain_type,
        native_token: chain.native_token,
        confirmations_required: chain.confirmations_required,
        block_time_secs: chain.block_time_secs,
        bridge_contract: chain.bridge_contract,
        htlc_contract: chain.htlc_contract,
        is_enabled: chain.is_enabled,
        supported_assets: asset_infos,
    };
    
    Ok(Json(ApiResponse::success(details)))
}

/// Detailed chain information
#[derive(Debug, Serialize)]
pub struct ChainDetails {
    pub chain_id: String,
    pub name: String,
    pub chain_type: String,
    pub native_token: String,
    pub confirmations_required: i32,
    pub block_time_secs: i32,
    pub bridge_contract: Option<String>,
    pub htlc_contract: Option<String>,
    pub is_enabled: bool,
    pub supported_assets: Vec<AssetInfo>,
}

/// Asset information
#[derive(Debug, Serialize)]
pub struct AssetInfo {
    pub symbol: String,
    pub name: String,
    pub contract_address: Option<String>,
    pub decimals: i16,
    pub min_amount: String,
    pub max_amount: String,
    pub is_enabled: bool,
}

/// Get chain status (health, block height, etc.)
pub async fn get_chain_status(
    State(state): State<AppState>,
    Path(chain_id): Path<String>,
) -> BridgeResult<Json<ApiResponse<ChainStatus>>> {
    // Verify chain exists
    if state.config.get_chain(&chain_id).is_none() {
        return Err(BridgeError::UnsupportedChain(chain_id));
    }
    
    // TODO: In production, query actual chain status
    // This is a placeholder implementation
    
    let status = ChainStatus {
        chain_id: chain_id.clone(),
        name: chain_id.clone(),
        is_healthy: true,
        current_block: 18500000, // Placeholder
        last_block_time: chrono::Utc::now(),
        pending_transfers: 5,
        gas_price: Some("30 gwei".to_string()),
        native_balance: Some("10.5 ETH".to_string()),
    };
    
    Ok(Json(ApiResponse::success(status)))
}

/// Get assets for a chain
pub async fn get_chain_assets(
    State(state): State<AppState>,
    Path(chain_id): Path<String>,
) -> BridgeResult<Json<ApiResponse<Vec<AssetInfo>>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    // Verify chain exists
    let _chain = repo
        .get_chain(&chain_id)
        .await?
        .ok_or(BridgeError::UnsupportedChain(chain_id.clone()))?;
    
    let assets = repo.get_chain_assets(&chain_id).await?;
    
    let asset_infos: Vec<AssetInfo> = assets
        .into_iter()
        .map(|a| AssetInfo {
            symbol: a.symbol,
            name: a.name,
            contract_address: a.contract_address,
            decimals: a.decimals,
            min_amount: a.min_amount.to_string(),
            max_amount: a.max_amount.to_string(),
            is_enabled: a.is_enabled,
        })
        .collect();
    
    Ok(Json(ApiResponse::success(asset_infos)))
}

/// Get all chain statuses
pub async fn list_chain_statuses(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<Vec<ChainStatus>>>> {
    let mut statuses = Vec::new();
    
    for (chain_id, chain_config) in state.config.enabled_chains() {
        // TODO: Query actual chain status
        let status = ChainStatus {
            chain_id: chain_id.clone(),
            name: chain_config.name.clone(),
            is_healthy: true,
            current_block: 18500000,
            last_block_time: chrono::Utc::now(),
            pending_transfers: 0,
            gas_price: Some("30 gwei".to_string()),
            native_balance: None,
        };
        statuses.push(status);
    }
    
    Ok(Json(ApiResponse::success(statuses)))
}
