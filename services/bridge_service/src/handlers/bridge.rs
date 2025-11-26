//! Bridge admin/management handlers

use axum::{
    extract::{Path, State},
    Json,
};
use serde::{Deserialize, Serialize};

use crate::db::BridgeRepository;
use crate::error::{BridgeError, BridgeResult};
use crate::models::*;
use crate::state::AppState;

/// Get bridge status
pub async fn get_bridge_status(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<BridgeStatus>>> {
    let repo = BridgeRepository::new(state.db.clone());
    let stats = repo.get_transfer_stats().await?;
    
    let enabled_chains: Vec<String> = state.config
        .enabled_chains()
        .iter()
        .map(|(id, _)| (*id).clone())
        .collect();
    
    let status = BridgeStatus {
        is_paused: state.config.security.paused,
        version: env!("CARGO_PKG_VERSION").to_string(),
        enabled_chains,
        total_transfers: stats.total_transfers,
        pending_transfers: stats.pending_transfers,
        total_volume_usd: stats.total_volume_usd.to_string(),
        total_fees_collected: stats.total_fees_collected.to_string(),
        uptime_seconds: 0, // TODO: Track actual uptime
    };
    
    Ok(Json(ApiResponse::success(status)))
}

/// Bridge status
#[derive(Debug, Serialize)]
pub struct BridgeStatus {
    pub is_paused: bool,
    pub version: String,
    pub enabled_chains: Vec<String>,
    pub total_transfers: i64,
    pub pending_transfers: i64,
    pub total_volume_usd: String,
    pub total_fees_collected: String,
    pub uptime_seconds: u64,
}

/// Pause/unpause bridge
#[derive(Debug, Deserialize)]
pub struct PauseBridgeRequest {
    pub paused: bool,
    pub reason: Option<String>,
}

pub async fn set_bridge_paused(
    State(_state): State<AppState>,
    Json(request): Json<PauseBridgeRequest>,
) -> BridgeResult<Json<ApiResponse<BridgePauseResponse>>> {
    // TODO: In production, this would require admin auth and update config
    
    let response = BridgePauseResponse {
        paused: request.paused,
        reason: request.reason,
        effective_at: chrono::Utc::now(),
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Pause response
#[derive(Debug, Serialize)]
pub struct BridgePauseResponse {
    pub paused: bool,
    pub reason: Option<String>,
    pub effective_at: chrono::DateTime<chrono::Utc>,
}

/// Get fee configuration
pub async fn get_fee_config(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<FeeConfigResponse>>> {
    let config = FeeConfigResponse {
        base_fee_bps: state.config.fees.base_fee_bps,
        min_fee_usd: state.config.fees.min_fee_usd,
        max_fee_usd: state.config.fees.max_fee_usd,
        dynamic_fees_enabled: state.config.fees.dynamic_fees_enabled,
        fee_recipient: state.config.fees.fee_recipient.clone(),
    };
    
    Ok(Json(ApiResponse::success(config)))
}

/// Fee configuration response
#[derive(Debug, Serialize)]
pub struct FeeConfigResponse {
    pub base_fee_bps: u32,
    pub min_fee_usd: f64,
    pub max_fee_usd: f64,
    pub dynamic_fees_enabled: bool,
    pub fee_recipient: String,
}

/// Update fee configuration
#[derive(Debug, Deserialize)]
pub struct UpdateFeeConfigRequest {
    pub base_fee_bps: Option<u32>,
    pub min_fee_usd: Option<f64>,
    pub max_fee_usd: Option<f64>,
    pub dynamic_fees_enabled: Option<bool>,
}

pub async fn update_fee_config(
    State(_state): State<AppState>,
    Json(request): Json<UpdateFeeConfigRequest>,
) -> BridgeResult<Json<ApiResponse<FeeConfigResponse>>> {
    // TODO: In production, this would require admin auth and update config
    
    let config = FeeConfigResponse {
        base_fee_bps: request.base_fee_bps.unwrap_or(30),
        min_fee_usd: request.min_fee_usd.unwrap_or(1.0),
        max_fee_usd: request.max_fee_usd.unwrap_or(1000.0),
        dynamic_fees_enabled: request.dynamic_fees_enabled.unwrap_or(true),
        fee_recipient: "0x...".to_string(),
    };
    
    Ok(Json(ApiResponse::success(config)))
}

/// Get security limits
pub async fn get_security_limits(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<SecurityLimits>>> {
    let limits = SecurityLimits {
        max_transfer_usd: state.config.security.max_transfer_usd,
        daily_limit_per_user_usd: state.config.security.daily_limit_per_user_usd,
        rate_limit_window_secs: state.config.security.rate_limit_window_secs,
        max_transfers_per_window: state.config.security.max_transfers_per_window,
        sanctions_screening_enabled: state.config.security.sanctions_screening_enabled,
    };
    
    Ok(Json(ApiResponse::success(limits)))
}

/// Security limits
#[derive(Debug, Serialize)]
pub struct SecurityLimits {
    pub max_transfer_usd: f64,
    pub daily_limit_per_user_usd: f64,
    pub rate_limit_window_secs: u64,
    pub max_transfers_per_window: u32,
    pub sanctions_screening_enabled: bool,
}

/// Get relayer status
pub async fn get_relayer_status(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<RelayerStatusResponse>>> {
    let status = RelayerStatusResponse {
        auto_relay_enabled: state.config.relayer.auto_relay_enabled,
        min_profit_threshold_usd: state.config.relayer.min_profit_threshold_usd,
        max_pending_per_chain: state.config.relayer.max_pending_per_chain,
        tx_timeout_secs: state.config.relayer.tx_timeout_secs,
        retry_attempts: state.config.relayer.retry_attempts,
    };
    
    Ok(Json(ApiResponse::success(status)))
}

/// Relayer status
#[derive(Debug, Serialize)]
pub struct RelayerStatusResponse {
    pub auto_relay_enabled: bool,
    pub min_profit_threshold_usd: f64,
    pub max_pending_per_chain: u32,
    pub tx_timeout_secs: u64,
    pub retry_attempts: u32,
}

/// Enable/disable chain
#[derive(Debug, Deserialize)]
pub struct SetChainEnabledRequest {
    pub enabled: bool,
    pub reason: Option<String>,
}

pub async fn set_chain_enabled(
    State(_state): State<AppState>,
    Path(chain_id): Path<String>,
    Json(request): Json<SetChainEnabledRequest>,
) -> BridgeResult<Json<ApiResponse<ChainEnabledResponse>>> {
    // TODO: In production, update database and config
    
    let response = ChainEnabledResponse {
        chain_id,
        enabled: request.enabled,
        reason: request.reason,
        effective_at: chrono::Utc::now(),
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Chain enabled response
#[derive(Debug, Serialize)]
pub struct ChainEnabledResponse {
    pub chain_id: String,
    pub enabled: bool,
    pub reason: Option<String>,
    pub effective_at: chrono::DateTime<chrono::Utc>,
}
