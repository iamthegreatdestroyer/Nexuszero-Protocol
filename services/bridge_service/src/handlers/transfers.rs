//! Transfer handlers for bridge operations

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
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
use crate::handlers::metrics;

/// Initiate a new bridge transfer
pub async fn initiate_transfer(
    State(state): State<AppState>,
    Json(request): Json<InitiateTransferRequest>,
) -> BridgeResult<Json<ApiResponse<InitiateTransferResponse>>> {
    // Validate request
    request.validate().map_err(BridgeError::from)?;
    
    // Check if bridge is paused
    if state.config.security.paused {
        return Err(BridgeError::ServiceUnavailable);
    }
    
    // Validate chains
    if !state.config.is_route_supported(&request.source_chain, &request.destination_chain) {
        return Err(BridgeError::RouteNotFound {
            source_chain: request.source_chain.clone(),
            destination: request.destination_chain.clone(),
            asset: request.asset_symbol.clone(),
        });
    }
    
    // Get route
    let repo = BridgeRepository::new(state.db.clone());
    let route = repo
        .get_route(&request.source_chain, &request.destination_chain, &request.asset_symbol)
        .await?
        .ok_or(BridgeError::RouteNotFound {
            source_chain: request.source_chain.clone(),
            destination: request.destination_chain.clone(),
            asset: request.asset_symbol.clone(),
        })?;
    
    // Validate amount
    if request.amount < route.min_amount {
        return Err(BridgeError::AmountTooSmall {
            minimum: route.min_amount.to_string(),
            amount: request.amount.to_string(),
        });
    }
    if request.amount > route.max_amount {
        return Err(BridgeError::AmountTooLarge {
            maximum: route.max_amount.to_string(),
            amount: request.amount.to_string(),
        });
    }
    
    // Check liquidity
    if request.amount > route.liquidity_available {
        return Err(BridgeError::InsufficientLiquidity {
            available: route.liquidity_available.to_string(),
            required: request.amount.to_string(),
        });
    }
    
    // Calculate fees
    let amount_usd = request.amount.to_string().parse::<f64>().unwrap_or(0.0);
    let bridge_fee = state.fee_calculator.calculate_bridge_fee(
        &request.source_chain,
        amount_usd,
    );
    let gas_fee = state.fee_calculator.estimate_gas_fee(&request.destination_chain).await;
    
    // Calculate receive amount
    let total_fees = bridge_fee + gas_fee;
    let receive_amount = request.amount - total_fees;
    
    // Generate deposit address (placeholder - would come from chain client)
    let deposit_address = format!("0x{:064x}", rand::random::<u64>());
    
    // Create transfer
    let user_id = Uuid::new_v4(); // Would come from auth
    let transfer = repo
        .create_transfer(
            user_id,
            &request.source_chain,
            &deposit_address,
            &request.destination_chain,
            &request.destination_address,
            &request.asset_symbol,
            None,
            request.amount,
            bridge_fee,
            TransferDirection::Lock,
        )
        .await?;
    
    // Calculate expiry
    let expires_at = chrono::Utc::now() + chrono::Duration::hours(1);
    
    // Update metrics
    metrics::TRANSFERS_ACTIVE
        .with_label_values(&[&request.source_chain, &request.destination_chain])
        .inc();
    
    let response = InitiateTransferResponse {
        transfer_id: transfer.transfer_id,
        deposit_address,
        deposit_amount: request.amount,
        bridge_fee,
        estimated_gas_fee: gas_fee,
        receive_amount,
        expires_at,
        instructions: format!(
            "Send {} {} to the deposit address within 1 hour to complete your transfer",
            request.amount, request.asset_symbol
        ),
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Get transfer by ID
pub async fn get_transfer(
    State(state): State<AppState>,
    Path(transfer_id): Path<String>,
) -> BridgeResult<Json<ApiResponse<BridgeTransfer>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let transfer = repo
        .get_transfer_by_transfer_id(&transfer_id)
        .await?
        .ok_or(BridgeError::TransferNotFound(transfer_id))?;
    
    Ok(Json(ApiResponse::success(transfer)))
}

/// List user transfers
#[derive(Debug, Deserialize)]
pub struct ListTransfersQuery {
    #[serde(default = "default_page")]
    pub page: i32,
    #[serde(default = "default_page_size")]
    pub page_size: i32,
}

fn default_page() -> i32 { 1 }
fn default_page_size() -> i32 { 20 }

pub async fn list_transfers(
    State(state): State<AppState>,
    Query(query): Query<ListTransfersQuery>,
) -> BridgeResult<Json<ApiResponse<PaginatedResponse<BridgeTransfer>>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let user_id = Uuid::new_v4(); // Would come from auth
    let (transfers, total) = repo.get_user_transfers(user_id, query.page, query.page_size).await?;
    
    let paginated = PaginatedResponse::new(transfers, total, query.page, query.page_size);
    
    Ok(Json(ApiResponse::success(paginated)))
}

/// Cancel a pending transfer
pub async fn cancel_transfer(
    State(state): State<AppState>,
    Path(transfer_id): Path<String>,
) -> BridgeResult<Json<ApiResponse<BridgeTransfer>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let transfer = repo
        .get_transfer_by_transfer_id(&transfer_id)
        .await?
        .ok_or(BridgeError::TransferNotFound(transfer_id.clone()))?;
    
    // Can only cancel pending transfers
    if transfer.status != TransferStatus::Pending {
        return Err(BridgeError::InvalidTransferState {
            transfer_id,
            expected: "pending".to_string(),
            actual: format!("{:?}", transfer.status),
        });
    }
    
    let updated = repo
        .update_transfer_status(transfer.id, TransferStatus::Cancelled, None)
        .await?;
    
    // Update metrics
    metrics::TRANSFERS_ACTIVE
        .with_label_values(&[&transfer.source_chain, &transfer.destination_chain])
        .dec();
    
    Ok(Json(ApiResponse::success(updated)))
}

/// Retry a failed transfer
#[derive(Debug, Deserialize)]
pub struct RetryTransferRequest {
    pub increase_gas: Option<bool>,
}

pub async fn retry_transfer(
    State(state): State<AppState>,
    Path(transfer_id): Path<String>,
    Json(request): Json<RetryTransferRequest>,
) -> BridgeResult<Json<ApiResponse<BridgeTransfer>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let transfer = repo
        .get_transfer_by_transfer_id(&transfer_id)
        .await?
        .ok_or(BridgeError::TransferNotFound(transfer_id.clone()))?;
    
    // Can only retry failed transfers
    if transfer.status != TransferStatus::Failed {
        return Err(BridgeError::InvalidTransferState {
            transfer_id,
            expected: "failed".to_string(),
            actual: format!("{:?}", transfer.status),
        });
    }
    
    // Check retry count
    if transfer.retry_count >= 3 {
        return Err(BridgeError::TransferFailed(
            "Maximum retry attempts exceeded".to_string()
        ));
    }
    
    // Reset status to claiming
    let updated = repo
        .update_transfer_status(transfer.id, TransferStatus::Claiming, None)
        .await?;
    
    // TODO: Trigger relay task with increased gas if requested
    
    Ok(Json(ApiResponse::success(updated)))
}

/// Get transfer history statistics
#[derive(Debug, Serialize)]
pub struct TransferHistoryStats {
    pub total_transfers: i64,
    pub completed_transfers: i64,
    pub total_volume_usd: Decimal,
    pub average_completion_time_secs: i64,
    pub success_rate: f64,
}

pub async fn get_transfer_stats(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<TransferHistoryStats>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let stats = repo.get_transfer_stats().await?;
    
    let success_rate = if stats.total_transfers > 0 {
        (stats.completed_transfers as f64) / (stats.total_transfers as f64) * 100.0
    } else {
        0.0
    };
    
    let history_stats = TransferHistoryStats {
        total_transfers: stats.total_transfers,
        completed_transfers: stats.completed_transfers,
        total_volume_usd: stats.total_volume_usd,
        average_completion_time_secs: 180, // TODO: Calculate from actual data
        success_rate,
    };
    
    Ok(Json(ApiResponse::success(history_stats)))
}
