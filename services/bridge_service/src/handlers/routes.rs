//! Route handlers

use axum::{
    extract::{Path, State},
    Json,
};
use serde::Serialize;

use crate::db::BridgeRepository;
use crate::error::{BridgeError, BridgeResult};
use crate::models::*;
use crate::state::AppState;

/// Get all available routes
pub async fn list_routes(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<Vec<RouteInfo>>>> {
    let repo = BridgeRepository::new(state.db.clone());
    let routes = repo.get_all_routes().await?;
    
    let route_infos: Vec<RouteInfo> = routes
        .into_iter()
        .map(|r| RouteInfo {
            id: r.id.to_string(),
            source_chain: r.source_chain,
            destination_chain: r.destination_chain,
            asset_symbol: r.asset_symbol,
            is_enabled: r.is_enabled,
            fee_bps: r.fee_bps,
            min_amount: r.min_amount.to_string(),
            max_amount: r.max_amount.to_string(),
            estimated_time_secs: r.estimated_time_secs,
            liquidity_available: r.liquidity_available.to_string(),
        })
        .collect();
    
    Ok(Json(ApiResponse::success(route_infos)))
}

/// Route information for API
#[derive(Debug, Serialize)]
pub struct RouteInfo {
    pub id: String,
    pub source_chain: String,
    pub destination_chain: String,
    pub asset_symbol: String,
    pub is_enabled: bool,
    pub fee_bps: i32,
    pub min_amount: String,
    pub max_amount: String,
    pub estimated_time_secs: i32,
    pub liquidity_available: String,
}

/// Get routes from a specific chain
pub async fn get_routes_from_chain(
    State(state): State<AppState>,
    Path(source_chain): Path<String>,
) -> BridgeResult<Json<ApiResponse<Vec<RouteInfo>>>> {
    let repo = BridgeRepository::new(state.db.clone());
    let all_routes = repo.get_all_routes().await?;
    
    let routes: Vec<RouteInfo> = all_routes
        .into_iter()
        .filter(|r| r.source_chain == source_chain && r.is_enabled)
        .map(|r| RouteInfo {
            id: r.id.to_string(),
            source_chain: r.source_chain,
            destination_chain: r.destination_chain,
            asset_symbol: r.asset_symbol,
            is_enabled: r.is_enabled,
            fee_bps: r.fee_bps,
            min_amount: r.min_amount.to_string(),
            max_amount: r.max_amount.to_string(),
            estimated_time_secs: r.estimated_time_secs,
            liquidity_available: r.liquidity_available.to_string(),
        })
        .collect();
    
    Ok(Json(ApiResponse::success(routes)))
}

/// Get specific route
pub async fn get_route(
    State(state): State<AppState>,
    Path((source, destination, asset)): Path<(String, String, String)>,
) -> BridgeResult<Json<ApiResponse<RouteInfo>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let route = repo
        .get_route(&source, &destination, &asset)
        .await?
        .ok_or(BridgeError::RouteNotFound {
            source_chain: source.clone(),
            destination: destination.clone(),
            asset: asset.clone(),
        })?;
    
    let route_info = RouteInfo {
        id: route.id.to_string(),
        source_chain: route.source_chain,
        destination_chain: route.destination_chain,
        asset_symbol: route.asset_symbol,
        is_enabled: route.is_enabled,
        fee_bps: route.fee_bps,
        min_amount: route.min_amount.to_string(),
        max_amount: route.max_amount.to_string(),
        estimated_time_secs: route.estimated_time_secs,
        liquidity_available: route.liquidity_available.to_string(),
    };
    
    Ok(Json(ApiResponse::success(route_info)))
}

/// Check route availability
#[derive(Debug, Serialize)]
pub struct RouteAvailability {
    pub source_chain: String,
    pub destination_chain: String,
    pub asset_symbol: String,
    pub is_available: bool,
    pub reason: Option<String>,
    pub liquidity_available: Option<String>,
    pub estimated_time_secs: Option<i32>,
}

pub async fn check_route_availability(
    State(state): State<AppState>,
    Path((source, destination, asset)): Path<(String, String, String)>,
) -> BridgeResult<Json<ApiResponse<RouteAvailability>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let route = repo.get_route(&source, &destination, &asset).await?;
    
    let availability = match route {
        Some(r) if r.is_enabled => RouteAvailability {
            source_chain: source,
            destination_chain: destination,
            asset_symbol: asset,
            is_available: true,
            reason: None,
            liquidity_available: Some(r.liquidity_available.to_string()),
            estimated_time_secs: Some(r.estimated_time_secs),
        },
        Some(_) => RouteAvailability {
            source_chain: source,
            destination_chain: destination,
            asset_symbol: asset,
            is_available: false,
            reason: Some("Route is temporarily disabled".to_string()),
            liquidity_available: None,
            estimated_time_secs: None,
        },
        None => RouteAvailability {
            source_chain: source,
            destination_chain: destination,
            asset_symbol: asset,
            is_available: false,
            reason: Some("Route not found".to_string()),
            liquidity_available: None,
            estimated_time_secs: None,
        },
    };
    
    Ok(Json(ApiResponse::success(availability)))
}

/// Get route matrix (all source-destination combinations)
#[derive(Debug, Serialize)]
pub struct RouteMatrix {
    pub chains: Vec<String>,
    pub routes: Vec<Vec<bool>>,
}

pub async fn get_route_matrix(
    State(state): State<AppState>,
) -> BridgeResult<Json<ApiResponse<RouteMatrix>>> {
    let repo = BridgeRepository::new(state.db.clone());
    
    let chains = repo.get_all_chains().await?;
    let routes = repo.get_all_routes().await?;
    
    let chain_ids: Vec<String> = chains.iter().map(|c| c.chain_id.clone()).collect();
    let chain_count = chain_ids.len();
    
    // Create matrix
    let mut matrix: Vec<Vec<bool>> = vec![vec![false; chain_count]; chain_count];
    
    for route in routes {
        if route.is_enabled {
            if let (Some(src_idx), Some(dst_idx)) = (
                chain_ids.iter().position(|c| c == &route.source_chain),
                chain_ids.iter().position(|c| c == &route.destination_chain),
            ) {
                matrix[src_idx][dst_idx] = true;
            }
        }
    }
    
    let route_matrix = RouteMatrix {
        chains: chain_ids,
        routes: matrix,
    };
    
    Ok(Json(ApiResponse::success(route_matrix)))
}
