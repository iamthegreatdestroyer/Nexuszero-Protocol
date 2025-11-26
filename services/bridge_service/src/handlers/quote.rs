//! Quote handlers for bridge transfers

use axum::{
    extract::{Path, State},
    Json,
};
use chrono::{Duration, Utc};
use rust_decimal::Decimal;
use uuid::Uuid;
use validator::Validate;

use crate::db::BridgeRepository;
use crate::error::{BridgeError, BridgeResult};
use crate::models::*;
use crate::state::AppState;
use crate::handlers::metrics;

/// Get a quote for a bridge transfer
pub async fn get_quote(
    State(state): State<AppState>,
    Json(request): Json<QuoteRequest>,
) -> BridgeResult<Json<ApiResponse<QuoteResponse>>> {
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
        .await?;
    
    let (route_available, liquidity_available, estimated_time_secs, fee_bps) = match route {
        Some(r) => {
            if !r.is_enabled {
                return Err(BridgeError::RouteDisabled {
                    source_chain: request.source_chain.clone(),
                    destination: request.destination_chain.clone(),
                });
            }
            (true, r.liquidity_available, r.estimated_time_secs as u32, r.fee_bps)
        }
        None => {
            // Use config defaults if no specific route
            (
                true,
                Decimal::from(1000000),
                300,
                state.config.fees.base_fee_bps as i32,
            )
        }
    };
    
    // Calculate fees
    let amount_f64 = request.amount.to_string().parse::<f64>().unwrap_or(0.0);
    
    // Bridge fee
    let bridge_fee = state.fee_calculator.calculate_bridge_fee(
        &request.source_chain,
        amount_f64,
    );
    let bridge_fee_usd = bridge_fee; // Assuming 1:1 for simplicity
    
    // Gas fee estimate
    let gas_fee = state.fee_calculator.estimate_gas_fee(&request.destination_chain).await;
    let gas_fee_usd = gas_fee * Decimal::from(2000); // ETH price placeholder
    
    // Total fees
    let total_fees = bridge_fee + gas_fee;
    let total_fees_usd = bridge_fee_usd + gas_fee_usd;
    
    // Output amount
    let output_amount = request.amount - bridge_fee;
    let output_amount_usd = output_amount; // Placeholder
    
    // Quote validity
    let valid_until = Utc::now() + Duration::minutes(5);
    let quote_id = format!("QT-{}", Uuid::new_v4().to_string().split('-').next().unwrap().to_uppercase());
    
    // Record metrics
    metrics::QUOTES_REQUESTED
        .with_label_values(&[&request.source_chain, &request.destination_chain])
        .inc();
    
    // Cache quote in Redis
    let mut redis = state.redis.clone();
    let quote_key = format!("{}quote:{}", state.config.redis.key_prefix, quote_id);
    let quote_data = serde_json::json!({
        "source_chain": request.source_chain,
        "destination_chain": request.destination_chain,
        "asset_symbol": request.asset_symbol,
        "input_amount": request.amount.to_string(),
        "output_amount": output_amount.to_string(),
        "bridge_fee": bridge_fee.to_string(),
        "gas_fee": gas_fee.to_string(),
        "valid_until": valid_until.to_rfc3339()
    });
    
    let _: Result<(), _> = redis::cmd("SETEX")
        .arg(&quote_key)
        .arg(300) // 5 minutes
        .arg(quote_data.to_string())
        .query_async(&mut redis)
        .await;
    
    let response = QuoteResponse {
        source_chain: request.source_chain,
        destination_chain: request.destination_chain,
        asset_symbol: request.asset_symbol,
        input_amount: request.amount,
        bridge_fee,
        bridge_fee_usd,
        gas_fee_estimate: gas_fee,
        gas_fee_usd,
        total_fees,
        total_fees_usd,
        output_amount,
        output_amount_usd,
        estimated_time_secs,
        route_available,
        liquidity_available,
        valid_until,
        quote_id,
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Execute a quote (initiate transfer)
pub async fn execute_quote(
    State(state): State<AppState>,
    Path(quote_id): Path<String>,
    Json(request): Json<ExecuteQuoteRequest>,
) -> BridgeResult<Json<ApiResponse<InitiateTransferResponse>>> {
    request.validate().map_err(BridgeError::from)?;
    
    // Retrieve quote from Redis
    let mut redis = state.redis.clone();
    let quote_key = format!("{}quote:{}", state.config.redis.key_prefix, quote_id);
    
    let quote_data: Option<String> = redis::cmd("GET")
        .arg(&quote_key)
        .query_async(&mut redis)
        .await
        .map_err(|_| BridgeError::QuoteNotFound(quote_id.clone()))?;
    
    let quote_data = quote_data
        .ok_or(BridgeError::QuoteNotFound(quote_id.clone()))?;
    
    let quote: serde_json::Value = serde_json::from_str(&quote_data)
        .map_err(|_| BridgeError::InternalError("Failed to parse quote".to_string()))?;
    
    // Check if quote is still valid
    let valid_until = quote["valid_until"]
        .as_str()
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
        .ok_or(BridgeError::QuoteExpired(quote_id.clone()))?;
    
    if Utc::now() > valid_until.with_timezone(&Utc) {
        return Err(BridgeError::QuoteExpired(quote_id.clone()));
    }
    
    // Get quote details
    let source_chain = quote["source_chain"].as_str().unwrap().to_string();
    let destination_chain = quote["destination_chain"].as_str().unwrap().to_string();
    let asset_symbol = quote["asset_symbol"].as_str().unwrap().to_string();
    let input_amount: Decimal = quote["input_amount"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap_or_default();
    let bridge_fee: Decimal = quote["bridge_fee"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap_or_default();
    let gas_fee: Decimal = quote["gas_fee"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap_or_default();
    let output_amount: Decimal = quote["output_amount"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap_or_default();
    
    // Create transfer
    let repo = BridgeRepository::new(state.db.clone());
    let user_id = Uuid::new_v4(); // Would come from auth
    
    // Generate deposit address
    let deposit_address = format!("0x{:064x}", rand::random::<u64>());
    
    let transfer = repo
        .create_transfer(
            user_id,
            &source_chain,
            &deposit_address,
            &destination_chain,
            &request.destination_address,
            &asset_symbol,
            None,
            input_amount,
            bridge_fee,
            TransferDirection::Lock,
        )
        .await?;
    
    // Delete quote from Redis (one-time use)
    let _: Result<(), _> = redis::cmd("DEL")
        .arg(&quote_key)
        .query_async(&mut redis)
        .await;
    
    // Record metrics
    metrics::QUOTES_EXECUTED
        .with_label_values(&[&source_chain, &destination_chain])
        .inc();
    
    let expires_at = Utc::now() + Duration::hours(1);
    
    let response = InitiateTransferResponse {
        transfer_id: transfer.transfer_id,
        deposit_address,
        deposit_amount: input_amount,
        bridge_fee,
        estimated_gas_fee: gas_fee,
        receive_amount: output_amount,
        expires_at,
        instructions: format!(
            "Send {} {} to the deposit address within 1 hour",
            input_amount, asset_symbol
        ),
    };
    
    Ok(Json(ApiResponse::success(response)))
}

/// Request to execute a quote
#[derive(Debug, serde::Deserialize, Validate)]
pub struct ExecuteQuoteRequest {
    /// Destination address
    #[validate(length(min = 1, max = 255))]
    pub destination_address: String,
}

/// Compare quotes across different routes
#[derive(Debug, serde::Deserialize, Validate)]
pub struct CompareQuotesRequest {
    pub source_chain: String,
    pub destination_chain: String,
    pub asset_symbol: String,
    pub amount: Decimal,
}

#[derive(Debug, serde::Serialize)]
pub struct QuoteComparison {
    pub quotes: Vec<QuoteResponse>,
    pub best_quote_id: Option<String>,
    pub best_output_amount: Option<Decimal>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

pub async fn compare_quotes(
    State(state): State<AppState>,
    Json(request): Json<CompareQuotesRequest>,
) -> BridgeResult<Json<ApiResponse<QuoteComparison>>> {
    request.validate().map_err(BridgeError::from)?;
    
    // For now, just return a single quote
    // In a full implementation, this would compare across multiple DEXs/bridges
    
    let quote_request = QuoteRequest {
        source_chain: request.source_chain.clone(),
        destination_chain: request.destination_chain.clone(),
        asset_symbol: request.asset_symbol.clone(),
        amount: request.amount,
    };
    
    let quote_response = get_quote(State(state), Json(quote_request)).await?;
    let quote = quote_response.0.data;
    
    let comparison = QuoteComparison {
        quotes: vec![quote.clone()],
        best_quote_id: Some(quote.quote_id.clone()),
        best_output_amount: Some(quote.output_amount),
        generated_at: Utc::now(),
    };
    
    Ok(Json(ApiResponse::success(comparison)))
}
