//! Health check handlers

use axum::{extract::State, http::StatusCode, Json};
use serde_json::{json, Value};
use crate::state::AppState;

/// Liveness probe - basic service health
pub async fn liveness() -> StatusCode {
    StatusCode::OK
}

/// Readiness probe - service ready to accept traffic
pub async fn readiness(State(state): State<AppState>) -> (StatusCode, Json<Value>) {
    let mut checks = vec![];
    let mut all_healthy = true;
    
    // Check database
    let db_healthy = sqlx::query("SELECT 1")
        .execute(&state.db)
        .await
        .is_ok();
    checks.push(json!({
        "name": "database",
        "healthy": db_healthy
    }));
    if !db_healthy {
        all_healthy = false;
    }
    
    // Check Redis
    let mut redis = state.redis.clone();
    let redis_healthy: Result<String, _> = redis::cmd("PING")
        .query_async(&mut redis)
        .await;
    let redis_healthy = redis_healthy.is_ok();
    checks.push(json!({
        "name": "redis",
        "healthy": redis_healthy
    }));
    if !redis_healthy {
        all_healthy = false;
    }
    
    // Check if bridge is paused
    let bridge_active = !state.config.security.paused;
    checks.push(json!({
        "name": "bridge_active",
        "healthy": bridge_active
    }));
    
    let status = if all_healthy {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    
    let response = json!({
        "status": if all_healthy { "healthy" } else { "unhealthy" },
        "service": "bridge-service",
        "version": env!("CARGO_PKG_VERSION"),
        "checks": checks,
        "timestamp": chrono::Utc::now().to_rfc3339()
    });
    
    (status, Json(response))
}

/// Detailed health information
pub async fn health_detail(State(state): State<AppState>) -> Json<Value> {
    let enabled_chains: Vec<String> = state.config
        .enabled_chains()
        .iter()
        .map(|(id, _)| (*id).clone())
        .collect();
    
    Json(json!({
        "service": "bridge-service",
        "version": env!("CARGO_PKG_VERSION"),
        "status": "operational",
        "bridge_paused": state.config.security.paused,
        "enabled_chains": enabled_chains,
        "htlc_config": {
            "default_timelock_secs": state.config.htlc.default_timelock_secs,
            "min_timelock_secs": state.config.htlc.min_timelock_secs,
            "max_timelock_secs": state.config.htlc.max_timelock_secs
        },
        "fee_config": {
            "base_fee_bps": state.config.fees.base_fee_bps,
            "min_fee_usd": state.config.fees.min_fee_usd,
            "max_fee_usd": state.config.fees.max_fee_usd
        },
        "security_config": {
            "max_transfer_usd": state.config.security.max_transfer_usd,
            "daily_limit_per_user_usd": state.config.security.daily_limit_per_user_usd,
            "sanctions_screening_enabled": state.config.security.sanctions_screening_enabled
        },
        "timestamp": chrono::Utc::now().to_rfc3339()
    }))
}
