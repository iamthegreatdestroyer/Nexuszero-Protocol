//! Health check handlers

use axum::{extract::State, http::StatusCode, Json};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::state::AppState;

/// Basic health check
pub async fn health_check() -> (StatusCode, Json<Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "status": "healthy",
            "service": "compliance_service",
            "version": env!("CARGO_PKG_VERSION")
        })),
    )
}

/// Readiness check - verifies all dependencies are available
pub async fn readiness_check(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    let is_healthy = state.is_healthy().await;
    
    if is_healthy {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ready",
                "checks": {
                    "database": "ok",
                    "redis": "ok",
                    "rule_engine": "ok"
                }
            })),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "status": "not_ready",
                "message": "One or more dependencies unavailable"
            })),
        )
    }
}

/// Liveness check
pub async fn liveness_check(State(state): State<Arc<AppState>>) -> (StatusCode, Json<Value>) {
    (
        StatusCode::OK,
        Json(json!({
            "status": "alive",
            "uptime_seconds": state.uptime_seconds()
        })),
    )
}
