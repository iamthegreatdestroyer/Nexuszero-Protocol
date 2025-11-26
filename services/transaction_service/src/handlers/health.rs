//! Health check handlers

use crate::state::AppState;
use axum::{extract::Extension, http::StatusCode, Json};
use serde::Serialize;
use std::sync::Arc;

/// Health check response
#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
}

/// Readiness response
#[derive(Serialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub checks: ReadinessChecks,
}

/// Individual readiness checks
#[derive(Serialize)]
pub struct ReadinessChecks {
    pub database: bool,
    pub redis: bool,
    pub privacy_service: bool,
    pub compliance_service: bool,
}

/// Health check endpoint
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        service: "nexuszero-transaction-service".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Readiness check endpoint
pub async fn readiness_check(
    Extension(state): Extension<Arc<AppState>>,
) -> (StatusCode, Json<ReadinessResponse>) {
    let db_healthy = state.check_db_health().await;
    let redis_healthy = state.check_redis_health().await;
    let privacy_healthy = state.check_privacy_service_health().await;
    let compliance_healthy = state.check_compliance_service_health().await;

    let all_ready = db_healthy && redis_healthy;

    let status = if all_ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };

    (
        status,
        Json(ReadinessResponse {
            ready: all_ready,
            checks: ReadinessChecks {
                database: db_healthy,
                redis: redis_healthy,
                privacy_service: privacy_healthy,
                compliance_service: compliance_healthy,
            },
        }),
    )
}
