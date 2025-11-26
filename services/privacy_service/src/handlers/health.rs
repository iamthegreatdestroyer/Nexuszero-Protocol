//! Health check handlers

use crate::state::AppState;
use axum::{extract::Extension, http::StatusCode, Json};
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
}

#[derive(Serialize)]
pub struct ReadinessResponse {
    pub ready: bool,
    pub checks: ReadinessChecks,
    pub queue_stats: crate::state::QueueStats,
}

#[derive(Serialize)]
pub struct ReadinessChecks {
    pub database: bool,
    pub redis: bool,
}

pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        service: "nexuszero-privacy-service".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn readiness_check(
    Extension(state): Extension<Arc<AppState>>,
) -> (StatusCode, Json<ReadinessResponse>) {
    let db_healthy = state.check_db_health().await;
    let redis_healthy = state.check_redis_health().await;
    let queue_stats = state.get_queue_stats().await;

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
            },
            queue_stats,
        }),
    )
}
