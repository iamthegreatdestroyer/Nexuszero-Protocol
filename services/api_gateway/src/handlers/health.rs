//! Health Check Handlers
//!
//! Provides liveness and readiness probes for Kubernetes/container orchestration

use crate::state::AppState;
use axum::{extract::Extension, http::StatusCode, Json};
use serde::Serialize;
use std::sync::Arc;

/// Health response
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Readiness response with detailed component status
#[derive(Debug, Serialize)]
pub struct ReadinessResponse {
    pub status: String,
    pub components: ComponentsHealth,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize)]
pub struct ComponentsHealth {
    pub database: ComponentStatus,
    pub redis: ComponentStatus,
    pub services: ServicesStatus,
}

#[derive(Debug, Serialize)]
pub struct ComponentStatus {
    pub status: String,
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
pub struct ServicesStatus {
    pub transaction: ComponentStatus,
    pub privacy: ComponentStatus,
    pub compliance: ComponentStatus,
    pub bridge: ComponentStatus,
}

/// Liveness probe - just checks if the service is running
pub async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: chrono::Utc::now(),
    })
}

/// Readiness probe - checks all dependencies
pub async fn readiness_check(
    Extension(state): Extension<Arc<AppState>>,
) -> Result<Json<ReadinessResponse>, (StatusCode, Json<ReadinessResponse>)> {
    let start = std::time::Instant::now();

    // Check database
    let db_start = std::time::Instant::now();
    let db_healthy = state.check_db_health().await;
    let db_latency = db_start.elapsed().as_millis() as u64;

    // Check Redis
    let redis_start = std::time::Instant::now();
    let redis_healthy = state.check_redis_health().await;
    let redis_latency = redis_start.elapsed().as_millis() as u64;

    // Check downstream services
    let services_health = state.check_services_health().await;

    let all_healthy = db_healthy && redis_healthy;
    let status = if all_healthy { "ready" } else { "not_ready" };

    let response = ReadinessResponse {
        status: status.to_string(),
        components: ComponentsHealth {
            database: ComponentStatus {
                status: if db_healthy { "healthy" } else { "unhealthy" }.to_string(),
                latency_ms: Some(db_latency),
            },
            redis: ComponentStatus {
                status: if redis_healthy { "healthy" } else { "unhealthy" }.to_string(),
                latency_ms: Some(redis_latency),
            },
            services: ServicesStatus {
                transaction: ComponentStatus {
                    status: if services_health.transaction {
                        "healthy"
                    } else {
                        "unhealthy"
                    }
                    .to_string(),
                    latency_ms: None,
                },
                privacy: ComponentStatus {
                    status: if services_health.privacy {
                        "healthy"
                    } else {
                        "unhealthy"
                    }
                    .to_string(),
                    latency_ms: None,
                },
                compliance: ComponentStatus {
                    status: if services_health.compliance {
                        "healthy"
                    } else {
                        "unhealthy"
                    }
                    .to_string(),
                    latency_ms: None,
                },
                bridge: ComponentStatus {
                    status: if services_health.bridge {
                        "healthy"
                    } else {
                        "unhealthy"
                    }
                    .to_string(),
                    latency_ms: None,
                },
            },
        },
        timestamp: chrono::Utc::now(),
    };

    tracing::debug!(
        "Readiness check completed in {}ms: {}",
        start.elapsed().as_millis(),
        status
    );

    if all_healthy {
        Ok(Json(response))
    } else {
        Err((StatusCode::SERVICE_UNAVAILABLE, Json(response)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_check() {
        let response = health_check().await;
        assert_eq!(response.status, "healthy");
        assert!(!response.version.is_empty());
    }
}
