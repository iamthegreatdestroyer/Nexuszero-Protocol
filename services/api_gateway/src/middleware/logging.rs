//! Request Logging Middleware
//!
//! Logs all incoming requests with relevant metadata

use axum::{
    body::Body,
    http::Request,
    middleware::Next,
    response::Response,
};
use std::time::Instant;

/// Request logging middleware
pub async fn request_logger(
    request: Request<Body>,
    next: Next,
) -> Response {
    let start = Instant::now();
    let method = request.method().clone();
    let uri = request.uri().clone();
    let version = request.version();

    // Get request ID if present
    let request_id = request
        .headers()
        .get("x-request-id")
        .and_then(|h| h.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Get user agent
    let user_agent = request
        .headers()
        .get("user-agent")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("unknown")
        .to_string();

    // Process the request
    let response = next.run(request).await;

    let duration = start.elapsed();
    let status = response.status();

    // Log the request
    tracing::info!(
        target: "http_access",
        request_id = %request_id,
        method = %method,
        uri = %uri,
        version = ?version,
        status = %status.as_u16(),
        duration_ms = %duration.as_millis(),
        user_agent = %user_agent,
        "Request completed"
    );

    // Log slow requests as warnings
    if duration.as_millis() > 1000 {
        tracing::warn!(
            request_id = %request_id,
            method = %method,
            uri = %uri,
            duration_ms = %duration.as_millis(),
            "Slow request detected"
        );
    }

    // Log errors
    if status.is_server_error() {
        tracing::error!(
            request_id = %request_id,
            method = %method,
            uri = %uri,
            status = %status.as_u16(),
            "Server error response"
        );
    } else if status.is_client_error() {
        tracing::debug!(
            request_id = %request_id,
            method = %method,
            uri = %uri,
            status = %status.as_u16(),
            "Client error response"
        );
    }

    response
}

/// Structured log entry for request
#[derive(Debug, serde::Serialize)]
pub struct RequestLog {
    pub request_id: String,
    pub method: String,
    pub uri: String,
    pub status: u16,
    pub duration_ms: u128,
    pub user_agent: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl RequestLog {
    pub fn new(
        request_id: String,
        method: String,
        uri: String,
        status: u16,
        duration_ms: u128,
        user_agent: String,
    ) -> Self {
        Self {
            request_id,
            method,
            uri,
            status,
            duration_ms,
            user_agent,
            timestamp: chrono::Utc::now(),
        }
    }
}
