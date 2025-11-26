//! Rate Limiting Middleware
//!
//! Implements token bucket rate limiting with Redis backend

use crate::error::ApiError;
use crate::state::AppState;
use axum::{
    body::Body,
    extract::ConnectInfo,
    http::Request,
    middleware::Next,
    response::Response,
    Extension,
};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Rate limiter middleware
pub async fn rate_limiter(
    Extension(state): Extension<Arc<AppState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<Body>,
    next: Next,
) -> Result<Response, ApiError> {
    // Skip rate limiting if disabled
    if !state.config.rate_limit.enabled {
        return Ok(next.run(request).await);
    }

    let ip = addr.ip().to_string();
    let window_duration = Duration::from_secs(1);
    let max_requests = state.config.rate_limit.requests_per_second;

    // Check and update rate limit
    let allowed = check_rate_limit(
        &state,
        &ip,
        max_requests,
        window_duration,
    )
    .await?;

    if !allowed {
        tracing::warn!("Rate limit exceeded for IP: {}", ip);
        return Err(ApiError::RateLimitExceeded);
    }

    // Add rate limit headers to response
    let mut response = next.run(request).await;

    response.headers_mut().insert(
        "X-RateLimit-Limit",
        max_requests.to_string().parse().unwrap(),
    );

    Ok(response)
}

/// Check and update rate limit for an IP
async fn check_rate_limit(
    state: &AppState,
    ip: &str,
    max_requests: u32,
    window: Duration,
) -> Result<bool, ApiError> {
    // Try Redis first (distributed rate limiting)
    if let Ok(allowed) = check_redis_rate_limit(state, ip, max_requests, window).await {
        return Ok(allowed);
    }

    // Fallback to in-memory rate limiting
    check_memory_rate_limit(state, ip, max_requests, window).await
}

/// Redis-based rate limiting using sliding window
async fn check_redis_rate_limit(
    state: &AppState,
    ip: &str,
    max_requests: u32,
    window: Duration,
) -> Result<bool, ApiError> {
    let mut conn = state.redis_conn().await?;
    let key = format!("ratelimit:{}", ip);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let window_ms = window.as_millis() as i64;

    // Use Redis transaction for atomic operations
    let result: (i64, ()) = redis::pipe()
        .atomic()
        // Remove old entries
        .cmd("ZREMRANGEBYSCORE")
        .arg(&key)
        .arg(0)
        .arg(now - window_ms)
        .ignore()
        // Add current request
        .cmd("ZADD")
        .arg(&key)
        .arg(now)
        .arg(now)
        .ignore()
        // Count requests in window
        .cmd("ZCARD")
        .arg(&key)
        // Set TTL
        .cmd("EXPIRE")
        .arg(&key)
        .arg(window.as_secs() as usize + 1)
        .ignore()
        .query_async(&mut conn)
        .await?;

    let count = result.0;
    Ok(count <= max_requests as i64)
}

/// In-memory rate limiting (fallback)
async fn check_memory_rate_limit(
    state: &AppState,
    ip: &str,
    max_requests: u32,
    window: Duration,
) -> Result<bool, ApiError> {
    let mut limiter = state.rate_limiter.write().await;
    let now = Instant::now();

    let entry = limiter
        .requests
        .entry(ip.to_string())
        .or_insert_with(|| crate::state::RequestCounter {
            count: 0,
            window_start: now,
        });

    // Reset window if expired
    if now.duration_since(entry.window_start) >= window {
        entry.count = 0;
        entry.window_start = now;
    }

    // Check limit
    if entry.count >= max_requests {
        return Ok(false);
    }

    // Increment counter
    entry.count += 1;
    Ok(true)
}

/// Rate limit by API key instead of IP
pub async fn rate_limit_by_key(
    state: &AppState,
    api_key: &str,
    max_requests: u32,
    window: Duration,
) -> Result<bool, ApiError> {
    let mut conn = state.redis_conn().await?;
    let key = format!("ratelimit:apikey:{}", api_key);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    let window_ms = window.as_millis() as i64;

    let result: (i64, ()) = redis::pipe()
        .atomic()
        .cmd("ZREMRANGEBYSCORE")
        .arg(&key)
        .arg(0)
        .arg(now - window_ms)
        .ignore()
        .cmd("ZADD")
        .arg(&key)
        .arg(now)
        .arg(now)
        .ignore()
        .cmd("ZCARD")
        .arg(&key)
        .cmd("EXPIRE")
        .arg(&key)
        .arg(window.as_secs() as usize + 1)
        .ignore()
        .query_async(&mut conn)
        .await?;

    Ok(result.0 <= max_requests as i64)
}

/// Get current rate limit status for debugging
pub async fn get_rate_limit_status(
    state: &AppState,
    ip: &str,
) -> Result<RateLimitStatus, ApiError> {
    let mut conn = state.redis_conn().await?;
    let key = format!("ratelimit:{}", ip);

    let count: i64 = redis::cmd("ZCARD")
        .arg(&key)
        .query_async(&mut conn)
        .await
        .unwrap_or(0);

    let ttl: i64 = redis::cmd("TTL")
        .arg(&key)
        .query_async(&mut conn)
        .await
        .unwrap_or(-1);

    Ok(RateLimitStatus {
        current_count: count as u32,
        max_requests: state.config.rate_limit.requests_per_second,
        remaining: (state.config.rate_limit.requests_per_second as i64 - count).max(0) as u32,
        reset_in_secs: ttl.max(0) as u32,
    })
}

/// Rate limit status information
#[derive(Debug, Clone, serde::Serialize)]
pub struct RateLimitStatus {
    pub current_count: u32,
    pub max_requests: u32,
    pub remaining: u32,
    pub reset_in_secs: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_status_serialize() {
        let status = RateLimitStatus {
            current_count: 50,
            max_requests: 100,
            remaining: 50,
            reset_in_secs: 30,
        };

        let json = serde_json::to_string(&status).unwrap();
        assert!(json.contains("\"current_count\":50"));
        assert!(json.contains("\"remaining\":50"));
    }
}
