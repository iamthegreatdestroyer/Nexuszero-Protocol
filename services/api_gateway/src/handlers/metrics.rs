//! Prometheus Metrics Handler
//!
//! Exposes Prometheus-compatible metrics endpoint

use axum::response::IntoResponse;
use prometheus::{Encoder, TextEncoder, Counter, Gauge, Histogram, HistogramOpts, Registry};
use lazy_static::lazy_static;

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Request metrics
    pub static ref HTTP_REQUESTS_TOTAL: Counter = Counter::new(
        "nexuszero_http_requests_total",
        "Total number of HTTP requests"
    ).unwrap();

    pub static ref HTTP_REQUEST_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "nexuszero_http_request_duration_seconds",
            "HTTP request duration in seconds"
        )
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();

    // Transaction metrics
    pub static ref TRANSACTIONS_CREATED: Counter = Counter::new(
        "nexuszero_transactions_created_total",
        "Total number of transactions created"
    ).unwrap();

    pub static ref TRANSACTIONS_BY_PRIVACY_LEVEL: prometheus::CounterVec = prometheus::CounterVec::new(
        prometheus::Opts::new(
            "nexuszero_transactions_by_privacy_level",
            "Transactions by privacy level"
        ),
        &["level"]
    ).unwrap();

    // Proof metrics
    pub static ref PROOFS_GENERATED: Counter = Counter::new(
        "nexuszero_proofs_generated_total",
        "Total number of proofs generated"
    ).unwrap();

    pub static ref PROOF_GENERATION_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "nexuszero_proof_generation_duration_seconds",
            "Proof generation duration in seconds"
        )
        .buckets(vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0])
    ).unwrap();

    // Connection metrics
    pub static ref ACTIVE_WEBSOCKET_CONNECTIONS: Gauge = Gauge::new(
        "nexuszero_active_websocket_connections",
        "Number of active WebSocket connections"
    ).unwrap();

    pub static ref DATABASE_CONNECTIONS: Gauge = Gauge::new(
        "nexuszero_database_connections",
        "Number of active database connections"
    ).unwrap();

    // Error metrics
    pub static ref ERRORS_TOTAL: prometheus::CounterVec = prometheus::CounterVec::new(
        prometheus::Opts::new(
            "nexuszero_errors_total",
            "Total number of errors"
        ),
        &["type", "code"]
    ).unwrap();

    // Rate limiting metrics
    pub static ref RATE_LIMIT_EXCEEDED: Counter = Counter::new(
        "nexuszero_rate_limit_exceeded_total",
        "Total number of rate limit exceeded events"
    ).unwrap();

    // Bridge metrics
    pub static ref BRIDGE_TRANSFERS_INITIATED: prometheus::CounterVec = prometheus::CounterVec::new(
        prometheus::Opts::new(
            "nexuszero_bridge_transfers_initiated",
            "Bridge transfers initiated by chain pair"
        ),
        &["source_chain", "target_chain"]
    ).unwrap();

    pub static ref BRIDGE_TRANSFERS_COMPLETED: prometheus::CounterVec = prometheus::CounterVec::new(
        prometheus::Opts::new(
            "nexuszero_bridge_transfers_completed",
            "Bridge transfers completed by chain pair"
        ),
        &["source_chain", "target_chain"]
    ).unwrap();
}

/// Initialize and register all metrics
pub fn init_metrics() {
    REGISTRY
        .register(Box::new(HTTP_REQUESTS_TOTAL.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(HTTP_REQUEST_DURATION.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(TRANSACTIONS_CREATED.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(TRANSACTIONS_BY_PRIVACY_LEVEL.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(PROOFS_GENERATED.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(PROOF_GENERATION_DURATION.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(ACTIVE_WEBSOCKET_CONNECTIONS.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(DATABASE_CONNECTIONS.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(ERRORS_TOTAL.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(RATE_LIMIT_EXCEEDED.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(BRIDGE_TRANSFERS_INITIATED.clone()))
        .unwrap();
    REGISTRY
        .register(Box::new(BRIDGE_TRANSFERS_COMPLETED.clone()))
        .unwrap();
}

/// Prometheus metrics endpoint handler
pub async fn prometheus_metrics() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();

    encoder.encode(&metric_families, &mut buffer).unwrap();

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        buffer,
    )
}

/// Helper function to record request duration
pub fn record_request_duration(duration: std::time::Duration) {
    HTTP_REQUEST_DURATION.observe(duration.as_secs_f64());
}

/// Helper function to increment error counter
pub fn record_error(error_type: &str, error_code: &str) {
    ERRORS_TOTAL
        .with_label_values(&[error_type, error_code])
        .inc();
}

/// Helper function to record transaction by privacy level
pub fn record_transaction(privacy_level: u8) {
    TRANSACTIONS_CREATED.inc();
    TRANSACTIONS_BY_PRIVACY_LEVEL
        .with_label_values(&[&privacy_level.to_string()])
        .inc();
}

/// Helper function to record proof generation
pub fn record_proof_generation(duration: std::time::Duration) {
    PROOFS_GENERATED.inc();
    PROOF_GENERATION_DURATION.observe(duration.as_secs_f64());
}

/// Helper function to record bridge transfer
pub fn record_bridge_transfer(source: &str, target: &str, completed: bool) {
    if completed {
        BRIDGE_TRANSFERS_COMPLETED
            .with_label_values(&[source, target])
            .inc();
    } else {
        BRIDGE_TRANSFERS_INITIATED
            .with_label_values(&[source, target])
            .inc();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_transaction() {
        record_transaction(3);
        // Just verify it doesn't panic
    }

    #[test]
    fn test_record_error() {
        record_error("validation", "INVALID_INPUT");
        // Just verify it doesn't panic
    }

    #[test]
    fn test_record_bridge_transfer() {
        record_bridge_transfer("ethereum", "polygon", false);
        record_bridge_transfer("ethereum", "polygon", true);
        // Just verify it doesn't panic
    }
}

