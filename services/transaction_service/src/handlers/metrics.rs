//! Prometheus metrics handler

use axum::response::IntoResponse;
use lazy_static::lazy_static;
use prometheus::{Counter, CounterVec, Gauge, Histogram, HistogramOpts, Opts, Registry, Encoder, TextEncoder};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Transaction metrics
    pub static ref TRANSACTIONS_CREATED: Counter = Counter::new(
        "transaction_service_transactions_created_total",
        "Total number of transactions created"
    ).unwrap();

    pub static ref TRANSACTIONS_BY_STATUS: CounterVec = CounterVec::new(
        Opts::new(
            "transaction_service_transactions_by_status",
            "Transactions by status"
        ),
        &["status"]
    ).unwrap();

    pub static ref TRANSACTIONS_BY_PRIVACY_LEVEL: CounterVec = CounterVec::new(
        Opts::new(
            "transaction_service_transactions_by_privacy_level",
            "Transactions by privacy level"
        ),
        &["level"]
    ).unwrap();

    // Proof metrics
    pub static ref PROOFS_REQUESTED: Counter = Counter::new(
        "transaction_service_proofs_requested_total",
        "Total number of proof generation requests"
    ).unwrap();

    pub static ref PROOF_QUEUE_LENGTH: Gauge = Gauge::new(
        "transaction_service_proof_queue_length",
        "Current proof generation queue length"
    ).unwrap();

    pub static ref PROOF_GENERATION_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "transaction_service_proof_generation_duration_seconds",
            "Proof generation duration in seconds"
        )
        .buckets(vec![0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0])
    ).unwrap();

    // Privacy morph metrics
    pub static ref PRIVACY_MORPHS: CounterVec = CounterVec::new(
        Opts::new(
            "transaction_service_privacy_morphs_total",
            "Privacy level changes"
        ),
        &["from_level", "to_level"]
    ).unwrap();

    // Batch metrics
    pub static ref BATCH_OPERATIONS: Counter = Counter::new(
        "transaction_service_batch_operations_total",
        "Total number of batch operations"
    ).unwrap();

    pub static ref BATCH_SIZE: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "transaction_service_batch_size",
            "Batch operation sizes"
        )
        .buckets(vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    ).unwrap();

    // Error metrics
    pub static ref ERRORS_TOTAL: CounterVec = CounterVec::new(
        Opts::new(
            "transaction_service_errors_total",
            "Total number of errors"
        ),
        &["type"]
    ).unwrap();
}

/// Initialize and register all metrics
pub fn init_metrics() {
    REGISTRY.register(Box::new(TRANSACTIONS_CREATED.clone())).ok();
    REGISTRY.register(Box::new(TRANSACTIONS_BY_STATUS.clone())).ok();
    REGISTRY.register(Box::new(TRANSACTIONS_BY_PRIVACY_LEVEL.clone())).ok();
    REGISTRY.register(Box::new(PROOFS_REQUESTED.clone())).ok();
    REGISTRY.register(Box::new(PROOF_QUEUE_LENGTH.clone())).ok();
    REGISTRY.register(Box::new(PROOF_GENERATION_DURATION.clone())).ok();
    REGISTRY.register(Box::new(PRIVACY_MORPHS.clone())).ok();
    REGISTRY.register(Box::new(BATCH_OPERATIONS.clone())).ok();
    REGISTRY.register(Box::new(BATCH_SIZE.clone())).ok();
    REGISTRY.register(Box::new(ERRORS_TOTAL.clone())).ok();
}

/// Prometheus metrics endpoint
pub async fn prometheus_metrics() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();

    // Gather metrics from default registry
    let metric_families = prometheus::gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    // Add custom registry metrics
    let custom_families = REGISTRY.gather();
    encoder.encode(&custom_families, &mut buffer).unwrap();

    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        buffer,
    )
}
