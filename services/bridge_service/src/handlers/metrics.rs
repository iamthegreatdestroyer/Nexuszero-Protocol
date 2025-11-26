//! Prometheus metrics handlers

use axum::response::IntoResponse;
use lazy_static::lazy_static;
use prometheus::{
    self, Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramOpts, HistogramVec,
    IntCounter, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry, TextEncoder,
};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();
    
    // Transfer metrics
    pub static ref TRANSFERS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_transfers_total", "Total number of bridge transfers"),
        &["source_chain", "destination_chain", "status"]
    ).unwrap();
    
    pub static ref TRANSFERS_ACTIVE: IntGaugeVec = IntGaugeVec::new(
        Opts::new("bridge_transfers_active", "Number of active transfers"),
        &["source_chain", "destination_chain"]
    ).unwrap();
    
    pub static ref TRANSFER_VOLUME_USD: CounterVec = CounterVec::new(
        Opts::new("bridge_transfer_volume_usd", "Total transfer volume in USD"),
        &["source_chain", "destination_chain", "asset"]
    ).unwrap();
    
    pub static ref TRANSFER_DURATION_SECONDS: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "bridge_transfer_duration_seconds",
            "Transfer completion time in seconds"
        ).buckets(vec![30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0]),
        &["source_chain", "destination_chain"]
    ).unwrap();
    
    // HTLC metrics
    pub static ref HTLC_CREATED: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_htlc_created_total", "Total HTLCs created"),
        &["chain"]
    ).unwrap();
    
    pub static ref HTLC_CLAIMED: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_htlc_claimed_total", "Total HTLCs claimed"),
        &["chain"]
    ).unwrap();
    
    pub static ref HTLC_REFUNDED: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_htlc_refunded_total", "Total HTLCs refunded"),
        &["chain"]
    ).unwrap();
    
    pub static ref HTLC_ACTIVE: IntGaugeVec = IntGaugeVec::new(
        Opts::new("bridge_htlc_active", "Number of active HTLCs"),
        &["chain", "status"]
    ).unwrap();
    
    // Chain metrics
    pub static ref CHAIN_BLOCK_HEIGHT: IntGaugeVec = IntGaugeVec::new(
        Opts::new("bridge_chain_block_height", "Current block height"),
        &["chain"]
    ).unwrap();
    
    pub static ref CHAIN_LATENCY_MS: GaugeVec = GaugeVec::new(
        Opts::new("bridge_chain_latency_ms", "Chain RPC latency in milliseconds"),
        &["chain"]
    ).unwrap();
    
    pub static ref CHAIN_HEALTHY: IntGaugeVec = IntGaugeVec::new(
        Opts::new("bridge_chain_healthy", "Chain health status (1=healthy, 0=unhealthy)"),
        &["chain"]
    ).unwrap();
    
    // Liquidity metrics
    pub static ref LIQUIDITY_AVAILABLE: GaugeVec = GaugeVec::new(
        Opts::new("bridge_liquidity_available", "Available liquidity"),
        &["chain", "asset"]
    ).unwrap();
    
    pub static ref LIQUIDITY_LOCKED: GaugeVec = GaugeVec::new(
        Opts::new("bridge_liquidity_locked", "Locked liquidity"),
        &["chain", "asset"]
    ).unwrap();
    
    pub static ref LIQUIDITY_UTILIZATION: GaugeVec = GaugeVec::new(
        Opts::new("bridge_liquidity_utilization", "Liquidity utilization ratio"),
        &["chain", "asset"]
    ).unwrap();
    
    // Fee metrics
    pub static ref FEES_COLLECTED_USD: CounterVec = CounterVec::new(
        Opts::new("bridge_fees_collected_usd", "Total fees collected in USD"),
        &["chain", "fee_type"]
    ).unwrap();
    
    // Request metrics
    pub static ref HTTP_REQUESTS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_http_requests_total", "Total HTTP requests"),
        &["method", "path", "status"]
    ).unwrap();
    
    pub static ref HTTP_REQUEST_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "bridge_http_request_duration_seconds",
            "HTTP request duration in seconds"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]),
        &["method", "path"]
    ).unwrap();
    
    // Quote metrics
    pub static ref QUOTES_REQUESTED: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_quotes_requested_total", "Total quotes requested"),
        &["source_chain", "destination_chain"]
    ).unwrap();
    
    pub static ref QUOTES_EXECUTED: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_quotes_executed_total", "Total quotes executed"),
        &["source_chain", "destination_chain"]
    ).unwrap();
    
    // Error metrics
    pub static ref ERRORS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_errors_total", "Total errors by type"),
        &["error_type", "chain"]
    ).unwrap();
    
    // Relayer metrics
    pub static ref RELAYER_TASKS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("bridge_relayer_tasks_total", "Total relayer tasks"),
        &["chain", "status"]
    ).unwrap();
    
    pub static ref RELAYER_GAS_USED: CounterVec = CounterVec::new(
        Opts::new("bridge_relayer_gas_used", "Total gas used by relayer"),
        &["chain"]
    ).unwrap();
}

/// Initialize and register all metrics
pub fn init_metrics() {
    REGISTRY.register(Box::new(TRANSFERS_TOTAL.clone())).unwrap();
    REGISTRY.register(Box::new(TRANSFERS_ACTIVE.clone())).unwrap();
    REGISTRY.register(Box::new(TRANSFER_VOLUME_USD.clone())).unwrap();
    REGISTRY.register(Box::new(TRANSFER_DURATION_SECONDS.clone())).unwrap();
    
    REGISTRY.register(Box::new(HTLC_CREATED.clone())).unwrap();
    REGISTRY.register(Box::new(HTLC_CLAIMED.clone())).unwrap();
    REGISTRY.register(Box::new(HTLC_REFUNDED.clone())).unwrap();
    REGISTRY.register(Box::new(HTLC_ACTIVE.clone())).unwrap();
    
    REGISTRY.register(Box::new(CHAIN_BLOCK_HEIGHT.clone())).unwrap();
    REGISTRY.register(Box::new(CHAIN_LATENCY_MS.clone())).unwrap();
    REGISTRY.register(Box::new(CHAIN_HEALTHY.clone())).unwrap();
    
    REGISTRY.register(Box::new(LIQUIDITY_AVAILABLE.clone())).unwrap();
    REGISTRY.register(Box::new(LIQUIDITY_LOCKED.clone())).unwrap();
    REGISTRY.register(Box::new(LIQUIDITY_UTILIZATION.clone())).unwrap();
    
    REGISTRY.register(Box::new(FEES_COLLECTED_USD.clone())).unwrap();
    
    REGISTRY.register(Box::new(HTTP_REQUESTS_TOTAL.clone())).unwrap();
    REGISTRY.register(Box::new(HTTP_REQUEST_DURATION.clone())).unwrap();
    
    REGISTRY.register(Box::new(QUOTES_REQUESTED.clone())).unwrap();
    REGISTRY.register(Box::new(QUOTES_EXECUTED.clone())).unwrap();
    
    REGISTRY.register(Box::new(ERRORS_TOTAL.clone())).unwrap();
    
    REGISTRY.register(Box::new(RELAYER_TASKS_TOTAL.clone())).unwrap();
    REGISTRY.register(Box::new(RELAYER_GAS_USED.clone())).unwrap();
}

/// Handler to expose Prometheus metrics
pub async fn metrics_handler() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    
    match encoder.encode_to_string(&metric_families) {
        Ok(metrics) => (
            axum::http::StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            metrics,
        ),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            format!("Failed to encode metrics: {}", e),
        ),
    }
}

/// Record a transfer
pub fn record_transfer(
    source_chain: &str,
    destination_chain: &str,
    asset: &str,
    status: &str,
    amount_usd: f64,
    duration_secs: Option<f64>,
) {
    TRANSFERS_TOTAL
        .with_label_values(&[source_chain, destination_chain, status])
        .inc();
    
    if status == "completed" {
        TRANSFER_VOLUME_USD
            .with_label_values(&[source_chain, destination_chain, asset])
            .inc_by(amount_usd);
        
        if let Some(duration) = duration_secs {
            TRANSFER_DURATION_SECONDS
                .with_label_values(&[source_chain, destination_chain])
                .observe(duration);
        }
    }
}

/// Record HTLC operation
pub fn record_htlc_operation(chain: &str, operation: &str) {
    match operation {
        "created" => HTLC_CREATED.with_label_values(&[chain]).inc(),
        "claimed" => HTLC_CLAIMED.with_label_values(&[chain]).inc(),
        "refunded" => HTLC_REFUNDED.with_label_values(&[chain]).inc(),
        _ => {}
    }
}

/// Update chain health
pub fn update_chain_health(chain: &str, healthy: bool, block_height: u64, latency_ms: f64) {
    CHAIN_HEALTHY
        .with_label_values(&[chain])
        .set(if healthy { 1 } else { 0 });
    CHAIN_BLOCK_HEIGHT
        .with_label_values(&[chain])
        .set(block_height as i64);
    CHAIN_LATENCY_MS
        .with_label_values(&[chain])
        .set(latency_ms);
}

/// Record error
pub fn record_error(error_type: &str, chain: &str) {
    ERRORS_TOTAL
        .with_label_values(&[error_type, chain])
        .inc();
}
