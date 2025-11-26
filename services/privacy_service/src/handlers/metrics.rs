//! Prometheus metrics handler

use axum::response::IntoResponse;
use lazy_static::lazy_static;
use prometheus::{Counter, CounterVec, Gauge, Histogram, HistogramOpts, Opts, Registry, Encoder, TextEncoder};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Proof generation metrics
    pub static ref PROOFS_GENERATED: Counter = Counter::new(
        "privacy_service_proofs_generated_total",
        "Total proofs generated"
    ).unwrap();

    pub static ref PROOFS_BY_LEVEL: CounterVec = CounterVec::new(
        Opts::new("privacy_service_proofs_by_level", "Proofs by privacy level"),
        &["level"]
    ).unwrap();

    pub static ref PROOF_GENERATION_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "privacy_service_proof_generation_seconds",
            "Proof generation duration"
        ).buckets(vec![0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0])
    ).unwrap();

    pub static ref PROOF_QUEUE_LENGTH: Gauge = Gauge::new(
        "privacy_service_proof_queue_length",
        "Current proof queue length"
    ).unwrap();

    pub static ref ACTIVE_PROOF_GENERATIONS: Gauge = Gauge::new(
        "privacy_service_active_proof_generations",
        "Currently active proof generations"
    ).unwrap();

    // Verification metrics
    pub static ref PROOFS_VERIFIED: Counter = Counter::new(
        "privacy_service_proofs_verified_total",
        "Total proofs verified"
    ).unwrap();

    pub static ref PROOF_VERIFICATION_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "privacy_service_proof_verification_seconds",
            "Proof verification duration"
        ).buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25])
    ).unwrap();

    // Privacy morph metrics
    pub static ref PRIVACY_MORPHS: CounterVec = CounterVec::new(
        Opts::new("privacy_service_morphs_total", "Privacy level morphs"),
        &["from", "to"]
    ).unwrap();

    // Disclosure metrics
    pub static ref DISCLOSURES_CREATED: Counter = Counter::new(
        "privacy_service_disclosures_created_total",
        "Total selective disclosures created"
    ).unwrap();

    pub static ref DISCLOSURES_VERIFIED: Counter = Counter::new(
        "privacy_service_disclosures_verified_total",
        "Total selective disclosures verified"
    ).unwrap();

    // Error metrics
    pub static ref ERRORS_TOTAL: CounterVec = CounterVec::new(
        Opts::new("privacy_service_errors_total", "Total errors"),
        &["type"]
    ).unwrap();
}

pub fn init_metrics() {
    REGISTRY.register(Box::new(PROOFS_GENERATED.clone())).ok();
    REGISTRY.register(Box::new(PROOFS_BY_LEVEL.clone())).ok();
    REGISTRY.register(Box::new(PROOF_GENERATION_DURATION.clone())).ok();
    REGISTRY.register(Box::new(PROOF_QUEUE_LENGTH.clone())).ok();
    REGISTRY.register(Box::new(ACTIVE_PROOF_GENERATIONS.clone())).ok();
    REGISTRY.register(Box::new(PROOFS_VERIFIED.clone())).ok();
    REGISTRY.register(Box::new(PROOF_VERIFICATION_DURATION.clone())).ok();
    REGISTRY.register(Box::new(PRIVACY_MORPHS.clone())).ok();
    REGISTRY.register(Box::new(DISCLOSURES_CREATED.clone())).ok();
    REGISTRY.register(Box::new(DISCLOSURES_VERIFIED.clone())).ok();
    REGISTRY.register(Box::new(ERRORS_TOTAL.clone())).ok();
}

pub async fn prometheus_metrics() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let mut buffer = Vec::new();

    let metric_families = prometheus::gather();
    encoder.encode(&metric_families, &mut buffer).unwrap();

    let custom_families = REGISTRY.gather();
    encoder.encode(&custom_families, &mut buffer).unwrap();

    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        buffer,
    )
}
