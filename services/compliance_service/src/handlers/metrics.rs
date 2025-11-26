//! Prometheus metrics handler

use axum::response::IntoResponse;
use lazy_static::lazy_static;
use prometheus::{
    Encoder, Histogram, HistogramOpts, IntCounter, IntCounterVec, IntGauge,
    Opts, Registry, TextEncoder,
};

lazy_static! {
    pub static ref REGISTRY: Registry = Registry::new();

    // Compliance check metrics
    pub static ref COMPLIANCE_CHECKS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("compliance_checks_total", "Total number of compliance checks"),
        &["jurisdiction", "status"]
    ).unwrap();

    pub static ref COMPLIANCE_CHECK_DURATION: Histogram = Histogram::with_opts(
        HistogramOpts::new("compliance_check_duration_seconds", "Compliance check duration")
            .buckets(vec![0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
    ).unwrap();

    // Risk assessment metrics
    pub static ref RISK_ASSESSMENTS_TOTAL: IntCounter = IntCounter::new(
        "risk_assessments_total", "Total number of risk assessments"
    ).unwrap();

    pub static ref HIGH_RISK_ENTITIES: IntGauge = IntGauge::new(
        "high_risk_entities", "Number of high-risk entities"
    ).unwrap();

    // SAR metrics
    pub static ref SARS_CREATED: IntCounterVec = IntCounterVec::new(
        Opts::new("sars_created_total", "Total SARs created"),
        &["jurisdiction"]
    ).unwrap();

    pub static ref SARS_SUBMITTED: IntCounterVec = IntCounterVec::new(
        Opts::new("sars_submitted_total", "Total SARs submitted"),
        &["jurisdiction"]
    ).unwrap();

    // Rule metrics
    pub static ref RULES_TRIGGERED: IntCounterVec = IntCounterVec::new(
        Opts::new("rules_triggered_total", "Total rules triggered"),
        &["rule_type", "severity"]
    ).unwrap();

    pub static ref ACTIVE_RULES: IntGauge = IntGauge::new(
        "active_rules", "Number of active compliance rules"
    ).unwrap();

    // KYC/AML metrics
    pub static ref KYC_VERIFICATIONS: IntCounterVec = IntCounterVec::new(
        Opts::new("kyc_verifications_total", "Total KYC verifications"),
        &["status"]
    ).unwrap();

    pub static ref AML_SCREENINGS: IntCounterVec = IntCounterVec::new(
        Opts::new("aml_screenings_total", "Total AML screenings"),
        &["result"]
    ).unwrap();

    pub static ref WATCHLIST_MATCHES: IntCounter = IntCounter::new(
        "watchlist_matches_total", "Total watchlist matches"
    ).unwrap();

    // Travel Rule metrics
    pub static ref TRAVEL_RULE_TRANSFERS: IntCounterVec = IntCounterVec::new(
        Opts::new("travel_rule_transfers_total", "Total Travel Rule transfers"),
        &["status"]
    ).unwrap();
}

/// Register all metrics
pub fn register_metrics() {
    REGISTRY.register(Box::new(COMPLIANCE_CHECKS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(COMPLIANCE_CHECK_DURATION.clone())).ok();
    REGISTRY.register(Box::new(RISK_ASSESSMENTS_TOTAL.clone())).ok();
    REGISTRY.register(Box::new(HIGH_RISK_ENTITIES.clone())).ok();
    REGISTRY.register(Box::new(SARS_CREATED.clone())).ok();
    REGISTRY.register(Box::new(SARS_SUBMITTED.clone())).ok();
    REGISTRY.register(Box::new(RULES_TRIGGERED.clone())).ok();
    REGISTRY.register(Box::new(ACTIVE_RULES.clone())).ok();
    REGISTRY.register(Box::new(KYC_VERIFICATIONS.clone())).ok();
    REGISTRY.register(Box::new(AML_SCREENINGS.clone())).ok();
    REGISTRY.register(Box::new(WATCHLIST_MATCHES.clone())).ok();
    REGISTRY.register(Box::new(TRAVEL_RULE_TRANSFERS.clone())).ok();
}

/// Metrics endpoint handler
pub async fn metrics() -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = REGISTRY.gather();
    let mut buffer = Vec::new();
    
    if encoder.encode(&metric_families, &mut buffer).is_err() {
        return (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics",
        ).into_response();
    }
    
    (
        [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        String::from_utf8(buffer).unwrap_or_default(),
    ).into_response()
}
