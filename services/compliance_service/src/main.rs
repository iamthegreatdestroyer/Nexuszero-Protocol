// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Compliance Service (RCL - Regulatory Compliance Layer)
//!
//! Multi-jurisdiction regulatory compliance service providing:
//! - Real-time compliance checking against configurable rule sets
//! - Multi-jurisdiction support (FATF, EU, US, APAC regulations)
//! - Automated SAR (Suspicious Activity Report) generation
//! - KYC/AML integration points
//! - Audit trail and compliance reporting
//! - Travel Rule compliance (VASP-to-VASP)

mod config;
mod db;
mod error;
mod handlers;
mod models;
mod rules;
mod services;
mod state;

use axum::{
    routing::{get, post, put, delete},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;

pub use config::ComplianceConfig;
pub use error::{ComplianceError, Result};
pub use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "compliance_service=debug,tower_http=debug".into()),
        )
        .json()
        .init();

    // Load configuration
    dotenvy::dotenv().ok();
    let config = ComplianceConfig::from_env()?;
    
    info!("Starting Compliance Service (RCL)");
    info!("Configured jurisdictions: {:?}", config.enabled_jurisdictions);

    // Initialize state
    let state = Arc::new(AppState::new(config.clone()).await?);
    
    // Start background workers
    state.start_background_workers();

    // Build router
    let app = create_router(state);

    // Run server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!("Compliance Service listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Create the main router with all routes
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        // Health and metrics
        .route("/health", get(handlers::health::health_check))
        .route("/health/ready", get(handlers::health::readiness_check))
        .route("/health/live", get(handlers::health::liveness_check))
        .route("/metrics", get(handlers::metrics::metrics))
        
        // Compliance checking endpoints
        .route("/api/v1/compliance/check", post(handlers::compliance::check_transaction))
        .route("/api/v1/compliance/check/batch", post(handlers::compliance::check_batch))
        .route("/api/v1/compliance/status/:tx_id", get(handlers::compliance::get_status))
        .route("/api/v1/compliance/history/:entity_id", get(handlers::compliance::get_history))
        
        // Risk assessment endpoints
        .route("/api/v1/risk/assess", post(handlers::risk::assess_risk))
        .route("/api/v1/risk/score/:entity_id", get(handlers::risk::get_risk_score))
        .route("/api/v1/risk/factors", get(handlers::risk::list_risk_factors))
        .route("/api/v1/risk/thresholds", get(handlers::risk::get_thresholds))
        .route("/api/v1/risk/thresholds", put(handlers::risk::update_thresholds))
        
        // Jurisdiction management
        .route("/api/v1/jurisdictions", get(handlers::jurisdiction::list_jurisdictions))
        .route("/api/v1/jurisdictions/:code", get(handlers::jurisdiction::get_jurisdiction))
        .route("/api/v1/jurisdictions/:code/rules", get(handlers::jurisdiction::get_rules))
        .route("/api/v1/jurisdictions/:code/thresholds", get(handlers::jurisdiction::get_thresholds))
        
        // SAR (Suspicious Activity Report) endpoints
        .route("/api/v1/sar", post(handlers::sar::create_sar))
        .route("/api/v1/sar", get(handlers::sar::list_sars))
        .route("/api/v1/sar/:id", get(handlers::sar::get_sar))
        .route("/api/v1/sar/:id", put(handlers::sar::update_sar))
        .route("/api/v1/sar/:id/submit", post(handlers::sar::submit_sar))
        
        // KYC/AML endpoints
        .route("/api/v1/kyc/verify", post(handlers::kyc::verify_identity))
        .route("/api/v1/kyc/status/:entity_id", get(handlers::kyc::get_kyc_status))
        .route("/api/v1/aml/screen", post(handlers::kyc::aml_screening))
        .route("/api/v1/aml/watchlist/check", post(handlers::kyc::watchlist_check))
        
        // Travel Rule endpoints
        .route("/api/v1/travel-rule/originator", post(handlers::travel_rule::submit_originator))
        .route("/api/v1/travel-rule/beneficiary", post(handlers::travel_rule::submit_beneficiary))
        .route("/api/v1/travel-rule/verify", post(handlers::travel_rule::verify_transfer))
        .route("/api/v1/travel-rule/vasp/:id", get(handlers::travel_rule::get_vasp_info))
        
        // Reporting endpoints
        .route("/api/v1/reports", get(handlers::reports::list_reports))
        .route("/api/v1/reports/:id", get(handlers::reports::get_report))
        .route("/api/v1/reports/generate", post(handlers::reports::generate_report))
        .route("/api/v1/reports/schedule", post(handlers::reports::schedule_report))
        
        // Audit endpoints
        .route("/api/v1/audit/logs", get(handlers::audit::get_audit_logs))
        .route("/api/v1/audit/export", post(handlers::audit::export_audit_logs))
        
        // Rules management
        .route("/api/v1/rules", get(handlers::rules::list_rules))
        .route("/api/v1/rules", post(handlers::rules::create_rule))
        .route("/api/v1/rules/:id", get(handlers::rules::get_rule))
        .route("/api/v1/rules/:id", put(handlers::rules::update_rule))
        .route("/api/v1/rules/:id", delete(handlers::rules::delete_rule))
        .route("/api/v1/rules/:id/test", post(handlers::rules::test_rule))
        
        // Middleware
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state)
}
