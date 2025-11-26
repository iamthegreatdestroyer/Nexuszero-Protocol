//! NexusZero Transaction Service
//!
//! Core transaction handling with Adaptive Privacy Morphing (APM) integration.
//! Manages the full transaction lifecycle from creation to finalization.
//!
//! # Architecture
//!
//! The Transaction Service is responsible for:
//! - Transaction creation and validation
//! - Privacy level assignment via APM integration
//! - Proof generation coordination
//! - Transaction state management
//! - Event emission for downstream services
//!
//! # Privacy Levels
//!
//! Transactions support 6 privacy levels (0-5):
//! - Level 0: Full transparency (public)
//! - Level 1: Sender-shielded (recipient visible)
//! - Level 2: Recipient-shielded (sender visible)
//! - Level 3: Amount-shielded (parties visible)
//! - Level 4: Full privacy (memo visible)
//! - Level 5: Maximum privacy (all data shielded)

mod config;
mod db;
mod error;
mod handlers;
mod models;
mod services;
mod state;

use axum::{
    routing::{get, post, put},
    Extension, Router,
};
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

pub use config::Config;
pub use error::TransactionError;
pub use models::*;
pub use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load environment variables
    dotenvy::dotenv().ok();

    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "transaction_service=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    tracing::info!("Starting NexusZero Transaction Service...");

    // Load configuration
    let config = Config::load()?;
    tracing::info!(host = %config.host, port = %config.port, "Configuration loaded");

    // Initialize application state
    let state = Arc::new(AppState::new(&config).await?);

    // Build router
    let app = create_router(state.clone());

    // Start server
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!(address = %addr, "Transaction Service listening");

    axum::serve(listener, app).await?;

    Ok(())
}

/// Create the API router with all routes
pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let middleware = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(cors);

    Router::new()
        // Health endpoints
        .route("/health", get(handlers::health::health_check))
        .route("/ready", get(handlers::health::readiness_check))
        .route("/metrics", get(handlers::metrics::prometheus_metrics))
        // Transaction CRUD
        .route("/api/v1/transactions", post(handlers::transaction::create_transaction))
        .route("/api/v1/transactions", get(handlers::transaction::list_transactions))
        .route("/api/v1/transactions/:id", get(handlers::transaction::get_transaction))
        .route("/api/v1/transactions/:id", put(handlers::transaction::update_transaction))
        .route("/api/v1/transactions/:id/cancel", post(handlers::transaction::cancel_transaction))
        // Transaction state transitions
        .route("/api/v1/transactions/:id/submit", post(handlers::transaction::submit_transaction))
        .route("/api/v1/transactions/:id/finalize", post(handlers::transaction::finalize_transaction))
        // Proof management
        .route("/api/v1/transactions/:id/proof", get(handlers::transaction::get_proof))
        .route("/api/v1/transactions/:id/proof", post(handlers::transaction::request_proof))
        // Batch operations
        .route("/api/v1/transactions/batch", post(handlers::batch::create_batch))
        .route("/api/v1/transactions/batch/:id", get(handlers::batch::get_batch_status))
        // Privacy level management
        .route("/api/v1/transactions/:id/privacy", get(handlers::privacy::get_privacy_level))
        .route("/api/v1/transactions/:id/privacy", put(handlers::privacy::update_privacy_level))
        .route("/api/v1/transactions/:id/privacy/morph", post(handlers::privacy::morph_privacy))
        // Compliance hooks
        .route("/api/v1/transactions/:id/compliance", get(handlers::compliance::get_compliance_status))
        .route("/api/v1/transactions/:id/selective-disclosure", post(handlers::compliance::create_selective_disclosure))
        // Analytics
        .route("/api/v1/analytics/summary", get(handlers::analytics::get_summary))
        .route("/api/v1/analytics/privacy-distribution", get(handlers::analytics::privacy_distribution))
        .layer(Extension(state))
        .layer(middleware)
}
