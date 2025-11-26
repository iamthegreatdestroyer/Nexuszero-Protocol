//! NexusZero Privacy Service
//!
//! Adaptive Privacy Morphing (APM) Engine
//!
//! # Overview
//!
//! The Privacy Service implements NexusZero's revolutionary 6-level privacy
//! spectrum, allowing users to dynamically adjust transaction privacy based
//! on their needs while maintaining cryptographic guarantees.
//!
//! # Privacy Levels
//!
//! | Level | Name              | Shielded Fields         | Proof Type       |
//! |-------|-------------------|-------------------------|------------------|
//! | 0     | Transparent       | None                    | No proof needed  |
//! | 1     | Sender-Shielded   | Sender address          | Partial ZK       |
//! | 2     | Recipient-Shielded| Recipient address       | Partial ZK       |
//! | 3     | Amount-Shielded   | Transaction amount      | Range proof      |
//! | 4     | Full Privacy      | Sender, recipient, amt  | Full Groth16     |
//! | 5     | Maximum           | All data + memo         | Full Groth16+    |
//!
//! # Core Features
//!
//! - **Privacy Morphing**: Dynamically change privacy level post-creation
//! - **Proof Generation**: ZK-SNARK proof generation using Groth16
//! - **Selective Disclosure**: Reveal specific fields for compliance
//! - **Privacy Recommendations**: AI-powered privacy level suggestions

mod config;
mod engine;
mod error;
mod handlers;
mod models;
mod proof;
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
pub use error::PrivacyError;
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
                .unwrap_or_else(|_| "privacy_service=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    tracing::info!("Starting NexusZero Privacy Service (APM Engine)...");

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

    tracing::info!(address = %addr, "Privacy Service listening");

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
        // Privacy levels
        .route("/api/v1/privacy/levels", get(handlers::privacy::list_levels))
        .route("/api/v1/privacy/levels/:level", get(handlers::privacy::get_level_details))
        // Privacy recommendations
        .route("/api/v1/privacy/recommend", post(handlers::privacy::recommend_level))
        .route("/api/v1/privacy/validate", post(handlers::privacy::validate_level))
        .route("/api/v1/privacy/cost", get(handlers::privacy::calculate_cost))
        // Privacy morphing
        .route("/api/v1/privacy/morph", post(handlers::morph::morph_privacy))
        .route("/api/v1/privacy/morph/:id", get(handlers::morph::get_morph_status))
        .route("/api/v1/privacy/morph/estimate", post(handlers::morph::estimate_morph))
        // Proof generation
        .route("/api/v1/proof/generate", post(handlers::proof::generate_proof))
        .route("/api/v1/proof/:id", get(handlers::proof::get_proof))
        .route("/api/v1/proof/:id/status", get(handlers::proof::get_proof_status))
        .route("/api/v1/proof/:id/cancel", post(handlers::proof::cancel_proof))
        .route("/api/v1/proof/verify", post(handlers::proof::verify_proof))
        .route("/api/v1/proof/batch", post(handlers::proof::batch_generate))
        // Selective disclosure
        .route("/api/v1/disclosure/create", post(handlers::disclosure::create_disclosure))
        .route("/api/v1/disclosure/:id", get(handlers::disclosure::get_disclosure))
        .route("/api/v1/disclosure/:id/verify", post(handlers::disclosure::verify_disclosure))
        .route("/api/v1/disclosure/:id/revoke", post(handlers::disclosure::revoke_disclosure))
        // Analytics
        .route("/api/v1/analytics/privacy", get(handlers::analytics::privacy_analytics))
        .route("/api/v1/analytics/proofs", get(handlers::analytics::proof_analytics))
        .layer(Extension(state))
        .layer(middleware)
}
