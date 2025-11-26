//! NexusZero API Gateway
//!
//! High-performance, secure API gateway providing:
//! - JWT/OAuth2 authentication
//! - Rate limiting with Redis backend
//! - Request routing to microservices
//! - Health monitoring & Prometheus metrics
//! - WebSocket support for real-time updates
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    API Gateway                          │
//! ├─────────────────────────────────────────────────────────┤
//! │  Rate Limiter → Auth → Router → Service Proxy          │
//! └─────────────────────────────────────────────────────────┘
//!           │              │              │
//!           ▼              ▼              ▼
//!    ┌──────────┐   ┌──────────┐   ┌──────────┐
//!    │Transaction│   │ Privacy  │   │Compliance│
//!    │ Service   │   │ Service  │   │ Service  │
//!    └──────────┘   └──────────┘   └──────────┘
//! ```

use axum::{
    extract::Extension,
    http::StatusCode,
    middleware,
    routing::{get, post, put},
    Router,
};
use std::sync::Arc;
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    request_id::MakeRequestUuid,
    trace::TraceLayer,
    ServiceBuilderExt,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod auth;
mod config;
mod error;
mod handlers;
mod middleware;
mod routes;
mod state;

use config::Config;
use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,tower_http=debug,axum=trace".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = Config::load()?;
    tracing::info!(
        "Starting NexusZero API Gateway v{}",
        env!("CARGO_PKG_VERSION")
    );
    tracing::info!("Environment: {}", config.environment);

    // Initialize application state
    let state = AppState::new(&config).await?;
    let state = Arc::new(state);

    // Build the router with all routes
    let app = build_router(state.clone());

    // Start metrics server on separate port
    let metrics_app = Router::new().route("/metrics", get(handlers::metrics::prometheus_metrics));

    let metrics_addr = format!("{}:{}", config.host, config.metrics_port);
    let metrics_listener = tokio::net::TcpListener::bind(&metrics_addr).await?;
    tracing::info!("Metrics server listening on {}", metrics_addr);

    tokio::spawn(async move {
        axum::serve(metrics_listener, metrics_app).await.unwrap();
    });

    // Start main server
    let addr = format!("{}:{}", config.host, config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("API Gateway listening on {}", addr);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Build the main application router with all middleware and routes
fn build_router(state: Arc<AppState>) -> Router {
    // Public routes (no auth required)
    let public_routes = Router::new()
        .route("/health", get(handlers::health::health_check))
        .route("/ready", get(handlers::health::readiness_check))
        .route("/api/v1/auth/login", post(handlers::auth::login))
        .route("/api/v1/auth/register", post(handlers::auth::register))
        .route("/api/v1/auth/refresh", post(handlers::auth::refresh_token));

    // Protected routes (auth required)
    let protected_routes = Router::new()
        // Authentication
        .route("/api/v1/auth/logout", post(handlers::auth::logout))
        .route("/api/v1/auth/me", get(handlers::auth::get_current_user))
        // Privacy Transactions
        .route(
            "/api/v1/transactions",
            get(handlers::transaction::list_transactions),
        )
        .route(
            "/api/v1/transactions",
            post(handlers::transaction::create_transaction),
        )
        .route(
            "/api/v1/transactions/:id",
            get(handlers::transaction::get_transaction),
        )
        .route(
            "/api/v1/transactions/:id/proof",
            get(handlers::transaction::get_proof),
        )
        .route(
            "/api/v1/transactions/:id/status",
            get(handlers::transaction::get_status),
        )
        // Privacy Levels (APM)
        .route(
            "/api/v1/privacy/levels",
            get(handlers::privacy::list_privacy_levels),
        )
        .route(
            "/api/v1/privacy/recommend",
            post(handlers::privacy::recommend_level),
        )
        .route("/api/v1/privacy/morph", post(handlers::privacy::morph_privacy))
        .route(
            "/api/v1/privacy/estimate",
            post(handlers::privacy::estimate_cost),
        )
        // Compliance (RCL)
        .route(
            "/api/v1/compliance/verify",
            post(handlers::compliance::verify_compliance),
        )
        .route(
            "/api/v1/compliance/selective-disclosure",
            post(handlers::compliance::selective_disclosure),
        )
        .route(
            "/api/v1/compliance/proofs",
            get(handlers::compliance::list_compliance_proofs),
        )
        .route(
            "/api/v1/compliance/proofs/:id",
            get(handlers::compliance::get_compliance_proof),
        )
        // Proof Generation
        .route("/api/v1/proofs/generate", post(handlers::proof::generate_proof))
        .route("/api/v1/proofs/verify", post(handlers::proof::verify_proof))
        .route("/api/v1/proofs/batch", post(handlers::proof::batch_generate))
        .route("/api/v1/proofs/:id", get(handlers::proof::get_proof))
        .route(
            "/api/v1/proofs/:id/status",
            get(handlers::proof::get_proof_status),
        )
        // Cross-Chain Bridge
        .route("/api/v1/bridge/quote", post(handlers::bridge::get_quote))
        .route(
            "/api/v1/bridge/initiate",
            post(handlers::bridge::initiate_transfer),
        )
        .route(
            "/api/v1/bridge/status/:id",
            get(handlers::bridge::get_status),
        )
        .route(
            "/api/v1/bridge/history",
            get(handlers::bridge::get_history),
        )
        .route(
            "/api/v1/bridge/supported-chains",
            get(handlers::bridge::get_supported_chains),
        )
        // Apply JWT auth middleware to protected routes
        .layer(middleware::from_fn_with_state(
            state.clone(),
            middleware::auth::jwt_auth,
        ));

    // WebSocket routes
    let ws_routes = Router::new()
        .route("/ws", get(handlers::websocket::ws_handler))
        .route("/ws/proofs", get(handlers::websocket::proof_status_ws));

    // Combine all routes
    Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .merge(ws_routes)
        .layer(Extension(state))
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(
            tower::ServiceBuilder::new()
                .set_x_request_id(MakeRequestUuid)
                .propagate_x_request_id(),
        )
        .layer(axum::middleware::from_fn(middleware::rate_limit::rate_limiter))
        .layer(axum::middleware::from_fn(middleware::logging::request_logger))
}
