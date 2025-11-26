//! Bridge Service - Cross-Chain Atomic Swaps
//! 
//! NexusZero Protocol - Phase 1
//! 
//! Enables secure cross-chain transfers using Hash Time-Locked Contracts (HTLCs).
//! Supports multiple chains including Ethereum, Polygon, Bitcoin, and Solana.

use axum::{
    routing::{get, post, put, delete},
    Router,
};
use sqlx::postgres::PgPoolOptions;
use std::net::SocketAddr;
use std::time::Duration;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
    timeout::TimeoutLayer,
    compression::CompressionLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod db;
mod error;
mod handlers;
mod models;
mod state;

use config::BridgeConfig;
use handlers::metrics;
use state::AppState;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "bridge_service=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();
    
    tracing::info!("Starting Bridge Service v{}", env!("CARGO_PKG_VERSION"));
    
    // Load configuration
    dotenvy::dotenv().ok();
    let config = BridgeConfig::default();
    
    // Initialize metrics
    metrics::init_metrics();
    
    // Create database connection pool
    tracing::info!("Connecting to database...");
    let db_pool = PgPoolOptions::new()
        .max_connections(config.database.max_connections)
        .min_connections(config.database.min_connections)
        .acquire_timeout(Duration::from_secs(config.database.connect_timeout_secs))
        .connect(&config.database.url)
        .await?;
    
    // Run migrations
    tracing::info!("Running database migrations...");
    sqlx::migrate!("./migrations").run(&db_pool).await?;
    
    // Create Redis connection
    tracing::info!("Connecting to Redis...");
    let redis_client = redis::Client::open(config.redis.url.clone())?;
    let redis_manager = redis::aio::ConnectionManager::new(redis_client).await?;
    
    // Create application state
    let state = AppState::new(db_pool, redis_manager, config.clone());
    
    // Initialize chain clients
    tracing::info!("Initializing chain clients...");
    state.chain_clients.initialize().await?;
    
    // Build router
    let app = create_router(state);
    
    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.service.port));
    tracing::info!("Bridge Service listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

/// Create the application router
fn create_router(state: AppState) -> Router {
    // CORS configuration
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);
    
    // Build middleware stack
    let middleware = ServiceBuilder::new()
        .layer(TraceLayer::new_for_http())
        .layer(TimeoutLayer::new(Duration::from_secs(30)))
        .layer(CompressionLayer::new())
        .layer(cors);
    
    // Health routes
    let health_routes = Router::new()
        .route("/live", get(handlers::health::liveness))
        .route("/ready", get(handlers::health::readiness))
        .route("/", get(handlers::health::health_detail));
    
    // Transfer routes
    let transfer_routes = Router::new()
        .route("/", post(handlers::transfers::initiate_transfer))
        .route("/", get(handlers::transfers::list_transfers))
        .route("/stats", get(handlers::transfers::get_transfer_stats))
        .route("/:transfer_id", get(handlers::transfers::get_transfer))
        .route("/:transfer_id/cancel", post(handlers::transfers::cancel_transfer))
        .route("/:transfer_id/retry", post(handlers::transfers::retry_transfer));
    
    // HTLC routes
    let htlc_routes = Router::new()
        .route("/", post(handlers::htlc::create_htlc))
        .route("/:htlc_id", get(handlers::htlc::get_htlc))
        .route("/:htlc_id/claim", post(handlers::htlc::claim_htlc))
        .route("/:htlc_id/refund", post(handlers::htlc::refund_htlc))
        .route("/:htlc_id/verify", get(handlers::htlc::verify_htlc))
        .route("/transfer/:transfer_id", get(handlers::htlc::get_htlcs_for_transfer));
    
    // Quote routes
    let quote_routes = Router::new()
        .route("/", post(handlers::quote::get_quote))
        .route("/compare", post(handlers::quote::compare_quotes))
        .route("/:quote_id/execute", post(handlers::quote::execute_quote));
    
    // Chain routes
    let chain_routes = Router::new()
        .route("/", get(handlers::chains::list_chains))
        .route("/status", get(handlers::chains::list_chain_statuses))
        .route("/:chain_id", get(handlers::chains::get_chain))
        .route("/:chain_id/status", get(handlers::chains::get_chain_status))
        .route("/:chain_id/assets", get(handlers::chains::get_chain_assets));
    
    // Route routes
    let route_routes = Router::new()
        .route("/", get(handlers::routes::list_routes))
        .route("/matrix", get(handlers::routes::get_route_matrix))
        .route("/from/:source", get(handlers::routes::get_routes_from_chain))
        .route("/:source/:destination/:asset", get(handlers::routes::get_route))
        .route("/:source/:destination/:asset/check", get(handlers::routes::check_route_availability));
    
    // Liquidity routes
    let liquidity_routes = Router::new()
        .route("/pools", get(handlers::liquidity::list_pools))
        .route("/pools/:chain/:asset", get(handlers::liquidity::get_pool))
        .route("/deposit", post(handlers::liquidity::deposit_liquidity))
        .route("/withdraw", post(handlers::liquidity::withdraw_liquidity))
        .route("/positions", get(handlers::liquidity::get_positions))
        .route("/claim", post(handlers::liquidity::claim_rewards));
    
    // Admin/Bridge management routes
    let bridge_routes = Router::new()
        .route("/status", get(handlers::bridge::get_bridge_status))
        .route("/pause", post(handlers::bridge::set_bridge_paused))
        .route("/fees", get(handlers::bridge::get_fee_config))
        .route("/fees", put(handlers::bridge::update_fee_config))
        .route("/limits", get(handlers::bridge::get_security_limits))
        .route("/relayer", get(handlers::bridge::get_relayer_status))
        .route("/chains/:chain_id/enabled", put(handlers::bridge::set_chain_enabled));
    
    // Combine all routes
    Router::new()
        .nest("/health", health_routes)
        .route("/metrics", get(handlers::metrics::metrics_handler))
        .nest("/api/v1/transfers", transfer_routes)
        .nest("/api/v1/htlc", htlc_routes)
        .nest("/api/v1/quotes", quote_routes)
        .nest("/api/v1/chains", chain_routes)
        .nest("/api/v1/routes", route_routes)
        .nest("/api/v1/liquidity", liquidity_routes)
        .nest("/api/v1/bridge", bridge_routes)
        .layer(middleware)
        .with_state(state)
}
