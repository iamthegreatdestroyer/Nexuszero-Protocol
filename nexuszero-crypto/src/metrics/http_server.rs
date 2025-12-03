//! HTTP Metrics Server for Prometheus Scraping
//!
//! Provides an HTTP endpoint for Prometheus to scrape ZK system metrics.
//!
//! # Example
//!
//! ```rust,no_run
//! use nexuszero_crypto::metrics::http_server::MetricsServer;
//!
//! #[tokio::main]
//! async fn main() {
//!     let server = MetricsServer::new("0.0.0.0:9090");
//!     server.run().await.expect("Metrics server failed");
//! }
//! ```

use hyper::{
    service::{make_service_fn, service_fn},
    Body, Method, Request, Response, Server, StatusCode,
};
use prometheus::{Encoder, TextEncoder};
use std::convert::Infallible;
use std::net::SocketAddr;
use tracing::{error, info};

use super::zk_metrics::{ZkMetrics, ZK_REGISTRY};

/// HTTP server for metrics exposition
pub struct MetricsServer {
    addr: SocketAddr,
}

impl MetricsServer {
    /// Create a new metrics server
    pub fn new(addr: impl Into<SocketAddr>) -> Self {
        Self { addr: addr.into() }
    }

    /// Create from string address
    pub fn from_str(addr: &str) -> Result<Self, std::net::AddrParseError> {
        Ok(Self {
            addr: addr.parse()?,
        })
    }

    /// Run the metrics server
    pub async fn run(&self) -> Result<(), hyper::Error> {
        // Initialize metrics
        if let Err(e) = ZkMetrics::init() {
            error!("Failed to initialize ZK metrics: {}", e);
        }

        let make_svc = make_service_fn(|_conn| async {
            Ok::<_, Infallible>(service_fn(handle_request))
        });

        let server = Server::bind(&self.addr).serve(make_svc);

        info!(address = %self.addr, "Starting ZK metrics server");

        server.await
    }
}

/// Handle incoming HTTP requests
async fn handle_request(req: Request<Body>) -> Result<Response<Body>, Infallible> {
    match (req.method(), req.uri().path()) {
        // Metrics endpoint
        (&Method::GET, "/metrics") => {
            let encoder = TextEncoder::new();
            let metric_families = ZK_REGISTRY.gather();

            let mut buffer = Vec::new();
            match encoder.encode(&metric_families, &mut buffer) {
                Ok(_) => {
                    let content_type = encoder.format_type();
                    Ok(Response::builder()
                        .status(StatusCode::OK)
                        .header("Content-Type", content_type)
                        .body(Body::from(buffer))
                        .unwrap())
                }
                Err(e) => {
                    error!("Failed to encode metrics: {}", e);
                    Ok(Response::builder()
                        .status(StatusCode::INTERNAL_SERVER_ERROR)
                        .body(Body::from(format!("Failed to encode metrics: {}", e)))
                        .unwrap())
                }
            }
        }

        // Health check endpoint
        (&Method::GET, "/health") => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"status":"healthy","component":"zk_metrics"}"#))
            .unwrap()),

        // Ready check endpoint
        (&Method::GET, "/ready") => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Body::from(r#"{"status":"ready"}"#))
            .unwrap()),

        // Root endpoint with info
        (&Method::GET, "/") => Ok(Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "text/html")
            .body(Body::from(
                r#"<!DOCTYPE html>
<html>
<head><title>NexusZero ZK Metrics</title></head>
<body>
<h1>NexusZero ZK Metrics Server</h1>
<p>Available endpoints:</p>
<ul>
<li><a href="/metrics">/metrics</a> - Prometheus metrics</li>
<li><a href="/health">/health</a> - Health check</li>
<li><a href="/ready">/ready</a> - Readiness check</li>
</ul>
</body>
</html>"#,
            ))
            .unwrap()),

        // 404 for everything else
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap()),
    }
}

/// Start metrics server in background
pub fn spawn_metrics_server(addr: &str) -> tokio::task::JoinHandle<()> {
    let addr = addr.to_string();
    tokio::spawn(async move {
        match MetricsServer::from_str(&addr) {
            Ok(server) => {
                if let Err(e) = server.run().await {
                    error!("Metrics server error: {}", e);
                }
            }
            Err(e) => {
                error!("Invalid metrics server address: {}", e);
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_endpoint() {
        // This would require setting up a test server
        // For now, just verify the handler logic compiles
        let req = Request::builder()
            .method(Method::GET)
            .uri("/metrics")
            .body(Body::empty())
            .unwrap();

        let resp = handle_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_health_endpoint() {
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();

        let resp = handle_request(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }
}
