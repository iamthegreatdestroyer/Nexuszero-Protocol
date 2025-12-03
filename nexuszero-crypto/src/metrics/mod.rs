//! Metrics collection for NexusZero Crypto
//!
//! This module provides comprehensive observability for the ZK proof system:
//!
//! - `zk_metrics`: ZK-specific metrics (proof generation, verification, circuit compilation)
//! - `error_tracking`: Structured error logging with Sentry/Loki integration
//! - `http_server`: HTTP server for Prometheus scraping

pub mod error_tracking;
pub mod http_server;
pub mod zk_metrics;

pub use error_tracking::{
    ErrorContext, ErrorSeverity, ErrorSummary, StructuredError,
    ZkErrorTracker, ErrorAggregator, AggregatedError,
};

pub use http_server::{MetricsServer, spawn_metrics_server};

pub use zk_metrics::{
    ProofType, SecurityLevel, ZkErrorCategory, ZkMetrics,
    ProofGenerationGuard, StageMetricsCollector,
};
