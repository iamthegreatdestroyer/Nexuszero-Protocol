//! Error Tracking and Structured Logging for ZK System
//!
//! Provides comprehensive error tracking with:
//! - Structured logging for Loki/ELK ingestion
//! - Sentry integration for error aggregation
//! - Correlation IDs for distributed tracing
//! - Error categorization and severity classification
//!
//! # Example
//!
//! ```rust,no_run
//! use nexuszero_crypto::metrics::{ZkErrorTracker, ErrorContext, ZkErrorCategory, ErrorSeverity};
//!
//! let tracker = ZkErrorTracker::global();
//!
//! // Track an error with context
//! tracker.track_error(
//!     ZkErrorCategory::ProofGeneration,
//!     ErrorSeverity::Error,
//!     ErrorContext::new()
//!         .with_proof_type("schnorr")
//!         .with_circuit_size(1024)
//!         .with_correlation_id("req-12345"),
//!     anyhow::anyhow!("Witness validation failed"),
//! );
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{error, info, span, warn, Level};

use super::zk_metrics::{ZkErrorCategory, ZkMetrics};

// ============================================================================
// ERROR CONTEXT
// ============================================================================

/// Rich context for error tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Unique correlation ID for distributed tracing
    pub correlation_id: Option<String>,
    /// Proof type being processed
    pub proof_type: Option<String>,
    /// Security level
    pub security_level: Option<String>,
    /// Circuit size (constraints)
    pub circuit_size: Option<usize>,
    /// Operation that failed
    pub operation: Option<String>,
    /// Stage within operation
    pub stage: Option<String>,
    /// User/tenant ID (if applicable)
    pub user_id: Option<String>,
    /// Request ID
    pub request_id: Option<String>,
    /// Service instance
    pub instance_id: Option<String>,
    /// Custom tags
    pub tags: HashMap<String, String>,
    /// Timestamp of error
    pub timestamp: u64,
    /// Duration of operation before failure (ms)
    pub duration_ms: Option<f64>,
    /// Memory usage at time of error
    pub memory_bytes: Option<usize>,
    /// Retry attempt number
    pub retry_attempt: Option<u32>,
}

impl ErrorContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self {
            correlation_id: None,
            proof_type: None,
            security_level: None,
            circuit_size: None,
            operation: None,
            stage: None,
            user_id: None,
            request_id: None,
            instance_id: None,
            tags: HashMap::new(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            duration_ms: None,
            memory_bytes: None,
            retry_attempt: None,
        }
    }

    /// Set correlation ID for distributed tracing
    pub fn with_correlation_id(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    /// Set proof type
    pub fn with_proof_type(mut self, proof_type: impl Into<String>) -> Self {
        self.proof_type = Some(proof_type.into());
        self
    }

    /// Set security level
    pub fn with_security_level(mut self, level: impl Into<String>) -> Self {
        self.security_level = Some(level.into());
        self
    }

    /// Set circuit size
    pub fn with_circuit_size(mut self, size: usize) -> Self {
        self.circuit_size = Some(size);
        self
    }

    /// Set operation name
    pub fn with_operation(mut self, op: impl Into<String>) -> Self {
        self.operation = Some(op.into());
        self
    }

    /// Set stage within operation
    pub fn with_stage(mut self, stage: impl Into<String>) -> Self {
        self.stage = Some(stage.into());
        self
    }

    /// Set user/tenant ID
    pub fn with_user_id(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Set request ID
    pub fn with_request_id(mut self, req_id: impl Into<String>) -> Self {
        self.request_id = Some(req_id.into());
        self
    }

    /// Set instance ID
    pub fn with_instance_id(mut self, instance: impl Into<String>) -> Self {
        self.instance_id = Some(instance.into());
        self
    }

    /// Add a custom tag
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Set duration before failure
    pub fn with_duration_ms(mut self, duration: f64) -> Self {
        self.duration_ms = Some(duration);
        self
    }

    /// Set memory usage
    pub fn with_memory_bytes(mut self, bytes: usize) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    /// Set retry attempt number
    pub fn with_retry_attempt(mut self, attempt: u32) -> Self {
        self.retry_attempt = Some(attempt);
        self
    }
}

impl Default for ErrorContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// ERROR SEVERITY
// ============================================================================

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ErrorSeverity {
    /// Debug-level: development only
    Debug,
    /// Info-level: informational
    Info,
    /// Warning: degraded but functional
    Warning,
    /// Error: operation failed
    Error,
    /// Critical: system stability at risk
    Critical,
    /// Fatal: unrecoverable
    Fatal,
}

impl ErrorSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorSeverity::Debug => "debug",
            ErrorSeverity::Info => "info",
            ErrorSeverity::Warning => "warning",
            ErrorSeverity::Error => "error",
            ErrorSeverity::Critical => "critical",
            ErrorSeverity::Fatal => "fatal",
        }
    }
}

// ============================================================================
// STRUCTURED ERROR
// ============================================================================

/// A structured error with full context for tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredError {
    /// Unique error ID
    pub error_id: String,
    /// Error category
    pub category: String,
    /// Error severity
    pub severity: String,
    /// Human-readable message
    pub message: String,
    /// Full error details (may include stack trace)
    pub details: Option<String>,
    /// Error context
    pub context: ErrorContext,
    /// Fingerprint for deduplication
    pub fingerprint: String,
    /// Number of occurrences (for aggregation)
    pub count: u64,
}

impl StructuredError {
    /// Create a new structured error
    pub fn new(
        category: ZkErrorCategory,
        severity: ErrorSeverity,
        message: impl Into<String>,
        context: ErrorContext,
    ) -> Self {
        let message = message.into();
        let error_id = generate_error_id();
        let fingerprint = generate_fingerprint(&category, &message, &context);

        Self {
            error_id,
            category: category.as_str().to_string(),
            severity: severity.as_str().to_string(),
            message,
            details: None,
            context,
            fingerprint,
            count: 1,
        }
    }

    /// Add error details (stack trace, etc.)
    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }

    /// Convert to JSON for logging
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|_| format!("{:?}", self))
    }

    /// Convert to Loki-compatible log line
    pub fn to_loki_line(&self) -> String {
        // Loki logfmt style
        let mut parts = vec![
            format!("level={}", self.severity),
            format!("error_id={}", self.error_id),
            format!("category={}", self.category),
            format!("message=\"{}\"", self.message.replace('"', "\\\"")),
        ];

        if let Some(ref corr_id) = self.context.correlation_id {
            parts.push(format!("correlation_id={}", corr_id));
        }
        if let Some(ref proof_type) = self.context.proof_type {
            parts.push(format!("proof_type={}", proof_type));
        }
        if let Some(ref op) = self.context.operation {
            parts.push(format!("operation={}", op));
        }
        if let Some(ref stage) = self.context.stage {
            parts.push(format!("stage={}", stage));
        }
        if let Some(duration) = self.context.duration_ms {
            parts.push(format!("duration_ms={:.2}", duration));
        }

        parts.join(" ")
    }
}

// ============================================================================
// ERROR TRACKER
// ============================================================================

/// Global error tracker for ZK system
pub struct ZkErrorTracker {
    /// Metrics instance
    metrics: &'static ZkMetrics,
    /// Error counter for IDs
    error_counter: AtomicU64,
    /// Instance ID
    instance_id: String,
    /// Sentry DSN (if configured)
    sentry_dsn: Option<String>,
}

impl ZkErrorTracker {
    /// Get the global error tracker instance
    pub fn global() -> &'static ZkErrorTracker {
        static INSTANCE: std::sync::OnceLock<ZkErrorTracker> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| ZkErrorTracker {
            metrics: ZkMetrics::global(),
            error_counter: AtomicU64::new(0),
            instance_id: std::env::var("INSTANCE_ID")
                .unwrap_or_else(|_| format!("nexuszero-{}", generate_short_id())),
            sentry_dsn: std::env::var("SENTRY_DSN").ok(),
        })
    }

    /// Initialize the error tracker
    pub fn init() {
        let _ = Self::global();
        info!(
            instance_id = %Self::global().instance_id,
            sentry_enabled = Self::global().sentry_dsn.is_some(),
            "ZK error tracker initialized"
        );
    }

    /// Track an error with full context
    pub fn track_error<E: std::fmt::Display>(
        &self,
        category: ZkErrorCategory,
        severity: ErrorSeverity,
        context: ErrorContext,
        error: E,
    ) {
        let message = error.to_string();

        // Create structured error
        let structured = StructuredError::new(category, severity, &message, context.clone());

        // Update metrics
        let proof_type_opt = context
            .proof_type
            .as_ref()
            .and_then(|s| parse_proof_type(s));

        self.metrics.record_error(
            category,
            proof_type_opt,
            severity.as_str(),
        );

        // Log to tracing (will be picked up by Loki/ELK)
        let span = span!(
            Level::ERROR,
            "zk_error",
            error_id = %structured.error_id,
            category = %structured.category,
            severity = %structured.severity,
            correlation_id = ?context.correlation_id,
            proof_type = ?context.proof_type,
            operation = ?context.operation,
            stage = ?context.stage,
        );

        let _enter = span.enter();

        match severity {
            ErrorSeverity::Debug | ErrorSeverity::Info => {
                info!(
                    message = %message,
                    error = %structured.to_json(),
                    "ZK operation info"
                );
            }
            ErrorSeverity::Warning => {
                warn!(
                    message = %message,
                    error = %structured.to_json(),
                    "ZK operation warning"
                );
            }
            ErrorSeverity::Error | ErrorSeverity::Critical | ErrorSeverity::Fatal => {
                error!(
                    message = %message,
                    error = %structured.to_json(),
                    "ZK operation error"
                );
            }
        }

        // Send to Sentry if configured and severity is high enough
        if matches!(
            severity,
            ErrorSeverity::Error | ErrorSeverity::Critical | ErrorSeverity::Fatal
        ) {
            self.send_to_sentry(&structured);
        }
    }

    /// Track a warning
    pub fn warn(&self, category: ZkErrorCategory, context: ErrorContext, message: impl Into<String>) {
        self.track_error(category, ErrorSeverity::Warning, context, message.into());
    }

    /// Track an error
    pub fn error<E: std::fmt::Display>(
        &self,
        category: ZkErrorCategory,
        context: ErrorContext,
        error: E,
    ) {
        self.track_error(category, ErrorSeverity::Error, context, error);
    }

    /// Track a critical error
    pub fn critical<E: std::fmt::Display>(
        &self,
        category: ZkErrorCategory,
        context: ErrorContext,
        error: E,
    ) {
        self.track_error(category, ErrorSeverity::Critical, context, error);
    }

    /// Track proof generation failure
    pub fn proof_generation_failed<E: std::fmt::Display>(
        &self,
        context: ErrorContext,
        error: E,
    ) {
        self.track_error(
            ZkErrorCategory::ProofGeneration,
            ErrorSeverity::Error,
            context.with_operation("proof_generation"),
            error,
        );
    }

    /// Track verification failure
    pub fn verification_failed<E: std::fmt::Display>(
        &self,
        context: ErrorContext,
        error: E,
    ) {
        self.track_error(
            ZkErrorCategory::VerificationFailed,
            ErrorSeverity::Error,
            context.with_operation("verification"),
            error,
        );
    }

    /// Track circuit compilation failure
    pub fn circuit_compilation_failed<E: std::fmt::Display>(
        &self,
        context: ErrorContext,
        error: E,
    ) {
        self.track_error(
            ZkErrorCategory::CircuitCompilation,
            ErrorSeverity::Error,
            context.with_operation("circuit_compilation"),
            error,
        );
    }

    /// Track witness mismatch
    pub fn witness_mismatch<E: std::fmt::Display>(
        &self,
        context: ErrorContext,
        error: E,
    ) {
        self.track_error(
            ZkErrorCategory::WitnessMismatch,
            ErrorSeverity::Error,
            context,
            error,
        );
    }

    /// Track resource exhaustion
    pub fn resource_exhaustion(&self, context: ErrorContext, resource: &str) {
        self.track_error(
            ZkErrorCategory::ResourceExhaustion,
            ErrorSeverity::Critical,
            context.with_tag("resource", resource),
            format!("Resource exhaustion: {}", resource),
        );
    }

    /// Generate a new correlation ID
    pub fn new_correlation_id(&self) -> String {
        format!(
            "{}-{}-{}",
            self.instance_id,
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            self.error_counter.fetch_add(1, Ordering::SeqCst)
        )
    }

    /// Send error to Sentry
    fn send_to_sentry(&self, _error: &StructuredError) {
        // Sentry integration would go here
        // This is a placeholder for when sentry-rust is added as a dependency
        #[cfg(feature = "sentry")]
        {
            if self.sentry_dsn.is_some() {
                // sentry::capture_message(&error.message, sentry::Level::Error);
            }
        }
    }
}

// ============================================================================
// ERROR AGGREGATOR
// ============================================================================

/// Aggregates errors for summary reporting
#[derive(Debug, Default)]
pub struct ErrorAggregator {
    /// Errors by fingerprint
    errors: HashMap<String, AggregatedError>,
    /// Window start time
    window_start: Option<SystemTime>,
    /// Window duration
    window_duration: Duration,
}

#[derive(Debug, Clone)]
pub struct AggregatedError {
    pub category: String,
    pub severity: String,
    pub message: String,
    pub count: u64,
    pub first_seen: SystemTime,
    pub last_seen: SystemTime,
    pub sample_context: ErrorContext,
}

impl ErrorAggregator {
    /// Create a new aggregator with specified window
    pub fn new(window_duration: Duration) -> Self {
        Self {
            errors: HashMap::new(),
            window_start: None,
            window_duration,
        }
    }

    /// Add an error to the aggregation
    pub fn add(&mut self, error: &StructuredError) {
        let now = SystemTime::now();

        // Reset window if expired
        if let Some(start) = self.window_start {
            if now.duration_since(start).unwrap_or_default() > self.window_duration {
                self.errors.clear();
                self.window_start = Some(now);
            }
        } else {
            self.window_start = Some(now);
        }

        // Aggregate by fingerprint
        self.errors
            .entry(error.fingerprint.clone())
            .and_modify(|agg| {
                agg.count += 1;
                agg.last_seen = now;
            })
            .or_insert_with(|| AggregatedError {
                category: error.category.clone(),
                severity: error.severity.clone(),
                message: error.message.clone(),
                count: 1,
                first_seen: now,
                last_seen: now,
                sample_context: error.context.clone(),
            });
    }

    /// Get top errors by count
    pub fn top_errors(&self, n: usize) -> Vec<&AggregatedError> {
        let mut errors: Vec<_> = self.errors.values().collect();
        errors.sort_by(|a, b| b.count.cmp(&a.count));
        errors.truncate(n);
        errors
    }

    /// Get error summary
    pub fn summary(&self) -> ErrorSummary {
        let total_errors: u64 = self.errors.values().map(|e| e.count).sum();
        let unique_errors = self.errors.len();

        let by_category: HashMap<String, u64> = self
            .errors
            .values()
            .fold(HashMap::new(), |mut acc, e| {
                *acc.entry(e.category.clone()).or_insert(0) += e.count;
                acc
            });

        let by_severity: HashMap<String, u64> = self
            .errors
            .values()
            .fold(HashMap::new(), |mut acc, e| {
                *acc.entry(e.severity.clone()).or_insert(0) += e.count;
                acc
            });

        ErrorSummary {
            total_errors,
            unique_errors,
            by_category,
            by_severity,
            window_duration: self.window_duration,
        }
    }
}

/// Summary of aggregated errors
#[derive(Debug, Clone, Serialize)]
pub struct ErrorSummary {
    pub total_errors: u64,
    pub unique_errors: usize,
    pub by_category: HashMap<String, u64>,
    pub by_severity: HashMap<String, u64>,
    pub window_duration: Duration,
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Generate a unique error ID
fn generate_error_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();

    let counter = COUNTER.fetch_add(1, Ordering::SeqCst);

    format!("err-{:x}-{:04x}", timestamp, counter & 0xFFFF)
}

/// Generate a short instance ID
fn generate_short_id() -> String {
    use std::time::UNIX_EPOCH;
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();

    format!("{:08x}", (timestamp & 0xFFFFFFFF) as u32)
}

/// Generate a fingerprint for error deduplication
fn generate_fingerprint(
    category: &ZkErrorCategory,
    message: &str,
    context: &ErrorContext,
) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();

    category.as_str().hash(&mut hasher);

    // Normalize message (remove numbers, hashes, etc.)
    let normalized_message = message
        .chars()
        .filter(|c| c.is_alphabetic() || c.is_whitespace())
        .collect::<String>();
    normalized_message.hash(&mut hasher);

    // Include relevant context in fingerprint
    context.operation.hash(&mut hasher);
    context.stage.hash(&mut hasher);
    context.proof_type.hash(&mut hasher);

    format!("{:016x}", hasher.finish())
}

/// Parse proof type from string
fn parse_proof_type(s: &str) -> Option<super::zk_metrics::ProofType> {
    use super::zk_metrics::ProofType;

    match s.to_lowercase().as_str() {
        "schnorr" => Some(ProofType::Schnorr),
        "ring_lwe" | "ringlwe" => Some(ProofType::RingLWE),
        "bulletproofs" => Some(ProofType::Bulletproofs),
        "groth16" => Some(ProofType::Groth16),
        "plonk" => Some(ProofType::Plonk),
        "halo2" => Some(ProofType::Halo2),
        "nova" => Some(ProofType::Nova),
        "quantum_lattice" => Some(ProofType::QuantumLattice),
        "hybrid_zk_lattice" => Some(ProofType::HybridZkLattice),
        "discrete_log" => Some(ProofType::DiscreteLog),
        "range_proof" => Some(ProofType::RangeProof),
        "membership_proof" => Some(ProofType::MembershipProof),
        _ => Some(ProofType::Custom),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_context_builder() {
        let ctx = ErrorContext::new()
            .with_correlation_id("test-123")
            .with_proof_type("schnorr")
            .with_operation("proof_generation")
            .with_stage("witness_validation")
            .with_circuit_size(1024);

        assert_eq!(ctx.correlation_id, Some("test-123".to_string()));
        assert_eq!(ctx.proof_type, Some("schnorr".to_string()));
        assert_eq!(ctx.circuit_size, Some(1024));
    }

    #[test]
    fn test_structured_error_creation() {
        let ctx = ErrorContext::new().with_proof_type("schnorr");
        let error = StructuredError::new(
            ZkErrorCategory::ProofGeneration,
            ErrorSeverity::Error,
            "Test error",
            ctx,
        );

        assert!(!error.error_id.is_empty());
        assert_eq!(error.category, "proof_generation");
        assert_eq!(error.severity, "error");
    }

    #[test]
    fn test_structured_error_to_json() {
        let ctx = ErrorContext::new();
        let error = StructuredError::new(
            ZkErrorCategory::InvalidInput,
            ErrorSeverity::Warning,
            "Invalid input",
            ctx,
        );

        let json = error.to_json();
        assert!(json.contains("invalid_input"));
        assert!(json.contains("warning"));
    }

    #[test]
    fn test_error_aggregator() {
        let mut aggregator = ErrorAggregator::new(Duration::from_secs(60));

        let ctx = ErrorContext::new();
        let error = StructuredError::new(
            ZkErrorCategory::ProofGeneration,
            ErrorSeverity::Error,
            "Test error",
            ctx,
        );

        aggregator.add(&error);
        aggregator.add(&error);
        aggregator.add(&error);

        let summary = aggregator.summary();
        assert_eq!(summary.total_errors, 3);
        assert_eq!(summary.unique_errors, 1);
    }

    #[test]
    fn test_fingerprint_stability() {
        let ctx1 = ErrorContext::new()
            .with_operation("proof_generation")
            .with_proof_type("schnorr");

        let ctx2 = ErrorContext::new()
            .with_operation("proof_generation")
            .with_proof_type("schnorr");

        let fp1 = generate_fingerprint(&ZkErrorCategory::ProofGeneration, "Error occurred", &ctx1);
        let fp2 = generate_fingerprint(&ZkErrorCategory::ProofGeneration, "Error occurred", &ctx2);

        assert_eq!(fp1, fp2);
    }

    #[test]
    fn test_loki_line_format() {
        let ctx = ErrorContext::new()
            .with_correlation_id("corr-123")
            .with_proof_type("schnorr")
            .with_operation("verification");

        let error = StructuredError::new(
            ZkErrorCategory::VerificationFailed,
            ErrorSeverity::Error,
            "Verification failed",
            ctx,
        );

        let line = error.to_loki_line();
        assert!(line.contains("level=error"));
        assert!(line.contains("correlation_id=corr-123"));
        assert!(line.contains("proof_type=schnorr"));
    }
}
