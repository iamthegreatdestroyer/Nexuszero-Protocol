//! Zero-Knowledge Proof System Metrics
//!
//! Comprehensive observability for ZK proof generation, verification,
//! circuit compilation, and error tracking.
//!
//! # Metrics Categories
//!
//! - **Proof Generation**: Timing, throughput, success rates
//! - **Verification**: Latency distributions, batch performance
//! - **Circuit Compilation**: Build times, constraint counts, optimization
//! - **Error Tracking**: Categorized failures, retry patterns
//! - **Resource Usage**: Memory, CPU, GPU utilization
//!
//! # Example
//!
//! ```rust,no_run
//! use nexuszero_crypto::metrics::zk_metrics::{ZkMetrics, ProofType};
//!
//! let metrics = ZkMetrics::global();
//!
//! // Track proof generation
//! let timer = metrics.proof_generation_timer(ProofType::Schnorr);
//! // ... generate proof ...
//! timer.observe_duration();
//!
//! // Record verification
//! metrics.record_verification(ProofType::RingLWE, 0.045, true);
//! ```

use lazy_static::lazy_static;
use prometheus::{
    Gauge, GaugeVec, Histogram, HistogramOpts, HistogramTimer,
    HistogramVec, IntCounterVec, IntGauge, IntGaugeVec, Opts, Registry,
};

// ============================================================================
// METRIC LABELS
// ============================================================================

/// Proof types for metric labeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProofType {
    Schnorr,
    RingLWE,
    Bulletproofs,
    Groth16,
    Plonk,
    Halo2,
    Nova,
    QuantumLattice,
    HybridZkLattice,
    DiscreteLog,
    RangeProof,
    MembershipProof,
    Custom,
}

impl ProofType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ProofType::Schnorr => "schnorr",
            ProofType::RingLWE => "ring_lwe",
            ProofType::Bulletproofs => "bulletproofs",
            ProofType::Groth16 => "groth16",
            ProofType::Plonk => "plonk",
            ProofType::Halo2 => "halo2",
            ProofType::Nova => "nova",
            ProofType::QuantumLattice => "quantum_lattice",
            ProofType::HybridZkLattice => "hybrid_zk_lattice",
            ProofType::DiscreteLog => "discrete_log",
            ProofType::RangeProof => "range_proof",
            ProofType::MembershipProof => "membership_proof",
            ProofType::Custom => "custom",
        }
    }
}

/// Security levels for metric labeling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SecurityLevel {
    Bit80,
    Bit128,
    Bit192,
    Bit256,
    PostQuantum128,
    PostQuantum256,
}

impl SecurityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            SecurityLevel::Bit80 => "80_bit",
            SecurityLevel::Bit128 => "128_bit",
            SecurityLevel::Bit192 => "192_bit",
            SecurityLevel::Bit256 => "256_bit",
            SecurityLevel::PostQuantum128 => "pq_128",
            SecurityLevel::PostQuantum256 => "pq_256",
        }
    }
}

/// Error categories for tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ZkErrorCategory {
    /// Invalid input parameters
    InvalidInput,
    /// Witness doesn't satisfy constraints
    WitnessMismatch,
    /// Circuit compilation failure
    CircuitCompilation,
    /// Proof generation failure
    ProofGeneration,
    /// Verification failure
    VerificationFailed,
    /// Trusted setup error
    TrustedSetup,
    /// Compression/decompression error
    Compression,
    /// Resource exhaustion (memory, time)
    ResourceExhaustion,
    /// Cryptographic error (hash, commitment)
    CryptoError,
    /// Network/serialization error
    Serialization,
    /// Internal/unknown error
    Internal,
}

impl ZkErrorCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            ZkErrorCategory::InvalidInput => "invalid_input",
            ZkErrorCategory::WitnessMismatch => "witness_mismatch",
            ZkErrorCategory::CircuitCompilation => "circuit_compilation",
            ZkErrorCategory::ProofGeneration => "proof_generation",
            ZkErrorCategory::VerificationFailed => "verification_failed",
            ZkErrorCategory::TrustedSetup => "trusted_setup",
            ZkErrorCategory::Compression => "compression",
            ZkErrorCategory::ResourceExhaustion => "resource_exhaustion",
            ZkErrorCategory::CryptoError => "crypto_error",
            ZkErrorCategory::Serialization => "serialization",
            ZkErrorCategory::Internal => "internal",
        }
    }
}

// ============================================================================
// METRICS DEFINITIONS
// ============================================================================

lazy_static! {
    /// Global ZK metrics registry
    pub static ref ZK_REGISTRY: Registry = Registry::new_custom(
        Some("nexuszero".to_string()),
        Some(std::collections::HashMap::from([
            ("component".to_string(), "zk_system".to_string())
        ]))
    ).unwrap();

    // ========================================================================
    // PROOF GENERATION METRICS
    // ========================================================================

    /// Total proofs generated (counter)
    pub static ref PROOFS_GENERATED_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("zk_proofs_generated_total", "Total number of proofs generated")
            .namespace("nexuszero"),
        &["proof_type", "security_level", "compressed"]
    ).unwrap();

    /// Proof generation duration histogram
    pub static ref PROOF_GENERATION_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_proof_generation_duration_seconds",
            "Time to generate ZK proofs"
        )
        .namespace("nexuszero")
        // Buckets: 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s
        .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]),
        &["proof_type", "security_level"]
    ).unwrap();

    /// Proof size histogram (bytes)
    pub static ref PROOF_SIZE_BYTES: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_proof_size_bytes",
            "Size of generated proofs in bytes"
        )
        .namespace("nexuszero")
        // Buckets: 64B, 256B, 1KB, 4KB, 16KB, 64KB, 256KB, 1MB
        .buckets(vec![64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0, 262144.0, 1048576.0]),
        &["proof_type", "compressed"]
    ).unwrap();

    /// Compression ratio achieved
    pub static ref COMPRESSION_RATIO: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_compression_ratio",
            "Proof compression ratio (original/compressed)"
        )
        .namespace("nexuszero")
        // Buckets: 1.0 (no compression), 1.5, 2.0, 3.0, 5.0, 10.0, 20.0
        .buckets(vec![1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]),
        &["proof_type"]
    ).unwrap();

    /// Active proof generations (in-flight)
    pub static ref ACTIVE_PROOF_GENERATIONS: IntGaugeVec = IntGaugeVec::new(
        Opts::new("zk_active_proof_generations", "Currently active proof generations")
            .namespace("nexuszero"),
        &["proof_type"]
    ).unwrap();

    /// Proof generation queue depth
    pub static ref PROOF_QUEUE_DEPTH: IntGauge = IntGauge::new(
        "nexuszero_zk_proof_queue_depth",
        "Number of proofs waiting in generation queue"
    ).unwrap();

    // ========================================================================
    // VERIFICATION METRICS
    // ========================================================================

    /// Total verifications performed
    pub static ref VERIFICATIONS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("zk_verifications_total", "Total verification attempts")
            .namespace("nexuszero"),
        &["proof_type", "result"]  // result: "valid", "invalid", "error"
    ).unwrap();

    /// Verification duration histogram
    pub static ref VERIFICATION_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_verification_duration_seconds",
            "Time to verify ZK proofs"
        )
        .namespace("nexuszero")
        // Target: <50ms. Buckets: 0.1ms, 0.5ms, 1ms, 5ms, 10ms, 25ms, 50ms, 100ms, 250ms
        .buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]),
        &["proof_type", "batch_size"]
    ).unwrap();

    /// Batch verification size distribution
    pub static ref BATCH_VERIFICATION_SIZE: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "zk_batch_verification_size",
            "Number of proofs in batch verification"
        )
        .namespace("nexuszero")
        .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    ).unwrap();

    // ========================================================================
    // CIRCUIT COMPILATION METRICS
    // ========================================================================

    /// Circuit compilation duration
    pub static ref CIRCUIT_COMPILATION_DURATION: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_circuit_compilation_seconds",
            "Time to compile circuits"
        )
        .namespace("nexuszero")
        // Buckets: 10ms, 50ms, 100ms, 500ms, 1s, 5s, 10s, 30s, 60s
        .buckets(vec![0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]),
        &["circuit_type"]
    ).unwrap();

    /// Circuit constraint count
    pub static ref CIRCUIT_CONSTRAINTS: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_circuit_constraints",
            "Number of constraints in compiled circuits"
        )
        .namespace("nexuszero")
        // Buckets: 100, 1K, 10K, 100K, 1M, 10M
        .buckets(vec![100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]),
        &["circuit_type"]
    ).unwrap();

    /// Circuit gate count
    pub static ref CIRCUIT_GATES: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_circuit_gates",
            "Number of gates in compiled circuits"
        )
        .namespace("nexuszero")
        .buckets(vec![100.0, 1000.0, 10000.0, 100000.0, 1000000.0, 10000000.0]),
        &["circuit_type", "gate_type"]
    ).unwrap();

    /// Compiled circuits cached
    pub static ref CIRCUIT_CACHE_SIZE: IntGauge = IntGauge::new(
        "nexuszero_zk_circuit_cache_size",
        "Number of compiled circuits in cache"
    ).unwrap();

    /// Circuit cache hits/misses
    pub static ref CIRCUIT_CACHE_HITS: IntCounterVec = IntCounterVec::new(
        Opts::new("zk_circuit_cache_operations", "Circuit cache hit/miss")
            .namespace("nexuszero"),
        &["result"]  // "hit", "miss"
    ).unwrap();

    // ========================================================================
    // ERROR TRACKING METRICS
    // ========================================================================

    /// Total errors by category
    pub static ref ZK_ERRORS_TOTAL: IntCounterVec = IntCounterVec::new(
        Opts::new("zk_errors_total", "Total ZK system errors by category")
            .namespace("nexuszero"),
        &["category", "proof_type", "severity"]  // severity: "warning", "error", "critical"
    ).unwrap();

    /// Error rate (derived from errors/operations)
    pub static ref ERROR_RATE: GaugeVec = GaugeVec::new(
        Opts::new("zk_error_rate", "Rolling error rate percentage")
            .namespace("nexuszero"),
        &["operation"]  // "generation", "verification", "compilation"
    ).unwrap();

    /// Retry attempts
    pub static ref RETRY_ATTEMPTS: IntCounterVec = IntCounterVec::new(
        Opts::new("zk_retry_attempts_total", "Number of retry attempts")
            .namespace("nexuszero"),
        &["operation", "attempt_number"]
    ).unwrap();

    /// Last error timestamp (for alerting)
    pub static ref LAST_ERROR_TIMESTAMP: GaugeVec = GaugeVec::new(
        Opts::new("zk_last_error_timestamp_seconds", "Unix timestamp of last error")
            .namespace("nexuszero"),
        &["category"]
    ).unwrap();

    // ========================================================================
    // RESOURCE USAGE METRICS
    // ========================================================================

    /// Peak memory during proof generation
    pub static ref PROOF_MEMORY_BYTES: HistogramVec = HistogramVec::new(
        HistogramOpts::new(
            "zk_proof_memory_bytes",
            "Peak memory usage during proof operations"
        )
        .namespace("nexuszero")
        // Buckets: 1MB, 10MB, 100MB, 500MB, 1GB, 2GB, 4GB
        .buckets(vec![
            1048576.0, 10485760.0, 104857600.0, 524288000.0,
            1073741824.0, 2147483648.0, 4294967296.0
        ]),
        &["operation"]
    ).unwrap();

    /// GPU utilization during proof generation
    pub static ref GPU_UTILIZATION: Gauge = Gauge::new(
        "nexuszero_zk_gpu_utilization_percent",
        "GPU utilization percentage during ZK operations"
    ).unwrap();

    /// GPU memory usage
    pub static ref GPU_MEMORY_BYTES: Gauge = Gauge::new(
        "nexuszero_zk_gpu_memory_bytes",
        "GPU memory usage in bytes"
    ).unwrap();

    /// Trusted setup progress (for ceremonies)
    pub static ref TRUSTED_SETUP_PROGRESS: GaugeVec = GaugeVec::new(
        Opts::new("zk_trusted_setup_progress", "Trusted setup ceremony progress")
            .namespace("nexuszero"),
        &["ceremony_id", "phase"]
    ).unwrap();

    // ========================================================================
    // OPTIMIZATION METRICS
    // ========================================================================

    /// Neural optimizer usage
    pub static ref NEURAL_OPTIMIZATION_USED: IntCounterVec = IntCounterVec::new(
        Opts::new("zk_neural_optimization_total", "Neural optimization usage")
            .namespace("nexuszero"),
        &["decision"]  // "used", "skipped"
    ).unwrap();

    /// Neural optimization speedup factor
    pub static ref NEURAL_SPEEDUP_FACTOR: Histogram = Histogram::with_opts(
        HistogramOpts::new(
            "zk_neural_speedup_factor",
            "Speedup achieved by neural optimization"
        )
        .namespace("nexuszero")
        .buckets(vec![0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0])
    ).unwrap();

    /// Parameter optimization decisions
    pub static ref PARAM_OPTIMIZATION_SOURCE: IntCounterVec = IntCounterVec::new(
        Opts::new("zk_param_optimization_source", "Source of parameter optimization")
            .namespace("nexuszero"),
        &["source"]  // "neural", "heuristic", "default", "cached"
    ).unwrap();
}

// ============================================================================
// METRICS API
// ============================================================================

/// High-level API for ZK metrics collection
pub struct ZkMetrics {
    /// Whether metrics are enabled
    enabled: bool,
}

impl ZkMetrics {
    /// Get the global ZK metrics instance
    pub fn global() -> &'static ZkMetrics {
        static INSTANCE: ZkMetrics = ZkMetrics { enabled: true };
        &INSTANCE
    }

    /// Initialize and register all metrics
    pub fn init() -> Result<(), prometheus::Error> {
        // Register all metrics with the ZK registry
        ZK_REGISTRY.register(Box::new(PROOFS_GENERATED_TOTAL.clone()))?;
        ZK_REGISTRY.register(Box::new(PROOF_GENERATION_DURATION.clone()))?;
        ZK_REGISTRY.register(Box::new(PROOF_SIZE_BYTES.clone()))?;
        ZK_REGISTRY.register(Box::new(COMPRESSION_RATIO.clone()))?;
        ZK_REGISTRY.register(Box::new(ACTIVE_PROOF_GENERATIONS.clone()))?;
        ZK_REGISTRY.register(Box::new(PROOF_QUEUE_DEPTH.clone()))?;

        ZK_REGISTRY.register(Box::new(VERIFICATIONS_TOTAL.clone()))?;
        ZK_REGISTRY.register(Box::new(VERIFICATION_DURATION.clone()))?;
        ZK_REGISTRY.register(Box::new(BATCH_VERIFICATION_SIZE.clone()))?;

        ZK_REGISTRY.register(Box::new(CIRCUIT_COMPILATION_DURATION.clone()))?;
        ZK_REGISTRY.register(Box::new(CIRCUIT_CONSTRAINTS.clone()))?;
        ZK_REGISTRY.register(Box::new(CIRCUIT_GATES.clone()))?;
        ZK_REGISTRY.register(Box::new(CIRCUIT_CACHE_SIZE.clone()))?;
        ZK_REGISTRY.register(Box::new(CIRCUIT_CACHE_HITS.clone()))?;

        ZK_REGISTRY.register(Box::new(ZK_ERRORS_TOTAL.clone()))?;
        ZK_REGISTRY.register(Box::new(ERROR_RATE.clone()))?;
        ZK_REGISTRY.register(Box::new(RETRY_ATTEMPTS.clone()))?;
        ZK_REGISTRY.register(Box::new(LAST_ERROR_TIMESTAMP.clone()))?;

        ZK_REGISTRY.register(Box::new(PROOF_MEMORY_BYTES.clone()))?;
        ZK_REGISTRY.register(Box::new(GPU_UTILIZATION.clone()))?;
        ZK_REGISTRY.register(Box::new(GPU_MEMORY_BYTES.clone()))?;
        ZK_REGISTRY.register(Box::new(TRUSTED_SETUP_PROGRESS.clone()))?;

        ZK_REGISTRY.register(Box::new(NEURAL_OPTIMIZATION_USED.clone()))?;
        ZK_REGISTRY.register(Box::new(NEURAL_SPEEDUP_FACTOR.clone()))?;
        ZK_REGISTRY.register(Box::new(PARAM_OPTIMIZATION_SOURCE.clone()))?;

        Ok(())
    }

    /// Get the ZK metrics registry for scraping
    pub fn registry() -> &'static Registry {
        &ZK_REGISTRY
    }

    // ========================================================================
    // PROOF GENERATION
    // ========================================================================

    /// Start a proof generation timer
    pub fn proof_generation_timer(
        &self,
        proof_type: ProofType,
        security_level: SecurityLevel,
    ) -> ProofGenerationGuard {
        ACTIVE_PROOF_GENERATIONS
            .with_label_values(&[proof_type.as_str()])
            .inc();

        ProofGenerationGuard {
            timer: Some(PROOF_GENERATION_DURATION
                .with_label_values(&[proof_type.as_str(), security_level.as_str()])
                .start_timer()),
            proof_type,
            security_level,
            completed: false,
        }
    }

    /// Record a completed proof generation
    pub fn record_proof_generated(
        &self,
        proof_type: ProofType,
        security_level: SecurityLevel,
        size_bytes: usize,
        compressed: bool,
        compression_ratio: f64,
    ) {
        PROOFS_GENERATED_TOTAL
            .with_label_values(&[
                proof_type.as_str(),
                security_level.as_str(),
                if compressed { "true" } else { "false" },
            ])
            .inc();

        PROOF_SIZE_BYTES
            .with_label_values(&[
                proof_type.as_str(),
                if compressed { "true" } else { "false" },
            ])
            .observe(size_bytes as f64);

        if compressed {
            COMPRESSION_RATIO
                .with_label_values(&[proof_type.as_str()])
                .observe(compression_ratio);
        }
    }

    // ========================================================================
    // VERIFICATION
    // ========================================================================

    /// Record a verification result
    pub fn record_verification(
        &self,
        proof_type: ProofType,
        duration_seconds: f64,
        valid: bool,
    ) {
        let result = if valid { "valid" } else { "invalid" };

        VERIFICATIONS_TOTAL
            .with_label_values(&[proof_type.as_str(), result])
            .inc();

        VERIFICATION_DURATION
            .with_label_values(&[proof_type.as_str(), "1"])
            .observe(duration_seconds);
    }

    /// Record batch verification
    pub fn record_batch_verification(
        &self,
        proof_type: ProofType,
        batch_size: usize,
        duration_seconds: f64,
        all_valid: bool,
    ) {
        let batch_label = match batch_size {
            1 => "1",
            2..=4 => "2-4",
            5..=16 => "5-16",
            17..=64 => "17-64",
            _ => "65+",
        };

        VERIFICATION_DURATION
            .with_label_values(&[proof_type.as_str(), batch_label])
            .observe(duration_seconds);

        BATCH_VERIFICATION_SIZE.observe(batch_size as f64);

        let result = if all_valid { "valid" } else { "invalid" };
        VERIFICATIONS_TOTAL
            .with_label_values(&[proof_type.as_str(), result])
            .inc_by(batch_size as u64);
    }

    // ========================================================================
    // CIRCUIT COMPILATION
    // ========================================================================

    /// Start a circuit compilation timer
    pub fn circuit_compilation_timer(&self, circuit_type: &str) -> HistogramTimer {
        CIRCUIT_COMPILATION_DURATION
            .with_label_values(&[circuit_type])
            .start_timer()
    }

    /// Record circuit statistics
    pub fn record_circuit_stats(
        &self,
        circuit_type: &str,
        constraints: usize,
        gates: usize,
        gate_type: &str,
    ) {
        CIRCUIT_CONSTRAINTS
            .with_label_values(&[circuit_type])
            .observe(constraints as f64);

        CIRCUIT_GATES
            .with_label_values(&[circuit_type, gate_type])
            .observe(gates as f64);
    }

    /// Record circuit cache operation
    pub fn record_circuit_cache(&self, hit: bool) {
        let result = if hit { "hit" } else { "miss" };
        CIRCUIT_CACHE_HITS.with_label_values(&[result]).inc();
    }

    /// Update circuit cache size
    pub fn set_circuit_cache_size(&self, size: i64) {
        CIRCUIT_CACHE_SIZE.set(size);
    }

    // ========================================================================
    // ERROR TRACKING
    // ========================================================================

    /// Record an error
    pub fn record_error(
        &self,
        category: ZkErrorCategory,
        proof_type: Option<ProofType>,
        severity: &str,
    ) {
        let proof_type_str = proof_type.map_or("none", |pt| pt.as_str());

        ZK_ERRORS_TOTAL
            .with_label_values(&[category.as_str(), proof_type_str, severity])
            .inc();

        // Update last error timestamp
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();

        LAST_ERROR_TIMESTAMP
            .with_label_values(&[category.as_str()])
            .set(timestamp);
    }

    /// Record a retry attempt
    pub fn record_retry(&self, operation: &str, attempt: u32) {
        let attempt_str = match attempt {
            1 => "1",
            2 => "2",
            3 => "3",
            _ => "4+",
        };

        RETRY_ATTEMPTS
            .with_label_values(&[operation, attempt_str])
            .inc();
    }

    /// Update rolling error rate
    pub fn update_error_rate(&self, operation: &str, rate: f64) {
        ERROR_RATE.with_label_values(&[operation]).set(rate);
    }

    // ========================================================================
    // RESOURCE USAGE
    // ========================================================================

    /// Record memory usage for an operation
    pub fn record_memory_usage(&self, operation: &str, bytes: usize) {
        PROOF_MEMORY_BYTES
            .with_label_values(&[operation])
            .observe(bytes as f64);
    }

    /// Update GPU metrics
    pub fn update_gpu_metrics(&self, utilization_percent: f64, memory_bytes: f64) {
        GPU_UTILIZATION.set(utilization_percent);
        GPU_MEMORY_BYTES.set(memory_bytes);
    }

    /// Update proof queue depth
    pub fn set_proof_queue_depth(&self, depth: i64) {
        PROOF_QUEUE_DEPTH.set(depth);
    }

    // ========================================================================
    // OPTIMIZATION TRACKING
    // ========================================================================

    /// Record neural optimization decision
    pub fn record_neural_optimization(&self, used: bool, speedup: f64) {
        let decision = if used { "used" } else { "skipped" };
        NEURAL_OPTIMIZATION_USED
            .with_label_values(&[decision])
            .inc();

        if used {
            NEURAL_SPEEDUP_FACTOR.observe(speedup);
        }
    }

    /// Record parameter optimization source
    pub fn record_param_source(&self, source: &str) {
        PARAM_OPTIMIZATION_SOURCE
            .with_label_values(&[source])
            .inc();
    }
}

// ============================================================================
// GUARD TYPES
// ============================================================================

/// RAII guard for proof generation timing
pub struct ProofGenerationGuard {
    timer: Option<HistogramTimer>,
    proof_type: ProofType,
    security_level: SecurityLevel,
    completed: bool,
}

impl ProofGenerationGuard {
    /// Mark the proof generation as complete and observe duration
    pub fn complete(&mut self) {
        if let Some(timer) = self.timer.take() {
            timer.observe_duration();
        }
        self.completed = true;

        ACTIVE_PROOF_GENERATIONS
            .with_label_values(&[self.proof_type.as_str()])
            .dec();
    }

    /// Mark as failed (still observes duration)
    pub fn fail(&mut self) {
        if let Some(timer) = self.timer.take() {
            timer.observe_duration();
        }
        self.completed = true;

        ACTIVE_PROOF_GENERATIONS
            .with_label_values(&[self.proof_type.as_str()])
            .dec();

        ZK_ERRORS_TOTAL
            .with_label_values(&[
                ZkErrorCategory::ProofGeneration.as_str(),
                self.proof_type.as_str(),
                "error",
            ])
            .inc();
    }
}

impl Drop for ProofGenerationGuard {
    fn drop(&mut self) {
        if !self.completed {
            // Dropped without explicit complete/fail - treat as timeout
            ACTIVE_PROOF_GENERATIONS
                .with_label_values(&[self.proof_type.as_str()])
                .dec();

            ZK_ERRORS_TOTAL
                .with_label_values(&[
                    ZkErrorCategory::ResourceExhaustion.as_str(),
                    self.proof_type.as_str(),
                    "warning",
                ])
                .inc();
        }
    }
}

// ============================================================================
// COLLECTOR FOR DETAILED STAGE METRICS
// ============================================================================

/// Collector for detailed proof generation stage metrics
pub struct StageMetricsCollector {
    stages: Vec<(String, std::time::Duration)>,
    start_time: std::time::Instant,
    current_stage: Option<(String, std::time::Instant)>,
}

impl StageMetricsCollector {
    /// Create a new stage metrics collector
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            start_time: std::time::Instant::now(),
            current_stage: None,
        }
    }

    /// Start a new stage
    pub fn start_stage(&mut self, name: &str) {
        // End previous stage if active
        if let Some((stage_name, start)) = self.current_stage.take() {
            self.stages.push((stage_name, start.elapsed()));
        }

        self.current_stage = Some((name.to_string(), std::time::Instant::now()));
    }

    /// End the current stage
    pub fn end_stage(&mut self) {
        if let Some((stage_name, start)) = self.current_stage.take() {
            self.stages.push((stage_name, start.elapsed()));
        }
    }

    /// Finalize and get all stage metrics
    pub fn finalize(mut self) -> Vec<(String, std::time::Duration)> {
        self.end_stage();
        self.stages
    }

    /// Get total elapsed time
    pub fn total_elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
}

impl Default for StageMetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_type_labels() {
        assert_eq!(ProofType::Schnorr.as_str(), "schnorr");
        assert_eq!(ProofType::RingLWE.as_str(), "ring_lwe");
        assert_eq!(ProofType::Groth16.as_str(), "groth16");
    }

    #[test]
    fn test_security_level_labels() {
        assert_eq!(SecurityLevel::Bit128.as_str(), "128_bit");
        assert_eq!(SecurityLevel::PostQuantum256.as_str(), "pq_256");
    }

    #[test]
    fn test_error_category_labels() {
        assert_eq!(ZkErrorCategory::InvalidInput.as_str(), "invalid_input");
        assert_eq!(ZkErrorCategory::VerificationFailed.as_str(), "verification_failed");
    }

    #[test]
    fn test_stage_metrics_collector() {
        let mut collector = StageMetricsCollector::new();

        collector.start_stage("validation");
        std::thread::sleep(std::time::Duration::from_millis(10));
        collector.end_stage();

        collector.start_stage("generation");
        std::thread::sleep(std::time::Duration::from_millis(20));
        collector.end_stage();

        let stages = collector.finalize();
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].0, "validation");
        assert_eq!(stages[1].0, "generation");
    }

    #[test]
    fn test_proof_generation_guard_complete() {
        let metrics = ZkMetrics::global();
        let guard = metrics.proof_generation_timer(ProofType::Schnorr, SecurityLevel::Bit128);
        guard.complete();
        // Should not panic
    }

    #[test]
    fn test_metrics_recording() {
        let metrics = ZkMetrics::global();

        // Record various metrics
        metrics.record_proof_generated(
            ProofType::Schnorr,
            SecurityLevel::Bit128,
            1024,
            true,
            2.5,
        );

        metrics.record_verification(ProofType::Schnorr, 0.025, true);
        metrics.record_circuit_cache(true);
        metrics.record_error(ZkErrorCategory::InvalidInput, Some(ProofType::Schnorr), "warning");
        metrics.record_neural_optimization(true, 1.5);
    }
}
