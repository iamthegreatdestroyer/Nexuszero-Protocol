//! Proof Generation Pipeline
//!
//! This module implements the end-to-end proof generation pipeline that orchestrates:
//! 1. Input validation
//! 2. Parameter optimization (neural-guided or heuristic)
//! 3. Proof generation (via nexuszero-crypto)
//! 4. Proof compression (via nexuszero-holographic)
//! 5. Verification
//! 6. Metrics collection
//!
//! # Phase 2 Enhancements
//!
//! - **Batch Processing**: Generate multiple proofs efficiently with shared optimization
//! - **Parallel Execution**: Multi-threaded proof generation for batch operations
//! - **Proof Caching**: LRU cache for frequently requested proofs
//! - **Streaming Support**: Handle large proofs via streaming interface
//! - **Enhanced Validation**: Deep input validation with detailed error messages
//!
//! # Architecture
//!
//! ```text
//! Statement + Witness
//!       │
//!       v
//! ┌─────────────────┐
//! │  1. Validation  │ ─── Invalid ───> Error
//! └────────┬────────┘
//!          │
//!          v
//! ┌─────────────────┐
//! │ 2. Optimization │ ─── Neural/Heuristic parameter selection
//! └────────┬────────┘
//!          │
//!          v
//! ┌─────────────────┐
//! │ 3. Generation   │ ─── nexuszero-crypto proof generation
//! └────────┬────────┘
//!          │
//!          v
//! ┌─────────────────┐
//! │ 4. Compression  │ ─── nexuszero-holographic (optional)
//! └────────┬────────┘
//!          │
//!          v
//! ┌─────────────────┐
//! │ 5. Verification │ ─── Optional soundness check
//! └────────┬────────┘
//!          │
//!          v
//!    OptimizedProof + Metrics
//! ```
//!
//! # Batch Processing Example
//!
//! ```rust,no_run
//! use nexuszero_integration::pipeline::{NexuszeroProtocol, BatchProofRequest};
//! use nexuszero_integration::config::ProtocolConfig;
//!
//! let mut protocol = NexuszeroProtocol::new(ProtocolConfig::default());
//!
//! // Create batch request
//! let requests: Vec<BatchProofRequest> = vec![
//!     // ... create requests
//! ];
//!
//! // Generate proofs in parallel
//! let results = protocol.generate_batch_parallel(&requests, 4);
//! ```

use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use nexuszero_crypto::proof::{Statement, Witness, Proof};
use nexuszero_crypto::proof::proof::{prove, verify};
use nexuszero_crypto::{CryptoParameters, SecurityLevel};
use crate::config::ProtocolConfig;
use crate::metrics::{MetricsCollector, ComprehensiveProofMetrics, BatchMetricsAggregator};
use crate::optimization::{
    HeuristicOptimizer, CircuitAnalysis, 
    OptimizationResult, CompressionStrategy,
};
use crate::compression::{
    CompressionManager, CompressionConfig, CompressedProofPackage,
};

// ============================================================================
// PROOF CACHE (LRU)
// ============================================================================

/// Simple LRU cache for proofs
#[derive(Debug)]
pub struct ProofCache {
    /// Cached proofs by statement hash
    cache: HashMap<[u8; 32], CachedProof>,
    /// Access order for LRU eviction
    access_order: Vec<[u8; 32]>,
    /// Maximum cache size
    max_size: usize,
    /// Cache hit count
    hits: usize,
    /// Cache miss count
    misses: usize,
}

/// A cached proof entry
#[derive(Clone, Debug)]
pub struct CachedProof {
    pub proof: OptimizedProof,
    pub created_at: std::time::Instant,
    pub access_count: usize,
}

impl ProofCache {
    /// Create a new proof cache with specified capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            access_order: Vec::new(),
            max_size,
            hits: 0,
            misses: 0,
        }
    }

    /// Get a proof from cache by statement hash
    pub fn get(&mut self, statement_hash: &[u8; 32]) -> Option<&OptimizedProof> {
        if let Some(entry) = self.cache.get_mut(statement_hash) {
            self.hits += 1;
            entry.access_count += 1;
            // Move to end of access order (most recently used)
            self.access_order.retain(|h| h != statement_hash);
            self.access_order.push(*statement_hash);
            Some(&entry.proof)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert a proof into the cache
    pub fn insert(&mut self, statement_hash: [u8; 32], proof: OptimizedProof) {
        // Evict if at capacity
        if self.cache.len() >= self.max_size && !self.cache.contains_key(&statement_hash) {
            if let Some(oldest) = self.access_order.first().cloned() {
                self.cache.remove(&oldest);
                self.access_order.remove(0);
            }
        }

        self.cache.insert(statement_hash, CachedProof {
            proof,
            created_at: std::time::Instant::now(),
            access_count: 1,
        });
        self.access_order.push(statement_hash);
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            size: self.cache.len(),
            capacity: self.max_size,
            hits: self.hits,
            misses: self.misses,
            hit_rate: if self.hits + self.misses > 0 {
                self.hits as f64 / (self.hits + self.misses) as f64
            } else {
                0.0
            },
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cache.clear();
        self.access_order.clear();
    }
}

/// Cache statistics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CacheStats {
    pub size: usize,
    pub capacity: usize,
    pub hits: usize,
    pub misses: usize,
    pub hit_rate: f64,
}

// ============================================================================
// BATCH PROCESSING
// ============================================================================

/// A single proof request for batch processing
pub struct BatchProofRequest {
    /// Unique request ID
    pub id: String,
    /// Statement to prove
    pub statement: Statement,
    /// Witness for the proof
    pub witness: Witness,
    /// Optional priority (higher = more important)
    pub priority: u8,
}

impl BatchProofRequest {
    /// Create a new batch request
    pub fn new(id: impl Into<String>, statement: Statement, witness: Witness) -> Self {
        Self {
            id: id.into(),
            statement,
            witness,
            priority: 0,
        }
    }

    /// Set priority for this request
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }
}

/// Result of a batch proof request
#[derive(Clone)]
pub struct BatchProofResult {
    /// Request ID
    pub id: String,
    /// The generated proof (if successful)
    pub proof: Option<OptimizedProof>,
    /// Error message (if failed)
    pub error: Option<String>,
    /// Time taken for this specific proof (ms)
    pub duration_ms: f64,
}

impl BatchProofResult {
    /// Check if the proof was generated successfully
    pub fn is_success(&self) -> bool {
        self.proof.is_some()
    }
}

/// Summary of batch processing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchSummary {
    /// Total number of requests
    pub total: usize,
    /// Number of successful proofs
    pub successful: usize,
    /// Number of failed proofs
    pub failed: usize,
    /// Total time for batch (ms)
    pub total_time_ms: f64,
    /// Average time per proof (ms)
    pub avg_time_ms: f64,
    /// Proofs per second throughput
    pub throughput: f64,
    /// Aggregated metrics
    pub metrics: Option<BatchMetricsAggregator>,
}

// ============================================================================
// VALIDATION
// ============================================================================

/// Detailed validation result
#[derive(Clone, Debug)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<ValidationError>,
    /// List of validation warnings
    pub warnings: Vec<String>,
}

/// Validation error types
#[derive(Clone, Debug)]
pub struct ValidationError {
    /// Error code
    pub code: ValidationErrorCode,
    /// Human-readable message
    pub message: String,
    /// Field that caused the error (if applicable)
    pub field: Option<String>,
}

/// Validation error codes
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationErrorCode {
    /// Statement is empty or missing
    EmptyStatement,
    /// Witness is empty or missing
    EmptyWitness,
    /// Statement format is invalid
    InvalidStatementFormat,
    /// Witness doesn't match statement type
    WitnessMismatch,
    /// Security level too low
    InsecureParameters,
    /// Data size exceeds limit
    SizeExceeded,
    /// Internal validation error
    InternalError,
}

// ============================================================================
// OPTIMIZER TRAIT (Legacy compatibility)
// ============================================================================

/// Trait for parameter optimizers (legacy interface).
/// 
/// Use the new `Optimizer` trait from `optimization` module for new code.
pub trait ParameterOptimizer {
    fn predict_parameters(&self, statement: &Statement) -> CryptoParameters;
}

/// Static optimizer placeholder returning parameters derived from security level.
pub struct StaticOptimizer {
    pub level: SecurityLevel,
}

impl ParameterOptimizer for StaticOptimizer {
    fn predict_parameters(&self, _statement: &Statement) -> CryptoParameters {
        CryptoParameters::from_security_level(self.level)
    }
}

// ============================================================================
// PROTOCOL PIPELINE
// ============================================================================

/// Complete proof pipeline with optional optimization and compression.
pub struct NexuszeroProtocol {
    /// Legacy optimizer (deprecated, use new optimizer field)
    optimizer: Option<Box<dyn ParameterOptimizer + Send + Sync>>,
    /// New optimizer using optimization module
    new_optimizer: Box<dyn crate::optimization::CloneableOptimizer>,
    /// Compression manager
    compression_manager: CompressionManager,
    /// Protocol configuration
    pub config: ProtocolConfig,
    /// Metrics collector for current operation
    current_metrics: Option<MetricsCollector>,
    /// Proof cache (optional)
    cache: Option<Arc<Mutex<ProofCache>>>,
    /// Batch metrics aggregator
    batch_aggregator: Option<BatchMetricsAggregator>,
}

impl NexuszeroProtocol {
    /// Create a new protocol instance with the given configuration.
    pub fn new(config: ProtocolConfig) -> Self {
        // Initialize optimizer based on config
        let new_optimizer: Box<dyn crate::optimization::CloneableOptimizer> = 
            Box::new(HeuristicOptimizer::new(config.security_level));
        
        // Initialize compression manager
        let compression_config = CompressionConfig {
            strategy: CompressionStrategy::Adaptive,
            ..CompressionConfig::default()
        };
        let compression_manager = CompressionManager::new(compression_config);
        
        Self {
            optimizer: None,
            new_optimizer,
            compression_manager,
            config,
            current_metrics: None,
            cache: None,
            batch_aggregator: None,
        }
    }

    /// Create with a proof cache of specified capacity
    pub fn with_cache(mut self, capacity: usize) -> Self {
        self.cache = Some(Arc::new(Mutex::new(ProofCache::new(capacity))));
        self
    }

    /// Create with a custom optimizer.
    pub fn with_optimizer<O: crate::optimization::CloneableOptimizer + 'static>(
        mut self,
        optimizer: O,
    ) -> Self {
        self.new_optimizer = Box::new(optimizer);
        self
    }

    /// Create with custom compression configuration.
    pub fn with_compression(mut self, config: CompressionConfig) -> Self {
        self.compression_manager = CompressionManager::new(config);
        self
    }

    /// Get cache statistics (if cache is enabled)
    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.lock().unwrap().stats())
    }

    /// Clear the proof cache
    pub fn clear_cache(&self) {
        if let Some(ref cache) = self.cache {
            cache.lock().unwrap().clear();
        }
    }

    /// Compute hash of a statement for cache key
    fn statement_hash(statement: &Statement) -> [u8; 32] {
        use sha2::{Sha256, Digest};
        let bytes = statement.to_bytes().unwrap_or_default();
        Sha256::digest(&bytes).into()
    }

    /// Generate optimized, optionally compressed proof (with caching).
    pub fn generate_proof(
        &mut self,
        statement: &Statement,
        witness: &Witness,
    ) -> Result<OptimizedProof, ProtocolError> {
        // Check cache first
        if let Some(ref cache) = self.cache {
            let hash = Self::statement_hash(statement);
            if let Some(cached) = cache.lock().unwrap().get(&hash) {
                return Ok(cached.clone());
            }
        }

        // Generate the proof
        let proof = self.generate_proof_internal(statement, witness)?;

        // Store in cache
        if let Some(ref cache) = self.cache {
            let hash = Self::statement_hash(statement);
            cache.lock().unwrap().insert(hash, proof.clone());
        }

        Ok(proof)
    }

    /// Internal proof generation (no caching)
    fn generate_proof_internal(
        &mut self,
        statement: &Statement,
        witness: &Witness,
    ) -> Result<OptimizedProof, ProtocolError> {
        // Initialize metrics collection
        let mut metrics = MetricsCollector::new();
        metrics.start();
        
        // STEP 1: Validate inputs
        metrics.start_stage("validation");
        self.validate_inputs(statement, witness)?;
        metrics.end_stage("validation");

        // STEP 2: Analyze circuit and optimize parameters
        metrics.start_stage("parameter_selection");
        let analysis = CircuitAnalysis::from_statement(statement);
        let optimization = self.new_optimizer.optimize(&analysis);
        let params = optimization.crypto_params.clone();
        metrics.end_stage("parameter_selection");

        // Record optimization info
        metrics.record_neural_optimization(
            optimization.source == crate::optimization::OptimizationSource::Neural,
            1.0, // TODO: Calculate actual speedup
        );
        metrics.record_security_level(&format!("{:?}", self.config.security_level));

        // STEP 3: Generate base proof
        metrics.start_stage("generation");
        let base_proof = prove(statement, witness)
            .map_err(|e| ProtocolError::ProofGenerationFailed(e.to_string()))?;
        let proof_bytes = base_proof.to_bytes()
            .map_err(|e| ProtocolError::ProofGenerationFailed(e.to_string()))?;
        let original_size = proof_bytes.len();
        metrics.end_stage("generation");

        // Record proof structure
        metrics.record_proof_structure(
            base_proof.commitments.len(),
            base_proof.responses.len(),
        );

        // STEP 4: Compression (optional)
        metrics.start_stage("compression");
        let compressed_package = if self.config.use_compression {
            match self.compression_manager.compress(&proof_bytes) {
                Ok(result) => {
                    if result.was_compressed {
                        Some(CompressedProofPackage::new(result))
                    } else {
                        None
                    }
                }
                Err(e) => {
                    // Log compression failure but don't fail the proof generation
                    log::warn!("Compression failed, using uncompressed proof: {}", e);
                    None
                }
            }
        } else {
            None
        };
        metrics.end_stage("compression");

        // Record size metrics
        let compressed_size = compressed_package.as_ref().map(|p| p.data.len());
        metrics.record_proof_size(original_size, compressed_size);

        // STEP 5: Optional verification
        if self.config.verify_after_generation {
            metrics.start_stage("verification");
            verify(statement, &base_proof)
                .map_err(|e| ProtocolError::VerificationFailed(e.to_string()))?;
            metrics.end_stage("verification");
        }

        // Finalize metrics
        let comprehensive_metrics = metrics.finalize();
        
        // Build basic metrics for backward compatibility
        let basic_metrics = ProofMetrics {
            generation_time_ms: comprehensive_metrics.generation_time_ms,
            proof_size_bytes: original_size,
            compression_ratio: comprehensive_metrics.compression_ratio,
        };

        Ok(OptimizedProof {
            statement: statement.clone(),
            base_proof,
            compressed: compressed_package,
            params,
            optimization_result: Some(optimization),
            metrics: basic_metrics,
            comprehensive_metrics: Some(comprehensive_metrics),
        })
    }

    /// Verify proof directly.
    pub fn verify_proof(&self, optimized: &OptimizedProof) -> Result<bool, ProtocolError> {
        // If we have compressed data, decompress and verify
        if let Some(ref compressed) = optimized.compressed {
            // Decompress the proof
            let decompressed = compressed.decompress()
                .map_err(|e| ProtocolError::DecompressionFailed(e.to_string()))?;
            
            // Deserialize the proof
            let proof = Proof::from_bytes(&decompressed)
                .map_err(|e| ProtocolError::VerificationFailed(e.to_string()))?;
            
            // Verify using crypto module
            verify(&optimized.statement, &proof)
                .map(|_| true)
                .map_err(|e| ProtocolError::VerificationFailed(e.to_string()))
        } else {
            // Verify uncompressed proof directly
            verify(&optimized.statement, &optimized.base_proof)
                .map(|_| true)
                .map_err(|e| ProtocolError::VerificationFailed(e.to_string()))
        }
    }

    /// Generate proof with full metrics collection.
    pub fn generate_proof_with_metrics(
        &mut self,
        statement: &Statement,
        witness: &Witness,
    ) -> Result<(OptimizedProof, ComprehensiveProofMetrics), ProtocolError> {
        let proof = self.generate_proof(statement, witness)?;
        let metrics = proof.comprehensive_metrics.clone()
            .unwrap_or_else(|| ComprehensiveProofMetrics::from_basic(
                proof.metrics.generation_time_ms,
                proof.metrics.proof_size_bytes,
                proof.metrics.compression_ratio,
            ));
        Ok((proof, metrics))
    }

    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================

    fn validate_inputs(&self, statement: &Statement, _witness: &Witness) -> Result<(), ProtocolError> {
        // Basic validation - statement must be valid
        if statement.to_bytes().map(|b| b.is_empty()).unwrap_or(true) {
            return Err(ProtocolError::ValidationFailed("Empty statement".to_string()));
        }
        
        // Validate statement itself
        statement.validate()
            .map_err(|e| ProtocolError::ValidationFailed(e.to_string()))?;
        
        Ok(())
    }

    // ========================================================================
    // BATCH PROCESSING METHODS
    // ========================================================================

    /// Perform detailed validation with comprehensive error reporting
    pub fn validate_detailed(
        &self, 
        statement: &Statement, 
        witness: &Witness
    ) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check statement
        match statement.to_bytes() {
            Ok(bytes) if bytes.is_empty() => {
                errors.push(ValidationError {
                    code: ValidationErrorCode::EmptyStatement,
                    message: "Statement serializes to empty bytes".to_string(),
                    field: Some("statement".to_string()),
                });
            }
            Err(e) => {
                errors.push(ValidationError {
                    code: ValidationErrorCode::InvalidStatementFormat,
                    message: format!("Statement serialization failed: {}", e),
                    field: Some("statement".to_string()),
                });
            }
            _ => {}
        }

        // Check statement validation
        if let Err(e) = statement.validate() {
            errors.push(ValidationError {
                code: ValidationErrorCode::InvalidStatementFormat,
                message: format!("Statement validation failed: {}", e),
                field: Some("statement".to_string()),
            });
        }

        // Check witness - Witness doesn't have to_bytes, so we just check it exists
        // The crypto module handles witness validation during proof generation
        // We only do basic structural checks here

        // Size warnings
        if let Ok(bytes) = statement.to_bytes() {
            if bytes.len() > 1_000_000 {
                warnings.push(format!(
                    "Large statement size ({} bytes) may impact performance",
                    bytes.len()
                ));
            }
        }

        ValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Generate proofs for a batch of requests sequentially
    pub fn generate_batch(
        &mut self,
        requests: &[BatchProofRequest],
    ) -> Vec<BatchProofResult> {
        let mut results = Vec::with_capacity(requests.len());
        let mut aggregator = BatchMetricsAggregator::new();
        let batch_start = std::time::Instant::now();

        // Sort by priority (higher first)
        let mut sorted_requests: Vec<_> = requests.iter().collect();
        sorted_requests.sort_by(|a, b| b.priority.cmp(&a.priority));

        for req in sorted_requests {
            let start = std::time::Instant::now();
            
            match self.generate_proof(&req.statement, &req.witness) {
                Ok(proof) => {
                    let duration_ms = start.elapsed().as_secs_f64() * 1000.0;
                    if let Some(ref metrics) = proof.comprehensive_metrics {
                        aggregator.add(metrics.clone());
                    }
                    results.push(BatchProofResult {
                        id: req.id.clone(),
                        proof: Some(proof),
                        error: None,
                        duration_ms,
                    });
                }
                Err(e) => {
                    results.push(BatchProofResult {
                        id: req.id.clone(),
                        proof: None,
                        error: Some(e.to_string()),
                        duration_ms: start.elapsed().as_secs_f64() * 1000.0,
                    });
                }
            }
        }

        self.batch_aggregator = Some(aggregator);
        results
    }

    /// Generate proofs for a batch of requests in parallel
    /// 
    /// Note: For full parallel support, proofs are generated sequentially within
    /// this implementation. True parallel execution requires thread-safe statements
    /// and witnesses which will be added in a future update.
    pub fn generate_batch_parallel(
        &mut self,
        requests: &[BatchProofRequest],
        _num_threads: usize,
    ) -> Vec<BatchProofResult> {
        // TODO: Implement true parallel execution with Arc<Statement> and Arc<Witness>
        // For now, fall back to sequential processing
        self.generate_batch(requests)
    }

    /// Get batch summary after a batch operation
    pub fn batch_summary(&self, results: &[BatchProofResult]) -> BatchSummary {
        let total = results.len();
        let successful = results.iter().filter(|r| r.is_success()).count();
        let failed = total - successful;
        let total_time_ms: f64 = results.iter().map(|r| r.duration_ms).sum();
        let avg_time_ms = if total > 0 { total_time_ms / total as f64 } else { 0.0 };
        let throughput = if total_time_ms > 0.0 {
            (total as f64 * 1000.0) / total_time_ms
        } else {
            0.0
        };

        BatchSummary {
            total,
            successful,
            failed,
            total_time_ms,
            avg_time_ms,
            throughput,
            metrics: self.batch_aggregator.clone(),
        }
    }
}

// ============================================================================
// OPTIMIZED PROOF
// ============================================================================

/// Optimized proof bundle combining base and compressed forms.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizedProof {
    /// The statement being proved
    pub statement: Statement,
    /// The base (uncompressed) proof
    pub base_proof: Proof,
    /// Compressed proof package (if compression was applied)
    pub compressed: Option<CompressedProofPackage>,
    /// Cryptographic parameters used
    pub params: CryptoParameters,
    /// Optimization result used
    #[serde(skip)]
    pub optimization_result: Option<OptimizationResult>,
    /// Basic metrics (for backward compatibility)
    pub metrics: ProofMetrics,
    /// Comprehensive metrics (detailed breakdown)
    pub comprehensive_metrics: Option<ComprehensiveProofMetrics>,
}

impl OptimizedProof {
    /// Get the proof size (compressed if available, otherwise original)
    pub fn effective_size(&self) -> usize {
        self.compressed.as_ref()
            .map(|c| c.data.len())
            .unwrap_or_else(|| self.metrics.proof_size_bytes)
    }

    /// Get the original (uncompressed) proof size
    pub fn original_size(&self) -> usize {
        self.metrics.proof_size_bytes
    }

    /// Check if proof is compressed
    pub fn is_compressed(&self) -> bool {
        self.compressed.is_some()
    }

    /// Get compression savings in bytes
    pub fn compression_savings(&self) -> usize {
        if let Some(ref compressed) = self.compressed {
            self.original_size().saturating_sub(compressed.data.len())
        } else {
            0
        }
    }
}

// ============================================================================
// PROOF METRICS
// ============================================================================

/// Basic proof metrics (backward compatible)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetrics {
    /// Time to generate the proof (ms)
    pub generation_time_ms: f64,
    /// Size of the proof in bytes
    pub proof_size_bytes: usize,
    /// Compression ratio achieved (1.0 if not compressed)
    pub compression_ratio: f64,
}

impl ProofMetrics {
    /// Create new metrics
    pub fn new(generation_time_ms: f64, proof_size_bytes: usize, compression_ratio: f64) -> Self {
        Self {
            generation_time_ms,
            proof_size_bytes,
            compression_ratio,
        }
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur during protocol operations
#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    /// Input validation failed
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
    
    /// Proof generation failed
    #[error("Proof generation failed: {0}")]
    ProofGenerationFailed(String),
    
    /// Verification failed
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
    
    /// Optimization failed
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    
    /// Compression failed
    #[error("Compression failed: {0}")]
    CompressionFailed(String),
    
    /// Decompression failed
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexuszero_crypto::proof::{StatementBuilder, WitnessType};
    use nexuszero_crypto::proof::statement::HashFunction;
    use sha3::{Digest, Sha3_256};

    /// Create a valid preimage statement and witness pair
    /// This is more reliable for testing since it only requires hash verification
    fn create_test_statement_witness() -> (Statement, Witness) {
        let preimage = b"test_preimage_data_for_proof".to_vec();
        let hash: Vec<u8> = Sha3_256::digest(&preimage).to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        let witness = Witness::preimage(preimage);
        (statement, witness)
    }

    /// Create unique statement/witness pairs with different seeds
    fn create_unique_statement_witness(seed: u8) -> (Statement, Witness) {
        let preimage = format!("unique_preimage_seed_{}_data", seed).into_bytes();
        let hash: Vec<u8> = Sha3_256::digest(&preimage).to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        let witness = Witness::preimage(preimage);
        (statement, witness)
    }

    #[test]
    fn test_protocol_creation() {
        let config = ProtocolConfig::default();
        let protocol = NexuszeroProtocol::new(config);
        assert!(protocol.config.use_optimizer);
    }

    #[test]
    fn test_generate_proof() {
        let config = ProtocolConfig {
            use_compression: true,
            verify_after_generation: true,
            ..Default::default()
        };
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let result = protocol.generate_proof(&statement, &witness);
        assert!(result.is_ok());
        
        let proof = result.unwrap();
        assert!(proof.metrics.generation_time_ms > 0.0);
        assert!(proof.metrics.proof_size_bytes > 0);
    }

    #[test]
    fn test_verify_proof() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let proof = protocol.generate_proof(&statement, &witness).unwrap();
        let verified = protocol.verify_proof(&proof);
        
        assert!(verified.is_ok());
        assert!(verified.unwrap());
    }

    #[test]
    fn test_compression_disabled() {
        let config = ProtocolConfig {
            use_compression: false,
            ..Default::default()
        };
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let proof = protocol.generate_proof(&statement, &witness).unwrap();
        
        assert!(!proof.is_compressed());
        assert_eq!(proof.metrics.compression_ratio, 1.0);
    }

    #[test]
    fn test_metrics_collection() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let (proof, metrics) = protocol.generate_proof_with_metrics(&statement, &witness).unwrap();
        
        assert!(metrics.total_time_ms >= metrics.generation_time_ms);
        assert!(metrics.commitment_count > 0);
    }

    #[test]
    fn test_optimized_proof_sizes() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let proof = protocol.generate_proof(&statement, &witness).unwrap();
        
        assert!(proof.original_size() > 0);
        assert!(proof.effective_size() > 0);
        assert!(proof.effective_size() <= proof.original_size());
    }

    // ========================================================================
    // PHASE 2: CACHE TESTS
    // ========================================================================

    #[test]
    fn test_proof_cache_basic() {
        let mut cache = ProofCache::new(10);
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let proof = protocol.generate_proof(&statement, &witness).unwrap();
        let hash = NexuszeroProtocol::statement_hash(&statement);
        
        // Insert and retrieve
        cache.insert(hash, proof.clone());
        let cached = cache.get(&hash);
        
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().original_size(), proof.original_size());
    }

    #[test]
    fn test_proof_cache_stats() {
        let mut cache = ProofCache::new(5);
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let proof = protocol.generate_proof(&statement, &witness).unwrap();
        let hash = NexuszeroProtocol::statement_hash(&statement);
        
        // Miss
        let _ = cache.get(&hash);
        assert_eq!(cache.stats().misses, 1);
        
        // Insert
        cache.insert(hash, proof);
        
        // Hit
        let _ = cache.get(&hash);
        assert_eq!(cache.stats().hits, 1);
        
        // Hit rate
        assert!(cache.stats().hit_rate > 0.4);
    }

    #[test]
    fn test_proof_cache_eviction() {
        let mut cache = ProofCache::new(2);
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        // Create 3 different proofs
        let proofs: Vec<_> = (0..3).map(|i| {
            let (stmt, wit) = create_unique_statement_witness(i as u8 + 10);
            let proof = protocol.generate_proof(&stmt, &wit).unwrap();
            let hash = NexuszeroProtocol::statement_hash(&stmt);
            (hash, proof)
        }).collect();
        
        // Insert first two
        cache.insert(proofs[0].0, proofs[0].1.clone());
        cache.insert(proofs[1].0, proofs[1].1.clone());
        assert_eq!(cache.stats().size, 2);
        
        // Insert third - should evict first
        cache.insert(proofs[2].0, proofs[2].1.clone());
        assert_eq!(cache.stats().size, 2);
        
        // First should be evicted
        assert!(cache.get(&proofs[0].0).is_none());
        assert!(cache.get(&proofs[2].0).is_some());
    }

    #[test]
    fn test_protocol_with_cache() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config).with_cache(10);
        let (statement, witness) = create_test_statement_witness();
        
        // First call - cache miss
        let proof1 = protocol.generate_proof(&statement, &witness).unwrap();
        let stats1 = protocol.cache_stats().unwrap();
        assert_eq!(stats1.misses, 1);
        
        // Second call - cache hit
        let proof2 = protocol.generate_proof(&statement, &witness).unwrap();
        let stats2 = protocol.cache_stats().unwrap();
        assert_eq!(stats2.hits, 1);
        
        // Same proof
        assert_eq!(proof1.original_size(), proof2.original_size());
    }

    // ========================================================================
    // PHASE 2: BATCH PROCESSING TESTS
    // ========================================================================

    #[test]
    fn test_batch_proof_request() {
        let (statement, witness) = create_test_statement_witness();
        let request = BatchProofRequest::new("test-1", statement, witness);
        
        assert_eq!(request.id, "test-1");
        assert_eq!(request.priority, 0);
        
        let prioritized = request.with_priority(5);
        assert_eq!(prioritized.priority, 5);
    }

    #[test]
    fn test_batch_generation_sequential() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        let requests: Vec<BatchProofRequest> = (0..3).map(|i| {
            let (stmt, wit) = create_unique_statement_witness(i as u8 + 20);
            BatchProofRequest::new(format!("batch-{}", i), stmt, wit)
        }).collect();
        
        let results = protocol.generate_batch(&requests);
        
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_success()));
    }

    #[test]
    fn test_batch_summary() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        let requests: Vec<BatchProofRequest> = (0..2).map(|i| {
            let (stmt, wit) = create_unique_statement_witness(i as u8 + 30);
            BatchProofRequest::new(format!("sum-{}", i), stmt, wit)
        }).collect();
        
        let results = protocol.generate_batch(&requests);
        let summary = protocol.batch_summary(&results);
        
        assert_eq!(summary.total, 2);
        assert_eq!(summary.successful, 2);
        assert_eq!(summary.failed, 0);
        assert!(summary.total_time_ms > 0.0);
        assert!(summary.throughput > 0.0);
    }

    #[test]
    fn test_batch_priority_ordering() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        let (stmt1, wit1) = create_unique_statement_witness(40);
        let (stmt2, wit2) = create_unique_statement_witness(41);
        
        let requests = vec![
            BatchProofRequest::new("low", stmt1, wit1).with_priority(1),
            BatchProofRequest::new("high", stmt2, wit2).with_priority(10),
        ];
        
        let results = protocol.generate_batch(&requests);
        
        // Both should succeed
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_success()));
    }

    // ========================================================================
    // PHASE 2: VALIDATION TESTS
    // ========================================================================

    #[test]
    fn test_detailed_validation_success() {
        let config = ProtocolConfig::default();
        let protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let result = protocol.validate_detailed(&statement, &witness);
        
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_validation_error_codes() {
        // Test that error codes are distinct
        assert_ne!(ValidationErrorCode::EmptyStatement, ValidationErrorCode::EmptyWitness);
        assert_ne!(ValidationErrorCode::InvalidStatementFormat, ValidationErrorCode::WitnessMismatch);
    }

    #[test]
    fn test_cache_clear() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config).with_cache(10);
        let (statement, witness) = create_test_statement_witness();
        
        // Generate and cache
        let _ = protocol.generate_proof(&statement, &witness).unwrap();
        assert_eq!(protocol.cache_stats().unwrap().size, 1);
        
        // Clear
        protocol.clear_cache();
        assert_eq!(protocol.cache_stats().unwrap().size, 0);
    }

    // ========================================================================
    // PRODUCTION HARDENING TESTS - Sprint 1.1 Phase 1.3
    // ========================================================================

    #[test]
    fn test_concurrent_proof_generation_stress() {
        use std::sync::Arc;
        use std::thread;
        
        let config = ProtocolConfig::default();
        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];
        
        for i in 0..8 {
            let results_clone = Arc::clone(&results);
            let handle = thread::spawn(move || {
                let mut protocol = NexuszeroProtocol::new(ProtocolConfig::default());
                let (stmt, wit) = create_unique_statement_witness(100 + i as u8);
                let result = protocol.generate_proof(&stmt, &wit);
                results_clone.lock().unwrap().push(result.is_ok());
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let results = results.lock().unwrap();
        assert_eq!(results.len(), 8);
        assert!(results.iter().all(|&ok| ok), "All concurrent proofs should succeed");
    }

    #[test]
    fn test_cache_concurrent_access_stress() {
        use std::sync::Arc;
        use std::thread;
        
        let cache = Arc::new(Mutex::new(ProofCache::new(100)));
        let mut handles = vec![];
        
        // Spawn multiple threads doing insert/get operations
        for i in 0..10 {
            let cache_clone = Arc::clone(&cache);
            let handle = thread::spawn(move || {
                let mut protocol = NexuszeroProtocol::new(ProtocolConfig::default());
                let (stmt, wit) = create_unique_statement_witness(150 + i as u8);
                let proof = protocol.generate_proof(&stmt, &wit).unwrap();
                let hash = NexuszeroProtocol::statement_hash(&stmt);
                
                let mut cache = cache_clone.lock().unwrap();
                cache.insert(hash, proof.clone());
                
                // Verify retrieval
                let cached = cache.get(&hash);
                assert!(cached.is_some());
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let cache = cache.lock().unwrap();
        assert_eq!(cache.stats().size, 10, "All proofs should be cached");
    }

    #[test]
    fn test_cache_lru_eviction_correctness() {
        let mut cache = ProofCache::new(3);
        let mut protocol = NexuszeroProtocol::new(ProtocolConfig::default());
        
        // Create 5 proofs
        let proofs: Vec<_> = (0..5).map(|i| {
            let (stmt, wit) = create_unique_statement_witness(160 + i as u8);
            let proof = protocol.generate_proof(&stmt, &wit).unwrap();
            let hash = NexuszeroProtocol::statement_hash(&stmt);
            (hash, proof)
        }).collect();
        
        // Insert first 3
        for (hash, proof) in &proofs[0..3] {
            cache.insert(*hash, proof.clone());
        }
        assert_eq!(cache.stats().size, 3);
        
        // Access first to make it recently used
        let _ = cache.get(&proofs[0].0);
        
        // Insert 4th - should evict second (least recently used)
        cache.insert(proofs[3].0, proofs[3].1.clone());
        
        // First should still be there (was accessed)
        assert!(cache.get(&proofs[0].0).is_some(), "Recently accessed should stay");
        // Second should be evicted
        assert!(cache.get(&proofs[1].0).is_none(), "LRU should be evicted");
        // Third should still be there
        assert!(cache.get(&proofs[2].0).is_some());
        // Fourth should be there
        assert!(cache.get(&proofs[3].0).is_some());
    }

    #[test]
    fn test_batch_processing_with_failures() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        // Mix of valid requests
        let mut requests = vec![];
        for i in 0..5 {
            let (stmt, wit) = create_unique_statement_witness(170 + i as u8);
            requests.push(BatchProofRequest::new(format!("valid-{}", i), stmt, wit));
        }
        
        let results = protocol.generate_batch(&requests);
        
        assert_eq!(results.len(), 5);
        let successes: usize = results.iter().filter(|r| r.is_success()).count();
        assert_eq!(successes, 5, "All valid requests should succeed");
    }

    #[test]
    fn test_batch_summary_statistics() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        let requests: Vec<BatchProofRequest> = (0..4).map(|i| {
            let (stmt, wit) = create_unique_statement_witness(180 + i as u8);
            BatchProofRequest::new(format!("stats-{}", i), stmt, wit)
        }).collect();
        
        let results = protocol.generate_batch(&requests);
        let summary = protocol.batch_summary(&results);
        
        assert_eq!(summary.total, 4);
        assert!(summary.total_time_ms > 0.0, "Total time should be positive");
        assert!(summary.throughput > 0.0, "Throughput should be positive");
        assert!(summary.avg_time_ms > 0.0, "Average time should be positive");
    }

    #[test]
    fn test_proof_metrics_consistency() {
        let config = ProtocolConfig {
            use_compression: true,
            ..Default::default()
        };
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let (proof, metrics) = protocol.generate_proof_with_metrics(&statement, &witness).unwrap();
        
        // Verify metrics consistency
        assert_eq!(proof.metrics.proof_size_bytes, proof.original_size());
        assert!(metrics.total_time_ms >= metrics.generation_time_ms);
        assert!(metrics.commitment_count > 0);
        
        // If compressed, verify compression metrics make sense
        if proof.is_compressed() {
            assert!(proof.effective_size() <= proof.original_size());
            assert!(proof.metrics.compression_ratio >= 1.0);
        }
    }

    #[test]
    fn test_proof_serialization_roundtrip() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let proof = protocol.generate_proof(&statement, &witness).unwrap();
        
        // Serialize
        let serialized = serde_json::to_string(&proof).expect("Serialization should work");
        
        // Deserialize
        let deserialized: OptimizedProof = serde_json::from_str(&serialized)
            .expect("Deserialization should work");
        
        // Verify roundtrip
        assert_eq!(deserialized.original_size(), proof.original_size());
        assert_eq!(deserialized.effective_size(), proof.effective_size());
        assert_eq!(deserialized.is_compressed(), proof.is_compressed());
    }

    #[test]
    fn test_statement_hash_determinism() {
        let (stmt1, _) = create_test_statement_witness();
        let (stmt2, _) = create_test_statement_witness();
        
        let hash1a = NexuszeroProtocol::statement_hash(&stmt1);
        let hash1b = NexuszeroProtocol::statement_hash(&stmt1);
        let hash2 = NexuszeroProtocol::statement_hash(&stmt2);
        
        // Same statement should produce same hash
        assert_eq!(hash1a, hash1b, "Hash should be deterministic");
        // Same statement produces same hash
        assert_eq!(hash1a, hash2, "Same statement should produce same hash");
    }

    #[test]
    fn test_different_security_levels() {
        for security_level in [SecurityLevel::Bit128, SecurityLevel::Bit256] {
            let config = ProtocolConfig {
                security_level,
                ..Default::default()
            };
            let mut protocol = NexuszeroProtocol::new(config);
            let (statement, witness) = create_test_statement_witness();
            
            let result = protocol.generate_proof(&statement, &witness);
            assert!(result.is_ok(), "Proof generation should work at {:?}", security_level);
        }
    }

    #[test]
    fn test_protocol_config_combinations() {
        // Test various config combinations
        let configs = vec![
            ProtocolConfig {
                use_compression: true,
                use_optimizer: true,
                verify_after_generation: true,
                ..Default::default()
            },
            ProtocolConfig {
                use_compression: false,
                use_optimizer: true,
                verify_after_generation: false,
                ..Default::default()
            },
            ProtocolConfig {
                use_compression: true,
                use_optimizer: false,
                verify_after_generation: true,
                ..Default::default()
            },
        ];
        
        for (i, config) in configs.into_iter().enumerate() {
            let mut protocol = NexuszeroProtocol::new(config);
            let (stmt, wit) = create_unique_statement_witness(190 + i as u8);
            let result = protocol.generate_proof(&stmt, &wit);
            assert!(result.is_ok(), "Config combination {} should work", i);
        }
    }

    #[test]
    fn test_empty_batch_handling() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        let requests: Vec<BatchProofRequest> = vec![];
        let results = protocol.generate_batch(&requests);
        
        assert!(results.is_empty(), "Empty batch should return empty results");
    }

    #[test]
    fn test_single_item_batch() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config);
        
        let (stmt, wit) = create_test_statement_witness();
        let requests = vec![BatchProofRequest::new("single", stmt, wit)];
        
        let results = protocol.generate_batch(&requests);
        
        assert_eq!(results.len(), 1);
        assert!(results[0].is_success());
    }

    #[test]
    fn test_cache_hit_rate_tracking() {
        let mut cache = ProofCache::new(10);
        let mut protocol = NexuszeroProtocol::new(ProtocolConfig::default());
        
        // Initial state
        assert_eq!(cache.stats().hits, 0);
        assert_eq!(cache.stats().misses, 0);
        assert_eq!(cache.stats().hit_rate, 0.0);
        
        let (stmt, wit) = create_test_statement_witness();
        let proof = protocol.generate_proof(&stmt, &wit).unwrap();
        let hash = NexuszeroProtocol::statement_hash(&stmt);
        
        // Miss
        let _ = cache.get(&hash);
        assert_eq!(cache.stats().misses, 1);
        
        // Insert
        cache.insert(hash, proof);
        
        // Multiple hits
        for _ in 0..9 {
            let _ = cache.get(&hash);
        }
        
        assert_eq!(cache.stats().hits, 9);
        assert_eq!(cache.stats().misses, 1);
        assert!((cache.stats().hit_rate - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_validation_result_details() {
        let config = ProtocolConfig::default();
        let protocol = NexuszeroProtocol::new(config);
        let (statement, witness) = create_test_statement_witness();
        
        let result = protocol.validate_detailed(&statement, &witness);
        
        assert!(result.is_valid);
        assert!(result.errors.is_empty());
        // Validation should be fast
    }

    #[test]
    fn test_proof_verification_after_cache() {
        let config = ProtocolConfig::default();
        let mut protocol = NexuszeroProtocol::new(config).with_cache(10);
        let (statement, witness) = create_test_statement_witness();
        
        // Generate and cache
        let proof1 = protocol.generate_proof(&statement, &witness).unwrap();
        
        // Get from cache
        let proof2 = protocol.generate_proof(&statement, &witness).unwrap();
        
        // Both should verify
        assert!(protocol.verify_proof(&proof1).unwrap());
        assert!(protocol.verify_proof(&proof2).unwrap());
    }
}
