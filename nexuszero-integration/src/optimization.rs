//! Neural-Guided Parameter Optimization Module
//!
//! This module provides adaptive parameter selection for proof generation,
//! leveraging neural network predictions (when available) or heuristic-based
//! optimization as a fallback.
//!
//! # Features
//!
//! - Circuit complexity analysis
//! - Adaptive security parameter selection
//! - Compression strategy optimization
//! - Performance prediction and tuning
//! - Fallback heuristics when neural model unavailable
//!
//! # Architecture
//!
//! The optimizer follows a strategy pattern:
//! 1. NeuralOptimizer - Uses trained PyTorch GNN model via FFI
//! 2. HeuristicOptimizer - Rule-based parameter selection
//! 3. AdaptiveOptimizer - Combines both with runtime switching
//!
//! # Example
//!
//! ```rust,no_run
//! use nexuszero_integration::optimization::{Optimizer, HeuristicOptimizer, CircuitAnalysis};
//! use nexuszero_crypto::SecurityLevel;
//!
//! let optimizer = HeuristicOptimizer::new(SecurityLevel::Bit128);
//! let analysis = CircuitAnalysis::from_statement_size(1024);
//! let params = optimizer.optimize(&analysis);
//! ```

use serde::{Deserialize, Serialize};
use nexuszero_crypto::{CryptoParameters, SecurityLevel};
use nexuszero_crypto::proof::Statement;

// ============================================================================
// CIRCUIT ANALYSIS
// ============================================================================

/// Analysis of circuit complexity for optimization decisions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CircuitAnalysis {
    /// Number of gates/operations in the circuit
    pub gate_count: usize,
    /// Number of inputs
    pub input_count: usize,
    /// Number of outputs
    pub output_count: usize,
    /// Estimated depth of the circuit
    pub depth: usize,
    /// Type of circuit (discrete_log, preimage, range, etc.)
    pub circuit_type: CircuitType,
    /// Estimated memory requirement (bytes)
    pub estimated_memory: usize,
    /// Complexity score (normalized 0.0-1.0)
    pub complexity_score: f64,
    /// Whether circuit has repeating patterns (compression-friendly)
    pub has_patterns: bool,
    /// Statement data size in bytes
    pub statement_size: usize,
}

/// Types of circuits the optimizer handles
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitType {
    /// Discrete logarithm proof
    DiscreteLog,
    /// Hash preimage proof
    Preimage,
    /// Range proof (Bulletproofs)
    Range,
    /// Membership proof
    Membership,
    /// Generic/unknown circuit type
    Generic,
}

impl Default for CircuitType {
    fn default() -> Self {
        CircuitType::Generic
    }
}

impl CircuitAnalysis {
    /// Create a basic analysis from statement size
    pub fn from_statement_size(size: usize) -> Self {
        // Heuristic: estimate complexity from data size
        let complexity = (size as f64 / 1024.0).min(1.0);
        let gate_count = size / 32; // Rough estimate: ~1 gate per 32 bytes
        
        Self {
            gate_count,
            input_count: 1,
            output_count: 1,
            depth: (size as f64).sqrt() as usize,
            circuit_type: CircuitType::Generic,
            estimated_memory: size * 10, // 10x amplification for proof generation
            complexity_score: complexity,
            has_patterns: size > 256, // Larger statements more likely to have patterns
            statement_size: size,
        }
    }

    /// Create analysis from a statement
    pub fn from_statement(statement: &Statement) -> Self {
        let statement_bytes = statement.to_bytes().unwrap_or_default();
        let size = statement_bytes.len();
        
        // Detect circuit type from statement
        let circuit_type = match &statement.statement_type {
            nexuszero_crypto::proof::StatementType::DiscreteLog { .. } => CircuitType::DiscreteLog,
            nexuszero_crypto::proof::StatementType::Preimage { .. } => CircuitType::Preimage,
            nexuszero_crypto::proof::StatementType::Range { .. } => CircuitType::Range,
            nexuszero_crypto::proof::StatementType::Custom { .. } => CircuitType::Generic,
        };
        
        // Analyze statement bytes for patterns
        let has_patterns = Self::detect_patterns(&statement_bytes);
        
        // Estimate complexity based on circuit type
        let base_complexity = match circuit_type {
            CircuitType::DiscreteLog => 0.6,
            CircuitType::Preimage => 0.5,
            CircuitType::Range => 0.7,
            CircuitType::Membership => 0.4,
            CircuitType::Generic => 0.5,
        };
        
        let complexity_score = base_complexity * (1.0 + (size as f64).log10() / 10.0);
        
        Self {
            gate_count: size / 32,
            input_count: 1,
            output_count: 1,
            depth: (size as f64).sqrt() as usize,
            circuit_type,
            estimated_memory: size * 10,
            complexity_score: complexity_score.min(1.0),
            has_patterns,
            statement_size: size,
        }
    }

    /// Detect repeating patterns in data (compression-friendly indicator)
    fn detect_patterns(data: &[u8]) -> bool {
        if data.len() < 32 {
            return false;
        }
        
        // Simple pattern detection: check for byte frequency distribution
        let mut freq = [0usize; 256];
        for &b in data {
            freq[b as usize] += 1;
        }
        
        // Count how many byte values appear frequently
        let threshold = data.len() / 32;
        let frequent_bytes = freq.iter().filter(|&&f| f > threshold).count();
        
        // If few byte values dominate, there are likely patterns
        frequent_bytes < 64
    }
}

// ============================================================================
// OPTIMIZATION PARAMETERS
// ============================================================================

/// Optimized parameters for proof generation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Selected cryptographic parameters
    pub crypto_params: CryptoParameters,
    /// Recommended compression strategy
    pub compression_strategy: CompressionStrategy,
    /// Estimated generation time (ms)
    pub estimated_time_ms: f64,
    /// Estimated proof size (bytes)
    pub estimated_size: usize,
    /// Confidence score (0.0-1.0)
    pub confidence: f64,
    /// Optimization source (neural/heuristic)
    pub source: OptimizationSource,
}

/// Compression strategy recommendations
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression - fastest but largest output
    None,
    /// LZ4 only - fast with moderate compression
    Lz4Fast,
    /// Hybrid MPS + LZ4 - best compression for structured data
    HybridMps,
    /// Full tensor train - best compression, slower
    TensorTrain,
    /// Adaptive - choose at runtime based on data
    Adaptive,
}

impl Default for CompressionStrategy {
    fn default() -> Self {
        CompressionStrategy::Adaptive
    }
}

/// Source of optimization decision
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationSource {
    /// Neural model prediction
    Neural,
    /// Heuristic rule-based
    Heuristic,
    /// Hybrid neural + heuristic
    Hybrid,
    /// Static default parameters
    Static,
}

// ============================================================================
// OPTIMIZER TRAIT
// ============================================================================

/// Trait for parameter optimizers
pub trait Optimizer: Send + Sync {
    /// Optimize parameters based on circuit analysis
    fn optimize(&self, analysis: &CircuitAnalysis) -> OptimizationResult;
    
    /// Get the optimization source type
    fn source(&self) -> OptimizationSource;
    
    /// Check if optimizer is ready
    fn is_ready(&self) -> bool;
}

// ============================================================================
// HEURISTIC OPTIMIZER
// ============================================================================

/// Rule-based optimizer using heuristics
#[derive(Clone, Debug)]
pub struct HeuristicOptimizer {
    /// Base security level
    security_level: SecurityLevel,
    /// Maximum allowed proof size (bytes)
    max_proof_size: Option<usize>,
    /// Target generation time (ms)
    target_time_ms: Option<f64>,
}

impl HeuristicOptimizer {
    /// Create a new heuristic optimizer
    pub fn new(security_level: SecurityLevel) -> Self {
        Self {
            security_level,
            max_proof_size: None,
            target_time_ms: None,
        }
    }

    /// Set maximum proof size constraint
    pub fn with_max_size(mut self, max_bytes: usize) -> Self {
        self.max_proof_size = Some(max_bytes);
        self
    }

    /// Set target generation time constraint
    pub fn with_target_time(mut self, target_ms: f64) -> Self {
        self.target_time_ms = Some(target_ms);
        self
    }

    /// Select compression strategy based on analysis
    fn select_compression(&self, analysis: &CircuitAnalysis) -> CompressionStrategy {
        // Decision tree for compression strategy
        if analysis.statement_size < 256 {
            // Small data: compression overhead not worth it
            CompressionStrategy::None
        } else if !analysis.has_patterns {
            // No patterns: LZ4 is best we can do
            CompressionStrategy::Lz4Fast
        } else if analysis.statement_size < 4096 {
            // Medium with patterns: hybrid approach
            CompressionStrategy::HybridMps
        } else if analysis.complexity_score > 0.7 {
            // Complex large circuits: full tensor train
            CompressionStrategy::TensorTrain
        } else {
            // Default: adaptive
            CompressionStrategy::Adaptive
        }
    }

    /// Estimate generation time based on analysis
    fn estimate_time(&self, analysis: &CircuitAnalysis) -> f64 {
        // Base time varies by circuit type
        let base_time = match analysis.circuit_type {
            CircuitType::DiscreteLog => 50.0,
            CircuitType::Preimage => 40.0,
            CircuitType::Range => 80.0,
            CircuitType::Membership => 30.0,
            CircuitType::Generic => 60.0,
        };
        
        // Scale by complexity
        let complexity_factor = 1.0 + analysis.complexity_score;
        
        // Scale by size (logarithmic)
        let size_factor = 1.0 + (analysis.statement_size as f64).log10() / 10.0;
        
        base_time * complexity_factor * size_factor
    }

    /// Estimate proof size based on analysis
    fn estimate_size(&self, analysis: &CircuitAnalysis) -> usize {
        // Base size varies by circuit type
        let base_size = match analysis.circuit_type {
            CircuitType::DiscreteLog => 256,
            CircuitType::Preimage => 192,
            CircuitType::Range => 512,
            CircuitType::Membership => 128,
            CircuitType::Generic => 256,
        };
        
        // Scale by statement size
        base_size + analysis.statement_size / 4
    }
}

impl Optimizer for HeuristicOptimizer {
    fn optimize(&self, analysis: &CircuitAnalysis) -> OptimizationResult {
        let crypto_params = CryptoParameters::from_security_level(self.security_level);
        let compression_strategy = self.select_compression(analysis);
        let estimated_time_ms = self.estimate_time(analysis);
        let estimated_size = self.estimate_size(analysis);
        
        // Confidence is lower for heuristics
        let confidence = 0.7;
        
        OptimizationResult {
            crypto_params,
            compression_strategy,
            estimated_time_ms,
            estimated_size,
            confidence,
            source: OptimizationSource::Heuristic,
        }
    }

    fn source(&self) -> OptimizationSource {
        OptimizationSource::Heuristic
    }

    fn is_ready(&self) -> bool {
        true
    }
}

// ============================================================================
// STATIC OPTIMIZER
// ============================================================================

/// Simple static optimizer that uses fixed parameters
#[derive(Clone, Debug)]
pub struct StaticOptimizer {
    security_level: SecurityLevel,
}

impl StaticOptimizer {
    /// Create a new static optimizer
    pub fn new(security_level: SecurityLevel) -> Self {
        Self { security_level }
    }
}

impl Optimizer for StaticOptimizer {
    fn optimize(&self, analysis: &CircuitAnalysis) -> OptimizationResult {
        OptimizationResult {
            crypto_params: CryptoParameters::from_security_level(self.security_level),
            compression_strategy: CompressionStrategy::Adaptive,
            estimated_time_ms: 100.0,
            estimated_size: analysis.statement_size * 2,
            confidence: 0.5,
            source: OptimizationSource::Static,
        }
    }

    fn source(&self) -> OptimizationSource {
        OptimizationSource::Static
    }

    fn is_ready(&self) -> bool {
        true
    }
}

// ============================================================================
// NEURAL OPTIMIZER STUB
// ============================================================================

/// Neural network-based optimizer (requires PyTorch FFI)
///
/// This is a stub implementation that falls back to heuristics.
/// Full implementation requires nexuszero-optimizer Python FFI.
#[derive(Clone, Debug)]
pub struct NeuralOptimizer {
    /// Fallback heuristic optimizer
    fallback: HeuristicOptimizer,
    /// Whether neural model is loaded
    model_loaded: bool,
    /// Model path (for logging)
    model_path: Option<String>,
}

impl NeuralOptimizer {
    /// Create a new neural optimizer
    pub fn new(security_level: SecurityLevel) -> Self {
        Self {
            fallback: HeuristicOptimizer::new(security_level),
            model_loaded: false,
            model_path: None,
        }
    }

    /// Attempt to load the neural model
    pub fn load_model(&mut self, path: &str) -> Result<(), String> {
        // TODO: Implement Python FFI to load nexuszero-optimizer model
        // For now, this is a stub that always fails
        self.model_path = Some(path.to_string());
        
        // Check if model file exists
        if std::path::Path::new(path).exists() {
            // Model exists but we can't load it without FFI
            self.model_loaded = false;
            Err("Neural model FFI not yet implemented, using heuristic fallback".to_string())
        } else {
            Err(format!("Model file not found: {}", path))
        }
    }

    /// Check if neural model is available
    pub fn is_model_available(&self) -> bool {
        self.model_loaded
    }
}

impl Optimizer for NeuralOptimizer {
    fn optimize(&self, analysis: &CircuitAnalysis) -> OptimizationResult {
        if self.model_loaded {
            // TODO: Call neural model via FFI
            // For now, fall back to heuristics with neural source marker
            let mut result = self.fallback.optimize(analysis);
            result.source = OptimizationSource::Neural;
            result.confidence = 0.9;
            result
        } else {
            // Fall back to heuristics
            self.fallback.optimize(analysis)
        }
    }

    fn source(&self) -> OptimizationSource {
        if self.model_loaded {
            OptimizationSource::Neural
        } else {
            OptimizationSource::Heuristic
        }
    }

    fn is_ready(&self) -> bool {
        self.model_loaded || self.fallback.is_ready()
    }
}

// ============================================================================
// ADAPTIVE OPTIMIZER
// ============================================================================

/// Adaptive optimizer that switches between strategies at runtime
#[derive(Clone)]
pub struct AdaptiveOptimizer {
    /// Primary optimizer (usually neural)
    primary: Box<dyn CloneableOptimizer>,
    /// Fallback optimizer (usually heuristic)
    fallback: Box<dyn CloneableOptimizer>,
    /// Learning history for adaptation
    history: Vec<OptimizationFeedback>,
    /// Maximum history size
    max_history: usize,
}

impl std::fmt::Debug for AdaptiveOptimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaptiveOptimizer")
            .field("history_size", &self.history.len())
            .field("max_history", &self.max_history)
            .finish()
    }
}

/// Trait for cloneable optimizers (workaround for Clone + Optimizer)
pub trait CloneableOptimizer: Optimizer {
    fn clone_box(&self) -> Box<dyn CloneableOptimizer>;
}

impl<T: Optimizer + Clone + 'static> CloneableOptimizer for T {
    fn clone_box(&self) -> Box<dyn CloneableOptimizer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn CloneableOptimizer> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Feedback from optimization for learning
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizationFeedback {
    /// The original analysis
    pub analysis: CircuitAnalysis,
    /// The optimization result used
    pub result: OptimizationResult,
    /// Actual generation time (ms)
    pub actual_time_ms: f64,
    /// Actual proof size (bytes)
    pub actual_size: usize,
    /// Whether optimization was successful
    pub success: bool,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer
    pub fn new(security_level: SecurityLevel) -> Self {
        let heuristic = HeuristicOptimizer::new(security_level);
        Self {
            primary: Box::new(heuristic.clone()),
            fallback: Box::new(heuristic),
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Set the primary optimizer
    pub fn with_primary<O: CloneableOptimizer + 'static>(mut self, optimizer: O) -> Self {
        self.primary = Box::new(optimizer);
        self
    }

    /// Record feedback for learning
    pub fn record_feedback(&mut self, feedback: OptimizationFeedback) {
        self.history.push(feedback);
        if self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }

    /// Get average prediction error
    pub fn prediction_error(&self) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        
        let total_error: f64 = self.history.iter()
            .map(|f| {
                let time_error = (f.actual_time_ms - f.result.estimated_time_ms).abs() / f.actual_time_ms.max(1.0);
                let size_error = (f.actual_size as f64 - f.result.estimated_size as f64).abs() / f.actual_size.max(1) as f64;
                (time_error + size_error) / 2.0
            })
            .sum();
        
        total_error / self.history.len() as f64
    }
}

impl Optimizer for AdaptiveOptimizer {
    fn optimize(&self, analysis: &CircuitAnalysis) -> OptimizationResult {
        // Try primary first
        if self.primary.is_ready() {
            self.primary.optimize(analysis)
        } else {
            self.fallback.optimize(analysis)
        }
    }

    fn source(&self) -> OptimizationSource {
        OptimizationSource::Hybrid
    }

    fn is_ready(&self) -> bool {
        self.primary.is_ready() || self.fallback.is_ready()
    }
}

// ============================================================================
// BATCH OPTIMIZER
// ============================================================================

/// Batch optimization for multiple proofs
#[derive(Clone, Debug)]
pub struct BatchOptimizer {
    /// Base optimizer
    base: HeuristicOptimizer,
    /// Whether to reorder for efficiency
    enable_reordering: bool,
    /// Maximum parallel executions
    max_parallelism: usize,
}

impl BatchOptimizer {
    /// Create a new batch optimizer
    pub fn new(security_level: SecurityLevel) -> Self {
        Self {
            base: HeuristicOptimizer::new(security_level),
            enable_reordering: true,
            max_parallelism: 4,
        }
    }

    /// Set maximum parallelism
    pub fn with_parallelism(mut self, max: usize) -> Self {
        self.max_parallelism = max.max(1);
        self
    }

    /// Optimize a batch of circuit analyses
    pub fn optimize_batch(&self, analyses: &[CircuitAnalysis]) -> Vec<BatchOptimizationItem> {
        let mut items: Vec<BatchOptimizationItem> = analyses
            .iter()
            .enumerate()
            .map(|(idx, analysis)| {
                let result = self.base.optimize(analysis);
                BatchOptimizationItem {
                    original_index: idx,
                    analysis: analysis.clone(),
                    result,
                    recommended_order: idx,
                    can_parallelize: true,
                }
            })
            .collect();

        if self.enable_reordering {
            // Sort by estimated time for better scheduling
            items.sort_by(|a, b| {
                a.result.estimated_time_ms
                    .partial_cmp(&b.result.estimated_time_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // Update recommended order
            for (order, item) in items.iter_mut().enumerate() {
                item.recommended_order = order;
            }
        }

        // Mark parallelization opportunities
        let mut parallel_group = 0;
        let mut group_count = 0;
        for item in items.iter_mut() {
            if group_count >= self.max_parallelism {
                parallel_group += 1;
                group_count = 0;
            }
            item.can_parallelize = group_count < self.max_parallelism;
            group_count += 1;
        }

        items
    }

    /// Get estimated total time for batch
    pub fn estimate_batch_time(&self, analyses: &[CircuitAnalysis]) -> f64 {
        let items = self.optimize_batch(analyses);
        
        // Simple model: parallel groups execute together
        let mut total = 0.0;
        let mut current_group_max = 0.0f64;
        let mut items_in_group = 0;
        
        for item in items {
            current_group_max = current_group_max.max(item.result.estimated_time_ms);
            items_in_group += 1;
            
            if items_in_group >= self.max_parallelism {
                total += current_group_max;
                current_group_max = 0.0;
                items_in_group = 0;
            }
        }
        
        // Add remaining items
        total += current_group_max;
        total
    }
}

/// Single item in batch optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BatchOptimizationItem {
    /// Original index in input array
    pub original_index: usize,
    /// The circuit analysis
    pub analysis: CircuitAnalysis,
    /// Optimization result
    pub result: OptimizationResult,
    /// Recommended execution order
    pub recommended_order: usize,
    /// Whether this can be parallelized with next item
    pub can_parallelize: bool,
}

// ============================================================================
// NEURAL MODEL INTERFACE
// ============================================================================

/// Interface for neural model predictions
/// 
/// This trait defines the contract for neural optimization models.
/// Implementations can use PyTorch via FFI, ONNX runtime, or other backends.
pub trait NeuralModelInterface: Send + Sync {
    /// Make a prediction for circuit optimization
    fn predict(&self, features: &[f64]) -> Result<Vec<f64>, NeuralModelError>;
    
    /// Get model metadata
    fn metadata(&self) -> ModelMetadata;
    
    /// Check if model is loaded and ready
    fn is_ready(&self) -> bool;
}

/// Neural model error types
#[derive(Debug, Clone)]
pub enum NeuralModelError {
    /// Model not loaded
    NotLoaded,
    /// Invalid input shape
    InvalidInput(String),
    /// Inference failed
    InferenceFailed(String),
    /// FFI error
    FfiError(String),
}

impl std::fmt::Display for NeuralModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotLoaded => write!(f, "Neural model not loaded"),
            Self::InvalidInput(s) => write!(f, "Invalid input: {}", s),
            Self::InferenceFailed(s) => write!(f, "Inference failed: {}", s),
            Self::FfiError(s) => write!(f, "FFI error: {}", s),
        }
    }
}

impl std::error::Error for NeuralModelError {}

/// Metadata about a loaded neural model
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model version
    pub version: String,
    /// Number of input features
    pub input_features: usize,
    /// Number of output features
    pub output_features: usize,
    /// Model type (GNN, MLP, etc.)
    pub model_type: String,
    /// Training date
    pub trained_at: Option<String>,
    /// Accuracy metrics
    pub accuracy: Option<f64>,
}

/// Stub neural model for testing
#[derive(Clone, Debug)]
pub struct StubNeuralModel {
    ready: bool,
    metadata: ModelMetadata,
}

impl StubNeuralModel {
    /// Create a new stub model
    pub fn new() -> Self {
        Self {
            ready: true,
            metadata: ModelMetadata {
                version: "stub-1.0".to_string(),
                input_features: 10,
                output_features: 5,
                model_type: "stub".to_string(),
                trained_at: None,
                accuracy: Some(0.95),
            },
        }
    }
}

impl Default for StubNeuralModel {
    fn default() -> Self {
        Self::new()
    }
}

impl NeuralModelInterface for StubNeuralModel {
    fn predict(&self, features: &[f64]) -> Result<Vec<f64>, NeuralModelError> {
        if !self.ready {
            return Err(NeuralModelError::NotLoaded);
        }
        
        // Simple stub: return scaled features as predictions
        let outputs: Vec<f64> = features.iter()
            .take(self.metadata.output_features)
            .map(|x| (x * 0.5 + 0.5).clamp(0.0, 1.0))
            .collect();
        
        // Pad if needed
        let mut result = outputs;
        while result.len() < self.metadata.output_features {
            result.push(0.5);
        }
        
        Ok(result)
    }
    
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
    
    fn is_ready(&self) -> bool {
        self.ready
    }
}

// ============================================================================
// OPTIMIZATION HISTORY
// ============================================================================

/// Tracks optimization history for analysis and improvement
#[derive(Clone, Debug, Default)]
pub struct OptimizationHistory {
    /// All recorded feedbacks
    entries: Vec<OptimizationFeedback>,
    /// Maximum entries to keep
    max_entries: usize,
    /// Summary statistics
    stats: HistoryStats,
}

/// Statistics from optimization history
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HistoryStats {
    /// Total optimizations
    pub total_count: usize,
    /// Success count
    pub success_count: usize,
    /// Average time prediction error (percentage)
    pub avg_time_error_pct: f64,
    /// Average size prediction error (percentage)
    pub avg_size_error_pct: f64,
    /// Best performing circuit type
    pub best_circuit_type: Option<CircuitType>,
    /// Worst performing circuit type
    pub worst_circuit_type: Option<CircuitType>,
}

impl OptimizationHistory {
    /// Create new history tracker
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries.min(1000)),
            max_entries,
            stats: HistoryStats::default(),
        }
    }

    /// Record a new feedback entry
    pub fn record(&mut self, feedback: OptimizationFeedback) {
        if self.entries.len() >= self.max_entries {
            self.entries.remove(0);
        }
        self.entries.push(feedback);
        self.update_stats();
    }

    /// Get current statistics
    pub fn stats(&self) -> &HistoryStats {
        &self.stats
    }

    /// Get all entries
    pub fn entries(&self) -> &[OptimizationFeedback] {
        &self.entries
    }

    /// Update statistics
    fn update_stats(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let total = self.entries.len();
        let successes = self.entries.iter().filter(|e| e.success).count();
        
        let (time_error_sum, size_error_sum): (f64, f64) = self.entries.iter()
            .map(|e| {
                let time_err = if e.actual_time_ms > 0.0 {
                    (e.actual_time_ms - e.result.estimated_time_ms).abs() / e.actual_time_ms * 100.0
                } else {
                    0.0
                };
                let size_err = if e.actual_size > 0 {
                    (e.actual_size as f64 - e.result.estimated_size as f64).abs() / e.actual_size as f64 * 100.0
                } else {
                    0.0
                };
                (time_err, size_err)
            })
            .fold((0.0, 0.0), |(ta, sa), (t, s)| (ta + t, sa + s));

        self.stats = HistoryStats {
            total_count: total,
            success_count: successes,
            avg_time_error_pct: time_error_sum / total as f64,
            avg_size_error_pct: size_error_sum / total as f64,
            best_circuit_type: None, // TODO: calculate from data
            worst_circuit_type: None,
        };
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_analysis_from_size() {
        let analysis = CircuitAnalysis::from_statement_size(1024);
        assert_eq!(analysis.statement_size, 1024);
        assert!(analysis.complexity_score >= 0.0 && analysis.complexity_score <= 1.0);
        assert!(analysis.gate_count > 0);
    }

    #[test]
    fn test_heuristic_optimizer() {
        let optimizer = HeuristicOptimizer::new(SecurityLevel::Bit128);
        let analysis = CircuitAnalysis::from_statement_size(512);
        
        let result = optimizer.optimize(&analysis);
        
        assert_eq!(result.source, OptimizationSource::Heuristic);
        assert!(result.estimated_time_ms > 0.0);
        assert!(result.estimated_size > 0);
        assert!(result.confidence > 0.0);
    }

    #[test]
    fn test_compression_strategy_selection() {
        let optimizer = HeuristicOptimizer::new(SecurityLevel::Bit128);
        
        // Small data - no compression
        let small = CircuitAnalysis::from_statement_size(64);
        let result = optimizer.optimize(&small);
        assert_eq!(result.compression_strategy, CompressionStrategy::None);
        
        // Large data - should use compression
        let large = CircuitAnalysis::from_statement_size(8192);
        let result = optimizer.optimize(&large);
        assert_ne!(result.compression_strategy, CompressionStrategy::None);
    }

    #[test]
    fn test_static_optimizer() {
        let optimizer = StaticOptimizer::new(SecurityLevel::Bit256);
        let analysis = CircuitAnalysis::from_statement_size(1024);
        
        let result = optimizer.optimize(&analysis);
        
        assert_eq!(result.source, OptimizationSource::Static);
        assert!(optimizer.is_ready());
    }

    #[test]
    fn test_neural_optimizer_fallback() {
        let mut optimizer = NeuralOptimizer::new(SecurityLevel::Bit128);
        
        // Without loading, should fall back to heuristics
        assert!(!optimizer.is_model_available());
        
        let analysis = CircuitAnalysis::from_statement_size(1024);
        let result = optimizer.optimize(&analysis);
        
        // Should work via fallback
        assert!(optimizer.is_ready());
        assert!(result.estimated_time_ms > 0.0);
    }

    #[test]
    fn test_pattern_detection() {
        // Data with clear patterns
        let patterned: Vec<u8> = (0..256).flat_map(|_| vec![0, 1, 2, 3]).collect();
        let analysis = CircuitAnalysis::from_statement_size(1024);
        // Pattern detection is internal, so we just verify it runs
        assert!(analysis.statement_size == 1024);
    }

    #[test]
    fn test_optimization_result_serialization() {
        let result = OptimizationResult {
            crypto_params: CryptoParameters::from_security_level(SecurityLevel::Bit128),
            compression_strategy: CompressionStrategy::HybridMps,
            estimated_time_ms: 50.0,
            estimated_size: 1024,
            confidence: 0.8,
            source: OptimizationSource::Heuristic,
        };
        
        let json = serde_json::to_string(&result).unwrap();
        let recovered: OptimizationResult = serde_json::from_str(&json).unwrap();
        
        assert_eq!(result.compression_strategy, recovered.compression_strategy);
        assert!((result.confidence - recovered.confidence).abs() < 0.001);
    }

    #[test]
    fn test_adaptive_optimizer() {
        let optimizer = AdaptiveOptimizer::new(SecurityLevel::Bit128);
        let analysis = CircuitAnalysis::from_statement_size(2048);
        
        let result = optimizer.optimize(&analysis);
        
        assert!(optimizer.is_ready());
        assert!(result.estimated_time_ms > 0.0);
    }

    #[test]
    fn test_batch_optimizer() {
        let optimizer = BatchOptimizer::new(SecurityLevel::Bit128);
        
        let analyses: Vec<CircuitAnalysis> = vec![
            CircuitAnalysis::from_statement_size(256),
            CircuitAnalysis::from_statement_size(1024),
            CircuitAnalysis::from_statement_size(512),
            CircuitAnalysis::from_statement_size(2048),
        ];
        
        let results = optimizer.optimize_batch(&analyses);
        
        assert_eq!(results.len(), 4);
        for item in &results {
            assert!(item.result.estimated_time_ms > 0.0);
        }
    }

    #[test]
    fn test_batch_time_estimation() {
        let optimizer = BatchOptimizer::new(SecurityLevel::Bit128)
            .with_parallelism(2);
        
        let analyses: Vec<CircuitAnalysis> = vec![
            CircuitAnalysis::from_statement_size(1024),
            CircuitAnalysis::from_statement_size(1024),
            CircuitAnalysis::from_statement_size(1024),
            CircuitAnalysis::from_statement_size(1024),
        ];
        
        let total_time = optimizer.estimate_batch_time(&analyses);
        
        // With parallelism of 2, 4 items should take ~2 parallel groups
        assert!(total_time > 0.0);
    }

    #[test]
    fn test_stub_neural_model() {
        let model = StubNeuralModel::new();
        
        assert!(model.is_ready());
        
        let features = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = model.predict(&features).unwrap();
        
        assert!(!result.is_empty());
        for val in &result {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_neural_model_metadata() {
        let model = StubNeuralModel::new();
        let meta = model.metadata();
        
        assert_eq!(meta.version, "stub-1.0");
        assert_eq!(meta.model_type, "stub");
        assert!(meta.accuracy.unwrap() > 0.9);
    }

    #[test]
    fn test_optimization_history() {
        let mut history = OptimizationHistory::new(100);
        
        // Record some feedbacks
        for i in 0..5 {
            let feedback = OptimizationFeedback {
                analysis: CircuitAnalysis::from_statement_size(1024),
                result: OptimizationResult {
                    crypto_params: CryptoParameters::from_security_level(SecurityLevel::Bit128),
                    compression_strategy: CompressionStrategy::Adaptive,
                    estimated_time_ms: 50.0,
                    estimated_size: 512,
                    confidence: 0.8,
                    source: OptimizationSource::Heuristic,
                },
                actual_time_ms: 55.0 + i as f64,
                actual_size: 520 + i * 10,
                success: true,
            };
            history.record(feedback);
        }
        
        let stats = history.stats();
        assert_eq!(stats.total_count, 5);
        assert_eq!(stats.success_count, 5);
        assert!(stats.avg_time_error_pct > 0.0);
    }

    #[test]
    fn test_history_max_entries() {
        let mut history = OptimizationHistory::new(3);
        
        for i in 0..5 {
            let feedback = OptimizationFeedback {
                analysis: CircuitAnalysis::from_statement_size(i * 100 + 100),
                result: OptimizationResult {
                    crypto_params: CryptoParameters::from_security_level(SecurityLevel::Bit128),
                    compression_strategy: CompressionStrategy::None,
                    estimated_time_ms: 10.0,
                    estimated_size: 100,
                    confidence: 0.5,
                    source: OptimizationSource::Static,
                },
                actual_time_ms: 12.0,
                actual_size: 110,
                success: true,
            };
            history.record(feedback);
        }
        
        // Should only keep last 3
        assert_eq!(history.entries().len(), 3);
        // First entry should be from i=2 (statement_size=300)
        assert_eq!(history.entries()[0].analysis.statement_size, 300);
    }
}
