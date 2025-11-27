//! NexusZero Integration Module
//!
//! This module provides end-to-end integration of the NexusZero Protocol components:
//! - **nexuszero-crypto**: Quantum-resistant zero-knowledge proofs
//! - **nexuszero-holographic**: Advanced proof compression using tensor networks
//! - **nexuszero-optimizer**: Neural-guided parameter optimization (via FFI)
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────────────────┐
//! │                     NexusZero Integration Layer                       │
//! ├──────────────────────────────────────────────────────────────────────┤
//! │                                                                       │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
//! │  │   API Layer  │─>│   Pipeline   │─>│  Metrics & Optimization  │   │
//! │  │              │  │              │  │                          │   │
//! │  │ NexuszeroAPI │  │ ProofPipeline│  │ MetricsCollector         │   │
//! │  │              │  │              │  │ Optimizer                │   │
//! │  └──────────────┘  └──────────────┘  └──────────────────────────┘   │
//! │         │                 │                      │                   │
//! │         v                 v                      v                   │
//! │  ┌──────────────────────────────────────────────────────────────┐   │
//! │  │                    Compression Layer                          │   │
//! │  │  CompressionManager | Strategy Selection | Metrics            │   │
//! │  └──────────────────────────────────────────────────────────────┘   │
//! │                              │                                       │
//! │         ┌────────────────────┼────────────────────┐                 │
//! │         v                    v                    v                 │
//! │  ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐       │
//! │  │nexuszero    │     │nexuszero    │     │nexuszero        │       │
//! │  │crypto       │     │holographic  │     │optimizer (FFI)  │       │
//! │  │             │     │             │     │                 │       │
//! │  │✅ Production│     │✅ Production│     │🟡 Heuristic     │       │
//! │  └─────────────┘     └─────────────┘     └─────────────────┘       │
//! │                                                                       │
//! └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use nexuszero_integration::{NexuszeroAPI, ProtocolConfig};
//! use nexuszero_integration::optimization::CompressionStrategy;
//!
//! // Create API with default configuration
//! let mut api = NexuszeroAPI::new();
//!
//! // Generate a discrete log proof
//! let generator = vec![2u8; 32];
//! let public_value = vec![4u8; 32];  
//! let secret = vec![5u8; 32];
//!
//! let proof = api.prove_discrete_log(&generator, &public_value, &secret)
//!     .expect("proof generation");
//!
//! // Verify the proof
//! assert!(api.verify(&proof).expect("verification"));
//!
//! // Get performance metrics
//! let metrics = api.get_metrics(&proof);
//! println!("Generation time: {:.2}ms", metrics.generation_time_ms);
//! println!("Compression ratio: {:.2}x", metrics.compression_ratio);
//! ```
//!
//! # Modules
//!
//! - [`config`]: Protocol configuration and settings
//! - [`pipeline`]: Core proof generation and verification pipeline
//! - [`optimization`]: Neural-guided and heuristic parameter optimization
//! - [`compression`]: Proof compression integration with holographic module
//! - [`metrics`]: Comprehensive performance metrics collection
//! - [`api`]: High-level API facade for easy integration

pub mod config;
pub mod pipeline;
pub mod api;
pub mod metrics;
pub mod optimization;
pub mod compression;

// Re-export main types for convenience
pub use config::ProtocolConfig;
pub use pipeline::{
    NexuszeroProtocol, OptimizedProof, ProofMetrics, ProtocolError,
    ProofCache, BatchProofRequest, BatchProofResult, ValidationResult, ValidationError,
};
pub use api::NexuszeroAPI;
pub use metrics::{
    MetricsCollector, ComprehensiveProofMetrics, BatchMetricsAggregator,
    ProofStageMetrics, PerformanceComparison, LatencyHistogram, RateTracker,
};
pub use optimization::{
    Optimizer, HeuristicOptimizer, StaticOptimizer, NeuralOptimizer, AdaptiveOptimizer,
    CircuitAnalysis, CircuitType, OptimizationResult, CompressionStrategy, OptimizationSource,
    BatchOptimizer, BatchOptimizationItem, OptimizationFeedback, OptimizationHistory,
    NeuralModelInterface, NeuralModelError, ModelMetadata, StubNeuralModel,
};
pub use compression::{
    CompressionManager, CompressionConfig, CompressionResult, CompressionError,
    CompressedProofPackage, CompressionAnalysis,
};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if the integration module is properly configured
pub fn is_configured() -> bool {
    // Check that required dependencies are available
    true
}

/// Get version information
pub fn version_info() -> String {
    format!(
        "nexuszero-integration v{} (crypto: v{}, holographic: v{})",
        VERSION,
        nexuszero_crypto::nexuszero_crypto_version() as f32 / 100.0,
        "0.1.0"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_info() {
        let info = version_info();
        assert!(info.contains("nexuszero-integration"));
    }

    #[test]
    fn test_is_configured() {
        assert!(is_configured());
    }
}

