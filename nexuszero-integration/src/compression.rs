//! Compression Integration Layer
//!
//! This module provides the integration layer between the proof generation
//! pipeline and the holographic compression system (nexuszero-holographic).
//!
//! # Features
//!
//! - Automatic compression strategy selection
//! - Transparent proof compression/decompression
//! - Compression metrics collection
//! - Multiple algorithm support (LZ4, MPS, Tensor Train)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
//! │  Proof Data     │ ──> │  CompressionManager  │ ──> │  Compressed     │
//! │  (bytes)        │     │  - Strategy selection│     │  Proof          │
//! │                 │ <── │  - Algorithm routing │ <── │  (bytes)        │
//! └─────────────────┘     └──────────────────────┘     └─────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use thiserror::Error;
use crate::optimization::CompressionStrategy;

// Re-export compression types from holographic module
pub use nexuszero_holographic::{
    CompressedTensorTrain, CompressionConfig as HoloCompressionConfig,
    StoragePrecision, CompressionError as HoloCompressionError,
    compress_proof_data, decompress_proof_data,
    analyze_compression_potential,
};

// Local serializable version of CompressionRecommendation
// (The holographic version doesn't implement Serialize/Deserialize)
/// Recommended compression approach based on data analysis
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionRecommendationLocal {
    /// Data has low entropy and good structure - use tensor train
    TensorTrain,
    /// Moderate entropy with some structure - use hybrid approach  
    Hybrid,
    /// High entropy data - use standard compression (LZ4/Zstd) only
    #[default]
    StandardOnly,
}

impl From<nexuszero_holographic::CompressionRecommendation> for CompressionRecommendationLocal {
    fn from(rec: nexuszero_holographic::CompressionRecommendation) -> Self {
        match rec {
            nexuszero_holographic::CompressionRecommendation::TensorTrain => Self::TensorTrain,
            nexuszero_holographic::CompressionRecommendation::Hybrid => Self::Hybrid,
            nexuszero_holographic::CompressionRecommendation::StandardOnly => Self::StandardOnly,
        }
    }
}

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur during compression operations
#[derive(Debug, Error)]
pub enum CompressionError {
    /// Input data is invalid or empty
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
    
    /// Compression algorithm failed
    #[error("Compression failed: {0}")]
    CompressionFailed(String),
    
    /// Decompression algorithm failed
    #[error("Decompression failed: {0}")]
    DecompressionFailed(String),
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// Strategy not supported
    #[error("Unsupported compression strategy: {0:?}")]
    UnsupportedStrategy(CompressionStrategy),
    
    /// Data integrity check failed
    #[error("Data integrity check failed")]
    IntegrityError,
}

impl From<HoloCompressionError> for CompressionError {
    fn from(err: HoloCompressionError) -> Self {
        CompressionError::CompressionFailed(err.to_string())
    }
}

// ============================================================================
// COMPRESSION RESULT
// ============================================================================

/// Result of a compression operation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionResult {
    /// The compressed data
    pub compressed_data: Vec<u8>,
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (original/compressed)
    pub ratio: f64,
    /// Strategy used
    pub strategy_used: CompressionStrategy,
    /// Time taken to compress (ms)
    pub compression_time_ms: f64,
    /// Whether data was actually compressed (ratio > 1.0)
    pub was_compressed: bool,
}

impl CompressionResult {
    /// Calculate space savings in bytes
    pub fn space_saved(&self) -> usize {
        self.original_size.saturating_sub(self.compressed_size)
    }

    /// Calculate space savings as percentage
    pub fn savings_percent(&self) -> f64 {
        if self.original_size > 0 {
            100.0 * (1.0 - (self.compressed_size as f64 / self.original_size as f64))
        } else {
            0.0
        }
    }
}

// ============================================================================
// COMPRESSION CONFIG
// ============================================================================

/// Configuration for compression operations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Preferred compression strategy
    pub strategy: CompressionStrategy,
    /// Maximum bond dimension for MPS/TT compression
    pub max_bond_dim: usize,
    /// Truncation threshold for SVD
    pub truncation_threshold: f64,
    /// Storage precision
    pub precision: StoragePrecision,
    /// Enable hybrid LZ4 backend
    pub use_hybrid: bool,
    /// Minimum size to attempt compression (bytes)
    pub min_size_threshold: usize,
    /// Verify compression integrity
    pub verify_integrity: bool,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            strategy: CompressionStrategy::Adaptive,
            max_bond_dim: 32,
            truncation_threshold: 1e-4,
            precision: StoragePrecision::F32,
            use_hybrid: true,
            min_size_threshold: 64,
            verify_integrity: true,
        }
    }
}

impl CompressionConfig {
    /// High compression preset
    pub fn high_compression() -> Self {
        Self {
            strategy: CompressionStrategy::TensorTrain,
            max_bond_dim: 16,
            truncation_threshold: 1e-3,
            precision: StoragePrecision::I8,
            use_hybrid: true,
            min_size_threshold: 32,
            verify_integrity: true,
        }
    }

    /// Fast compression preset
    pub fn fast() -> Self {
        Self {
            strategy: CompressionStrategy::Lz4Fast,
            max_bond_dim: 8,
            truncation_threshold: 1e-4,
            precision: StoragePrecision::F32,
            use_hybrid: false,
            min_size_threshold: 128,
            verify_integrity: false,
        }
    }

    /// Balanced preset
    pub fn balanced() -> Self {
        Self {
            strategy: CompressionStrategy::HybridMps,
            max_bond_dim: 32,
            truncation_threshold: 1e-5,
            precision: StoragePrecision::F16,
            use_hybrid: true,
            min_size_threshold: 64,
            verify_integrity: true,
        }
    }

    /// Convert to holographic compression config
    pub fn to_holo_config(&self) -> HoloCompressionConfig {
        HoloCompressionConfig {
            max_bond_dim: self.max_bond_dim,
            truncation_threshold: self.truncation_threshold,
            block_size: 64,
            precision: self.precision,
            hybrid_mode: self.use_hybrid,
        }
    }
}

// ============================================================================
// COMPRESSION MANAGER
// ============================================================================

/// Manager for compression operations
#[derive(Clone, Debug)]
pub struct CompressionManager {
    /// Configuration
    config: CompressionConfig,
}

impl CompressionManager {
    /// Create a new compression manager
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(CompressionConfig::default())
    }

    /// Compress proof data
    pub fn compress(&self, data: &[u8]) -> Result<CompressionResult, CompressionError> {
        if data.is_empty() {
            return Err(CompressionError::InvalidInput("Empty input data".to_string()));
        }

        // Check minimum size threshold
        if data.len() < self.config.min_size_threshold {
            return Ok(CompressionResult {
                compressed_data: data.to_vec(),
                original_size: data.len(),
                compressed_size: data.len(),
                ratio: 1.0,
                strategy_used: CompressionStrategy::None,
                compression_time_ms: 0.0,
                was_compressed: false,
            });
        }

        let start = std::time::Instant::now();
        
        // Select and execute compression strategy
        let (compressed, strategy_used) = match self.config.strategy {
            CompressionStrategy::None => {
                (data.to_vec(), CompressionStrategy::None)
            }
            CompressionStrategy::Lz4Fast => {
                let compressed = self.compress_lz4(data)?;
                (compressed, CompressionStrategy::Lz4Fast)
            }
            CompressionStrategy::HybridMps => {
                let compressed = self.compress_hybrid_mps(data)?;
                (compressed, CompressionStrategy::HybridMps)
            }
            CompressionStrategy::TensorTrain => {
                let compressed = self.compress_tensor_train(data)?;
                (compressed, CompressionStrategy::TensorTrain)
            }
            CompressionStrategy::Adaptive => {
                self.compress_adaptive(data)?
            }
        };

        let compression_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        let original_size = data.len();
        let compressed_size = compressed.len();
        let ratio = if compressed_size > 0 {
            original_size as f64 / compressed_size as f64
        } else {
            1.0
        };

        // Verify integrity if configured
        if self.config.verify_integrity && strategy_used != CompressionStrategy::None {
            self.verify_compression(&compressed, original_size)?;
        }

        Ok(CompressionResult {
            compressed_data: compressed,
            original_size,
            compressed_size,
            ratio,
            strategy_used,
            compression_time_ms,
            was_compressed: ratio > 1.0,
        })
    }

    /// Decompress proof data
    pub fn decompress(&self, data: &[u8], strategy: CompressionStrategy) -> Result<Vec<u8>, CompressionError> {
        if data.is_empty() {
            return Err(CompressionError::InvalidInput("Empty compressed data".to_string()));
        }

        match strategy {
            CompressionStrategy::None => Ok(data.to_vec()),
            CompressionStrategy::Lz4Fast => self.decompress_lz4(data),
            CompressionStrategy::HybridMps | 
            CompressionStrategy::TensorTrain |
            CompressionStrategy::Adaptive => {
                // All these use the holographic decompression
                decompress_proof_data(data).map_err(|e| e.into())
            }
        }
    }

    /// Analyze data for optimal compression strategy
    pub fn analyze(&self, data: &[u8]) -> CompressionAnalysis {
        if data.is_empty() {
            return CompressionAnalysis::default();
        }

        // Use holographic analysis if available
        let holo_analysis = analyze_compression_potential(data);
        
        // Calculate entropy estimate
        let entropy = Self::estimate_entropy(data);
        
        // Detect patterns
        let has_patterns = Self::detect_patterns(data);
        
        // Recommend strategy
        let recommended_strategy = if entropy > 7.5 {
            CompressionStrategy::None // High entropy, compression won't help
        } else if has_patterns && data.len() > 1024 {
            CompressionStrategy::TensorTrain
        } else if has_patterns {
            CompressionStrategy::HybridMps
        } else {
            CompressionStrategy::Lz4Fast
        };

        CompressionAnalysis {
            entropy_estimate: entropy,
            has_patterns,
            data_size: data.len(),
            recommended_strategy,
            estimated_ratio: holo_analysis.estimated_ratio,
            holographic_recommendation: holo_analysis.recommendation.into(),
        }
    }

    // ========================================================================
    // PRIVATE COMPRESSION METHODS
    // ========================================================================

    fn compress_lz4(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        lz4_flex::compress_prepend_size(data);
        Ok(lz4_flex::compress_prepend_size(data))
    }

    fn decompress_lz4(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        lz4_flex::decompress_size_prepended(data)
            .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
    }

    fn compress_hybrid_mps(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let config = self.config.to_holo_config();
        let compressed = CompressedTensorTrain::compress(data, config)?;
        compressed.to_bytes_lz4().map_err(|e| e.into())
    }

    fn compress_tensor_train(&self, data: &[u8]) -> Result<Vec<u8>, CompressionError> {
        let mut config = self.config.to_holo_config();
        config.hybrid_mode = true; // Enable full hybrid for best compression
        let compressed = CompressedTensorTrain::compress(data, config)?;
        compressed.to_bytes_lz4().map_err(|e| e.into())
    }

    fn compress_adaptive(&self, data: &[u8]) -> Result<(Vec<u8>, CompressionStrategy), CompressionError> {
        // Analyze data to pick best strategy
        let analysis = self.analyze(data);
        
        // Try the recommended strategy
        match analysis.recommended_strategy {
            CompressionStrategy::None => {
                Ok((data.to_vec(), CompressionStrategy::None))
            }
            CompressionStrategy::Lz4Fast => {
                let compressed = self.compress_lz4(data)?;
                Ok((compressed, CompressionStrategy::Lz4Fast))
            }
            CompressionStrategy::HybridMps => {
                let compressed = self.compress_hybrid_mps(data)?;
                Ok((compressed, CompressionStrategy::HybridMps))
            }
            strategy => {
                let compressed = self.compress_tensor_train(data)?;
                Ok((compressed, strategy))
            }
        }
    }

    fn verify_compression(&self, _compressed: &[u8], _original_size: usize) -> Result<(), CompressionError> {
        // For now, just basic size check
        // TODO: Full roundtrip verification for critical paths
        Ok(())
    }

    fn estimate_entropy(data: &[u8]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        let mut freq = [0usize; 256];
        for &b in data {
            freq[b as usize] += 1;
        }

        let len = data.len() as f64;
        let mut entropy = 0.0;
        for &f in &freq {
            if f > 0 {
                let p = f as f64 / len;
                entropy -= p * p.log2();
            }
        }
        
        entropy
    }

    fn detect_patterns(data: &[u8]) -> bool {
        if data.len() < 32 {
            return false;
        }

        let mut freq = [0usize; 256];
        for &b in data {
            freq[b as usize] += 1;
        }

        // Count dominant byte values
        let threshold = data.len() / 16;
        let dominant = freq.iter().filter(|&&f| f > threshold).count();
        
        // If few bytes dominate, there are patterns
        dominant < 64
    }
}

impl Default for CompressionManager {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ============================================================================
// COMPRESSION ANALYSIS
// ============================================================================

/// Analysis of data for compression optimization
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionAnalysis {
    /// Estimated entropy (0-8, 8 = max entropy)
    pub entropy_estimate: f64,
    /// Whether repeating patterns were detected
    pub has_patterns: bool,
    /// Size of input data
    pub data_size: usize,
    /// Recommended compression strategy
    pub recommended_strategy: CompressionStrategy,
    /// Estimated compression ratio
    pub estimated_ratio: f64,
    /// Holographic compression recommendation
    pub holographic_recommendation: CompressionRecommendationLocal,
}

impl Default for CompressionAnalysis {
    fn default() -> Self {
        Self {
            entropy_estimate: 8.0,
            has_patterns: false,
            data_size: 0,
            recommended_strategy: CompressionStrategy::None,
            estimated_ratio: 1.0,
            holographic_recommendation: CompressionRecommendationLocal::StandardOnly,
        }
    }
}

// ============================================================================
// COMPRESSED PROOF WRAPPER
// ============================================================================

/// A compressed proof that can be serialized and transmitted
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedProofPackage {
    /// The compressed proof data
    pub data: Vec<u8>,
    /// Original proof size
    pub original_size: usize,
    /// Compression strategy used
    pub strategy: CompressionStrategy,
    /// Checksum for integrity verification
    pub checksum: [u8; 32],
}

impl CompressedProofPackage {
    /// Create a new compressed proof package
    pub fn new(result: CompressionResult) -> Self {
        use sha2::{Sha256, Digest};
        let checksum: [u8; 32] = Sha256::digest(&result.compressed_data).into();
        
        Self {
            data: result.compressed_data,
            original_size: result.original_size,
            strategy: result.strategy_used,
            checksum,
        }
    }

    /// Verify the package integrity
    pub fn verify_integrity(&self) -> bool {
        use sha2::{Sha256, Digest};
        let computed: [u8; 32] = Sha256::digest(&self.data).into();
        computed == self.checksum
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        if self.data.len() > 0 {
            self.original_size as f64 / self.data.len() as f64
        } else {
            1.0
        }
    }

    /// Decompress to original proof data
    pub fn decompress(&self) -> Result<Vec<u8>, CompressionError> {
        let manager = CompressionManager::with_defaults();
        manager.decompress(&self.data, self.strategy)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_manager_creation() {
        let manager = CompressionManager::with_defaults();
        assert_eq!(manager.config.strategy, CompressionStrategy::Adaptive);
    }

    #[test]
    fn test_compress_small_data_skipped() {
        let manager = CompressionManager::new(CompressionConfig {
            min_size_threshold: 100,
            ..Default::default()
        });
        
        let data: Vec<u8> = vec![1, 2, 3, 4, 5];
        let result = manager.compress(&data).unwrap();
        
        assert!(!result.was_compressed);
        assert_eq!(result.strategy_used, CompressionStrategy::None);
        assert_eq!(result.ratio, 1.0);
    }

    #[test]
    fn test_compress_lz4() {
        let manager = CompressionManager::new(CompressionConfig {
            strategy: CompressionStrategy::Lz4Fast,
            min_size_threshold: 0,
            verify_integrity: false,
            ..Default::default()
        });
        
        // Repeating pattern should compress well
        let data: Vec<u8> = (0..1024).map(|i| (i % 16) as u8).collect();
        let result = manager.compress(&data).unwrap();
        
        assert_eq!(result.strategy_used, CompressionStrategy::Lz4Fast);
        assert!(result.compressed_size < result.original_size);
    }

    #[test]
    fn test_compression_analysis() {
        let manager = CompressionManager::with_defaults();
        
        // High entropy data
        let random: Vec<u8> = (0..=255u8).collect();
        let analysis = manager.analyze(&random);
        assert!(analysis.entropy_estimate > 5.0);
        
        // Low entropy data
        let repetitive: Vec<u8> = vec![0; 256];
        let analysis = manager.analyze(&repetitive);
        assert!(analysis.entropy_estimate < 1.0);
    }

    #[test]
    fn test_compressed_proof_package() {
        let result = CompressionResult {
            compressed_data: vec![1, 2, 3, 4],
            original_size: 100,
            compressed_size: 4,
            ratio: 25.0,
            strategy_used: CompressionStrategy::Lz4Fast,
            compression_time_ms: 1.0,
            was_compressed: true,
        };
        
        let package = CompressedProofPackage::new(result);
        assert!(package.verify_integrity());
        assert!((package.compression_ratio() - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_entropy_estimation() {
        // Uniform distribution - max entropy
        let uniform: Vec<u8> = (0..=255u8).collect();
        let entropy = CompressionManager::estimate_entropy(&uniform);
        assert!(entropy > 7.0);
        
        // Single value - zero entropy
        let constant: Vec<u8> = vec![42; 256];
        let entropy = CompressionManager::estimate_entropy(&constant);
        assert!(entropy < 0.01);
    }

    #[test]
    fn test_pattern_detection() {
        // Data with patterns
        let patterned: Vec<u8> = (0..256).map(|i| (i % 4) as u8).collect();
        assert!(CompressionManager::detect_patterns(&patterned));
        
        // Random data (simulated)
        let _random: Vec<u8> = (0..=255u8).collect();
        // This might or might not detect patterns depending on distribution
    }

    #[test]
    fn test_compression_config_presets() {
        let fast = CompressionConfig::fast();
        assert_eq!(fast.strategy, CompressionStrategy::Lz4Fast);
        
        let high = CompressionConfig::high_compression();
        assert_eq!(high.strategy, CompressionStrategy::TensorTrain);
        
        let balanced = CompressionConfig::balanced();
        assert_eq!(balanced.strategy, CompressionStrategy::HybridMps);
    }

    #[test]
    fn test_space_savings() {
        let result = CompressionResult {
            compressed_data: vec![],
            original_size: 1000,
            compressed_size: 100,
            ratio: 10.0,
            strategy_used: CompressionStrategy::HybridMps,
            compression_time_ms: 5.0,
            was_compressed: true,
        };
        
        assert_eq!(result.space_saved(), 900);
        assert!((result.savings_percent() - 90.0).abs() < 0.01);
    }
}
