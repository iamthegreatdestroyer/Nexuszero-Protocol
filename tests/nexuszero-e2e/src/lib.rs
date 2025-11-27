//! NexusZero End-to-End Test Library
//!
//! This library provides shared test utilities and test suite organization
//! for the NexusZero Protocol E2E testing framework.
//!
//! # Integration with Real Modules
//!
//! This test suite integrates with:
//! - `nexuszero-crypto`: Bulletproof range proofs, Ring-LWE encryption
//! - `nexuszero-holographic`: MPS compression, holographic encoding
//! - Privacy services (via privacy_service crate when available)

pub mod utils;

// Re-export commonly used test utilities
pub use utils::{generate_deterministic_bytes, generate_random_bytes, TestMetrics, Timer};

// Re-export real cryptographic modules for E2E testing
pub use nexuszero_crypto::{
    CryptoError, CryptoParameters, CryptoResult, SecurityLevel,
    proof::BulletproofRangeProof as RawBulletproofRangeProof,
};

// Re-export holographic compression modules
pub use nexuszero_holographic::{
    compression::mps_compressed::{CompressedMPS, MPSConfig, MPSError, QuantizedTensor},
    compression::encoder_new::{HolographicEncoder, EncoderConfig, CompressedProof, CompressionStats},
    compression::mps_v2::{
        CompressedTensorTrain, CompressionConfig, StoragePrecision,
        compress_proof_data as raw_compress_proof_data, 
        decompress_proof_data as raw_decompress_proof_data,
    },
};

// ============================================================================
// WRAPPER TYPES FOR E2E TESTING
// ============================================================================
// These wrappers provide a simpler API for E2E tests while using the real
// underlying implementations.

use serde::{Serialize, Deserialize};

/// Wrapped BulletproofRangeProof with convenience methods for E2E testing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BulletproofRangeProof {
    inner: RawBulletproofRangeProof,
    num_bits: usize,
}

impl BulletproofRangeProof {
    /// Create a new wrapped proof
    pub fn new(inner: RawBulletproofRangeProof, num_bits: usize) -> Self {
        Self { inner, num_bits }
    }

    /// Get the size of the proof in bytes
    pub fn size_bytes(&self) -> usize {
        // Calculate approximate size based on proof components
        self.inner.commitment.len() 
            + self.inner.bit_commitments.iter().map(|c| c.len()).sum::<usize>()
            + self.inner.inner_product_proof.left_commitments.iter().map(|c| c.len()).sum::<usize>()
            + self.inner.inner_product_proof.right_commitments.iter().map(|c| c.len()).sum::<usize>()
            + self.inner.inner_product_proof.final_a.len()
            + self.inner.inner_product_proof.final_b.len()
            + self.inner.challenges.len() * 32
    }

    /// Serialize proof to bytes (for compression)
    pub fn to_bytes(&self) -> Vec<u8> {
        bincode::serialize(&self.inner).unwrap_or_default()
    }

    /// Deserialize proof from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        let inner: RawBulletproofRangeProof = bincode::deserialize(bytes)
            .map_err(|e| CryptoError::ProofError(e.to_string()))?;
        let num_bits = inner.bit_commitments.len();
        Ok(Self { inner, num_bits })
    }

    /// Get the commitment from the proof
    pub fn commitment(&self) -> &[u8] {
        &self.inner.commitment
    }

    /// Get the number of bits
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Access inner proof
    pub fn inner(&self) -> &RawBulletproofRangeProof {
        &self.inner
    }
}

/// Generate a range proof for E2E testing
pub fn prove_range(value: u64, blinding: &[u8; 32], num_bits: u8) -> CryptoResult<BulletproofRangeProof> {
    let inner = nexuszero_crypto::proof::prove_range(value, blinding, num_bits as usize)?;
    Ok(BulletproofRangeProof::new(inner, num_bits as usize))
}

/// Verify a range proof for E2E testing
pub fn verify_range(proof: &BulletproofRangeProof) -> CryptoResult<()> {
    nexuszero_crypto::proof::verify_range(&proof.inner, &proof.inner.commitment, proof.num_bits)
}

/// Compress proof data with configuration
pub fn compress_proof_data(data: &[u8], _config: &CompressionConfig) -> Result<CompressedData, CompressionError> {
    // Use the raw compression function with default config
    let compressed_bytes = raw_compress_proof_data(data)
        .map_err(|e| CompressionError::CompressionFailed(e.to_string()))?;
    Ok(CompressedData { data: compressed_bytes, original_size: data.len() })
}

/// Decompress proof data
pub fn decompress_proof_data(compressed: &CompressedData) -> Result<Vec<u8>, CompressionError> {
    raw_decompress_proof_data(&compressed.data)
        .map_err(|e| CompressionError::DecompressionFailed(e.to_string()))
}

/// Compressed data wrapper
#[derive(Clone, Debug)]
pub struct CompressedData {
    pub data: Vec<u8>,
    pub original_size: usize,
}

impl CompressedData {
    /// Get the serialized size of the compressed data
    pub fn serialized_size(&self) -> usize {
        self.data.len()
    }
}

/// Compression errors for E2E testing
#[derive(Debug, Clone)]
pub enum CompressionError {
    CompressionFailed(String),
    DecompressionFailed(String),
}

impl std::fmt::Display for CompressionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompressionError::CompressionFailed(s) => write!(f, "Compression failed: {}", s),
            CompressionError::DecompressionFailed(s) => write!(f, "Decompression failed: {}", s),
        }
    }
}

impl std::error::Error for CompressionError {}

// ============================================================================
// TEST CONFIGURATION
// ============================================================================

/// Test configuration for E2E tests
#[derive(Clone, Debug)]
pub struct E2ETestConfig {
    /// Security level for crypto operations
    pub security_level: SecurityLevel,
    /// Compression configuration
    pub compression_config: MPSConfig,
    /// Number of iterations for performance tests
    pub iterations: usize,
    /// Enable verbose output
    pub verbose: bool,
}

impl Default for E2ETestConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Bit128,
            compression_config: MPSConfig::default(),
            iterations: 100,
            verbose: false,
        }
    }
}

impl E2ETestConfig {
    /// Create a quick test configuration for CI
    pub fn quick() -> Self {
        Self {
            security_level: SecurityLevel::Bit128,
            compression_config: MPSConfig::fast(),
            iterations: 10,
            verbose: false,
        }
    }

    /// Create an exhaustive test configuration
    pub fn exhaustive() -> Self {
        Self {
            security_level: SecurityLevel::Bit256,
            compression_config: MPSConfig::lossless(),
            iterations: 1000,
            verbose: true,
        }
    }
}
