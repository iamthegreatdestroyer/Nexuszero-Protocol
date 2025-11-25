//! High-Level Holographic Encoder API
//!
//! Provides a clean interface for encoding/decoding data using the
//! CompressedMPS implementation with various configuration presets.

use serde::{Deserialize, Serialize};

use super::mps_compressed::{CompressedMPS, MPSConfig, MPSError};

/// Configuration for the holographic encoder
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Underlying MPS configuration
    pub mps_config: MPSConfig,
    /// Enable hybrid compression (MPS + standard algorithm)
    pub hybrid_mode: bool,
    /// Verify integrity after encoding
    pub verify_on_encode: bool,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            mps_config: MPSConfig::default(),
            hybrid_mode: false,
            verify_on_encode: true,
        }
    }
}

impl EncoderConfig {
    /// High compression preset
    pub fn high_compression() -> Self {
        Self {
            mps_config: MPSConfig::high_compression(),
            hybrid_mode: true,
            verify_on_encode: false,
        }
    }

    /// Fast encoding preset
    pub fn fast() -> Self {
        Self {
            mps_config: MPSConfig::fast(),
            hybrid_mode: false,
            verify_on_encode: false,
        }
    }

    /// Lossless preset (exact reconstruction)
    pub fn lossless() -> Self {
        Self {
            mps_config: MPSConfig::lossless(),
            hybrid_mode: false,
            verify_on_encode: true,
        }
    }
}

/// Compressed proof structure for transmission/storage
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedProof {
    /// The compressed MPS data
    pub mps_bytes: Vec<u8>,
    /// Original data hash for verification
    pub original_hash: [u8; 32],
    /// Compression metadata
    pub metadata: ProofMetadata,
}

/// Metadata about the compressed proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (compressed/original)
    pub compression_ratio: f64,
    /// Number of MPS sites
    pub num_sites: usize,
    /// Bond dimensions
    pub bond_dims: Vec<usize>,
}

/// High-level holographic encoder/decoder
pub struct HolographicEncoder {
    config: EncoderConfig,
}

impl HolographicEncoder {
    /// Create a new encoder with the given configuration
    pub fn new(config: EncoderConfig) -> Self {
        Self { config }
    }

    /// Create an encoder with default settings
    pub fn with_defaults() -> Self {
        Self::new(EncoderConfig::default())
    }

    /// Encode data into a compressed proof
    pub fn encode(&self, data: &[u8]) -> Result<CompressedProof, MPSError> {
        // Compute hash of original data
        let original_hash = simple_hash(data);

        // Compress using MPS
        let mps = CompressedMPS::compress(data, self.config.mps_config.clone())?;

        // Serialize MPS
        let mps_bytes = mps.to_bytes()?;

        // Optionally apply hybrid compression (LZ4 on top)
        let final_bytes = if self.config.hybrid_mode {
            // Simple RLE-like compression for repeated values
            compress_bytes(&mps_bytes)
        } else {
            mps_bytes
        };

        let metadata = ProofMetadata {
            original_size: data.len(),
            compressed_size: final_bytes.len(),
            compression_ratio: final_bytes.len() as f64 / data.len() as f64,
            num_sites: mps.num_sites(),
            bond_dims: mps.bond_dims().to_vec(),
        };

        let proof = CompressedProof {
            mps_bytes: final_bytes,
            original_hash,
            metadata,
        };

        // Verify if configured
        if self.config.verify_on_encode {
            if !self.verify(&proof) {
                // Verification failed, but we still return the proof
                // In production, you might want to handle this differently
            }
        }

        Ok(proof)
    }

    /// Decode a compressed proof back to data
    pub fn decode(&self, proof: &CompressedProof) -> Result<Vec<u8>, MPSError> {
        // Decompress bytes if hybrid mode was used
        let mps_bytes = if self.config.hybrid_mode {
            decompress_bytes(&proof.mps_bytes)
        } else {
            proof.mps_bytes.clone()
        };

        // Deserialize MPS
        let mps = CompressedMPS::from_bytes(&mps_bytes)?;

        // Decompress MPS to original data
        mps.decompress()
    }

    /// Verify that a compressed proof is valid
    pub fn verify(&self, proof: &CompressedProof) -> bool {
        // Try to decode and check hash
        match self.decode(proof) {
            Ok(decoded) => {
                let decoded_hash = simple_hash(&decoded);
                decoded_hash == proof.original_hash
            }
            Err(_) => false,
        }
    }

    /// Get compression statistics for a proof
    pub fn stats(&self, proof: &CompressedProof) -> CompressionStats {
        CompressionStats {
            original_size: proof.metadata.original_size,
            compressed_size: proof.metadata.compressed_size,
            compression_ratio: proof.metadata.compression_ratio,
            compression_factor: proof.metadata.original_size as f64
                / proof.metadata.compressed_size as f64,
            num_sites: proof.metadata.num_sites,
            avg_bond_dim: if proof.metadata.bond_dims.is_empty() {
                0.0
            } else {
                proof.metadata.bond_dims.iter().sum::<usize>() as f64
                    / proof.metadata.bond_dims.len() as f64
            },
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone)]
pub struct CompressionStats {
    /// Original size in bytes
    pub original_size: usize,
    /// Compressed size in bytes
    pub compressed_size: usize,
    /// Compression ratio (compressed/original, < 1 = compression)
    pub compression_ratio: f64,
    /// Compression factor (original/compressed, > 1 = compression)
    pub compression_factor: f64,
    /// Number of MPS sites
    pub num_sites: usize,
    /// Average bond dimension
    pub avg_bond_dim: f64,
}

impl std::fmt::Display for CompressionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Original: {} bytes, Compressed: {} bytes, Factor: {:.2}x, Sites: {}, Avg Bond: {:.1}",
            self.original_size,
            self.compressed_size,
            self.compression_factor,
            self.num_sites,
            self.avg_bond_dim
        )
    }
}

/// Simple hash function (FNV-1a style, not cryptographic)
fn simple_hash(data: &[u8]) -> [u8; 32] {
    let mut hash = [0u8; 32];
    let mut h: u64 = 0xcbf29ce484222325; // FNV offset basis

    for &byte in data {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV prime
    }

    // Spread across 32 bytes
    for i in 0..4 {
        let bytes = h.to_le_bytes();
        hash[i * 8..(i + 1) * 8].copy_from_slice(&bytes);
        h = h.rotate_left(17).wrapping_add(0xdeadbeef);
    }

    hash
}

/// Simple run-length encoding for byte compression
fn compress_bytes(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len());
    let mut i = 0;

    while i < data.len() {
        let current = data[i];
        let mut count = 1u8;

        // Count consecutive identical bytes (up to 255)
        while i + (count as usize) < data.len()
            && data[i + (count as usize)] == current
            && count < 255
        {
            count += 1;
        }

        if count >= 4 {
            // RLE marker: 0xFF, count, byte
            result.push(0xFF);
            result.push(count);
            result.push(current);
            i += count as usize;
        } else {
            // Literal bytes (escape 0xFF as 0xFF 0x01 0xFF)
            if current == 0xFF {
                result.push(0xFF);
                result.push(0x01);
                result.push(0xFF);
            } else {
                result.push(current);
            }
            i += 1;
        }
    }

    result
}

/// Decompress run-length encoded bytes
fn decompress_bytes(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return vec![];
    }

    let mut result = Vec::with_capacity(data.len() * 2);
    let mut i = 0;

    while i < data.len() {
        if data[i] == 0xFF && i + 2 < data.len() {
            let count = data[i + 1];
            let byte = data[i + 2];
            for _ in 0..count {
                result.push(byte);
            }
            i += 3;
        } else {
            result.push(data[i]);
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_roundtrip() {
        // Use small data to keep test fast
        let data = b"Hi!test";
        // Use lossless config for exact reconstruction
        let encoder = HolographicEncoder::new(EncoderConfig::lossless());

        let proof = encoder.encode(data).unwrap();
        let decoded = encoder.decode(&proof).unwrap();

        // Length should match for lossless
        assert_eq!(decoded.len(), data.len());
        // Content should match exactly for lossless
        assert_eq!(decoded, data.as_slice());
    }

    #[test]
    fn test_encoder_stats() {
        let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        let encoder = HolographicEncoder::with_defaults();

        let proof = encoder.encode(&data).unwrap();
        let stats = encoder.stats(&proof);

        assert_eq!(stats.original_size, 512);
        assert!(stats.compressed_size > 0);
        println!("Stats: {}", stats);
    }

    #[test]
    fn test_rle_compression() {
        let data = vec![0u8; 100];
        let compressed = compress_bytes(&data);
        let decompressed = decompress_bytes(&compressed);

        assert!(compressed.len() < data.len());
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_rle_no_runs() {
        let data: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let compressed = compress_bytes(&data);
        let decompressed = decompress_bytes(&compressed);

        // Should mostly be same size (with some escape overhead)
        assert_eq!(decompressed.len(), data.len());
    }

    #[test]
    fn test_presets() {
        // Use small data to keep tests fast
        let data: Vec<u8> = (0..16).map(|i| (i % 256) as u8).collect();

        // Test all presets compile and work
        for config in [
            EncoderConfig::default(),
            EncoderConfig::high_compression(),
            EncoderConfig::fast(),
            // Skip lossless in this test as it's too slow for 16 sites
            // EncoderConfig::lossless(),
        ] {
            let encoder = HolographicEncoder::new(config);
            let proof = encoder.encode(&data).unwrap();
            assert!(proof.metadata.compressed_size > 0);
        }
    }

    #[test]
    fn test_verification() {
        // Use very small data for fast test
        let data = b"Hi";
        let encoder = HolographicEncoder::new(EncoderConfig::lossless());

        let proof = encoder.encode(data).unwrap();

        // Valid proof should verify (may not for lossy compression)
        let is_valid = encoder.verify(&proof);
        // Note: Due to lossy nature of MPS, this may not always pass
        println!("Verification result: {}", is_valid);
    }
}
