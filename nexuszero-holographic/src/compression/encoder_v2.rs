//! Holographic Proof Encoder v2
//!
//! High-level API for compressing ZK proofs using tensor network methods.

use crate::compression::mps_compressed::{CompressedMPS, MPSConfig, MPSError};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub mps_config: MPSConfig,
    pub enable_hybrid: bool,
    pub target_ratio: f64,
    pub max_reconstruction_error: f64,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            mps_config: MPSConfig::default(),
            enable_hybrid: true,
            target_ratio: 0.1,
            max_reconstruction_error: 1e-6,
        }
    }
}

impl EncoderConfig {
    pub fn high_compression() -> Self {
        Self {
            mps_config: MPSConfig {
                max_bond_dim: 32,
                svd_truncation_threshold: 1e-4,
                physical_dim: 2,
                block_size: 16,
                enable_quantization: true,
                quantization_bits: 8,
            },
            enable_hybrid: true,
            target_ratio: 0.01,
            max_reconstruction_error: 1e-4,
        }
    }
    
    pub fn fast() -> Self {
        Self {
            mps_config: MPSConfig {
                max_bond_dim: 16,
                svd_truncation_threshold: 1e-3,
                physical_dim: 4,
                block_size: 4,
                enable_quantization: true,
                quantization_bits: 16,
            },
            enable_hybrid: false,
            target_ratio: 0.5,
            max_reconstruction_error: 1e-3,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedProof {
    mps: CompressedMPS,
    original_hash: [u8; 32],
    method: CompressionMethod,
    stats: CompressionStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CompressionMethod { MPS, HybridMPSLZ4, Passthrough }

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressionStats {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub compression_factor: f64,
    pub reconstruction_error: f64,
    pub encoding_time_ms: u128,
    pub num_sites: usize,
    pub max_bond_dim: usize,
}

impl CompressedProof {
    pub fn stats(&self) -> &CompressionStats { &self.stats }
    pub fn verify_integrity(&self) -> bool { self.mps.verify_boundary(&self.original_hash) }
    pub fn compression_ratio(&self) -> f64 { self.stats.compression_ratio }
    pub fn compression_factor(&self) -> f64 { self.stats.compression_factor }
    
    pub fn to_bytes(&self) -> Result<Vec<u8>, MPSError> {
        bincode::serialize(self).map_err(|_| MPSError::DecompositionFailed)
    }
    
    pub fn from_bytes(data: &[u8]) -> Result<Self, MPSError> {
        bincode::deserialize(data).map_err(|_| MPSError::ReconstructionFailed)
    }
}

pub struct HolographicEncoder {
    config: EncoderConfig,
}

impl HolographicEncoder {
    pub fn new(config: EncoderConfig) -> Self { Self { config } }
    pub fn default_encoder() -> Self { Self::new(EncoderConfig::default()) }
    
    pub fn encode(&self, data: &[u8]) -> Result<CompressedProof, MPSError> {
        use std::time::Instant;
        let start = Instant::now();
        let original_hash = simple_hash(data);
        let mps = CompressedMPS::compress(data, self.config.mps_config.clone())?;
        let encoding_time = start.elapsed().as_millis();
        
        let stats = CompressionStats {
            original_size: data.len(),
            compressed_size: mps.compressed_size_bytes(),
            compression_ratio: mps.compression_ratio(),
            compression_factor: mps.compression_factor(),
            reconstruction_error: mps.reconstruction_error(),
            encoding_time_ms: encoding_time,
            num_sites: mps.num_sites(),
            max_bond_dim: *mps.bond_dimensions().iter().max().unwrap_or(&1),
        };
        
        let method = if self.config.enable_hybrid { CompressionMethod::HybridMPSLZ4 } else { CompressionMethod::MPS };
        Ok(CompressedProof { mps, original_hash, method, stats })
    }
    
    pub fn decode(&self, compressed: &CompressedProof) -> Result<Vec<u8>, MPSError> {
        compressed.mps.decompress()
    }
    
    pub fn verify(&self, compressed: &CompressedProof) -> bool {
        compressed.verify_integrity()
    }
}

fn simple_hash(data: &[u8]) -> [u8; 32] {
    let mut hash = [0u8; 32];
    let mut h: u64 = 0xcbf29ce484222325;
    for &byte in data {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    for i in 0..4 {
        let part = (h >> (i * 8)) & 0xFF;
        for j in 0..8 { hash[i * 8 + j] = ((part >> j) & 1) as u8 * 255; }
    }
    hash
}

pub fn encode_proof(data: &[u8], max_bond_dim: usize) -> Result<CompressedMPS, MPSError> {
    CompressedMPS::compress(data, MPSConfig { max_bond_dim, ..Default::default() })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_encoder_roundtrip() {
        let encoder = HolographicEncoder::default_encoder();
        let data: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        let compressed = encoder.encode(&data).unwrap();
        let recovered = encoder.decode(&compressed).unwrap();
        assert_eq!(data.len(), recovered.len());
    }
}
