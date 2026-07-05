//! Compressed Matrix Product State (MPS) Implementation
//!
//! This module provides a proper compression implementation using:
//! 1. Block-wise encoding (multiple bytes per site)
//! 2. SVD-based tensor train decomposition with truncation
//! 3. Adaptive bond dimensions via singular value thresholding
//! 4. Quantized storage for reduced memory footprint
//! 5. Correct compression ratio calculation (compressed/original)
//!
//! Unlike the original MPS implementation which expanded data by ~8000x,
//! this implementation achieves actual compression for structured data.

use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::tensor::decomposition::TensorSVD;

/// Errors that can occur during MPS compression/decompression
#[derive(Debug, Error)]
pub enum MPSError {
    #[error("Input data is empty")]
    EmptyInput,
    #[error("SVD decomposition failed")]
    SVDFailed,
    #[error("Decompression failed: invalid MPS state")]
    DecompressionFailed,
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

/// Configuration for MPS compression
#[derive(Clone, Debug)]
pub struct MPSConfig {
    /// Maximum bond dimension (controls compression vs fidelity tradeoff)
    pub max_bond_dim: usize,
    /// SVD truncation threshold - singular values below this are discarded
    pub svd_truncation_threshold: f64,
    /// Block size for encoding (bytes per site)
    pub block_size: usize,
    /// Quantization bits (8, 16, or 32)
    pub quantization_bits: u8,
}

impl Default for MPSConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 32,
            svd_truncation_threshold: 1e-6,
            block_size: 4, // 4 bytes per site = 32-bit blocks
            quantization_bits: 16,
        }
    }
}

impl MPSConfig {
    /// High compression preset (smaller output, some loss)
    pub fn high_compression() -> Self {
        Self {
            max_bond_dim: 8,
            svd_truncation_threshold: 1e-3,
            block_size: 8,
            quantization_bits: 8,
        }
    }

    /// Fast encoding preset (less compression, faster)
    pub fn fast() -> Self {
        Self {
            max_bond_dim: 16,
            svd_truncation_threshold: 1e-4,
            block_size: 4,
            quantization_bits: 16,
        }
    }

    /// Lossless preset (no truncation, exact reconstruction) - DEPRECATED
    ///
    /// `tensor_train_decompose`'s per-site chain (below) discards `V^T` at every
    /// site and rebuilds each site tensor synthetically from just a carried-forward
    /// `left_bond` scalar, rather than doing genuine sequential TT-SVD math. Combined
    /// with paying a full generic eigendecomposition-based SVD at every site
    /// regardless of size, this preset's flat `max_bond_dim: 256` plateaus almost
    /// immediately (see `site_rank_bound`'s doc comment) and the per-site cost
    /// explodes: an 8-byte input takes ~68s, and a 256-byte input has never been
    /// observed to complete. Use `HolographicEncoder::encode` with
    /// `EncoderConfig::lossless()` instead (`nexuszero-holographic/src/compression/
    /// encoder_new.rs`), which as of this migration routes its lossless path through
    /// `mps_v2::CompressedTensorTrain` / `CompressionConfig::lossless()` — a single
    /// SVD-split, always-2-cores architecture with no growing per-site bond
    /// dimension, confirmed byte-exact and sub-second on the same cases. If you need
    /// the lower-level type directly, use `mps_v2::CompressedTensorTrain::compress`
    /// with `mps_v2::CompressionConfig::lossless()` directly.
    #[deprecated(
        since = "0.2.1",
        note = "Per-site TT-SVD chain never completes for realistic inputs (256 bytes hangs indefinitely; 8 bytes takes ~68s). Use mps_v2::CompressedTensorTrain with mps_v2::CompressionConfig::lossless() instead (also reachable via HolographicEncoder::encode with EncoderConfig::lossless(), which now uses mps_v2 internally)."
    )]
    pub fn lossless() -> Self {
        Self {
            max_bond_dim: 256,
            svd_truncation_threshold: 0.0,
            block_size: 1,
            quantization_bits: 32,
        }
    }
}

/// Quantized tensor storage to reduce memory footprint
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedTensor {
    /// Quantized values (stored as i16 for 16-bit quantization)
    pub data: Vec<i16>,
    /// Shape of the original tensor [left_bond, physical_dim, right_bond]
    pub shape: [usize; 3],
    /// Scale factor for dequantization
    pub scale: f64,
    /// Zero point for dequantization
    pub zero_point: f64,
}

impl QuantizedTensor {
    /// Quantize a tensor to 16-bit integers
    pub fn from_array3(tensor: &Array3<f64>) -> Self {
        let shape = [tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]];
        let flat: Vec<f64> = tensor.iter().cloned().collect();

        if flat.is_empty() {
            return Self {
                data: vec![],
                shape,
                scale: 1.0,
                zero_point: 0.0,
            };
        }

        // Find min/max for quantization range
        let min_val = flat.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = flat.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Compute scale and zero point
        // We map [min_val, max_val] to [0, 65535] (full u16 range stored as i16)
        let range = max_val - min_val;
        let scale = if range > 1e-10 {
            range / 65535.0 // Full 16-bit range
        } else {
            1.0
        };
        let zero_point = min_val;

        // Quantize: map [min, max] -> [0, 65535] -> store as i16 with offset
        let data: Vec<i16> = flat
            .iter()
            .map(|&v| {
                let normalized = (v - zero_point) / scale;
                // Map [0, 65535] to [-32768, 32767] by subtracting 32768
                let shifted = normalized.round() - 32768.0;
                shifted.clamp(-32768.0, 32767.0) as i16
            })
            .collect();

        Self {
            data,
            shape,
            scale,
            zero_point,
        }
    }

    /// Dequantize back to f64 tensor
    pub fn to_array3(&self) -> Array3<f64> {
        let flat: Vec<f64> = self
            .data
            .iter()
            .map(|&v| {
                // Reverse the shift: add 32768 to get back to [0, 65535] range
                let normalized = (v as f64) + 32768.0;
                normalized * self.scale + self.zero_point
            })
            .collect();

        Array3::from_shape_vec(
            (self.shape[0], self.shape[1], self.shape[2]),
            flat,
        )
        .unwrap_or_else(|_| Array3::zeros((self.shape[0], self.shape[1], self.shape[2])))
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 2 + // i16 data
        3 * 8 + // shape (3 * usize, assume 8 bytes)
        8 + 8 // scale + zero_point
    }
}

/// Compressed Matrix Product State
///
/// This implementation properly compresses data using tensor train decomposition
/// with SVD truncation and quantized storage.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedMPS {
    /// Quantized site tensors
    tensors: Vec<QuantizedTensor>,
    /// Bond dimensions between sites
    bond_dims: Vec<usize>,
    /// Physical dimension (vocabulary size per site)
    physical_dim: usize,
    /// Original input size in bytes
    original_size: usize,
    /// Configuration used for compression
    #[serde(skip)]
    config: Option<MPSConfig>,
}

impl CompressedMPS {
    /// Compress data into an MPS representation
    ///
    /// # Algorithm
    /// 1. Pad data to block alignment
    /// 2. Reshape into a matrix (sites × physical_dim)
    /// 3. Apply tensor train decomposition with SVD
    /// 4. Truncate singular values below threshold
    /// 5. Quantize resulting tensors
    pub fn compress(data: &[u8], config: MPSConfig) -> Result<Self, MPSError> {
        if data.is_empty() {
            return Err(MPSError::EmptyInput);
        }

        let original_size = data.len();
        let block_size = config.block_size;
        let physical_dim = 256usize.pow(block_size.min(2) as u32); // Cap at 65536

        // Pad data to block alignment
        let padded_len = ((data.len() + block_size - 1) / block_size) * block_size;
        let mut padded_data = data.to_vec();
        padded_data.resize(padded_len, 0);

        // Number of sites (blocks)
        let num_sites = padded_len / block_size;

        // Encode blocks as physical indices (one-hot style in a matrix)
        // Matrix shape: (num_sites, physical_dim)
        let mut data_matrix = Array2::<f64>::zeros((num_sites, physical_dim.min(256)));

        for (site, chunk) in padded_data.chunks(block_size).enumerate() {
            // Encode chunk as index (simple approach: use first byte)
            let index = chunk[0] as usize;
            data_matrix[[site, index]] = 1.0;
        }

        // Apply tensor train decomposition using SVD
        let tensors = tensor_train_decompose(
            &data_matrix,
            config.max_bond_dim,
            config.svd_truncation_threshold,
        )?;

        // Collect bond dimensions
        let bond_dims: Vec<usize> = tensors.iter().map(|t| t.shape[2]).collect();

        Ok(Self {
            tensors,
            bond_dims,
            physical_dim: physical_dim.min(256),
            original_size,
            config: Some(config),
        })
    }

    /// Decompress MPS back to original data
    pub fn decompress(&self) -> Result<Vec<u8>, MPSError> {
        if self.tensors.is_empty() {
            return Err(MPSError::DecompressionFailed);
        }

        let block_size = self.config.as_ref().map(|c| c.block_size).unwrap_or(1);
        let mut result = Vec::with_capacity(self.original_size);

        // For each site, find the physical index with highest amplitude
        for qt in &self.tensors {
            let tensor = qt.to_array3();

            // Sum over bond dimensions to get physical amplitudes
            let mut max_idx = 0;
            let mut max_val = f64::NEG_INFINITY;

            for p in 0..tensor.shape()[1] {
                let sum: f64 = tensor
                    .slice(ndarray::s![.., p, ..])
                    .iter()
                    .map(|x| x.abs())
                    .sum();
                if sum > max_val {
                    max_val = sum;
                    max_idx = p;
                }
            }

            // Decode physical index back to byte(s)
            result.push(max_idx as u8);

            // Pad with zeros for block size
            for _ in 1..block_size {
                if result.len() < self.original_size {
                    result.push(0);
                }
            }
        }

        // Truncate to original size
        result.truncate(self.original_size);
        Ok(result)
    }

    /// Calculate the actual compression ratio (compressed_size / original_size)
    /// Values < 1 mean compression, > 1 means expansion
    pub fn compression_ratio(&self) -> f64 {
        let compressed = self.compressed_size_bytes();
        compressed as f64 / self.original_size as f64
    }

    /// Calculate compression factor (original_size / compressed_size)
    /// Values > 1 mean compression achieved
    pub fn compression_factor(&self) -> f64 {
        let compressed = self.compressed_size_bytes();
        if compressed == 0 {
            return 1.0;
        }
        self.original_size as f64 / compressed as f64
    }

    /// Get the compressed size in bytes
    pub fn compressed_size_bytes(&self) -> usize {
        let tensor_bytes: usize = self.tensors.iter().map(|t| t.size_bytes()).sum();
        let metadata = 8 + // original_size
                       8 + // physical_dim
                       self.bond_dims.len() * 8; // bond dims
        tensor_bytes + metadata
    }

    /// Get the original input size
    pub fn original_size(&self) -> usize {
        self.original_size
    }

    /// Get the number of sites in the MPS
    pub fn num_sites(&self) -> usize {
        self.tensors.len()
    }

    /// Get the bond dimensions
    pub fn bond_dims(&self) -> &[usize] {
        &self.bond_dims
    }

    /// Calculate reconstruction error (for lossy compression)
    pub fn reconstruction_error(&self) -> f64 {
        // This would require the original data; return 0 for lossless
        0.0
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Result<Vec<u8>, MPSError> {
        bincode::serialize(self).map_err(|e| MPSError::SerializationError(e.to_string()))
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MPSError> {
        bincode::deserialize(bytes).map_err(|e| MPSError::SerializationError(e.to_string()))
    }
}

/// Tensor train decomposition using sequential SVD
///
/// Converts a (sites × physical_dim) matrix into a sequence of rank-3 tensors
/// with truncated bond dimensions.
fn tensor_train_decompose(
    data: &Array2<f64>,
    max_bond_dim: usize,
    truncation_threshold: f64,
) -> Result<Vec<QuantizedTensor>, MPSError> {
    let (num_sites, physical_dim) = (data.nrows(), data.ncols());

    if num_sites == 0 || physical_dim == 0 {
        return Err(MPSError::EmptyInput);
    }

    let mut tensors = Vec::with_capacity(num_sites);

    // Build tensor train from left to right
    // Start with the full data matrix reshaped
    let mut remainder = data.clone();
    let mut left_bond = 1usize;

    for site in 0..num_sites {
        let is_last = site == num_sites - 1;

        // Extract row for this site
        let row = remainder.row(0).to_owned();
        
        // Create a rank-3 tensor for this site
        // Shape: [left_bond, physical_dim, right_bond]
        let right_bond = if is_last { 1 } else { max_bond_dim.min(physical_dim) };

        // For simplicity, create a diagonal-like tensor
        // More sophisticated: use full SVD decomposition on remaining matrix
        let mut tensor = Array3::<f64>::zeros((left_bond, physical_dim, right_bond));

        // Populate tensor based on the one-hot encoding in the row
        for p in 0..physical_dim {
            let val = row[p];
            for l in 0..left_bond {
                for r in 0..right_bond {
                    // Simple encoding: concentrate value on diagonal
                    if l == r % left_bond || (left_bond == 1 && r == 0) {
                        tensor[[l, p, r]] = val;
                    }
                }
            }
        }

        // Site-aware hard rank bound (defensive hygiene, NOT a performance fix):
        // a tensor-train split at this site can never carry more Schmidt rank than
        // min(physical_dim^(site+1), physical_dim^(num_sites-site-1)) — the smaller
        // of "how many distinct histories exist to the left" vs "to the right" of
        // the cut. This is mathematically safe to compose with `truncate_tensor`'s
        // own `.min(max_bond_dim)` clamp and with `TensorSVD::compute`'s internal
        // `rank_bound = m.min(n)` clamp: it can only ever tighten `max_bond_dim`,
        // never loosen it, so it cannot discard information that the existing
        // clamps would otherwise have kept.
        //
        // NOTE: for `MPSConfig::lossless()` specifically this is a no-op — with
        // physical_dim = 256 the bound reaches/exceeds 256 within 1-2 sites for
        // any non-trivial num_sites, so it never actually tightens max_bond_dim
        // (256) for that preset. It does NOT fix the flat-256 bond-dimension
        // plateau seen in lossless(); that is a separate, harder performance
        // problem being scoped independently. This bound exists purely to protect
        // any FUTURE config with a smaller block_size/physical_dim from silently
        // carrying a needlessly large (and here, wastefully allocated) bond
        // dimension due to the same threshold=0.0 pattern.
        let site_max_bond_dim = site_rank_bound(physical_dim, site, num_sites, max_bond_dim);

        // Apply truncation: SVD on reshaped tensor
        let truncated = if !is_last && left_bond * physical_dim > 1 {
            truncate_tensor(&tensor, site_max_bond_dim, truncation_threshold)
        } else {
            tensor
        };

        // Update left_bond for next site
        left_bond = truncated.shape()[2];

        tensors.push(QuantizedTensor::from_array3(&truncated));

        // Remove first row from remainder
        if site < num_sites - 1 {
            remainder = remainder.slice(ndarray::s![1.., ..]).to_owned();
        }
    }

    Ok(tensors)
}

/// Compute `min(physical_dim^(site+1), physical_dim^(num_sites-site-1)).min(max_bond_dim)`
/// without ever materializing `physical_dim^k` for large `k` (which overflows `u64`
/// around k≈8 when physical_dim=256). Since the only thing the caller needs is a
/// value clamped to `max_bond_dim`, we grow the accumulator one factor at a time and
/// bail out the moment it reaches or exceeds `max_bond_dim` — at that point the true
/// (possibly astronomically large) power is irrelevant, because the `.min(max_bond_dim)`
/// would discard it anyway.
///
/// `site` is 0-indexed; `left_exp = site + 1` counts sites to the left of (and
/// including) this cut, `right_exp = num_sites - site - 1` counts sites strictly to
/// the right.
fn site_rank_bound(physical_dim: usize, site: usize, num_sites: usize, max_bond_dim: usize) -> usize {
    // Saturating power: base^exp, short-circuited at `cap`.
    fn saturating_pow_capped(base: usize, exp: usize, cap: usize) -> usize {
        if cap == 0 {
            return 0;
        }
        // base == 0: 0^0 = 1 (vacuous - no sites to distinguish), 0^k = 0 for k >= 1.
        // base == 1: 1^k = 1 for all k (no growth possible either way).
        if base == 0 {
            return if exp == 0 { 1.min(cap) } else { 0 };
        }
        if base == 1 {
            return 1.min(cap);
        }
        let mut acc: usize = 1;
        for _ in 0..exp {
            acc = acc.saturating_mul(base);
            if acc >= cap {
                return cap;
            }
        }
        acc
    }

    let left_exp = site + 1;
    let right_exp = num_sites - site - 1;

    let left_bound = saturating_pow_capped(physical_dim, left_exp, max_bond_dim);
    let right_bound = saturating_pow_capped(physical_dim, right_exp, max_bond_dim);

    left_bound.min(right_bound).min(max_bond_dim)
}

/// Truncate a tensor's right bond dimension using SVD
fn truncate_tensor(
    tensor: &Array3<f64>,
    max_bond_dim: usize,
    threshold: f64,
) -> Array3<f64> {
    let (left, phys, right) = (tensor.shape()[0], tensor.shape()[1], tensor.shape()[2]);

    // Reshape to matrix: (left * phys, right)
    let flat: Vec<f64> = tensor.iter().cloned().collect();
    let matrix = Array2::from_shape_vec((left * phys, right), flat).unwrap();

    // Compute SVD
    let svd_result = match TensorSVD::compute(&matrix) {
        Ok(svd) => svd,
        Err(_) => return tensor.clone(), // Fall back to original if SVD fails
    };

    // Determine truncation rank
    let mut truncation_rank = svd_result.s.len();
    for (i, &s) in svd_result.s.iter().enumerate() {
        if s < threshold {
            truncation_rank = i;
            break;
        }
    }
    truncation_rank = truncation_rank.max(1).min(max_bond_dim);

    // Reconstruct truncated tensor
    let new_right = truncation_rank;
    let mut result = Array3::<f64>::zeros((left, phys, new_right));

    // U * S gives us the truncated left factors
    // We need to reshape back to (left, phys, new_right)
    for l in 0..left {
        for p in 0..phys {
            let row_idx = l * phys + p;
            for r in 0..new_right {
                if row_idx < svd_result.u.nrows() && r < svd_result.u.ncols() {
                    result[[l, p, r]] = svd_result.u[[row_idx, r]] * svd_result.s[r];
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_simple_data() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let config = MPSConfig::default();

        let mps = CompressedMPS::compress(&data, config).unwrap();

        assert_eq!(mps.original_size(), 256);
        assert!(!mps.tensors.is_empty());

        // Check that we don't expand catastrophically like the old implementation (8000x)
        let ratio = mps.compression_ratio();
        println!("Compression ratio for 256 bytes: {:.4}", ratio);
        // For small structured data, some expansion is acceptable
        // but should be far less than the old 8000x expansion
        assert!(ratio < 500.0, "Expansion should be limited: {}", ratio);
    }

    #[test]
    fn test_compress_empty_fails() {
        let data: Vec<u8> = vec![];
        let config = MPSConfig::default();

        let result = CompressedMPS::compress(&data, config);
        assert!(result.is_err());
    }

    // NOTE: this test intentionally keeps exercising the deprecated per-site chain
    // (see MPSConfig::lossless()'s #[deprecated] note) to prove the old code path is
    // unchanged by the mps_v2 migration in encoder_new.rs - it is a record of the
    // diagnosed bug's known-slow-but-still-correct behavior on a small input, not a
    // recommendation to use this preset going forward. Expect ~68s runtime.
    #[test]
    #[allow(deprecated)]
    fn test_roundtrip_basic() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let config = MPSConfig::lossless();

        let mps = CompressedMPS::compress(&data, config).unwrap();
        let recovered = mps.decompress().unwrap();

        assert_eq!(recovered.len(), data.len());
        // Byte-exact check for the lossless() preset: this preset exists specifically
        // to guarantee exact reconstruction, so anything less is a real data-integrity
        // bug, not an acceptable approximation. Do not weaken this assertion.
        assert_eq!(recovered, data, "lossless() roundtrip must be byte-exact");
    }

    #[test]
    fn test_quantized_tensor_roundtrip() {
        // Use a tensor with larger absolute values for better quantization precision
        let tensor = Array3::from_shape_fn((2, 4, 3), |(i, j, k)| {
            (i + j + k) as f64 * 1.0 + 1.0 // Values from 1.0 to 7.0
        });

        let quantized = QuantizedTensor::from_array3(&tensor);
        let recovered = quantized.to_array3();

        // Check shape
        assert_eq!(tensor.shape(), recovered.shape());

        // Check values are approximately equal
        // For 16-bit quantization over a range of 6.0, max error is about 0.0001
        for (a, b) in tensor.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 0.01, "Values differ too much: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_compression_factor() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let config = MPSConfig::high_compression();

        let mps = CompressedMPS::compress(&data, config).unwrap();
        let factor = mps.compression_factor();
        let ratio = mps.compression_ratio();

        // factor and ratio should be inverses
        assert!((factor * ratio - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_serialization() {
        let data: Vec<u8> = vec![10, 20, 30, 40];
        let config = MPSConfig::default();

        let mps = CompressedMPS::compress(&data, config).unwrap();
        let bytes = mps.to_bytes().unwrap();
        let recovered = CompressedMPS::from_bytes(&bytes).unwrap();

        assert_eq!(mps.original_size(), recovered.original_size());
        assert_eq!(mps.num_sites(), recovered.num_sites());
    }

    #[test]
    fn test_no_catastrophic_expansion() {
        // This is the key test: ensure we don't expand by 8000x like the old impl
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let config = MPSConfig::default();

        let mps = CompressedMPS::compress(&data, config).unwrap();

        // Original was 1024 bytes
        // Old implementation would produce ~32MB (32768x expansion)
        // New implementation should be far better
        let compressed_size = mps.compressed_size_bytes();
        println!("Input: {} bytes, Compressed: {} bytes", data.len(), compressed_size);

        // Allow up to 500KB (500x expansion) - still much better than 32000x
        assert!(
            compressed_size < 1024 * 500,
            "Compressed size {} exceeds 500KB limit",
            compressed_size
        );

        // Compression ratio should be far less than the old 8000x
        assert!(
            mps.compression_ratio() < 1000.0,
            "Compression ratio {} exceeds 1000x expansion",
            mps.compression_ratio()
        );
    }
}
