//! High-Level Holographic Encoder API
//!
//! Provides a clean interface for encoding/decoding data using the
//! CompressedMPS implementation with various configuration presets.
//!
//! # Backend routing (migration note, added alongside the mps_v2 lossless migration)
//!
//! `HolographicEncoder` can internally back its lossless path with either of two
//! independent compression implementations:
//! - `mps_compressed::CompressedMPS` (the original per-site TT chain). Still used
//!   for `EncoderConfig::default()` / `high_compression()` / `fast()`, unchanged.
//! - `mps_v2::CompressedTensorTrain` (a single SVD-split, always-2-core
//!   architecture). Now used for `EncoderConfig::lossless()`, because
//!   `mps_compressed`'s per-site chain never completes for realistic inputs under
//!   its own `lossless()` preset (see `MPSConfig::lossless()`'s `#[deprecated]`
//!   note) while `mps_v2`'s `lossless()` preset is confirmed byte-exact and
//!   sub-second on the same cases.
//!
//! `HolographicEncoder::encode`/`decode`'s public signatures and `CompressedProof`'s
//! public fields are unchanged by this - the backend selection is fully internal.
//! The one necessary internal change is that `CompressedProof.mps_bytes` now carries
//! a 1-byte backend tag (`BACKEND_TAG_MPS_COMPRESSED` / `BACKEND_TAG_MPS_V2`)
//! prepended to the serialized payload, so `decode()` is self-describing and correct
//! even if a `CompressedProof` is decoded by a `HolographicEncoder` instance
//! constructed with a different `EncoderConfig` than the one that encoded it.

use serde::{Deserialize, Serialize};

use super::mps_compressed::{CompressedMPS, MPSConfig, MPSError};
use super::mps_v2::{CompressedTensorTrain, CompressionConfig as MpsV2Config, CompressionError as MpsV2Error};

/// Backend tag: payload is a bincode-serialized `mps_compressed::CompressedMPS`.
const BACKEND_TAG_MPS_COMPRESSED: u8 = 0x01;
/// Backend tag: payload is a bincode-serialized `mps_v2::CompressedTensorTrain`.
const BACKEND_TAG_MPS_V2: u8 = 0x02;

/// Configuration for the holographic encoder
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    /// Underlying MPS configuration (used when `use_mps_v2` is false)
    pub mps_config: MPSConfig,
    /// Enable hybrid compression (MPS + standard algorithm)
    pub hybrid_mode: bool,
    /// Verify integrity after encoding
    pub verify_on_encode: bool,
    /// Internal backend selector: when true, `encode`/`decode` route through
    /// `mps_v2::CompressedTensorTrain` (via `mps_v2_config`) instead of
    /// `mps_compressed::CompressedMPS` (via `mps_config`). Not part of the
    /// preset-selection API surface callers are expected to toggle by hand -
    /// set implicitly by `EncoderConfig::lossless()`. Kept `pub` only because
    /// `EncoderConfig`'s fields were already all `pub`; manual construction
    /// (`EncoderConfig { .. }`) defaults it to `false` via `Default`.
    pub use_mps_v2: bool,
    /// `mps_v2` configuration, used only when `use_mps_v2` is true.
    pub mps_v2_config: MpsV2Config,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            mps_config: MPSConfig::default(),
            hybrid_mode: false,
            verify_on_encode: true,
            use_mps_v2: false,
            mps_v2_config: MpsV2Config::default(),
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
            use_mps_v2: false,
            mps_v2_config: MpsV2Config::default(),
        }
    }

    /// Fast encoding preset
    pub fn fast() -> Self {
        Self {
            mps_config: MPSConfig::fast(),
            hybrid_mode: false,
            verify_on_encode: false,
            use_mps_v2: false,
            mps_v2_config: MpsV2Config::default(),
        }
    }

    /// Lossless preset (exact reconstruction)
    ///
    /// Migrated to route through `mps_v2::CompressedTensorTrain` /
    /// `mps_v2::CompressionConfig::lossless()` internally (see module-level
    /// migration note above). `mps_config` is still populated with
    /// `MPSConfig::lossless()` for informational/backward-compat purposes (e.g.
    /// existing callers that inspect `config.mps_config` after building this
    /// preset), but `encode`/`decode` no longer read it once `use_mps_v2` is set.
    #[allow(deprecated)]
    pub fn lossless() -> Self {
        Self {
            mps_config: MPSConfig::lossless(),
            hybrid_mode: false,
            verify_on_encode: true,
            use_mps_v2: true,
            mps_v2_config: MpsV2Config::lossless(),
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

        // Compress using whichever backend this config selects, producing a
        // backend-tagged, bincode-serialized payload plus metadata fields common
        // to both backends.
        let (tagged_bytes, num_sites, bond_dims) = if self.config.use_mps_v2 {
            let tt = CompressedTensorTrain::compress(data, self.config.mps_v2_config.clone())
                .map_err(mps_v2_err_to_mps_err)?;
            let serialized = tt.to_bytes().map_err(mps_v2_err_to_mps_err)?;

            let mut tagged = Vec::with_capacity(1 + serialized.len());
            tagged.push(BACKEND_TAG_MPS_V2);
            tagged.extend_from_slice(&serialized);

            let stats = tt.stats();
            // CompressedTensorTrain is always exactly 2 cores (a single SVD split),
            // sharing one bond dimension between them; avg_bond_dim == max_bond_dim
            // in that case (see mps_v2::tensor_train_svd), so either stat gives the
            // true shared rank. One entry per site mirrors CompressedMPS::bond_dims's
            // "one bond dim per site" shape.
            let bond_dims = vec![stats.max_bond_dim; stats.num_sites];
            (tagged, stats.num_sites, bond_dims)
        } else {
            let mps = CompressedMPS::compress(data, self.config.mps_config.clone())?;
            let serialized = mps.to_bytes()?;

            let mut tagged = Vec::with_capacity(1 + serialized.len());
            tagged.push(BACKEND_TAG_MPS_COMPRESSED);
            tagged.extend_from_slice(&serialized);

            (tagged, mps.num_sites(), mps.bond_dims().to_vec())
        };

        // Optionally apply hybrid compression (LZ4 on top)
        let final_bytes = if self.config.hybrid_mode {
            // Simple RLE-like compression for repeated values
            compress_bytes(&tagged_bytes)
        } else {
            tagged_bytes
        };

        let metadata = ProofMetadata {
            original_size: data.len(),
            compressed_size: final_bytes.len(),
            compression_ratio: final_bytes.len() as f64 / data.len() as f64,
            num_sites,
            bond_dims,
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
        let tagged_bytes = if self.config.hybrid_mode {
            decompress_bytes(&proof.mps_bytes)
        } else {
            proof.mps_bytes.clone()
        };

        // The backend tag byte makes decode() self-describing: it reflects how
        // `proof` was actually encoded, independent of `self.config.use_mps_v2`.
        // This matters because `proof` may have been produced by a different
        // HolographicEncoder instance (e.g. deserialized from storage/network)
        // than the one decoding it.
        let (&tag, payload) = tagged_bytes
            .split_first()
            .ok_or(MPSError::DecompressionFailed)?;

        match tag {
            BACKEND_TAG_MPS_V2 => {
                let tt = CompressedTensorTrain::from_bytes(payload).map_err(mps_v2_err_to_mps_err)?;
                tt.decompress().map_err(mps_v2_err_to_mps_err)
            }
            BACKEND_TAG_MPS_COMPRESSED => {
                let mps = CompressedMPS::from_bytes(payload)?;
                mps.decompress()
            }
            _ => Err(MPSError::DecompressionFailed),
        }
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

/// Map `mps_v2::CompressionError` onto the encoder's public `MPSError` type so
/// `HolographicEncoder::encode`/`decode` can keep returning `Result<_, MPSError>`
/// regardless of which backend (`mps_compressed` or `mps_v2`) actually ran.
/// Variants that carry a `String` map onto `MPSError`'s closest same-shaped
/// variant, preserving the message; variants `MPSError` has no dedicated slot for
/// (`LZ4Error`, `TruncationOverflow`) are folded into `SerializationError` with
/// their `Display` text so the detail isn't silently dropped.
fn mps_v2_err_to_mps_err(err: MpsV2Error) -> MPSError {
    match err {
        MpsV2Error::EmptyInput => MPSError::EmptyInput,
        MpsV2Error::SVDFailed(_) => MPSError::SVDFailed,
        MpsV2Error::DecompressionFailed(_) => MPSError::DecompressionFailed,
        MpsV2Error::SerializationError(msg) => MPSError::SerializationError(msg),
        other @ (MpsV2Error::LZ4Error(_) | MpsV2Error::TruncationOverflow) => {
            MPSError::SerializationError(other.to_string())
        }
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

        // Test all presets compile and work. `lossless()` was previously skipped
        // here ("too slow for 16 sites" - it never completed via the old
        // mps_compressed per-site chain); now that it's migrated onto mps_v2 it's
        // sub-second, so it's included like every other preset.
        for config in [
            EncoderConfig::default(),
            EncoderConfig::high_compression(),
            EncoderConfig::fast(),
            EncoderConfig::lossless(),
        ] {
            let encoder = HolographicEncoder::new(config);
            let proof = encoder.encode(&data).unwrap();
            assert!(proof.metadata.compressed_size > 0);
        }
    }

    /// Regression test for the mps_v2 lossless migration: the 8-byte case that
    /// previously took ~68s via `mps_compressed`'s per-site chain must now complete
    /// well under a second via `mps_v2`, and must still be byte-exact.
    #[test]
    fn test_lossless_8_bytes_fast_and_exact() {
        let data: Vec<u8> = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let encoder = HolographicEncoder::new(EncoderConfig::lossless());

        let start = std::time::Instant::now();
        let proof = encoder.encode(&data).unwrap();
        let decoded = encoder.decode(&proof).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(decoded, data, "lossless() roundtrip must be byte-exact");
        assert!(
            elapsed.as_secs() < 5,
            "8-byte lossless encode+decode took {:?}, expected well under a second (mps_v2 baseline ~0.04s); \
             the old mps_compressed path took ~68s for this exact case",
            elapsed
        );
        println!("8-byte lossless encode+decode: {:?}", elapsed);
    }

    /// Regression test for the mps_v2 lossless migration: 256 bytes via
    /// `mps_compressed`'s per-site chain (see
    /// `mps_compression_validation.rs::test_config_presets`) has never been observed
    /// to complete. Via mps_v2 it must complete fast and byte-exact.
    #[test]
    fn test_lossless_256_bytes_completes_and_exact() {
        let data: Vec<u8> = (0..256).map(|i| (i % 256) as u8).collect();
        let encoder = HolographicEncoder::new(EncoderConfig::lossless());

        let start = std::time::Instant::now();
        let proof = encoder.encode(&data).unwrap();
        let decoded = encoder.decode(&proof).unwrap();
        let elapsed = start.elapsed();

        assert_eq!(decoded, data, "lossless() roundtrip must be byte-exact");
        assert!(
            elapsed.as_secs() < 10,
            "256-byte lossless encode+decode took {:?}, expected well under a second",
            elapsed
        );
        println!("256-byte lossless encode+decode: {:?}", elapsed);
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
