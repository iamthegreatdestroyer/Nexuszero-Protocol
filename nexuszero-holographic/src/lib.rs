// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

pub mod tensor;
pub mod compression;
pub mod verification;
pub mod utils;
pub mod cli;

// Re-export the deprecated MPS for backward compatibility
#[allow(deprecated)]
pub use compression::mps::MPS;

// Re-export the new compression implementations
pub use compression::mps_compressed::{CompressedMPS, MPSConfig, MPSError, QuantizedTensor};
pub use compression::encoder_new::{HolographicEncoder, EncoderConfig, CompressedProof, CompressionStats};

// Re-export v2 compression with full features (LZ4, multi-precision, entropy analysis)
pub use compression::mps_v2::{
    CompressedTensorTrain, CompressionConfig, StoragePrecision,
    QuantizedTensorV2, TensorTrainStats, CompressionError,
    CompressionAnalysis, CompressionRecommendation,
    analyze_compression_potential, compress_proof_data, decompress_proof_data,
};

// Re-export neural-enhanced compression
pub use compression::neural::{
    NeuralCompressor, NeuralConfig, NeuralCompressedData, NeuralError,
    QuantizationParams, NeuralAnalysis, Device as NeuralDevice,
    neural_compress, neural_decompress,
};
