pub mod mps;
pub mod mps_compressed;  // Fixed implementation with actual compression
pub mod mps_v2;          // NEW: Full reference implementation with LZ4, multi-precision, entropy analysis
pub mod hybrid;          // NEW: Practical hybrid compression (LZ4 + RLE + Delta)
pub mod encoder_new;     // High-level encoder API
pub mod boundary;
pub mod peps;
pub mod encoder;
pub mod decoder;
pub mod neural;          // Neural-enhanced compression with learned quantization

// Re-export neural types for convenience
pub use neural::{NeuralCompressor, NeuralConfig, NeuralCompressedData, NeuralError, QuantizationParams};

// Re-export hybrid compression for practical use
pub use hybrid::{HybridCompressor, HybridConfig, HybridCompressed, HybridStats, CompressionStrategy};