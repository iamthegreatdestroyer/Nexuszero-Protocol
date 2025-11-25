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
