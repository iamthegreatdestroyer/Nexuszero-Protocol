//! Nexuszero Crypto - Quantum-Resistant Zero-Knowledge Proof System
//!
//! This library provides lattice-based cryptographic primitives for building
//! zero-knowledge proof systems that are resistant to quantum attacks.
//!
//! # Features
//!
//! - **LWE Encryption**: Learning With Errors based encryption
//! - **Ring-LWE**: Efficient ring-based variant with NTT optimization
//! - **Zero-Knowledge Proofs**: Statement/Witness/Proof system
//! - **Parameter Selection**: Automatic security parameter optimization
//!
//! # Example
//!
//! ```rust,no_run
//! use nexuszero_crypto::{SecurityLevel, CryptoParameters};
//!
//! // Select security parameters
//! let params = CryptoParameters::from_security_level(SecurityLevel::Bit128);
//!
//! // Create statement and witness (examples to be implemented)
//! // let statement = ...;
//! // let witness = ...;
//! // let proof = prove(&statement, &witness, &params)?;
//! // verify(&statement, &proof, &params)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod lattice;
pub mod params;
pub mod proof;
pub mod utils;

// Re-export commonly used types
pub use params::{CryptoParameters, SecurityLevel};

/// Custom error type for cryptographic operations
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    /// Invalid security parameter
    #[error("Invalid security parameter: {0}")]
    InvalidParameter(String),

    /// Encryption/Decryption error
    #[error("Encryption error: {0}")]
    EncryptionError(String),

    /// Proof generation error
    #[error("Proof generation failed: {0}")]
    ProofError(String),

    /// Verification error
    #[error("Verification failed: {0}")]
    VerificationError(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Mathematical operation error
    #[error("Math error: {0}")]
    MathError(String),
}

/// Result type for cryptographic operations
pub type CryptoResult<T> = Result<T, CryptoError>;

/// Trait for lattice-based cryptographic parameters
pub trait LatticeParameters {
    /// Get the dimension parameter
    fn dimension(&self) -> usize;

    /// Get the modulus
    fn modulus(&self) -> u64;

    /// Get the error distribution parameter
    fn sigma(&self) -> f64;

    /// Validate parameters
    fn validate(&self) -> CryptoResult<()>;
}

/// Trait for proof systems
pub trait ProofSystem {
    /// Statement type
    type Statement;

    /// Witness type
    type Witness;

    /// Proof type
    type Proof;

    /// Generate a proof
    fn prove(
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> CryptoResult<Self::Proof>;

    /// Verify a proof
    fn verify(statement: &Self::Statement, proof: &Self::Proof) -> CryptoResult<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_library_initialization() {
        // Basic smoke test
        assert!(true);
    }
}
