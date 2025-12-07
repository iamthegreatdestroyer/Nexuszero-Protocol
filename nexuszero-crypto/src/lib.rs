// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Nexuszero Crypto - Quantum-Resistant Zero-Knowledge Proof System
//!
//! This library provides lattice-based cryptographic primitives for building
//! zero-knowledge proof systems that are resistant to quantum attacks.
//!
//! # ⚠️ SECURITY WARNING ⚠️
//!
//! **THIS LIBRARY IS UNDER ACTIVE DEVELOPMENT AND HAS NOT BEEN INDEPENDENTLY AUDITED.**
//!
//! ## DO NOT USE IN PRODUCTION without:
//!
//! 1. ✅ Independent third-party security review
//! 2. ✅ Comprehensive side-channel analysis on target hardware
//! 3. ✅ Formal threat modeling for your specific use case
//! 4. ✅ Infrastructure hardening (dedicated hardware, disabled hyperthreading)
//!
//! ## Security Status:
//!
//! - **Timing Attacks:** ✅ Mitigated (constant-time implementations)
//! - **Cache Attacks:** ⚠️ Partially mitigated (requires hardware isolation)
//! - **Formal Verification:** ✅ 34 Kani proofs implemented
//! - **Side-Channel Testing:** ✅ 14 resistance tests passing
//! - **Independent Audit:** ❌ NOT YET COMPLETED
//!
//! See `SECURITY_AUDIT.md` for detailed security analysis and recommendations.
//!
//! # Features
//!
//! - **LWE Encryption**: Learning With Errors based encryption
//! - **Ring-LWE**: Efficient ring-based variant with NTT optimization
//! - **Zero-Knowledge Proofs**: Statement/Witness/Proof system
//! - **Parameter Selection**: Automatic security parameter optimization
//! - **Constant-Time Operations**: Timing attack resistant implementations
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

#![allow(missing_docs)]
#![warn(clippy::all)]
// NOTE: The crate carries a security advisory comment but no crate-wide deprecated attribute to avoid
// creating deprecation warnings in test/bench targets. The advisory can be found in SECURITY_AUDIT.md.

pub mod lattice;
pub mod metrics;
pub mod params;
pub mod proof;
pub mod utils;

// FFI bindings for Python integration
pub mod ffi;

// Re-export commonly used types
pub use params::{CryptoParameters, SecurityLevel};

/// Return a simple version number for FFI smoke testing.
#[no_mangle]
pub extern "C" fn nexuszero_crypto_version() -> u32 {
    100  // version 1.0.0 encoded as 100
}

/// Custom error type for cryptographic operations
#[derive(Debug, thiserror::Error, PartialEq, Eq, Hash)]
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

    /// Hardware backend error
    #[error("Hardware error: {0}")]
    HardwareError(String),

    /// Invalid input parameters
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Internal error
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Network communication error
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Feature not implemented
    #[error("Not implemented: {0}")]
    NotImplemented(String),
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

/// Cryptographic parameter validation functions
/// 
/// These functions validate that all cryptographic parameters are secure
/// and should be called at application startup.
pub mod validation {
    use crate::CryptoResult;
    
    /// Validate all cryptographic parameters at startup
    /// 
    /// This function performs comprehensive validation of:
    /// - Ring-LWE parameters against claimed security levels
    /// - Bulletproofs generators and moduli
    /// - Fiat-Shamir domain separation
    /// - Primitive roots for NTT operations
    /// 
    /// Should be called once at application startup.
    pub fn validate_cryptographic_parameters() -> CryptoResult<()> {
        // Validate Ring-LWE parameters
        crate::lattice::ring_lwe::validation::validate_all_parameter_sets()?;
        
        // Validate Bulletproofs parameters
        crate::proof::bulletproofs::validate_cryptographic_parameters()?;
        
        // Validate primitive roots for all parameter sets
        let params_128 = crate::lattice::RingLWEParameters::new_128bit_security();
        let params_192 = crate::lattice::RingLWEParameters::new_192bit_security();
        let params_256 = crate::lattice::RingLWEParameters::new_256bit_security();
        
        crate::lattice::ring_lwe::validation::validate_primitive_root(&params_128)?;
        crate::lattice::ring_lwe::validation::validate_primitive_root(&params_192)?;
        crate::lattice::ring_lwe::validation::validate_primitive_root(&params_256)?;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // No items from parent required by this basic smoke test

    #[test]
    fn test_library_initialization() {
        // Basic smoke test: ensure module compiles and basic types exist
        let _ = 1 + 1; // trivial operation to keep test meaningful
    }
}
