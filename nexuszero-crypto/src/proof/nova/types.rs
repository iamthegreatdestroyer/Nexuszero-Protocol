// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Nova type definitions and error handling
//!
//! This module defines the core types used throughout the Nova integration,
//! including curve types, scalar fields, and error handling.

use std::fmt;
use thiserror::Error;
use serde::{Serialize, Deserialize};

// Re-export pasta curve types when nova feature is enabled
#[cfg(feature = "nova")]
pub use pasta_curves::{
    pallas::{Point as PallasPoint, Scalar as PallasScalar},
    vesta::{Point as VestaPoint, Scalar as VestaScalar},
    Fp, Fq,
};

#[cfg(feature = "nova")]
pub use ff::{Field, PrimeField};

/// Type alias for the primary curve group element (Pallas)
#[cfg(feature = "nova")]
pub type G1 = PallasPoint;

/// Type alias for the secondary curve group element (Vesta)
#[cfg(feature = "nova")]
pub type G2 = VestaPoint;

/// Type alias for the primary scalar field (Pallas scalar)
#[cfg(feature = "nova")]
pub type Scalar = PallasScalar;

/// Type alias for the secondary scalar field (Vesta scalar)
#[cfg(feature = "nova")]
pub type Scalar2 = VestaScalar;

// Placeholder types when nova feature is not enabled
#[cfg(not(feature = "nova"))]
pub type G1 = [u8; 32];
#[cfg(not(feature = "nova"))]
pub type G2 = [u8; 32];
#[cfg(not(feature = "nova"))]
pub type Scalar = [u8; 32];
#[cfg(not(feature = "nova"))]
pub type Scalar2 = [u8; 32];

/// Curve type selection for Nova proving
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CurveType {
    /// Pallas curve (recommended for Nova)
    #[default]
    Pallas,
    /// Vesta curve (secondary curve in Nova)
    Vesta,
    /// BN254 curve (for Ethereum compatibility)
    Bn254,
    /// Grumpkin curve (for recursive SNARKs)
    Grumpkin,
}

impl CurveType {
    /// Get the curve's scalar field size in bits
    pub fn scalar_bits(&self) -> usize {
        match self {
            Self::Pallas => 255,
            Self::Vesta => 255,
            Self::Bn254 => 254,
            Self::Grumpkin => 254,
        }
    }

    /// Get the curve's base field size in bits
    pub fn base_bits(&self) -> usize {
        match self {
            Self::Pallas => 255,
            Self::Vesta => 255,
            Self::Bn254 => 254,
            Self::Grumpkin => 254,
        }
    }
}

/// Nova-specific error types
#[derive(Error, Debug, Clone)]
pub enum NovaError {
    /// R1CS constraint system error
    #[error("R1CS error: {0}")]
    R1CSError(String),

    /// Folding operation failed
    #[error("Folding error: {0}")]
    FoldingError(String),

    /// Invalid circuit configuration
    #[error("Invalid circuit: {0}")]
    InvalidCircuit(String),

    /// Witness generation failed
    #[error("Witness generation error: {0}")]
    WitnessError(String),

    /// Proof verification failed
    #[error("Verification failed: {0}")]
    VerificationError(String),

    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Feature not enabled
    #[error("Nova feature not enabled")]
    FeatureNotEnabled,

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Step function error
    #[error("Step function error: {0}")]
    StepFunctionError(String),

    /// Public parameter generation error
    #[error("Public parameter error: {0}")]
    PublicParameterError(String),

    /// IVC proof error
    #[error("IVC proof error: {0}")]
    IVCProofError(String),

    /// Accumulator error
    #[error("Accumulator error: {0}")]
    AccumulatorError(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid state
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// Hardware/GPU acceleration error
    #[error("Hardware error: {0}")]
    HardwareError(String),
}

/// Result type for Nova operations
pub type NovaResult<T> = Result<T, NovaError>;

/// Nova proof size estimation
#[derive(Debug, Clone, Copy)]
pub struct ProofSizeEstimate {
    /// Size of uncompressed proof in bytes
    pub uncompressed_bytes: usize,
    /// Size of compressed proof in bytes
    pub compressed_bytes: usize,
    /// Number of group elements
    pub group_elements: usize,
    /// Number of field elements
    pub field_elements: usize,
}

impl ProofSizeEstimate {
    /// Estimate proof size for given number of steps
    pub fn for_steps(num_steps: usize) -> Self {
        // Nova proof size is roughly O(log n) in the number of steps
        // These are approximations based on Nova paper benchmarks
        let log_steps = (num_steps as f64).log2().ceil() as usize;
        
        Self {
            // Base proof ~10KB + ~500 bytes per log step
            uncompressed_bytes: 10_000 + (log_steps * 500),
            // Compression typically achieves 60-70% of original
            compressed_bytes: 6_500 + (log_steps * 350),
            // 2 group elements per step of IVC recursion
            group_elements: 2 * log_steps + 4,
            // ~10 field elements base + 2 per log step
            field_elements: 10 + (2 * log_steps),
        }
    }
}

/// Security level configuration for Nova
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NovaSecurityLevel {
    /// 128-bit security (recommended for most applications)
    Bit128,
    /// 192-bit security (high security applications)
    Bit192,
    /// 256-bit security (maximum security)
    Bit256,
}

impl Default for NovaSecurityLevel {
    fn default() -> Self {
        Self::Bit128
    }
}

impl NovaSecurityLevel {
    /// Get the number of rounds for this security level
    pub fn rounds(&self) -> usize {
        match self {
            Self::Bit128 => 24,
            Self::Bit192 => 36,
            Self::Bit256 => 48,
        }
    }

    /// Get the constraint system size multiplier
    pub fn constraint_multiplier(&self) -> usize {
        match self {
            Self::Bit128 => 1,
            Self::Bit192 => 2,
            Self::Bit256 => 3,
        }
    }
}

/// Circuit configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitParams {
    /// Maximum number of constraints
    pub max_constraints: usize,
    /// Maximum number of variables
    pub max_variables: usize,
    /// Maximum number of public inputs
    pub max_public_inputs: usize,
    /// Security level
    pub security_level: NovaSecurityLevel,
}

impl Default for CircuitParams {
    fn default() -> Self {
        Self {
            max_constraints: 1_000_000,
            max_variables: 1_000_000,
            max_public_inputs: 100,
            security_level: NovaSecurityLevel::Bit128,
        }
    }
}

/// Metrics for Nova proof generation and verification
#[derive(Debug, Clone, Default)]
pub struct NovaMetrics {
    /// Number of folding steps completed
    pub folding_steps: usize,
    /// Number of steps folded
    pub steps_folded: usize,
    /// Total constraints in the circuit
    pub total_constraints: usize,
    /// Time spent generating proof (microseconds)
    pub proof_generation_us: u64,
    /// Proving time in milliseconds
    pub proving_time_ms: u64,
    /// Time spent verifying proof (microseconds)
    pub verification_us: u64,
    /// Memory usage in bytes
    pub memory_bytes: usize,
    /// Number of R1CS instances folded
    pub instances_folded: usize,
}

impl fmt::Display for NovaMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "NovaMetrics {{ steps: {}, constraints: {}, proof_time: {}μs, verify_time: {}μs, memory: {} bytes }}",
            self.folding_steps,
            self.total_constraints,
            self.proof_generation_us,
            self.verification_us,
            self.memory_bytes
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_size_estimate() {
        let estimate = ProofSizeEstimate::for_steps(100);
        assert!(estimate.uncompressed_bytes > 0);
        assert!(estimate.compressed_bytes < estimate.uncompressed_bytes);
        
        // Larger step count should have larger proof
        let large_estimate = ProofSizeEstimate::for_steps(10_000);
        assert!(large_estimate.uncompressed_bytes > estimate.uncompressed_bytes);
    }

    #[test]
    fn test_security_levels() {
        assert_eq!(NovaSecurityLevel::Bit128.rounds(), 24);
        assert_eq!(NovaSecurityLevel::Bit256.rounds(), 48);
    }

    #[test]
    fn test_nova_error_display() {
        let err = NovaError::R1CSError("constraint mismatch".to_string());
        assert!(err.to_string().contains("R1CS error"));
    }
}
