// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Nova Folding Scheme Integration
//!
//! This module provides integration with the Nova folding scheme for
//! Incrementally Verifiable Computation (IVC) and Proof-Carrying Data (PCD).
//!
//! # Overview
//!
//! Nova is a folding scheme that enables efficient incremental verification
//! of computations. Instead of generating a new proof for each step, Nova
//! "folds" multiple computation steps into a single proof that grows only
//! logarithmically with the number of steps.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    Nova Folding Architecture                        │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
//! │  │   Step 1    │    │   Step 2    │    │   Step N    │            │
//! │  │  Circuit    │───▶│  Circuit    │───▶│  Circuit    │            │
//! │  └─────────────┘    └─────────────┘    └─────────────┘            │
//! │         │                  │                  │                   │
//! │         ▼                  ▼                  ▼                   │
//! │  ┌─────────────────────────────────────────────────────┐          │
//! │  │              R1CS Conversion Layer                  │          │
//! │  │  (Statement/Witness → R1CS Constraint System)       │          │
//! │  └─────────────────────────────────────────────────────┘          │
//! │                          │                                        │
//! │                          ▼                                        │
//! │  ┌─────────────────────────────────────────────────────┐          │
//! │  │              Nova Folding Engine                    │          │
//! │  │  (IVC accumulator + Folding proof)                  │          │
//! │  └─────────────────────────────────────────────────────┘          │
//! │                          │                                        │
//! │                          ▼                                        │
//! │  ┌─────────────────────────────────────────────────────┐          │
//! │  │           Compressed IVC Proof (SNARK)              │          │
//! │  │  (Final verification in constant time)              │          │
//! │  └─────────────────────────────────────────────────────┘          │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Features
//!
//! - **IVC (Incrementally Verifiable Computation)**: Prove long-running
//!   computations step-by-step with constant-time verification
//! - **PCD (Proof-Carrying Data)**: Compose proofs from different sources
//! - **Efficient Recursion**: O(log n) proof size growth
//! - **Pasta Curves**: Uses Pallas/Vesta cycle for efficient recursion
//!
//! # Example
//!
//! ```rust,ignore
//! use nexuszero_crypto::proof::nova::{NovaProver, NovaConfig, StepCircuit};
//!
//! // Define a step circuit
//! struct MyStepCircuit { /* ... */ }
//!
//! impl StepCircuit for MyStepCircuit {
//!     // Implementation
//! }
//!
//! // Create Nova prover
//! let config = NovaConfig::default();
//! let prover = NovaProver::new(config)?;
//!
//! // Generate IVC proof over multiple steps
//! let ivc_proof = prover.prove_ivc(&circuit, num_steps)?;
//!
//! // Verify in constant time
//! let is_valid = prover.verify_ivc(&ivc_proof)?;
//! ```

#[cfg(feature = "nova")]
pub mod r1cs;
#[cfg(feature = "nova")]
pub mod folding;
#[cfg(feature = "nova")]
pub mod prover;
#[cfg(feature = "nova")]
pub mod circuits;
#[cfg(feature = "nova")]
pub mod types;
#[cfg(feature = "nova")]
pub mod recursive;
#[cfg(feature = "nova")]
pub mod gpu;
#[cfg(feature = "nova")]
pub mod integration;

#[cfg(feature = "nova")]
pub use r1cs::{R1CSConverter, R1CSConstraintSystem, R1CSInstance, R1CSWitness, R1CSVariable, LinearCombination, R1CSConstraint};
#[cfg(feature = "nova")]
pub use folding::{FoldingEngine, FoldedInstance, FoldingProof, FoldedWitness, FoldingConfig};
#[cfg(feature = "nova")]
pub use prover::{NovaProver, NovaConfig, NovaProof, IVCProof, NovaPublicParams, CompressionLevel};
#[cfg(feature = "nova")]
pub use circuits::{StepCircuit, TrivialCircuit, MinRootCircuit, HashChainCircuit, MerkleUpdateCircuit, CircuitMetadata};
#[cfg(feature = "nova")]
pub use types::{NovaError, NovaResult, NovaSecurityLevel, CurveType, NovaMetrics, ProofSizeEstimate, CircuitParams};
#[cfg(feature = "nova")]
pub use recursive::{RecursiveProver, RecursiveConfig, RecursiveProof, IVCChain, RecursiveStep, RecursiveVerificationResult};
#[cfg(feature = "nova")]
pub use gpu::{NovaGPU, GPUConfig, GPUMetrics, GPUAccelerationManager, ScalarPoint, NTTDomain, MSMResult, NTTResult, CommitmentResult};
#[cfg(feature = "nova")]
pub use integration::{
    NovaSystem, NovaSystemConfig, ProofRequest, ProofResult, CircuitType, HashChainType,
    VerificationRequest, VerificationResult, VerificationStatus,
    ProofTiming, ProofMetrics, SystemMetrics, BatchProver, StreamProver,
};

/// Module version for compatibility checking
pub const NOVA_MODULE_VERSION: &str = "0.1.0";

/// Check if Nova feature is enabled
pub fn is_nova_enabled() -> bool {
    cfg!(feature = "nova")
}

#[cfg(not(feature = "nova"))]
pub mod stub {
    //! Stub implementations when Nova feature is disabled
    
    use crate::{CryptoError, CryptoResult};
    
    /// Stub NovaProver that returns errors when feature is disabled
    pub struct NovaProver;
    
    impl NovaProver {
        pub fn new(_config: ()) -> CryptoResult<Self> {
            Err(CryptoError::NotImplemented(
                "Nova feature is not enabled. Compile with --features nova".to_string()
            ))
        }
    }
}

#[cfg(not(feature = "nova"))]
pub use stub::*;
