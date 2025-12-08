//! Zero-knowledge proof system components
#![allow(clippy::module_inception)]

//!
//! This module provides Statement, Witness, and Proof structures
//! for constructing zero-knowledge proofs.

pub mod bulletproofs;
pub mod proof;
pub mod schnorr;
pub mod statement;
pub mod witness;
pub mod witness_manager;
pub mod plugins;

// New modular architecture modules
pub mod circuit;
pub mod prover;
pub mod verifier;
pub mod witness_dsl;
pub mod hardware_acceleration;
pub mod distributed_verification;
pub mod performance_optimization;

// Nova folding scheme (feature-gated)
#[cfg(feature = "nova")]
pub mod nova;

// Re-export main types
pub use proof::{Proof, ProofMetadata};
pub use statement::{Statement, StatementBuilder, StatementType};
pub use witness::{Witness, WitnessType};
pub use witness_manager::{WitnessManager, DefaultWitnessManager, WitnessMetadata, CachedWitness, ValidationConstraints, WitnessGenerationConfig, TransformationResult, CacheStats};
pub use bulletproofs::{BulletproofRangeProof, prove_range, verify_range};
pub use schnorr::{
    SchnorrPrivateKey, SchnorrPublicKey, SchnorrSignature, SchnorrParams,
    schnorr_keygen, schnorr_sign, schnorr_verify
};
pub use plugins::{
    ProofPluginEnum, ProofRegistry, ProofType, SetupParams, VerificationKey, ProverKey,
    CircuitComponent, CircuitInfo, PluginInfo,
    SchnorrPlugin, BulletproofsPlugin, Groth16Plugin, PlonkPlugin
};

// Re-export new modular types
pub use circuit::{Circuit, CircuitEngine, Variable, Constraint};
pub use prover::{Prover, ProverConfig, ProverCapabilities, ProverRegistry, ProofStrategy};
pub use verifier::{Verifier, VerifierConfig, VerifierCapabilities, VerifierRegistry, VerificationStrategy, VerificationRequirements};
pub use witness_dsl::{WitnessGenerator, WitnessGenerationPlan, WitnessBuilder, WitnessGeneratorRegistry, WitnessStrategy};
#[cfg(feature = "hardware-acceleration")]
pub use hardware_acceleration::{GPUVerifier, HardwareProver, HardwareType};
#[cfg(feature = "tpu")]
pub use hardware_acceleration::TPUVerifier;
pub use distributed_verification::{DistributedVerifier, ByzantineDistributedVerifier, VerificationNode, DistributedConfig, LoadBalancingStrategy};
pub use performance_optimization::{ParallelBatchProver, OptimizedBatchVerifier, AdaptiveProver, PerformanceMonitor, PerformanceMetrics, PerformanceThresholds, PerformanceAlert};

// Nova re-exports when feature enabled
#[cfg(feature = "nova")]
pub use nova::{
    NovaProver, NovaConfig, NovaProof, IVCProof, 
    StepCircuit, TrivialCircuit, MinRootCircuit,
    R1CSConverter, R1CSConstraintSystem, R1CSInstance, R1CSWitness,
    FoldingEngine, FoldedInstance, FoldingProof,
    NovaError, NovaResult,
};
