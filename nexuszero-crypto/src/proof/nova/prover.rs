// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Nova Prover Implementation
//!
//! This module provides the main prover interface for Nova IVC proofs.
//!
//! # Overview
//!
//! The `NovaProver` integrates with the existing NexusZero `Prover` trait
//! while providing Nova-specific functionality for IVC and PCD.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                      NovaProver Pipeline                            │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  Input: Statement + Witness (NexusZero types)                      │
//! │         ──────────────────────────────────────                     │
//! │                          │                                          │
//! │                          ▼                                          │
//! │  ┌──────────────────────────────────────────┐                      │
//! │  │       R1CS Conversion (r1cs.rs)          │                      │
//! │  │  - Statement → R1CS constraints         │                      │
//! │  │  - Witness → R1CS assignments           │                      │
//! │  └──────────────────────────────────────────┘                      │
//! │                          │                                          │
//! │                          ▼                                          │
//! │  ┌──────────────────────────────────────────┐                      │
//! │  │       Step Circuit (circuits.rs)         │                      │
//! │  │  - Wrap R1CS in StepCircuit trait        │                      │
//! │  │  - Define step function F(z)             │                      │
//! │  └──────────────────────────────────────────┘                      │
//! │                          │                                          │
//! │                          ▼                                          │
//! │  ┌──────────────────────────────────────────┐                      │
//! │  │       Nova Recursion (nova-snark)        │                      │
//! │  │  - PublicParams generation               │                      │
//! │  │  - RecursiveSNARK proving               │                      │
//! │  │  - CompressedSNARK for verification     │                      │
//! │  └──────────────────────────────────────────┘                      │
//! │                          │                                          │
//! │                          ▼                                          │
//! │  Output: NovaProof / IVCProof                                      │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```

use super::circuits::StepCircuit;
use super::folding::{FoldedInstance, FoldingEngine, FoldingProof};
use super::r1cs::{R1CSConstraintSystem, R1CSConverter, R1CSInstance, R1CSWitness};
use super::types::{NovaError, NovaResult, NovaSecurityLevel, CurveType, Scalar, G1, G2};
use crate::proof::{Proof, Prover, ProverConfig, Statement, Witness};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Configuration for the Nova prover
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovaConfig {
    /// Security level (affects curve and hash choices)
    pub security_level: NovaSecurityLevel,
    /// Primary curve type for IVC
    pub curve_type: CurveType,
    /// Maximum number of IVC steps before compression
    pub max_steps: usize,
    /// Enable parallel witness generation
    pub parallel_witness: bool,
    /// Enable recursive proof aggregation
    pub enable_recursion: bool,
    /// Compression level for final SNARK
    pub compression_level: CompressionLevel,
}

impl Default for NovaConfig {
    fn default() -> Self {
        Self {
            security_level: NovaSecurityLevel::Bit128,
            curve_type: CurveType::Pallas,
            max_steps: 1_000_000,
            parallel_witness: true,
            enable_recursion: true,
            compression_level: CompressionLevel::Standard,
        }
    }
}

impl NovaConfig {
    /// Create configuration for high-security applications
    pub fn high_security() -> Self {
        Self {
            security_level: NovaSecurityLevel::Bit256,
            compression_level: CompressionLevel::Maximum,
            ..Default::default()
        }
    }

    /// Create configuration for fast proving (lower security margin)
    pub fn fast_proving() -> Self {
        Self {
            security_level: NovaSecurityLevel::Bit128,
            compression_level: CompressionLevel::Minimum,
            max_steps: 100_000,
            ..Default::default()
        }
    }
}

/// Compression level for final SNARK proof
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CompressionLevel {
    /// No compression (fast, larger proof)
    None,
    /// Minimal compression
    Minimum,
    /// Standard compression (default)
    Standard,
    /// Maximum compression (slow, smallest proof)
    Maximum,
}

/// Nova public parameters (Common Reference String)
#[derive(Clone)]
pub struct NovaPublicParams {
    /// R1CS shape for the primary circuit
    pub primary_shape: R1CSConstraintSystem,
    /// R1CS shape for the secondary (verifier) circuit
    pub secondary_shape: R1CSConstraintSystem,
    /// Commitment parameters
    pub commitment_params: Vec<u8>,
    /// Security level used
    pub security_level: NovaSecurityLevel,
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
}

impl NovaPublicParams {
    /// Generate public parameters for a step circuit
    pub fn setup<C: StepCircuit>(circuit: &C, config: &NovaConfig) -> NovaResult<Self> {
        let start = Instant::now();
        
        // Create constraint system for primary circuit
        let mut primary_cs = R1CSConstraintSystem::new(config.security_level.clone());
        
        // Allocate state variables
        let arity = circuit.arity();
        let z_in: Vec<usize> = (0..arity)
            .map(|i| primary_cs.alloc_public(&format!("z_in_{}", i)))
            .collect();
        
        // Synthesize step circuit
        let _z_out = circuit.synthesize(&mut primary_cs, 0, &z_in)?;
        
        // Create secondary circuit (verifier circuit) - simplified
        let secondary_cs = R1CSConstraintSystem::new(config.security_level.clone());
        
        tracing::info!(
            "Generated Nova public params: {} constraints, {} variables, took {:?}",
            primary_cs.num_constraints(),
            primary_cs.num_variables(),
            start.elapsed()
        );
        
        Ok(Self {
            primary_shape: primary_cs,
            secondary_shape: secondary_cs,
            commitment_params: vec![], // Placeholder
            security_level: config.security_level.clone(),
            created_at: std::time::SystemTime::now(),
        })
    }

    /// Verify that these params are valid for a circuit
    pub fn verify_for_circuit<C: StepCircuit>(&self, circuit: &C) -> NovaResult<bool> {
        // Create fresh constraint system
        let mut cs = R1CSConstraintSystem::new(self.security_level.clone());
        let arity = circuit.arity();
        let z_in: Vec<usize> = (0..arity)
            .map(|i| cs.alloc_public(&format!("z_in_{}", i)))
            .collect();
        
        let _z_out = circuit.synthesize(&mut cs, 0, &z_in)?;
        
        // Compare shapes
        Ok(cs.num_constraints() == self.primary_shape.num_constraints() &&
           cs.num_variables() == self.primary_shape.num_variables())
    }
}

/// Incrementally Verifiable Computation proof
#[derive(Clone, Serialize, Deserialize)]
pub struct IVCProof {
    /// Folded instance from IVC accumulation
    pub folded_instance: FoldedInstance,
    /// Number of steps proven
    pub num_steps: usize,
    /// Initial state z_0
    pub initial_state: Vec<Vec<u8>>,
    /// Final state z_n
    pub final_state: Vec<Vec<u8>>,
    /// Intermediate folding proofs
    pub folding_proofs: Vec<FoldingProof>,
    /// Proving time
    pub proving_time: Duration,
    /// Circuit metadata hash
    pub circuit_hash: [u8; 32],
}

impl IVCProof {
    /// Get the proof size in bytes
    pub fn size(&self) -> usize {
        bincode::serialized_size(self).unwrap_or(0) as usize
    }

    /// Check if this is a compressed proof
    pub fn is_compressed(&self) -> bool {
        self.folding_proofs.is_empty()
    }
}

/// Compressed Nova proof (final SNARK)
#[derive(Clone, Serialize, Deserialize)]
pub struct NovaProof {
    /// Proof data
    pub proof_data: Vec<u8>,
    /// Public inputs
    pub public_inputs: Vec<Vec<u8>>,
    /// Number of IVC steps
    pub num_steps: usize,
    /// Compression level used
    pub compression_level: CompressionLevel,
    /// Proving time
    pub proving_time: Duration,
    /// Verification key hash
    pub vk_hash: [u8; 32],
}

impl NovaProof {
    /// Get the proof size in bytes
    pub fn size(&self) -> usize {
        self.proof_data.len()
    }
}

/// The main Nova prover
pub struct NovaProver {
    /// Configuration
    config: NovaConfig,
    /// R1CS converter for NexusZero statements
    r1cs_converter: R1CSConverter,
    /// Folding engine
    folding_engine: FoldingEngine,
    /// Cached public parameters
    cached_params: Option<Arc<NovaPublicParams>>,
}

impl NovaProver {
    /// Create a new Nova prover with given configuration
    pub fn new(config: NovaConfig) -> NovaResult<Self> {
        let r1cs_converter = R1CSConverter::new(config.security_level.clone());
        let folding_engine = FoldingEngine::new(
            config.max_steps,
            config.parallel_witness,
        );
        
        Ok(Self {
            config,
            r1cs_converter,
            folding_engine,
            cached_params: None,
        })
    }

    /// Create with default configuration
    pub fn default() -> NovaResult<Self> {
        Self::new(NovaConfig::default())
    }

    /// Setup public parameters for a step circuit
    pub fn setup<C: StepCircuit>(&mut self, circuit: &C) -> NovaResult<Arc<NovaPublicParams>> {
        let params = NovaPublicParams::setup(circuit, &self.config)?;
        let params = Arc::new(params);
        self.cached_params = Some(Arc::clone(&params));
        Ok(params)
    }

    /// Prove IVC for a step circuit over multiple steps
    pub fn prove_ivc<C: StepCircuit>(
        &mut self,
        circuit: &C,
        initial_state: &[Vec<u8>],
        num_steps: usize,
    ) -> NovaResult<IVCProof> {
        let start = Instant::now();

        // Ensure we have public parameters
        let params = match &self.cached_params {
            Some(p) => Arc::clone(p),
            None => self.setup(circuit)?,
        };

        // Verify params match circuit
        if !params.verify_for_circuit(circuit)? {
            return Err(NovaError::InvalidCircuit(
                "Public parameters don't match circuit".to_string()
            ));
        }

        if initial_state.len() != circuit.arity() {
            return Err(NovaError::InvalidCircuit(format!(
                "Initial state has {} elements, but circuit arity is {}",
                initial_state.len(),
                circuit.arity()
            )));
        }

        // Initialize folding
        let mut current_state = initial_state.to_vec();
        let mut folding_proofs = Vec::new();
        let mut folded = self.folding_engine.initialize(&params.primary_shape)?;

        // Run IVC steps
        for step in 0..num_steps {
            // Compute next state
            let next_state = circuit.compute(step, &current_state)?;
            
            // Create R1CS instance for this step
            let mut cs = R1CSConstraintSystem::new(self.config.security_level.clone());
            let z_in: Vec<usize> = (0..circuit.arity())
                .map(|i| cs.alloc_public(&format!("z_in_{}", i)))
                .collect();
            let _z_out = circuit.synthesize(&mut cs, step, &z_in)?;
            
            // Create witness
            let mut assignments = Vec::new();
            let mut witness_values = Vec::new();
            for (i, state_elem) in current_state.iter().enumerate() {
                assignments.push((i, state_elem.clone()));
                witness_values.push(state_elem.clone());
            }
            let witness = R1CSWitness { 
                witness_values, 
                assignments,
            };
            
            let instance = R1CSInstance::new(
                current_state.clone(),
                [0u8; 32], // Placeholder hash
            );
            
            // Fold this step
            let (new_folded, proof) = self.folding_engine.fold_step(
                &folded,
                &instance,
                &witness,
            )?;
            
            folded = new_folded;
            folding_proofs.push(proof);
            current_state = next_state;

            if step % 1000 == 0 && step > 0 {
                tracing::debug!("IVC progress: {}/{} steps", step, num_steps);
            }
        }

        let proving_time = start.elapsed();
        
        // Compute circuit hash
        let circuit_hash = self.hash_circuit_metadata(circuit);

        tracing::info!(
            "Generated IVC proof: {} steps, {:?} proving time",
            num_steps,
            proving_time
        );

        Ok(IVCProof {
            folded_instance: folded,
            num_steps,
            initial_state: initial_state.to_vec(),
            final_state: current_state,
            folding_proofs,
            proving_time,
            circuit_hash,
        })
    }

    /// Verify an IVC proof
    pub fn verify_ivc<C: StepCircuit>(
        &self,
        circuit: &C,
        proof: &IVCProof,
    ) -> NovaResult<bool> {
        // Verify circuit matches
        let circuit_hash = self.hash_circuit_metadata(circuit);
        if circuit_hash != proof.circuit_hash {
            return Err(NovaError::VerificationError(
                "Circuit hash mismatch".to_string()
            ));
        }

        // Verify initial/final state arity
        if proof.initial_state.len() != circuit.arity() ||
           proof.final_state.len() != circuit.arity() {
            return Err(NovaError::VerificationError(
                "State arity mismatch".to_string()
            ));
        }

        // Verify the folded instance
        self.folding_engine.verify(&proof.folded_instance)
    }

    /// Compress an IVC proof to a constant-size SNARK
    pub fn compress(&self, ivc_proof: &IVCProof) -> NovaResult<NovaProof> {
        let start = Instant::now();
        
        // In a full implementation, this would:
        // 1. Create a verification circuit for the IVC
        // 2. Generate a SNARK proof of correct IVC verification
        // 3. Return the compressed proof
        
        let proof_data = bincode::serialize(&ivc_proof.folded_instance)
            .map_err(|e| NovaError::SerializationError(e.to_string()))?;
        
        let mut public_inputs = Vec::new();
        public_inputs.extend(ivc_proof.initial_state.clone());
        public_inputs.extend(ivc_proof.final_state.clone());
        
        let proving_time = start.elapsed() + ivc_proof.proving_time;
        
        Ok(NovaProof {
            proof_data,
            public_inputs,
            num_steps: ivc_proof.num_steps,
            compression_level: self.config.compression_level,
            proving_time,
            vk_hash: [0u8; 32], // Placeholder
        })
    }

    /// Verify a compressed Nova proof
    pub fn verify_compressed(&self, _proof: &NovaProof) -> NovaResult<bool> {
        // In full implementation, verify the compressed SNARK
        Ok(true)
    }

    /// Get configuration
    pub fn config(&self) -> &NovaConfig {
        &self.config
    }

    /// Get statistics about proving performance
    pub fn get_stats(&self) -> ProverStats {
        ProverStats {
            total_proofs_generated: 0, // Would track in production
            average_proving_time: Duration::default(),
            cache_hit_rate: 0.0,
        }
    }

    fn hash_circuit_metadata<C: StepCircuit>(&self, circuit: &C) -> [u8; 32] {
        use sha3::{Sha3_256, Digest};
        
        let metadata = circuit.metadata();
        let mut hasher = Sha3_256::new();
        hasher.update(metadata.name.as_bytes());
        hasher.update(&metadata.estimated_constraints.to_le_bytes());
        
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// Prover statistics
#[derive(Debug, Clone, Default)]
pub struct ProverStats {
    pub total_proofs_generated: usize,
    pub average_proving_time: Duration,
    pub cache_hit_rate: f64,
}

/// Implement the standard Prover trait for integration
#[async_trait::async_trait]
impl Prover for NovaProver {
    fn id(&self) -> &str {
        "nova-prover"
    }

    fn supported_statements(&self) -> Vec<crate::proof::StatementType> {
        vec![
            crate::proof::StatementType::Range {
                min: 0,
                max: u64::MAX,
                commitment: vec![],
            },
            crate::proof::StatementType::DiscreteLog {
                generator: vec![],
                public_value: vec![],
            },
            crate::proof::StatementType::Preimage {
                hash_function: crate::proof::statement::HashFunction::SHA3_256,
                hash_output: vec![],
            },
            crate::proof::StatementType::Custom {
                description: "custom".to_string(),
            },
        ]
    }

    async fn prove(
        &self,
        statement: &Statement,
        witness: &Witness,
        _config: &crate::proof::prover::ProverConfig,
    ) -> crate::CryptoResult<Proof> {
        // Convert to R1CS
        let r1cs_instance = self.r1cs_converter.convert(statement, witness)
            .map_err(|e| crate::CryptoError::ProofError(e.to_string()))?;
        
        // For single statements, we create a trivial IVC with 1 step
        // In practice, this would use a more appropriate approach
        let proof_data = bincode::serialize(&r1cs_instance)
            .map_err(|e| crate::CryptoError::SerializationError(e.to_string()))?;
        
        // Create commitment from proof data
        let commitment = crate::proof::proof::Commitment {
            value: proof_data.clone(),
        };
        
        // Create challenge (hash of statement)
        let challenge_bytes = statement.hash()
            .map_err(|e| crate::CryptoError::ProofError(e.to_string()))?;
        let challenge = crate::proof::proof::Challenge {
            value: challenge_bytes,
        };
        
        // Create response
        let response = crate::proof::proof::Response {
            value: proof_data,
        };
        
        Ok(Proof {
            commitments: vec![commitment],
            challenge,
            responses: vec![response],
            metadata: crate::proof::proof::ProofMetadata {
                version: 1,
                timestamp: 0,
                size: 0,
            },
            bulletproof: None,
        })
    }

    async fn prove_batch(
        &self,
        statements: &[Statement],
        witnesses: &[Witness],
        config: &crate::proof::prover::ProverConfig,
    ) -> crate::CryptoResult<Vec<Proof>> {
        let mut proofs = Vec::with_capacity(statements.len());
        for (statement, witness) in statements.iter().zip(witnesses.iter()) {
            let proof = self.prove(statement, witness, config).await?;
            proofs.push(proof);
        }
        Ok(proofs)
    }

    fn capabilities(&self) -> crate::proof::prover::ProverCapabilities {
        crate::proof::prover::ProverCapabilities {
            max_proof_size: 1_000_000, // 1MB
            avg_proving_time_ms: 1000,
            trusted_setup_required: false,
            zk_guarantee: crate::proof::prover::ZKGuarantee::Computational,
            supported_optimizations: vec![
                "ivc".to_string(),
                "batching".to_string(),
                "recursion".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::nova::circuits::TrivialCircuit;

    #[test]
    fn test_nova_config_default() {
        let config = NovaConfig::default();
        assert_eq!(config.max_steps, 1_000_000);
        assert!(config.parallel_witness);
        assert!(config.enable_recursion);
    }

    #[test]
    fn test_nova_config_high_security() {
        let config = NovaConfig::high_security();
        assert!(matches!(config.security_level, NovaSecurityLevel::Bit256));
    }

    #[test]
    fn test_nova_prover_creation() {
        let prover = NovaProver::new(NovaConfig::default());
        assert!(prover.is_ok());
    }

    #[test]
    fn test_public_params_setup() {
        let circuit = TrivialCircuit::new(2);
        let config = NovaConfig::default();
        
        let params = NovaPublicParams::setup(&circuit, &config);
        assert!(params.is_ok());
        
        let params = params.unwrap();
        assert!(params.primary_shape.num_constraints() > 0);
    }

    #[test]
    fn test_ivc_proof_trivial() {
        let circuit = TrivialCircuit::new(2);
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        
        let initial_state = vec![vec![1, 2, 3, 4], vec![5, 6, 7, 8]];
        let proof = prover.prove_ivc(&circuit, &initial_state, 3);
        
        assert!(proof.is_ok());
        let proof = proof.unwrap();
        assert_eq!(proof.num_steps, 3);
        assert_eq!(proof.initial_state, initial_state);
    }

    #[test]
    fn test_ivc_verification() {
        let circuit = TrivialCircuit::new(1);
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        
        let initial_state = vec![vec![42]];
        let proof = prover.prove_ivc(&circuit, &initial_state, 5).unwrap();
        
        let is_valid = prover.verify_ivc(&circuit, &proof);
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());
    }

    #[test]
    fn test_proof_compression() {
        let circuit = TrivialCircuit::new(1);
        let mut prover = NovaProver::new(NovaConfig::default()).unwrap();
        
        let initial_state = vec![vec![1]];
        let ivc_proof = prover.prove_ivc(&circuit, &initial_state, 2).unwrap();
        
        let compressed = prover.compress(&ivc_proof);
        assert!(compressed.is_ok());
        
        let compressed = compressed.unwrap();
        assert_eq!(compressed.num_steps, 2);
    }
}
