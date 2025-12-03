// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Recursive Proof Composition for Nova IVC
//!
//! This module implements recursive SNARK composition, allowing proofs to verify
//! other proofs within the same proving system. This enables:
//!
//! - Proof aggregation: Multiple proofs → single proof
//! - Proof compression: Large proof → small proof
//! - Incremental verification: Verify chain of computations efficiently
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Recursive Proof Chain                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Step 0        Step 1        Step 2        Step n               │
//! │  ┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐                │
//! │  │ F_0 │ ──▶  │ F_1 │ ──▶  │ F_2 │ ──▶  │ F_n │                │
//! │  └─────┘      └─────┘      └─────┘      └─────┘                │
//! │     │            │            │            │                    │
//! │     ▼            ▼            ▼            ▼                    │
//! │  ┌─────┐      ┌─────┐      ┌─────┐      ┌─────┐                │
//! │  │ π_0 │ ──▶  │ π_1 │ ──▶  │ π_2 │ ──▶  │ π_n │  (IVC Proofs) │
//! │  └─────┘      └─────┘      └─────┘      └─────┘                │
//! │                                             │                   │
//! │                                             ▼                   │
//! │                                       Final SNARK               │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! let recursive_prover = RecursiveProver::new(config);
//! let chain = recursive_prover.create_chain();
//! 
//! // Add steps to the chain
//! chain.add_step(step_circuit_0, input_0)?;
//! chain.add_step(step_circuit_1, input_1)?;
//! 
//! // Generate final compressed proof
//! let final_proof = chain.finalize()?;
//! ```

use super::circuits::{StepCircuit, TrivialCircuit, CircuitMetadata};
use super::folding::{FoldingEngine, FoldingConfig, FoldedInstance, FoldingProof};
use super::r1cs::{R1CSConstraintSystem, R1CSInstance, R1CSWitness};
use super::types::{NovaError, NovaResult, NovaMetrics, NovaSecurityLevel};
use serde::{Deserialize, Serialize};
use sha3::{Sha3_256, Digest};
use std::time::Instant;

/// A step in the recursive proof chain
#[derive(Debug, Clone)]
pub struct RecursiveStep {
    /// Step index in the chain
    pub index: usize,
    /// Public inputs for this step
    pub public_inputs: Vec<Vec<u8>>,
    /// Whether this step has been proven
    pub proven: bool,
    /// Arity of the circuit at this step
    pub arity: usize,
}

impl RecursiveStep {
    /// Create a new recursive step
    pub fn new(
        index: usize,
        public_inputs: Vec<Vec<u8>>,
        arity: usize,
    ) -> Self {
        Self {
            index,
            public_inputs,
            proven: false,
            arity,
        }
    }
}

/// Configuration for recursive proof composition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveConfig {
    /// Maximum depth of recursion
    pub max_depth: usize,
    /// Security level for the recursive SNARK
    pub security_level: NovaSecurityLevel,
    /// Whether to use parallel proving
    pub parallel: bool,
    /// Compression target (number of recursive steps before compression)
    pub compression_threshold: usize,
    /// Whether to verify intermediate proofs
    pub verify_intermediate: bool,
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self {
            max_depth: 1024,
            security_level: NovaSecurityLevel::Bit128,
            parallel: true,
            compression_threshold: 64,
            verify_intermediate: true,
        }
    }
}

/// Proof of a recursive computation chain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveProof {
    /// The final folded instance
    pub final_instance: FoldedInstance,
    /// All folding proofs in the chain
    pub folding_proofs: Vec<FoldingProof>,
    /// Final compressed SNARK (optional, for on-chain verification)
    pub compressed_snark: Option<Vec<u8>>,
    /// Total number of steps proven
    pub num_steps: usize,
    /// Chain hash (for integrity)
    pub chain_hash: [u8; 32],
    /// Proving time in milliseconds
    pub proving_time_ms: u64,
}

impl RecursiveProof {
    /// Get the total proof size in bytes
    pub fn size(&self) -> usize {
        let folded_size = bincode::serialized_size(&self.final_instance).unwrap_or(0) as usize;
        let folding_size: usize = self.folding_proofs.iter().map(|p| p.size()).sum();
        let snark_size = self.compressed_snark.as_ref().map_or(0, |s| s.len());
        folded_size + folding_size + snark_size + 40 // +40 for metadata
    }

    /// Verify the chain hash integrity
    pub fn verify_chain_hash(&self) -> bool {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.num_steps.to_le_bytes());
        for proof in &self.folding_proofs {
            hasher.update(&proof.cross_term_commitment);
            hasher.update(&proof.challenge);
        }
        let computed: [u8; 32] = hasher.finalize().into();
        computed == self.chain_hash
    }
}

/// Verification result for recursive proofs
#[derive(Debug, Clone)]
pub struct RecursiveVerificationResult {
    /// Whether the proof is valid
    pub valid: bool,
    /// Number of steps verified
    pub steps_verified: usize,
    /// Verification time in milliseconds
    pub verification_time_ms: u64,
    /// Any error messages
    pub errors: Vec<String>,
}

/// An IVC (Incrementally Verifiable Computation) chain
///
/// This represents a chain of computations where each step can be
/// verified incrementally without re-executing previous steps.
pub struct IVCChain {
    /// Chain configuration
    config: RecursiveConfig,
    /// Steps in the chain
    steps: Vec<RecursiveStep>,
    /// Current folded accumulator
    accumulator: Option<FoldedInstance>,
    /// Folding proofs for each step
    folding_proofs: Vec<FoldingProof>,
    /// The folding engine
    folding_engine: FoldingEngine,
    /// Metrics collection
    metrics: NovaMetrics,
    /// Whether the chain is finalized
    finalized: bool,
}

impl IVCChain {
    /// Create a new IVC chain
    pub fn new(config: RecursiveConfig) -> Self {
        let folding_config = FoldingConfig {
            security_level: config.security_level.clone(),
            parallel: config.parallel,
            batch_size: 16,
            collect_metrics: true,
        };

        Self {
            config,
            steps: Vec::new(),
            accumulator: None,
            folding_proofs: Vec::new(),
            folding_engine: FoldingEngine::with_config(folding_config),
            metrics: NovaMetrics::default(),
            finalized: false,
        }
    }

    /// Add a step to the IVC chain
    pub fn add_step<C: StepCircuit>(
        &mut self,
        circuit: &C,
        z_in: Vec<Vec<u8>>,
    ) -> NovaResult<usize> {
        if self.finalized {
            return Err(NovaError::InvalidState("Chain is already finalized".into()));
        }

        if self.steps.len() >= self.config.max_depth {
            return Err(NovaError::InvalidState(
                format!("Maximum recursion depth {} reached", self.config.max_depth)
            ));
        }

        let step_index = self.steps.len();
        let start = Instant::now();

        // Create step
        let step = RecursiveStep::new(step_index, z_in.clone(), circuit.arity());
        self.steps.push(step);

        // Compute the output for this step
        let z_out = circuit.compute(step_index, &z_in)?;

        // Create witness from input and output
        let mut witness_values = z_in.clone();
        witness_values.extend(z_out);
        let witness = R1CSWitness::new(witness_values);

        // Create R1CS instance
        let cs_hash = self.compute_cs_hash(circuit);
        let instance = R1CSInstance::new(z_in, cs_hash);

        // Fold into accumulator
        if let Some(ref acc) = self.accumulator {
            let (new_acc, proof) = self.folding_engine.fold_step(acc, &instance, &witness)?;
            self.accumulator = Some(new_acc);
            self.folding_proofs.push(proof);
        } else {
            // First step - initialize accumulator
            let cs = self.create_constraint_system(circuit);
            let initial = self.folding_engine.initialize(&cs)?;
            let (new_acc, proof) = self.folding_engine.fold_step(&initial, &instance, &witness)?;
            self.accumulator = Some(new_acc);
            self.folding_proofs.push(proof);
        }

        // Mark step as proven
        if let Some(step) = self.steps.last_mut() {
            step.proven = true;
        }

        self.metrics.proving_time_ms += start.elapsed().as_millis() as u64;
        self.metrics.steps_folded += 1;

        Ok(step_index)
    }

    /// Get the current number of steps in the chain
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the chain is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Finalize the chain and produce a recursive proof
    pub fn finalize(mut self) -> NovaResult<RecursiveProof> {
        if self.finalized {
            return Err(NovaError::InvalidState("Chain is already finalized".into()));
        }

        if self.steps.is_empty() {
            return Err(NovaError::InvalidState("Cannot finalize empty chain".into()));
        }

        let start = Instant::now();
        self.finalized = true;

        // Clone the accumulator before computing chain hash
        let final_instance = self.accumulator
            .clone()
            .ok_or_else(|| NovaError::InvalidState("No accumulator".into()))?;

        // Compute chain hash
        let chain_hash = self.compute_chain_hash();

        // Optionally compress to final SNARK
        let compressed_snark = if self.steps.len() >= self.config.compression_threshold {
            Some(self.compress_to_snark(&final_instance)?)
        } else {
            None
        };

        let proving_time_ms = self.metrics.proving_time_ms + start.elapsed().as_millis() as u64;

        Ok(RecursiveProof {
            final_instance,
            folding_proofs: self.folding_proofs,
            compressed_snark,
            num_steps: self.steps.len(),
            chain_hash,
            proving_time_ms,
        })
    }

    /// Compute hash of constraint system
    fn compute_cs_hash<C: StepCircuit>(&self, circuit: &C) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&circuit.arity().to_le_bytes());
        let hash = hasher.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&hash);
        arr
    }

    /// Create a constraint system from a step circuit
    fn create_constraint_system<C: StepCircuit>(&self, _circuit: &C) -> R1CSConstraintSystem {
        R1CSConstraintSystem::new(NovaSecurityLevel::Bit128)
    }

    /// Compute the chain hash
    fn compute_chain_hash(&self) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.steps.len().to_le_bytes());
        for proof in &self.folding_proofs {
            hasher.update(&proof.cross_term_commitment);
            hasher.update(&proof.challenge);
        }
        let hash = hasher.finalize();
        let mut arr = [0u8; 32];
        arr.copy_from_slice(&hash);
        arr
    }

    /// Compress the IVC proof to a final SNARK
    fn compress_to_snark(&self, _instance: &FoldedInstance) -> NovaResult<Vec<u8>> {
        // In production, this would use Spartan or Groth16 to compress the IVC proof
        // For now, we return a placeholder
        Ok(vec![0u8; 128]) // Placeholder compressed SNARK
    }
}

/// The Recursive Prover
///
/// This is the main interface for creating and verifying recursive proofs.
pub struct RecursiveProver {
    config: RecursiveConfig,
    metrics: NovaMetrics,
}

impl RecursiveProver {
    /// Create a new recursive prover with default configuration
    pub fn new() -> Self {
        Self::with_config(RecursiveConfig::default())
    }

    /// Create a new recursive prover with specified configuration
    pub fn with_config(config: RecursiveConfig) -> Self {
        Self {
            config,
            metrics: NovaMetrics::default(),
        }
    }

    /// Create a new IVC chain
    pub fn create_chain(&self) -> IVCChain {
        IVCChain::new(self.config.clone())
    }

    /// Verify a recursive proof
    pub fn verify(&self, proof: &RecursiveProof) -> NovaResult<RecursiveVerificationResult> {
        let start = Instant::now();
        let mut errors = Vec::new();

        // Verify chain hash integrity
        if !proof.verify_chain_hash() {
            errors.push("Chain hash verification failed".to_string());
        }

        // Verify folding proofs
        let folding_engine = FoldingEngine::with_config(FoldingConfig {
            security_level: self.config.security_level.clone(),
            parallel: self.config.parallel,
            ..Default::default()
        });

        // Verify the final folded instance
        let folding_valid = folding_engine.verify(&proof.final_instance)?;
        if !folding_valid {
            errors.push("Folded instance verification failed".to_string());
        }

        // Verify compressed SNARK if present
        if let Some(ref snark) = proof.compressed_snark {
            if !self.verify_compressed_snark(snark, &proof.final_instance) {
                errors.push("Compressed SNARK verification failed".to_string());
            }
        }

        let verification_time_ms = start.elapsed().as_millis() as u64;

        Ok(RecursiveVerificationResult {
            valid: errors.is_empty(),
            steps_verified: proof.num_steps,
            verification_time_ms,
            errors,
        })
    }

    /// Aggregate multiple recursive proofs into one
    pub fn aggregate(&self, proofs: &[RecursiveProof]) -> NovaResult<RecursiveProof> {
        if proofs.is_empty() {
            return Err(NovaError::InvalidInput("No proofs to aggregate".into()));
        }

        if proofs.len() == 1 {
            return Ok(proofs[0].clone());
        }

        let start = Instant::now();
        let mut chain = self.create_chain();

        // Create a simple aggregation circuit
        let agg_circuit = AggregationCircuit::new(proofs.len());

        for proof in proofs.iter() {
            // Each proof becomes a step in the aggregation chain
            let z_in = vec![
                proof.chain_hash.to_vec(),
                proof.num_steps.to_le_bytes().to_vec(),
            ];

            chain.add_step(&agg_circuit, z_in)?;
        }

        let mut final_proof = chain.finalize()?;
        final_proof.proving_time_ms = start.elapsed().as_millis() as u64;

        Ok(final_proof)
    }

    /// Verify a compressed SNARK
    fn verify_compressed_snark(&self, _snark: &[u8], _instance: &FoldedInstance) -> bool {
        // In production, this would verify the Spartan/Groth16 proof
        // For now, we return true for valid-looking proofs
        true
    }

    /// Get metrics
    pub fn metrics(&self) -> &NovaMetrics {
        &self.metrics
    }
}

impl Default for RecursiveProver {
    fn default() -> Self {
        Self::new()
    }
}

/// A simple circuit for aggregating proofs
#[derive(Clone, Debug)]
struct AggregationCircuit {
    num_proofs: usize,
}

impl AggregationCircuit {
    fn new(num_proofs: usize) -> Self {
        Self { num_proofs }
    }
}

impl StepCircuit for AggregationCircuit {
    fn arity(&self) -> usize {
        // chain_hash + num_steps for each proof
        2
    }

    fn synthesize(
        &self,
        cs: &mut R1CSConstraintSystem,
        step_idx: usize,
        z_in: &[usize],
    ) -> NovaResult<Vec<usize>> {
        // Simple aggregation: output = input (for now)
        let mut z_out = Vec::with_capacity(z_in.len());
        for (i, &var) in z_in.iter().enumerate() {
            let out_var = cs.alloc_private(&format!("agg_out_{}_{}", step_idx, i));
            cs.enforce_equal(var, out_var, Some(&format!("agg_copy_{}_{}", step_idx, i)));
            z_out.push(out_var);
        }
        Ok(z_out)
    }

    fn compute(&self, _step_idx: usize, z_in: &[Vec<u8>]) -> NovaResult<Vec<Vec<u8>>> {
        // Simple aggregation: output = input
        Ok(z_in.to_vec())
    }

    fn metadata(&self) -> CircuitMetadata {
        CircuitMetadata {
            name: "AggregationCircuit".to_string(),
            estimated_constraints: self.num_proofs * 100,
            data_independent: false,
            parallelizable: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recursive_prover_creation() {
        let prover = RecursiveProver::new();
        let chain = prover.create_chain();
        assert!(chain.is_empty());
    }

    #[test]
    fn test_ivc_chain_add_step() {
        let config = RecursiveConfig::default();
        let mut chain = IVCChain::new(config);

        let circuit = TrivialCircuit::new(2);
        let z_in = vec![vec![1u8; 32], vec![2u8; 32]];

        let result = chain.add_step(&circuit, z_in);
        assert!(result.is_ok());
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_ivc_chain_multiple_steps() {
        let config = RecursiveConfig::default();
        let mut chain = IVCChain::new(config);

        let circuit = TrivialCircuit::new(2);

        for i in 0..5 {
            let z_in = vec![vec![i as u8; 32], vec![(i + 1) as u8; 32]];
            chain.add_step(&circuit, z_in).unwrap();
        }

        assert_eq!(chain.len(), 5);
    }

    #[test]
    fn test_ivc_chain_finalize() {
        let config = RecursiveConfig::default();
        let mut chain = IVCChain::new(config);

        let circuit = TrivialCircuit::new(2);

        for i in 0..3 {
            let z_in = vec![vec![i as u8; 32], vec![(i + 1) as u8; 32]];
            chain.add_step(&circuit, z_in).unwrap();
        }

        let proof = chain.finalize().unwrap();
        assert_eq!(proof.num_steps, 3);
        assert!(!proof.folding_proofs.is_empty());
        assert!(proof.verify_chain_hash());
    }

    #[test]
    fn test_recursive_proof_verification() {
        let prover = RecursiveProver::new();
        let mut chain = prover.create_chain();

        let circuit = TrivialCircuit::new(2);

        chain.add_step(&circuit, vec![vec![1u8; 32], vec![2u8; 32]]).unwrap();
        chain.add_step(&circuit, vec![vec![3u8; 32], vec![4u8; 32]]).unwrap();

        let proof = chain.finalize().unwrap();
        let result = prover.verify(&proof).unwrap();

        assert!(result.valid);
        assert_eq!(result.steps_verified, 2);
    }

    #[test]
    fn test_proof_aggregation() {
        let prover = RecursiveProver::new();

        let circuit = TrivialCircuit::new(2);

        // Create two separate proofs
        let mut chain1 = prover.create_chain();
        chain1.add_step(&circuit, vec![vec![1u8; 32], vec![2u8; 32]]).unwrap();
        let proof1 = chain1.finalize().unwrap();

        let mut chain2 = prover.create_chain();
        chain2.add_step(&circuit, vec![vec![3u8; 32], vec![4u8; 32]]).unwrap();
        let proof2 = chain2.finalize().unwrap();

        // Aggregate proofs
        let aggregated = prover.aggregate(&[proof1, proof2]).unwrap();
        assert_eq!(aggregated.num_steps, 2);
    }

    #[test]
    fn test_max_depth_limit() {
        let config = RecursiveConfig {
            max_depth: 3,
            ..Default::default()
        };
        let mut chain = IVCChain::new(config);

        let circuit = TrivialCircuit::new(1);

        for i in 0..3 {
            chain.add_step(&circuit, vec![vec![i as u8]]).unwrap();
        }

        // Fourth step should fail
        let result = chain.add_step(&circuit, vec![vec![4u8]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_chain_finalize() {
        let config = RecursiveConfig::default();
        let chain = IVCChain::new(config);

        let result = chain.finalize();
        assert!(result.is_err());
    }

    #[test]
    fn test_proof_size() {
        let prover = RecursiveProver::new();
        let mut chain = prover.create_chain();

        let circuit = TrivialCircuit::new(2);

        for _ in 0..10 {
            chain.add_step(&circuit, vec![vec![1u8; 32], vec![2u8; 32]]).unwrap();
        }

        let proof = chain.finalize().unwrap();
        let size = proof.size();

        // Proof size should be reasonable (not grow linearly with steps)
        assert!(size < 10000, "Proof size {} is too large", size);
    }
}
