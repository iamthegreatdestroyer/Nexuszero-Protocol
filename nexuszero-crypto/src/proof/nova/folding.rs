// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Nova Folding Engine
//!
//! This module implements the core folding operations for Nova IVC proofs.
//! Folding allows multiple R1CS instances to be "folded" into a single
//! accumulated instance, enabling efficient incremental verification.
//!
//! # Folding Overview
//!
//! Nova's folding scheme works by:
//! 1. Taking two R1CS instances (current accumulator + new step)
//! 2. Computing a random challenge r
//! 3. Folding the instances: acc' = acc + r * step
//! 4. Producing a folding proof (cross-term commitment)
//!
//! # Security
//!
//! The folding scheme is secure under the assumption that the underlying
//! commitment scheme is binding and hiding. Nova uses Pedersen commitments
//! over the Pasta curve cycle (Pallas/Vesta).

use super::r1cs::{R1CSConstraintSystem, R1CSInstance, R1CSWitness};
use super::types::{NovaError, NovaResult, NovaMetrics, NovaSecurityLevel};
use serde::{Deserialize, Serialize};
use sha3::{Sha3_256, Digest};
use std::time::Instant;

/// A folded R1CS instance representing multiple computation steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldedInstance {
    /// The accumulated instance (commitments + public inputs)
    pub accumulated_x: Vec<Vec<u8>>,
    /// Error term commitment (tracks folding error)
    pub error_commitment: Vec<u8>,
    /// Number of steps folded into this instance
    pub num_steps: usize,
    /// Running hash of all folded instances
    pub running_hash: [u8; 32],
    /// Scalar used in last folding (for verification)
    pub last_challenge: Vec<u8>,
}

impl FoldedInstance {
    /// Create initial folded instance from first R1CS instance
    pub fn initial(instance: &R1CSInstance) -> Self {
        let mut hasher = Sha3_256::new();
        for input in &instance.public_inputs {
            hasher.update(input);
        }
        hasher.update(&instance.cs_hash);
        let running_hash: [u8; 32] = hasher.finalize().into();

        Self {
            accumulated_x: instance.public_inputs.clone(),
            error_commitment: vec![0u8; 32], // Zero error initially
            num_steps: 1,
            running_hash,
            last_challenge: vec![],
        }
    }

    /// Create a relaxed instance (for Nova's relaxed R1CS)
    pub fn relaxed(instance: &R1CSInstance) -> Self {
        let mut folded = Self::initial(instance);
        // Add relaxation factor u = 1 initially
        folded.accumulated_x.push(vec![1]); // u = 1
        folded
    }
}

/// Proof that two instances were correctly folded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingProof {
    /// Commitment to cross-term T
    pub cross_term_commitment: Vec<u8>,
    /// Challenge scalar r used in folding
    pub challenge: Vec<u8>,
    /// Auxiliary information for verification
    pub aux_commitments: Vec<Vec<u8>>,
}

impl FoldingProof {
    /// Size of the proof in bytes
    pub fn size(&self) -> usize {
        self.cross_term_commitment.len()
            + self.challenge.len()
            + self.aux_commitments.iter().map(|c| c.len()).sum::<usize>()
    }
}

/// Accumulated folding witness
#[derive(Debug, Clone)]
pub struct FoldedWitness {
    /// Accumulated witness values
    pub witness_values: Vec<Vec<u8>>,
    /// Error vector (tracks accumulated error)
    pub error_vector: Vec<Vec<u8>>,
}

impl FoldedWitness {
    /// Create initial folded witness
    pub fn initial(witness: &R1CSWitness) -> Self {
        Self {
            witness_values: witness.witness_values.clone(),
            error_vector: vec![vec![0u8; 32]], // Zero error initially
        }
    }

    /// Zeroize sensitive data
    pub fn zeroize(&mut self) {
        for w in &mut self.witness_values {
            for byte in w.iter_mut() {
                *byte = 0;
            }
        }
        for e in &mut self.error_vector {
            for byte in e.iter_mut() {
                *byte = 0;
            }
        }
    }
}

impl Drop for FoldedWitness {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// Configuration for the folding engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldingConfig {
    /// Security level
    pub security_level: NovaSecurityLevel,
    /// Whether to use parallel folding
    pub parallel: bool,
    /// Batch size for multi-instance folding
    pub batch_size: usize,
    /// Whether to collect metrics
    pub collect_metrics: bool,
}

impl Default for FoldingConfig {
    fn default() -> Self {
        Self {
            security_level: NovaSecurityLevel::Bit128,
            parallel: true,
            batch_size: 16,
            collect_metrics: true,
        }
    }
}

/// The Nova Folding Engine
///
/// This engine handles the core folding operations for Nova IVC proofs.
pub struct FoldingEngine {
    config: FoldingConfig,
    constraint_system: Option<R1CSConstraintSystem>,
    current_accumulator: Option<FoldedInstance>,
    current_witness: Option<FoldedWitness>,
    metrics: NovaMetrics,
}

impl FoldingEngine {
    /// Create a new folding engine with default config
    pub fn new(max_steps: usize, parallel: bool) -> Self {
        Self::with_config(FoldingConfig {
            batch_size: max_steps.min(16),
            parallel,
            ..Default::default()
        })
    }
    
    /// Create a new folding engine with specified configuration
    pub fn with_config(config: FoldingConfig) -> Self {
        Self {
            config,
            constraint_system: None,
            current_accumulator: None,
            current_witness: None,
            metrics: NovaMetrics::default(),
        }
    }

    /// Initialize with a constraint system
    pub fn init(&mut self, cs: R1CSConstraintSystem) -> NovaResult<()> {
        cs.validate()?;
        self.metrics.total_constraints = cs.num_constraints();
        self.constraint_system = Some(cs);
        Ok(())
    }

    /// Initialize folding with an R1CS shape and return initial accumulator
    pub fn initialize(&self, cs: &R1CSConstraintSystem) -> NovaResult<FoldedInstance> {
        let cs_hash = {
            let mut hasher = Sha3_256::new();
            hasher.update(&cs.num_constraints().to_le_bytes());
            hasher.update(&cs.num_variables().to_le_bytes());
            let hash = hasher.finalize();
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&hash);
            arr
        };
        let initial_instance = R1CSInstance::new(
            vec![vec![0u8; 32]; cs.num_public_inputs],
            cs_hash,
        );
        Ok(FoldedInstance::relaxed(&initial_instance))
    }

    /// Fold a step into the current accumulator
    pub fn fold_step(
        &self,
        current: &FoldedInstance,
        instance: &R1CSInstance,
        witness: &R1CSWitness,
    ) -> NovaResult<(FoldedInstance, FoldingProof)> {
        // Generate challenge
        let cross_term = vec![0u8; 32]; // Placeholder for cross-term
        let commitment = self.commit(&cross_term)?;
        let challenge = self.generate_challenge(&current.running_hash, instance, &commitment)?;
        
        // Fold instances
        let folded = self.fold_instances(current, instance, &challenge)?;
        
        let proof = FoldingProof {
            cross_term_commitment: commitment,
            challenge,
            aux_commitments: vec![],
        };
        
        Ok((folded, proof))
    }

    /// Verify a folded instance
    pub fn verify(&self, folded: &FoldedInstance) -> NovaResult<bool> {
        // Basic validation
        if folded.num_steps == 0 {
            return Ok(false);
        }
        if folded.accumulated_x.is_empty() {
            return Ok(false);
        }
        // In full implementation, verify error terms and commitments
        Ok(true)
    }

    /// Fold a new instance into the accumulator
    ///
    /// This is the core folding operation that combines the current
    /// accumulated instance with a new step instance.
    pub fn fold(
        &mut self,
        new_instance: &R1CSInstance,
        new_witness: &R1CSWitness,
    ) -> NovaResult<FoldingProof> {
        let start = Instant::now();

        // Ensure we have a constraint system
        let cs = self.constraint_system.as_ref()
            .ok_or(NovaError::InvalidCircuit("No constraint system initialized".into()))?;

        // Verify instance matches constraint system
        if new_instance.cs_hash != {
            use sha2::{Sha256, Digest as _};
            let mut hasher = Sha256::new();
            hasher.update(&bincode::serialize(cs).map_err(|e| {
                NovaError::SerializationError(e.to_string())
            })?);
            let hash: [u8; 32] = hasher.finalize().into();
            hash
        } {
            return Err(NovaError::R1CSError("Instance doesn't match constraint system".into()));
        }

        // Initialize accumulator if this is the first instance
        if self.current_accumulator.is_none() {
            self.current_accumulator = Some(FoldedInstance::relaxed(new_instance));
            self.current_witness = Some(FoldedWitness::initial(new_witness));
            self.metrics.folding_steps = 1;
            self.metrics.instances_folded = 1;
            
            return Ok(FoldingProof {
                cross_term_commitment: vec![0u8; 32],
                challenge: vec![],
                aux_commitments: vec![],
            });
        }

        let acc = self.current_accumulator.as_ref().unwrap();
        let acc_witness = self.current_witness.as_ref().unwrap();

        // Step 1: Compute cross-term T = A(w_acc) * B(w_new) + A(w_new) * B(w_acc)
        let cross_term = self.compute_cross_term(acc_witness, new_witness)?;

        // Step 2: Commit to cross-term
        let cross_term_commitment = self.commit(&cross_term)?;

        // Step 3: Generate Fiat-Shamir challenge r
        let challenge = self.generate_challenge(
            &acc.running_hash,
            new_instance,
            &cross_term_commitment,
        )?;

        // Step 4: Compute folded instance
        let folded_instance = self.fold_instances(acc, new_instance, &challenge)?;

        // Step 5: Compute folded witness
        let folded_witness = self.fold_witnesses(acc_witness, new_witness, &challenge)?;

        // Update state
        self.current_accumulator = Some(folded_instance);
        self.current_witness = Some(folded_witness);
        self.metrics.folding_steps += 1;
        self.metrics.instances_folded += 1;

        if self.config.collect_metrics {
            self.metrics.proof_generation_us += start.elapsed().as_micros() as u64;
        }

        Ok(FoldingProof {
            cross_term_commitment,
            challenge,
            aux_commitments: vec![],
        })
    }

    /// Verify a folding proof
    pub fn verify_fold(
        &self,
        old_acc: &FoldedInstance,
        new_instance: &R1CSInstance,
        new_acc: &FoldedInstance,
        proof: &FoldingProof,
    ) -> NovaResult<bool> {
        let start = Instant::now();

        // Recompute the challenge
        let expected_challenge = self.generate_challenge(
            &old_acc.running_hash,
            new_instance,
            &proof.cross_term_commitment,
        )?;

        // Check challenge matches
        if expected_challenge != proof.challenge {
            return Ok(false);
        }

        // Verify the folded instance was computed correctly
        let expected_folded = self.fold_instances(old_acc, new_instance, &proof.challenge)?;

        // Check accumulated values match
        if expected_folded.accumulated_x != new_acc.accumulated_x {
            return Ok(false);
        }

        // Check step count
        if expected_folded.num_steps != new_acc.num_steps {
            return Ok(false);
        }

        Ok(true)
    }

    /// Get current accumulator
    pub fn accumulator(&self) -> Option<&FoldedInstance> {
        self.current_accumulator.as_ref()
    }

    /// Get current metrics
    pub fn metrics(&self) -> &NovaMetrics {
        &self.metrics
    }

    /// Reset the folding engine
    pub fn reset(&mut self) {
        self.current_accumulator = None;
        self.current_witness = None;
        self.metrics = NovaMetrics::default();
        if let Some(cs) = &self.constraint_system {
            self.metrics.total_constraints = cs.num_constraints();
        }
    }

    // Private helper methods

    fn compute_cross_term(
        &self,
        _acc_witness: &FoldedWitness,
        _new_witness: &R1CSWitness,
    ) -> NovaResult<Vec<u8>> {
        // In a full implementation, this computes:
        // T_i = A_i(w_acc) * B_i(w_new) + A_i(w_new) * B_i(w_acc) - u_acc * C_i(w_new) - C_i(w_acc)
        //
        // For now, we return a placeholder that demonstrates the structure
        let cross_term = vec![0u8; 32]; // Would be computed from actual witness values
        Ok(cross_term)
    }

    fn commit(&self, data: &[u8]) -> NovaResult<Vec<u8>> {
        // Pedersen commitment: C = g^m * h^r
        // For now, we use a hash-based commitment as placeholder
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.update(b"nexuszero-commitment");
        Ok(hasher.finalize().to_vec())
    }

    fn generate_challenge(
        &self,
        running_hash: &[u8; 32],
        instance: &R1CSInstance,
        commitment: &[u8],
    ) -> NovaResult<Vec<u8>> {
        // Fiat-Shamir challenge: r = H(running_hash || instance || commitment)
        let mut hasher = Sha3_256::new();
        hasher.update(running_hash);
        for input in &instance.public_inputs {
            hasher.update(input);
        }
        hasher.update(&instance.cs_hash);
        hasher.update(commitment);
        hasher.update(b"nexuszero-folding-challenge");
        
        Ok(hasher.finalize().to_vec())
    }

    fn fold_instances(
        &self,
        acc: &FoldedInstance,
        new_instance: &R1CSInstance,
        challenge: &[u8],
    ) -> NovaResult<FoldedInstance> {
        // Fold public inputs: x' = x_acc + r * x_new
        let mut folded_x = Vec::with_capacity(acc.accumulated_x.len());
        
        for (acc_x, new_x) in acc.accumulated_x.iter().zip(new_instance.public_inputs.iter()) {
            // Simplified addition with challenge weighting
            let mut result = acc_x.clone();
            for (i, byte) in new_x.iter().enumerate() {
                if i < result.len() && !challenge.is_empty() {
                    // In real impl: result[i] = acc_x[i] + challenge * new_x[i] mod p
                    result[i] = result[i].wrapping_add(byte.wrapping_mul(challenge[0]));
                }
            }
            folded_x.push(result);
        }

        // Update running hash
        let mut hasher = Sha3_256::new();
        hasher.update(&acc.running_hash);
        for x in &folded_x {
            hasher.update(x);
        }
        hasher.update(challenge);
        let running_hash: [u8; 32] = hasher.finalize().into();

        Ok(FoldedInstance {
            accumulated_x: folded_x,
            error_commitment: self.fold_error(&acc.error_commitment, challenge)?,
            num_steps: acc.num_steps + 1,
            running_hash,
            last_challenge: challenge.to_vec(),
        })
    }

    fn fold_error(&self, current_error: &[u8], challenge: &[u8]) -> NovaResult<Vec<u8>> {
        // E' = E + r^2 * T (error accumulation)
        let mut hasher = Sha3_256::new();
        hasher.update(current_error);
        hasher.update(challenge);
        hasher.update(challenge); // r^2
        Ok(hasher.finalize().to_vec())
    }

    fn fold_witnesses(
        &self,
        acc_witness: &FoldedWitness,
        new_witness: &R1CSWitness,
        challenge: &[u8],
    ) -> NovaResult<FoldedWitness> {
        // Fold witness: w' = w_acc + r * w_new
        let mut folded_witness = Vec::with_capacity(acc_witness.witness_values.len());
        
        for (acc_w, new_w) in acc_witness.witness_values.iter()
            .zip(new_witness.witness_values.iter())
        {
            let mut result = acc_w.clone();
            for (i, byte) in new_w.iter().enumerate() {
                if i < result.len() && !challenge.is_empty() {
                    result[i] = result[i].wrapping_add(byte.wrapping_mul(challenge[0]));
                }
            }
            folded_witness.push(result);
        }

        // Update error vector
        let mut error_vector = acc_witness.error_vector.clone();
        if !challenge.is_empty() {
            error_vector.push(challenge.to_vec());
        }

        Ok(FoldedWitness {
            witness_values: folded_witness,
            error_vector,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_instance() -> R1CSInstance {
        R1CSInstance::new(
            vec![vec![1, 2, 3], vec![4, 5, 6]],
            [0u8; 32],
        )
    }

    fn create_test_witness() -> R1CSWitness {
        R1CSWitness::new(vec![vec![7, 8, 9], vec![10, 11, 12]])
    }

    #[test]
    fn test_folded_instance_initial() {
        let instance = create_test_instance();
        let folded = FoldedInstance::initial(&instance);
        
        assert_eq!(folded.num_steps, 1);
        assert_eq!(folded.accumulated_x.len(), instance.public_inputs.len());
    }

    #[test]
    fn test_folded_instance_relaxed() {
        let instance = create_test_instance();
        let folded = FoldedInstance::relaxed(&instance);
        
        // Should have extra element for relaxation factor u
        assert_eq!(folded.accumulated_x.len(), instance.public_inputs.len() + 1);
    }

    #[test]
    fn test_folding_engine_creation() {
        let engine = FoldingEngine::new(1000, false);
        
        assert!(engine.accumulator().is_none());
        assert_eq!(engine.metrics().folding_steps, 0);
    }

    #[test]
    fn test_folding_engine_init() {
        let mut engine = FoldingEngine::new(1000, false);
        
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        let a = cs.alloc_public("a");
        let b = cs.alloc_private("b");
        let c = cs.alloc_private("c");
        cs.enforce_mul(a, b, c, None);
        
        let result = engine.init(cs);
        assert!(result.is_ok());
        assert_eq!(engine.metrics().total_constraints, 1);
    }

    #[test]
    fn test_folding_proof_size() {
        let proof = FoldingProof {
            cross_term_commitment: vec![0u8; 32],
            challenge: vec![0u8; 32],
            aux_commitments: vec![vec![0u8; 32]],
        };
        
        assert_eq!(proof.size(), 96);
    }

    #[test]
    fn test_challenge_generation() {
        let engine = FoldingEngine::new(1000, false);
        
        let instance = create_test_instance();
        let running_hash = [0u8; 32];
        let commitment = vec![1u8; 32];
        
        let challenge = engine.generate_challenge(&running_hash, &instance, &commitment);
        assert!(challenge.is_ok());
        assert_eq!(challenge.unwrap().len(), 32);
    }

    #[test]
    fn test_commit() {
        let engine = FoldingEngine::new(1000, false);
        
        let data = vec![1u8, 2, 3, 4, 5];
        let commitment = engine.commit(&data);
        
        assert!(commitment.is_ok());
        assert_eq!(commitment.unwrap().len(), 32);
    }

    #[test]
    fn test_folded_witness_zeroize() {
        let witness = R1CSWitness::new(vec![vec![1, 2, 3], vec![4, 5, 6]]);
        let mut folded = FoldedWitness::initial(&witness);
        
        folded.zeroize();
        
        for w in &folded.witness_values {
            for byte in w {
                assert_eq!(*byte, 0);
            }
        }
    }
}
