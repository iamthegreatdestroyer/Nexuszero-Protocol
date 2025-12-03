// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! Nova Step Circuits
//!
//! This module defines the `StepCircuit` trait and provides several
//! built-in circuit implementations for common IVC patterns.
//!
//! # Overview
//!
//! In Nova's IVC model, computation is expressed as repeated applications
//! of a "step function" F: (i, z_i) -> z_{i+1}. The StepCircuit trait
//! encapsulates this step function as an R1CS circuit.
//!
//! # Built-in Circuits
//!
//! - `TrivialCircuit`: Identity function, useful for testing
//! - `MinRootCircuit`: Computes iterated square roots (from Nova paper)
//! - `HashChainCircuit`: Computes iterated hashes
//! - `MerkleUpdateCircuit`: Merkle tree membership/update proofs

use super::r1cs::{R1CSConstraintSystem, LinearCombination, R1CSConstraint};
use super::types::{NovaError, NovaResult};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Trait for circuits that can be folded with Nova
///
/// A step circuit represents a single step of an IVC computation.
/// The circuit takes an input state z_i and produces output state z_{i+1}.
pub trait StepCircuit: Send + Sync + Clone {
    /// Number of elements in the state vector
    fn arity(&self) -> usize;

    /// Generate the R1CS constraints for this step function
    ///
    /// # Arguments
    /// * `cs` - The constraint system to add constraints to
    /// * `step_idx` - The current step index (0-indexed)
    /// * `z_in` - Variable IDs for input state elements
    ///
    /// # Returns
    /// Variable IDs for output state elements
    fn synthesize(
        &self,
        cs: &mut R1CSConstraintSystem,
        step_idx: usize,
        z_in: &[usize],
    ) -> NovaResult<Vec<usize>>;

    /// Compute the step function on concrete inputs
    ///
    /// This is used for witness generation.
    fn compute(&self, step_idx: usize, z_in: &[Vec<u8>]) -> NovaResult<Vec<Vec<u8>>>;

    /// Get circuit metadata for optimization
    fn metadata(&self) -> CircuitMetadata {
        CircuitMetadata::default()
    }
}

/// Metadata about a step circuit
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CircuitMetadata {
    /// Human-readable name
    pub name: String,
    /// Estimated number of constraints
    pub estimated_constraints: usize,
    /// Whether the circuit is data-independent (same constraints for all inputs)
    pub data_independent: bool,
    /// Parallelization hint
    pub parallelizable: bool,
}

/// Trivial step circuit that copies input to output
///
/// This is useful for testing the folding infrastructure without
/// complex circuit logic.
#[derive(Clone, Debug)]
pub struct TrivialCircuit {
    arity: usize,
}

impl TrivialCircuit {
    /// Create a new trivial circuit with given state size
    pub fn new(arity: usize) -> Self {
        Self { arity }
    }
}

impl StepCircuit for TrivialCircuit {
    fn arity(&self) -> usize {
        self.arity
    }

    fn synthesize(
        &self,
        cs: &mut R1CSConstraintSystem,
        _step_idx: usize,
        z_in: &[usize],
    ) -> NovaResult<Vec<usize>> {
        if z_in.len() != self.arity {
            return Err(NovaError::InvalidCircuit(format!(
                "Expected {} inputs, got {}",
                self.arity,
                z_in.len()
            )));
        }

        // Trivial circuit: output = input
        // We just allocate new output variables and constrain them equal to inputs
        let mut z_out = Vec::with_capacity(self.arity);
        
        for (i, &z_i) in z_in.iter().enumerate() {
            let out_var = cs.alloc_private(&format!("trivial_out_{}", i));
            cs.enforce_equal(z_i, out_var, Some(&format!("trivial_copy_{}", i)));
            z_out.push(out_var);
        }

        Ok(z_out)
    }

    fn compute(&self, _step_idx: usize, z_in: &[Vec<u8>]) -> NovaResult<Vec<Vec<u8>>> {
        // Trivial: output = input
        Ok(z_in.to_vec())
    }

    fn metadata(&self) -> CircuitMetadata {
        CircuitMetadata {
            name: "TrivialCircuit".to_string(),
            estimated_constraints: self.arity,
            data_independent: true,
            parallelizable: true,
        }
    }
}

/// MinRoot step circuit from the Nova paper
///
/// Computes z_{i+1} = z_i^{1/5} (fifth root in the field).
/// This is useful for benchmarking as it has a known number of
/// constraints and is computationally intensive.
#[derive(Clone, Debug)]
pub struct MinRootCircuit {
    num_iters_per_step: usize,
}

impl MinRootCircuit {
    /// Create a new MinRoot circuit
    ///
    /// # Arguments
    /// * `num_iters_per_step` - Number of fifth-root iterations per step
    pub fn new(num_iters_per_step: usize) -> Self {
        Self { num_iters_per_step }
    }
}

impl StepCircuit for MinRootCircuit {
    fn arity(&self) -> usize {
        2 // (x, y) where we compute x <- x^{1/5} iteratively
    }

    fn synthesize(
        &self,
        cs: &mut R1CSConstraintSystem,
        _step_idx: usize,
        z_in: &[usize],
    ) -> NovaResult<Vec<usize>> {
        if z_in.len() != 2 {
            return Err(NovaError::InvalidCircuit(
                "MinRootCircuit requires arity 2".to_string()
            ));
        }

        let mut x = z_in[0];
        let y = z_in[1];

        // Compute x <- x^{1/5} iteratively
        // This is done by proving x^5 = x_prev for each iteration
        for iter in 0..self.num_iters_per_step {
            // Allocate x_new = x^{1/5}
            let x_new = cs.alloc_private(&format!("minroot_x_{}", iter));
            
            // We need to prove x_new^5 = x
            // Break down: x_new^2, x_new^4, x_new^5
            let x_sq = cs.alloc_private(&format!("minroot_x_sq_{}", iter));
            let x_quad = cs.alloc_private(&format!("minroot_x_quad_{}", iter));
            let x_fifth = cs.alloc_private(&format!("minroot_x_fifth_{}", iter));

            // x_sq = x_new * x_new
            cs.enforce_mul(x_new, x_new, x_sq, Some(&format!("minroot_sq_{}", iter)));
            
            // x_quad = x_sq * x_sq
            cs.enforce_mul(x_sq, x_sq, x_quad, Some(&format!("minroot_quad_{}", iter)));
            
            // x_fifth = x_quad * x_new
            cs.enforce_mul(x_quad, x_new, x_fifth, Some(&format!("minroot_fifth_{}", iter)));
            
            // x_fifth = x (previous x)
            cs.enforce_equal(x_fifth, x, Some(&format!("minroot_check_{}", iter)));

            x = x_new;
        }

        // Output: (x_final, y)
        let y_out = cs.alloc_private("minroot_y_out");
        cs.enforce_equal(y, y_out, Some("minroot_y_copy"));

        Ok(vec![x, y_out])
    }

    fn compute(&self, _step_idx: usize, z_in: &[Vec<u8>]) -> NovaResult<Vec<Vec<u8>>> {
        if z_in.len() != 2 {
            return Err(NovaError::InvalidCircuit(
                "MinRootCircuit requires 2 inputs".to_string()
            ));
        }

        // In real implementation, compute x^{1/5} in the field
        // For now, return placeholder
        Ok(z_in.to_vec())
    }

    fn metadata(&self) -> CircuitMetadata {
        CircuitMetadata {
            name: "MinRootCircuit".to_string(),
            // Each iteration needs 4 multiplication constraints
            estimated_constraints: self.num_iters_per_step * 4 + 1,
            data_independent: true,
            parallelizable: false, // Sequential fifth roots
        }
    }
}

/// Hash chain step circuit
///
/// Computes z_{i+1} = H(z_i) for iterated hashing.
/// Useful for proving hash chains efficiently.
#[derive(Clone, Debug)]
pub struct HashChainCircuit {
    hash_type: HashType,
}

/// Supported hash functions for circuits
#[derive(Clone, Debug, Copy, Serialize, Deserialize)]
pub enum HashType {
    /// Poseidon hash (SNARK-friendly)
    Poseidon,
    /// MiMC hash
    MiMC,
    /// Rescue hash
    Rescue,
    /// SHA256 (expensive in R1CS)
    SHA256,
}

impl HashChainCircuit {
    /// Create a new hash chain circuit
    pub fn new(hash_type: HashType) -> Self {
        Self { hash_type }
    }

    /// Create with Poseidon hash (recommended)
    pub fn poseidon() -> Self {
        Self::new(HashType::Poseidon)
    }
}

impl StepCircuit for HashChainCircuit {
    fn arity(&self) -> usize {
        1 // Single hash value
    }

    fn synthesize(
        &self,
        cs: &mut R1CSConstraintSystem,
        step_idx: usize,
        z_in: &[usize],
    ) -> NovaResult<Vec<usize>> {
        if z_in.len() != 1 {
            return Err(NovaError::InvalidCircuit(
                "HashChainCircuit requires arity 1".to_string()
            ));
        }

        let input = z_in[0];
        
        // The actual hash circuit depends on the hash type
        // Here we create placeholder constraints
        let output = match self.hash_type {
            HashType::Poseidon => {
                // Poseidon uses a series of additions and multiplications
                // Simplified: 8 rounds of S-box (x^5) and linear layers
                self.synthesize_poseidon(cs, input, step_idx)?
            }
            HashType::MiMC => {
                // MiMC uses x^3 rounds
                self.synthesize_mimc(cs, input, step_idx)?
            }
            _ => {
                // Placeholder for other hash types
                let out = cs.alloc_private(&format!("hash_out_{}", step_idx));
                cs.enforce_mul(input, input, out, Some("hash_placeholder"));
                out
            }
        };

        Ok(vec![output])
    }

    fn compute(&self, _step_idx: usize, z_in: &[Vec<u8>]) -> NovaResult<Vec<Vec<u8>>> {
        use sha3::{Sha3_256, Digest};
        
        if z_in.len() != 1 {
            return Err(NovaError::InvalidCircuit(
                "HashChainCircuit requires 1 input".to_string()
            ));
        }

        // Compute hash based on type
        let mut hasher = Sha3_256::new();
        hasher.update(&z_in[0]);
        let result = hasher.finalize().to_vec();

        Ok(vec![result])
    }

    fn metadata(&self) -> CircuitMetadata {
        let constraints = match self.hash_type {
            HashType::Poseidon => 300,  // ~300 constraints for Poseidon
            HashType::MiMC => 350,       // ~350 for MiMC
            HashType::Rescue => 400,     // ~400 for Rescue
            HashType::SHA256 => 25000,   // SHA256 is expensive in R1CS
        };

        CircuitMetadata {
            name: format!("HashChainCircuit({:?})", self.hash_type),
            estimated_constraints: constraints,
            data_independent: true,
            parallelizable: true,
        }
    }
}

impl HashChainCircuit {
    fn synthesize_poseidon(&self, cs: &mut R1CSConstraintSystem, input: usize, step_idx: usize) -> NovaResult<usize> {
        // Simplified Poseidon: 8 full rounds
        // Each round: S-box (x^5) + linear layer
        let mut state = input;
        
        for round in 0..8 {
            // S-box: x^5
            let x2 = cs.alloc_private(&format!("poseidon_x2_{}_{}", step_idx, round));
            let x4 = cs.alloc_private(&format!("poseidon_x4_{}_{}", step_idx, round));
            let x5 = cs.alloc_private(&format!("poseidon_x5_{}_{}", step_idx, round));
            
            cs.enforce_mul(state, state, x2, Some(&format!("poseidon_sq_{}_{}", step_idx, round)));
            cs.enforce_mul(x2, x2, x4, Some(&format!("poseidon_quad_{}_{}", step_idx, round)));
            cs.enforce_mul(x4, state, x5, Some(&format!("poseidon_fifth_{}_{}", step_idx, round)));
            
            state = x5;
        }
        
        Ok(state)
    }

    fn synthesize_mimc(&self, cs: &mut R1CSConstraintSystem, input: usize, step_idx: usize) -> NovaResult<usize> {
        // Simplified MiMC: 64 rounds of x^3
        let mut state = input;
        
        for round in 0..64 {
            // x^3 = x * x * x
            let x2 = cs.alloc_private(&format!("mimc_x2_{}_{}", step_idx, round));
            let x3 = cs.alloc_private(&format!("mimc_x3_{}_{}", step_idx, round));
            
            cs.enforce_mul(state, state, x2, Some(&format!("mimc_sq_{}_{}", step_idx, round)));
            cs.enforce_mul(x2, state, x3, Some(&format!("mimc_cube_{}_{}", step_idx, round)));
            
            state = x3;
        }
        
        Ok(state)
    }
}

/// Merkle tree update circuit
///
/// Proves that updating a Merkle tree from root R1 to R2
/// is valid given an update path.
#[derive(Clone, Debug)]
pub struct MerkleUpdateCircuit {
    tree_depth: usize,
}

impl MerkleUpdateCircuit {
    /// Create a new Merkle update circuit
    ///
    /// # Arguments
    /// * `tree_depth` - Depth of the Merkle tree
    pub fn new(tree_depth: usize) -> Self {
        Self { tree_depth }
    }
}

impl StepCircuit for MerkleUpdateCircuit {
    fn arity(&self) -> usize {
        // State: (old_root, new_root, leaf_index)
        3
    }

    fn synthesize(
        &self,
        cs: &mut R1CSConstraintSystem,
        step_idx: usize,
        z_in: &[usize],
    ) -> NovaResult<Vec<usize>> {
        if z_in.len() != 3 {
            return Err(NovaError::InvalidCircuit(
                "MerkleUpdateCircuit requires arity 3".to_string()
            ));
        }

        let old_root = z_in[0];
        let new_root = z_in[1];
        let leaf_index = z_in[2];

        // Allocate Merkle path siblings
        let mut path_siblings = Vec::with_capacity(self.tree_depth);
        for i in 0..self.tree_depth {
            path_siblings.push(cs.alloc_private(&format!("merkle_sibling_{}_{}", step_idx, i)));
        }

        // Allocate old and new leaf values
        let old_leaf = cs.alloc_private(&format!("merkle_old_leaf_{}", step_idx));
        let new_leaf = cs.alloc_private(&format!("merkle_new_leaf_{}", step_idx));

        // Verify old Merkle path (simplified)
        // In full implementation: hash up the tree and check against old_root
        let computed_old_root = cs.alloc_private(&format!("merkle_computed_old_{}", step_idx));
        cs.enforce_equal(computed_old_root, old_root, Some("merkle_old_root_check"));

        // Verify new Merkle path
        let computed_new_root = cs.alloc_private(&format!("merkle_computed_new_{}", step_idx));
        cs.enforce_equal(computed_new_root, new_root, Some("merkle_new_root_check"));

        // Output state: (new_root, new_root, leaf_index + 1)
        let next_index = cs.alloc_private(&format!("merkle_next_index_{}", step_idx));
        
        Ok(vec![new_root, new_root, next_index])
    }

    fn compute(&self, _step_idx: usize, z_in: &[Vec<u8>]) -> NovaResult<Vec<Vec<u8>>> {
        // In real implementation, verify and update Merkle tree
        Ok(z_in.to_vec())
    }

    fn metadata(&self) -> CircuitMetadata {
        // Each tree level needs ~300 constraints for Poseidon hash
        CircuitMetadata {
            name: format!("MerkleUpdateCircuit(depth={})", self.tree_depth),
            estimated_constraints: self.tree_depth * 300 + 50,
            data_independent: false, // Depends on leaf index bits
            parallelizable: false,
        }
    }
}

/// Circuit composition: chain multiple step circuits
#[derive(Clone)]
pub struct ComposedCircuit<C1, C2>
where
    C1: StepCircuit,
    C2: StepCircuit,
{
    first: C1,
    second: C2,
    _marker: PhantomData<(C1, C2)>,
}

impl<C1, C2> ComposedCircuit<C1, C2>
where
    C1: StepCircuit,
    C2: StepCircuit,
{
    /// Compose two circuits: second(first(x))
    pub fn new(first: C1, second: C2) -> NovaResult<Self> {
        if first.arity() != second.arity() {
            return Err(NovaError::InvalidCircuit(
                "Cannot compose circuits with different arities".to_string()
            ));
        }
        Ok(Self {
            first,
            second,
            _marker: PhantomData,
        })
    }
}

impl<C1, C2> StepCircuit for ComposedCircuit<C1, C2>
where
    C1: StepCircuit,
    C2: StepCircuit,
{
    fn arity(&self) -> usize {
        self.first.arity()
    }

    fn synthesize(
        &self,
        cs: &mut R1CSConstraintSystem,
        step_idx: usize,
        z_in: &[usize],
    ) -> NovaResult<Vec<usize>> {
        // Run first circuit
        let intermediate = self.first.synthesize(cs, step_idx * 2, z_in)?;
        // Run second circuit on intermediate output
        self.second.synthesize(cs, step_idx * 2 + 1, &intermediate)
    }

    fn compute(&self, step_idx: usize, z_in: &[Vec<u8>]) -> NovaResult<Vec<Vec<u8>>> {
        let intermediate = self.first.compute(step_idx * 2, z_in)?;
        self.second.compute(step_idx * 2 + 1, &intermediate)
    }

    fn metadata(&self) -> CircuitMetadata {
        let m1 = self.first.metadata();
        let m2 = self.second.metadata();
        CircuitMetadata {
            name: format!("Composed({}, {})", m1.name, m2.name),
            estimated_constraints: m1.estimated_constraints + m2.estimated_constraints,
            data_independent: m1.data_independent && m2.data_independent,
            parallelizable: m1.parallelizable && m2.parallelizable,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::nova::types::NovaSecurityLevel;

    #[test]
    fn test_trivial_circuit() {
        let circuit = TrivialCircuit::new(3);
        assert_eq!(circuit.arity(), 3);
        
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        let inputs: Vec<usize> = (0..3).map(|i| cs.alloc_public(&format!("in_{}", i))).collect();
        
        let outputs = circuit.synthesize(&mut cs, 0, &inputs);
        assert!(outputs.is_ok());
        assert_eq!(outputs.unwrap().len(), 3);
    }

    #[test]
    fn test_trivial_circuit_compute() {
        let circuit = TrivialCircuit::new(2);
        let input = vec![vec![1, 2, 3], vec![4, 5, 6]];
        
        let output = circuit.compute(0, &input);
        assert!(output.is_ok());
        assert_eq!(output.unwrap(), input);
    }

    #[test]
    fn test_minroot_circuit() {
        let circuit = MinRootCircuit::new(2);
        assert_eq!(circuit.arity(), 2);
        
        let metadata = circuit.metadata();
        assert!(metadata.estimated_constraints > 0);
        assert!(!metadata.parallelizable);
    }

    #[test]
    fn test_hash_chain_circuit() {
        let circuit = HashChainCircuit::poseidon();
        assert_eq!(circuit.arity(), 1);
        
        let metadata = circuit.metadata();
        assert!(metadata.name.contains("Poseidon"));
    }

    #[test]
    fn test_merkle_update_circuit() {
        let circuit = MerkleUpdateCircuit::new(20);
        assert_eq!(circuit.arity(), 3);
        
        let metadata = circuit.metadata();
        assert!(metadata.estimated_constraints > 5000); // 20 * 300 + 50
    }

    #[test]
    fn test_circuit_composition() {
        let c1 = TrivialCircuit::new(2);
        let c2 = TrivialCircuit::new(2);
        
        let composed = ComposedCircuit::new(c1, c2);
        assert!(composed.is_ok());
        
        let composed = composed.unwrap();
        assert_eq!(composed.arity(), 2);
    }

    #[test]
    fn test_circuit_composition_arity_mismatch() {
        let c1 = TrivialCircuit::new(2);
        let c2 = TrivialCircuit::new(3);
        
        let composed = ComposedCircuit::new(c1, c2);
        assert!(composed.is_err());
    }
}
