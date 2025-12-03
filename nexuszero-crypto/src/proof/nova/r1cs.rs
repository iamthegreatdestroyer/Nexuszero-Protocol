// Copyright (c) 2025 NexusZero Protocol
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

//! R1CS (Rank-1 Constraint System) Conversion Layer
//!
//! This module provides the abstraction layer for converting NexusZero's
//! Statement/Witness model to R1CS constraint systems compatible with Nova.
//!
//! # R1CS Overview
//!
//! R1CS represents computations as a system of quadratic constraints:
//! ```text
//! (A · z) ⊙ (B · z) = (C · z)
//! ```
//! where:
//! - A, B, C are sparse matrices
//! - z is the witness vector [1, x, w] (constant, public inputs, private witness)
//! - ⊙ is element-wise (Hadamard) product
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────┐
//! │  NexusZero Statement│
//! │  + Witness          │
//! └──────────┬──────────┘
//!            │
//!            ▼
//! ┌──────────────────────┐
//! │   R1CS Converter     │
//! │  ┌────────────────┐  │
//! │  │ Variable Alloc │  │
//! │  │ Constraint Gen │  │
//! │  │ Witness Map    │  │
//! │  └────────────────┘  │
//! └──────────┬───────────┘
//!            │
//!            ▼
//! ┌──────────────────────┐
//! │   R1CS Instance      │
//! │  + R1CS Witness      │
//! │  (Nova-compatible)   │
//! └──────────────────────┘
//! ```

use super::types::{NovaError, NovaResult, NovaSecurityLevel};
use crate::proof::{Statement, StatementType, Witness};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[cfg(feature = "nova")]
use bellpepper_core::{
    num::AllocatedNum,
    ConstraintSystem, SynthesisError,
};

#[cfg(feature = "nova")]
use ff::PrimeField;

/// Represents a variable in the R1CS constraint system
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct R1CSVariable {
    /// Unique identifier for this variable
    pub id: usize,
    /// Human-readable name (for debugging)
    pub name: String,
    /// Whether this is a public input
    pub is_public: bool,
    /// Bit width of the variable
    pub bit_width: usize,
}

impl R1CSVariable {
    /// Create a new public variable
    pub fn public(id: usize, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            is_public: true,
            bit_width: 256, // Default to 256-bit field elements
        }
    }

    /// Create a new private (witness) variable
    pub fn private(id: usize, name: impl Into<String>) -> Self {
        Self {
            id,
            name: name.into(),
            is_public: false,
            bit_width: 256,
        }
    }

    /// Set the bit width
    pub fn with_bit_width(mut self, bits: usize) -> Self {
        self.bit_width = bits;
        self
    }
}

/// Linear combination of variables: Σ(coefficient * variable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearCombination {
    /// Terms in the linear combination: (variable_id, coefficient)
    pub terms: Vec<(usize, Vec<u8>)>,
}

impl LinearCombination {
    /// Create an empty linear combination
    pub fn zero() -> Self {
        Self { terms: vec![] }
    }

    /// Create from a single variable with coefficient 1
    pub fn from_variable(var_id: usize) -> Self {
        Self {
            terms: vec![(var_id, vec![1])],
        }
    }

    /// Add a term to the linear combination
    pub fn add_term(&mut self, var_id: usize, coefficient: Vec<u8>) {
        self.terms.push((var_id, coefficient));
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Number of terms
    pub fn len(&self) -> usize {
        self.terms.len()
    }
}

/// A single R1CS constraint: A * B = C
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R1CSConstraint {
    /// Left input linear combination
    pub a: LinearCombination,
    /// Right input linear combination
    pub b: LinearCombination,
    /// Output linear combination
    pub c: LinearCombination,
    /// Optional constraint name for debugging
    pub name: Option<String>,
}

impl R1CSConstraint {
    /// Create a new constraint
    pub fn new(a: LinearCombination, b: LinearCombination, c: LinearCombination) -> Self {
        Self { a, b, c, name: None }
    }

    /// Add a name for debugging
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Complete R1CS constraint system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R1CSConstraintSystem {
    /// All variables in the system
    pub variables: Vec<R1CSVariable>,
    /// All constraints
    pub constraints: Vec<R1CSConstraint>,
    /// Number of public inputs
    pub num_public_inputs: usize,
    /// Number of private witness values
    pub num_witness: usize,
    /// Security level
    pub security_level: NovaSecurityLevel,
}

impl R1CSConstraintSystem {
    /// Create a new empty constraint system
    pub fn new(security_level: NovaSecurityLevel) -> Self {
        Self {
            variables: vec![],
            constraints: vec![],
            num_public_inputs: 0,
            num_witness: 0,
            security_level,
        }
    }

    /// Allocate a new public input variable
    pub fn alloc_public(&mut self, name: impl Into<String>) -> usize {
        let id = self.variables.len();
        self.variables.push(R1CSVariable::public(id, name));
        self.num_public_inputs += 1;
        id
    }

    /// Allocate a new private witness variable
    pub fn alloc_private(&mut self, name: impl Into<String>) -> usize {
        let id = self.variables.len();
        self.variables.push(R1CSVariable::private(id, name));
        self.num_witness += 1;
        id
    }

    /// Add a constraint to the system
    pub fn add_constraint(&mut self, constraint: R1CSConstraint) {
        self.constraints.push(constraint);
    }

    /// Add a multiplication constraint: a * b = c
    pub fn enforce_mul(&mut self, a: usize, b: usize, c: usize, name: Option<&str>) {
        let constraint = R1CSConstraint::new(
            LinearCombination::from_variable(a),
            LinearCombination::from_variable(b),
            LinearCombination::from_variable(c),
        );
        let constraint = if let Some(n) = name {
            constraint.with_name(n)
        } else {
            constraint
        };
        self.add_constraint(constraint);
    }

    /// Add an equality constraint: a = b (implemented as a * 1 = b)
    pub fn enforce_equal(&mut self, a: usize, b: usize, name: Option<&str>) {
        let mut a_lc = LinearCombination::from_variable(a);
        let b_lc = LinearCombination::from_variable(b);
        
        // We need to express a = b as a constraint
        // Use: a * 1 = b
        let constraint = R1CSConstraint::new(
            a_lc,
            LinearCombination { terms: vec![(0, vec![1])] }, // Constant 1
            b_lc,
        );
        let constraint = if let Some(n) = name {
            constraint.with_name(n)
        } else {
            constraint
        };
        self.add_constraint(constraint);
    }

    /// Number of constraints
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Total number of variables
    pub fn num_variables(&self) -> usize {
        self.variables.len()
    }

    /// Validate the constraint system
    pub fn validate(&self) -> NovaResult<()> {
        // Check all variable references are valid
        for (i, constraint) in self.constraints.iter().enumerate() {
            for (var_id, _) in &constraint.a.terms {
                if *var_id >= self.variables.len() {
                    return Err(NovaError::R1CSError(format!(
                        "Invalid variable reference {} in constraint {} (A)",
                        var_id, i
                    )));
                }
            }
            for (var_id, _) in &constraint.b.terms {
                if *var_id >= self.variables.len() {
                    return Err(NovaError::R1CSError(format!(
                        "Invalid variable reference {} in constraint {} (B)",
                        var_id, i
                    )));
                }
            }
            for (var_id, _) in &constraint.c.terms {
                if *var_id >= self.variables.len() {
                    return Err(NovaError::R1CSError(format!(
                        "Invalid variable reference {} in constraint {} (C)",
                        var_id, i
                    )));
                }
            }
        }
        Ok(())
    }
}

/// R1CS instance (public inputs + structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct R1CSInstance {
    /// Public input values
    pub public_inputs: Vec<Vec<u8>>,
    /// Hash of the constraint system structure
    #[serde(default)]
    pub cs_hash: [u8; 32],
    /// Shape hash for compatibility (alias)
    #[serde(default)]
    pub shape_hash: [u8; 32],
}

impl R1CSInstance {
    /// Create a new R1CS instance
    pub fn new(public_inputs: Vec<Vec<u8>>, cs_hash: [u8; 32]) -> Self {
        Self {
            public_inputs,
            cs_hash,
            shape_hash: cs_hash,
        }
    }
}

/// R1CS witness (private values satisfying the constraints)
#[derive(Debug, Clone)]
pub struct R1CSWitness {
    /// Private witness values
    pub witness_values: Vec<Vec<u8>>,
    /// Variable assignments (variable_id, value)
    pub assignments: Vec<(usize, Vec<u8>)>,
}

impl R1CSWitness {
    /// Create a new witness from values
    pub fn new(values: Vec<Vec<u8>>) -> Self {
        let assignments = values.iter().enumerate()
            .map(|(i, v)| (i, v.clone()))
            .collect();
        Self { 
            witness_values: values,
            assignments,
        }
    }
    
    /// Create from assignments
    pub fn from_assignments(assignments: Vec<(usize, Vec<u8>)>) -> Self {
        let witness_values = assignments.iter().map(|(_, v)| v.clone()).collect();
        Self {
            witness_values,
            assignments,
        }
    }

    /// Zeroize sensitive witness data
    pub fn zeroize(&mut self) {
        for value in &mut self.witness_values {
            for byte in value.iter_mut() {
                *byte = 0;
            }
        }
        for (_, value) in &mut self.assignments {
            for byte in value.iter_mut() {
                *byte = 0;
            }
        }
    }
}

impl Drop for R1CSWitness {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// Converter from NexusZero Statement/Witness to R1CS
pub struct R1CSConverter {
    security_level: NovaSecurityLevel,
}

impl R1CSConverter {
    /// Create a new converter with default settings
    pub fn new(security_level: NovaSecurityLevel) -> Self {
        Self { security_level }
    }

    /// Create a converter with default security level
    pub fn default() -> Self {
        Self {
            security_level: NovaSecurityLevel::default(),
        }
    }

    /// Create a converter with specific security level (alias)
    pub fn with_security_level(security_level: NovaSecurityLevel) -> Self {
        Self { security_level }
    }
    
    /// Convert statement and witness to R1CS instance
    pub fn convert(&self, statement: &Statement, witness: &Witness) -> NovaResult<R1CSInstance> {
        let cs = self.convert_statement(statement)?;
        let _ = self.convert_witness(statement, witness, &cs)?;
        
        // Create instance with public inputs from statement
        let public_inputs = statement.public_inputs.iter()
            .map(|v| v.clone())
            .collect();
        
        let cs_hash = {
            use sha3::{Sha3_256, Digest};
            let mut hasher = Sha3_256::new();
            hasher.update(&cs.num_constraints().to_le_bytes());
            hasher.update(&cs.num_variables().to_le_bytes());
            let hash = hasher.finalize();
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&hash);
            arr
        };
        
        Ok(R1CSInstance::new(public_inputs, cs_hash))
    }

    /// Convert a NexusZero Statement to R1CS constraint system
    pub fn convert_statement(&self, statement: &Statement) -> NovaResult<R1CSConstraintSystem> {
        let mut cs = R1CSConstraintSystem::new(self.security_level);

        match &statement.statement_type {
            StatementType::DiscreteLog { generator, public_value } => {
                self.convert_discrete_log(&mut cs, generator, public_value)?;
            }
            StatementType::Preimage { hash_function, hash_output } => {
                self.convert_preimage(&mut cs, hash_output)?;
            }
            StatementType::Range { min, max, commitment } => {
                self.convert_range(&mut cs, *min, *max, commitment)?;
            }
            StatementType::Equality { left, right } => {
                self.convert_equality(&mut cs, left, right)?;
            }
            StatementType::Custom { type_id, data } => {
                self.convert_custom(&mut cs, type_id, data)?;
            }
        }

        cs.validate()?;
        Ok(cs)
    }

    /// Convert witness values to R1CS witness
    pub fn convert_witness(
        &self,
        statement: &Statement,
        witness: &Witness,
        cs: &R1CSConstraintSystem,
    ) -> NovaResult<R1CSWitness> {
        let mut witness_values = Vec::with_capacity(cs.num_witness);

        // Extract witness data based on statement type
        match &statement.statement_type {
            StatementType::DiscreteLog { .. } => {
                // Discrete log witness is the exponent
                witness_values.push(witness.data.clone());
            }
            StatementType::Preimage { .. } => {
                // Preimage witness is the preimage itself
                witness_values.push(witness.data.clone());
            }
            StatementType::Range { .. } => {
                // Range witness is the value and possibly bit decomposition
                witness_values.push(witness.data.clone());
                // Add bit decomposition if needed
                let value = u64::from_le_bytes(
                    witness.data.get(..8)
                        .and_then(|s| s.try_into().ok())
                        .unwrap_or([0u8; 8])
                );
                for i in 0..64 {
                    let bit = ((value >> i) & 1) as u8;
                    witness_values.push(vec![bit]);
                }
            }
            StatementType::Equality { .. } => {
                witness_values.push(witness.data.clone());
            }
            StatementType::Custom { .. } => {
                witness_values.push(witness.data.clone());
            }
        }

        Ok(R1CSWitness::new(witness_values))
    }

    /// Create R1CS instance from statement
    pub fn create_instance(
        &self,
        statement: &Statement,
        cs: &R1CSConstraintSystem,
    ) -> NovaResult<R1CSInstance> {
        let mut public_inputs = Vec::new();

        // Extract public inputs based on statement type
        match &statement.statement_type {
            StatementType::DiscreteLog { generator, public_value } => {
                public_inputs.push(generator.clone());
                public_inputs.push(public_value.clone());
            }
            StatementType::Preimage { hash_output, .. } => {
                public_inputs.push(hash_output.clone());
            }
            StatementType::Range { min, max, commitment } => {
                public_inputs.push(min.to_le_bytes().to_vec());
                public_inputs.push(max.to_le_bytes().to_vec());
                public_inputs.push(commitment.clone());
            }
            StatementType::Equality { left, right } => {
                public_inputs.push(left.clone());
                public_inputs.push(right.clone());
            }
            StatementType::Custom { data, .. } => {
                public_inputs.push(data.clone());
            }
        }

        // Compute hash of constraint system
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&bincode::serialize(cs).map_err(|e| {
            NovaError::SerializationError(format!("Failed to serialize CS: {}", e))
        })?);
        let cs_hash: [u8; 32] = hasher.finalize().into();

        Ok(R1CSInstance {
            public_inputs,
            cs_hash,
        })
    }

    // Private conversion methods for each statement type

    fn convert_discrete_log(
        &self,
        cs: &mut R1CSConstraintSystem,
        generator: &[u8],
        public_value: &[u8],
    ) -> NovaResult<()> {
        // Public inputs: generator G, public value Y
        let g = cs.alloc_public("generator");
        let y = cs.alloc_public("public_value");

        // Private witness: exponent x such that G^x = Y
        let x = cs.alloc_private("exponent");

        // For now, we encode this as a placeholder constraint
        // Full implementation would use point multiplication in the circuit
        cs.enforce_mul(g, x, y, Some("discrete_log_check"));

        Ok(())
    }

    fn convert_preimage(
        &self,
        cs: &mut R1CSConstraintSystem,
        hash_output: &[u8],
    ) -> NovaResult<()> {
        // Public input: hash output
        let h = cs.alloc_public("hash_output");

        // Private witness: preimage
        let preimage = cs.alloc_private("preimage");

        // Intermediate variables for hash computation
        // This is a simplified placeholder - real implementation would
        // decompose the hash function into R1CS constraints
        let hash_result = cs.alloc_private("hash_result");
        
        // Constraint: hash(preimage) = h
        cs.enforce_equal(hash_result, h, Some("preimage_check"));

        Ok(())
    }

    fn convert_range(
        &self,
        cs: &mut R1CSConstraintSystem,
        min: u64,
        max: u64,
        commitment: &[u8],
    ) -> NovaResult<()> {
        // Public inputs
        let min_var = cs.alloc_public("min");
        let max_var = cs.alloc_public("max");
        let commitment_var = cs.alloc_public("commitment");

        // Private witness: the actual value
        let value = cs.alloc_private("value");

        // Bit decomposition for range check
        let bit_width = 64;
        let mut bit_vars = Vec::with_capacity(bit_width);
        for i in 0..bit_width {
            bit_vars.push(cs.alloc_private(&format!("bit_{}", i)));
        }

        // Constraints:
        // 1. Each bit is boolean (b * b = b)
        for (i, &bit) in bit_vars.iter().enumerate() {
            cs.enforce_mul(bit, bit, bit, Some(&format!("bit_{}_boolean", i)));
        }

        // 2. Value reconstructs from bits (simplified)
        // In a real implementation, we'd sum bits * powers of 2

        // 3. Value >= min and value <= max
        // This requires additional comparison gadgets

        Ok(())
    }

    fn convert_equality(
        &self,
        cs: &mut R1CSConstraintSystem,
        left: &[u8],
        right: &[u8],
    ) -> NovaResult<()> {
        let l = cs.alloc_public("left");
        let r = cs.alloc_public("right");
        let witness = cs.alloc_private("equality_witness");

        cs.enforce_equal(l, r, Some("equality_check"));

        Ok(())
    }

    fn convert_custom(
        &self,
        cs: &mut R1CSConstraintSystem,
        type_id: &str,
        data: &[u8],
    ) -> NovaResult<()> {
        // Custom statements need custom conversion logic
        // For now, we just create placeholder constraints
        let input = cs.alloc_public("custom_input");
        let witness = cs.alloc_private("custom_witness");
        let output = cs.alloc_private("custom_output");

        cs.enforce_mul(input, witness, output, Some("custom_constraint"));

        Ok(())
    }
}

impl Default for R1CSConverter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::statement::HashFunction;

    #[test]
    fn test_r1cs_constraint_system_creation() {
        let mut cs = R1CSConstraintSystem::new(NovaSecurityLevel::Bit128);
        
        let a = cs.alloc_public("input_a");
        let b = cs.alloc_private("witness_b");
        let c = cs.alloc_private("output_c");

        cs.enforce_mul(a, b, c, Some("test_mul"));

        assert_eq!(cs.num_public_inputs, 1);
        assert_eq!(cs.num_witness, 2);
        assert_eq!(cs.num_constraints(), 1);
        assert!(cs.validate().is_ok());
    }

    #[test]
    fn test_linear_combination() {
        let mut lc = LinearCombination::zero();
        assert!(lc.is_empty());

        lc.add_term(0, vec![1]);
        lc.add_term(1, vec![2]);
        assert_eq!(lc.len(), 2);
    }

    #[test]
    fn test_r1cs_converter_discrete_log() {
        let converter = R1CSConverter::new();
        
        let statement = Statement {
            statement_type: StatementType::DiscreteLog {
                generator: vec![1, 2, 3],
                public_value: vec![4, 5, 6],
            },
            commitment: None,
            timestamp: 0,
            metadata: None,
        };

        let result = converter.convert_statement(&statement);
        assert!(result.is_ok());
        
        let cs = result.unwrap();
        assert!(cs.num_public_inputs >= 2); // generator and public value
        assert!(cs.num_constraints() > 0);
    }

    #[test]
    fn test_r1cs_converter_range() {
        let converter = R1CSConverter::new();
        
        let statement = Statement {
            statement_type: StatementType::Range {
                min: 0,
                max: 100,
                commitment: vec![1, 2, 3, 4],
            },
            commitment: None,
            timestamp: 0,
            metadata: None,
        };

        let result = converter.convert_statement(&statement);
        assert!(result.is_ok());
        
        let cs = result.unwrap();
        // Should have many constraints for bit decomposition
        assert!(cs.num_constraints() > 0);
    }

    #[test]
    fn test_r1cs_variable_creation() {
        let public = R1CSVariable::public(0, "test_public");
        assert!(public.is_public);
        assert_eq!(public.bit_width, 256);

        let private = R1CSVariable::private(1, "test_private")
            .with_bit_width(64);
        assert!(!private.is_public);
        assert_eq!(private.bit_width, 64);
    }
}
