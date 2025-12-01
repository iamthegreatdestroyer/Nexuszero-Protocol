//! Schnorr proof plugin implementation
//!
//! This module provides a plugin implementation for Schnorr-style sigma protocol proofs,
//! including discrete log and preimage proofs.

use crate::proof::plugins::{
    ProofType, SetupParams, VerificationKey, ProverKey, CircuitInfo
};
use crate::proof::{Statement, Witness, Proof, StatementType};
use crate::{CryptoResult};
use std::collections::HashMap;

/// Schnorr proof plugin implementation
#[derive(Debug, Clone)]
pub struct SchnorrPlugin;

impl SchnorrPlugin {
    /// Create a new Schnorr plugin instance
    pub fn new() -> Self {
        Self
    }

    /// Get the proof type for this plugin
    pub fn proof_type(&self) -> ProofType {
        ProofType::Schnorr
    }

    /// Get the name for this plugin
    pub fn name(&self) -> &'static str {
        "Schnorr Sigma Protocol"
    }

    /// Get the version for this plugin
    pub fn version(&self) -> &'static str {
        "1.0.0"
    }

    /// Get supported statements for this plugin
    pub fn supported_statements(&self) -> Vec<StatementType> {
        vec![
            StatementType::DiscreteLog {
                generator: vec![],
                public_value: vec![],
            },
            StatementType::Preimage {
                hash_function: crate::proof::statement::HashFunction::SHA256,
                hash_output: vec![],
            },
        ]
    }

    /// Setup the proof system
    pub async fn setup(&self, _params: &SetupParams) -> CryptoResult<(ProverKey, VerificationKey)> {
        // Schnorr proofs don't require trusted setup - keys are derived from parameters
        let prover_key = ProverKey {
            data: vec![], // No special prover key needed
            key_type: "schnorr".to_string(),
            proof_type: ProofType::Schnorr,
        };

        let verification_key = VerificationKey {
            data: vec![], // No special verification key needed
            key_type: "schnorr".to_string(),
            proof_type: ProofType::Schnorr,
        };

        Ok((prover_key, verification_key))
    }

    /// Generate a proof
    pub async fn prove(&self, statement: &Statement, witness: &Witness, _prover_key: &ProverKey) -> CryptoResult<Proof> {
        // Use existing proof generation logic
        crate::proof::proof::prove(statement, witness)
    }

    /// Verify a proof
    pub async fn verify(&self, statement: &Statement, proof: &Proof, _verification_key: &VerificationKey) -> CryptoResult<bool> {
        // Use existing proof verification logic
        match crate::proof::proof::verify(statement, proof) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Serialize the plugin
    pub fn serialize(&self) -> CryptoResult<Vec<u8>> {
        // Schnorr plugin has no internal state to serialize
        Ok(vec![])
    }

    /// Get circuit info
    pub fn circuit_info(&self, statement: &Statement) -> CircuitInfo {
        let (constraints, variables) = match &statement.statement_type {
            StatementType::DiscreteLog { .. } => (3, 2), // Basic sigma protocol
            StatementType::Preimage { .. } => (5, 3),   // Hash-based protocol
            _ => (1, 1), // Fallback
        };

        CircuitInfo {
            constraints,
            variables,
            proof_size_bytes: 128, // Typical Schnorr proof size
            verification_time_ms: 1, // Very fast verification
            metadata: HashMap::from([
                ("protocol".to_string(), "Sigma".to_string()),
                ("interactive".to_string(), "false".to_string()),
            ]),
        }
    }
}

impl Default for SchnorrPlugin {
    fn default() -> Self {
        Self::new()
    }
}
