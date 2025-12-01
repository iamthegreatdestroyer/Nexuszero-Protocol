//! Modular prover architecture
//!
//! This module provides a trait-based prover system that allows
//! different proof systems to be plugged in seamlessly.

use crate::proof::{Statement, Witness, Proof};
use crate::{CryptoError, CryptoResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for prover execution
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverConfig {
    /// Security level
    pub security_level: crate::SecurityLevel,
    /// Optimization settings
    pub optimizations: HashMap<String, serde_json::Value>,
    /// Backend-specific parameters
    pub backend_params: HashMap<String, serde_json::Value>,
}

/// Abstract prover trait
#[async_trait]
pub trait Prover: Send + Sync {
    /// Prover identifier
    fn id(&self) -> &str;

    /// Supported statement types
    fn supported_statements(&self) -> Vec<crate::proof::StatementType>;

    /// Generate a proof for the given statement and witness
    async fn prove(&self, statement: &Statement, witness: &Witness, config: &ProverConfig) -> CryptoResult<Proof>;

    /// Batch proof generation
    async fn prove_batch(&self, statements: &[Statement], witnesses: &[Witness], config: &ProverConfig) -> CryptoResult<Vec<Proof>>;

    /// Get prover capabilities and metadata
    fn capabilities(&self) -> ProverCapabilities;
}

/// Prover capabilities and metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProverCapabilities {
    /// Maximum proof size in bytes
    pub max_proof_size: usize,
    /// Average proving time in milliseconds
    pub avg_proving_time_ms: u64,
    /// Trusted setup required
    pub trusted_setup_required: bool,
    /// Zero-knowledge guarantee level
    pub zk_guarantee: ZKGuarantee,
    /// Supported optimizations
    pub supported_optimizations: Vec<String>,
}

/// Zero-knowledge guarantee levels
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ZKGuarantee {
    /// Perfect zero-knowledge
    Perfect,
    /// Computational zero-knowledge
    Computational,
    /// Statistical zero-knowledge
    Statistical,
    /// Honest-verifier zero-knowledge
    HonestVerifier,
}

/// Strategy pattern for proof generation
pub enum ProofStrategy {
    /// Direct proof generation
    Direct(DirectProver),
    /// Circuit-based proof generation
    Circuit(CircuitProver),
    /// Recursive proof composition
    Recursive(RecursiveProver),
}

/// Direct prover implementation (current approach)
pub struct DirectProver;

#[async_trait]
impl Prover for DirectProver {
    fn id(&self) -> &str { "direct" }

    fn supported_statements(&self) -> Vec<crate::proof::StatementType> {
        use crate::proof::StatementType::*;
        vec![
            DiscreteLog { generator: vec![], public_value: vec![] },
            Preimage { hash_function: crate::proof::statement::HashFunction::SHA3_256, hash_output: vec![] },
            Range { min: 0, max: 0, commitment: vec![] },
        ]
    }

    async fn prove(&self, statement: &Statement, witness: &Witness, _config: &ProverConfig) -> CryptoResult<Proof> {
        crate::proof::proof::prove(statement, witness)
    }

    async fn prove_batch(&self, statements: &[Statement], witnesses: &[Witness], _config: &ProverConfig) -> CryptoResult<Vec<Proof>> {
        let statements_and_witnesses: Vec<(Statement, Witness)> = statements.iter()
            .zip(witnesses.iter())
            .map(|(s, w)| (s.clone(), (*w).clone()))
            .collect();
        crate::proof::proof::prove_batch(&statements_and_witnesses)
    }

    fn capabilities(&self) -> ProverCapabilities {
        ProverCapabilities {
            max_proof_size: 2048,
            avg_proving_time_ms: 50,
            trusted_setup_required: false,
            zk_guarantee: ZKGuarantee::Perfect,
            supported_optimizations: vec!["batch".to_string(), "constant-time".to_string()],
        }
    }
}

/// Circuit-based prover (future extension)
pub struct CircuitProver {
    engine: crate::proof::circuit::CircuitEngine,
}

#[async_trait]
impl Prover for CircuitProver {
    fn id(&self) -> &str { "circuit" }

    fn supported_statements(&self) -> Vec<crate::proof::StatementType> {
        // Support all statement types through circuit composition
        vec![] // Would be populated based on registered circuits
    }

    async fn prove(&self, statement: &Statement, witness: &Witness, _config: &ProverConfig) -> CryptoResult<Proof> {
        // Convert statement to circuit ID
        let circuit_id = self.statement_to_circuit_id(statement)?;
        let inputs = self.extract_inputs(statement, witness)?;

        self.engine.prove(&circuit_id, inputs).await
    }

    async fn prove_batch(&self, _statements: &[Statement], _witnesses: &[Witness], _config: &ProverConfig) -> CryptoResult<Vec<Proof>> {
        Err(CryptoError::NotImplemented("Circuit batch proving not implemented".to_string()))
    }

    fn capabilities(&self) -> ProverCapabilities {
        ProverCapabilities {
            max_proof_size: 4096,
            avg_proving_time_ms: 200,
            trusted_setup_required: true,
            zk_guarantee: ZKGuarantee::Computational,
            supported_optimizations: vec!["circuit-optimization".to_string(), "parallel".to_string()],
        }
    }
}

impl CircuitProver {
    fn statement_to_circuit_id(&self, _statement: &Statement) -> CryptoResult<String> {
        Err(CryptoError::NotImplemented("Circuit mapping not implemented".to_string()))
    }

    fn extract_inputs(&self, _statement: &Statement, _witness: &Witness) -> CryptoResult<HashMap<String, Vec<u8>>> {
        Err(CryptoError::NotImplemented("Input extraction not implemented".to_string()))
    }
}

/// Recursive prover for proof composition
pub struct RecursiveProver {
    base_prover: Box<dyn Prover>,
}

#[async_trait]
impl Prover for RecursiveProver {
    fn id(&self) -> &str { "recursive" }

    fn supported_statements(&self) -> Vec<crate::proof::StatementType> {
        self.base_prover.supported_statements()
    }

    async fn prove(&self, statement: &Statement, witness: &Witness, config: &ProverConfig) -> CryptoResult<Proof> {
        // Generate base proof
        let _base_proof = self.base_prover.prove(statement, witness, config).await?;

        // Generate proof of correct proof generation (meta-proof)
        // This would prove that the base proof was generated correctly
        Err(CryptoError::NotImplemented("Recursive proving not implemented".to_string()))
    }

    async fn prove_batch(&self, statements: &[Statement], witnesses: &[Witness], config: &ProverConfig) -> CryptoResult<Vec<Proof>> {
        self.base_prover.prove_batch(statements, witnesses, config).await
    }

    fn capabilities(&self) -> ProverCapabilities {
        let mut base_caps = self.base_prover.capabilities();
        base_caps.zk_guarantee = ZKGuarantee::Computational; // Downgraded due to recursion
        base_caps.supported_optimizations.push("recursion".to_string());
        base_caps
    }
}

/// Prover registry for dynamic loading
pub struct ProverRegistry {
    provers: HashMap<String, Box<dyn Prover>>,
}

impl ProverRegistry {
    pub fn new() -> Self {
        Self {
            provers: HashMap::new(),
        }
    }

    pub fn register(&mut self, prover: Box<dyn Prover>) {
        self.provers.insert(prover.id().to_string(), prover);
    }

    pub fn get(&self, id: &str) -> Option<&Box<dyn Prover>> {
        self.provers.get(id)
    }

    pub fn list(&self) -> Vec<String> {
        self.provers.keys().cloned().collect()
    }
}