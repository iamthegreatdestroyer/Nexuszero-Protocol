use crate::pipeline::{NexuszeroProtocol, OptimizedProof, ProtocolError};
use crate::config::ProtocolConfig;
use nexuszero_crypto::proof::{StatementBuilder, Statement, Witness};
use nexuszero_crypto::proof::statement::HashFunction;

/// High-level API facade for protocol usage.
pub struct NexuszeroAPI {
    protocol: NexuszeroProtocol,
}

impl NexuszeroAPI {
    /// Initialize with default configuration.
    pub fn new() -> Self { Self { protocol: NexuszeroProtocol::new(ProtocolConfig::default()) } }
    /// Initialize with custom configuration.
    pub fn with_config(config: ProtocolConfig) -> Self { Self { protocol: NexuszeroProtocol::new(config) } }

    /// Generate discrete log proof (placeholder mapping).
    pub fn prove_discrete_log(&mut self, generator: &[u8], public_value: &[u8], secret_exponent: &[u8]) -> Result<OptimizedProof, ProtocolError> {
        let statement = StatementBuilder::new().discrete_log(generator.to_vec(), public_value.to_vec()).build().map_err(|e| ProtocolError::ProofGenerationFailed(e.to_string()))?;
        let witness = Witness::discrete_log(secret_exponent.to_vec());
        self.protocol.generate_proof(&statement, &witness)
    }

    /// Generate hash preimage proof.
    pub fn prove_preimage(&mut self, hash_function: HashFunction, hash_output: &[u8], preimage: &[u8]) -> Result<OptimizedProof, ProtocolError> {
        let statement = StatementBuilder::new().preimage(hash_function, hash_output.to_vec()).build().map_err(|e| ProtocolError::ProofGenerationFailed(e.to_string()))?;
        let witness = Witness::preimage(preimage.to_vec());
        self.protocol.generate_proof(&statement, &witness)
    }

    /// Verify optimized proof (compressed or uncompressed path).
    pub fn verify(&self, proof: &OptimizedProof) -> Result<bool, ProtocolError> { self.protocol.verify_proof(proof) }

    /// Retrieve metrics from proof.
    pub fn get_metrics(&self, proof: &OptimizedProof) -> crate::pipeline::ProofMetrics { proof.metrics.clone() }
}
