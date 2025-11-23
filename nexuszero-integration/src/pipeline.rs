use serde::{Deserialize, Serialize};
use nexuszero_crypto::proof::{Statement, Witness, Proof};
use nexuszero_crypto::proof::proof::{prove, verify};
use nexuszero_crypto::{CryptoParameters, SecurityLevel};
use nexuszero_holographic::MPS; // re-exported from holographic crate
use crate::config::ProtocolConfig;

/// Trait for parameter optimizers (stub for future neural implementation).
pub trait ParameterOptimizer {
    fn predict_parameters(&self, _statement: &Statement) -> CryptoParameters;
}

/// Static optimizer placeholder returning parameters derived from security level.
pub struct StaticOptimizer {
    pub level: SecurityLevel,
}

impl ParameterOptimizer for StaticOptimizer {
    fn predict_parameters(&self, _statement: &Statement) -> CryptoParameters {
        CryptoParameters::from_security_level(self.level)
    }
}

/// Complete proof pipeline with optional optimization and compression.
pub struct NexuszeroProtocol {
    optimizer: Option<Box<dyn ParameterOptimizer + Send + Sync>>,
    pub config: ProtocolConfig,
}

impl NexuszeroProtocol {
    pub fn new(config: ProtocolConfig) -> Self {
        Self { optimizer: None, config }
    }

    /// Generate optimized, optionally compressed proof.
    pub fn generate_proof(&mut self, statement: &Statement, witness: &Witness) -> Result<OptimizedProof, ProtocolError> {
        // STEP 1: Select parameters (currently not used by prove(), stored for metadata only)
        let params = if self.config.use_optimizer {
            if self.optimizer.is_none() {
                self.optimizer = Some(Box::new(StaticOptimizer { level: self.config.security_level }));
            }
            self.optimizer.as_ref().unwrap().predict_parameters(statement)
        } else {
            CryptoParameters::from_security_level(self.config.security_level)
        };

        // STEP 2: Generate base proof (crypto crate prove currently ignores params)
        let start = std::time::Instant::now();
        let base_proof = prove(statement, witness).map_err(|e| ProtocolError::ProofGenerationFailed(e.to_string()))?;
        let gen_time = start.elapsed().as_secs_f64() * 1000.0;

        // STEP 3: Compression (optional)
        let compressed = if self.config.use_compression {
            let proof_bytes = base_proof.to_bytes().map_err(|e| ProtocolError::ProofGenerationFailed(e.to_string()))?;
            // Simple heuristic: bond dimension capped at 16
            match MPS::from_proof_data(&proof_bytes, 16) {
                Ok(mps) => Some(mps),
                Err(e) => return Err(ProtocolError::CompressionFailed(format!("MPS error: {e}"))),
            }
        } else { None };

        // STEP 4: Metrics
        let compression_ratio = compressed.as_ref().map(|m| m.compression_ratio()).unwrap_or(1.0);
        let proof_size_bytes = base_proof.size();

        Ok(OptimizedProof {
            statement: statement.clone(),
            base_proof,
            compressed,
            params,
            metrics: ProofMetrics {
                generation_time_ms: gen_time,
                proof_size_bytes,
                compression_ratio,
            },
        })
    }

    /// Verify proof directly (compressed path uses boundary verification shortcut).
    pub fn verify_proof(&self, optimized: &OptimizedProof) -> Result<bool, ProtocolError> {
        if let Some(ref mps) = optimized.compressed {
            let boundary = extract_boundary(&optimized.statement);
            Ok(mps.verify_boundary(&boundary))
        } else {
            verify(&optimized.statement, &optimized.base_proof)
                .map(|_| true)
                .map_err(|e| ProtocolError::VerificationFailed(e.to_string()))
        }
    }
}

/// Derive boundary vector from statement hash for compressed verification stub.
fn extract_boundary(statement: &Statement) -> Vec<f64> {
    let hash = statement.hash().unwrap_or([0u8;32]);
    hash.iter().map(|b| (*b as f64) / 255.0).collect()
}

/// Optimized proof bundle combining base and compressed forms.
#[derive(Clone, Serialize, Deserialize)]
pub struct OptimizedProof {
    pub statement: Statement,
    pub base_proof: Proof,
    pub compressed: Option<MPS>,
    pub params: CryptoParameters,
    pub metrics: ProofMetrics,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetrics {
    pub generation_time_ms: f64,
    pub proof_size_bytes: usize,
    pub compression_ratio: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("Proof generation failed: {0}")] ProofGenerationFailed(String),
    #[error("Verification failed: {0}")] VerificationFailed(String),
    #[error("Optimization failed: {0}")] OptimizationFailed(String),
    #[error("Compression failed: {0}")] CompressionFailed(String),
}
