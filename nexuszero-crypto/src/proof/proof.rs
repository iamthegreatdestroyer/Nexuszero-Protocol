//! Proof generation and verification
//!
//! This module implements the core prove/verify algorithms.

use crate::proof::{Statement, Witness, Prover, Verifier, ProverConfig, VerifierConfig, ProverRegistry, VerifierRegistry, ProverCapabilities, VerifierCapabilities};
use crate::proof::prover::ZKGuarantee;
use crate::proof::verifier::VerificationGuarantee;
use crate::{CryptoError, CryptoResult};
use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use std::sync::Arc;

#[cfg(feature = "lazy_static")]
use lazy_static::lazy_static;

#[cfg(feature = "lazy_static")]
lazy_static! {
    /// Global prover registry instance
    pub static ref GLOBAL_PROVER_REGISTRY: Arc<ProverRegistry> = {
        let mut registry = ProverRegistry::new();
        registry.register(Box::new(LegacyProver));

        // Register hardware-accelerated provers if available
        #[cfg(feature = "hardware-acceleration")]
        {
            registry.register(Box::new(crate::proof::hardware_acceleration::HardwareProver::new(
                crate::proof::hardware_acceleration::HardwareType::GPU,
            )));
            registry.register(Box::new(crate::proof::hardware_acceleration::HardwareProver::new(
                crate::proof::hardware_acceleration::HardwareType::TPU,
            )));
        }

        Arc::new(registry)
    };

    /// Global verifier registry instance
    pub static ref GLOBAL_VERIFIER_REGISTRY: Arc<VerifierRegistry> = {
        let mut registry = VerifierRegistry::new();
        registry.register(Box::new(LegacyVerifier));

        // Register hardware-accelerated verifiers if available
        #[cfg(feature = "gpu")]
        {
            // Note: GPU verifier registration would happen asynchronously
            // For now, we register a placeholder that will be replaced at runtime
            registry.register(Box::new(crate::proof::verifier::HardwareVerifier::new(
                crate::proof::verifier::HardwareType::GPU,
            )));
        }

        #[cfg(feature = "tpu")]
        {
            registry.register(Box::new(crate::proof::verifier::HardwareVerifier::new(
                crate::proof::verifier::HardwareType::TPU,
            )));
        }

        Arc::new(registry)
    };
}

/// A zero-knowledge proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Proof {
    /// Commitment phase values
    pub commitments: Vec<Commitment>,
    /// Challenge from Fiat-Shamir transform
    pub challenge: Challenge,
    /// Response phase values
    pub responses: Vec<Response>,
    /// Proof metadata
    pub metadata: ProofMetadata,
    /// Bulletproof range proof (if applicable). Always serialized to preserve
    /// positional field ordering for bincode. (Using skip_serializing_if with
    /// bincode causes deserialization EOF due to missing field bytes.)
    pub bulletproof: Option<crate::proof::bulletproofs::BulletproofRangeProof>,
}

/// Commitment in the proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Commitment {
    /// Commitment value
    pub value: Vec<u8>,
}

/// Challenge value
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Challenge {
    /// Challenge bytes - expanded to 64 bytes for enhanced security
    pub value: Vec<u8>,
}

/// Response in the proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Response {
    /// Response value
    pub value: Vec<u8>,
}

/// Proof metadata - optimized for minimal size
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Proof system version
    pub version: u8,
    /// Timestamp of generation (set to 0 for size optimization)
    pub timestamp: u64,
    /// Size in bytes (computed dynamically, serialized as 0 to save space)
    pub size: usize,
}

impl Proof {
    /// Validate proof structure
    pub fn validate(&self) -> CryptoResult<()> {
        if self.commitments.is_empty() {
            return Err(CryptoError::ProofError("No commitments".to_string()));
        }
        if self.responses.is_empty() {
            return Err(CryptoError::ProofError("No responses".to_string()));
        }
        // Basic structural validation: ensure sizes are within reasonable limits
        // to guard against malformed or malicious proofs.
        for c in &self.commitments {
            if c.value.is_empty() || c.value.len() > 1024 {
                return Err(CryptoError::ProofError("Invalid commitment size".to_string()));
            }
        }
        for r in &self.responses {
            if r.value.is_empty() || r.value.len() > 512 {
                return Err(CryptoError::ProofError("Invalid response size".to_string()));
            }
        }
        Ok(())
    }

    /// Serialize proof to bytes
    pub fn to_bytes(&self) -> CryptoResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to serialize proof: {}", e))
        })
    }

    /// Deserialize proof from bytes
    pub fn from_bytes(bytes: &[u8]) -> CryptoResult<Self> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to deserialize proof: {}", e))
        })
    }

    /// Get proof size (computed from serialization)
    pub fn size(&self) -> usize {
        // Size is computed on-demand from serialization
        self.to_bytes().map(|b| b.len()).unwrap_or(0)
    }
}

// ============================================================================
// Legacy Prover/Verifier Implementations
// ============================================================================

/// Legacy prover that wraps the existing monolithic prove function
pub struct LegacyProver;

#[async_trait]
impl Prover for LegacyProver {
    fn id(&self) -> &str {
        "legacy"
    }

    fn supported_statements(&self) -> Vec<crate::proof::StatementType> {
        use crate::proof::StatementType::*;
        vec![
            DiscreteLog { generator: vec![], public_value: vec![] },
            Preimage { hash_function: HashFunction::SHA256, hash_output: vec![] },
            Range { min: 0, max: 0, commitment: vec![] },
        ]
    }

    async fn prove(&self, statement: &Statement, witness: &Witness, config: &ProverConfig) -> CryptoResult<Proof> { 
        prove(statement, witness)
    }

    async fn prove_batch(&self, statements: &[Statement], witnesses: &[Witness], config: &ProverConfig) -> CryptoResult<Vec<Proof>> { 
        // Convert to the format expected by the existing prove_batch function
        let statements_and_witnesses: Vec<(Statement, Witness)> = statements.iter()
            .zip(witnesses.iter())
            .map(|(s, w)| (s.clone(), w.clone()))
            .collect();
        prove_batch(&statements_and_witnesses)
    }    fn capabilities(&self) -> crate::proof::ProverCapabilities {
        crate::proof::ProverCapabilities {
            max_proof_size: 16384,
            avg_proving_time_ms: 10,
            trusted_setup_required: false,
            zk_guarantee: ZKGuarantee::Computational,
            supported_optimizations: vec!["parallel".to_string()],
        }
    }
}

/// Legacy verifier that wraps the existing monolithic verify function
pub struct LegacyVerifier;

#[async_trait]
impl Verifier for LegacyVerifier {
    fn id(&self) -> &str {
        "legacy"
    }

    fn supported_statements(&self) -> Vec<crate::proof::StatementType> {
        use crate::proof::StatementType::*;
        vec![
            DiscreteLog { generator: vec![], public_value: vec![] },
            Preimage { hash_function: HashFunction::SHA256, hash_output: vec![] },
            Range { min: 0, max: 0, commitment: vec![] },
        ]
    }

    async fn verify(&self, statement: &Statement, proof: &Proof, config: &VerifierConfig) -> CryptoResult<bool> {
        verify(statement, proof).map(|_| true)
    }

    async fn verify_batch(&self, statements: &[Statement], proofs: &[Proof], config: &VerifierConfig) -> CryptoResult<Vec<bool>> {
        // Convert to the format expected by the existing verify_batch function
        let statements_and_proofs: Vec<(Statement, Proof)> = statements.iter()
            .zip(proofs.iter())
            .map(|(s, p)| (s.clone(), p.clone()))
            .collect();
        
        // Call verify_batch and convert the result
        verify_batch(&statements_and_proofs)?;
        Ok(vec![true; statements.len()]) // All verifications succeeded
    }

    fn capabilities(&self) -> crate::proof::VerifierCapabilities {
        crate::proof::VerifierCapabilities {
            max_proof_size: 16384,
            avg_verification_time_ms: 5,
            trusted_setup_required: false,
            verification_guarantee: VerificationGuarantee::Computational,
            supported_optimizations: vec!["parallel".to_string()],
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

use rand::Rng;
use num_bigint::BigUint;
use crate::proof::statement::{HashFunction, StatementType};

/// Generate random blinding factors for commitments
fn generate_blinding_factors(count: usize, size: usize) -> Vec<Vec<u8>> {
    let mut rng = rand::thread_rng();
    (0..count)
        .map(|_| (0..size).map(|_| rng.gen::<u8>()).collect())
        .collect()
}

/// Convert challenge to scalar for arithmetic
fn challenge_to_bigint(challenge: &[u8]) -> BigUint {
    BigUint::from_bytes_be(challenge)
}

/// Modular addition: (a + b) mod m, padded to 32 bytes
#[allow(dead_code)]
fn add_mod(a: &[u8], b: &[u8], modulus: &BigUint) -> Vec<u8> {
    let a_big = BigUint::from_bytes_be(a);
    let b_big = BigUint::from_bytes_be(b);
    let result = (a_big + b_big) % modulus;
    let mut bytes = result.to_bytes_be();
    
    // Pad to 32 bytes if needed
    while bytes.len() < 32 {
        bytes.insert(0, 0);
    }
    bytes
}

/// Modular multiplication: (a * b) mod m, padded to 32 bytes
#[allow(dead_code)]
fn mul_mod(a: &BigUint, b: &[u8], modulus: &BigUint) -> Vec<u8> {
    let b_big = BigUint::from_bytes_be(b);
    let result = (a * b_big) % modulus;
    let mut bytes = result.to_bytes_be();
    
    // Pad to 32 bytes if needed
    while bytes.len() < 32 {
        bytes.insert(0, 0);
    }
    bytes
}

/// Modular exponentiation: base^exp mod modulus (constant-time)
fn mod_exp(base: &[u8], exp: &[u8], modulus: &[u8]) -> Vec<u8> {
    use crate::utils::ct_modpow;
    
    let base_big = BigUint::from_bytes_be(base);
    let exp_big = BigUint::from_bytes_be(exp);
    let mod_big = BigUint::from_bytes_be(modulus);
    
    let result = ct_modpow(&base_big, &exp_big, &mod_big);
    result.to_bytes_be()
}

// ============================================================================
// Commitment Phase Functions
// ============================================================================

/// Commit for discrete log proof: t = g^r
fn commit_discrete_log(generator: &[u8], blinding: &[u8]) -> CryptoResult<Commitment> {
    // Use a simple modulus for demonstration (in production, use proper group)
    let modulus_bytes = vec![0xFF; 32]; // 2^256 - 1 approximation
    
    let t = mod_exp(generator, blinding, &modulus_bytes);
    
    Ok(Commitment { value: t })
}

/// Commit for preimage proof: commitment to randomness
fn commit_preimage(blinding: &[u8]) -> CryptoResult<Commitment> {
    use sha3::{Digest, Sha3_256};
    
    // Commit to blinding factor
    let mut hasher = Sha3_256::new();
    hasher.update(blinding);
    let commitment = hasher.finalize().to_vec();
    
    Ok(Commitment { value: commitment })
}

/// Pedersen commitment for range proof: C = g^v * h^r (constant-time)
fn commit_range(value: u64, blinding_r: &[u8], generator_g: &[u8], generator_h: &[u8]) -> CryptoResult<Commitment> {
    use num_bigint::BigUint;
    use crate::utils::constant_time::ct_modpow;
    
    let modulus_bytes = vec![0xFF; 32];
    let mod_big = BigUint::from_bytes_be(&modulus_bytes);
    
    // g^v (constant-time)
    let g_big = BigUint::from_bytes_be(generator_g);
    let v_bytes = value.to_be_bytes();
    let v_big = BigUint::from_bytes_be(&v_bytes);
    let g_v = ct_modpow(&g_big, &v_big, &mod_big);
    
    // h^r (constant-time)
    let h_big = BigUint::from_bytes_be(generator_h);
    let r_big = BigUint::from_bytes_be(blinding_r);
    let h_r = ct_modpow(&h_big, &r_big, &mod_big);
    
    // C = g^v * h^r (mod p)
    let commitment = (g_v * h_r) % &mod_big;
    
    Ok(Commitment { value: commitment.to_bytes_be() })
}

// ============================================================================
// Response Phase Functions
// ============================================================================

/// Compute response for discrete log: s = r + c*x (no modulus, let it be large)
fn compute_discrete_log_response(
    secret: &[u8],
    blinding: &[u8],
    challenge: &[u8],
) -> CryptoResult<Response> {
    let c = challenge_to_bigint(challenge);
    let r = BigUint::from_bytes_be(blinding);
    let x = BigUint::from_bytes_be(secret);
    
    // s = r + c*x (no modular reduction - let response be arbitrary size)
    let s = r + (c * x);
    
    Ok(Response { value: s.to_bytes_be() })
}

/// Compute response for preimage: reveal blinding XOR challenge
fn compute_preimage_response(
    blinding: &[u8],
    challenge: &[u8],
) -> CryptoResult<Response> {
    // Simple response: blinding XOR first bytes of challenge
    let mut response = blinding.to_vec();
    for (i, byte) in response.iter_mut().enumerate() {
        if i < challenge.len() {
            *byte ^= challenge[i];
        }
    }
    
    Ok(Response { value: response })
}

/// Compute response for range proof: s = r_blinding + c*r_witness
fn compute_range_response(
    witness_blinding: &[u8],
    commitment_blinding: &[u8],
    challenge: &[u8],
) -> CryptoResult<Response> {
    use num_bigint::BigUint;
    
    let c = challenge_to_bigint(challenge);
    let r_commit = BigUint::from_bytes_be(commitment_blinding);
    let r_witness = BigUint::from_bytes_be(witness_blinding);
    
    // s = r_commit + c * r_witness
    let s = r_commit + (c * r_witness);
    
    Ok(Response { value: s.to_bytes_be() })
}

// ============================================================================
// Verification Functions
// ============================================================================

/// Verify discrete log proof: check g^s = t * h^c (constant-time)
fn verify_discrete_log_proof(
    generator: &[u8],
    public_value: &[u8],
    commitment: &Commitment,
    challenge: &Challenge,
    response: &Response,
) -> CryptoResult<()> {
    use crate::utils::constant_time::ct_modpow;
    
    let modulus_bytes = vec![0xFF; 32];
    let mod_big = BigUint::from_bytes_be(&modulus_bytes);
    
    // Compute g^s (mod p) - constant-time
    let gs_big = {
        let gen_big = BigUint::from_bytes_be(generator);
        let response_big = BigUint::from_bytes_be(&response.value);
        ct_modpow(&gen_big, &response_big, &mod_big)
    };
    
    // Compute h^c (mod p) - constant-time
    let hc_big = {
        let h_big = BigUint::from_bytes_be(public_value);
        let c_big = BigUint::from_bytes_be(&challenge.value);
        ct_modpow(&h_big, &c_big, &mod_big)
    };
    
    // Compute t * h^c (mod p)
    let right_side = {
        let t_big = BigUint::from_bytes_be(&commitment.value);
        (t_big * hc_big) % &mod_big
    };
    
    // Verify g^s = t * h^c (mod p)
    if gs_big == right_side {
        Ok(())
    } else {
        Err(CryptoError::VerificationError(
            "Discrete log proof verification failed".to_string(),
        ))
    }
}

/// Verify preimage proof
fn verify_preimage_proof(
    hash_function: &HashFunction,
    hash_output: &[u8],
    commitment: &Commitment,
    challenge: &Challenge,
    response: &Response,
) -> CryptoResult<()> {
    use sha3::{Digest, Sha3_256};
    use sha2::Sha256;
    
    // Recompute blinding from response XOR challenge
    let mut blinding = response.value.clone();
    for (i, byte) in blinding.iter_mut().enumerate() {
        if i < challenge.value.len() {
            *byte ^= challenge.value[i];
        }
    }
    
    // Verify commitment matches
    let recomputed_commitment = match hash_function {
        HashFunction::SHA3_256 => {
            let mut hasher = Sha3_256::new();
            hasher.update(&blinding);
            hasher.finalize().to_vec()
        }
        HashFunction::SHA256 => {
            let mut hasher = Sha256::new();
            hasher.update(&blinding);
            hasher.finalize().to_vec()
        }
        HashFunction::Blake3 => {
            // Blake3 not yet implemented
            return Err(CryptoError::VerificationError(
                "Blake3 hash function not yet supported".to_string(),
            ));
        }
    };
    
    if recomputed_commitment != commitment.value {
        return Err(CryptoError::VerificationError(
            "Preimage commitment verification failed".to_string(),
        ));
    }
    
    // Note: In a real preimage proof, we'd verify the actual preimage
    // This is a simplified version showing the structure
    
    // Verify hash output matches (in real implementation, witness would be revealed selectively)
    if hash_output.is_empty() {
        return Err(CryptoError::VerificationError(
            "Invalid hash output".to_string(),
        ));
    }
    
    Ok(())
}

// ============================================================================
// Main Proof Generation Function
// ============================================================================

/// Generate a zero-knowledge proof
/// 
/// # Arguments
/// * `statement` - The public statement being proven
/// * `witness` - The secret knowledge (NOT transmitted)
/// 
/// # Returns
/// A proof that can be publicly verified
/// 
/// # Implementation
/// Uses Schnorr-style protocol with Fiat-Shamir transform:
/// 1. Commitment: Generate random blinding factors
/// 2. Challenge: Hash statement + commitments (Fiat-Shamir)
/// 3. Response: Combine witness, blinding, and challenge
pub fn prove(statement: &Statement, witness: &Witness) -> CryptoResult<Proof> {
    // PHASE 1: Validate inputs
    if !witness.satisfies_statement(statement) {
        return Err(CryptoError::ProofError(
            "Witness does not satisfy statement".to_string(),
        ));
    }
    
    // PHASE 2: Commitment Phase
    // Generate random blinding factors (must be kept for response phase)
    let blinding = generate_blinding_factors(1, 32);
    
    let commitments = match &statement.statement_type {
        StatementType::DiscreteLog { generator, .. } => {
            vec![commit_discrete_log(generator, &blinding[0])?]
        }
        StatementType::Preimage { .. } => {
            vec![commit_preimage(&blinding[0])?]
        }
        StatementType::Range { .. } => {
            // For Bulletproofs, commitment is generated within the protocol
            // Placeholder commitment for compatibility - use cryptographically secure generators
            let gen_g = crate::proof::bulletproofs::generator_g().to_bytes_be();
            let gen_h = crate::proof::bulletproofs::generator_h().to_bytes_be();
            vec![commit_range(0, &blinding[0], &gen_g, &gen_h)?]
        }
        StatementType::Custom { .. } => {
            return Err(CryptoError::ProofError(
                "Custom statements not yet supported".to_string(),
            ));
        }
    };
    
    // PHASE 3: Challenge Phase (Fiat-Shamir)
    let challenge = compute_challenge(statement, &commitments)?;
    
    // PHASE 4: Response Phase
    // Use the SAME blinding factors from commitment phase
    let responses = match &statement.statement_type {
        StatementType::DiscreteLog { .. } => {
            // Get secret from witness
            let secret = witness.get_secret_bytes()
                .map_err(|e| CryptoError::ProofError(e.to_string()))?;
            vec![compute_discrete_log_response(&secret, &blinding[0], &challenge.value)?]
        }
        StatementType::Preimage { .. } => {
            vec![compute_preimage_response(&blinding[0], &challenge.value)?]
        }
        StatementType::Range { .. } => {
            // Bulletproofs handles response internally
            vec![compute_range_response(&blinding[0], &blinding[0], &challenge.value)?]
        }
        StatementType::Custom { .. } => {
            return Err(CryptoError::ProofError(
                "Unsupported statement type".to_string(),
            ));
        }
    };
    
    // PHASE 4.5: Generate Bulletproof for Range statements
    let bulletproof = match &statement.statement_type {
        StatementType::Range { min, max, .. } => {
            // Extract value and blinding from witness
            let secret = witness.get_secret_bytes()
                .map_err(|e| CryptoError::ProofError(e.to_string()))?;
            
            // First 8 bytes are value
            let mut value_bytes = [0u8; 8];
            value_bytes.copy_from_slice(&secret[0..8]);
            let value = u64::from_be_bytes(value_bytes);
            
            // Rest is blinding
            let witness_blinding = if secret.len() > 8 { &secret[8..] } else { &blinding[0] };
            
            // Compute number of bits needed for range
            let range_size = max - min;
            let num_bits = if range_size == 0 {
                1
            } else {
                64 - range_size.leading_zeros() as usize
            };
            
            // Generate Bulletproof directly on the actual value so that the
            // Bulletproof commitment matches the statement commitment provided.
            // NOTE: This does NOT enforce the lower bound (min) inside the
            // Bulletproof itself; min/max are enforced by the statement/witness
            // satisfaction check. Future improvement: normalize (value - min)
            // and adjust verification to reapply the offset securely.
            if *min == 0 {
                Some(crate::proof::bulletproofs::prove_range(
                    value,
                    witness_blinding,
                    num_bits,
                )?)
            } else {
                Some(crate::proof::bulletproofs::prove_range_offset(
                    value,
                    *min,
                    witness_blinding,
                    num_bits,
                )?)
            }
        }
        _ => None,
    };
    
    // PHASE 5: Package proof
    let metadata = ProofMetadata {
        version: 1,
        timestamp: 0, // Set to 0 for size optimization
        size: 0, // Size computed dynamically to save space
    };
    
    Ok(Proof {
        commitments,
        challenge,
        responses,
        metadata,
        bulletproof,
    })
}

/// Verify a zero-knowledge proof
/// 
/// # Arguments
/// * `statement` - The public statement
/// * `proof` - The proof to verify
/// 
/// # Returns
/// `Ok(())` if proof is valid, error otherwise
/// 
/// # Implementation
/// 1. Validate proof structure
/// 2. Recompute challenge and verify it matches
/// 3. Verify proof equation (statement-type specific)
pub fn verify(statement: &Statement, proof: &Proof) -> CryptoResult<()> {
    // PHASE 1: Validate proof structure
    proof.validate()?;
    
    // PHASE 2: Recompute challenge
    let recomputed_challenge = compute_challenge(statement, &proof.commitments)?;
    
    // Verify challenge matches (critical for security!)
    if recomputed_challenge.value != proof.challenge.value {
        return Err(CryptoError::VerificationError(
            "Challenge verification failed - possible tampering".to_string(),
        ));
    }
    
    // PHASE 3: Verify responses (statement-type specific)
    match &statement.statement_type {
        StatementType::DiscreteLog {
            generator,
            public_value,
        } => {
            if proof.commitments.is_empty() || proof.responses.is_empty() {
                return Err(CryptoError::VerificationError(
                    "Invalid proof structure".to_string(),
                ));
            }
            
            verify_discrete_log_proof(
                generator,
                public_value,
                &proof.commitments[0],
                &proof.challenge,
                &proof.responses[0],
            )?;
        }
        StatementType::Preimage {
            hash_function,
            hash_output,
        } => {
            if proof.commitments.is_empty() || proof.responses.is_empty() {
                return Err(CryptoError::VerificationError(
                    "Invalid proof structure".to_string(),
                ));
            }
            
            verify_preimage_proof(
                hash_function,
                hash_output,
                &proof.commitments[0],
                &proof.challenge,
                &proof.responses[0],
            )?;
        }
        StatementType::Range { min, max, commitment } => {
            // Use Bulletproofs for cryptographic range enforcement
            if let Some(ref bulletproof) = proof.bulletproof {
                // Compute number of bits for range verification
                let range_size = max - min;
                let num_bits = if range_size == 0 {
                    1
                } else {
                    64 - range_size.leading_zeros() as usize
                };
                
                // Verify Bulletproof
                // Temporarily use unified verification path until offset linkage strengthened.
                crate::proof::bulletproofs::verify_range(
                    bulletproof,
                    commitment,
                    num_bits,
                )?;
            } else {
                // Fallback to simplified verification for backward compatibility (constant-time)
                if proof.commitments.is_empty() || proof.responses.is_empty() {
                    return Err(CryptoError::VerificationError(
                        "Invalid proof structure".to_string(),
                    ));
                }
                
                use num_bigint::BigUint;
                use crate::utils::constant_time::ct_modpow;
                
                let modulus_bytes = vec![0xFF; 32];
                let mod_big = BigUint::from_bytes_be(&modulus_bytes);
                
                let gen_h = crate::proof::bulletproofs::generator_h().to_bytes_be();
                let t_big = BigUint::from_bytes_be(&proof.commitments[0].value);
                let c_big = BigUint::from_bytes_be(commitment);
                let challenge_big = BigUint::from_bytes_be(&proof.challenge.value);
                let c_pow_c = ct_modpow(&c_big, &challenge_big, &mod_big);
                let left_side = (t_big * c_pow_c) % &mod_big;
                
                let h_big = BigUint::from_bytes_be(&gen_h);
                let s_big = BigUint::from_bytes_be(&proof.responses[0].value);
                let right_side = ct_modpow(&h_big, &s_big, &mod_big);
                
                if left_side != right_side {
                    return Err(CryptoError::VerificationError(
                        "Range proof commitment equation failed".to_string(),
                    ));
                }
            }
        }
        StatementType::Custom { .. } => {
            return Err(CryptoError::VerificationError(
                "Custom statements not yet supported".to_string(),
            ));
        }
    }
    
    // PHASE 4: All checks passed
    Ok(())
}

/// Compute Fiat-Shamir challenge
pub fn compute_challenge(statement: &Statement, commitments: &[Commitment]) -> CryptoResult<Challenge> {
    use sha3::{Digest, Sha3_512};

    let mut hasher = Sha3_512::new();

    // Domain separation: Include protocol identifier
    hasher.update(b"NEXUSZERO-ZK-PROOF");
    hasher.update(b"Fiat-Shamir-Challenge-v1");

    // Hash statement
    let stmt_bytes = statement.to_bytes()?;
    hasher.update(&stmt_bytes);

    // Hash all commitments
    for commitment in commitments {
        hasher.update(&commitment.value);
    }

    let hash_output = hasher.finalize();
    let challenge_bytes = hash_output.to_vec();

    Ok(Challenge {
        value: challenge_bytes,
    })
}

// ============================================================================
// Parallel Proof Generation Functions
// ============================================================================

/// Generate multiple proofs in parallel using Rayon
///
/// # Arguments
/// * `statements_and_witnesses` - Vector of (Statement, Witness) tuples
///
/// # Returns
/// Vector of proofs, one for each input pair
///
/// # Performance
/// This function uses Rayon to parallelize proof generation across multiple cores.
/// Expected speedup: >3x on 4-core CPUs for batches of 4+ proofs.
///
/// # Example
/// ```ignore
/// use nexuszero_crypto::proof::{Statement, Witness};
/// use nexuszero_crypto::proof::proof::prove_batch;
/// use nexuszero_crypto::proof::statement::StatementBuilder;
/// 
/// // Create multiple statements and witnesses
/// let statements_and_witnesses = vec![
///     // (statement1, witness1),
///     // (statement2, witness2),
///     // ...
/// ];
/// 
/// // Generate proofs in parallel
/// let proofs = prove_batch(&statements_and_witnesses)?;
/// ```
pub fn prove_batch(
    statements_and_witnesses: &[(Statement, Witness)],
) -> CryptoResult<Vec<Proof>> {
    use rayon::prelude::*;
    
    // Validate inputs
    if statements_and_witnesses.is_empty() {
        return Ok(Vec::new());
    }
    
    // Generate proofs in parallel
    let results: Vec<CryptoResult<Proof>> = statements_and_witnesses
        .par_iter()
        .map(|(statement, witness)| prove(statement, witness))
        .collect();
    
    // Collect results, propagating any errors
    let mut proofs = Vec::with_capacity(results.len());
    for result in results {
        proofs.push(result?);
    }
    
    Ok(proofs)
}

/// Verify multiple proofs in parallel using Rayon
///
/// # Arguments
/// * `statements_and_proofs` - Vector of (Statement, Proof) tuples
///
/// # Returns
/// `Ok(())` if all proofs verify successfully, error otherwise
///
/// # Performance
/// This function uses Rayon to parallelize proof verification across multiple cores.
/// Expected speedup: >3x on 4-core CPUs for batches of 4+ proofs.
pub fn verify_batch(
    statements_and_proofs: &[(Statement, Proof)],
) -> CryptoResult<()> {
    use rayon::prelude::*;
    
    // Validate inputs
    if statements_and_proofs.is_empty() {
        return Ok(());
    }
    
    // Verify proofs in parallel
    let results: Vec<CryptoResult<()>> = statements_and_proofs
        .par_iter()
        .map(|(statement, proof)| verify(statement, proof))
        .collect();
    
    // Check that all verifications succeeded
    for result in results {
        result?;
    }
    
    Ok(())
}

// ============================================================================
// Modular Proof Functions (Registry-based)
// ============================================================================

/// Generate a proof using the modular registry system
pub async fn prove_modular(
    statement: &Statement,
    witness: &Witness,
    prover_id: Option<&str>,
) -> CryptoResult<Proof> {
    #[cfg(feature = "lazy_static")]
    {
        let registry = Arc::clone(&GLOBAL_PROVER_REGISTRY);
        let config = ProverConfig {
            security_level: crate::SecurityLevel::Bit256,
            optimizations: std::collections::HashMap::new(),
            backend_params: std::collections::HashMap::new(),
        };

        let prover = if let Some(id) = prover_id {
            registry.get(id).ok_or_else(|| CryptoError::InvalidParameter("Prover not found".to_string()))?
        } else {
            // For now, just use the legacy prover
            registry.get("legacy").ok_or_else(|| CryptoError::InvalidParameter("Legacy prover not found".to_string()))?
        };

        prover.prove(statement, witness, &config).await
    }

    #[cfg(not(feature = "lazy_static"))]
    {
        // Fallback to legacy implementation
        prove(statement, witness)
    }
}

/// Verify a proof using the modular registry system
pub async fn verify_modular(
    statement: &Statement,
    proof: &Proof,
    verifier_id: Option<&str>,
) -> CryptoResult<bool> {
    #[cfg(feature = "lazy_static")]
    {
        let registry = Arc::clone(&GLOBAL_VERIFIER_REGISTRY);
        let config = VerifierConfig {
            security_level: crate::SecurityLevel::Bit256,
            optimizations: std::collections::HashMap::new(),
            backend_params: std::collections::HashMap::new(),
        };

        let verifier = if let Some(id) = verifier_id {
            registry.get(id).ok_or_else(|| CryptoError::InvalidParameter("Verifier not found".to_string()))?
        } else {
            // For now, just use the legacy verifier
            registry.get("legacy").ok_or_else(|| CryptoError::InvalidParameter("Legacy verifier not found".to_string()))?
        };

        verifier.verify(statement, proof, &config).await
    }

    #[cfg(not(feature = "lazy_static"))]
    {
        // Fallback to legacy implementation
        verify(statement, proof).map(|_| true)
    }
}

/// Batch prove using the modular registry system
pub async fn prove_batch_modular(
    statements: &[Statement],
    witnesses: &[Witness],
    prover_id: Option<&str>,
) -> CryptoResult<Vec<Proof>> {
    #[cfg(feature = "lazy_static")]
    {
        let registry = Arc::clone(&GLOBAL_PROVER_REGISTRY);
        let config = ProverConfig {
            security_level: crate::SecurityLevel::Bit256,
            optimizations: std::collections::HashMap::new(),
            backend_params: std::collections::HashMap::new(),
        };

        let prover = if let Some(id) = prover_id {
            registry.get(id).ok_or_else(|| CryptoError::InvalidParameter("Prover not found".to_string()))?
        } else if !statements.is_empty() {
            // For now, just use the legacy prover
            registry.get("legacy").ok_or_else(|| CryptoError::InvalidParameter("Legacy prover not found".to_string()))?
        } else {
            return Ok(vec![]);
        };

        prover.prove_batch(statements, witnesses, &config).await
    }

    #[cfg(not(feature = "lazy_static"))]
    {
        // Fallback to legacy implementation
        prove_batch(statements, witnesses)
    }
}

/// Batch verify using the modular registry system
pub async fn verify_batch_modular(
    statements: &[Statement],
    proofs: &[Proof],
    verifier_id: Option<&str>,
) -> CryptoResult<Vec<bool>> {
    #[cfg(feature = "lazy_static")]
    {
        let registry = Arc::clone(&GLOBAL_VERIFIER_REGISTRY);
        let config = VerifierConfig {
            security_level: crate::SecurityLevel::Bit256,
            optimizations: std::collections::HashMap::new(),
            backend_params: std::collections::HashMap::new(),
        };

        let verifier = if let Some(id) = verifier_id {
            registry.get(id).ok_or_else(|| CryptoError::InvalidParameter("Verifier not found".to_string()))?
        } else if !statements.is_empty() {
            // For now, just use the legacy verifier
            registry.get("legacy").ok_or_else(|| CryptoError::InvalidParameter("Legacy verifier not found".to_string()))?
        } else {
            return Ok(vec![]);
        };

        verifier.verify_batch(statements, proofs, &config).await
    }

    #[cfg(not(feature = "lazy_static"))]
    {
        // Fallback to legacy implementation
        verify_batch(statements, proofs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::constant_time::ct_modpow;

    #[test]
    fn test_proof_structure() {
        let proof = Proof {
            commitments: vec![Commitment {
                value: vec![1, 2, 3],
            }],
            challenge: Challenge { value: vec![0u8; 32] },
            responses: vec![Response {
                value: vec![4, 5, 6],
            }],
            metadata: ProofMetadata {
                version: 1,
                timestamp: 0,
                size: 100,
            },
            bulletproof: None,
        };

        assert!(proof.validate().is_ok());
    }

    #[test]
    fn test_fiat_shamir_consistency() {
        use crate::proof::statement::{HashFunction, StatementBuilder};

        let stmt = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, vec![0u8; 32])
            .build()
            .unwrap();

        let commitments = vec![Commitment {
            value: vec![1, 2, 3],
        }];

        let c1 = compute_challenge(&stmt, &commitments).unwrap();
        let c2 = compute_challenge(&stmt, &commitments).unwrap();

        assert_eq!(c1.value, c2.value);
    }

    #[test]
    fn test_discrete_log_proof_generation() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;

        // Create statement: prove knowledge of x where g^x = h
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        
        // Compute public_value = generator^secret (mod p)
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();

        let statement = StatementBuilder::new()
            .discrete_log(generator.clone(), public_value.clone())
            .build()
            .unwrap();

        // Create witness with secret exponent
        let witness = Witness::discrete_log(secret);

        // Generate proof
        let result = prove(&statement, &witness);
        assert!(result.is_ok(), "Proof generation should succeed");

        let proof = result.unwrap();
        assert!(!proof.commitments.is_empty());
        assert!(!proof.responses.is_empty());
        assert_eq!(proof.metadata.version, 1);
    }

    #[test]
    fn test_discrete_log_proof_verification() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;

        // Create statement with proper discrete log relationship
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        
        // Compute public_value = generator^secret (mod p)
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();

        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();

        // Create witness and generate proof
        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();

        // Verify proof
        let result = verify(&statement, &proof);
        assert!(result.is_ok(), "Proof verification should succeed");
    }

    #[test]
    fn test_preimage_proof_generation() {
        use crate::proof::statement::{HashFunction, StatementBuilder};
        use sha3::{Digest, Sha3_256};

        // Create preimage and its hash
        let preimage = b"secret message".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();

        // Create statement
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();

        // Create witness
        let witness = Witness::preimage(preimage);

        // Generate proof
        let result = prove(&statement, &witness);
        assert!(result.is_ok(), "Preimage proof generation should succeed");

        let proof = result.unwrap();
        assert!(!proof.commitments.is_empty());
        assert!(!proof.responses.is_empty());
    }

    #[test]
    fn test_preimage_proof_verification() {
        use crate::proof::statement::{HashFunction, StatementBuilder};
        use sha3::{Digest, Sha3_256};

        // Create preimage and hash
        let preimage = b"secret message".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();

        // Create statement and witness
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        let witness = Witness::preimage(preimage);

        // Generate and verify proof
        let proof = prove(&statement, &witness).unwrap();
        let result = verify(&statement, &proof);
        
        assert!(result.is_ok(), "Preimage proof verification should succeed");
    }

    #[test]
    fn test_proof_tampering_detection() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;

        // Generate original proof with correct discrete log relationship
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        
        // Compute public_value = generator^secret (mod p)
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();

        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement, &witness).unwrap();

        // Tamper with challenge
        proof.challenge.value[0] ^= 0xFF;

        // Verification should fail
        let result = verify(&statement, &proof);
        assert!(result.is_err(), "Tampered proof should fail verification");
        
        if let Err(e) = result {
            match e {
                CryptoError::VerificationError(msg) => {
                    assert!(msg.contains("Challenge"));
                }
                _ => panic!("Expected VerificationError"),
            }
        }
    }

    #[test]
    fn test_invalid_witness_rejection() {
        use crate::proof::statement::{HashFunction, StatementBuilder};
        use sha3::{Digest, Sha3_256};

        // Create statement with specific hash
        let correct_preimage = b"correct secret".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&correct_preimage);
        let hash = hasher.finalize().to_vec();

        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();

        // Create witness with WRONG preimage
        let wrong_preimage = b"wrong secret".to_vec();
        let witness = Witness::preimage(wrong_preimage);

        // Proof generation should fail (witness doesn't satisfy statement)
        let result = prove(&statement, &witness);
        assert!(result.is_err(), "Invalid witness should be rejected");
    }

    #[test]
    fn test_proof_soundness() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;

        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);

        // Generate 5 different statements and prove each one correctly
        for i in 1..=5 {
            let secret = vec![i as u8; 32];
            
            // Compute correct public_value = generator^secret (mod p)
            let secret_big = BigUint::from_bytes_be(&secret);
            let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator.clone(), public_value)
                .build()
                .unwrap();
            
            let witness = Witness::discrete_log(secret);
            
            let proof_result = prove(&statement, &witness);
            assert!(proof_result.is_ok(), "Proof {} generation failed", i);
            
            let proof = proof_result.unwrap();
            let verify_result = verify(&statement, &proof);
            assert!(verify_result.is_ok(), "Proof {} verification failed", i);
        }
    }

    #[test]
    fn test_proof_serialization() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;

        // Generate proof with correct discrete log relationship
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        
        // Compute public_value = generator^secret (mod p)
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();

        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();

        // Serialize
        let bytes = proof.to_bytes().unwrap();
        assert!(!bytes.is_empty());
        
        // Print proof size for reference
        println!("Discrete log proof size: {} bytes", bytes.len());

        // Deserialize
        let deserialized = Proof::from_bytes(&bytes).unwrap();

        // Verify deserialized proof
        let result = verify(&statement, &deserialized);
        assert!(result.is_ok(), "Deserialized proof should verify");
    }
    
    #[test]
    fn test_proof_size_measurements() {
        use crate::proof::statement::{StatementBuilder, HashFunction};
        use sha3::{Digest, Sha3_256};
        use num_bigint::BigUint;
        
        // Test discrete log proof size
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();
        let bytes = proof.to_bytes().unwrap();
        
        println!("Discrete log proof: {} bytes", bytes.len());
        
        // Test preimage proof size
        let preimage = b"test message for proof".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let statement2 = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        let witness2 = Witness::preimage(preimage);
        let proof2 = prove(&statement2, &witness2).unwrap();
        let bytes2 = proof2.to_bytes().unwrap();
        
        println!("Preimage proof: {} bytes", bytes2.len());
        
        // Test range proof size with Bulletproofs
        let value: u64 = 15;
        let blinding = vec![0xAA; 32];
        let commitment = crate::proof::bulletproofs::pedersen_commit(value, &blinding).unwrap();
        
        let statement3 = StatementBuilder::new()
            .range(10, 20, commitment)
            .build()
            .unwrap();
        let witness3 = Witness::range(value, blinding);
        let proof3 = prove(&statement3, &witness3).unwrap();
        let bytes3 = proof3.to_bytes().unwrap();
        
        println!("Range proof (with Bulletproof): {} bytes", bytes3.len());
        
        // Verify all proofs still work
        assert!(verify(&statement, &proof).is_ok());
        assert!(verify(&statement2, &proof2).is_ok());
        assert!(verify(&statement3, &proof3).is_ok());
    }

    #[test]
    fn test_proof_metadata() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;

        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        
        // Compute public_value = generator^secret (mod p)
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();

        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();

        // Check metadata
        assert_eq!(proof.metadata.version, 1);
        // Timestamp is now optional for size optimization
        assert!(proof.metadata.timestamp == 0);
        // Size is computed dynamically
        assert!(proof.size() > 0);
    }

    #[test]
    fn test_proof_validation_errors() {
        let bad_proof_empty = Proof { commitments: vec![], challenge: Challenge { value: vec![0u8;32] }, responses: vec![Response{ value: vec![1]}], metadata: ProofMetadata{version:1,timestamp:0,size:0}, bulletproof: None };        
        assert!(bad_proof_empty.validate().is_err());
        let bad_proof_no_responses = Proof { commitments: vec![Commitment{ value: vec![1]}], challenge: Challenge { value: vec![0u8;32] }, responses: vec![], metadata: ProofMetadata{version:1,timestamp:0,size:0}, bulletproof: None };        
        assert!(bad_proof_no_responses.validate().is_err());
    }

    #[test]
    fn test_discrete_log_commitment_tamper_failure() {
        use crate::proof::statement::StatementBuilder; use num_bigint::BigUint;
        let generator = vec![2u8;32]; let secret = vec![42u8;32];
        let modulus_bytes = vec![0xFF;32];
        let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        let statement = StatementBuilder::new().discrete_log(generator.clone(), public_value).build().unwrap();
        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement,&witness).unwrap();
        // tamper commitment (will cause challenge mismatch rather than equation failure)
        proof.commitments[0].value[0] ^= 0xAA; // flip byte
        let result = verify(&statement,&proof);
        assert!(matches!(result, Err(CryptoError::VerificationError(msg)) if msg.contains("Challenge")));
    }

    #[test]
    fn test_discrete_log_response_tamper_failure() {
        use crate::proof::statement::StatementBuilder; use num_bigint::BigUint;
        let generator = vec![2u8;32]; let secret = vec![42u8;32];
        let modulus_bytes = vec![0xFF;32];
        let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        let statement = StatementBuilder::new().discrete_log(generator.clone(), public_value).build().unwrap();
        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement,&witness).unwrap();
        // Tamper response drastically (zero out bytes to force equation failure)
        let len = proof.responses[0].value.len();
        proof.responses[0].value = vec![0u8; len];
        let result = verify(&statement,&proof);
        assert!(result.is_err());
        if let Err(CryptoError::VerificationError(msg)) = result { assert!(msg.contains("Discrete log") || msg.contains("verification failed")); } else { panic!("Expected VerificationError"); }
    }

    #[test]
    fn test_preimage_response_tamper_commitment_failure() {
        use crate::proof::statement::{HashFunction, StatementBuilder}; use sha3::{Digest,Sha3_256};
        let preimage = b"secret msg".to_vec(); let mut hasher = Sha3_256::new(); hasher.update(&preimage); let hash = hasher.finalize().to_vec();
        let statement = StatementBuilder::new().preimage(HashFunction::SHA3_256, hash).build().unwrap();
        let witness = Witness::preimage(preimage);
        let mut proof = prove(&statement,&witness).unwrap();
        // Tamper response so recomputed blinding mismatches commitment
        proof.responses[0].value[0] ^= 0xAA;
        let result = verify(&statement,&proof);
        assert!(matches!(result, Err(CryptoError::VerificationError(msg)) if msg.contains("commitment")));
    }

    #[test]
    fn test_truncated_proof_deserialization_failure() {
        use crate::proof::statement::StatementBuilder; use num_bigint::BigUint;
        let generator = vec![2u8;32]; let secret = vec![42u8;32];
        let modulus_bytes = vec![0xFF;32]; let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes); let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        let statement = StatementBuilder::new().discrete_log(generator.clone(), public_value).build().unwrap();
        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement,&witness).unwrap();
        let bytes = proof.to_bytes().unwrap();
        let truncated = &bytes[..bytes.len()/4];
        let result = Proof::from_bytes(truncated);
        assert!(result.is_err());
    }

    #[test]
    fn test_commitment_reordering_challenge_mismatch() {
        use crate::proof::statement::StatementBuilder; use num_bigint::BigUint;
        let generator = vec![2u8;32]; let secret = vec![42u8;32];
        let modulus_bytes = vec![0xFF;32]; let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes); let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        let statement = StatementBuilder::new().discrete_log(generator.clone(), public_value).build().unwrap();
        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement,&witness).unwrap();
        // Duplicate commitment causing challenge mismatch
        let first = proof.commitments[0].clone();
        proof.commitments.push(first); // modifies commitment list
        let result = verify(&statement,&proof);
        assert!(matches!(result, Err(CryptoError::VerificationError(msg)) if msg.contains("Challenge")));
    }

    #[test]
    fn test_manual_blake3_preimage_verification_failure() {
        use crate::proof::statement::{StatementBuilder, HashFunction};
        // Build statement with Blake3 (unsupported in verify_preimage_proof)
        let statement = StatementBuilder::new()
            .preimage(HashFunction::Blake3, vec![1,2,3,4])
            .build()
            .unwrap();
        // Manually fabricate a proof with one commitment/response and matching challenge
        let commitments = vec![Commitment { value: vec![0xAA; 32] }];
        let challenge = compute_challenge(&statement,&commitments).unwrap();
        let responses = vec![Response { value: vec![0xBB; 32] }];
        let proof = Proof { commitments, challenge, responses, metadata: ProofMetadata { version:1, timestamp:0, size:0 }, bulletproof: None };
        let result = verify(&statement,&proof);
        assert!(matches!(result, Err(CryptoError::VerificationError(msg)) if msg.contains("Blake3")));
    }

    // ===================== Added Edge Case Verification Tests (Wave 4) =====================

    #[test]
    fn test_range_proof_generation_and_verification_success() {
        use crate::proof::statement::StatementBuilder;
        
        // Test with Bulletproofs providing full cryptographic range enforcement
        let value: u64 = 15;
        let blinding = vec![0xAA; 32];
        
        // Create commitment using Bulletproofs commitment function
        let commitment = crate::proof::bulletproofs::pedersen_commit(value, &blinding).unwrap();
        
        let statement = StatementBuilder::new()
            .range(10, 20, commitment.clone())
            .build()
            .unwrap();
        let witness = Witness::range(value, blinding);
        
        // Generate proof with full Bulletproofs protocol
        let proof = prove(&statement, &witness).unwrap();
        
        // Verify proof has Bulletproof component
        assert!(proof.bulletproof.is_some(), "Proof should contain Bulletproof");
        
        // Verify proof cryptographically
        let result = verify(&statement, &proof);
        assert!(result.is_ok(), "Valid range proof with Bulletproofs should verify");
    }

    #[test]
    fn test_range_proof_out_of_range_failure() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        // Witness has out-of-range value 25, should fail at prove() stage
        let value: u64 = 25;
        let blinding = vec![0xBB; 16];
        
        let modulus_bytes = vec![0xFF; 32];
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let gen_g = vec![2u8; 32];
        let gen_h = vec![3u8; 32];
        
        let g_big = BigUint::from_bytes_be(&gen_g);
        let h_big = BigUint::from_bytes_be(&gen_h);
        let v_big = BigUint::from(value);
        let r_big = BigUint::from_bytes_be(&blinding);
        
        let g_v = ct_modpow(&g_big, &v_big, &mod_big);
        let h_r = ct_modpow(&h_big, &r_big, &mod_big);
        let commitment = (g_v * h_r) % &mod_big;
        let commitment_bytes = commitment.to_bytes_be();
        
        let statement = StatementBuilder::new().range(10, 20, commitment_bytes).build().unwrap();
        let witness = Witness::range(value, blinding);
        // Should fail because witness value 25 is outside [10, 20]
        let result = prove(&statement, &witness);
        assert!(result.is_err(), "Proof generation should fail for out-of-range witness");
    }

    #[test]
    fn test_custom_statement_verification_not_supported() {
        use crate::proof::statement::{Statement, StatementType};
        // Manually construct a Custom statement (builder has no helper)
        let statement = Statement {
            statement_type: StatementType::Custom { description: "demo".to_string() },
            version: 1,
        };
        // Fabricate a minimal proof
        let commitments = vec![Commitment { value: vec![9,9,9] }];
        let challenge = compute_challenge(&statement, &commitments).unwrap();
        let responses = vec![Response { value: vec![8,8,8] }];
        let proof = Proof { commitments, challenge, responses, metadata: ProofMetadata { version:1, timestamp:0, size:0 }, bulletproof: None };
        let result = verify(&statement, &proof);
        assert!(matches!(result, Err(CryptoError::VerificationError(msg)) if msg.contains("Custom statements")));
    }

    #[test]
    fn test_preimage_verification_invalid_empty_hash_output() {
        use crate::proof::statement::{Statement, StatementType, HashFunction};
        use sha3::{Digest, Sha3_256};
        // Manually construct preimage statement with EMPTY hash output to reach error branch
        let statement = Statement {
            statement_type: StatementType::Preimage { hash_function: HashFunction::SHA3_256, hash_output: vec![] },
            version: 1,
        };
        // Construct consistent commitment/response so commitment check passes
        let blinding: Vec<u8> = (0..32).map(|i| (i as u8) ^ 0xA5).collect();
        let mut hasher = Sha3_256::new(); hasher.update(&blinding); let commitment_bytes = hasher.finalize().to_vec();
        let commitments = vec![Commitment { value: commitment_bytes }];
        let challenge = compute_challenge(&statement, &commitments).unwrap();
        // response = blinding XOR challenge
        let mut response_bytes = blinding.clone();
        for (i, b) in response_bytes.iter_mut().enumerate() { if i < challenge.value.len() { *b ^= challenge.value[i]; } }
        let responses = vec![Response { value: response_bytes }];
        let proof = Proof { commitments, challenge, responses, metadata: ProofMetadata { version:1, timestamp:0, size:0 }, bulletproof: None };
        let result = verify(&statement, &proof);
        assert!(matches!(result, Err(CryptoError::VerificationError(msg)) if msg.contains("Invalid hash output")));
    }

    // ===================== Additional Proof Verification Edge Cases =====================

    #[test]
    fn test_proof_with_mismatched_commitment_response_counts() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        // Create valid statement
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement, &witness).unwrap();
        
        // Add extra commitment to mismatch counts
        proof.commitments.push(Commitment { value: vec![0xFF; 32] });
        
        // Should fail during verification due to challenge mismatch
        let result = verify(&statement, &proof);
        assert!(result.is_err(), "Proof with mismatched commitment count should fail");
    }

    #[test]
    fn test_proof_verification_with_all_zero_challenge() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement, &witness).unwrap();
        
        // Set challenge to all zeros (weak challenge)
        proof.challenge.value = vec![0u8; 32];
        
        // Verification should fail due to challenge mismatch
        let result = verify(&statement, &proof);
        assert!(result.is_err(), "Proof with manipulated zero challenge should fail");
    }

    #[test]
    fn test_proof_verification_with_oversized_response() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        
        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement, &witness).unwrap();
        
        // Replace response with oversized value (1024 bytes instead of ~32)
        proof.responses[0].value = vec![0xFF; 1024];
        
        // Verification should fail during equation check
        let result = verify(&statement, &proof);
        assert!(result.is_err(), "Proof with oversized response should fail verification");
    }

    #[test]
    fn test_range_proof_with_zero_range() {
        use crate::proof::statement::StatementBuilder;
        
        // Create statement with zero-width range [10, 10]
        let value: u64 = 10;
        let blinding = vec![0xCC; 32];
        let commitment = crate::proof::bulletproofs::pedersen_commit(value, &blinding).unwrap();
        
        // Zero-width range should be rejected by StatementBuilder
        let statement = StatementBuilder::new()
            .range(10, 10, commitment)
            .build();
        
        // Should fail with InvalidParameter error
        assert!(statement.is_err(), "Zero-width range [10,10] should be rejected");
        
        if let Err(e) = statement {
            // Verify it's the expected error
            match e {
                CryptoError::InvalidParameter(msg) => {
                    assert!(msg.contains("min must be less than max"), 
                           "Error should indicate min < max requirement");
                }
                _ => panic!("Expected InvalidParameter error for zero-width range"),
            }
        }
    }

    // ===================== Witness Validation Corner Cases =====================

    #[test]
    fn test_witness_with_empty_secret_data() {
        use crate::proof::statement::{HashFunction, StatementBuilder};
        use sha3::{Digest, Sha3_256};
        
        // Create statement expecting non-empty preimage
        let preimage = b"test".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        // Create witness with empty preimage
        let empty_witness = Witness::preimage(vec![]);
        
        // Should fail satisfaction check
        assert!(!empty_witness.satisfies_statement(&statement), 
                "Witness with empty secret should not satisfy statement");
    }

    #[test]
    fn test_witness_range_value_at_exact_boundaries() {
        use crate::proof::statement::StatementBuilder;
        
        // Test exact minimum boundary (10)
        let min_value: u64 = 10;
        let blinding_min = vec![0xDD; 32];
        let commitment_min = crate::proof::bulletproofs::pedersen_commit(min_value, &blinding_min).unwrap();
        
        let statement_min = StatementBuilder::new()
            .range(10, 20, commitment_min)
            .build()
            .unwrap();
        
        let witness_min = Witness::range(min_value, blinding_min);
        assert!(witness_min.satisfies_statement(&statement_min),
                "Witness at exact minimum boundary should satisfy statement");
        
        // Test exact maximum boundary (19, since range is [10, 20) exclusive)
        let max_value: u64 = 19;
        let blinding_max = vec![0xEE; 32];
        let commitment_max = crate::proof::bulletproofs::pedersen_commit(max_value, &blinding_max).unwrap();
        
        let statement_max = StatementBuilder::new()
            .range(10, 20, commitment_max)
            .build()
            .unwrap();
        
        let witness_max = Witness::range(max_value, blinding_max);
        assert!(witness_max.satisfies_statement(&statement_max),
                "Witness at exact maximum boundary should satisfy statement");
    }

    #[test]
    fn test_witness_satisfies_wrong_statement_type() {
        use crate::proof::statement::{HashFunction, StatementBuilder};
        use sha3::{Digest, Sha3_256};
        
        // Create discrete log witness
        let secret = vec![42u8; 32];
        let dlog_witness = Witness::discrete_log(secret);
        
        // Create preimage statement (different type)
        let preimage = b"test".to_vec();
        let mut hasher = Sha3_256::new();
        hasher.update(&preimage);
        let hash = hasher.finalize().to_vec();
        
        let preimage_statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        
        // Discrete log witness should not satisfy preimage statement
        assert!(!dlog_witness.satisfies_statement(&preimage_statement),
                "Witness of wrong type should not satisfy statement");
    }

    // ===================== Parallel Proof Generation Tests =====================

    #[test]
    fn test_prove_batch_empty() {
        let batch: Vec<(Statement, Witness)> = vec![];
        let proofs = prove_batch(&batch).unwrap();
        assert_eq!(proofs.len(), 0);
    }

    #[test]
    fn test_prove_batch_single() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(secret);
        
        let batch = vec![(statement.clone(), witness)];
        let proofs = prove_batch(&batch).unwrap();
        
        assert_eq!(proofs.len(), 1);
        assert!(verify(&statement, &proofs[0]).is_ok());
    }

    #[test]
    fn test_prove_batch_multiple() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let gen_big = BigUint::from_bytes_be(&generator);
        
        // Create multiple statement-witness pairs
        let mut batch = Vec::new();
        for i in 1..=5 {
            let secret = vec![i as u8; 32];
            let secret_big = BigUint::from_bytes_be(&secret);
            let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator.clone(), public_value)
                .build()
                .unwrap();
            let witness = Witness::discrete_log(secret);
            
            batch.push((statement, witness));
        }
        
        // Generate proofs in parallel
        let proofs = prove_batch(&batch).unwrap();
        
        assert_eq!(proofs.len(), 5);
        
        // Verify all proofs
        for (i, proof) in proofs.iter().enumerate() {
            assert!(verify(&batch[i].0, proof).is_ok(), 
                    "Proof {} should verify", i);
        }
    }

    #[test]
    fn test_verify_batch_empty() {
        let batch: Vec<(Statement, Proof)> = vec![];
        assert!(verify_batch(&batch).is_ok());
    }

    #[test]
    fn test_verify_batch_single() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let secret = vec![42u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();
        
        let batch = vec![(statement, proof)];
        assert!(verify_batch(&batch).is_ok());
    }

    #[test]
    fn test_verify_batch_multiple() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let gen_big = BigUint::from_bytes_be(&generator);
        
        // Create and prove multiple statements
        let mut batch = Vec::new();
        for i in 1..=5 {
            let secret = vec![i as u8; 32];
            let secret_big = BigUint::from_bytes_be(&secret);
            let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator.clone(), public_value)
                .build()
                .unwrap();
            let witness = Witness::discrete_log(secret);
            let proof = prove(&statement, &witness).unwrap();
            
            batch.push((statement, proof));
        }
        
        // Verify all proofs in parallel
        assert!(verify_batch(&batch).is_ok());
    }

    #[test]
    fn test_verify_batch_with_invalid_proof() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let gen_big = BigUint::from_bytes_be(&generator);
        
        // Create valid and invalid proofs
        let mut batch = Vec::new();
        
        // Add 2 valid proofs
        for i in 1..=2 {
            let secret = vec![i as u8; 32];
            let secret_big = BigUint::from_bytes_be(&secret);
            let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator.clone(), public_value)
                .build()
                .unwrap();
            let witness = Witness::discrete_log(secret);
            let proof = prove(&statement, &witness).unwrap();
            
            batch.push((statement, proof));
        }
        
        // Add 1 invalid proof (tampered)
        let secret = vec![3u8; 32];
        let secret_big = BigUint::from_bytes_be(&secret);
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator.clone(), public_value)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(secret);
        let mut proof = prove(&statement, &witness).unwrap();
        
        // Tamper with the proof
        proof.challenge.value[0] ^= 0xFF;
        
        batch.push((statement, proof));
        
        // Batch verification should fail
        assert!(verify_batch(&batch).is_err());
    }

    #[test]
    fn test_batch_proof_consistency() {
        use crate::proof::statement::StatementBuilder;
        use num_bigint::BigUint;
        
        let generator = vec![2u8; 32];
        let modulus_bytes = vec![0xFF; 32];
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let gen_big = BigUint::from_bytes_be(&generator);
        
        // Create batch of statement-witness pairs
        let mut statements_and_witnesses = Vec::new();
        for i in 1..=4 {
            let secret = vec![i as u8; 32];
            let secret_big = BigUint::from_bytes_be(&secret);
            let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
            
            let statement = StatementBuilder::new()
                .discrete_log(generator.clone(), public_value)
                .build()
                .unwrap();
            let witness = Witness::discrete_log(secret);
            
            statements_and_witnesses.push((statement, witness));
        }
        
        // Generate proofs in batch
        let batch_proofs = prove_batch(&statements_and_witnesses).unwrap();
        
        // Generate proofs individually
        let individual_proofs: Vec<Proof> = statements_and_witnesses
            .iter()
            .map(|(stmt, wit)| prove(stmt, wit).unwrap())
            .collect();
        
        // Both methods should produce valid proofs
        assert_eq!(batch_proofs.len(), individual_proofs.len());
        
        for i in 0..batch_proofs.len() {
            assert!(verify(&statements_and_witnesses[i].0, &batch_proofs[i]).is_ok());
            assert!(verify(&statements_and_witnesses[i].0, &individual_proofs[i]).is_ok());
        }
    }
}
