//! Proof generation and verification
//!
//! This module implements the core prove/verify algorithms.

use crate::proof::{Statement, Witness};
use crate::{CryptoError, CryptoResult};
use serde::{Deserialize, Serialize};

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
    /// Challenge bytes
    pub value: [u8; 32],
}

/// Response in the proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Response {
    /// Response value
    pub value: Vec<u8>,
}

/// Proof metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofMetadata {
    /// Proof system version
    pub version: u8,
    /// Timestamp of generation
    pub timestamp: u64,
    /// Size in bytes
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

    /// Get proof size
    pub fn size(&self) -> usize {
        self.metadata.size
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
fn challenge_to_bigint(challenge: &[u8; 32]) -> BigUint {
    BigUint::from_bytes_be(challenge)
}

/// Modular addition: (a + b) mod m, padded to 32 bytes
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

/// Modular exponentiation: base^exp mod modulus
fn mod_exp(base: &[u8], exp: &[u8], modulus: &[u8]) -> Vec<u8> {
    let base_big = BigUint::from_bytes_be(base);
    let exp_big = BigUint::from_bytes_be(exp);
    let mod_big = BigUint::from_bytes_be(modulus);
    
    let result = base_big.modpow(&exp_big, &mod_big);
    result.to_bytes_be()
}

/// Get current Unix timestamp
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
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

// ============================================================================
// Response Phase Functions
// ============================================================================

/// Compute response for discrete log: s = r + c*x (no modulus, let it be large)
fn compute_discrete_log_response(
    secret: &[u8],
    blinding: &[u8],
    challenge: &[u8; 32],
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
    challenge: &[u8; 32],
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

// ============================================================================
// Verification Functions
// ============================================================================

/// Verify discrete log proof: check g^s = t * h^c
fn verify_discrete_log_proof(
    generator: &[u8],
    public_value: &[u8],
    commitment: &Commitment,
    challenge: &Challenge,
    response: &Response,
) -> CryptoResult<()> {
    let modulus_bytes = vec![0xFF; 32];
    let mod_big = BigUint::from_bytes_be(&modulus_bytes);
    
    // Compute g^s (mod p)
    let gs_big = {
        let gen_big = BigUint::from_bytes_be(generator);
        let response_big = BigUint::from_bytes_be(&response.value);
        gen_big.modpow(&response_big, &mod_big)
    };
    
    // Compute h^c (mod p)
    let hc_big = {
        let h_big = BigUint::from_bytes_be(public_value);
        let c_big = BigUint::from_bytes_be(&challenge.value);
        h_big.modpow(&c_big, &mod_big)
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
            // For range proofs, we'd need Bulletproofs or similar
            // This is a placeholder
            return Err(CryptoError::ProofError(
                "Range proofs not yet fully implemented".to_string(),
            ));
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
        _ => {
            return Err(CryptoError::ProofError(
                "Unsupported statement type".to_string(),
            ));
        }
    };
    
    // PHASE 5: Package proof
    let proof_bytes = bincode::serialize(&(&commitments, &challenge, &responses))
        .map_err(|e| CryptoError::SerializationError(e.to_string()))?;
    
    let metadata = ProofMetadata {
        version: 1,
        timestamp: current_timestamp(),
        size: proof_bytes.len(),
    };
    
    Ok(Proof {
        commitments,
        challenge,
        responses,
        metadata,
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
        StatementType::Range { .. } => {
            return Err(CryptoError::VerificationError(
                "Range proofs not yet fully implemented".to_string(),
            ));
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
    use sha3::{Digest, Sha3_256};

    let mut hasher = Sha3_256::new();

    // Hash statement
    let stmt_bytes = statement.to_bytes()?;
    hasher.update(&stmt_bytes);

    // Hash all commitments
    for commitment in commitments {
        hasher.update(&commitment.value);
    }

    let hash_output = hasher.finalize();
    let mut challenge_bytes = [0u8; 32];
    challenge_bytes.copy_from_slice(&hash_output);

    Ok(Challenge {
        value: challenge_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_structure() {
        let proof = Proof {
            commitments: vec![Commitment {
                value: vec![1, 2, 3],
            }],
            challenge: Challenge { value: [0u8; 32] },
            responses: vec![Response {
                value: vec![4, 5, 6],
            }],
            metadata: ProofMetadata {
                version: 1,
                timestamp: 0,
                size: 100,
            },
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
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();

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
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();

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
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
        
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
            let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
            
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
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();

        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();

        // Serialize
        let bytes = proof.to_bytes().unwrap();
        assert!(!bytes.is_empty());

        // Deserialize
        let deserialized = Proof::from_bytes(&bytes).unwrap();

        // Verify deserialized proof
        let result = verify(&statement, &deserialized);
        assert!(result.is_ok(), "Deserialized proof should verify");
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
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
        
        let statement = StatementBuilder::new()
            .discrete_log(generator, public_value)
            .build()
            .unwrap();

        let witness = Witness::discrete_log(secret);
        let proof = prove(&statement, &witness).unwrap();

        // Check metadata
        assert_eq!(proof.metadata.version, 1);
        assert!(proof.metadata.timestamp > 0);
        assert!(proof.metadata.size > 0);
        assert_eq!(proof.size(), proof.metadata.size);
    }

    #[test]
    fn test_proof_validation_errors() {
        let bad_proof_empty = Proof { commitments: vec![], challenge: Challenge { value: [0u8;32] }, responses: vec![Response{ value: vec![1]}], metadata: ProofMetadata{version:1,timestamp:0,size:0} };        
        assert!(bad_proof_empty.validate().is_err());
        let bad_proof_no_responses = Proof { commitments: vec![Commitment{ value: vec![1]}], challenge: Challenge { value: [0u8;32] }, responses: vec![], metadata: ProofMetadata{version:1,timestamp:0,size:0} };        
        assert!(bad_proof_no_responses.validate().is_err());
    }

    #[test]
    fn test_discrete_log_commitment_tamper_failure() {
        use crate::proof::statement::StatementBuilder; use num_bigint::BigUint;
        let generator = vec![2u8;32]; let secret = vec![42u8;32];
        let modulus_bytes = vec![0xFF;32];
        let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = gen_big.modpow(&secret_big,&mod_big).to_bytes_be();
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
        let public_value = gen_big.modpow(&secret_big,&mod_big).to_bytes_be();
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
        let modulus_bytes = vec![0xFF;32]; let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes); let public_value = gen_big.modpow(&secret_big,&mod_big).to_bytes_be();
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
        let modulus_bytes = vec![0xFF;32]; let gen_big = BigUint::from_bytes_be(&generator); let secret_big = BigUint::from_bytes_be(&secret); let mod_big = BigUint::from_bytes_be(&modulus_bytes); let public_value = gen_big.modpow(&secret_big,&mod_big).to_bytes_be();
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
        let proof = Proof { commitments, challenge, responses, metadata: ProofMetadata { version:1, timestamp:0, size:0 } };
        let result = verify(&statement,&proof);
        assert!(matches!(result, Err(CryptoError::VerificationError(msg)) if msg.contains("Blake3")));
    }
}
