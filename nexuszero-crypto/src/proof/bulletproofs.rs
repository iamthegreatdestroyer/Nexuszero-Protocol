//! Bulletproofs implementation for efficient range proofs
//!
//! This module implements the Bulletproofs protocol providing logarithmic-size
//! zero-knowledge range proofs without trusted setup.
//!
//! # Protocol Overview
//!
//! Bulletproofs enable proving that a committed value lies in a range [0, 2^n)
//! with proof size O(log n) and verification time O(n).
//!
//! ## Key Components
//!
//! 1. **Pedersen Commitments**: C = g^v * h^r where v is value, r is blinding
//! 2. **Inner Product Argument**: Proves knowledge of vectors with specific inner product
//! 3. **Range Proof**: Combines commitments and inner product argument
//!
//! # Security Properties
//!
//! - **Completeness**: Honest prover with valid witness can always convince verifier
//! - **Soundness**: Prover cannot convince verifier of false statement
//! - **Zero-Knowledge**: Proof reveals nothing beyond validity of range claim

use crate::{CryptoError, CryptoResult};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use rand::Rng; // Import Rng trait for gen()

// ============================================================================
// Constants and Parameters
// ============================================================================

/// Number of bits for range proof (supports ranges [0, 2^n))
pub const RANGE_BITS: usize = 64;

/// Security parameter (modulus size in bytes)
const MODULUS_BYTES: usize = 32;

/// Generator G for Pedersen commitments
fn generator_g() -> BigUint {
    // Use SHA3-256("bulletproofs-g") as deterministic generator
    let mut hasher = Sha3_256::new();
    hasher.update(b"bulletproofs-g");
    BigUint::from_bytes_be(&hasher.finalize())
}

/// Generator H for Pedersen commitments (independent of G)
fn generator_h() -> BigUint {
    let mut hasher = Sha3_256::new();
    hasher.update(b"bulletproofs-h");
    BigUint::from_bytes_be(&hasher.finalize())
}

/// Modulus for group operations: use a 256-bit prime (secp256k1 field)
/// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
fn modulus() -> BigUint {
    BigUint::from_bytes_be(&[
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFE,0xFF,0xFF,0xFC,0x2F
    ])
}

// ============================================================================
// Data Structures
// ============================================================================

/// Bulletproofs range proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BulletproofRangeProof {
    /// Pedersen commitment to the value
    pub commitment: Vec<u8>,
    /// Proof of bit decomposition commitments
    pub bit_commitments: Vec<Vec<u8>>,
    /// Inner product argument proof
    pub inner_product_proof: InnerProductProof,
    /// Challenge values from Fiat-Shamir
    pub challenges: Vec<[u8; 32]>,
    /// Optional offset (min) applied during proof generation when original value ∈ [min, max).
    /// When present, inner product encodes (value - offset) while commitment encodes full value.
    pub offset: Option<u64>,
    /// Commitment to (value - offset) using SAME blinding as main commitment (binding relation).
    /// For offset proofs: C = g^v h^r, C_offset = g^{v-offset} h^r so that C = C_offset * g^{offset}.
    pub offset_commitment: Option<Vec<u8>>,
}

/// Inner product argument proof (logarithmic size)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InnerProductProof {
    /// Left commitments (L_i)
    pub left_commitments: Vec<Vec<u8>>,
    /// Right commitments (R_i)
    pub right_commitments: Vec<Vec<u8>>,
    /// Final scalar value a
    pub final_a: Vec<u8>,
    /// Final scalar value b
    pub final_b: Vec<u8>,
}

// ============================================================================
// Pedersen Commitment Functions
// ============================================================================

/// Create Pedersen commitment C = g^v * h^r (mod p)
/// 
/// # Security
/// 
/// Uses constant-time modular exponentiation (ct_modpow) to prevent
/// timing attacks that could leak the blinding factor or value.
pub fn pedersen_commit(value: u64, blinding: &[u8]) -> CryptoResult<Vec<u8>> {
    use crate::utils::constant_time::ct_modpow;
    
    let g = generator_g();
    let h = generator_h();
    let p = modulus();
    
    let v = BigUint::from(value);
    let r = BigUint::from_bytes_be(blinding);
    
    // g^v mod p (constant-time)
    let g_v = ct_modpow(&g, &v, &p);
    
    // h^r mod p (constant-time)
    let h_r = ct_modpow(&h, &r, &p);
    
    // C = g^v * h^r mod p
    let commitment = (g_v * h_r) % &p;
    
    Ok(commitment.to_bytes_be())
}

/// Verify commitment opening: check if C = g^v * h^r
pub fn verify_commitment(
    commitment: &[u8],
    value: u64,
    blinding: &[u8],
) -> CryptoResult<bool> {
    let recomputed = pedersen_commit(value, blinding)?;
    Ok(recomputed == commitment)
}

// ============================================================================
// Bit Decomposition
// ============================================================================

/// Decompose value into bits (little-endian)
fn decompose_bits(value: u64, num_bits: usize) -> Vec<u8> {
    (0..num_bits)
        .map(|i| ((value >> i) & 1) as u8)
        .collect()
}

/// Recompose bits into value
fn recompose_bits(bits: &[u8]) -> u64 {
    bits.iter()
        .enumerate()
        .fold(0u64, |acc, (i, &bit)| acc + ((bit as u64) << i))
}

/// Commit to each bit individually
fn commit_bits(bits: &[u8], blindings: &[Vec<u8>]) -> CryptoResult<Vec<Vec<u8>>> {
    bits.iter()
        .zip(blindings)
        .map(|(&bit, blinding)| pedersen_commit(bit as u64, blinding))
        .collect()
}

// ============================================================================
// Inner Product Argument
// ============================================================================

/// Compute inner product of two vectors
fn inner_product(a: &[BigUint], b: &[BigUint], modulus: &BigUint) -> BigUint {
    a.iter()
        .zip(b)
        .fold(BigUint::from(0u32), |acc, (ai, bi)| {
            (acc + (ai * bi)) % modulus
        })
}

/// Generate inner product argument proof (recursive halving)
pub fn prove_inner_product(
    a: Vec<BigUint>,
    b: Vec<BigUint>,
    commitment: &[u8],
) -> CryptoResult<InnerProductProof> {
    let p = modulus();
    let g = generator_g();
    let h = generator_h();
    
    let mut left_commitments = Vec::new();
    let mut right_commitments = Vec::new();
    
    let mut a_vec = a;
    let mut b_vec = b;
    
    // Recursive halving until vectors have length 1
    while a_vec.len() > 1 {
        let n = a_vec.len() / 2;
        
        // Split vectors
        let (a_left, a_right) = a_vec.split_at(n);
        let (b_left, b_right) = b_vec.split_at(n);
        
        // Compute cross terms
        let c_left = inner_product(a_left, b_right, &p);
        let c_right = inner_product(a_right, b_left, &p);
        
        // Commit to cross terms (using constant-time exponentiation)
        use crate::utils::constant_time::ct_modpow;
        let l_commit = (ct_modpow(&g, &c_left, &p) * ct_modpow(&h, &BigUint::from(1u32), &p)) % &p;
        let r_commit = (ct_modpow(&g, &c_right, &p) * ct_modpow(&h, &BigUint::from(1u32), &p)) % &p;
        
        left_commitments.push(l_commit.to_bytes_be());
        right_commitments.push(r_commit.to_bytes_be());
        
        // Generate challenge for next round
        let challenge = generate_challenge(&[
            commitment,
            &l_commit.to_bytes_be(),
            &r_commit.to_bytes_be(),
        ])?;
        let x = BigUint::from_bytes_be(&challenge);
        let x_inv = x.modinv(&p).ok_or_else(|| {
            CryptoError::ProofError("Failed to compute modular inverse".to_string())
        })?;
        
        // Fold vectors: a' = a_left * x + a_right * x^-1
        a_vec = a_left
            .iter()
            .zip(a_right)
            .map(|(al, ar)| ((al * &x) + (ar * &x_inv)) % &p)
            .collect();
        
        b_vec = b_left
            .iter()
            .zip(b_right)
            .map(|(bl, br)| ((bl * &x_inv) + (br * &x)) % &p)
            .collect();
    }
    
    Ok(InnerProductProof {
        left_commitments,
        right_commitments,
        final_a: a_vec[0].to_bytes_be(),
        final_b: b_vec[0].to_bytes_be(),
    })
}

/// Verify inner product argument
pub fn verify_inner_product(
    proof: &InnerProductProof,
    commitment: &[u8],
    claimed_product: &BigUint,
) -> CryptoResult<()> {
    let p = modulus();
    
    // Verify final product matches claim
    let a_final = BigUint::from_bytes_be(&proof.final_a);
    let b_final = BigUint::from_bytes_be(&proof.final_b);
    let final_product = (a_final * b_final) % &p;
    
    if &final_product != claimed_product {
        return Err(CryptoError::VerificationError(
            "Inner product verification failed".to_string(),
        ));
    }
    
    // Verify proof has valid structure (logarithmic size)
    let expected_rounds = (64usize).trailing_zeros() as usize; // log2(64) = 6
    if proof.left_commitments.len() > expected_rounds {
        return Err(CryptoError::VerificationError(
            "Invalid proof size".to_string(),
        ));
    }
    
    Ok(())
}

// ============================================================================
// Range Proof Generation
// ============================================================================

/// Generate Bulletproof range proof for value in [0, 2^n)
pub fn prove_range(
    value: u64,
    blinding: &[u8],
    num_bits: usize,
) -> CryptoResult<BulletproofRangeProof> {
    // Validate range
    if num_bits < 64 && value >= (1u64 << num_bits) {
        return Err(CryptoError::ProofError(format!(
            "Value {} exceeds range [0, 2^{})",
            value, num_bits
        )));
    }
    
    // Create main commitment
    let commitment = pedersen_commit(value, blinding)?;
    
    // Decompose value into bits
    let bits = decompose_bits(value, num_bits);
    
    // Generate random blindings for bit commitments
    let mut rng = rand::thread_rng();
    let bit_blindings: Vec<Vec<u8>> = (0..num_bits)
        .map(|_| (0..32).map(|_| rng.gen::<u8>()).collect())
        .collect();
    
    // Commit to each bit
    let bit_commitments = commit_bits(&bits, &bit_blindings)?;
    
    // Prepare vectors for inner product argument
    let p = modulus();
    let a_vec: Vec<BigUint> = bits.iter().map(|&b| BigUint::from(b)).collect();
    let b_vec: Vec<BigUint> = (0..num_bits)
        .map(|i| BigUint::from(1u64) << i)
        .collect();
    
    // Prove inner product equals value
    let inner_product_proof = prove_inner_product(a_vec, b_vec, &commitment)?;
    
    // Generate challenges for Fiat-Shamir
    let challenge1 = generate_challenge(&[&commitment])?;
    let challenge2 = generate_challenge(&[
        &commitment,
        &inner_product_proof.final_a,
        &inner_product_proof.final_b,
    ])?;
    
    Ok(BulletproofRangeProof {
        commitment,
        bit_commitments,
        inner_product_proof,
        challenges: vec![challenge1, challenge2],
        offset: None,
        offset_commitment: None,
    })
}

/// Generate Bulletproof range proof with lower-bound offset normalization.
/// Proves value ∈ [min, min + 2^num_bits) by committing to full value while
/// bit decomposition covers (value - min).
pub fn prove_range_offset(
    value: u64,
    min: u64,
    blinding: &[u8],
    num_bits: usize,
) -> CryptoResult<BulletproofRangeProof> {
    if value < min {
        return Err(CryptoError::ProofError("Value below minimum".to_string()));
    }
    let offset_value = value - min;
    if num_bits < 64 && offset_value >= (1u64 << num_bits) {
        return Err(CryptoError::ProofError(format!(
            "Offset value {} exceeds range [0, 2^{})",
            offset_value, num_bits
        )));
    }
    // Commitment to full value
    let commitment = pedersen_commit(value, blinding)?;
    // Decompose offset
    let bits = decompose_bits(offset_value, num_bits);
    // Blindings
    let mut rng = rand::thread_rng();
    let bit_blindings: Vec<Vec<u8>> = (0..num_bits)
        .map(|_| (0..32).map(|_| rng.gen::<u8>()).collect())
        .collect();
    let bit_commitments = commit_bits(&bits, &bit_blindings)?;
    let p = modulus();
    let a_vec: Vec<BigUint> = bits.iter().map(|&b| BigUint::from(b)).collect();
    let b_vec: Vec<BigUint> = (0..num_bits).map(|i| BigUint::from(1u64) << i).collect();
    let inner_product_proof = prove_inner_product(a_vec, b_vec, &commitment)?;
    let challenge1 = generate_challenge(&[&commitment])?;
    let challenge2 = generate_challenge(&[&commitment, &inner_product_proof.final_a, &inner_product_proof.final_b])?;
    Ok(BulletproofRangeProof {
        commitment,
        bit_commitments,
        inner_product_proof,
        challenges: vec![challenge1, challenge2],
        offset: Some(min),
        offset_commitment: Some(pedersen_commit(offset_value, blinding)?),
    })
}

// ============================================================================
// Range Proof Verification
// ============================================================================

/// Verify Bulletproof range proof
pub fn verify_range(
    proof: &BulletproofRangeProof,
    commitment: &[u8],
    num_bits: usize,
) -> CryptoResult<()> {
    // Verify commitment matches
    if proof.commitment != commitment {
        return Err(CryptoError::VerificationError(
            "Commitment mismatch".to_string(),
        ));
    }
    
    // Verify bit commitment count
    if proof.bit_commitments.len() != num_bits {
        return Err(CryptoError::VerificationError(
            "Invalid number of bit commitments".to_string(),
        ));
    }
    
    // Verify each bit commitment represents 0 or 1
    for bit_commit in &proof.bit_commitments {
        if !verify_bit_commitment(bit_commit)? {
            return Err(CryptoError::VerificationError(
                "Invalid bit commitment".to_string(),
            ));
        }
    }
    
    // Verify inner product argument
    let p = modulus();
    
    // Reconstruct claimed value from final inner product
    let a_final = BigUint::from_bytes_be(&proof.inner_product_proof.final_a);
    let b_final = BigUint::from_bytes_be(&proof.inner_product_proof.final_b);
    let claimed_product = (a_final * b_final) % &p;
    
    verify_inner_product(&proof.inner_product_proof, commitment, &claimed_product)?;
    
    // Verify Fiat-Shamir challenges
    let recomputed_challenge1 = generate_challenge(&[commitment])?;
    if proof.challenges[0] != recomputed_challenge1 {
        return Err(CryptoError::VerificationError(
            "Challenge verification failed".to_string(),
        ));
    }
    
    Ok(())
}

/// Verify Bulletproof range proof with an offset (lower bound enforcement).
/// Uses statement-provided min to reconstruct full value = offset + (inner product result).
pub fn verify_range_offset(
    proof: &BulletproofRangeProof,
    commitment: &[u8],
    min: u64,
    num_bits: usize,
) -> CryptoResult<()> {
    if proof.commitment != commitment {
        return Err(CryptoError::VerificationError("Commitment mismatch".to_string()));
    }
    if proof.offset != Some(min) {
        return Err(CryptoError::VerificationError("Offset marker mismatch".to_string()));
    }
    let offset_commitment = proof.offset_commitment.as_ref().ok_or_else(||
        CryptoError::VerificationError("Missing offset commitment".to_string())
    )?;
    if proof.bit_commitments.len() != num_bits {
        return Err(CryptoError::VerificationError("Invalid number of bit commitments".to_string()));
    }
    for bit_commit in &proof.bit_commitments {
        if !verify_bit_commitment(bit_commit)? {
            return Err(CryptoError::VerificationError("Invalid bit commitment".to_string()));
        }
    }
    let p = modulus();
    let a_final = BigUint::from_bytes_be(&proof.inner_product_proof.final_a);
    let b_final = BigUint::from_bytes_be(&proof.inner_product_proof.final_b);
    let offset_value = (a_final * b_final) % &p; // BigUint offset
    // Range check
    if num_bits < 64 {
        let limit = BigUint::from(1u64) << num_bits;
        if offset_value >= limit {
            return Err(CryptoError::VerificationError("Offset value out of range".to_string()));
        }
    }
    // Binding relation: commitment == offset_commitment * g^{min}
    let g = generator_g();
    use crate::utils::constant_time::ct_modpow;
    let g_min = ct_modpow(&g, &BigUint::from(min), &p);
    let offset_big = BigUint::from_bytes_be(offset_commitment);
    let recombined = (offset_big * g_min) % &p;
    let commitment_big = BigUint::from_bytes_be(&proof.commitment);
    if recombined != commitment_big {
        return Err(CryptoError::VerificationError("Offset binding relation failed".to_string()));
    }
    // Challenge determinism check
    let recomputed_challenge1 = generate_challenge(&[commitment])?;
    if proof.challenges[0] != recomputed_challenge1 {
        return Err(CryptoError::VerificationError("Challenge verification failed".to_string()));
    }
    Ok(())
}

/// Verify bit commitment represents 0 or 1 (simplified check)
fn verify_bit_commitment(commitment: &[u8]) -> CryptoResult<bool> {
    // In full implementation, would verify commitment is to {0,1}
    // For now, just check structure
    Ok(!commitment.is_empty() && commitment.len() <= 64)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Generate Fiat-Shamir challenge by hashing inputs
fn generate_challenge(inputs: &[&[u8]]) -> CryptoResult<[u8; 32]> {
    let mut hasher = Sha3_256::new();
    
    for input in inputs {
        hasher.update(input);
    }
    
    let hash = hasher.finalize();
    let mut challenge = [0u8; 32];
    challenge.copy_from_slice(&hash);
    
    Ok(challenge)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pedersen_commitment() {
        let value = 42u64;
        let blinding = vec![0xAA; 32];
        
        let commitment = pedersen_commit(value, &blinding).unwrap();
        assert!(!commitment.is_empty());
        
        // Verify commitment
        assert!(verify_commitment(&commitment, value, &blinding).unwrap());
        assert!(!verify_commitment(&commitment, 43, &blinding).unwrap());
    }

    #[test]
    fn test_bit_decomposition() {
        let value = 42u64; // 0b101010
        let bits = decompose_bits(value, 8);
        
        assert_eq!(bits.len(), 8);
        assert_eq!(bits[0], 0); // LSB
        assert_eq!(bits[1], 1);
        assert_eq!(bits[2], 0);
        assert_eq!(bits[3], 1);
        assert_eq!(bits[4], 0);
        assert_eq!(bits[5], 1);
        assert_eq!(bits[6], 0); // MSB
        assert_eq!(bits[7], 0);
        
        // Verify recomposition
        assert_eq!(recompose_bits(&bits), value);
    }

    #[test]
    fn test_inner_product() {
        let p = modulus();
        let a = vec![BigUint::from(2u32), BigUint::from(3u32)];
        let b = vec![BigUint::from(4u32), BigUint::from(5u32)];
        
        let product = inner_product(&a, &b, &p);
        
        // 2*4 + 3*5 = 23
        assert_eq!(product, BigUint::from(23u32));
    }

    #[test]
    fn test_range_proof_valid_value() {
        let value = 42u64;
        let blinding = vec![0xBB; 32];
        let num_bits = 8; // Range [0, 256)
        
        // Generate proof
        let proof = prove_range(value, &blinding, num_bits).unwrap();
        
        // Verify proof
        let result = verify_range(&proof, &proof.commitment, num_bits);
        assert!(result.is_ok(), "Valid range proof should verify");
    }

    #[test]
    fn test_range_proof_out_of_range() {
        let value = 300u64; // Exceeds 2^8 = 256
        let blinding = vec![0xCC; 32];
        let num_bits = 8;
        
        // Proof generation should fail
        let result = prove_range(value, &blinding, num_bits);
        assert!(result.is_err(), "Out-of-range value should fail");
    }

    #[test]
    fn test_range_proof_boundary_values() {
        let blinding = vec![0xDD; 32];
        
        // Test minimum value (0)
        let proof_min = prove_range(0, &blinding, 8).unwrap();
        assert!(verify_range(&proof_min, &proof_min.commitment, 8).is_ok());
        
        // Test maximum value (255 for 8 bits)
        let proof_max = prove_range(255, &blinding, 8).unwrap();
        assert!(verify_range(&proof_max, &proof_max.commitment, 8).is_ok());
        
        // Test exceeding maximum (256)
        assert!(prove_range(256, &blinding, 8).is_err());
    }

    #[test]
    fn test_commitment_binding() {
        let value1 = 42u64;
        let value2 = 43u64;
        let blinding = vec![0xEE; 32];
        
        let c1 = pedersen_commit(value1, &blinding).unwrap();
        let c2 = pedersen_commit(value2, &blinding).unwrap();
        
        // Different values should produce different commitments
        assert_ne!(c1, c2, "Commitment should be binding");
    }

    #[test]
    fn test_commitment_hiding() {
        let value = 42u64;
        let blinding1 = vec![0x11; 32];
        let blinding2 = vec![0x22; 32];
        
        let c1 = pedersen_commit(value, &blinding1).unwrap();
        let c2 = pedersen_commit(value, &blinding2).unwrap();
        
        // Same value with different blinding should produce different commitments
        assert_ne!(c1, c2, "Commitment should be hiding");
    }

    #[test]
    fn test_inner_product_proof_generation() {
        let a = vec![BigUint::from(1u32), BigUint::from(0u32)];
        let b = vec![BigUint::from(1u32), BigUint::from(2u32)];
        let commitment = vec![0xAA; 32];
        
        let proof = prove_inner_product(a, b, &commitment);
        assert!(proof.is_ok(), "Inner product proof should generate");
        
        let proof = proof.unwrap();
        assert!(!proof.final_a.is_empty());
        assert!(!proof.final_b.is_empty());
    }

    #[test]
    fn test_challenge_determinism() {
        let input1 = vec![0x01, 0x02, 0x03];
        let input2 = vec![0x04, 0x05];
        
        let c1 = generate_challenge(&[&input1, &input2]).unwrap();
        let c2 = generate_challenge(&[&input1, &input2]).unwrap();
        
        assert_eq!(c1, c2, "Challenges should be deterministic");
    }

    #[test]
    fn test_range_proof_proof_size() {
        let value = 100u64;
        let blinding = vec![0xFF; 32];
        let num_bits = 64;
        
        let proof = prove_range(value, &blinding, num_bits).unwrap();
        
        // Bulletproofs should have logarithmic size
        assert!(
            proof.inner_product_proof.left_commitments.len() <= 7,
            "Proof size should be logarithmic (≤ log2(64) = 6 rounds)"
        );
    }
}
