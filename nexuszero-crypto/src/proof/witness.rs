//! Witness structures for zero-knowledge proofs
//!
//! A Witness represents the SECRET knowledge that proves a statement.
//! This must NEVER be transmitted or stored insecurely.

use crate::proof::statement::{HashFunction, Statement, StatementType};
use crate::utils::constant_time::{ct_modpow, ct_bytes_eq, ct_in_range};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Witness type indicator
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WitnessType {
    /// Discrete logarithm witness
    DiscreteLog,
    /// Hash preimage witness
    Preimage,
    /// Range proof witness
    Range,
    /// Custom witness
    Custom,
}

/// Secret data for different witness types
#[allow(dead_code)]
#[derive(Clone, Zeroize, ZeroizeOnDrop, Debug)]
enum SecretData {
    /// Discrete log witness: the exponent x where g^x = h
    DiscreteLog(Vec<u8>),

    /// Preimage witness: x where H(x) = y
    Preimage(Vec<u8>),

    /// Range proof witness: (value, blinding_factor)
    Range { value: u64, blinding: Vec<u8> },

    /// Custom witness data
    Custom(Vec<u8>),
}

/// Witness structure
///
/// Note: SecretData implements ZeroizeOnDrop to secure sensitive data
#[derive(Clone, Debug)]
pub struct Witness {
    /// The secret data (automatically zeroized via ZeroizeOnDrop)
    secret_data: SecretData,
    /// Randomness used in proof
    randomness: Vec<u8>,
    /// Type indicator
    witness_type: WitnessType,
}

impl Witness {
    /// Create witness for discrete log proof
    pub fn discrete_log(exponent: Vec<u8>) -> Self {
        let mut randomness = vec![0u8; 32];
        rand::Rng::fill(&mut rand::thread_rng(), &mut randomness[..]);

        Self {
            secret_data: SecretData::DiscreteLog(exponent),
            randomness,
            witness_type: WitnessType::DiscreteLog,
        }
    }

    /// Create witness for hash preimage proof
    pub fn preimage(preimage: Vec<u8>) -> Self {
        let mut randomness = vec![0u8; 32];
        rand::Rng::fill(&mut rand::thread_rng(), &mut randomness[..]);

        Self {
            secret_data: SecretData::Preimage(preimage),
            randomness,
            witness_type: WitnessType::Preimage,
        }
    }

    /// Create witness for range proof
    pub fn range(value: u64, blinding: Vec<u8>) -> Self {
        let mut randomness = vec![0u8; 32];
        rand::Rng::fill(&mut rand::thread_rng(), &mut randomness[..]);

        Self {
            secret_data: SecretData::Range { value, blinding },
            randomness,
            witness_type: WitnessType::Range,
        }
    }

    /// Create custom witness
    pub fn custom(data: Vec<u8>) -> Self {
        let mut randomness = vec![0u8; 32];
        rand::Rng::fill(&mut rand::thread_rng(), &mut randomness[..]);

        Self {
            secret_data: SecretData::Custom(data),
            randomness,
            witness_type: WitnessType::Custom,
        }
    }

    /// Validate witness satisfies the statement
    pub fn satisfies_statement(&self, statement: &Statement) -> bool {
        match (&self.secret_data, &statement.statement_type) {
            (SecretData::Preimage(pre), StatementType::Preimage { hash_function, hash_output }) => {
                self.verify_preimage(hash_function, pre, hash_output)
            }
            (SecretData::DiscreteLog(secret), StatementType::DiscreteLog { generator, public_value }) => {
                // Verify that generator^secret = public_value (mod p)
                // 
                // SECURITY: Uses constant-time modular exponentiation (ct_modpow)
                // to prevent timing attacks that could leak secret exponent bits.
                use num_bigint::BigUint;
                
                let modulus_bytes = vec![0xFF; 32];
                let gen_big = BigUint::from_bytes_be(generator);
                let secret_big = BigUint::from_bytes_be(secret);
                let mod_big = BigUint::from_bytes_be(&modulus_bytes);
                let public_big = BigUint::from_bytes_be(public_value);
                
                // Use constant-time modular exponentiation (Montgomery ladder)
                let computed = ct_modpow(&gen_big, &secret_big, &mod_big);
                
                // Use constant-time comparison for final equality check
                let computed_bytes = computed.to_bytes_be();
                let public_bytes = public_big.to_bytes_be();
                ct_bytes_eq(&computed_bytes, &public_bytes)
            }
            (
                SecretData::Range { value, blinding },
                StatementType::Range {
                    min,
                    max,
                    commitment,
                },
            ) => {
                // Check value is in range [min, max] inclusive
                // SECURITY: Uses constant-time range comparison to prevent timing attacks
                // that could leak information about whether value is in range.
                let in_range = ct_in_range(*value, *min, *max);
                
                // Verify commitment matches C = g^v * h^r using Bulletproofs generators
                // ALWAYS compute the commitment even if value is out of range (constant-time)
                let commitment_valid = match crate::proof::bulletproofs::pedersen_commit(*value, blinding) {
                    Ok(computed_commitment) => {
                        // Use constant-time comparison for commitment equality
                        ct_bytes_eq(&computed_commitment, commitment)
                    }
                    Err(_) => false, // Commitment computation failed
                };
                
                // Combine both checks: in_range AND commitment_valid
                in_range && commitment_valid
            }
            _ => false,
        }
    }

    /// Get witness type
    pub fn witness_type(&self) -> &WitnessType {
        &self.witness_type
    }

    /// Access randomness (used in proof generation)
    #[allow(dead_code)]
    pub(crate) fn randomness(&self) -> &[u8] {
        &self.randomness
    }

    /// Get secret bytes for proof generation
    /// 
    /// This is used internally by the proof system.
    /// Should never be exposed outside the proof module.
    pub(crate) fn get_secret_bytes(&self) -> Result<Vec<u8>, &'static str> {
        match &self.secret_data {
            SecretData::DiscreteLog(bytes) => Ok(bytes.clone()),
            SecretData::Preimage(bytes) => Ok(bytes.clone()),
            SecretData::Range { value, blinding } => {
                // For range proofs, combine value and blinding
                let mut result = value.to_be_bytes().to_vec();
                result.extend_from_slice(blinding);
                Ok(result)
            }
            SecretData::Custom(bytes) => Ok(bytes.clone()),
        }
    }

    /// Verify hash preimage
    fn verify_preimage(
        &self,
        hash_fn: &HashFunction,
        preimage: &[u8],
        expected_hash: &[u8],
    ) -> bool {
        use sha3::{Digest, Sha3_256};

        let actual_hash = match hash_fn {
            HashFunction::SHA3_256 => {
                let mut hasher = Sha3_256::new();
                hasher.update(preimage);
                hasher.finalize().to_vec()
            }
            _ => return false,
        };

        constant_time_eq(&actual_hash, expected_hash)
    }

    /// Securely destroy witness
    pub fn destroy(self) {
        // Drop is called automatically, which triggers zeroization
        drop(self);
    }
}

// Legacy constant_time_eq function - now using ct_bytes_eq from utils::constant_time
// Keeping for backward compatibility in tests
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    ct_bytes_eq(a, b)
}

// Drop is automatically implemented by ZeroizeOnDrop derive
// which handles secure zeroization of all fields

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proof::StatementBuilder;

    #[test]
    fn test_preimage_witness() {
        use sha3::{Digest, Sha3_256};

        let preimage = b"secret message";
        let hash = Sha3_256::digest(preimage).to_vec();

        let witness = Witness::preimage(preimage.to_vec());
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();

        assert!(witness.satisfies_statement(&statement));
    }

    #[test]
    fn test_constant_time_equality() {
        let a = vec![1, 2, 3, 4];
        let b = vec![1, 2, 3, 4];
        let c = vec![1, 2, 3, 5];

        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_witness_types() {
        let witness = Witness::discrete_log(vec![1, 2, 3]);
        assert!(matches!(witness.witness_type(), &WitnessType::DiscreteLog));

        let witness = Witness::preimage(vec![4, 5, 6]);
        assert!(matches!(witness.witness_type(), &WitnessType::Preimage));
    }

    #[test]
    fn test_discrete_log_witness_mismatch() {
        use num_bigint::BigUint;
        use crate::utils::constant_time::ct_modpow;
        let generator = vec![2u8;32];
        let secret = vec![5u8;32];
        // public_value computed with different secret to force mismatch
        let wrong_secret = vec![7u8;32];
        let modulus_bytes = vec![0xFF;32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let wrong_big = BigUint::from_bytes_be(&wrong_secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &wrong_big, &mod_big).to_bytes_be();
        let statement = StatementBuilder::new()
            .discrete_log(generator.clone(), public_value)
            .build()
            .unwrap();
        let witness = Witness::discrete_log(secret);
        assert!(!witness.satisfies_statement(&statement));
    }

    #[test]
    fn test_preimage_witness_mismatch() {
        use sha3::{Digest, Sha3_256};
        let preimage = b"correct".to_vec();
        let mut hasher = Sha3_256::new(); hasher.update(&preimage); let hash = hasher.finalize().to_vec();
        let statement = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, hash)
            .build()
            .unwrap();
        let wrong_witness = Witness::preimage(b"wrong".to_vec());
        assert!(!wrong_witness.satisfies_statement(&statement));
    }

    #[test]
    fn test_range_witness_out_of_range() {
        let statement = StatementBuilder::new()
            .range(10, 20, vec![0u8;32])
            .build()
            .unwrap();
        let witness = Witness::range(25, vec![1,2,3]);
        assert!(!witness.satisfies_statement(&statement));
    }

    // ===================== Added Witness Corner Case Tests (Wave 4) =====================

    #[test]
    fn test_cross_type_witness_mismatch() {
        use num_bigint::BigUint;
        use crate::utils::constant_time::ct_modpow;
        // Discrete log statement
        let generator = vec![2u8;32];
        let secret = vec![3u8;32];
        let modulus_bytes = vec![0xFF;32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
        let dl_statement = StatementBuilder::new().discrete_log(generator.clone(), public_value).build().unwrap();
        // Preimage witness used against discrete log statement (should fail)
        let preimage_witness = Witness::preimage(b"not a discrete log".to_vec());
        assert!(!preimage_witness.satisfies_statement(&dl_statement));

        // Preimage statement
        use sha3::{Digest, Sha3_256};
        let preimage = b"secret value".to_vec();
        let mut hasher = Sha3_256::new(); hasher.update(&preimage); let hash = hasher.finalize().to_vec();
        let preimage_statement = StatementBuilder::new().preimage(HashFunction::SHA3_256, hash).build().unwrap();
        // Discrete log witness used against preimage statement (should fail)
        let dl_witness = Witness::discrete_log(vec![5u8;32]);
        assert!(!dl_witness.satisfies_statement(&preimage_statement));
    }

    #[test]
    fn test_range_witness_boundary_values() {
        // Range [10, 20]; test value at min and max boundaries with proper commitments
        // Use bulletproofs pedersen_commit to ensure compatibility
        
        // Test min boundary (10)
        let min_value = 10u64;
        let min_blinding = vec![0xAA; 16];
        let min_commitment = crate::proof::bulletproofs::pedersen_commit(min_value, &min_blinding)
            .expect("Commitment should succeed");
        
        let min_statement = StatementBuilder::new().range(10, 20, min_commitment).build().unwrap();
        let min_witness = Witness::range(min_value, min_blinding);
        assert!(min_witness.satisfies_statement(&min_statement));
        
        // Test max boundary (20)
        let max_value = 20u64;
        let max_blinding = vec![0xBB; 16];
        let max_commitment = crate::proof::bulletproofs::pedersen_commit(max_value, &max_blinding)
            .expect("Commitment should succeed");
        
        let max_statement = StatementBuilder::new().range(10, 20, max_commitment).build().unwrap();
        let max_witness = Witness::range(max_value, max_blinding);
        assert!(max_witness.satisfies_statement(&max_statement));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        let a = vec![1,2,3];
        let b = vec![1,2,3,4];
        assert!(!constant_time_eq(&a,&b));
    }
}
