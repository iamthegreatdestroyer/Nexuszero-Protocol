//! Witness structures for zero-knowledge proofs
//!
//! A Witness represents the SECRET knowledge that proves a statement.
//! This must NEVER be transmitted or stored insecurely.

use crate::proof::statement::{HashFunction, Statement, StatementType};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Witness type indicator
#[derive(Clone, Debug)]
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
#[derive(Zeroize, ZeroizeOnDrop)]
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

    /// Validate witness satisfies the statement
    pub fn satisfies_statement(&self, statement: &Statement) -> bool {
        match (&self.secret_data, &statement.statement_type) {
            (SecretData::Preimage(pre), StatementType::Preimage { hash_function, hash_output }) => {
                self.verify_preimage(hash_function, pre, hash_output)
            }
            (SecretData::DiscreteLog(secret), StatementType::DiscreteLog { generator, public_value }) => {
                // Verify that generator^secret = public_value (mod p)
                use num_bigint::BigUint;
                
                let modulus_bytes = vec![0xFF; 32];
                let gen_big = BigUint::from_bytes_be(generator);
                let secret_big = BigUint::from_bytes_be(secret);
                let mod_big = BigUint::from_bytes_be(&modulus_bytes);
                let public_big = BigUint::from_bytes_be(public_value);
                
                let computed = gen_big.modpow(&secret_big, &mod_big);
                computed == public_big
            }
            (
                SecretData::Range { value, .. },
                StatementType::Range {
                    min,
                    max,
                    commitment: _,
                },
            ) => *value >= *min && *value <= *max,
            _ => false,
        }
    }

    /// Get witness type
    pub fn witness_type(&self) -> &WitnessType {
        &self.witness_type
    }

    /// Access randomness (used in proof generation)
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

/// Constant-time byte array equality
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }

    result == 0
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
        let generator = vec![2u8;32];
        let secret = vec![5u8;32];
        // public_value computed with different secret to force mismatch
        let wrong_secret = vec![7u8;32];
        let modulus_bytes = vec![0xFF;32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let wrong_big = BigUint::from_bytes_be(&wrong_secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = gen_big.modpow(&wrong_big, &mod_big).to_bytes_be();
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
        // Discrete log statement
        let generator = vec![2u8;32];
        let secret = vec![3u8;32];
        let modulus_bytes = vec![0xFF;32];
        let gen_big = BigUint::from_bytes_be(&generator);
        let secret_big = BigUint::from_bytes_be(&secret);
        let mod_big = BigUint::from_bytes_be(&modulus_bytes);
        let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
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
        // Range [10, 20]; test value at min and max boundaries
        let statement = StatementBuilder::new().range(10, 20, vec![0u8;32]).build().unwrap();
        let min_witness = Witness::range(10, vec![0xAA;16]);
        let max_witness = Witness::range(20, vec![0xBB;16]);
        assert!(min_witness.satisfies_statement(&statement));
        assert!(max_witness.satisfies_statement(&statement));
    }

    #[test]
    fn test_constant_time_eq_different_lengths() {
        let a = vec![1,2,3];
        let b = vec![1,2,3,4];
        assert!(!constant_time_eq(&a,&b));
    }
}
