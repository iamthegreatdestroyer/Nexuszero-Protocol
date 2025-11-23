//! Statement structures for zero-knowledge proofs
//!
//! A Statement represents the public claim being proven.

use crate::{CryptoError, CryptoResult};
use serde::{Deserialize, Serialize};

/// Types of statements that can be proven
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum StatementType {
    /// Prove knowledge of discrete log: g^x = h
    DiscreteLog {
        /// Generator
        generator: Vec<u8>,
        /// Public value
        public_value: Vec<u8>,
    },

    /// Prove knowledge of hash preimage: H(x) = y
    Preimage {
        /// Hash function used
        hash_function: HashFunction,
        /// Hash output
        hash_output: Vec<u8>,
    },

    /// Prove x ∈ [min, max] (range proof)
    Range {
        /// Minimum value
        min: u64,
        /// Maximum value
        max: u64,
        /// Commitment to value
        commitment: Vec<u8>,
    },

    /// Custom statement (placeholder)
    Custom {
        /// Description
        description: String,
    },
}

/// Supported hash functions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum HashFunction {
    /// SHA3-256
    SHA3_256,
    /// SHA-256
    SHA256,
    /// BLAKE3
    Blake3,
}

/// Complete statement for a zero-knowledge proof
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Statement {
    /// Type and parameters of the statement
    pub statement_type: StatementType,
    /// Version for future compatibility
    pub version: u8,
}

impl Statement {
    /// Validate statement consistency
    pub fn validate(&self) -> CryptoResult<()> {
        match &self.statement_type {
            StatementType::DiscreteLog { generator, public_value } => {
                if generator.is_empty() || public_value.is_empty() {
                    return Err(CryptoError::InvalidParameter(
                        "Discrete log statement requires non-empty values".to_string(),
                    ));
                }
            }
            StatementType::Preimage { hash_output, .. } => {
                if hash_output.is_empty() {
                    return Err(CryptoError::InvalidParameter(
                        "Preimage statement requires non-empty hash".to_string(),
                    ));
                }
            }
            StatementType::Range { min, max, .. } => {
                if min >= max {
                    return Err(CryptoError::InvalidParameter(
                        "Range min must be less than max".to_string(),
                    ));
                }
            }
            StatementType::Custom { .. } => {}
        }
        Ok(())
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> CryptoResult<Vec<u8>> {
        bincode::serialize(self).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to serialize statement: {}", e))
        })
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> CryptoResult<Self> {
        bincode::deserialize(bytes).map_err(|e| {
            CryptoError::SerializationError(format!("Failed to deserialize statement: {}", e))
        })
    }

    /// Compute hash of statement
    pub fn hash(&self) -> CryptoResult<[u8; 32]> {
        use sha3::{Digest, Sha3_256};

        let bytes = self.to_bytes()?;
        let mut hasher = Sha3_256::new();
        hasher.update(&bytes);
        let result = hasher.finalize();

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        Ok(hash)
    }
}

/// Builder for constructing statements
pub struct StatementBuilder {
    statement_type: Option<StatementType>,
}

impl StatementBuilder {
    /// Create a new statement builder
    pub fn new() -> Self {
        Self {
            statement_type: None,
        }
    }

    /// Build a discrete log statement
    pub fn discrete_log(mut self, generator: Vec<u8>, public_value: Vec<u8>) -> Self {
        self.statement_type = Some(StatementType::DiscreteLog {
            generator,
            public_value,
        });
        self
    }

    /// Build a preimage statement
    pub fn preimage(mut self, hash_function: HashFunction, hash_output: Vec<u8>) -> Self {
        self.statement_type = Some(StatementType::Preimage {
            hash_function,
            hash_output,
        });
        self
    }

    /// Build a range statement
    pub fn range(mut self, min: u64, max: u64, commitment: Vec<u8>) -> Self {
        self.statement_type = Some(StatementType::Range {
            min,
            max,
            commitment,
        });
        self
    }

    /// Build the statement
    pub fn build(self) -> CryptoResult<Statement> {
        let statement_type = self.statement_type.ok_or_else(|| {
            CryptoError::InvalidParameter("Statement type not set".to_string())
        })?;

        let statement = Statement {
            statement_type,
            version: 1,
        };

        statement.validate()?;
        Ok(statement)
    }
}

impl Default for StatementBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_statement_builder() {
        let stmt = StatementBuilder::new()
            .discrete_log(vec![1, 2, 3], vec![4, 5, 6])
            .build()
            .unwrap();

        assert!(matches!(stmt.statement_type, StatementType::DiscreteLog { .. }));
    }

    #[test]
    fn test_statement_serialization() {
        let stmt = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, vec![0u8; 32])
            .build()
            .unwrap();

        let bytes = stmt.to_bytes().unwrap();
        let recovered = Statement::from_bytes(&bytes).unwrap();

        assert_eq!(stmt.version, recovered.version);
    }

    #[test]
    fn test_invalid_discrete_log_statement() {
        let result = StatementBuilder::new()
            .discrete_log(vec![], vec![1,2,3])
            .build();
        assert!(result.is_err());
        let result2 = StatementBuilder::new()
            .discrete_log(vec![1,2,3], vec![])
            .build();
        assert!(result2.is_err());
    }

    #[test]
    fn test_invalid_preimage_statement_empty_hash() {
        let result = StatementBuilder::new()
            .preimage(HashFunction::SHA3_256, vec![])
            .build();
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_range_statement_min_greater_equal_max() {
        let result = StatementBuilder::new()
            .range(10, 10, vec![0u8;32])
            .build();
        assert!(result.is_err());
        let result2 = StatementBuilder::new()
            .range(20, 10, vec![0u8;32])
            .build();
        assert!(result2.is_err());
    }

    #[test]
    fn test_builder_missing_type_rejection() {
        let builder = StatementBuilder::new();
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_statement_hash_length() {
        let stmt = StatementBuilder::new()
            .discrete_log(vec![1,2,3], vec![4,5,6])
            .build()
            .unwrap();
        let hash = stmt.hash().unwrap();
        assert_eq!(hash.len(), 32);
    }
}
