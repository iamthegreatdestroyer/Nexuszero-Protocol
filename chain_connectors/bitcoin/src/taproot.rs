//! Taproot utilities for Bitcoin transactions.

use bitcoin::hashes::Hash;
use bitcoin::key::UntweakedPublicKey;
use bitcoin::taproot::{TaprootBuilder, TaprootSpendInfo};
use bitcoin::ScriptBuf;
use sha2::{Digest, Sha256};

use crate::error::BitcoinError;

/// Build a NexusZero-specific Taproot output.
pub struct TaprootOutputBuilder {
    internal_key: UntweakedPublicKey,
    scripts: Vec<ScriptBuf>,
}

impl TaprootOutputBuilder {
    /// Create a new Taproot builder with internal key.
    pub fn new(internal_key: UntweakedPublicKey) -> Self {
        Self {
            internal_key,
            scripts: Vec::new(),
        }
    }

    /// Add a script to the Taproot tree.
    pub fn add_script(mut self, script: ScriptBuf) -> Self {
        self.scripts.push(script);
        self
    }

    /// Build the Taproot spend info.
    pub fn build(self) -> Result<TaprootSpendInfo, BitcoinError> {
        let secp = bitcoin::secp256k1::Secp256k1::new();

        if self.scripts.is_empty() {
            // Key-path only spend
            Ok(TaprootBuilder::new()
                .finalize(&secp, self.internal_key)
                .map_err(|_| BitcoinError::TaprootError("Failed to finalize taproot".to_string()))?)
        } else {
            // Script-path spend
            let mut builder = TaprootBuilder::new();

            for script in self.scripts {
                builder = builder
                    .add_leaf(0, script)
                    .map_err(|_| BitcoinError::TaprootError("Failed to add leaf".to_string()))?;
            }

            builder
                .finalize(&secp, self.internal_key)
                .map_err(|_| BitcoinError::TaprootError("Failed to finalize taproot".to_string()))
        }
    }
}

/// Create a commitment script for proof embedding.
pub fn create_commitment_script(proof_hash: &[u8; 32]) -> ScriptBuf {
    use bitcoin::opcodes::all::*;

    ScriptBuf::builder()
        .push_slice(proof_hash)
        .push_opcode(OP_DROP)
        .push_opcode(OP_PUSHNUM_1)
        .into_script()
}

/// Hash proof data to create a commitment.
pub fn hash_proof_commitment(
    proof: &[u8],
    sender_commitment: &[u8; 32],
    recipient_commitment: &[u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"NexusZero/v1/proof");
    hasher.update(proof);
    hasher.update(sender_commitment);
    hasher.update(recipient_commitment);
    
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commitment_hash() {
        let proof = vec![1, 2, 3, 4];
        let sender = [1u8; 32];
        let recipient = [2u8; 32];

        let hash = hash_proof_commitment(&proof, &sender, &recipient);
        assert_eq!(hash.len(), 32);
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn test_commitment_script() {
        let hash = [1u8; 32];
        let script = create_commitment_script(&hash);
        assert!(!script.is_empty());
    }
}
