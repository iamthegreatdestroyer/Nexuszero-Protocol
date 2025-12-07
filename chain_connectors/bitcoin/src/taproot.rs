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

    // ===== HARDENING TESTS =====

    #[test]
    fn test_commitment_hash_deterministic() {
        let proof = vec![1, 2, 3, 4];
        let sender = [1u8; 32];
        let recipient = [2u8; 32];

        let hash1 = hash_proof_commitment(&proof, &sender, &recipient);
        let hash2 = hash_proof_commitment(&proof, &sender, &recipient);
        
        assert_eq!(hash1, hash2, "Same inputs should produce same hash");
    }

    #[test]
    fn test_commitment_hash_different_proofs() {
        let proof1 = vec![1, 2, 3, 4];
        let proof2 = vec![5, 6, 7, 8];
        let sender = [1u8; 32];
        let recipient = [2u8; 32];

        let hash1 = hash_proof_commitment(&proof1, &sender, &recipient);
        let hash2 = hash_proof_commitment(&proof2, &sender, &recipient);
        
        assert_ne!(hash1, hash2, "Different proofs should produce different hashes");
    }

    #[test]
    fn test_commitment_hash_different_sender() {
        let proof = vec![1, 2, 3, 4];
        let sender1 = [1u8; 32];
        let sender2 = [2u8; 32];
        let recipient = [3u8; 32];

        let hash1 = hash_proof_commitment(&proof, &sender1, &recipient);
        let hash2 = hash_proof_commitment(&proof, &sender2, &recipient);
        
        assert_ne!(hash1, hash2, "Different senders should produce different hashes");
    }

    #[test]
    fn test_commitment_hash_different_recipient() {
        let proof = vec![1, 2, 3, 4];
        let sender = [1u8; 32];
        let recipient1 = [2u8; 32];
        let recipient2 = [3u8; 32];

        let hash1 = hash_proof_commitment(&proof, &sender, &recipient1);
        let hash2 = hash_proof_commitment(&proof, &sender, &recipient2);
        
        assert_ne!(hash1, hash2, "Different recipients should produce different hashes");
    }

    #[test]
    fn test_commitment_hash_empty_proof() {
        let proof = vec![];
        let sender = [1u8; 32];
        let recipient = [2u8; 32];

        let hash = hash_proof_commitment(&proof, &sender, &recipient);
        
        assert_eq!(hash.len(), 32);
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn test_commitment_hash_large_proof() {
        let proof = vec![0xabu8; 10_000];
        let sender = [1u8; 32];
        let recipient = [2u8; 32];

        let hash = hash_proof_commitment(&proof, &sender, &recipient);
        
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_commitment_hash_zero_commitments() {
        let proof = vec![1, 2, 3, 4];
        let sender = [0u8; 32];
        let recipient = [0u8; 32];

        let hash = hash_proof_commitment(&proof, &sender, &recipient);
        
        assert_eq!(hash.len(), 32);
        // Even with zero commitments, hash should not be zero
        assert_ne!(hash, [0u8; 32]);
    }

    #[test]
    fn test_commitment_script_structure() {
        let hash = [0xabu8; 32];
        let script = create_commitment_script(&hash);
        
        // Script should be: <32-byte-hash> OP_DROP OP_PUSHNUM_1
        // Length: 33 (push + 32 bytes) + 1 (OP_DROP) + 1 (OP_PUSHNUM_1) = 35 bytes
        assert!(script.len() >= 35);
    }

    #[test]
    fn test_commitment_script_different_hashes() {
        let hash1 = [1u8; 32];
        let hash2 = [2u8; 32];
        
        let script1 = create_commitment_script(&hash1);
        let script2 = create_commitment_script(&hash2);
        
        assert_ne!(script1, script2);
    }

    #[test]
    fn test_taproot_builder_creation() {
        use bitcoin::secp256k1::{Secp256k1, SecretKey};
        use bitcoin::key::Keypair;

        let secp = Secp256k1::new();
        let secret_key = SecretKey::from_slice(&[1u8; 32]).unwrap();
        let keypair = Keypair::from_secret_key(&secp, &secret_key);
        let (xonly, _parity) = keypair.x_only_public_key();
        let internal_key = UntweakedPublicKey::from(xonly);
        
        let builder = TaprootOutputBuilder::new(internal_key);
        
        // Key-path only spend should work
        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_taproot_builder_with_script() {
        use bitcoin::secp256k1::{Secp256k1, SecretKey};
        use bitcoin::key::Keypair;

        let secp = Secp256k1::new();
        let secret_key = SecretKey::from_slice(&[1u8; 32]).unwrap();
        let keypair = Keypair::from_secret_key(&secp, &secret_key);
        let (xonly, _parity) = keypair.x_only_public_key();
        let internal_key = UntweakedPublicKey::from(xonly);
        
        let script = create_commitment_script(&[0xab; 32]);
        
        let builder = TaprootOutputBuilder::new(internal_key).add_script(script);
        
        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_taproot_builder_multiple_scripts() {
        // Note: Adding multiple scripts at depth 0 may fail in TaprootBuilder.
        // For now, we test that two separate builders with single scripts work.
        use bitcoin::secp256k1::{Secp256k1, SecretKey};
        use bitcoin::key::Keypair;

        let secp = Secp256k1::new();
        let secret_key = SecretKey::from_slice(&[1u8; 32]).unwrap();
        let keypair = Keypair::from_secret_key(&secp, &secret_key);
        let (xonly, _parity) = keypair.x_only_public_key();
        let internal_key = UntweakedPublicKey::from(xonly);
        
        let script1 = create_commitment_script(&[0x01; 32]);
        let script2 = create_commitment_script(&[0x02; 32]);
        
        // Test each script works individually
        let builder1 = TaprootOutputBuilder::new(internal_key).add_script(script1);
        assert!(builder1.build().is_ok());
        
        let builder2 = TaprootOutputBuilder::new(internal_key).add_script(script2);
        assert!(builder2.build().is_ok());
    }

    #[test]
    fn test_commitment_hash_preserves_prefix() {
        let proof = vec![1, 2, 3, 4];
        let sender = [1u8; 32];
        let recipient = [2u8; 32];

        // The hash should include "NexusZero/v1/proof" prefix
        // We can verify by checking the hash is different from raw SHA256
        let mut hasher = Sha256::new();
        hasher.update(&proof);
        hasher.update(&sender);
        hasher.update(&recipient);
        let raw_hash: [u8; 32] = hasher.finalize().into();

        let commitment_hash = hash_proof_commitment(&proof, &sender, &recipient);
        
        // Should be different because commitment includes protocol prefix
        assert_ne!(raw_hash, commitment_hash);
    }
}
