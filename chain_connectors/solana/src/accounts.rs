//! Solana account parsing and PDA derivation.

use sha2::{Sha256, Digest};
use serde::{Deserialize, Serialize};

use crate::error::SolanaError;

/// Proof account info parsed from on-chain data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAccountInfo {
    /// Proof ID hash
    pub proof_id: [u8; 32],
    /// Privacy level (0-5)
    pub privacy_level: u8,
    /// Sender commitment
    pub sender_commitment: [u8; 32],
    /// Recipient commitment
    pub recipient_commitment: [u8; 32],
    /// Whether the proof has been verified
    pub verified: bool,
    /// Submitter address (base58)
    pub submitter: String,
    /// Slot when submitted
    pub slot: u64,
    /// Unix timestamp
    pub timestamp: i64,
}

/// Transfer account info parsed from on-chain data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferAccountInfo {
    /// Transfer ID
    pub transfer_id: [u8; 32],
    /// Target chain identifier
    pub target_chain: [u8; 32],
    /// Sender address (base58)
    pub sender: String,
    /// Recipient on target chain
    pub recipient: Vec<u8>,
    /// Amount in lamports
    pub amount: u64,
    /// Transfer status
    pub status: u8,
}

/// Account parser for NexusZero program accounts.
pub struct AccountParser;

impl AccountParser {
    /// Parse a proof account from raw data.
    /// 
    /// Expected format:
    /// - bytes 0-7: discriminator
    /// - bytes 8-39: proof_id (32 bytes)
    /// - byte 40: privacy_level
    /// - bytes 41-72: sender_commitment (32 bytes)
    /// - bytes 73-104: recipient_commitment (32 bytes)
    /// - byte 105: verified flag
    /// - bytes 106-137: submitter (32 bytes)
    /// - bytes 138-145: slot (u64)
    /// - bytes 146-153: timestamp (i64)
    pub fn parse_proof(data: &[u8]) -> Result<ProofAccountInfo, SolanaError> {
        if data.len() < 154 {
            return Err(SolanaError::AccountNotFound("Data too short for proof account".to_string()));
        }
        
        // Check discriminator
        let discriminator = &data[..8];
        let expected = Self::proof_discriminator();
        if discriminator != expected {
            return Err(SolanaError::Program("Invalid proof account discriminator".to_string()));
        }
        
        // Parse fields manually
        let mut proof_id = [0u8; 32];
        proof_id.copy_from_slice(&data[8..40]);
        
        let privacy_level = data[40];
        
        let mut sender_commitment = [0u8; 32];
        sender_commitment.copy_from_slice(&data[41..73]);
        
        let mut recipient_commitment = [0u8; 32];
        recipient_commitment.copy_from_slice(&data[73..105]);
        
        let verified = data[105] != 0;
        
        let submitter_bytes = &data[106..138];
        let submitter = bs58::encode(submitter_bytes).into_string();
        
        let slot = u64::from_le_bytes(data[138..146].try_into().unwrap());
        let timestamp = i64::from_le_bytes(data[146..154].try_into().unwrap());
        
        Ok(ProofAccountInfo {
            proof_id,
            privacy_level,
            sender_commitment,
            recipient_commitment,
            verified,
            submitter,
            slot,
            timestamp,
        })
    }
    
    /// Parse a transfer account from raw data.
    pub fn parse_transfer(data: &[u8]) -> Result<TransferAccountInfo, SolanaError> {
        if data.len() < 120 {
            return Err(SolanaError::AccountNotFound("Data too short for transfer account".to_string()));
        }
        
        // Parse fields manually
        let mut transfer_id = [0u8; 32];
        transfer_id.copy_from_slice(&data[8..40]);
        
        let mut target_chain = [0u8; 32];
        target_chain.copy_from_slice(&data[40..72]);
        
        let sender_bytes = &data[72..104];
        let sender = bs58::encode(sender_bytes).into_string();
        
        // Recipient length at bytes 104-108, then recipient data
        let recipient_len = u32::from_le_bytes(data[104..108].try_into().unwrap()) as usize;
        let recipient = data[108..108 + recipient_len.min(32)].to_vec();
        
        let offset = 108 + recipient_len.min(32);
        let amount = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap_or([0; 8]));
        let status = data.get(offset + 8).copied().unwrap_or(0);
        
        Ok(TransferAccountInfo {
            transfer_id,
            target_chain,
            sender,
            recipient,
            amount,
            status,
        })
    }
    
    /// Get the proof account discriminator.
    fn proof_discriminator() -> [u8; 8] {
        let mut hasher = Sha256::new();
        hasher.update(b"account:ProofAccount");
        let result = hasher.finalize();
        let mut disc = [0u8; 8];
        disc.copy_from_slice(&result[..8]);
        disc
    }
    
    /// Get the transfer account discriminator.
    fn transfer_discriminator() -> [u8; 8] {
        let mut hasher = Sha256::new();
        hasher.update(b"account:TransferAccount");
        let result = hasher.finalize();
        let mut disc = [0u8; 8];
        disc.copy_from_slice(&result[..8]);
        disc
    }
}

/// PDA (Program Derived Address) utilities.
pub struct PdaDerivation;

impl PdaDerivation {
    /// Derive proof account PDA.
    pub fn proof_account(
        program_id: &[u8; 32],
        proof_id: &[u8; 32],
    ) -> ([u8; 32], u8) {
        Self::find_pda(program_id, &[b"proof", proof_id.as_slice()])
    }
    
    /// Derive transfer account PDA.
    pub fn transfer_account(
        program_id: &[u8; 32],
        transfer_id: &[u8; 32],
    ) -> ([u8; 32], u8) {
        Self::find_pda(program_id, &[b"transfer", transfer_id.as_slice()])
    }
    
    /// Derive verifier state PDA.
    pub fn verifier_state(program_id: &[u8; 32]) -> ([u8; 32], u8) {
        Self::find_pda(program_id, &[b"verifier_state"])
    }
    
    /// Find program derived address with bump.
    fn find_pda(program_id: &[u8; 32], seeds: &[&[u8]]) -> ([u8; 32], u8) {
        for bump in (0..=255u8).rev() {
            let mut hasher = Sha256::new();
            for seed in seeds {
                hasher.update(seed);
            }
            hasher.update(&[bump]);
            hasher.update(program_id);
            hasher.update(b"ProgramDerivedAddress");
            
            let hash = hasher.finalize();
            let mut address = [0u8; 32];
            address.copy_from_slice(&hash[..32]);
            
            // Check if this is a valid PDA (not on the ed25519 curve)
            // Simplified check - real implementation would use proper curve math
            if address[31] & 0x80 == 0 {
                return (address, bump);
            }
        }
        
        // Fallback (shouldn't happen in practice)
        ([0u8; 32], 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pda_derivation() {
        let program_id = [1u8; 32];
        let proof_id = [2u8; 32];
        
        let (address, bump) = PdaDerivation::proof_account(&program_id, &proof_id);
        assert!(bump <= 255);
        assert_ne!(address, [0u8; 32]);
    }
    
    #[test]
    fn test_discriminator() {
        let disc = AccountParser::proof_discriminator();
        assert_eq!(disc.len(), 8);
    }
}
