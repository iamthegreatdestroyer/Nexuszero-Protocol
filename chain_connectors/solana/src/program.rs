//! NexusZero Solana program types and instruction builders.

use serde::{Deserialize, Serialize};

/// Instruction discriminators for NexusZero program.
#[derive(Debug, Clone, Copy)]
#[repr(u8)]
pub enum NexusZeroInstruction {
    /// Submit a proof for verification
    SubmitProof = 0,
    /// Verify a proof
    VerifyProof = 1,
    /// Initialize the verifier state
    Initialize = 2,
    /// Bridge transfer initiation
    InitiateBridge = 3,
    /// Complete bridge transfer
    CompleteBridge = 4,
}

/// Proof submission instruction data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmitProofData {
    /// Privacy level (0-5)
    pub privacy_level: u8,
    /// Sender commitment
    pub sender_commitment: [u8; 32],
    /// Recipient commitment
    pub recipient_commitment: [u8; 32],
    /// Proof data
    pub proof: Vec<u8>,
}

/// Verify proof instruction data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyProofData {
    /// Proof ID to verify
    pub proof_id: [u8; 32],
}

/// Bridge initiation instruction data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitiateBridgeData {
    /// Target chain ID
    pub target_chain: [u8; 32],
    /// Recipient on target chain
    pub recipient: Vec<u8>,
    /// Amount to bridge (lamports)
    pub amount: u64,
    /// Proof data
    pub proof: Vec<u8>,
}

/// Complete bridge instruction data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompleteBridgeData {
    /// Transfer ID
    pub transfer_id: [u8; 32],
    /// Relayer proof
    pub relayer_proof: Vec<u8>,
}

/// Proof account data stored on-chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofAccount {
    /// Account discriminator
    pub discriminator: [u8; 8],
    /// Proof ID
    pub proof_id: [u8; 32],
    /// Privacy level
    pub privacy_level: u8,
    /// Sender commitment
    pub sender_commitment: [u8; 32],
    /// Recipient commitment
    pub recipient_commitment: [u8; 32],
    /// Whether proof is verified
    pub verified: bool,
    /// Submitter public key
    pub submitter: [u8; 32],
    /// Slot when submitted
    pub slot: u64,
    /// Unix timestamp
    pub timestamp: i64,
}

/// Bridge transfer account data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferAccount {
    /// Account discriminator
    pub discriminator: [u8; 8],
    /// Transfer ID
    pub transfer_id: [u8; 32],
    /// Source chain (Solana)
    pub source_chain: [u8; 32],
    /// Target chain
    pub target_chain: [u8; 32],
    /// Sender
    pub sender: [u8; 32],
    /// Recipient
    pub recipient: Vec<u8>,
    /// Amount
    pub amount: u64,
    /// Proof ID
    pub proof_id: [u8; 32],
    /// Status
    pub status: TransferStatus,
    /// Initiation slot
    pub init_slot: u64,
    /// Completion slot
    pub complete_slot: Option<u64>,
}

/// Transfer status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum TransferStatus {
    Initiated = 0,
    Relaying = 1,
    ReadyToComplete = 2,
    Completed = 3,
    Failed = 4,
    Expired = 5,
}

/// Instruction builder helper.
pub struct InstructionBuilder;

impl InstructionBuilder {
    /// Build submit proof instruction data.
    pub fn submit_proof(
        privacy_level: u8,
        sender_commitment: [u8; 32],
        recipient_commitment: [u8; 32],
        proof: Vec<u8>,
    ) -> Vec<u8> {
        let data = SubmitProofData {
            privacy_level,
            sender_commitment,
            recipient_commitment,
            proof,
        };
        
        let mut result = vec![NexusZeroInstruction::SubmitProof as u8];
        result.extend_from_slice(&bincode::serialize(&data).unwrap_or_default());
        result
    }
    
    /// Build verify proof instruction data.
    pub fn verify_proof(proof_id: [u8; 32]) -> Vec<u8> {
        let data = VerifyProofData { proof_id };
        
        let mut result = vec![NexusZeroInstruction::VerifyProof as u8];
        result.extend_from_slice(&bincode::serialize(&data).unwrap_or_default());
        result
    }
    
    /// Build initiate bridge instruction data.
    pub fn initiate_bridge(
        target_chain: [u8; 32],
        recipient: Vec<u8>,
        amount: u64,
        proof: Vec<u8>,
    ) -> Vec<u8> {
        let data = InitiateBridgeData {
            target_chain,
            recipient,
            amount,
            proof,
        };
        
        let mut result = vec![NexusZeroInstruction::InitiateBridge as u8];
        result.extend_from_slice(&bincode::serialize(&data).unwrap_or_default());
        result
    }
    
    /// Build complete bridge instruction data.
    pub fn complete_bridge(transfer_id: [u8; 32], relayer_proof: Vec<u8>) -> Vec<u8> {
        let data = CompleteBridgeData {
            transfer_id,
            relayer_proof,
        };
        
        let mut result = vec![NexusZeroInstruction::CompleteBridge as u8];
        result.extend_from_slice(&bincode::serialize(&data).unwrap_or_default());
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_submit_proof_instruction() {
        let data = InstructionBuilder::submit_proof(
            3,
            [1u8; 32],
            [2u8; 32],
            vec![0, 1, 2, 3],
        );
        
        assert_eq!(data[0], NexusZeroInstruction::SubmitProof as u8);
        assert!(data.len() > 1);
    }
    
    #[test]
    fn test_verify_proof_instruction() {
        let data = InstructionBuilder::verify_proof([0u8; 32]);
        assert_eq!(data[0], NexusZeroInstruction::VerifyProof as u8);
    }
    
    #[test]
    fn test_transfer_status() {
        assert_eq!(TransferStatus::Initiated as u8, 0);
        assert_eq!(TransferStatus::Completed as u8, 3);
    }
}
