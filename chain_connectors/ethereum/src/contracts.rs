//! NexusZero smart contract definitions and ABI

use alloy_sol_types::sol;

// Define the NexusZero Verifier contract interface
sol! {
    /// NexusZero Verifier Contract
    #[sol(rpc)]
    contract NexusZeroVerifier {
        /// Submit a privacy proof for on-chain verification
        function submitProof(
            bytes calldata proof,
            bytes32 senderCommitment,
            bytes32 recipientCommitment,
            uint8 privacyLevel
        ) external returns (bytes32 proofId);

        /// Verify a previously submitted proof
        function verifyProof(bytes32 proofId) external view returns (bool);

        /// Get proof details
        function getProofDetails(bytes32 proofId) external view returns (
            bytes32 hash,
            uint8 level,
            uint64 timestamp,
            bool verified,
            address submitter
        );

        /// Get total number of proofs submitted
        function totalProofs() external view returns (uint256);

        /// Get proof count by privacy level
        function proofCountByLevel(uint8 level) external view returns (uint256);

        /// Event emitted when a proof is submitted
        event ProofSubmitted(
            bytes32 indexed proofId,
            address indexed submitter,
            uint8 privacyLevel
        );

        /// Event emitted when a proof is verified
        event ProofVerified(
            bytes32 indexed proofId,
            bool success
        );
    }
}

// Define the NexusZero Bridge contract interface
sol! {
    /// NexusZero Bridge Contract
    #[sol(rpc)]
    contract NexusZeroBridge {
        /// Initiate a cross-chain transfer
        function initiateTransfer(
            bytes32 targetChain,
            bytes calldata proof,
            bytes calldata recipient,
            uint256 amount
        ) external payable returns (bytes32 transferId);

        /// Complete an incoming transfer
        function completeTransfer(
            bytes32 transferId,
            bytes calldata relayerProof
        ) external;

        /// Get transfer status
        /// Returns: 0=pending, 1=relaying, 2=ready, 3=completed, 4=failed, 5=expired
        function getTransferStatus(bytes32 transferId) external view returns (uint8);

        /// Get transfer details
        function getTransferDetails(bytes32 transferId) external view returns (
            bytes32 sourceChain,
            bytes32 targetChain,
            address sender,
            bytes memory recipient,
            uint256 amount,
            uint8 status,
            uint64 timestamp
        );

        /// Refund an expired or failed transfer
        function refundTransfer(bytes32 transferId) external;

        /// Event emitted when a transfer is initiated
        event TransferInitiated(
            bytes32 indexed transferId,
            bytes32 indexed targetChain,
            address indexed sender,
            uint256 amount
        );

        /// Event emitted when a transfer is completed
        event TransferCompleted(
            bytes32 indexed transferId,
            address relayer
        );

        /// Event emitted when a transfer fails
        event TransferFailed(
            bytes32 indexed transferId,
            string reason
        );
    }
}

// Define HTLC contract interface
sol! {
    /// Hash Time-Locked Contract
    #[sol(rpc)]
    contract NexusZeroHTLC {
        /// Create a new HTLC
        function createHTLC(
            address recipient,
            bytes32 hashLock,
            uint256 timeoutBlocks
        ) external payable returns (bytes32 htlcId);

        /// Redeem an HTLC with the preimage
        function redeem(
            bytes32 htlcId,
            bytes32 preimage
        ) external;

        /// Refund an expired HTLC
        function refund(bytes32 htlcId) external;

        /// Get HTLC details
        function getHTLC(bytes32 htlcId) external view returns (
            address sender,
            address recipient,
            uint256 amount,
            bytes32 hashLock,
            uint256 timeoutBlock,
            uint8 status  // 0=active, 1=redeemed, 2=refunded
        );

        /// Event emitted when HTLC is created
        event HTLCCreated(
            bytes32 indexed htlcId,
            address indexed sender,
            address indexed recipient,
            uint256 amount,
            bytes32 hashLock,
            uint256 timeoutBlock
        );

        /// Event emitted when HTLC is redeemed
        event HTLCRedeemed(
            bytes32 indexed htlcId,
            bytes32 preimage
        );

        /// Event emitted when HTLC is refunded
        event HTLCRefunded(
            bytes32 indexed htlcId
        );
    }
}

/// Contract addresses for different networks
pub struct ContractAddresses {
    /// Verifier contract address
    pub verifier: alloy_primitives::Address,
    /// Bridge contract address (optional)
    pub bridge: Option<alloy_primitives::Address>,
    /// HTLC contract address (optional)
    pub htlc: Option<alloy_primitives::Address>,
}

impl ContractAddresses {
    /// Create from hex strings
    pub fn from_strings(
        verifier: &str,
        bridge: Option<&str>,
        htlc: Option<&str>,
    ) -> Result<Self, String> {
        let parse_addr = |s: &str| -> Result<alloy_primitives::Address, String> {
            s.parse()
                .map_err(|e| format!("Invalid address '{}': {}", s, e))
        };

        Ok(Self {
            verifier: parse_addr(verifier)?,
            bridge: bridge.map(parse_addr).transpose()?,
            htlc: htlc.map(parse_addr).transpose()?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contract_addresses_parsing() {
        let result = ContractAddresses::from_strings(
            "0x1234567890123456789012345678901234567890",
            Some("0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"),
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_address() {
        let result = ContractAddresses::from_strings("invalid", None, None);
        assert!(result.is_err());
    }
}
