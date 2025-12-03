// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025 NexusZero Protocol
//
// This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
// Licensed under the GNU Affero General Public License v3.0 or later.
// Commercial licensing available at https://nexuszero.io/licensing
//
// NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
// are trademarks of NexusZero Protocol. All Rights Reserved.

pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "./NexusZeroVerifier.sol";

/**
 * @title NexusZeroBridge
 * @notice Cross-chain bridge with privacy-preserving transfers
 * @dev Integrates with NexusZeroVerifier for ZK proof verification
 */
contract NexusZeroBridge is AccessControl, Pausable, ReentrancyGuard {
    using SafeERC20 for IERC20;

    // ============================================================
    // ROLES
    // ============================================================
    bytes32 public constant RELAYER_ROLE = keccak256("RELAYER_ROLE");
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");

    // ============================================================
    // STRUCTS
    // ============================================================

    /// @notice Bridge transfer request
    struct TransferRequest {
        bytes32 requestId;
        address sender;
        bytes32 destinationChain;
        bytes32 recipientCommitment;  // Hidden recipient
        address token;
        uint256 amount;
        NexusZeroVerifier.PrivacyLevel privacyLevel;
        bytes32 nullifier;
        uint256 timestamp;
        TransferStatus status;
    }

    /// @notice Transfer status
    enum TransferStatus {
        Pending,
        Proving,
        Submitted,
        Completed,
        Failed,
        Refunded
    }

    /// @notice Supported chain configuration
    struct ChainConfig {
        bytes32 chainId;
        bool isActive;
        uint256 minTransfer;
        uint256 maxTransfer;
        uint256 baseFee;
        uint256 privacyMultiplier;  // Fee multiplier for higher privacy
    }

    // ============================================================
    // STATE VARIABLES
    // ============================================================

    /// @notice Reference to the verifier contract
    NexusZeroVerifier public immutable verifier;

    /// @notice Transfer requests by ID
    mapping(bytes32 => TransferRequest) public transfers;

    /// @notice Chain configurations
    mapping(bytes32 => ChainConfig) public chains;

    /// @notice Supported tokens
    mapping(address => bool) public supportedTokens;

    /// @notice User transfer history
    mapping(address => bytes32[]) public userTransfers;

    /// @notice Nonce for request IDs
    uint256 public nonce;

    /// @notice Total value locked
    mapping(address => uint256) public totalValueLocked;

    /// @notice Treasury address for fees
    address public treasury;

    // ============================================================
    // EVENTS
    // ============================================================

    event TransferInitiated(
        bytes32 indexed requestId,
        address indexed sender,
        bytes32 indexed destinationChain,
        address token,
        uint256 amount,
        NexusZeroVerifier.PrivacyLevel privacyLevel
    );

    event TransferCompleted(
        bytes32 indexed requestId,
        bytes32 proofHash
    );

    event TransferFailed(
        bytes32 indexed requestId,
        string reason
    );

    event TransferRefunded(
        bytes32 indexed requestId,
        address indexed sender,
        uint256 amount
    );

    event ChainConfigured(
        bytes32 indexed chainId,
        bool isActive,
        uint256 minTransfer,
        uint256 maxTransfer
    );

    event TokenSupported(
        address indexed token,
        bool isSupported
    );

    // ============================================================
    // ERRORS
    // ============================================================

    error UnsupportedChain(bytes32 chainId);
    error UnsupportedToken(address token);
    error AmountBelowMinimum(uint256 amount, uint256 minimum);
    error AmountAboveMaximum(uint256 amount, uint256 maximum);
    error InsufficientFee(uint256 required, uint256 provided);
    error TransferNotFound(bytes32 requestId);
    error InvalidTransferStatus(TransferStatus current, TransferStatus required);
    error UnauthorizedRelayer(address caller);

    // ============================================================
    // CONSTRUCTOR
    // ============================================================

    constructor(address _verifier, address _treasury) {
        verifier = NexusZeroVerifier(_verifier);
        treasury = _treasury;

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
        _grantRole(RELAYER_ROLE, msg.sender);

        // Configure default chains
        _configureChain(keccak256("ethereum"), true, 0.01 ether, 1000 ether, 0.001 ether, 100);
        _configureChain(keccak256("polygon"), true, 0.001 ether, 100 ether, 0.0001 ether, 100);
        _configureChain(keccak256("arbitrum"), true, 0.001 ether, 500 ether, 0.0005 ether, 100);
    }

    // ============================================================
    // EXTERNAL FUNCTIONS
    // ============================================================

    /**
     * @notice Initiate a privacy-preserving cross-chain transfer
     * @param destinationChain Target chain identifier
     * @param recipientCommitment Commitment hiding the recipient address
     * @param token Token to transfer (address(0) for native)
     * @param amount Amount to transfer
     * @param privacyLevel Desired privacy level
     * @param nullifier Nullifier for double-spend prevention
     * @return requestId The unique transfer request ID
     */
    function initiateTransfer(
        bytes32 destinationChain,
        bytes32 recipientCommitment,
        address token,
        uint256 amount,
        NexusZeroVerifier.PrivacyLevel privacyLevel,
        bytes32 nullifier
    ) external payable nonReentrant whenNotPaused returns (bytes32 requestId) {
        // Validate chain
        ChainConfig storage chainConfig = chains[destinationChain];
        if (!chainConfig.isActive) revert UnsupportedChain(destinationChain);

        // Validate token
        if (token != address(0) && !supportedTokens[token]) {
            revert UnsupportedToken(token);
        }

        // Validate amount
        if (amount < chainConfig.minTransfer) {
            revert AmountBelowMinimum(amount, chainConfig.minTransfer);
        }
        if (amount > chainConfig.maxTransfer) {
            revert AmountAboveMaximum(amount, chainConfig.maxTransfer);
        }

        // Calculate fee based on privacy level
        uint256 fee = _calculateFee(chainConfig, privacyLevel);
        
        if (token == address(0)) {
            // Native token transfer
            if (msg.value < amount + fee) {
                revert InsufficientFee(amount + fee, msg.value);
            }
        } else {
            // ERC20 transfer
            if (msg.value < fee) {
                revert InsufficientFee(fee, msg.value);
            }
            IERC20(token).safeTransferFrom(msg.sender, address(this), amount);
        }

        // Generate request ID
        requestId = keccak256(abi.encode(
            msg.sender,
            destinationChain,
            recipientCommitment,
            amount,
            ++nonce,
            block.timestamp
        ));

        // Create transfer request
        transfers[requestId] = TransferRequest({
            requestId: requestId,
            sender: msg.sender,
            destinationChain: destinationChain,
            recipientCommitment: recipientCommitment,
            token: token,
            amount: amount,
            privacyLevel: privacyLevel,
            nullifier: nullifier,
            timestamp: block.timestamp,
            status: TransferStatus.Pending
        });

        // Track user transfers
        userTransfers[msg.sender].push(requestId);

        // Update TVL
        totalValueLocked[token] += amount;

        // Send fee to treasury
        if (fee > 0) {
            (bool success, ) = treasury.call{value: fee}("");
            require(success, "Fee transfer failed");
        }

        emit TransferInitiated(
            requestId,
            msg.sender,
            destinationChain,
            token,
            amount,
            privacyLevel
        );

        return requestId;
    }

    /**
     * @notice Complete a transfer with proof verification
     * @param requestId The transfer request ID
     * @param proof The ZK proof
     * @param publicInputs Public inputs for verification
     */
    function completeTransfer(
        bytes32 requestId,
        NexusZeroVerifier.Groth16Proof calldata proof,
        uint256[] calldata publicInputs
    ) external onlyRole(RELAYER_ROLE) nonReentrant {
        TransferRequest storage transfer = transfers[requestId];
        
        if (transfer.timestamp == 0) revert TransferNotFound(requestId);
        if (transfer.status != TransferStatus.Pending && transfer.status != TransferStatus.Proving) {
            revert InvalidTransferStatus(transfer.status, TransferStatus.Pending);
        }

        // Get circuit ID for privacy level
        bytes32 circuitId = verifier.circuitIds(transfer.privacyLevel);

        // Create commitment from amount (simplified)
        bytes32 commitment = keccak256(abi.encode(
            transfer.recipientCommitment,
            transfer.amount,
            requestId
        ));

        // Submit proof to verifier and record proofId
        bytes32 proofId = verifier.submitProof{value: verifier.verificationFeeByLevel(transfer.privacyLevel)}(
            proof.a,
            proof.b,
            proof.c,
            publicInputs,
            circuitId,
            bytes32(0), // senderCommitment: unused here
            transfer.recipientCommitment,
            transfer.privacyLevel
        );

        // Verify proof via verifier by id
        bool isValid = verifier.verifyProofById{value: verifier.verificationFeeByLevel(transfer.privacyLevel)}(
            proofId,
            transfer.nullifier,
            commitment
        );

        if (isValid) {
            transfer.status = TransferStatus.Completed;
            
            // Update TVL
            totalValueLocked[transfer.token] -= transfer.amount;

            bytes32 proofHash = proofId;
            emit TransferCompleted(requestId, proofHash);
        } else {
            transfer.status = TransferStatus.Failed;
            emit TransferFailed(requestId, "Proof verification failed");
        }
    }

    /**
     * @notice Request refund for a failed transfer
     * @param requestId The transfer request ID
     */
    function requestRefund(bytes32 requestId) external nonReentrant {
        TransferRequest storage transfer = transfers[requestId];
        
        if (transfer.timestamp == 0) revert TransferNotFound(requestId);
        require(transfer.sender == msg.sender, "Not transfer sender");
        
        // Allow refund if failed or pending for too long (24 hours)
        bool canRefund = transfer.status == TransferStatus.Failed ||
            (transfer.status == TransferStatus.Pending && 
             block.timestamp > transfer.timestamp + 24 hours);
        
        require(canRefund, "Cannot refund at this time");

        transfer.status = TransferStatus.Refunded;
        
        // Update TVL
        totalValueLocked[transfer.token] -= transfer.amount;

        // Refund
        if (transfer.token == address(0)) {
            (bool success, ) = transfer.sender.call{value: transfer.amount}("");
            require(success, "Refund failed");
        } else {
            IERC20(transfer.token).safeTransfer(transfer.sender, transfer.amount);
        }

        emit TransferRefunded(requestId, transfer.sender, transfer.amount);
    }

    /**
     * @notice Get transfer details
     * @param requestId The transfer request ID
     * @return transfer The transfer request
     */
    function getTransfer(bytes32 requestId) external view returns (TransferRequest memory) {
        return transfers[requestId];
    }

    /**
     * @notice Get user's transfer history
     * @param user The user address
     * @return requestIds Array of transfer request IDs
     */
    function getUserTransfers(address user) external view returns (bytes32[] memory) {
        return userTransfers[user];
    }

    /**
     * @notice Calculate fee for a transfer
     * @param destinationChain Target chain
     * @param privacyLevel Privacy level
     * @return fee The calculated fee
     */
    function calculateFee(
        bytes32 destinationChain,
        NexusZeroVerifier.PrivacyLevel privacyLevel
    ) external view returns (uint256 fee) {
        ChainConfig storage chainConfig = chains[destinationChain];
        return _calculateFee(chainConfig, privacyLevel);
    }

    // ============================================================
    // ADMIN FUNCTIONS
    // ============================================================

    /**
     * @notice Configure a supported chain
     */
    function configureChain(
        bytes32 chainId,
        bool isActive,
        uint256 minTransfer,
        uint256 maxTransfer,
        uint256 baseFee,
        uint256 privacyMultiplier
    ) external onlyRole(OPERATOR_ROLE) {
        _configureChain(chainId, isActive, minTransfer, maxTransfer, baseFee, privacyMultiplier);
    }

    /**
     * @notice Set token support status
     */
    function setTokenSupport(address token, bool isSupported) external onlyRole(OPERATOR_ROLE) {
        supportedTokens[token] = isSupported;
        emit TokenSupported(token, isSupported);
    }

    /**
     * @notice Update treasury address
     */
    function setTreasury(address _treasury) external onlyRole(DEFAULT_ADMIN_ROLE) {
        require(_treasury != address(0), "Invalid treasury");
        treasury = _treasury;
    }

    /**
     * @notice Pause the contract
     */
    function pause() external onlyRole(OPERATOR_ROLE) {
        _pause();
    }

    /**
     * @notice Unpause the contract
     */
    function unpause() external onlyRole(OPERATOR_ROLE) {
        _unpause();
    }

    // ============================================================
    // INTERNAL FUNCTIONS
    // ============================================================

    function _configureChain(
        bytes32 chainId,
        bool isActive,
        uint256 minTransfer,
        uint256 maxTransfer,
        uint256 baseFee,
        uint256 privacyMultiplier
    ) internal {
        chains[chainId] = ChainConfig({
            chainId: chainId,
            isActive: isActive,
            minTransfer: minTransfer,
            maxTransfer: maxTransfer,
            baseFee: baseFee,
            privacyMultiplier: privacyMultiplier
        });

        emit ChainConfigured(chainId, isActive, minTransfer, maxTransfer);
    }

    function _calculateFee(
        ChainConfig storage config,
        NexusZeroVerifier.PrivacyLevel privacyLevel
    ) internal view returns (uint256) {
        uint256 levelMultiplier = uint256(privacyLevel) + 1;
        return config.baseFee * levelMultiplier * config.privacyMultiplier / 100;
    }

    // ============================================================
    // RECEIVE
    // ============================================================

    receive() external payable {}
}
