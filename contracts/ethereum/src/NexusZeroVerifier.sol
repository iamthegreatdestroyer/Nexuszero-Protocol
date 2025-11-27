// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title NexusZeroVerifier
 * @notice Core contract for verifying zero-knowledge proofs in NexusZero Protocol
 * @dev Implements Groth16 proof verification with multi-level privacy support
 */
contract NexusZeroVerifier is AccessControl, Pausable, ReentrancyGuard {
    // ============================================================
    // ROLES
    // ============================================================
    bytes32 public constant OPERATOR_ROLE = keccak256("OPERATOR_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");

    // ============================================================
    // STRUCTS
    // ============================================================
    
    /// @notice Groth16 proof structure
    struct Groth16Proof {
        uint256[2] a;      // G1 point
        uint256[2][2] b;   // G2 point  
        uint256[2] c;      // G1 point
    }

    /// @notice Verification key for a specific circuit
    struct VerificationKey {
        uint256[2] alpha;
        uint256[2][2] beta;
        uint256[2][2] gamma;
        uint256[2][2] delta;
        uint256[2][] ic;
        bool isActive;
    }

    /// @notice Privacy level enumeration (matches Rust implementation)
    enum PrivacyLevel {
        Transparent,      // Level 0 - No privacy
        Pseudonymous,     // Level 1 - Address pseudonymity
        Confidential,     // Level 2 - Amount hidden
        Private,          // Level 3 - Full transaction privacy
        Anonymous,        // Level 4 - Sender/receiver hidden
        Sovereign         // Level 5 - Maximum privacy
    }

    /// @notice Proof metadata for tracking
    struct ProofRecord {
        bytes32 proofHash;
        PrivacyLevel level;
        address submitter;
        uint256 timestamp;
        bool isValid;
    }

    // ============================================================
    // STATE VARIABLES
    // ============================================================

    /// @notice Verification keys indexed by circuit ID
    mapping(bytes32 => VerificationKey) public verificationKeys;
    
    /// @notice Recorded proofs indexed by proof hash
    mapping(bytes32 => ProofRecord) public proofRecords;
    
    /// @notice Nullifier set to prevent double-spending
    mapping(bytes32 => bool) public nullifiers;
    
    /// @notice Commitment set for transaction outputs
    mapping(bytes32 => bool) public commitments;
    
    /// @notice Total proofs verified per privacy level
    mapping(PrivacyLevel => uint256) public proofCountByLevel;
    
    /// @notice Circuit IDs for each privacy level
    mapping(PrivacyLevel => bytes32) public circuitIds;

    /// @notice Total number of proofs verified
    uint256 public totalProofsVerified;
    
    /// @notice Proof verification fee in wei
    uint256 public verificationFee;

    // ============================================================
    // EVENTS
    // ============================================================

    event ProofVerified(
        bytes32 indexed proofHash,
        PrivacyLevel level,
        address indexed submitter,
        uint256 timestamp
    );

    event VerificationKeyRegistered(
        bytes32 indexed circuitId,
        PrivacyLevel level,
        address indexed registrar
    );

    event NullifierUsed(
        bytes32 indexed nullifier,
        bytes32 indexed proofHash
    );

    event CommitmentAdded(
        bytes32 indexed commitment,
        bytes32 indexed proofHash
    );

    event VerificationFeeUpdated(
        uint256 oldFee,
        uint256 newFee
    );

    // ============================================================
    // ERRORS
    // ============================================================

    error InvalidProof();
    error NullifierAlreadyUsed(bytes32 nullifier);
    error CommitmentAlreadyExists(bytes32 commitment);
    error CircuitNotRegistered(bytes32 circuitId);
    error InsufficientFee(uint256 required, uint256 provided);
    error InvalidVerificationKey();
    error ProofAlreadySubmitted(bytes32 proofHash);

    // ============================================================
    // CONSTRUCTOR
    // ============================================================

    constructor() {
        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(OPERATOR_ROLE, msg.sender);
        _grantRole(VERIFIER_ROLE, msg.sender);
        
        verificationFee = 0.001 ether;
    }

    // ============================================================
    // EXTERNAL FUNCTIONS
    // ============================================================

    /**
     * @notice Register a verification key for a circuit
     * @param circuitId Unique identifier for the circuit
     * @param level Privacy level this circuit supports
     * @param vk The verification key components
     */
    function registerVerificationKey(
        bytes32 circuitId,
        PrivacyLevel level,
        VerificationKey calldata vk
    ) external onlyRole(OPERATOR_ROLE) {
        if (vk.ic.length == 0) revert InvalidVerificationKey();
        
        verificationKeys[circuitId] = vk;
        verificationKeys[circuitId].isActive = true;
        circuitIds[level] = circuitId;
        
        emit VerificationKeyRegistered(circuitId, level, msg.sender);
    }

    /**
     * @notice Verify a zero-knowledge proof
     * @param circuitId The circuit ID for this proof type
     * @param proof The Groth16 proof
     * @param publicInputs Public inputs to the circuit
     * @param nullifier Nullifier to prevent double-spending
     * @param commitment Output commitment
     * @return success Whether the proof was valid
     */
    function verifyProof(
        bytes32 circuitId,
        Groth16Proof calldata proof,
        uint256[] calldata publicInputs,
        bytes32 nullifier,
        bytes32 commitment
    ) external payable nonReentrant whenNotPaused returns (bool success) {
        // Check fee
        if (msg.value < verificationFee) {
            revert InsufficientFee(verificationFee, msg.value);
        }

        // Check circuit exists
        VerificationKey storage vk = verificationKeys[circuitId];
        if (!vk.isActive) revert CircuitNotRegistered(circuitId);

        // Check nullifier not used
        if (nullifiers[nullifier]) {
            revert NullifierAlreadyUsed(nullifier);
        }

        // Check commitment doesn't exist
        if (commitments[commitment]) {
            revert CommitmentAlreadyExists(commitment);
        }

        // Compute proof hash
        bytes32 proofHash = keccak256(abi.encode(
            proof.a,
            proof.b,
            proof.c,
            publicInputs
        ));

        // Check proof not already submitted
        if (proofRecords[proofHash].timestamp != 0) {
            revert ProofAlreadySubmitted(proofHash);
        }

        // Verify the proof (Groth16 verification)
        success = _verifyGroth16(vk, proof, publicInputs);
        
        if (!success) revert InvalidProof();

        // Record nullifier and commitment
        nullifiers[nullifier] = true;
        commitments[commitment] = true;

        // Determine privacy level from circuit
        PrivacyLevel level = _getLevelFromCircuit(circuitId);

        // Record proof
        proofRecords[proofHash] = ProofRecord({
            proofHash: proofHash,
            level: level,
            submitter: msg.sender,
            timestamp: block.timestamp,
            isValid: true
        });

        // Update counters
        totalProofsVerified++;
        proofCountByLevel[level]++;

        emit ProofVerified(proofHash, level, msg.sender, block.timestamp);
        emit NullifierUsed(nullifier, proofHash);
        emit CommitmentAdded(commitment, proofHash);

        return true;
    }

    /**
     * @notice Batch verify multiple proofs
     * @param circuitIds Array of circuit IDs
     * @param proofs Array of proofs
     * @param publicInputsArray Array of public inputs arrays
     * @param nullifiersList Array of nullifiers
     * @param commitmentsList Array of commitments
     * @return results Array of verification results
     */
    function batchVerifyProofs(
        bytes32[] calldata circuitIds,
        Groth16Proof[] calldata proofs,
        uint256[][] calldata publicInputsArray,
        bytes32[] calldata nullifiersList,
        bytes32[] calldata commitmentsList
    ) external payable nonReentrant whenNotPaused returns (bool[] memory results) {
        uint256 count = proofs.length;
        require(
            circuitIds.length == count &&
            publicInputsArray.length == count &&
            nullifiersList.length == count &&
            commitmentsList.length == count,
            "Array length mismatch"
        );

        uint256 totalFee = verificationFee * count;
        if (msg.value < totalFee) {
            revert InsufficientFee(totalFee, msg.value);
        }

        results = new bool[](count);
        
        for (uint256 i = 0; i < count; i++) {
            // Skip if nullifier already used
            if (nullifiers[nullifiersList[i]]) {
                results[i] = false;
                continue;
            }

            VerificationKey storage vk = verificationKeys[circuitIds[i]];
            if (!vk.isActive) {
                results[i] = false;
                continue;
            }

            results[i] = _verifyGroth16(vk, proofs[i], publicInputsArray[i]);
            
            if (results[i]) {
                nullifiers[nullifiersList[i]] = true;
                commitments[commitmentsList[i]] = true;
                totalProofsVerified++;
            }
        }

        return results;
    }

    /**
     * @notice Check if a nullifier has been used
     * @param nullifier The nullifier to check
     * @return used Whether the nullifier has been used
     */
    function isNullifierUsed(bytes32 nullifier) external view returns (bool used) {
        return nullifiers[nullifier];
    }

    /**
     * @notice Check if a commitment exists
     * @param commitment The commitment to check
     * @return exists Whether the commitment exists
     */
    function commitmentExists(bytes32 commitment) external view returns (bool exists) {
        return commitments[commitment];
    }

    /**
     * @notice Get proof record by hash
     * @param proofHash The proof hash
     * @return record The proof record
     */
    function getProofRecord(bytes32 proofHash) external view returns (ProofRecord memory record) {
        return proofRecords[proofHash];
    }

    /**
     * @notice Update verification fee
     * @param newFee New fee in wei
     */
    function setVerificationFee(uint256 newFee) external onlyRole(OPERATOR_ROLE) {
        uint256 oldFee = verificationFee;
        verificationFee = newFee;
        emit VerificationFeeUpdated(oldFee, newFee);
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

    /**
     * @notice Withdraw accumulated fees
     * @param to Address to send fees to
     */
    function withdrawFees(address payable to) external onlyRole(DEFAULT_ADMIN_ROLE) {
        uint256 balance = address(this).balance;
        require(balance > 0, "No fees to withdraw");
        (bool success, ) = to.call{value: balance}("");
        require(success, "Transfer failed");
    }

    // ============================================================
    // INTERNAL FUNCTIONS
    // ============================================================

    /**
     * @dev Verify a Groth16 proof using the pairing check
     * @param vk Verification key
     * @param proof The proof
     * @param publicInputs Public inputs
     * @return valid Whether the proof is valid
     */
    function _verifyGroth16(
        VerificationKey storage vk,
        Groth16Proof calldata proof,
        uint256[] calldata publicInputs
    ) internal view returns (bool valid) {
        require(publicInputs.length + 1 == vk.ic.length, "Invalid input length");

        // Compute the linear combination vk_x
        uint256[2] memory vk_x = vk.ic[0];
        
        for (uint256 i = 0; i < publicInputs.length; i++) {
            // vk_x = vk_x + publicInputs[i] * vk.ic[i+1]
            (uint256 x, uint256 y) = _scalarMul(vk.ic[i + 1], publicInputs[i]);
            (vk_x[0], vk_x[1]) = _pointAdd(vk_x[0], vk_x[1], x, y);
        }

        // Pairing check:
        // e(A, B) = e(alpha, beta) * e(vk_x, gamma) * e(C, delta)
        return _pairingCheck(
            proof.a,
            proof.b,
            vk.alpha,
            vk.beta,
            vk_x,
            vk.gamma,
            proof.c,
            vk.delta
        );
    }

    /**
     * @dev Get privacy level from circuit ID
     */
    function _getLevelFromCircuit(bytes32 circuitId) internal view returns (PrivacyLevel) {
        for (uint8 i = 0; i <= uint8(PrivacyLevel.Sovereign); i++) {
            if (circuitIds[PrivacyLevel(i)] == circuitId) {
                return PrivacyLevel(i);
            }
        }
        return PrivacyLevel.Transparent;
    }

    // ============================================================
    // ELLIPTIC CURVE OPERATIONS (BN254/alt_bn128)
    // ============================================================

    uint256 constant FIELD_MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    uint256 constant CURVE_ORDER = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    /**
     * @dev Scalar multiplication on G1
     */
    function _scalarMul(uint256[2] memory p, uint256 s) internal view returns (uint256 x, uint256 y) {
        uint256[3] memory input;
        input[0] = p[0];
        input[1] = p[1];
        input[2] = s;
        
        uint256[2] memory result;
        bool success;
        
        assembly {
            success := staticcall(sub(gas(), 2000), 7, input, 0x60, result, 0x40)
        }
        require(success, "Scalar multiplication failed");
        
        return (result[0], result[1]);
    }

    /**
     * @dev Point addition on G1
     */
    function _pointAdd(
        uint256 x1, uint256 y1,
        uint256 x2, uint256 y2
    ) internal view returns (uint256 x, uint256 y) {
        uint256[4] memory input;
        input[0] = x1;
        input[1] = y1;
        input[2] = x2;
        input[3] = y2;
        
        uint256[2] memory result;
        bool success;
        
        assembly {
            success := staticcall(sub(gas(), 2000), 6, input, 0x80, result, 0x40)
        }
        require(success, "Point addition failed");
        
        return (result[0], result[1]);
    }

    /**
     * @dev Pairing check for Groth16 verification
     */
    function _pairingCheck(
        uint256[2] memory a,
        uint256[2][2] memory b,
        uint256[2] memory alpha,
        uint256[2][2] memory beta,
        uint256[2] memory vk_x,
        uint256[2][2] memory gamma,
        uint256[2] memory c,
        uint256[2][2] memory delta
    ) internal view returns (bool) {
        // Negate a for the pairing equation
        uint256[2] memory negA;
        negA[0] = a[0];
        negA[1] = FIELD_MODULUS - (a[1] % FIELD_MODULUS);

        // Build pairing input: 4 pairs of (G1, G2) points
        uint256[24] memory input;
        
        // -A, B
        input[0] = negA[0];
        input[1] = negA[1];
        input[2] = b[0][1];
        input[3] = b[0][0];
        input[4] = b[1][1];
        input[5] = b[1][0];
        
        // alpha, beta
        input[6] = alpha[0];
        input[7] = alpha[1];
        input[8] = beta[0][1];
        input[9] = beta[0][0];
        input[10] = beta[1][1];
        input[11] = beta[1][0];
        
        // vk_x, gamma
        input[12] = vk_x[0];
        input[13] = vk_x[1];
        input[14] = gamma[0][1];
        input[15] = gamma[0][0];
        input[16] = gamma[1][1];
        input[17] = gamma[1][0];
        
        // C, delta
        input[18] = c[0];
        input[19] = c[1];
        input[20] = delta[0][1];
        input[21] = delta[0][0];
        input[22] = delta[1][1];
        input[23] = delta[1][0];

        uint256[1] memory result;
        bool success;
        
        assembly {
            success := staticcall(sub(gas(), 2000), 8, input, 768, result, 0x20)
        }
        
        require(success, "Pairing check failed");
        return result[0] == 1;
    }

    // ============================================================
    // RECEIVE
    // ============================================================

    receive() external payable {}
}
