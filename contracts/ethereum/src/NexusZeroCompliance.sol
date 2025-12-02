// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "./NexusZeroVerifier.sol";

/**
 * @title NexusZeroCompliance
 * @notice Privacy-preserving compliance verification using zero-knowledge proofs
 * @dev Enables selective disclosure for regulatory requirements without revealing sensitive data
 */
contract NexusZeroCompliance is AccessControl, Pausable {
    // ============================================================
    // ROLES
    // ============================================================
    bytes32 public constant COMPLIANCE_OFFICER_ROLE = keccak256("COMPLIANCE_OFFICER_ROLE");
    bytes32 public constant VERIFIER_ROLE = keccak256("VERIFIER_ROLE");

    // ============================================================
    // STRUCTS
    // ============================================================

    /// @notice Compliance proof types
    enum ComplianceType {
        AgeVerification,           // Prove age > threshold
        AccreditedInvestor,        // Prove accredited status
        NotSanctioned,             // Prove not on sanctions list
        JurisdictionAllowed,       // Prove jurisdiction is allowed
        TransactionLimit,          // Prove within transaction limits
        SourceOfFunds,             // Prove legitimate source
        KYCCompleted,              // Prove KYC without revealing data
        AMLCleared                 // Prove AML clearance
    }

    /// @notice Compliance attestation record
    struct ComplianceAttestation {
        bytes32 attestationId;
        address subject;              // Who the attestation is about
        ComplianceType complianceType;
        bytes32 proofHash;            // Hash of the ZK proof
        bytes32 commitment;           // Pedersen commitment to the claim
        uint256 issuedAt;
        uint256 expiresAt;
        bool isValid;
        bytes32 issuerCommitment;     // Hidden issuer identity
    }

    /// @notice Jurisdiction profile
    struct JurisdictionProfile {
        bytes32 jurisdictionId;
        bool isActive;
        bool requiresKYC;
        bool requiresAccreditation;
        uint256 transactionLimit;
        uint256[] blockedJurisdictions;
    }

    // ============================================================
    // STATE VARIABLES
    // ============================================================

    /// @notice Reference to the verifier contract
    NexusZeroVerifier public immutable verifier;

    /// @notice Attestations by ID
    mapping(bytes32 => ComplianceAttestation) public attestations;

    /// @notice User attestations
    mapping(address => bytes32[]) public userAttestations;

    /// @notice Active attestations by type for a user
    mapping(address => mapping(ComplianceType => bytes32)) public activeAttestation;

    /// @notice Jurisdiction profiles
    mapping(bytes32 => JurisdictionProfile) public jurisdictions;

    /// @notice Trusted attestation issuers (commitment => trusted)
    mapping(bytes32 => bool) public trustedIssuers;

    /// @notice Nonce for attestation IDs
    uint256 public nonce;

    /// @notice Circuit IDs for each compliance type
    mapping(ComplianceType => bytes32) public complianceCircuits;

    // ============================================================
    // EVENTS
    // ============================================================

    event AttestationCreated(
        bytes32 indexed attestationId,
        address indexed subject,
        ComplianceType complianceType,
        uint256 expiresAt
    );

    event AttestationRevoked(
        bytes32 indexed attestationId,
        address indexed revoker,
        string reason
    );

    event AttestationVerified(
        bytes32 indexed attestationId,
        address indexed verifier,
        bool isValid
    );

    event JurisdictionConfigured(
        bytes32 indexed jurisdictionId,
        bool isActive
    );

    event TrustedIssuerUpdated(
        bytes32 indexed issuerCommitment,
        bool isTrusted
    );

    // ============================================================
    // ERRORS
    // ============================================================

    error AttestationNotFound(bytes32 attestationId);
    error AttestationExpired(bytes32 attestationId);
    error AttestationInvalid(bytes32 attestationId);
    error UntrustedIssuer(bytes32 issuerCommitment);
    error InvalidProof();
    error JurisdictionNotAllowed(bytes32 jurisdictionId);

    // ============================================================
    // CONSTRUCTOR
    // ============================================================

    constructor(address _verifier) {
        verifier = NexusZeroVerifier(_verifier);

        _grantRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _grantRole(COMPLIANCE_OFFICER_ROLE, msg.sender);
        _grantRole(VERIFIER_ROLE, msg.sender);

        // Configure default jurisdictions
        _configureJurisdiction(
            keccak256("US"),
            true,
            true,   // requires KYC
            true,   // requires accreditation for some
            1000000 ether,
            new uint256[](0)
        );

        _configureJurisdiction(
            keccak256("EU"),
            true,
            true,
            false,
            500000 ether,
            new uint256[](0)
        );
    }

    // ============================================================
    // EXTERNAL FUNCTIONS
    // ============================================================

    /**
     * @notice Create a compliance attestation with ZK proof
     * @param subject Address the attestation is about
     * @param complianceType Type of compliance being attested
     * @param proof ZK proof of compliance
     * @param publicInputs Public inputs to the circuit
     * @param commitment Pedersen commitment to the claim
     * @param issuerCommitment Commitment hiding issuer identity
     * @param validityPeriod How long the attestation is valid (seconds)
     * @return attestationId The created attestation ID
     */
    function createAttestation(
        address subject,
        ComplianceType complianceType,
        NexusZeroVerifier.Groth16Proof calldata proof,
        uint256[] calldata publicInputs,
        bytes32 commitment,
        bytes32 issuerCommitment,
        uint256 validityPeriod
    ) external onlyRole(COMPLIANCE_OFFICER_ROLE) whenNotPaused returns (bytes32 attestationId) {
        // Verify issuer is trusted
        if (!trustedIssuers[issuerCommitment]) {
            revert UntrustedIssuer(issuerCommitment);
        }

        // Get circuit for this compliance type
        bytes32 circuitId = complianceCircuits[complianceType];

        // Create nullifier from subject and type
        bytes32 nullifier = keccak256(abi.encode(subject, complianceType, block.timestamp));

        // Submit the proof
        bytes32 proofId = verifier.submitProof{value: verifier.verificationFeeByLevel(NexusZeroVerifier.PrivacyLevel.Pseudonymous)}(
            proof.a,
            proof.b,
            proof.c,
            publicInputs,
            circuitId,
            bytes32(0), // sender commitment not needed for compliance
            commitment,
            NexusZeroVerifier.PrivacyLevel.Pseudonymous
        );

        // Verify the proof on-chain via stored proof id
        bool isValid = verifier.verifyProofById{value: verifier.verificationFeeByLevel(NexusZeroVerifier.PrivacyLevel.Pseudonymous)}(
            proofId,
            nullifier,
            commitment
        );

        if (!isValid) revert InvalidProof();

        // Generate attestation ID
        attestationId = keccak256(abi.encode(
            subject,
            complianceType,
            commitment,
            ++nonce,
            block.timestamp
        ));

        // Compute proof hash
        bytes32 proofHash = keccak256(abi.encode(proof.a, proof.b, proof.c));

        // Create attestation
        attestations[attestationId] = ComplianceAttestation({
            attestationId: attestationId,
            subject: subject,
            complianceType: complianceType,
            proofHash: proofHash,
            commitment: commitment,
            issuedAt: block.timestamp,
            expiresAt: block.timestamp + validityPeriod,
            isValid: true,
            issuerCommitment: issuerCommitment
        });

        // Track user attestations
        userAttestations[subject].push(attestationId);
        
        // Update active attestation
        activeAttestation[subject][complianceType] = attestationId;

        emit AttestationCreated(attestationId, subject, complianceType, block.timestamp + validityPeriod);

        return attestationId;
    }

    /**
     * @notice Verify a compliance attestation is valid
     * @param attestationId The attestation to verify
     * @return isValid Whether the attestation is currently valid
     */
    function verifyAttestation(bytes32 attestationId) external view returns (bool isValid) {
        ComplianceAttestation storage attestation = attestations[attestationId];
        
        if (attestation.issuedAt == 0) return false;
        if (!attestation.isValid) return false;
        if (block.timestamp > attestation.expiresAt) return false;
        
        return true;
    }

    /**
     * @notice Check if a user has a valid attestation of a specific type
     * @param user The user to check
     * @param complianceType The type of compliance to check
     * @return hasValid Whether the user has a valid attestation
     * @return attestationId The attestation ID if valid
     */
    function hasValidAttestation(
        address user,
        ComplianceType complianceType
    ) external view returns (bool hasValid, bytes32 attestationId) {
        attestationId = activeAttestation[user][complianceType];
        
        if (attestationId == bytes32(0)) return (false, bytes32(0));
        
        ComplianceAttestation storage attestation = attestations[attestationId];
        
        if (!attestation.isValid) return (false, attestationId);
        if (block.timestamp > attestation.expiresAt) return (false, attestationId);
        
        return (true, attestationId);
    }

    /**
     * @notice Check multiple compliance requirements at once
     * @param user The user to check
     * @param requiredTypes Array of required compliance types
     * @return allValid Whether all requirements are met
     * @return results Individual results for each type
     */
    function checkCompliance(
        address user,
        ComplianceType[] calldata requiredTypes
    ) external view returns (bool allValid, bool[] memory results) {
        results = new bool[](requiredTypes.length);
        allValid = true;

        for (uint256 i = 0; i < requiredTypes.length; i++) {
            bytes32 attestationId = activeAttestation[user][requiredTypes[i]];
            
            if (attestationId == bytes32(0)) {
                results[i] = false;
                allValid = false;
                continue;
            }

            ComplianceAttestation storage attestation = attestations[attestationId];
            results[i] = attestation.isValid && block.timestamp <= attestation.expiresAt;
            
            if (!results[i]) allValid = false;
        }

        return (allValid, results);
    }

    /**
     * @notice Revoke an attestation
     * @param attestationId The attestation to revoke
     * @param reason Reason for revocation
     */
    function revokeAttestation(
        bytes32 attestationId,
        string calldata reason
    ) external onlyRole(COMPLIANCE_OFFICER_ROLE) {
        ComplianceAttestation storage attestation = attestations[attestationId];
        
        if (attestation.issuedAt == 0) revert AttestationNotFound(attestationId);
        
        attestation.isValid = false;

        // Clear active attestation if this was it
        if (activeAttestation[attestation.subject][attestation.complianceType] == attestationId) {
            activeAttestation[attestation.subject][attestation.complianceType] = bytes32(0);
        }

        emit AttestationRevoked(attestationId, msg.sender, reason);
    }

    /**
     * @notice Get attestation details
     * @param attestationId The attestation ID
     * @return attestation The attestation record
     */
    function getAttestation(bytes32 attestationId) external view returns (ComplianceAttestation memory) {
        return attestations[attestationId];
    }

    /**
     * @notice Get all attestations for a user
     * @param user The user address
     * @return attestationIds Array of attestation IDs
     */
    function getUserAttestations(address user) external view returns (bytes32[] memory) {
        return userAttestations[user];
    }

    // ============================================================
    // ADMIN FUNCTIONS
    // ============================================================

    /**
     * @notice Configure a jurisdiction
     */
    function configureJurisdiction(
        bytes32 jurisdictionId,
        bool isActive,
        bool requiresKYC,
        bool requiresAccreditation,
        uint256 transactionLimit,
        uint256[] calldata blockedJurisdictions
    ) external onlyRole(COMPLIANCE_OFFICER_ROLE) {
        _configureJurisdiction(
            jurisdictionId,
            isActive,
            requiresKYC,
            requiresAccreditation,
            transactionLimit,
            blockedJurisdictions
        );
    }

    /**
     * @notice Set trusted issuer status
     */
    function setTrustedIssuer(bytes32 issuerCommitment, bool isTrusted) external onlyRole(DEFAULT_ADMIN_ROLE) {
        trustedIssuers[issuerCommitment] = isTrusted;
        emit TrustedIssuerUpdated(issuerCommitment, isTrusted);
    }

    /**
     * @notice Set compliance circuit ID
     */
    function setComplianceCircuit(
        ComplianceType complianceType,
        bytes32 circuitId
    ) external onlyRole(DEFAULT_ADMIN_ROLE) {
        complianceCircuits[complianceType] = circuitId;
    }

    /**
     * @notice Pause the contract
     */
    function pause() external onlyRole(COMPLIANCE_OFFICER_ROLE) {
        _pause();
    }

    /**
     * @notice Unpause the contract
     */
    function unpause() external onlyRole(COMPLIANCE_OFFICER_ROLE) {
        _unpause();
    }

    // ============================================================
    // INTERNAL FUNCTIONS
    // ============================================================

    function _configureJurisdiction(
        bytes32 jurisdictionId,
        bool isActive,
        bool requiresKYC,
        bool requiresAccreditation,
        uint256 transactionLimit,
        uint256[] memory blockedJurisdictions
    ) internal {
        jurisdictions[jurisdictionId] = JurisdictionProfile({
            jurisdictionId: jurisdictionId,
            isActive: isActive,
            requiresKYC: requiresKYC,
            requiresAccreditation: requiresAccreditation,
            transactionLimit: transactionLimit,
            blockedJurisdictions: blockedJurisdictions
        });

        emit JurisdictionConfigured(jurisdictionId, isActive);
    }
}
