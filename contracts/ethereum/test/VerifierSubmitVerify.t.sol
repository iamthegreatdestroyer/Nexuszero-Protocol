// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "../src/NexusZeroVerifier.sol";
import "forge-std/Test.sol";

contract VerifierSubmitVerifyTest is Test {
    NexusZeroVerifier verifier;

    function setUp() public {
        verifier = new NexusZeroVerifier();
    }

    function testSubmitAndMarkVerified() public {
        // Dummy Groth16 proof: 0s
        NexusZeroVerifier.Groth16Proof memory proof;
        proof.a = [uint256(0), uint256(0)];
        proof.b = [[uint256(0), uint256(0)], [uint256(0), uint256(0)]];
        proof.c = [uint256(0), uint256(0)];
        uint256[] memory publicInputs = new uint256[](0);

        // Submit proof and measure gas
        uint256 gstart = gasleft();
        bytes32 proofId = verifier.submitProof{value: 0.001 ether}(bytes32(0), proof, publicInputs, bytes32(0), bytes32(0), NexusZeroVerifier.PrivacyLevel.Transparent);
        uint256 gend = gasleft();
        uint256 gasUsed = gstart - gend;
        emit log_named_uint("submitProof gas used", gasUsed);
        assert(proofId != bytes32(0));

        // Mark as verified via operator helper
        verifier.operatorMarkProofVerified(proofId, true);

        // Now verify via view
        bool isValid = verifier.verifyProof(proofId);
        assert(isValid == true);
    }
}
