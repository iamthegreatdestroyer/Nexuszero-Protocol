// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "forge-std/Test.sol";
import "../src/NexusZeroVerifier.sol";
import "../src/NexusZeroBridge.sol";
import "@openzeppelin/contracts/token/ERC20/ERC20.sol";

contract DummyToken is ERC20 {
    constructor() ERC20("Dummy","DUM") { _mint(msg.sender, 1e24); }
}

contract BridgeSubmitCompleteTest is Test {
    NexusZeroVerifier verifier;
    NexusZeroBridge bridge;
    DummyToken token;

    function setUp() public {
        verifier = new NexusZeroVerifier();
        bridge = new NexusZeroBridge(address(verifier), address(this));
        token = new DummyToken();
        bridge.grantRole(bridge.OPERATOR_ROLE(), address(this));
        bridge.grantRole(bridge.RELAYER_ROLE(), address(this));
    }

    function testInitiateAndComplete() public {
        // Approve token and initiate
        token.approve(address(bridge), 1000);
        bytes32 requestId = bridge.initiateTransfer{value: 0}(keccak256("polygon"), bytes32(0), address(token), 1000, NexusZeroVerifier.PrivacyLevel.Transparent, bytes32(0));
        assert(requestId != bytes32(0));

        // Submit proof via verifier manually
        NexusZeroVerifier.Groth16Proof memory proof;
        proof.a = [uint256(0), uint256(0)];
        proof.b = [[uint256(0), uint256(0)], [uint256(0), uint256(0)]];
        proof.c = [uint256(0), uint256(0)];
        uint256[] memory pub = new uint256[](0);

        bytes32 proofId = verifier.submitProofRaw{value: 0.001 ether}(abi.encodePacked(proof.a[0], proof.a[1]), bytes32(0), bytes32(0), bytes32(0), NexusZeroVerifier.PrivacyLevel.Transparent);
        verifier.operatorMarkProofVerified(proofId, true);

        // Complete the transfer
        bridge.completeTransfer(requestId, proof, pub);

        // Check transfer status
        (,,address sender,,,,) = bridge.getTransferDetails(requestId);
        assert(sender == address(this));
    }
}
