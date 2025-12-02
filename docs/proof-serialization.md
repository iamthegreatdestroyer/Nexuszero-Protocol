# Proof Serialization & Cross-Chain Message Format

This document defines the on-chain and cross-chain proof serialization formats used by NexusZero.

## Groth16 Proof Format (EVM - canonical)

- a: 2 \* 32-byte big-endian integers (a.x, a.y) => 64 bytes
- b: 2 _ 2 _ 32-byte big-endian integers (b[0].x, b[0].y, b[1].x, b[1].y) => 128 bytes
- c: 2 \* 32-byte integers => 64 bytes
- Total: 256 bytes

When calling `submitProof(a,b,c, publicInputs, circuitId, senderCommit, recipientCommit, privacyLevel)`, the structured tuples are used explicitly. For ABI compatibility, the connector may pass `a,b,c` as `uint256[2]`, `uint256[2][2]`, `uint256[2]` respectively and public inputs as `uint256[]`.

## Cross-Chain Bridge Transfer Message

- The canonical message includes:

  - transferId: bytes32
  - sourceChain: bytes32
  - targetChain: bytes32
  - amount: uint256
  - token: address
  - recipientCommitment: bytes32
  - proofId: bytes32 (returned after `submitProof`)

- On the receiving chain, relayer calls `completeTransfer` with `transferId` and `proofId`, and `proofId` is verified by the Verifier using `verifyProofById`.

## Connector Behavior

- EVM connectors parse the proof bytes into `a,b,c` typed values and call `submitProof(...)` structured function.
- If proof_public_inputs are unavailable in the main proof bytes, connectors should include them in the `publicInputs` param (uint256[]).

## Versioning & Backwards Compatibility

- Maintain a `proof_type` and `proof_version` in `ProofMetadata` to allow future changes in proof encoding.
- On-chain code should store a `proof_type` metadata string to help off-chain tooling decode proof bytes.

## Notes

- For non-Groth16 chains (e.g., Solana), ensure program-side verification matches proof type and any discrepancy results in proof rejection.
- Keep serialization deterministic: big-endian representation and fixed-size integer encodings.
