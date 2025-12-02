# Proof Mechanism Decision - NexusZero Protocol

## Decision

- On-chain verification will use Groth16 (BN254/alt_bn128) for now for gas efficiency and wide compatibility with EVM chains.
- Off-chain, the project will maintain a quantum-resistant stack (lattice-based proofs) for research pipelines and to enable future migration plans (e.g., zk-STARKs or lattice-based SNARK primitives when feasible).

## Rationale

- Groth16 provides a small proof size and a reliable pairing-based verifier (using EVM precompiles) that is currently cost-effective for on-chain verification.
- Quantum-resistant proof systems are an active research area for on-chain friendly verification (STARKs are large but may be reduced using recursive proof systems). For now, we prioritize a practical, deployable system.

## Migration Plan

- Monitor advances in on-chain proof verification for STARKs or lattice-based schemes.
- Evaluate recursive proofs (using Groth16 or aggregation) to compress multiple verifications into one on-chain check.
- Develop a compatibility layer enabling proof type rotation: `proof_type` metadata in `ProofMetadata` and `proofType` in on-chain records.

## Security Notes

- Groth16 is not quantum-resistant. If post-quantum adversary protection is required, consider offloading sensitive or long-lived attestations to specialized L2 or non-revocable off-chain attestations.
