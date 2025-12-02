# Dynamic Verification Fees

This file documents how fees are configured and recommended adjustments.

- `NexusZeroVerifier` exposes a `mapping(PrivacyLevel => uint256) public verificationFeeByLevel`.
- Contracts and connectors should call `verifier.verificationFeeByLevel(uint8(privacyLevel))` to determine the expected fee.
- Connectors should estimate gas using the provided `estimate_fee` methods and use those to set `msg.value` when calling `submitProof` or `submitProofRaw`.

Recommendations:

- Maintain conservative default fees on mainnet and tune via `setVerificationFeeForLevel` when applying to live networks.
- Use chain-specific estimation in connectors (price \* gas).
