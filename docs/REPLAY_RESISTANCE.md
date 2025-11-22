# Replay Resistance in Nexuszero Proof Protocol

## Threat Model

A valid zero-knowledge proof generated in one protocol instance could be replayed in another context if:

- The statement bytes are identical.
- The commitment list is identical.
- Fiat-Shamir challenge derives only from those deterministic elements.

Currently `compute_challenge(statement, commitments)` hashes only statement serialization + commitment values. This protects integrity but does **not** bind proofs to an application session or domain.

## Recommended Enhancements

1. Domain Separation:
   - Add a constant tag (e.g. `b"NEXUSZERO_ZK_V1"`) at the start of the hash transcript.
2. Context Binding:
   - Extend `Statement` with optional `context: Vec<u8>` (session ID, channel binding, nonce).
   - Include `context` bytes in challenge hashing.
3. Nonce / Anti-Replay Token:
   - Verifier supplies a fresh nonce; prover incorporates nonce into commitments (or added to transcript prior to hashing).
4. Expiry / Freshness:
   - Include timestamp window and reject proofs older than threshold.

## Implementation Sketch

```rust
pub fn compute_challenge(statement: &Statement, commitments: &[Commitment], context: Option<&[u8]>) -> CryptoResult<Challenge> {
    use sha3::{Digest, Sha3_256};
    let mut hasher = Sha3_256::new();
    hasher.update(b"NEXUSZERO_ZK_V1");
    if let Some(ctx) = context { hasher.update(ctx); }
    hasher.update(statement.to_bytes()?);
    for c in commitments { hasher.update(&c.value); }
    let hash = hasher.finalize();
    let mut out = [0u8;32]; out.copy_from_slice(&hash); Ok(Challenge{ value: out })
}
```

## Verification Adjustments

- Verifier supplies/records context and checks it matches expected session.
- Reject proofs missing required context or with stale timestamp.

## Security Gains

| Mechanism     | Prevents                             |
| ------------- | ------------------------------------ |
| Domain tag    | Cross-protocol confusion             |
| Context field | Cross-session replay                 |
| Nonce         | Immediate replay within same session |
| Expiry        | Long-term archival replay            |

## Next Steps

1. Add `context: Option<Vec<u8>>` to `Statement` and builder.
2. Modify challenge computation to accept context.
3. Add tests: replay without context fails; replay with incorrect context fails.
4. Document in README and protocol spec.
