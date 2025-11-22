# Pedersen Commitment Binding & Hiding Analysis

## Construction

C = g^v \* h^r mod p

- p: 256-bit prime (secp256k1 field) ensuring multiplicative group structure.
- g,h: Independent generators derived via SHA3-256 hashes of distinct labels.
- v: Value (range proven value or discrete input)
- r: Blinding factor (32 random bytes interpreted as big-endian integer)

## Binding Property

Under the Discrete Log assumption in the group modulo p, finding two openings (v,r) != (v',r') such that:

```
g^v * h^r ≡ g^{v'} * h^{r'} (mod p)
```

Implies:

```
g^{v-v'} ≡ h^{r'-r} (mod p)
```

Which yields a non-trivial discrete logarithm relation between g and h. Because g and h are independently hashed, probability that h = g^x is negligible; thus forging a second opening breaks DL. Hence commitment is computationally binding.

## Hiding Property

Given commitment C, distribution over C for fixed v and random r is (close to) uniform in subgroup due to multiplication by h^r. Without knowledge of r, extracting v reduces to solving discrete log or brute-force search over v when v space small. Using large blinding space (≥ 256 bits of entropy) yields statistical hiding for practical purposes.

## Collision Resistance & Generator Independence

- Generators constructed: g = H("bulletproofs-g"), h = H("bulletproofs-h")
- Distinct labels avoid related-output risk. Optional improvement: additional rejection sampling to ensure neither generator maps to low-order element (not required here as p is prime and hash output < p).

## Range Proof Interaction

When proving v ∈ [min, max), protocol now decomposes (v - min) while committing to v. Inner product argument reveals no direct information about v beyond membership, since decomposition bits remain hidden behind separate bit commitments.

## Side-Channel Considerations

- Modular exponentiation uses constant-time algorithm (`ct_modpow`).
- Ensure blinding scalar reduction avoids variable-time big integer divisions (already using repeated squaring). Audit for any secret-dependent branching in blinding generation (none; RNG usage acceptable).

## Recommended Hardening

1. Enforce r ∈ [0, p-1] by reducing big-endian bytes modulo p explicitly.
2. Add subgroup order checks (not needed—prime field ensures order p-1; but could ensure generators not 0 or 1 trivial values).
3. Integrate transcript domain separation for multi-protocol safety.

## Summary

Pedersen commitments as implemented are binding under DL and hiding with 256-bit random blinding. With prime modulus and independent generators, security goals are met. Remaining tasks: formal proofs in protocol spec, implement r reduction, and add context binding.
