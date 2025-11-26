This PR implements optional un-blinding for the `ct_modpow_blinded` function in `nexuszero-crypto::utils::constant_time`.

Key changes:

- Adds `ct_modpow_blinded_with_order(base, exponent, modulus, Option<&BigUint>) -> BigUint` to permit optional un-blinding when `group_order` is provided (private `r` generation ensures `r` is invertible modulo `group_order`).
- Adds `ct_modpow_blinded_with_r(base, exponent, modulus, r, Option<&BigUint>)` helper for deterministic testing / control over blinding factor.
- Implemented `gcd_biguint` and `modinv_biguint` helper functions that compute GCD and modular inverse for BigUint using `BigInt` arithmetic.
- Extended docs and added unit tests verifying: (a) un-blinding with known `group_order` yields the same result as `ct_modpow`, (b) blinded result differs when no group order is provided, and (c) deterministic un-blind with a provided `r` works as expected.

Notes and considerations:

- The un-blind operation requires that the multiplicative group order is provided and `r` is invertible modulo that order. For prime modulus p, group order should be p-1.
- The function maintains backward compatibility: `ct_modpow_blinded(base, exponent, modulus)` keeps the original behavior (returns blinded result).
- Additional integration tests and performance/security review are recommended for production use (invoking modular inverse has performance and potential timing implications).

Please review: I can tidy up naming, error reporting, or rework behavior if you'd prefer a strict error when `r` is not invertible instead of re-generating.
