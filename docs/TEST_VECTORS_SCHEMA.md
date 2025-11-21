# Test Vectors Unified Schema

Version: 1.0 (initial consolidation)

## Top-Level Structure (per file)

```jsonc
{
  "test_suite": "<string>",
  "version": "1.0",
  "description": "<string>",
  "vectors": [ /* vector objects */ ],
  // Optional for proof vectors:
  "security_properties": [ {"property": "completeness", ...} ]
}
```

## LWE Vector Object

```jsonc
{
  "name": "LWE-128-Simple",
  "security_level": 128,
  "parameters": {
    "dimension": 512,
    "modulus": 12289,
    "sigma": 3.2
  },
  // optional deterministic seeds retained for future use:
  "secret_key": { "s": [1, 0, 1, 0, 1] },
  "public_key": { "A_seed": "hex", "b_seed": "hex" },
  "test_cases": [
    {
      "plaintext_bit": 0,
      "encryption_seed": "aaaaaaaaaaaaaaaa", // optional
      "expected_ciphertext": { "c1_hash": "...", "c2_mod": 6144 }, // optional
      "decryption_result": 0 // optional (if absent we assert decrypt == plaintext_bit)
    }
  ]
}
```

## Ring-LWE Vector Object

```jsonc
{
  "name": "Ring-LWE-128-Polynomial-Operations",
  "security_level": 128,
  "parameters": {
    "degree": 256,
    "modulus": 7681,
    "sigma": 3.2
  },
  "polynomial_tests": [
    {
      "operation": "addition",
      "poly_a": [1, 2, 3, 0, 0],
      "poly_b": [4, 5, 6, 0, 0],
      "expected": [5, 7, 9, 0, 0],
      "description": "Simple polynomial addition"
    }
  ],
  "encryption_tests": [
    {
      "message": [1, 0, 0, 0], // raw bytes, interpreted bit-wise (b & 1)
      "encryption_seed": "test_seed_128", // optional
      "expected_properties": { "c1_degree": 255 } // optional
    }
  ],
  "ntt_tests": [
    {
      "test": "ntt_invertibility",
      "input_polynomial": [1, 2, 3, 4, 5]
    }
  ]
}
```

## Proof Vector Object

```jsonc
{
  "name": "Discrete-Log-Proof-Basic",
  "proof_type": "discrete_log", // or "preimage"
  "parameters": {
    /* type-dependent */
  },
  "proof_components": {
    /* reference values for deterministic mode */
  },
  "verification": { "expected_result": "valid" },
  "test_cases": [
    /* optional structured cases */
  ]
}
```

## Parsing & Execution Rules

- Missing optional fields are ignored safely.
- For LWE: m (samples) derived as 2 \* dimension if not provided.
- For Ring-LWE encryption_tests.message: bytes converted to bits via (byte & 1).
- For Proofs: classification by `name` fallback when `proof_type` missing.
- If `decryption_result` present: assert decrypted == provided; else assert equality with original plaintext.
- Polynomial tests presently support only `operation == "addition"`; others may extend with `expected_schoolbook` etc.

## Validation Guidelines

- `dimension`, `degree` must be > 0.
- `modulus` must be >= dimension (basic safety); prime preference handled outside schema.
- `sigma` > 0.
- `plaintext_bit` must be 0 or 1.

## Future Extensions

- Deterministic seeding for reproducible ciphertext hashing.
- Rich polynomial test operations: subtraction, multiplication (schoolbook vs NTT).
- NTT invertibility verification.
- Structured proof tampering scenarios.
- Versioned schema migration (`version` field).

## Rationale

This unified schema preserves existing JSON assets while enabling progressive enhancement without breaking backward compatibility. Optional fields allow richer deterministic or analytical validation when cryptographic hashing and reproducible randomness are integrated.

---

Maintained by: Nexuszero Protocol
Version: 1.0
"
