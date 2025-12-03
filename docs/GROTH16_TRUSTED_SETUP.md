# Groth16 Trusted Setup Ceremony Guide

## Overview

The Groth16 proving system used in NexusZero requires a **trusted setup ceremony** to generate the Common Reference String (CRS). This document outlines the security considerations, ceremony process, and best practices for conducting a trusted setup.

## Why Trusted Setup is Required

Groth16 proofs achieve their efficiency through a preprocessing phase that generates:

1. **Proving Key (pk)**: Used by provers to generate proofs
2. **Verification Key (vk)**: Used by verifiers to check proofs
3. **Toxic Waste**: Random values that MUST be destroyed

### Security Implications

If the toxic waste (τ, α, β values) is compromised:

- **An attacker can forge proofs** for any statement
- **The entire system's soundness is broken**
- There is **NO cryptographic evidence** of compromise

## Ceremony Architecture

### Phase 1: Powers of Tau

The Powers of Tau ceremony generates:

```
τ¹, τ², τ³, ..., τⁿ (in both G₁ and G₂)
α·τⁱ, β·τⁱ
```

**Properties:**

- Universal - can be reused across different circuits
- Sequential contributions from multiple participants
- Only ONE honest participant needed for security

### Phase 2: Circuit-Specific Setup

After Powers of Tau, a circuit-specific phase generates:

- Proving key elements specific to the circuit structure
- Verification key for the specific computation

## Recommended Ceremony Process

### Prerequisites

1. **Hardware Requirements**

   - Secure, air-gapped machine (preferred)
   - Minimum 16GB RAM for large circuits
   - Fast SSD storage

2. **Software Requirements**
   - Verified ceremony software (snarkjs, circom, or custom)
   - Entropy sources (hardware RNG preferred)

### Step-by-Step Process

#### 1. Initial Setup

```bash
# Generate initial parameters (coordinator)
snarkjs powersoftau new bn128 <power> pot_0000.ptau -v

# Power should be at least log2(circuit_constraints)
# For NexusZero: power >= 20 (1M+ constraints)
```

#### 2. Participant Contributions

```bash
# Each participant runs:
snarkjs powersoftau contribute pot_<prev>.ptau pot_<next>.ptau \
    --name="Participant <N>" \
    --entropy="<random_input>" \
    -v

# Verify contribution
snarkjs powersoftau verify pot_<next>.ptau
```

#### 3. Entropy Collection (CRITICAL)

Each participant MUST use multiple entropy sources:

- `/dev/urandom` (Linux) or `CryptGenRandom` (Windows)
- Hardware RNG (if available)
- Keyboard/mouse movement timing
- Custom entropy from personal data

**Example Multi-Source Entropy:**

```bash
# Combine multiple entropy sources
ENTROPY=$(cat /dev/urandom | head -c 1024 | sha256sum | cut -d' ' -f1)
ENTROPY+=$(date +%s%N | sha256sum | cut -d' ' -f1)
ENTROPY+=$(ps aux | sha256sum | cut -d' ' -f1)
```

#### 4. Finalization (Coordinator)

```bash
# Apply random beacon (after all contributions)
snarkjs powersoftau beacon pot_final.ptau pot_beacon.ptau \
    <beacon_hash> <iterations>

# Prepare for phase 2
snarkjs powersoftau prepare phase2 pot_beacon.ptau pot_final.ptau -v

# Verify entire ceremony
snarkjs powersoftau verify pot_final.ptau
```

#### 5. Circuit-Specific Phase 2

```bash
# Generate initial phase 2 file
snarkjs groth16 setup circuit.r1cs pot_final.ptau circuit_0000.zkey

# Contributors participate
snarkjs zkey contribute circuit_0000.zkey circuit_0001.zkey \
    --name="Participant 1" -v

# Export verification key
snarkjs zkey export verificationkey circuit_final.zkey verification_key.json
```

### Toxic Waste Destruction

**CRITICAL: Each participant MUST ensure toxic waste destruction:**

1. **Memory Clearing**

   - Use secure memory wiping (our `zeroize` implementation)
   - Power cycle the machine after contribution

2. **No Backups**

   - Never write random values to persistent storage
   - Disable swap during contribution

3. **Hardware Destruction (Maximum Security)**
   - For high-value applications, physically destroy contribution hardware
   - At minimum, perform secure disk wipe

## Verification

### Ceremony Transcript Verification

Anyone can verify the ceremony:

```bash
# Verify the entire chain of contributions
snarkjs powersoftau verify pot_final.ptau

# Verify phase 2
snarkjs zkey verify circuit.r1cs pot_final.ptau circuit_final.zkey
```

### What Verification Proves

✅ All contributions are mathematically valid
✅ Parameters are correctly computed
✅ No structural issues in the ceremony

### What Verification CANNOT Prove

❌ Whether toxic waste was actually destroyed
❌ Whether any participant was compromised
❌ Whether the entropy was truly random

## NexusZero-Specific Considerations

### Circuit Specifications

| Parameter       | Value             | Notes                  |
| --------------- | ----------------- | ---------------------- |
| Curve           | BN254 (alt_bn128) | EVM-compatible         |
| Max Constraints | 2^20              | ~1M constraints        |
| Powers of Tau   | 2^21              | Supports future growth |

### Multi-Party Ceremony Recommendations

For NexusZero's production deployment:

1. **Minimum Participants**: 20+ independent contributors
2. **Geographic Distribution**: Contributors from 10+ countries
3. **Institutional Diversity**: Mix of individuals, companies, academics
4. **Transparent Process**:
   - Public attestations from each participant
   - Published transcript hashes
   - Video recordings of contributions (optional)

### Fallback: Transparent Alternatives

If trusted setup is unacceptable for your use case, consider:

1. **Bulletproofs** (already implemented in NexusZero)

   - No trusted setup required
   - Larger proofs, slower verification
   - Suitable for range proofs

2. **PLONK with Universal Setup**

   - One-time setup reusable for all circuits
   - Updatable reference string

3. **STARKs**
   - No trusted setup (transparent)
   - Larger proofs but post-quantum secure

## Security Checklist

### Before Ceremony

- [ ] Verify ceremony software integrity (reproducible builds)
- [ ] Prepare air-gapped contribution machine
- [ ] Collect multiple entropy sources
- [ ] Disable networking during contribution

### During Ceremony

- [ ] Verify previous contributions before adding yours
- [ ] Use unique, high-entropy randomness
- [ ] Do NOT save random values anywhere
- [ ] Complete contribution in single session

### After Ceremony

- [ ] Securely wipe contribution machine
- [ ] Power cycle / restart the machine
- [ ] Publish attestation with contribution hash
- [ ] Verify final parameters independently

## Incident Response

### If Toxic Waste Compromise Suspected

1. **Immediate Actions**

   - Halt all proof generation/verification
   - Notify all protocol users
   - Do NOT accept any new proofs

2. **Recovery**

   - Conduct new ceremony from scratch
   - Require increased participant count
   - Consider migrating to transparent system

3. **Post-Mortem**
   - Document compromise vector
   - Update security procedures
   - Consider hardware security modules (HSMs)

## References

- [Zcash Powers of Tau Ceremony](https://z.cash/technology/paramgen/)
- [BGM17: Scalable Multi-Party Computation](https://eprint.iacr.org/2017/1050)
- [snarkjs Documentation](https://github.com/iden3/snarkjs)
- [Groth16 Original Paper](https://eprint.iacr.org/2016/260)

## Version History

| Version | Date       | Changes               |
| ------- | ---------- | --------------------- |
| 1.0     | 2025-12-02 | Initial documentation |

---

**Security Contact**: For trusted setup security concerns, contact the NexusZero security team.
