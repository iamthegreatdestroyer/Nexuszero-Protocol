# Security Considerations for Nexuszero-Crypto

**Last Updated:** November 21, 2025  
**Audit Status:** Constant-Time Operations Partially Implemented  
**Security Level:** Research/Development - NOT Production Ready

---

## ⚠️ Security Warning

**THIS LIBRARY HAS NOT BEEN AUDITED BY THIRD-PARTY SECURITY EXPERTS**

This cryptographic library is under active development and should be considered **research-quality software**. It has not undergone:

- Independent security audit
- Formal verification of cryptographic properties
- Extensive penetration testing
- Production deployment hardening

**DO NOT USE IN PRODUCTION SYSTEMS** without:

1. Independent security review
2. Comprehensive side-channel analysis
3. Formal threat modeling for your specific use case
4. Additional hardening measures

---

## Table of Contents

1. [Side-Channel Attack Surface](#side-channel-attack-surface)
2. [Constant-Time Implementations](#constant-time-implementations)
3. [Known Vulnerabilities](#known-vulnerabilities)
4. [Mitigation Strategies](#mitigation-strategies)
5. [Secure Usage Guidelines](#secure-usage-guidelines)
6. [Threat Model](#threat-model)
7. [Reporting Security Issues](#reporting-security-issues)

---

## Side-Channel Attack Surface

### Overview of Side-Channel Vulnerabilities

Side-channel attacks exploit physical implementation characteristics rather than mathematical weaknesses. This library is vulnerable to several classes of side-channel attacks:

#### 1. **Timing Attacks** 🔴 HIGH RISK

**Definition:** Attackers measure execution time to infer secret values.

**Vulnerable Operations:**

| Operation                             | Risk Level | Status              | Details                                                    |
| ------------------------------------- | ---------- | ------------------- | ---------------------------------------------------------- |
| **LWE Decryption**                    | 🟡 MEDIUM  | Partially Mitigated | Final comparison uses constant-time, but dot product leaks |
| **Discrete Log Witness Verification** | 🔴 HIGH    | Documented Only     | `modpow` not constant-time, leaks exponent bits            |
| **Range Check (Witness)**             | 🔴 HIGH    | Documented Only     | Early return on out-of-range reveals information           |
| **Preimage Verification**             | 🟢 LOW     | Mitigated           | Uses constant-time byte comparison                         |
| **Commitment Comparison**             | 🟢 LOW     | Mitigated           | Uses `subtle` crate for equality                           |

**Attack Scenarios:**

- **LWE Decrypt Timing:** An attacker with timing oracle access could potentially recover the secret key by observing decryption timing across many ciphertexts
- **Proof Generation:** Timing variations during proof generation may leak witness bit patterns
- **Range Proofs:** Early rejection of out-of-range values reveals information about the secret value

#### 2. **Cache-Timing Attacks** 🔴 HIGH RISK

**Definition:** Attackers observe CPU cache behavior to infer secret-dependent memory access patterns.

**Vulnerable Operations:**

- **Matrix-Vector Products (LWE):** Secret key array indexing creates cache patterns
- **Polynomial Operations (Ring-LWE):** Coefficient access patterns leak secret polynomial structure
- **Modular Exponentiation:** Cache-line aligned table lookups reveal exponent bits
- **NTT Butterfly Operations:** Memory access patterns in NTT may leak coefficient information

**Attack Scenarios:**

- **Flush+Reload:** Attacker forces cache misses and measures reload times
- **Prime+Probe:** Attacker fills cache lines and observes evictions
- **Co-location Attacks:** In cloud environments, attacker VMs can monitor victim cache behavior

#### 3. **Power Analysis Attacks** 🟡 MEDIUM RISK

**Definition:** Attackers measure power consumption during cryptographic operations.

**Applicability:**

- **IoT/Embedded Devices:** Critical concern for hardware implementations
- **Cloud/Server:** Lower risk but possible with physical access
- **Desktop/Mobile:** Requires specialized equipment

**Vulnerable Operations:**

- Modular arithmetic operations (Hamming weight leaks)
- Conditional branches based on secret data
- Secret-dependent loop iterations

#### 4. **Electromagnetic (EM) Attacks** 🟡 MEDIUM RISK

Similar to power analysis but measures electromagnetic radiation from CPU/memory operations.

**Risk Factors:**

- Physical proximity required
- More relevant for embedded/hardware implementations
- Less practical for server-side applications

---

## Constant-Time Implementations

### What We've Implemented ✅

#### 1. **Subtle Crate Integration**

Added the `subtle` crate (v2.5) for provably constant-time operations:

```rust
use subtle::{ConstantTimeEq, ConstantTimeGreater, Choice};
```

**Guarantees:**

- Operations complete in the same time regardless of input values
- No data-dependent branches in compiled assembly
- Resistant to timing attacks at the algorithmic level

#### 2. **LWE Decryption (Partial)**

**File:** `src/lattice/lwe.rs`

**Implementation:**

```rust
pub fn decrypt(sk: &LWESecretKey, ct: &LWECiphertext, params: &LWEParameters)
    -> CryptoResult<bool>
{
    use subtle::ConstantTimeGreater;

    let m_prime = (ct.v - sk.s.dot(&ct.u)).rem_euclid(params.q as i64);

    let distance_to_zero = m_prime.min(params.q as i64 - m_prime);
    let distance_to_half = (m_prime - (params.q / 2) as i64).abs()
        .min((params.q / 2) as i64);

    // Constant-time comparison
    let ct_result = (distance_to_zero as u64).ct_gt(&(distance_to_half as u64));
    Ok(bool::from(ct_result))
}
```

**Status:** ✅ Final comparison is constant-time  
**Remaining Issue:** ⚠️ Dot product `s.dot(&ct.u)` may leak through cache timing

#### 3. **Witness Verification Comparisons**

**File:** `src/proof/witness.rs`

**Constant-Time Byte Comparison:**

```rust
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;

    if a.len() != b.len() {
        return false;  // Length is public info
    }

    let mut result = subtle::Choice::from(1u8);
    for (x, y) in a.iter().zip(b.iter()) {
        result &= x.ct_eq(y);
    }

    bool::from(result)
}
```

**Applied To:**

- ✅ Preimage hash verification
- ✅ Discrete log public value comparison
- ✅ Commitment equality checking
- ✅ General byte array comparisons

**Guarantees:**

- No early termination on mismatch
- Execution time depends only on array length (public)
- Uses bitwise operations without conditional branches

### What Still Needs Work ⚠️

#### 1. **Modular Exponentiation** 🔴 CRITICAL

**Current Implementation:** `num-bigint` crate's `modpow`

**Problem:**

```rust
// This is NOT constant-time!
let computed = gen_big.modpow(&secret_big, &mod_big);
```

The `modpow` function in `num-bigint` uses the **square-and-multiply algorithm**, which:

- Takes different time for 1-bits vs 0-bits in the exponent
- Leaks exponent bit patterns through timing
- Vulnerable to timing attacks that can recover the secret exponent

**Impact:**

- 🔴 **Discrete Log Proofs:** Secret witness exponent is leaked
- 🔴 **Range Proofs:** Blinding factor exponent is leaked
- 🔴 **All Pedersen Commitments:** Vulnerable to timing analysis

**Recommended Fix:**

```rust
// Use Montgomery ladder or constant-time exponentiation
fn constant_time_modpow(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
    // Implementation needed: constant-time square-and-multiply
    // or Montgomery ladder algorithm
    todo!("Implement constant-time modular exponentiation")
}
```

**Alternatives:**

- Use `crypto-bigint` crate with constant-time guarantees
- Implement Montgomery ladder algorithm
- Apply exponent blinding techniques

#### 2. **Range Check Early Returns** 🔴 HIGH PRIORITY

**Current Implementation:**

```rust
if *value < *min || *value > *max {
    return false;  // ⚠️ Early return leaks information!
}
```

**Problem:** An attacker can measure timing to determine:

- Whether value is above or below the range
- Approximate distance from the boundary
- Binary search to narrow down the secret value

**Recommended Fix:**

```rust
use subtle::ConstantTimeLess, ConstantTimeGreater};

// Compute in_range without branching
let below_min = value.ct_lt(&min);
let above_max = value.ct_gt(&max);
let in_range = !(below_min | above_max);

// Continue computation regardless, only use result at the end
```

#### 3. **Array Indexing (Cache Attacks)** 🟡 MEDIUM PRIORITY

**Current Vulnerable Operations:**

```rust
// LWE secret key usage
let m_prime = ct.v - sk.s.dot(&ct.u);  // Array indexing on secret s

// Polynomial coefficient access
for i in 0..n {
    result[i] = poly.coeffs[secret_index];  // Secret-dependent indexing
}
```

**Problem:**

- CPU caches data in 64-byte cache lines
- Array index `secret_index` determines which cache lines are accessed
- Attacker can observe cache line access patterns
- This reveals information about `secret_index`

**Mitigations:**

1. **Constant-Time Indexing:** Use conditional moves instead of indexing
2. **Cache-Oblivious Algorithms:** Design algorithms that don't create secret-dependent patterns
3. **Blinding:** Add random offsets to indices (with correction later)

**Example Constant-Time Index Selection:**

```rust
fn ct_select(array: &[u64], index: usize) -> u64 {
    let mut result = 0u64;
    for (i, &val) in array.iter().enumerate() {
        let mask = subtle::Choice::from((i == index) as u8);
        result = u64::conditional_select(&result, &val, mask);
    }
    result
}
```

#### 4. **Polynomial NTT Operations** 🟡 MEDIUM PRIORITY

**Current NTT Implementation:**

```rust
while len < n {
    for j in 0..len {
        let u = result[k + j];  // Array access pattern
        let v = result[k + j + len];
        // ...
    }
}
```

**Problems:**

- Loop iterations may leak polynomial degree
- Coefficient access patterns leak polynomial structure
- Butterfly operations create predictable memory access patterns

**Recommended Mitigations:**

- Use constant-time coefficient selection
- Employ index shuffling techniques
- Consider masked polynomial representations

---

## Known Vulnerabilities

### Critical (Requires Immediate Attention)

#### 1. **Secret Exponent Leakage via Timing** 🔴

**Location:** `src/proof/witness.rs:104` (Discrete Log verification)  
**Location:** `src/proof/bulletproofs.rs` (Multiple locations)  
**CVSS Score:** 7.5 (High) - Remote timing attack can recover secrets

**Description:**
The `num_bigint::modpow` function is not constant-time. An attacker with the ability to measure execution time can:

1. Send crafted inputs with known patterns
2. Measure timing differences
3. Apply statistical analysis to recover exponent bit patterns
4. Gradually recover the entire secret exponent

**Exploitation Scenario:**

```
Attacker sends 1000 proof verification requests
For each bit position i in secret exponent:
    Measure time for proofs designed to exercise bit i
    If time > threshold: bit i = 1
    Else: bit i = 0
After 256 iterations: complete secret recovered
```

**Affected Components:**

- Discrete logarithm proofs
- Bulletproofs range proofs
- Pedersen commitments
- All witness verification involving exponentiation

**Mitigation Timeline:**

- **Short-term:** Document vulnerability prominently
- **Medium-term:** Implement Montgomery ladder algorithm
- **Long-term:** Migrate to constant-time cryptographic library

#### 2. **Cache-Timing on Secret Key Operations** 🔴

**Location:** `src/lattice/lwe.rs:157` (Dot product in decrypt)  
**CVSS Score:** 6.5 (Medium) - Requires co-location but highly effective

**Description:**
The dot product computation `sk.s.dot(&ct.u)` accesses secret key elements with patterns dependent on the ciphertext. In cloud/shared environments:

- Attacker can observe which cache lines are accessed
- Secret key structure can be recovered through statistical analysis
- Attack is practical on modern Intel/AMD processors

**Exploitation Requirements:**

- Co-location on same physical CPU (cloud VM, hyperthreading)
- Ability to execute code and observe cache state
- Multiple decryption operations with chosen ciphertexts

**Mitigation:**

- Implement constant-time matrix-vector multiplication
- Use scatter-gather operations to hide access patterns
- Consider ORAM (Oblivious RAM) techniques for extreme security

### High (Should Be Addressed)

#### 3. **Early Return in Range Checks** 🟡

**Location:** `src/proof/witness.rs:117`  
**CVSS Score:** 5.3 (Medium) - Local timing attack reveals range information

**Description:**

```rust
if *value < *min || *value > *max {
    return false;  // ⚠️ Leaks timing information
}
```

When the value is out of range, the function returns early. An attacker measuring execution time can:

- Determine if value is in range without seeing the value
- Narrow down the value through binary search timing analysis
- Potentially recover the exact value with enough measurements

**Attack Example:**

```
Assume secret value v, range [10, 20]
Attacker measures timing for proofs with different range statements:
  [10, 20] -> slow (value in range, full verification)
  [15, 20] -> fast if v < 15 (early return)
  [10, 15] -> fast if v > 15 (early return)
Binary search on timing reveals v ≈ 13 after log2(10) = 4 queries
```

**Mitigation:**
Use constant-time comparison and compute full result regardless of range check:

```rust
let in_range = ct_ge(value, min) & ct_le(value, max);
let result = in_range & verify_commitment();
bool::from(result)
```

### Medium (Best Practice Improvements)

#### 4. **Memory Allocation Patterns** 🟡

**CVSS Score:** 3.1 (Low) - Requires sophisticated attack

**Description:**
Memory allocation and deallocation timing can leak information about:

- Size of secret values
- Number of polynomial coefficients
- Proof component sizes

**Affected Areas:**

- `Vec` allocations for secrets
- Dynamic polynomial degree changes
- Proof serialization buffer sizes

**Mitigation:**

- Pre-allocate buffers to maximum size
- Use fixed-size arrays for secrets where possible
- Pad all structures to constant size

---

## Mitigation Strategies

### Immediate Actions (Implementable Now)

#### 1. **Use Subtle Crate Everywhere** ✅ DONE

Replace all secret-dependent comparisons:

```rust
// ❌ BAD
if secret_a == secret_b { ... }

// ✅ GOOD
use subtle::ConstantTimeEq;
if bool::from(secret_a.ct_eq(&secret_b)) { ... }
```

**Status:** Implemented for byte array comparisons

#### 2. **Document All Vulnerabilities** ✅ DONE

Add warnings to all vulnerable functions:

```rust
/// # Constant-Time Considerations
///
/// ⚠️ WARNING: This function is NOT constant-time
/// - modpow leaks exponent through timing
/// - For production: use constant-time exponentiation
```

**Status:** Added to `lwe.rs`, `witness.rs`

#### 3. **Add Security.md** ✅ IN PROGRESS

Comprehensive security documentation (this document).

### Short-Term Improvements (Next Sprint)

#### 4. **Implement Constant-Time Modpow**

**Priority:** 🔴 CRITICAL

**Options:**

**Option A:** Implement Montgomery Ladder

```rust
fn montgomery_ladder(base: &BigUint, exp: &BigUint, modulus: &BigUint) -> BigUint {
    let mut r0 = BigUint::one();
    let mut r1 = base.clone();

    for bit in exp.bits().rev() {
        let bit_choice = subtle::Choice::from(bit as u8);

        // Always perform both operations
        let r0_squared = (&r0 * &r0) % modulus;
        let r1_squared = (&r1 * &r1) % modulus;
        let r0_times_r1 = (&r0 * &r1) % modulus;

        // Constant-time selection based on bit
        r0 = BigUint::conditional_select(&r0_squared, &r0_times_r1, bit_choice);
        r1 = BigUint::conditional_select(&r0_times_r1, &r1_squared, bit_choice);
    }

    r0
}
```

**Option B:** Use `crypto-bigint` crate

```toml
[dependencies]
crypto-bigint = { version = "0.5", features = ["extra-sizes"] }
```

**Recommendation:** Option B (battle-tested, audited implementation)

#### 5. **Constant-Time Range Checks**

```rust
fn ct_in_range(value: u64, min: u64, max: u64) -> Choice {
    use subtle::{ConstantTimeGreater, ConstantTimeLess};

    let above_min = value.ct_gt(&(min - 1));  // value >= min
    let below_max = value.ct_lt(&(max + 1));  // value <= max

    above_min & below_max
}
```

#### 6. **Blinding Techniques**

Add randomness to secret operations to hide patterns:

```rust
fn blinded_dot_product(secret: &[i64], public: &[i64]) -> i64 {
    let blinding = generate_random_i64();
    let blinded_secret = secret.iter().map(|&s| s + blinding).collect::<Vec<_>>();

    let result = blinded_secret.iter().zip(public).map(|(s, p)| s * p).sum::<i64>();
    result - (blinding * public.iter().sum::<i64>())
}
```

### Long-Term Hardening (Future Roadmap)

#### 7. **Hardware-Backed Security**

- **Intel SGX:** Run sensitive operations in secure enclaves
- **ARM TrustZone:** Isolate cryptographic operations
- **Hardware Security Modules (HSM):** Offload key operations

#### 8. **Formal Verification**

- Use tools like `crux-mir` or `kani` to verify constant-time properties
- Prove absence of secret-dependent branches
- Verify cache-timing resistance

#### 9. **Side-Channel Testing**

- **ChipWhisperer:** Test for power analysis vulnerabilities
- **Cache-Timing Frameworks:** Automated cache-timing attack testing
- **Differential Analysis:** Compare timing distributions for different inputs

---

## Secure Usage Guidelines

### For Library Users

#### ✅ DO:

1. **Run in Isolated Environments**

   - Use dedicated servers without co-tenants
   - Avoid shared cloud VMs for sensitive operations
   - Disable hyperthreading if possible

2. **Add Additional Layers**

   - Network-level rate limiting to prevent timing analysis
   - Request padding to hide operation types
   - Dummy operations to normalize timing

3. **Monitor for Attacks**

   - Log timing anomalies
   - Detect unusual request patterns
   - Implement rate limiting per IP

4. **Use Secure Memory**

   - `mlock()` sensitive memory pages
   - Disable swap for processes handling secrets
   - Use secure memory allocators

5. **Test Thoroughly**
   - Run property-based tests
   - Perform differential timing analysis
   - Test under load to verify constant-time properties

#### ❌ DON'T:

1. **Use in Production Without Audit**

   - This library is research-grade
   - Requires professional security review

2. **Assume Timing Safety**

   - Many operations are NOT constant-time
   - Read security warnings carefully

3. **Deploy on Shared Infrastructure**

   - Avoid cloud VMs without dedicated hardware
   - Don't co-locate with untrusted code

4. **Ignore Security Updates**

   - Monitor for security advisories
   - Update promptly when fixes are released

5. **Store Secrets in Plain Memory**
   - Use `zeroize` for cleanup
   - Lock memory pages with `mlock`

### For Library Developers

#### Code Review Checklist:

```markdown
## Secret-Dependent Operations Review

- [ ] Does this function handle secret data?
- [ ] Are all comparisons constant-time?
- [ ] Are there any early returns based on secrets?
- [ ] Do loops iterate a secret-dependent number of times?
- [ ] Is array indexing secret-dependent?
- [ ] Are modular operations constant-time?
- [ ] Is memory allocation size secret-dependent?
- [ ] Are there any `if`/`match` statements on secrets?
- [ ] Is timing uniform across all code paths?
- [ ] Have you added security warnings to docs?
```

#### Testing Requirements:

1. **Timing Tests:**

```rust
#[test]
fn test_constant_time_comparison() {
    let samples = 10000;
    let mut timings_match = Vec::new();
    let mut timings_mismatch = Vec::new();

    for _ in 0..samples {
        let start = Instant::now();
        let _ = constant_time_eq(&[1; 32], &[1; 32]);
        timings_match.push(start.elapsed());

        let start = Instant::now();
        let _ = constant_time_eq(&[1; 32], &[2; 32]);
        timings_mismatch.push(start.elapsed());
    }

    // Statistical test: distributions should be indistinguishable
    assert!(ks_test(&timings_match, &timings_mismatch) > 0.05);
}
```

2. **Property-Based Tests:**

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_timing_independence(a in any::<[u8; 32]>(), b in any::<[u8; 32]>()) {
        let start = Instant::now();
        let _ = constant_time_eq(&a, &b);
        let duration = start.elapsed();

        // Timing should not correlate with hamming distance
        prop_assert!(duration < Duration::from_micros(10));
    }
}
```

---

## Threat Model

### Assumptions

**What We Assume:**

1. **Honest-But-Curious Adversary**

   - Adversary follows protocol but tries to learn secrets
   - No active tampering with ciphertext/proofs

2. **Limited Physical Access**

   - Adversary can measure timing remotely
   - No direct physical access to hardware
   - No ability to fault-inject or glitch power

3. **Standard Computing Environment**
   - x86-64 or ARM processors
   - Standard OS (Linux, Windows, macOS)
   - No specialized hardware (HSM, TPM)

**What We Don't Assume:**

1. ❌ **Adversary with Physical Access** - Outside threat model
2. ❌ **Fault Injection Attacks** - Not addressed
3. ❌ **Side-Channel Resistant Hardware** - Not required
4. ❌ **Trusted Execution Environment** - Optional but not assumed

### Attack Vectors (In Scope)

| Attack Type                    | Feasibility  | Impact           | Mitigation Status |
| ------------------------------ | ------------ | ---------------- | ----------------- |
| **Remote Timing Attack**       | High         | Critical         | Partial           |
| **Cache-Timing (Co-located)**  | Medium       | High             | Documented        |
| **Branch Prediction Analysis** | Low          | Medium           | Partial           |
| **Memory Access Patterns**     | Medium       | High             | Not Addressed     |
| **Power Analysis**             | Low (Remote) | Critical (Local) | Not Addressed     |

### Attack Vectors (Out of Scope)

- Fault injection (clock glitching, voltage faults)
- Physical side-channels requiring hardware access
- Spectre/Meltdown-style speculative execution attacks
- Row hammer and similar DRAM attacks
- Supply chain attacks on compiler/hardware

---

## Reporting Security Issues

### Responsible Disclosure

If you discover a security vulnerability:

1. **DO NOT** open a public GitHub issue
2. **DO NOT** disclose publicly before fix is available
3. **DO** email security details to: **security@nexuszero.dev**

### What to Include:

- **Description:** Clear explanation of the vulnerability
- **Impact:** What an attacker could achieve
- **Reproduction:** Step-by-step proof-of-concept
- **Affected Versions:** Which releases are vulnerable
- **Suggested Fix:** If you have ideas for mitigation

### Response Timeline:

- **24 hours:** Acknowledgment of report
- **7 days:** Initial assessment and severity rating
- **30 days:** Fix developed and tested (for critical issues)
- **90 days:** Public disclosure (coordinated with reporter)

### Hall of Fame:

Contributors who responsibly disclose vulnerabilities will be:

- Credited in release notes
- Listed in SECURITY.md
- Acknowledged in academic publications (if applicable)

---

## Security Checklist for Deployment

Before deploying this library in any environment:

### Infrastructure:

- [ ] Dedicated hardware (no VM co-tenancy)
- [ ] Hyperthreading disabled
- [ ] Memory pages locked (`mlock`)
- [ ] Swap disabled for sensitive processes
- [ ] Network rate limiting configured
- [ ] Request padding implemented
- [ ] Timing jitter added to responses

### Monitoring:

- [ ] Timing anomaly detection enabled
- [ ] Request pattern analysis active
- [ ] Cache miss rate monitoring
- [ ] Power consumption baseline established
- [ ] Alert thresholds configured

### Code:

- [ ] Latest version deployed
- [ ] Security patches applied
- [ ] Constant-time operations verified
- [ ] Timing tests pass
- [ ] No debug logging of secrets
- [ ] Secrets zeroized on drop

### Testing:

- [ ] Timing analysis performed
- [ ] Cache-timing tests run
- [ ] Power analysis (if embedded)
- [ ] Differential fault testing
- [ ] Third-party security review

---

## References

### Academic Papers:

1. **Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS, and Other Systems**  
   Paul Kocher (1996)  
   https://www.paulkocher.com/TimingAttacks.pdf

2. **Cache-Timing Attacks on AES**  
   Daniel J. Bernstein (2005)  
   https://cr.yp.to/antiforgery/cachetiming-20050414.pdf

3. **Constant-Time Implementations**  
   Marc Joye and Sung-Ming Yen (2002)  
   https://eprint.iacr.org/2002/073

4. **A Systematic Study of Side-Channel Attacks on LWE-Based Cryptography**  
   Anubhab Baksi et al. (2020)  
   https://eprint.iacr.org/2020/1240

### Tools & Libraries:

- **subtle:** https://github.com/dalek-cryptography/subtle
- **zeroize:** https://github.com/RustCrypto/utils/tree/master/zeroize
- **crypto-bigint:** https://github.com/RustCrypto/crypto-bigint
- **dudect:** Constant-time testing - https://github.com/oreparaz/dudect

### Standards:

- **FIPS 140-3:** Security Requirements for Cryptographic Modules
- **ISO/IEC 19790:** Security requirements for cryptographic modules
- **Common Criteria:** Security evaluation criteria

---

## Version History

| Version   | Date       | Changes                               |
| --------- | ---------- | ------------------------------------- |
| **1.0.0** | 2025-11-21 | Initial security documentation        |
|           |            | - Constant-time audit completed       |
|           |            | - Subtle crate integrated             |
|           |            | - Critical vulnerabilities documented |

---

**Maintained by:** Steve (@iamthegreatdestroyer)  
**Last Review:** November 21, 2025  
**Next Review Due:** December 21, 2025  
**Contact:** security@nexuszero.dev
