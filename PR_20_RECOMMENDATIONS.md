# PR #20 Recommendations: Fix CI/Coverage and NTT Implementation

## Critical Issues Summary

PR #20 introduces CI/CD improvements for code coverage, but contains several critical implementation errors in the NTT (Number Theoretic Transform) code and missing dependencies.

## Issue 1: Missing Rayon Dependency (BLOCKER)

### Problem
```rust
// File: nexuszero-crypto/src/proof/proof.rs:687
use rayon::prelude::*;
// ERROR: rayon is not declared as a dependency
```

### Fix
Add to `nexuszero-crypto/Cargo.toml`:

```toml
[dependencies]
# ... existing dependencies ...
rayon = "1.8"
```

### Verification
```bash
cd nexuszero-crypto
cargo check
cargo test
```

## Issue 2: Incorrect NTT Omega Calculation (BLOCKER)

### Problem
```rust
// File: nexuszero-crypto/src/lattice/ring_lwe.rs:395
let omega_pow = 1u64; // WRONG: omega_pow is never updated

for i in (0..n).step_by(2 * m) {
    for j in 0..m {
        let t = (a[i + j + m] as u128 * omega_pow as u128 % q as u128) as u64;
        let u = a[i + j];
        a[i + j] = (u + t) % q;
        a[i + j + m] = (u + q - t) % q;
    }
    // omega_pow should be updated here!
}
```

### Mathematical Background

The NTT (Number Theoretic Transform) is analogous to the FFT but operates in modular arithmetic. For correct computation:

- `omega` = primitive n-th root of unity modulo q
- `omega_pow` = omega^k for iteration k
- Each butterfly operation uses a different power of omega

### Correct Implementation

```rust
/// Performs in-place forward NTT on input array
/// 
/// # Parameters
/// - `a`: Input/output array (length must be power of 2)
/// - `n`: Length of array (must be power of 2)
/// - `q`: Modulus (must be prime where q ≡ 1 (mod 2n))
/// - `omega`: Primitive n-th root of unity modulo q
pub fn ntt_forward_correct(a: &mut [u64], n: usize, q: u64, omega: u64) {
    assert!(n.is_power_of_two(), "n must be a power of 2");
    assert!(a.len() == n, "Array length must equal n");
    
    // Bit-reversal permutation
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j >= bit {
            j -= bit;
            bit >>= 1;
        }
        j += bit;
        if i < j {
            a.swap(i, j);
        }
    }
    
    // Cooley-Tukey NTT
    let mut m = 2;
    while m <= n {
        // Compute omega_m = omega^(n/m)
        let omega_m = mod_pow(omega, (n / m) as u64, q);
        
        for i in (0..n).step_by(m) {
            let mut omega_pow = 1u64;
            
            for j in 0..(m / 2) {
                let t = mod_mul(a[i + j + m / 2], omega_pow, q);
                let u = a[i + j];
                
                a[i + j] = mod_add(u, t, q);
                a[i + j + m / 2] = mod_sub(u, t, q);
                
                // CRITICAL: Update omega_pow for next iteration
                omega_pow = mod_mul(omega_pow, omega_m, q);
            }
        }
        
        m *= 2;
    }
}

/// Modular multiplication: (a * b) mod q
#[inline]
fn mod_mul(a: u64, b: u64, q: u64) -> u64 {
    ((a as u128 * b as u128) % q as u128) as u64
}

/// Modular addition: (a + b) mod q
#[inline]
fn mod_add(a: u64, b: u64, q: u64) -> u64 {
    let sum = a + b;
    if sum >= q { sum - q } else { sum }
}

/// Modular subtraction: (a - b) mod q
#[inline]
fn mod_sub(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}

/// Modular exponentiation: a^b mod q
fn mod_pow(mut a: u64, mut b: u64, q: u64) -> u64 {
    let mut result = 1u64;
    a %= q;
    while b > 0 {
        if b & 1 == 1 {
            result = mod_mul(result, a, q);
        }
        a = mod_mul(a, a, q);
        b >>= 1;
    }
    result
}
```

### Test to Verify Correctness

```rust
#[test]
fn test_ntt_correctness() {
    // Example: n=8, q=17 (17 ≡ 1 mod 16)
    let n = 8;
    let q = 17;
    let omega = 3; // 3 is a primitive 8th root of unity mod 17
    
    let mut a = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let original = a.clone();
    
    // Forward NTT
    ntt_forward_correct(&mut a, n, q, omega);
    
    // Inverse NTT (omega_inv = omega^(n-1))
    let omega_inv = mod_pow(omega, (n - 1) as u64, q);
    ntt_forward_correct(&mut a, n, q, omega_inv);
    
    // Scale by n^-1 mod q
    let n_inv = mod_pow(n as u64, q - 2, q); // Fermat's little theorem
    for x in a.iter_mut() {
        *x = mod_mul(*x, n_inv, q);
    }
    
    // Should recover original
    assert_eq!(a, original, "NTT round-trip failed");
}
```

## Issue 3: Fake SIMD Implementations (HIGH PRIORITY)

### Problem: butterfly_avx2

```rust
// File: nexuszero-crypto/src/lattice/ring_lwe.rs:305
#[cfg(target_feature = "avx2")]
unsafe fn butterfly_avx2(a: &mut [u64], i: usize, j: usize, m: usize, omega_pow: u64, q: u64) {
    // Claims to use AVX2 but only has scalar code!
    for k in 0..4 {
        let t = (a[i + j + m + k] as u128 * omega_pow as u128 % q as u128) as u64;
        let u = a[i + j + k];
        a[i + j + k] = (u + t) % q;
        a[i + j + m + k] = (u + q - t) % q;
    }
}
```

### Problem: butterfly_neon

```rust
// File: nexuszero-crypto/src/lattice/ring_lwe.rs:340
#[cfg(target_arch = "aarch64")]
unsafe fn butterfly_neon(a: &mut [u64], i: usize, j: usize, m: usize, omega_pow: u64, q: u64) {
    // Claims to use NEON but only has scalar code!
    for k in 0..2 {
        let t = (a[i + j + m + k] as u128 * omega_pow as u128 % q as u128) as u64;
        let u = a[i + j + k];
        a[i + j + k] = (u + t) % q;
        a[i + j + m + k] = (u + q - t) % q;
    }
}
```

### Recommended Fixes

**Option 1: Remove the fake SIMD functions**
```rust
// Remove butterfly_avx2 and butterfly_neon entirely
// Use only the scalar butterfly implementation
```

**Option 2: Implement actual SIMD (Advanced)**

For AVX2:
```rust
#[cfg(target_feature = "avx2")]
unsafe fn butterfly_avx2_real(a: &mut [u64], i: usize, j: usize, m: usize, omega_pow: u64, q: u64) {
    use std::arch::x86_64::*;
    
    // Load 4 elements at a time using __m256i
    let u_vec = _mm256_loadu_si256(a[i+j..].as_ptr() as *const __m256i);
    let v_vec = _mm256_loadu_si256(a[i+j+m..].as_ptr() as *const __m256i);
    
    // Multiply v_vec by omega_pow
    let omega_vec = _mm256_set1_epi64x(omega_pow as i64);
    let q_vec = _mm256_set1_epi64x(q as i64);
    
    // ... actual SIMD operations ...
    
    // Note: Proper SIMD NTT is complex and requires careful implementation
}
```

**Recommendation:** Use Option 1 (remove fake SIMD) unless you have SIMD expertise and can properly implement and test the optimizations.

## Issue 4: PowerShell on Linux (MEDIUM PRIORITY)

### Problem
```yaml
# File: .github/workflows/nightly-coverage.yml:37
- name: Update coverage badge
  run: ./scripts/update-coverage.ps1  # PowerShell on Linux!
  
- name: Update history
  run: ./scripts/update-coverage-history.ps1  # PowerShell on Linux!
```

### Issues with PowerShell on Linux
- Uses Windows-specific cmdlets (`Get-Command`, `Get-Content`, `Add-Content`, `Test-Path`)
- May not work reliably with PowerShell Core on Ubuntu
- Adds unnecessary dependency

### Recommended Fix: Convert to Bash

Create `scripts/update-coverage.sh`:
```bash
#!/bin/bash
set -euo pipefail

# Update coverage badge
COVERAGE_PERCENT=$(grep -oP 'line-rate="\K[0-9.]+' cobertura.xml | awk '{print int($1 * 100)}')

# Generate badge SVG
BADGE_COLOR="red"
if [ "$COVERAGE_PERCENT" -ge 90 ]; then
    BADGE_COLOR="brightgreen"
elif [ "$COVERAGE_PERCENT" -ge 75 ]; then
    BADGE_COLOR="yellow"
elif [ "$COVERAGE_PERCENT" -ge 60 ]; then
    BADGE_COLOR="orange"
fi

# Update badge using shields.io or similar
echo "Coverage: ${COVERAGE_PERCENT}% - Color: ${BADGE_COLOR}"

# Update README badge link if needed
if [ -f README.md ]; then
    sed -i "s/coverage-[0-9]*%/coverage-${COVERAGE_PERCENT}%/g" README.md
fi
```

Create `scripts/update-coverage-history.sh`:
```bash
#!/bin/bash
set -euo pipefail

# Extract coverage percentage
COVERAGE_PERCENT=$(grep -oP 'line-rate="\K[0-9.]+' cobertura.xml | awk '{print int($1 * 100)}')
TIMESTAMP=$(date -Iseconds)
COMMIT_SHA=$(git rev-parse --short HEAD)

# Append to history file
HISTORY_FILE="coverage_history.json"
if [ ! -f "$HISTORY_FILE" ]; then
    echo "[]" > "$HISTORY_FILE"
fi

# Add new entry
jq --arg timestamp "$TIMESTAMP" \
   --arg commit "$COMMIT_SHA" \
   --argjson coverage "$COVERAGE_PERCENT" \
   '. += [{"timestamp": $timestamp, "commit": $commit, "coverage": $coverage}]' \
   "$HISTORY_FILE" > "${HISTORY_FILE}.tmp"

mv "${HISTORY_FILE}.tmp" "$HISTORY_FILE"

echo "Added coverage entry: ${COVERAGE_PERCENT}% at ${TIMESTAMP}"
```

Update workflow:
```yaml
- name: Update coverage badge
  run: |
    chmod +x ./scripts/update-coverage.sh
    ./scripts/update-coverage.sh
  
- name: Update history
  run: |
    chmod +x ./scripts/update-coverage-history.sh
    ./scripts/update-coverage-history.sh
```

## Issue 5: Non-Constant-Time Operation in Test (LOW PRIORITY)

### Problem
```rust
// File: nexuszero-crypto/src/proof/proof.rs:1058
let public_value = gen_big.modpow(&secret_big, &mod_big).to_bytes_be();
```

### Fix
```rust
let public_value = ct_modpow(&gen_big, &secret_big, &mod_big).to_bytes_be();
```

This maintains consistency with the rest of the codebase which emphasizes constant-time operations.

## Issue 6: Expensive size() Method (MEDIUM PRIORITY)

### Problem
```rust
// File: nexuszero-crypto/src/proof/proof.rs:88
pub fn size(&self) -> usize {
    self.to_bytes().len()  // Serializes on every call!
}
```

### Fix Option 1: Cache on Creation
```rust
pub struct Proof {
    // ... existing fields ...
    cached_size: Option<usize>,
}

impl Proof {
    pub fn size(&mut self) -> usize {
        if let Some(size) = self.cached_size {
            return size;
        }
        
        let size = self.to_bytes().len();
        self.cached_size = Some(size);
        size
    }
}
```

### Fix Option 2: Compute Without Serialization
```rust
pub fn size(&self) -> usize {
    // Compute size without actually serializing
    let mut size = 0;
    size += 32; // a: CompressedRistretto (32 bytes)
    size += 32; // s: Scalar (32 bytes)
    size += self.challenges.len() * 32; // Vec<Scalar>
    size += self.responses.len() * 32; // Vec<Scalar>
    size
}
```

## Issue 7: Unused Import (LOW PRIORITY)

### Problem
```python
# File: nexuszero-crypto/test_ffi.py:10
import os  # Not used
```

### Fix
```python
# Remove unused import
# import os
```

## Testing Requirements

### Test 1: NTT Round-Trip Test
```rust
#[test]
fn test_ntt_round_trip() {
    let n = 256;
    let q = 7340033; // Prime, q ≡ 1 (mod 2n)
    let omega = compute_primitive_root(n, q);
    
    let mut data: Vec<u64> = (0..n).map(|i| (i * 123 + 456) % q).collect();
    let original = data.clone();
    
    ntt_forward(&mut data, n, q, omega);
    ntt_inverse(&mut data, n, q, omega);
    
    assert_eq!(data, original, "NTT round-trip should recover original data");
}
```

### Test 2: Coverage Threshold Test
```yaml
# Add to CI workflow
- name: Check coverage threshold
  run: |
    coverage=$(cargo tarpaulin --output-format Json | jq '.files | map(.coverage) | add / length')
    if (( $(echo "$coverage < 90" | bc -l) )); then
      echo "Coverage ${coverage}% is below 90% threshold"
      exit 1
    fi
```

## Action Items

### Critical (Must Fix)
1. ✅ Add rayon dependency to Cargo.toml
2. ✅ Fix NTT omega_pow calculation
3. ✅ Add NTT correctness tests

### High Priority (Should Fix)
4. ✅ Remove or properly implement SIMD functions
5. ✅ Convert PowerShell scripts to bash

### Medium Priority (Recommended)
6. ✅ Fix expensive size() method
7. ✅ Use ct_modpow in test for consistency

### Low Priority (Nice to Have)
8. ✅ Remove unused import

## Verification Steps

```bash
# 1. Add rayon dependency
cd nexuszero-crypto
# Edit Cargo.toml
cargo check

# 2. Fix NTT implementation
# Edit src/lattice/ring_lwe.rs
cargo test ring_lwe

# 3. Remove fake SIMD
# Edit src/lattice/ring_lwe.rs
cargo test --features simd

# 4. Convert scripts
cd scripts
# Create .sh versions
chmod +x *.sh
./update-coverage.sh
./update-coverage-history.sh

# 5. Run full test suite
cd ..
cargo test --workspace
cargo clippy --workspace --all-targets
```

## Status

🔴 **BLOCKED - DO NOT MERGE**

Critical algorithmic errors must be fixed before merging.

## Estimated Effort

- Add rayon dependency: 15 minutes
- Fix NTT omega calculation: 2-3 hours
- Remove/fix SIMD: 1-2 hours
- Convert PowerShell to bash: 1-2 hours
- Fix other issues: 1-2 hours
- Testing and validation: 2-3 hours
- **Total: 7-13 hours**
