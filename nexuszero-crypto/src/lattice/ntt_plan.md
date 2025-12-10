# NTT Implementation Plan - Quick Reference

## Overview

Implement high-performance Number Theoretic Transform (NTT) for Ring-LWE polynomial multiplication with 4-16x speedup using CPU SIMD and optional GPU acceleration.

## File Structure

```
nexuszero-crypto/src/lattice/
├── ntt.rs              # Core NTT implementation (scalar baseline)
├── ntt_simd.rs         # SIMD optimizations (AVX2/AVX-512/NEON)
├── ntt_gpu.rs          # GPU kernels (CUDA/OpenCL) - optional
└── ring_lwe.rs         # Updated to use NTT for polynomial multiplication
```

## Phase 2: SIMD Implementation (Current Priority)

### Step 1: Core NTT Module (`ntt.rs`)

```rust
// nexuszero-crypto/src/lattice/ntt.rs

use num_bigint::BigUint;
use crate::CryptoResult;

/// NTT context with precomputed twiddle factors
pub struct NTTContext {
    pub n: usize,              // Polynomial degree (power of 2)
    pub q: u64,                // Prime modulus
    pub omega: u64,            // Primitive n-th root of unity
    pub psi: u64,              // 2n-th root of unity (for negacyclic)
    pub twiddles: Vec<u64>,    // Forward twiddle factors
    pub inv_twiddles: Vec<u64>,// Inverse twiddle factors
    pub inv_n: u64,            // n^(-1) mod q
}

impl NTTContext {
    /// Create NTT context for given parameters
    pub fn new(n: usize, q: u64) -> CryptoResult<Self> {
        // Verify n is power of 2
        // Find primitive root
        // Precompute twiddles
        // ...
    }
}

/// Forward NTT (scalar baseline)
pub fn ntt_forward(a: &[u64], ctx: &NTTContext) -> Vec<u64> {
    // Cooley-Tukey algorithm
    // ...
}

/// Inverse NTT (scalar baseline)
pub fn ntt_inverse(a: &[u64], ctx: &NTTContext) -> Vec<u64> {
    // ...
}

/// Polynomial multiplication via NTT
pub fn polymul_ntt(a: &[u64], b: &[u64], ctx: &NTTContext) -> Vec<u64> {
    let a_ntt = ntt_forward(a, ctx);
    let b_ntt = ntt_forward(b, ctx);
    let c_ntt = pointwise_mul(&a_ntt, &b_ntt, ctx.q);
    ntt_inverse(&c_ntt, ctx)
}
```

### Step 2: Modular Arithmetic Primitives

```rust
// Helper functions for modular arithmetic

#[inline(always)]
pub fn mod_add(a: u64, b: u64, q: u64) -> u64 {
    let sum = a + b;
    if sum >= q { sum - q } else { sum }
}

#[inline(always)]
pub fn mod_sub(a: u64, b: u64, q: u64) -> u64 {
    if a >= b { a - b } else { a + q - b }
}

/// Barrett reduction for fast modular multiplication
#[inline(always)]
pub fn mod_mul_barrett(a: u64, b: u64, q: u64, mu: u64) -> u64 {
    let prod = a as u128 * b as u128;
    let quot = ((prod * mu as u128) >> 64) as u64;
    let rem = (prod as u64).wrapping_sub(quot.wrapping_mul(q));
    if rem >= q { rem - q } else { rem }
}

/// Find primitive n-th root of unity modulo q
pub fn find_primitive_root(n: usize, q: u64) -> Option<u64> {
    // Trial and error or use known roots for standard parameters
    // For q=12289, n=256: ω=49
    // ...
}
```

### Step 3: SIMD Module (`ntt_simd.rs`)

```rust
// nexuszero-crypto/src/lattice/ntt_simd.rs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::ntt::{NTTContext};

/// AVX2 NTT implementation (4x 64-bit parallel)
#[cfg(target_feature = "avx2")]
#[target_feature(enable = "avx2")]
pub unsafe fn ntt_forward_avx2(a: &[u64], ctx: &NTTContext) -> Vec<u64> {
    let mut output = a.to_vec();
    let q_vec = _mm256_set1_epi64x(ctx.q as i64);

    // Process 4 butterflies at a time
    for stage in 0..ctx.n.trailing_zeros() {
        let m = 1 << stage;
        let stride = 2 * m;

        for i in (0..ctx.n).step_by(stride) {
            for j in (i..i+m).step_by(4) {
                // Load 4 elements
                let a_vec = _mm256_loadu_si256(output[j..].as_ptr() as *const __m256i);
                let b_vec = _mm256_loadu_si256(output[j+m..].as_ptr() as *const __m256i);
                let omega_vec = _mm256_loadu_si256(ctx.twiddles[...].as_ptr() as *const __m256i);

                // Butterfly: t = b * ω, a' = a + t, b' = a - t
                // (vectorized modular arithmetic)
                // ...
            }
        }
    }

    output
}

/// Runtime dispatch based on CPU features
pub fn ntt_forward_auto(a: &[u64], ctx: &NTTContext) -> Vec<u64> {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { ntt_forward_avx512(a, ctx) };
        } else if is_x86_feature_detected!("avx2") {
            return unsafe { ntt_forward_avx2(a, ctx) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { ntt_forward_neon(a, ctx) };
        }
    }

    super::ntt::ntt_forward(a, ctx)
}
```

### Step 4: Integration with Ring-LWE

```rust
// nexuszero-crypto/src/lattice/ring_lwe.rs

use super::ntt::{NTTContext, polymul_ntt};
use super::ntt_simd::ntt_forward_auto;

// Update RingLWECiphertext to store NTT context
pub struct RingLWECiphertext {
    pub c0: Vec<u64>,
    pub c1: Vec<u64>,
    pub params: RingLWEParameters,
    ntt_ctx: NTTContext,  // Add this
}

// Update encrypt function
pub fn encrypt(
    public_key: &RingLWEPublicKey,
    message: &[u64],
    params: &RingLWEParameters,
    rng: &mut impl RngCore,
) -> CryptoResult<RingLWECiphertext> {
    // Create NTT context
    let ntt_ctx = NTTContext::new(params.n, params.q)?;

    // Generate error polynomials e0, e1, e2
    // ...

    // Polynomial multiplications using NTT
    let u = polymul_ntt(&r, &public_key.a, &ntt_ctx);  // r * a
    let v = polymul_ntt(&r, &public_key.b, &ntt_ctx);  // r * b

    // c0 = u + e0, c1 = v + e1 + encode(m)
    // ...
}
```

### Step 5: Testing

```rust
// nexuszero-crypto/src/lattice/ntt.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_correctness() {
        let n = 256;
        let q = 12289;
        let ctx = NTTContext::new(n, q).unwrap();

        let input: Vec<u64> = (0..n as u64).map(|x| x % q).collect();
        let ntt_result = ntt_forward(&input, &ctx);
        let recovered = ntt_inverse(&ntt_result, &ctx);

        for i in 0..n {
            assert_eq!(input[i], recovered[i], "NTT roundtrip failed at index {}", i);
        }
    }

    #[test]
    fn test_polymul_ntt_vs_naive() {
        let n = 256;
        let q = 12289;
        let ctx = NTTContext::new(n, q).unwrap();

        let a: Vec<u64> = (0..n as u64).map(|x| x % q).collect();
        let b: Vec<u64> = (0..n as u64).map(|x| (x * 2) % q).collect();

        let result_ntt = polymul_ntt(&a, &b, &ctx);
        let result_naive = polymul_naive(&a, &b, q);  // Implement this for comparison

        for i in 0..n {
            assert_eq!(result_ntt[i], result_naive[i], "Polymul mismatch at index {}", i);
        }
    }

    #[test]
    fn test_simd_matches_scalar() {
        let n = 256;
        let q = 12289;
        let ctx = NTTContext::new(n, q).unwrap();

        let input: Vec<u64> = (0..n as u64).map(|x| x % q).collect();

        let scalar_result = ntt_forward(&input, &ctx);

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                let simd_result = unsafe { ntt_forward_avx2(&input, &ctx) };
                assert_eq!(scalar_result, simd_result, "AVX2 doesn't match scalar");
            }
        }
    }
}
```

### Step 6: Benchmarking

```rust
// benches/ntt_benchmark.rs

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nexuszero_crypto::lattice::ntt::*;
use nexuszero_crypto::lattice::ntt_simd::*;

fn bench_ntt_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_forward");

    for n in [256, 512, 1024, 2048, 4096] {
        let ctx = NTTContext::new(n, 12289).unwrap();
        let input: Vec<u64> = (0..n as u64).collect();

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| ntt_forward(black_box(&input), black_box(&ctx)));
        });

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            group.bench_with_input(BenchmarkId::new("avx2", n), &n, |b, _| {
                b.iter(|| unsafe { ntt_forward_avx2(black_box(&input), black_box(&ctx)) });
            });
        }

        group.bench_with_input(BenchmarkId::new("auto", n), &n, |b, _| {
            b.iter(|| ntt_forward_auto(black_box(&input), black_box(&ctx)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ntt_sizes);
criterion_main!(benches);
```

## Known Parameters for Quick Start

### Standard Ring-LWE Parameters with NTT Support

| Security Level    | n    | q     | ω (primitive root) |
| ----------------- | ---- | ----- | ------------------ |
| Toy (demo)        | 256  | 12289 | 49                 |
| Level 1 (128-bit) | 512  | 12289 | 49                 |
| Level 3 (192-bit) | 1024 | 40961 | 7                  |
| Level 5 (256-bit) | 2048 | 40961 | 7                  |

These are NTT-friendly primes: q ≡ 1 (mod 2n)

## Expected Performance Targets

| n    | Scalar | AVX2  | AVX-512 | Speedup (AVX2) |
| ---- | ------ | ----- | ------- | -------------- |
| 256  | 12 μs  | 4 μs  | 2 μs    | 3.0x           |
| 512  | 28 μs  | 8 μs  | 4 μs    | 3.5x           |
| 1024 | 65 μs  | 16 μs | 9 μs    | 4.1x           |
| 2048 | 145 μs | 35 μs | 18 μs   | 4.1x           |
| 4096 | 320 μs | 75 μs | 38 μs   | 4.3x           |

## Next Steps

1. ✅ Research complete (see `docs/NTT_HARDWARE_ACCELERATION.md`)
2. ⬜ Implement `ntt.rs` - scalar baseline
3. ⬜ Implement `ntt_simd.rs` - AVX2 optimization
4. ⬜ Add comprehensive tests
5. ⬜ Integrate with `ring_lwe.rs`
6. ⬜ Run benchmarks
7. ⬜ Document performance improvements

## References

- Full research document: `docs/NTT_HARDWARE_ACCELERATION.md`
- Intel HEXL: https://github.com/intel/hexl
- Microsoft SEAL: https://github.com/microsoft/SEAL
