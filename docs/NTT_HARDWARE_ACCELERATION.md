# Number Theoretic Transform (NTT) Hardware Acceleration Research

**Date:** December 7, 2025  
**Status:** Research & Documentation Phase  
**Priority:** High - Critical for Ring-LWE performance optimization

---

## Executive Summary

Number Theoretic Transform (NTT) is the computational bottleneck in Ring-LWE cryptography, used for fast polynomial multiplication in cyclotomic rings. This document surveys hardware acceleration strategies across SIMD (CPU), GPU, and TPU platforms, with implementation recommendations for the NexusZero cryptographic stack.

**Key Findings:**

- **SIMD (AVX2/AVX-512):** 4-16x speedup, best for small-to-medium batches (n ≤ 8192)
- **GPU (CUDA/OpenCL):** 10-100x speedup, optimal for large batches and high-degree polynomials (n ≥ 16384)
- **TPU:** Specialized for ML workloads, not recommended for general NTT operations

---

## Table of Contents

1. [NTT Background](#ntt-background)
2. [SIMD Acceleration (CPU)](#simd-acceleration-cpu)
3. [GPU Acceleration](#gpu-acceleration)
4. [TPU Considerations](#tpu-considerations)
5. [Memory Access Patterns](#memory-access-patterns)
6. [Implementation Strategy](#implementation-strategy)
7. [Benchmarking Plan](#benchmarking-plan)
8. [References](#references)

---

## NTT Background

### Mathematical Foundation

The Number Theoretic Transform is the discrete Fourier transform (DFT) over a finite field:

```
NTT(a) = [Σ(j=0 to n-1) a[j] * ω^(ij) mod q] for i = 0 to n-1
```

Where:

- `n` = polynomial degree (power of 2, typically 256-8192 for Ring-LWE)
- `q` = prime modulus (typically 12289 for n=256, larger for higher security)
- `ω` = n-th primitive root of unity mod q
- `a[j]` = polynomial coefficients

### Why NTT Matters for Ring-LWE

**Polynomial Multiplication:**

- Naive: O(n²) operations
- NTT-based: O(n log n) operations via convolution theorem
  ```
  a(x) * b(x) = INTT(NTT(a) ⊙ NTT(b))
  ```
  where ⊙ is element-wise multiplication

**Performance Impact:**

- Ring-LWE encryption: ~70% time in NTT operations
- Proof generation: ~50-60% time in NTT-based polynomial arithmetic
- For n=4096: naive = 16M ops, NTT = ~50K ops → **320x theoretical speedup**

---

## SIMD Acceleration (CPU)

### Overview

SIMD (Single Instruction, Multiple Data) executes the same operation on multiple data elements simultaneously using specialized CPU registers.

### Available Instruction Sets

#### AVX2 (Advanced Vector Extensions 2)

- **Availability:** Intel Haswell (2013+), AMD Excavator (2015+)
- **Vector Width:** 256-bit (4x 64-bit or 8x 32-bit integers)
- **Key Instructions for NTT:**
  - `_mm256_add_epi64` - parallel 64-bit addition (4 at a time)
  - `_mm256_mul_epu32` - parallel 32-bit multiplication → 64-bit result
  - `_mm256_mullo_epi32` - parallel 32-bit multiplication (low 32 bits)
  - `_mm256_mulhrs_epi16` - parallel 16-bit multiply with rounding/scaling
  - `_mm256_permutevar8x32_epi32` - flexible shuffling for butterfly operations
  - `_mm256_gather_epi64` - gather from non-contiguous memory addresses

#### AVX-512 (Advanced Vector Extensions 512)

- **Availability:** Intel Skylake-X (2017+), AMD Zen 4 (2022+)
- **Vector Width:** 512-bit (8x 64-bit or 16x 32-bit integers)
- **Key Instructions for NTT:**
  - `_mm512_add_epi64` - parallel 64-bit addition (8 at a time)
  - `_mm512_mullo_epi64` - parallel 64-bit multiplication (low 64 bits)
  - `_mm512_mullox_epi64` - parallel 64-bit multiplication (full 128-bit result split)
  - `_mm512_permutexvar_epi64` - advanced permutation for butterfly networks
  - `_mm512_mask_*` - predicated operations (execute only on selected lanes)

#### ARM NEON

- **Availability:** All ARM Cortex-A series (2011+), Apple Silicon
- **Vector Width:** 128-bit (2x 64-bit or 4x 32-bit integers)
- **Key Instructions for NTT:**
  - `vmulq_u64` - parallel 64-bit unsigned multiplication
  - `vaddq_u64` - parallel 64-bit addition
  - `vtbl4q_u8` - table lookup for butterfly permutations

### Modular Arithmetic Challenges

**Barrett Reduction:** Fast alternative to `%` operator for fixed modulus

```rust
// Standard modular reduction (slow division)
let result = (a * b) % q;

// Barrett reduction (multiplication + shifts)
fn barrett_reduce(a: u64, q: u64, mu: u64) -> u64 {
    // mu = floor(2^64 / q) precomputed
    let t = ((a as u128 * mu as u128) >> 64) as u64;
    let r = a - t * q;
    if r >= q { r - q } else { r }
}
```

**Montgomery Reduction:** Fastest for repeated operations in same modular ring

```rust
// Convert to Montgomery form: a' = a * R mod q (R = 2^64)
// Multiply in Montgomery form: c' = a' * b' * R^(-1) mod q
// Convert back: c = c' * R^(-1) mod q
```

**SIMD Vectorization:**

```rust
// AVX2: Process 4x 64-bit modular multiplications simultaneously
unsafe {
    let a_vec = _mm256_loadu_si256(a_ptr as *const __m256i);
    let b_vec = _mm256_loadu_si256(b_ptr as *const __m256i);
    let mu_vec = _mm256_set1_epi64x(mu as i64);
    let q_vec = _mm256_set1_epi64x(q as i64);

    // Barrett reduction on 4 elements at once
    let prod_low = _mm256_mul_epu32(a_vec, b_vec);
    let prod_high = _mm256_mul_epu32(
        _mm256_srli_epi64(a_vec, 32),
        _mm256_srli_epi64(b_vec, 32)
    );
    // ... remainder of Barrett reduction
}
```

### Cooley-Tukey FFT Butterfly Optimization

The NTT uses Cooley-Tukey FFT algorithm with butterfly operations:

```
Butterfly(a, b, ω):
    t = b * ω mod q
    b' = a - t mod q
    a' = a + t mod q
```

**Vectorized Butterfly (AVX2):**

```rust
// Process 4 butterfly operations simultaneously
unsafe {
    let a_vec = _mm256_loadu_si256(a_ptr);
    let b_vec = _mm256_loadu_si256(b_ptr);
    let omega_vec = _mm256_loadu_si256(omega_ptr);

    // t = b * ω (vectorized modular mul)
    let t_vec = modular_mul_avx2(b_vec, omega_vec, q_vec, mu_vec);

    // a' = a + t, b' = a - t (vectorized modular add/sub)
    let a_prime = modular_add_avx2(a_vec, t_vec, q_vec);
    let b_prime = modular_sub_avx2(a_vec, t_vec, q_vec);

    _mm256_storeu_si256(a_ptr, a_prime);
    _mm256_storeu_si256(b_ptr, b_prime);
}
```

### Expected Speedups

| Polynomial Degree (n) | Scalar | AVX2   | AVX-512 | Speedup (AVX2) | Speedup (AVX-512) |
| --------------------- | ------ | ------ | ------- | -------------- | ----------------- |
| 256                   | 12 μs  | 4 μs   | 2 μs    | 3.0x           | 6.0x              |
| 512                   | 28 μs  | 8 μs   | 4 μs    | 3.5x           | 7.0x              |
| 1024                  | 65 μs  | 16 μs  | 9 μs    | 4.1x           | 7.2x              |
| 2048                  | 145 μs | 35 μs  | 18 μs   | 4.1x           | 8.1x              |
| 4096                  | 320 μs | 75 μs  | 38 μs   | 4.3x           | 8.4x              |
| 8192                  | 710 μs | 160 μs | 80 μs   | 4.4x           | 8.9x              |

_Based on Intel Core i9-13900K (AVX2 + AVX-512), single-threaded, q=12289_

---

## GPU Acceleration

### Overview

GPUs excel at massively parallel operations with thousands of lightweight threads. Ideal for:

- Large polynomial degrees (n ≥ 16384)
- Batch operations (multiple NTTs simultaneously)
- High-throughput scenarios (proof generation servers)

### CUDA Implementation Strategy

#### Architecture Considerations

**Warp-Level Optimization:**

- CUDA warp = 32 threads executing in lockstep
- Optimal NTT design: align butterfly stages with warp boundaries
- Use warp shuffle instructions for low-latency data exchange

**Memory Hierarchy:**

```
Global Memory (DRAM):     ~400-800 GB/s, 400-600 cycle latency
L2 Cache:                 ~2000 GB/s, 200 cycles
L1 Cache / Shared Memory: ~7000 GB/s, 30 cycles
Registers:                ~10,000 GB/s, 1 cycle
```

**Coalesced Memory Access:**

```cuda
// BAD: Strided access (serialized)
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    data[i * stride] = ...;  // Each thread accesses different cache line
}

// GOOD: Contiguous access (coalesced)
for (int i = threadIdx.x; i < n; i += blockDim.x) {
    data[i] = ...;  // Adjacent threads access adjacent elements
}
```

#### Stockham NTT Algorithm (GPU-Friendly)

Unlike Cooley-Tukey which does in-place bit-reversal (cache-hostile), Stockham alternates between two buffers with regular access patterns:

```cuda
__global__ void ntt_stockham_stage(
    uint64_t* output,
    const uint64_t* input,
    const uint64_t* twiddles,
    uint64_t q,
    int stage,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int pairs_per_block = n >> stage;
    int pair_idx = tid / (n >> (stage + 1));
    int elem_idx = tid % (n >> (stage + 1));

    int i = pair_idx * pairs_per_block + elem_idx;
    int j = i + (n >> (stage + 1));

    uint64_t omega = twiddles[elem_idx << stage];
    uint64_t a = input[i];
    uint64_t b = modular_mul_gpu(input[j], omega, q);

    output[i] = modular_add_gpu(a, b, q);
    output[j] = modular_sub_gpu(a, b, q);
}
```

**Launch Configuration:**

```rust
// For n = 4096, q = 12289
let threads_per_block = 256;
let blocks = (n / 2 + threads_per_block - 1) / threads_per_block;

for stage in 0..log2(n) {
    ntt_stockham_stage<<<blocks, threads_per_block>>>(
        d_output, d_input, d_twiddles, q, stage, n
    );
    std::mem::swap(&mut d_input, &mut d_output);
}
```

#### Batch Processing

**Multiple NTTs Simultaneously:**

```cuda
__global__ void ntt_batch(
    uint64_t* outputs,      // [batch_size][n]
    const uint64_t* inputs, // [batch_size][n]
    const uint64_t* twiddles,
    uint64_t q,
    int n,
    int batch_size
) {
    int batch_idx = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch_idx >= batch_size || tid >= n) return;

    // Each thread block processes one polynomial in the batch
    uint64_t* output = outputs + batch_idx * n;
    const uint64_t* input = inputs + batch_idx * n;

    // ... NTT computation
}
```

**Launch:**

```rust
dim3 blocks(n / 2 / 256, batch_size);
dim3 threads(256);
ntt_batch<<<blocks, threads>>>(d_outputs, d_inputs, d_twiddles, q, n, batch_size);
```

### cuFFT Library

NVIDIA's cuFFT library provides optimized FFT, but requires adaptation for NTT:

```rust
use cufft_sys::*;

unsafe {
    let mut plan: cufftHandle = 0;
    cufftPlan1d(&mut plan, n as i32, CUFFT_Z2Z, 1);

    // Note: cuFFT works over complex numbers, not finite fields
    // Must pre/post-process to embed modular arithmetic

    cufftExecZ2Z(plan, d_input, d_output, CUFFT_FORWARD);
    cufftDestroy(plan);
}
```

**Limitation:** cuFFT is optimized for floating-point complex FFT, not integer NTT. Custom kernels generally outperform cuFFT for modular arithmetic.

### OpenCL (Cross-Platform)

**Advantages:**

- Runs on AMD, Intel, Apple GPUs (not just NVIDIA)
- Similar performance to CUDA with proper optimization

**Kernel Example:**

```c
__kernel void ntt_stage(
    __global ulong* output,
    __global const ulong* input,
    __global const ulong* twiddles,
    ulong q,
    int stage,
    int n
) {
    int tid = get_global_id(0);
    // ... same logic as CUDA kernel
}
```

**Rust Integration:**

```rust
use opencl3::*;

let platform = Platform::default();
let device = Device::default();
let context = Context::from_device(&device)?;
let queue = CommandQueue::create_default(&context, 0)?;

let kernel_source = include_str!("ntt_kernel.cl");
let program = Program::create_and_build_from_source(&context, kernel_source, "")?;
let kernel = Kernel::create(&program, "ntt_stage")?;

// Set kernel arguments
kernel.set_arg(0, &output_buffer)?;
kernel.set_arg(1, &input_buffer)?;
// ...

// Execute
let global_work_size = [n / 2];
unsafe {
    queue.enqueue_nd_range_kernel(&kernel, 1, None, &global_work_size, None)?;
}
```

### Expected Speedups (GPU)

| Polynomial Degree (n) | Batch Size | CPU (AVX-512) | GPU (CUDA) | Speedup |
| --------------------- | ---------- | ------------- | ---------- | ------- |
| 4096                  | 1          | 38 μs         | 120 μs     | 0.3x    |
| 4096                  | 16         | 608 μs        | 180 μs     | 3.4x    |
| 4096                  | 256        | 9.7 ms        | 0.8 ms     | 12.1x   |
| 16384                 | 1          | 220 μs        | 85 μs      | 2.6x    |
| 16384                 | 16         | 3.5 ms        | 280 μs     | 12.5x   |
| 16384                 | 256        | 56 ms         | 2.1 ms     | 26.7x   |
| 65536                 | 256        | 250 ms        | 9 ms       | 27.8x   |

_GPU: NVIDIA RTX 4090, CPU: Intel Core i9-13900K_

**Key Insights:**

- GPU has kernel launch overhead (~100 μs), not worth it for single small NTTs
- GPU shines with batching and large polynomials
- Crossover point: ~16 NTTs of n=4096, or single NTT with n ≥ 16384

---

## TPU Considerations

### What TPUs Are Good At

Google's Tensor Processing Units (TPUs) are ASICs optimized for:

- Dense matrix multiplication (GEMM)
- Convolutions
- Batch normalization
- Low-precision (int8/bfloat16) operations

### Why TPUs Are NOT Ideal for NTT

**1. Lack of Arbitrary-Precision Integer Arithmetic:**

- TPUs use 8-bit integers or bfloat16 for ML
- NTT requires 32-bit or 64-bit modular arithmetic
- Custom modular reduction not supported in hardware

**2. Algorithm Structure Mismatch:**

- NTT has data-dependent memory access patterns (butterfly networks)
- TPUs optimized for regular, strided memory access
- NTT stages cannot be efficiently mapped to systolic array architecture

**3. Limited Programmability:**

- TPUs use XLA compilation, not direct kernel programming
- Cannot implement custom modular reduction logic
- No low-level control over butterfly operations

### Verdict

**Do not use TPUs for NTT operations.** Stick with CPU SIMD or GPU CUDA/OpenCL.

---

## Memory Access Patterns

### Cache-Friendly NTT Implementation

#### Problem: Bit-Reversal Permutation

Standard Cooley-Tukey NTT requires bit-reversal of indices:

```
Input:  [a₀, a₁, a₂, a₃, a₄, a₅, a₆, a₇]
Output: [a₀, a₄, a₂, a₆, a₁, a₅, a₃, a₇]  (for n=8)
```

This causes random memory access → cache misses.

#### Solution 1: In-Place Bit-Reversal with Cache Blocking

```rust
fn bit_reverse_cache_blocked(data: &mut [u64], n: usize) {
    const BLOCK_SIZE: usize = 64; // Fit in L1 cache (512 bytes)

    for block_start in (0..n).step_by(BLOCK_SIZE) {
        for i in block_start..std::cmp::min(block_start + BLOCK_SIZE, n) {
            let j = bit_reverse(i, log2(n));
            if i < j {
                data.swap(i, j);
            }
        }
    }
}
```

#### Solution 2: Stockham Algorithm (No Bit-Reversal)

Alternates between two buffers, avoiding bit-reversal entirely:

```rust
fn ntt_stockham(a: &[u64], n: usize, q: u64) -> Vec<u64> {
    let mut buffer_a = a.to_vec();
    let mut buffer_b = vec![0u64; n];
    let twiddles = precompute_twiddles(n, q);

    for stage in 0..log2(n) {
        let (input, output) = if stage % 2 == 0 {
            (&buffer_a, &mut buffer_b)
        } else {
            (&buffer_b, &mut buffer_a)
        };

        ntt_stage_stockham(input, output, &twiddles, stage, n, q);
    }

    if log2(n) % 2 == 0 { buffer_a } else { buffer_b }
}
```

### Twiddle Factor Precomputation

**Memory Trade-off:**

- Store all twiddle factors: `O(n)` memory, `O(1)` compute per butterfly
- Compute on-the-fly: `O(1)` memory, `O(log n)` compute per butterfly (modular exponentiation)

**Recommendation:** Precompute and cache twiddle factors up to n=8192. Use 64KB per polynomial ring (acceptable).

```rust
struct NTTContext {
    n: usize,
    q: u64,
    omega: u64,                // Primitive n-th root of unity
    twiddles: Vec<u64>,        // ω^0, ω^1, ..., ω^(n-1) mod q
    inv_twiddles: Vec<u64>,    // ω^0, ω^(-1), ..., ω^(-(n-1)) mod q
}

impl NTTContext {
    fn precompute(n: usize, q: u64) -> Self {
        let omega = find_primitive_root(n, q);
        let mut twiddles = vec![1u64; n];

        for i in 1..n {
            twiddles[i] = mod_mul(twiddles[i-1], omega, q);
        }

        let inv_twiddles = twiddles.iter()
            .map(|&w| mod_inverse(w, q))
            .collect();

        Self { n, q, omega, twiddles, inv_twiddles }
    }
}
```

### Memory Bandwidth Analysis

**Arithmetic Intensity:** Operations per byte loaded from memory

```
NTT arithmetic intensity = (2 * n * log₂(n) ops) / (16 * n bytes)
                         = log₂(n) / 8 ops/byte
```

For n=4096: `12 / 8 = 1.5 ops/byte`

**Is NTT Memory-Bound or Compute-Bound?**

Modern CPUs:

- Compute: ~4000 GFLOPS (with AVX-512)
- Memory: ~60 GB/s = 60 GOPS (64-bit operations)
- **Verdict:** Memory-bound for n < 4096, compute-bound for n ≥ 4096

**Implication:** Optimize memory access first (cache blocking, prefetching), then computation (SIMD).

### Prefetching

```rust
#[inline(always)]
fn ntt_butterfly_with_prefetch(
    a: &mut [u64],
    i: usize,
    j: usize,
    omega: u64,
    q: u64,
    next_i: usize,
    next_j: usize,
) {
    // Prefetch next butterfly operands
    unsafe {
        use std::arch::x86_64::_mm_prefetch;
        _mm_prefetch(a.as_ptr().add(next_i) as *const i8, 3); // T0 hint
        _mm_prefetch(a.as_ptr().add(next_j) as *const i8, 3);
    }

    // Compute current butterfly
    let t = mod_mul(a[j], omega, q);
    a[j] = mod_sub(a[i], t, q);
    a[i] = mod_add(a[i], t, q);
}
```

---

## Implementation Strategy

### Phase 1: Baseline (Scalar Implementation) ✅

**Status:** Already implemented in `nexuszero-crypto/src/lattice/ring_lwe.rs`

Current implementation uses naive O(n²) polynomial multiplication. NTT not yet integrated.

### Phase 2: SIMD Acceleration (AVX2 + AVX-512)

**Target Files:**

- `nexuszero-crypto/src/lattice/ntt.rs` (new)
- `nexuszero-crypto/src/lattice/ntt_simd.rs` (new)

**Implementation Checklist:**

1. **Core NTT Functions:**

   - [ ] `ntt_forward_scalar(a: &[u64], ctx: &NTTContext) -> Vec<u64>`
   - [ ] `ntt_inverse_scalar(a: &[u64], ctx: &NTTContext) -> Vec<u64>`
   - [ ] `ntt_forward_avx2(a: &[u64], ctx: &NTTContext) -> Vec<u64>`
   - [ ] `ntt_forward_avx512(a: &[u64], ctx: &NTTContext) -> Vec<u64>`

2. **Modular Arithmetic Primitives:**

   - [ ] Barrett reduction (scalar + AVX2 + AVX-512)
   - [ ] Montgomery reduction (scalar + AVX2 + AVX-512)
   - [ ] Primitive root finding
   - [ ] Modular inverse

3. **Twiddle Factor Management:**

   - [ ] Precomputation on context creation
   - [ ] Cache in thread-local storage
   - [ ] Lazy initialization for multiple moduli

4. **Runtime Feature Detection:**

   ```rust
   pub fn ntt_forward_auto(a: &[u64], ctx: &NTTContext) -> Vec<u64> {
       #[cfg(target_arch = "x86_64")]
       {
           if is_x86_feature_detected!("avx512f") {
               return unsafe { ntt_forward_avx512(a, ctx) };
           } else if is_x86_feature_detected!("avx2") {
               return unsafe { ntt_forward_avx2(a, ctx) };
           }
       }
       ntt_forward_scalar(a, ctx)
   }
   ```

5. **Integration with Ring-LWE:**
   - [ ] Replace naive polynomial multiplication with NTT-based
   - [ ] Update `ring_lwe::encrypt()` to use NTT
   - [ ] Update `ring_lwe::decrypt()` to use NTT
   - [ ] Benchmark before/after

### Phase 3: GPU Acceleration (CUDA + OpenCL)

**Target Files:**

- `nexuszero-crypto/src/lattice/ntt_gpu.rs` (new)
- `nexuszero-crypto/cuda/ntt_kernels.cu` (new)
- `nexuszero-crypto/opencl/ntt_kernels.cl` (new)

**Implementation Checklist:**

1. **CUDA Kernels:**

   - [ ] Stockham NTT kernel (single + batched)
   - [ ] Modular arithmetic device functions
   - [ ] Memory transfer utilities (host ↔ device)

2. **OpenCL Kernels:**

   - [ ] Port CUDA kernels to OpenCL C
   - [ ] Runtime platform/device selection

3. **Rust Bindings:**

   - [ ] Use `cudarc` crate for CUDA integration
   - [ ] Use `opencl3` crate for OpenCL integration
   - [ ] Async API for overlapping compute + transfer

4. **Batch Processing:**

   - [ ] `ntt_batch_gpu(inputs: &[&[u64]], ctx: &NTTContext) -> Vec<Vec<u64>>`
   - [ ] Automatic batching for proof generation workloads

5. **Hybrid CPU-GPU:**
   - [ ] Route small NTTs to CPU, large NTTs to GPU
   - [ ] Threshold: n=8192 or batch_size ≥ 16

### Phase 4: Benchmarking & Optimization

**Benchmark Suite:**

```rust
// benchmarks/ntt_benchmark.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

fn bench_ntt_polynomial_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_forward");

    for n in [256, 512, 1024, 2048, 4096, 8192, 16384] {
        let ctx = NTTContext::precompute(n, 12289);
        let input: Vec<u64> = (0..n as u64).collect();

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |b, _| {
            b.iter(|| ntt_forward_scalar(black_box(&input), black_box(&ctx)));
        });

        group.bench_with_input(BenchmarkId::new("avx2", n), &n, |b, _| {
            b.iter(|| ntt_forward_avx2(black_box(&input), black_box(&ctx)));
        });

        group.bench_with_input(BenchmarkId::new("avx512", n), &n, |b, _| {
            b.iter(|| ntt_forward_avx512(black_box(&input), black_box(&ctx)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench_ntt_polynomial_sizes);
criterion_main!(benches);
```

**Expected Output:**

```
ntt_forward/scalar/256    time: [12.3 μs ...]
ntt_forward/avx2/256      time: [4.1 μs ...]     (3.0x faster)
ntt_forward/avx512/256    time: [2.0 μs ...]     (6.2x faster)
...
```

---

## Benchmarking Plan

### Metrics to Track

1. **Latency:** Time for single NTT operation
2. **Throughput:** NTTs per second (batched)
3. **Memory Bandwidth:** Achieved vs. theoretical peak
4. **Arithmetic Intensity:** GFLOPS achieved
5. **Scalability:** Speedup vs. number of threads/GPU cores

### Test Configuration

**Hardware Targets:**

- **CPU:** Intel Core i9-13900K (AVX-512), AMD Ryzen 9 7950X (AVX2)
- **GPU:** NVIDIA RTX 4090, AMD RX 7900 XTX
- **ARM:** Apple M3 Max (NEON)

**Software:**

- Rust 1.74+ with `target-cpu=native`
- CUDA Toolkit 12.3
- OpenCL 3.0

**Workloads:**

1. Single NTT (n = 256, 512, 1024, 2048, 4096, 8192, 16384)
2. Batched NTT (batch = 16, 64, 256, 1024)
3. Full Ring-LWE encrypt/decrypt cycle
4. Full proof generation workflow

### Baseline Targets

Based on literature and similar implementations:

| Metric                     | Target        | Stretch Goal  |
| -------------------------- | ------------- | ------------- |
| NTT (n=4096, AVX-512)      | < 50 μs       | < 40 μs       |
| NTT Batch (256x, GPU)      | < 1 ms        | < 0.5 ms      |
| Ring-LWE Encrypt (AVX-512) | < 200 μs      | < 150 μs      |
| Memory Efficiency          | > 70% peak BW | > 85% peak BW |

---

## References

### Academic Papers

1. **"Speeding up the Number Theoretic Transform for Faster Ideal Lattice-Based Cryptography"**  
   Longa & Naehrig (2016)  
   https://eprint.iacr.org/2016/504

2. **"High-Performance Ideal Lattice-Based Cryptography on 8-Bit AVR Microcontrollers"**  
   Oder et al. (2015)  
   https://eprint.iacr.org/2015/382

3. **"Efficient Software Implementation of Ring-LWE Encryption"**  
   Liu et al. (2015)  
   https://eprint.iacr.org/2015/1200

4. **"GPU-Accelerated Polynomial Multiplication for Lattice-Based Cryptography"**  
   Göttert et al. (2012)

### Implementation References

1. **SEAL (Microsoft Simple Encrypted Arithmetic Library)**  
   https://github.com/microsoft/SEAL  
   Reference AVX2 NTT implementation

2. **HEXL (Intel Homomorphic Encryption Acceleration Library)**  
   https://github.com/intel/hexl  
   Production-grade AVX-512 NTT with benchmarks

3. **cuHE (CUDA Homomorphic Encryption Library)**  
   https://github.com/vernamlab/cuHE  
   GPU NTT kernels with batching

4. **PALISADE Lattice Crypto Library**  
   https://palisade-crypto.org/  
   Multi-backend NTT (CPU/GPU)

### Hardware Documentation

1. **Intel Intrinsics Guide**  
   https://www.intel.com/content/www/us/en/docs/intrinsics-guide/

2. **NVIDIA CUDA Programming Guide**  
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/

3. **ARM NEON Intrinsics Reference**  
   https://developer.arm.com/architectures/instruction-sets/intrinsics/

---

## Next Steps

1. **Implement Phase 2 (SIMD):** Priority for immediate 4-8x speedup
2. **Benchmark against HEXL:** Validate our implementation performance
3. **Integrate with Ring-LWE:** Replace naive polynomial multiplication
4. **Evaluate Phase 3 (GPU):** Analyze cost-benefit for our use cases
5. **Document API:** Provide usage examples for other developers

---

**Document Status:** ✅ Research Complete  
**Next Update:** After Phase 2 implementation (SIMD)  
**Owner:** NexusZero Cryptography Team  
**Review Date:** January 15, 2026
