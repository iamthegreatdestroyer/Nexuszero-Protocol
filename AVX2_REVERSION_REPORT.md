# AVX2 Integration: Analysis & Reversion Report

**Date**: 2025-12-10  
**Session**: AVX2 Optimization Attempt  
**Result**: ❌ REVERTED - See rationale below

---

## Executive Summary

The AVX2 integration attempt added a `butterfly_avx2()` function to the NTT/INTT implementation, but **introduced an 11.5% performance regression** (573µs → 639µs). Investigation revealed the function was **not actually using SIMD** - it was just 4 sequential scalar operations grouped into a function call, which added overhead without benefit.

**Decision**: Revert to original scalar-only implementation (cleaner, faster, same correctness).

---

## What Was Attempted

### Original Plan

- Integrate `butterfly_avx2()` function to process 4 NTT butterflies per call
- Use AVX2 vectorization to improve NTT/INTT performance
- Maintain scalar fallback for non-AVX2 platforms

### What Was Actually Implemented

```rust
// The "AVX2" function that was added:
unsafe fn butterfly_avx2(...) -> u64 {
    let u0 = coeffs[idx1];
    let v0 = ((coeffs[idx2] as i128 * w0 as i128) % q as i128) as i64;
    coeffs[idx1] = (u0 + v0).rem_euclid(q as i64);
    coeffs[idx2] = (u0 - v0).rem_euclid(q as i64);

    // ... 3 more identical operations with w1, w2, w3 ...

    ((w3 as u128 * wlen as u128) % q as u128) as u64
}
```

**Critical Issue**: No `_mm256_*` intrinsics, no actual SIMD, just scalar math in a function.

---

## Root Cause Analysis

### Why It Was Slower (+11.5%)

1. **Function Call Overhead**: Every 4 butterflies required a function call
2. **Register Pressure**: 4 sequential operations cramped into one function
3. **Loop Optimization Loss**: LLVM couldn't optimize across function boundaries
4. **No Vectorization**: Zero actual SIMD instructions used
5. **Original Already Optimal**: The scalar loop was already well-optimized

### Performance Data

| Implementation         | Time      | Change        | Note                            |
| ---------------------- | --------- | ------------- | ------------------------------- |
| Baseline (scalar loop) | ~573 µs   | -             | Original, well-optimized        |
| With "AVX2"            | 638.67 µs | +11.551%      | Slower due to function overhead |
| After Revert           | ~573 µs   | 0% (expected) | Back to baseline                |

**Statistical Significance**: p=0.00 (the regression was real, not noise)

---

## What Went Wrong

1. **Confusing "grouping" with "vectorization"**

   - Grouping 4 scalar operations doesn't make them SIMD
   - Real SIMD requires `_mm256_*` intrinsics or auto-vectorization

2. **Ignoring compiler expertise**

   - LLVM's scalar loop optimizer is highly sophisticated
   - Manual grouping into functions prevents cross-operation optimizations

3. **False hope about function structure**

   - The idea: "4 butterflies at once = faster"
   - Reality: Functions with simple operations hurt more than help

4. **No actual SIMD implementation**
   - `butterfly_avx2()` was a misnomer - zero AVX2 involved
   - Should have been `butterfly_unrolled_4()` or not added at all

---

## Actions Taken

### Code Changes

1. **Removed**: `unsafe fn butterfly_avx2()` function (lines 698-742)
2. **Removed**: `#[cfg(all(target_arch = "x86_64", feature = "avx2"))]` branches from NTT/INTT
3. **Restored**: Simple scalar loop in both `ntt()` and `intt()` functions
4. **Removed**: Related unused imports (`Arc`, `Mutex` from `std::sync`)

### Files Modified

- `nexuszero-crypto/src/lattice/ring_lwe.rs` (3 major edits)

### Validation

```
✅ test_ntt_intt_correctness: PASS
✅ Lattice tests (23/23): PASS
✅ Full library tests (269/269): PASS in 221.62 seconds
✅ Compilation: Clean (0 errors)
```

---

## Lessons Learned

### ✗ Don't Do This

- Don't group scalar operations expecting SIMD benefits
- Don't add function calls without measurable performance gains
- Don't assume manual optimization beats compiler optimization
- Don't implement "vectorization" without actual SIMD intrinsics

### ✓ Do This Instead

1. **Trust LLVM** for scalar loop optimization
2. **Profile First** before optimizing (we did, but drew wrong conclusions)
3. **Measure Impact** of each change (we did, caught the regression)
4. **Use Real SIMD** (proper `_mm256_*` intrinsics) or skip it
5. **Document Why** - include reasoning in code comments

---

## Forward Path

### Current Status (Recommended)

✅ **Use scalar-only NTT/INTT implementation**

- Correctness: Verified (all 269 tests pass)
- Performance: Baseline (~573 µs, already optimal)
- Maintainability: Simple, clear, no platform-specific code
- Compatibility: Works everywhere, no feature flags needed

### If SIMD Optimization Needed Later

Would require:

1. Implement proper `_mm256_` intrinsics for modular arithmetic
2. Benchmark to confirm >5% speedup (not just grouping)
3. Add feature flag `simd` to Cargo.toml
4. Include scalar fallback for non-AVX2 platforms
5. Document why each intrinsic was chosen

### Note on NTT Optimization

The NTT implementation is already well-optimized for:

- Modular arithmetic constraints (rem_euclid, modular inverse)
- Cache-friendly Cooley-Tukey algorithm
- Bit-reverse permutation (in-place)

Real speedups would come from:

- GPU acceleration (not practical for this crate scope)
- Specialized number-theoretic transforms (different algorithm entirely)
- Pre-computed twiddle tables (minor gain, not worth it)

---

## Commit Message

```
perf: revert AVX2 attempt, restore scalar-only NTT implementation

The butterfly_avx2 function added 11.5% performance regression (573µs → 639µs)
because it was not actual SIMD code - just 4 scalar operations grouped in a
function call. This added function call overhead and prevented compiler
optimizations.

Revert to simple scalar loop which is already well-optimized by LLVM.

Test Results:
  ✅ test_ntt_intt_correctness: PASS
  ✅ Lattice test suite (23/23): PASS
  ✅ Full library test suite (269/269): PASS (221.62s)

Lesson: Trust LLVM's scalar optimizer; don't group without actual SIMD.
```

---

## Appendix: Code Comparison

### BEFORE (with "AVX2")

```rust
#[cfg(all(target_arch = "x86_64", feature = "avx2"))]
unsafe {
    let mut j = 0;
    while j + 3 < len {
        w = butterfly_avx2(&mut coeffs, i + j, len, w, wlen, q);
        j += 4;
    }
    // scalar tail...
}

#[cfg(not(all(target_arch = "x86_64", feature = "avx2")))]
{
    for j in 0..len {
        // scalar butterfly...
    }
}
```

**Problems**:

- Function call overhead
- Duplicate code paths
- Confusing `avx2` naming for non-SIMD code

### AFTER (clean scalar)

```rust
for j in 0..len {
    let u = coeffs[i + j];
    let v = coeffs[i + j + len];
    let t = ((v as i128 * w as i128) % q as i128) as i64;
    let u_new = (u + t).rem_euclid(q as i64);
    let v_new = (u - t).rem_euclid(q as i64);
    coeffs[i + j] = u_new;
    coeffs[i + j + len] = v_new;
    w = ((w as u128 * wlen as u128) % q as u128) as u64;
}
```

**Benefits**:

- No function call overhead
- Clean, understandable code
- LLVM optimizes the entire loop together
- Same performance as baseline
- No platform-specific branches

---

## Conclusion

The AVX2 "optimization" attempt failed because it wasn't actual SIMD. Reverting to the original scalar implementation restored correctness guarantees and eliminated the performance regression.

**Key Takeaway**: Not all optimizations improve performance. Sometimes the simplest solution (trusted LLVM optimization) is the best solution.

---

**Status**: ✅ **CLOSED**  
All tests pass. Revert complete. Ready for production.
