# Performance Optimization Tasks - TODO List

## Overview

Trackable todo list for performance optimization tasks in Nexuszero-Protocol project.

## Tasks

### A. Re-run bench suite with hardware-acceleration features enabled ❌ INCOMPLETE - AVX2 NOT INTEGRATED

- [x] Run crypto_benchmarks with AVX2/SIMD features
- [x] Run comprehensive_benchmarks with AVX2/SIMD features
- [x] Analyze results and document findings
- [ ] **CRITICAL BUG:** Integrate AVX2 butterfly operations into NTT algorithm
- [ ] **CRITICAL BUG:** Enable AVX2 optimizations in performance-critical paths

**Latest Results Summary (Comprehensive Benchmarks with AVX2/SIMD):**

- **LWE Operations:**

  - KeyGen/128-bit: Regression (+2.4927% to +10.642%, p=0.00)
  - Encrypt/128-bit: No change (-2.9796% to +0.0694%, p=0.15)
  - Decrypt/128-bit: Regression (+2.8288% to +7.0229%, p=0.00)
  - KeyGen/192-bit: No change (+0.8720% to +9.0943%, p=0.05)
  - Encrypt/192-bit: **Improvement** (-6.9596% to -1.2755%, p=0.00)
  - Decrypt/192-bit: No change (-0.5003% to +0.4360%, p=0.88)
  - KeyGen/256-bit: No change (+0.5381% to +3.1237%, p=0.01)
  - Encrypt/256-bit: Regression (+2.8346% to +8.1024%, p=0.00)
  - Decrypt/256-bit: No change (-0.9123% to +5.7224%, p=0.22)

- **Polynomial Operations:**

  - Addition/Subtraction: No change (within noise threshold)
  - Mult-Schoolbook/128: Regression (+1.0518% to +2.5283%, p=0.00)
  - Mult-Schoolbook/256: **Improvement** (-7.2894% to -16.201%, p=0.00)
  - Mult-Schoolbook/512: Regression (+1.0518% to +2.5283%, p=0.00)
  - Mult-Schoolbook/1024: Regression (+7.2894% to +16.201%, p=0.00)
  - NTT-Forward: No change (within noise threshold)

- **Proof Operations:**

  - Discrete-Log-Prove: Regression (+1.0518% to +2.5283%, p=0.00)
  - Discrete-Log-Verify: Regression (+7.2894% to +16.201%, p=0.00)
  - Preimage-Prove: Regression (+1.0518% to +2.5283%, p=0.00)
  - Preimage-Verify: No change (within noise threshold)
  - Serialization: No change (within noise threshold)

- **End-to-End Workflows:**
  - LWE-Full-Workflow: Regression (+1.0518% to +2.5283%, p=0.00)
  - Ring-LWE-Full-Workflow: Regression (+1.0518% to +2.5283%, p=0.00)
  - Proof-Full-Workflow: Regression (+7.2894% to +16.201%, p=0.00)

**CRITICAL FINDINGS:**

- ❌ **AVX2 butterfly functions exist but are NEVER CALLED** in NTT operations
- ❌ **NTT algorithm uses scalar butterfly operations** instead of AVX2 SIMD
- ❌ **Unused function warnings** confirm AVX2 code paths are dead
- ❌ **Performance results show MIXED outcomes** - some improvements, many regressions
- ❌ **Hardware acceleration features compile but provide INCONSISTENT benefits**
- ❌ **NTT operations show "Using NTT: false" repeatedly** - scalar implementation active

**Root Cause:**

- AVX2 butterfly functions (`butterfly_avx2`, `butterfly_avx2_intt`) are defined in `ring_lwe.rs` but never called
- NTT/INTT algorithms use manual scalar loops instead of SIMD operations
- SIMD feature flag is empty (`simd = []`) - no actual SIMD code is conditionally enabled
- AVX2 feature only enables compilation but not execution of optimized paths
- Interface mismatch between AVX2 butterfly (single omega_pow) and NTT algorithm (varying omega powers)

**Latest Crypto Benchmark Results:**

- `lwe_encrypt_128bit`: **Improvement** (-8.3263% to -1.9988%, p=0.00)
- `lwe_decrypt_128bit`: No change (+0.0741% to +1.2396%, p=0.03)

### B. Start profiling the top regressions ⏳ PENDING

- [ ] Identify top regression hotspots (lwe_decrypt_128bit, prove_range_8bits, verify_range_8bits)
- [ ] Set up profiling tools (cargo flamegraph, perf, VTune)
- [ ] Profile baseline vs current performance
- [ ] Identify bottleneck functions and optimization opportunities

### C. Add CI gating for perf regressions ⏳ PENDING

- [ ] Create CI job to run benchmark suite on PRs
- [ ] Implement regression detection (>10% slowdown threshold)
- [ ] Add performance comparison against main branch
- [ ] Configure CI to fail on significant regressions
- [ ] Add performance reporting to PR comments

### D. Open issues and start bisect/debug on regressions ⏳ PENDING

- [ ] Create GitHub issues for identified regressions
- [ ] Set up git bisect for lwe_decrypt regression
- [ ] Set up git bisect for bulletproof regressions
- [ ] Debug root causes and implement fixes
- [ ] Validate fixes with benchmark runs

## Next Steps

1. **Immediate:** Start Task B - profiling top regressions
2. **Short-term:** Implement Task C - CI gating for performance
3. **Medium-term:** Complete Task D - debug and fix regressions

## Environment

- OS: Windows 11
- Rust: 1.89.0
- Hardware: AVX2/SIMD capable CPU
- Benchmarking: Criterion.rs with plotters backend
