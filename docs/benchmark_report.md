# Benchmark Report
Generated: 2025-12-09T15:43:48.526098 UTC

## Environment
- OS: Windows 11 AMD64
- rustc: rustc 1.89.0 (29483883e 2025-08-04)
- cargo: cargo 1.89.0 (c24e10642 2025-06-23)

## Benchmarks (compared to baseline)
| Benchmark | Baseline (us) | Current (us) | Change (%) |
|---|---:|---:|---:|
| lwe_decrypt_128bit | 33.458 | 38.313 | 14.51% |
| lwe_encrypt_128bit | 513.053 | 483.990 | -5.66% |
| prove_range_8bits | 6490.207 | 10256.735 | 58.03% |
| verify_range_8bits | 3.389 | 3.902 | 15.12% |