# Benchmark Report
Generated: 2025-12-09T20:54:14.269091 UTC

## Environment
- OS: Windows 11 AMD64
- rustc: rustc 1.89.0 (29483883e 2025-08-04)
- cargo: cargo 1.89.0 (c24e10642 2025-06-23)

## Benchmarks (compared to baseline)
| Benchmark | Baseline (us) | Current (us) | Change (%) |
|---|---:|---:|---:|
| lwe_decrypt_128bit | 33.458 | 27.795 | -16.93% |
| lwe_encrypt_128bit | 513.053 | 373.039 | -27.29% |
| prove_range_8bits | 6490.207 | 8523.942 | 31.34% |
| verify_range_8bits | 3.389 | 3.028 | -10.65% |