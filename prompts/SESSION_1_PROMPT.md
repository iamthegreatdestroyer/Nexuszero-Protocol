# Copilot Prompt: Week 1 – Performance Benchmarking Report (Task 1.6)

You are assisting with performance benchmarking for the Rust crate `nexuszero-crypto`. Your goal is to run Criterion benchmarks and produce a concise benchmarking report with key metrics.

## Actions

1. Verify environment:
   - Run: `rustc --version` and `cargo --version`
   - Run: `cargo test -p nexuszero-crypto`
2. Run benchmarks:
   - Run: `cargo bench -p nexuszero-crypto --benches`
3. Collect results:
   - Locate Criterion reports under `nexuszero-crypto/target/criterion/`
   - Extract mean, std dev, and throughput for:
     - `lwe_encrypt_128bit`
     - `prove_discrete_log_128bit`
     - `verify_discrete_log_128bit`
4. Update reports:
   - Append a new dated section to `benchmark_output.txt` with:
     - System info (CPU model, cores/threads)
     - Rust/cargo versions
     - Commands executed
     - Key metrics summary
   - Update `nexuszero-crypto/BENCHMARK_RESULTS.md` with a table of key results and a short analysis.

## Format Requirements

- Include date/time in headers.
- Use code fences for commands and outputs.
- Keep the analysis to 4-6 sentences focusing on trends and targets vs actuals.

## Acceptance Criteria

- Criterion benchmarks run without errors.
- `benchmark_output.txt` updated with a clear, reproducible log.
- `nexuszero-crypto/BENCHMARK_RESULTS.md` updated with metrics and brief analysis.
- Metrics mapped against targets in `scripts/WEEK_1_CRYPTOGRAPHY_MODULE_PROMPTS.md` (Performance Targets section).
