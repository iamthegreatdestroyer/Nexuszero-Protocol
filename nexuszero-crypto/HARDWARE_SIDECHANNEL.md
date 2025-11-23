# Hardware-Backed Security & Side-Channel Testing

## Overview

This crate includes two advanced security frameworks:

1. **Hardware-Backed Security Integration** (`utils/hardware.rs`)
2. **Side-Channel Testing Framework** (`utils/sidechannel.rs`)

## Hardware-Backed Security

### Supported Backends

- **Intel SGX** (Software Guard Extensions): Secure enclaves with memory encryption
- **ARM TrustZone**: Secure/normal world isolation
- **HSM** (Hardware Security Module): PKCS#11 compatible devices
- **Software Fallback**: Constant-time implementations without hardware isolation

### Usage

```rust
use nexuszero_crypto::utils::hardware::{select_backend, HardwareBackend};

// Automatically select best available backend
let backend = select_backend();

// Perform secure modular exponentiation
let result = backend.secure_modpow(&base, &exp, &modulus)?;

// Generate hardware RNG
let random = backend.secure_random(32)?;

// Seal keys to hardware
backend.seal_key("my_key", &key_data)?;

// Generate attestation report
let report = backend.attest()?;
```

### Platform Requirements

**Intel SGX:**

- Intel CPU with SGX support (Skylake+)
- Linux kernel with SGX driver
- AESM service running

**ARM TrustZone:**

- ARM Cortex-A processor with TrustZone
- OP-TEE or proprietary TEE OS
- Kernel driver for secure world communication

**HSM:**

- Physical HSM device (YubiHSM, Nitrokey, etc.)
- PKCS#11 library installed
- Device path configuration

### Optional Dependencies

```toml
# For Intel SGX support
[dependencies]
sgx_tstd = { version = "1.1", optional = true }
sgx_types = { version = "1.1", optional = true }

# For HSM support
[dependencies]
pkcs11 = { version = "0.5", optional = true }
```

## Side-Channel Testing Framework

### Components

#### 1. Differential Timing Analysis (dudect)

Statistical test for constant-time properties using Welch's t-test:

```rust
use nexuszero_crypto::utils::sidechannel::test_constant_time;

let result = test_constant_time(
    |secret_bit| {
        let start = Instant::now();
        my_crypto_operation(secret_bit);
        start.elapsed()
    },
    1000, // number of samples
);

if !result.is_constant_time {
    println!("WARNING: Timing leak detected! t={:?}", result.t_statistic);
}
```

#### 2. Cache-Timing Attack Simulation

Simulates Flush+Reload and Prime+Probe attacks:

```rust
use nexuszero_crypto::utils::sidechannel::CacheSimulator;

let mut cache = CacheSimulator::new();

// Simulate victim operation
cache.access(secret_dependent_address);

// Attacker flushes
cache.flush(target_address);

// Analyze patterns
let analysis = cache.analyze_patterns();
if !analysis.suspicious_lines.is_empty() {
    println!("Potential cache-timing vulnerability detected");
}
```

#### 3. Memory Access Pattern Analysis

Tracks memory access regularity:

```rust
use nexuszero_crypto::utils::sidechannel::{MemoryTracer, AccessType};

let mut tracer = MemoryTracer::new();

tracer.record(address, size, AccessType::Read);
tracer.mark_secret_dependent(); // if access depends on secret

let regularity = tracer.analyze_regularity();
// regularity close to 1.0 = constant-time
```

### Running Side-Channel Tests

```bash
# Run all side-channel tests
cargo test --test hardware_sidechannel_tests

# Run specific test
cargo test --test hardware_sidechannel_tests test_dudect_constant_time

# With detailed output
cargo test --test hardware_sidechannel_tests -- --nocapture
```

### Integration with External Tools

#### ChipWhisperer

For power/EM analysis:

```bash
# Install ChipWhisperer
pip install chipwhisperer

# Run capture and analysis (requires hardware)
python scripts/chipwhisperer_analysis.py
```

#### Intel PT (Processor Trace)

For execution flow analysis:

```bash
# Enable Intel PT
echo 1 | sudo tee /sys/kernel/debug/tracing/events/intel_pt/enable

# Capture trace during crypto operation
perf record -e intel_pt//u cargo test

# Analyze trace
perf script --itrace=i1t -F time,ip,sym
```

#### dudect CLI

External dudect tool for comprehensive testing:

```bash
# Install dudect
cargo install dudect-bencher

# Run dudect benchmark
dudect-bencher --benchmark ct_operations
```

## Best Practices

### Hardware Backend Selection

1. **Prefer Hardware**: Always try hardware backends first for maximum security
2. **Validate Attestation**: Verify attestation reports in production
3. **Fallback Strategy**: Implement graceful degradation to software backend
4. **Key Management**: Use sealed storage for long-term keys

### Side-Channel Testing

1. **Test Early**: Integrate side-channel tests in CI/CD
2. **Multiple Metrics**: Use dudect + cache simulation + memory tracing
3. **Statistical Rigor**: Collect sufficient samples (>1000) for dudect
4. **Isolated Environment**: Run timing tests on dedicated hardware when possible
5. **Regression Testing**: Track t-statistics over time

### Continuous Monitoring

```bash
# Add to CI pipeline
cargo test --test hardware_sidechannel_tests

# Monitor for regressions
cargo test --test property_timing_tests
```

## Limitations

- **Hardware Backend**: Implementations are currently placeholders; full SGX/TrustZone integration requires platform-specific SDKs
- **Cache Simulation**: Simplified model; real cache behavior is more complex
- **Timing Tests**: Subject to system noise; may require multiple runs or isolated hardware
- **Attestation**: Mock implementations; production requires proper PKI infrastructure

## Future Enhancements

- [ ] Full SGX enclave implementation with sealing/attestation
- [ ] ARM TrustZone TA (Trusted Application) integration
- [ ] PKCS#11 HSM driver support
- [ ] ChipWhisperer Python bindings
- [ ] Automated dudect regression tracking
- [ ] Cache simulator with realistic timing models
- [ ] Power analysis integration (DPA/CPA)
- [ ] EM emission analysis

## References

- [Intel SGX Developer Guide](https://software.intel.com/content/www/us/en/develop/topics/software-guard-extensions.html)
- [ARM TrustZone Documentation](https://developer.arm.com/ip-products/security-ip/trustzone)
- [dudect Paper](https://eprint.iacr.org/2016/1123.pdf)
- [Cache-Timing Attacks](https://www.usenix.org/system/files/conference/usenixsecurity14/sec14-paper-yarom.pdf)
- [Constant-Time Crypto Coding](https://github.com/veorq/cryptocoding)

## License

Same as parent crate (MIT).
