# NexusZero Integration API - Deployment Guide

**Version:** 0.1.0  
**Date:** November 27, 2025  
**Status:** Production Ready

---

## Overview

This guide covers deploying the NexusZero Integration API, which provides a unified interface for quantum-resistant zero-knowledge proof generation and verification. The integration layer orchestrates three core modules: crypto, holographic compression, and neural optimization.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Performance Tuning](#performance-tuning)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **OS:** Linux (Ubuntu 20.04+), macOS (10.15+), or Windows (10+ with WSL2)
- **CPU:** 2-core x64 processor (4+ cores recommended)
- **RAM:** 4GB (8GB+ recommended for high throughput)
- **Storage:** 1GB free space
- **Rust:** 1.70+ (for building from source)

### Recommended Production Setup

- **CPU:** 4+ cores (8+ cores for high throughput)
- **RAM:** 8GB+ (16GB+ for concurrent proof generation)
- **Storage:** SSD with 10GB+ free space
- **Network:** 100Mbps+ connection for dependency downloads

## Installation Methods

### Method 1: Cargo (Recommended for Development)

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Clone the repository
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol

# Build the integration library
cargo build --release --package nexuszero-integration

# Run tests to verify installation
cargo test --package nexuszero-integration
```

### Method 2: Docker (Recommended for Production)

```bash
# Pull the pre-built image
docker pull iamthegreatdestroyer/nexuszero-integration:latest

# Or build from source
git clone https://github.com/iamthegreatdestroyer/Nexuszero-Protocol.git
cd Nexuszero-Protocol
docker build -t nexuszero-integration -f deployment/Dockerfile .

# Run the container
docker run -p 8080:8080 nexuszero-integration
```

### Method 3: Docker Compose (Full Stack)

For a complete deployment including all NexusZero services:

```yaml
# docker-compose.yml
version: "3.8"
services:
  nexuszero-integration:
    image: iamthegreatdestroyer/nexuszero-integration:latest
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - NEXUSZERO_SECURITY_LEVEL=128
      - NEXUSZERO_COMPRESSION_ENABLED=true
    volumes:
      - ./config:/app/config:ro
    restart: unless-stopped

  # Optional: Include monitoring stack
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

```bash
# Start the services
docker-compose up -d

# Check status
docker-compose ps
```

## Configuration

### Environment Variables

| Variable                            | Default | Description                                     |
| ----------------------------------- | ------- | ----------------------------------------------- |
| `RUST_LOG`                          | `info`  | Logging level (error, warn, info, debug, trace) |
| `NEXUSZERO_SECURITY_LEVEL`          | `128`   | Security level in bits (128, 256)               |
| `NEXUSZERO_COMPRESSION_ENABLED`     | `true`  | Enable/disable proof compression                |
| `NEXUSZERO_OPTIMIZER_ENABLED`       | `true`  | Enable/disable neural optimization              |
| `NEXUSZERO_MAX_PROOF_SIZE`          | `10000` | Maximum proof size in bytes                     |
| `NEXUSZERO_MAX_VERIFY_TIME`         | `50.0`  | Maximum verification time in milliseconds       |
| `NEXUSZERO_VERIFY_AFTER_GENERATION` | `false` | Verify proofs immediately after generation      |

### Configuration File

Create a `config/integration.toml` file:

```toml
[protocol]
security_level = "Bit128"  # Bit128, Bit256
use_compression = true
use_optimizer = true
max_proof_size = 10000
max_verify_time = 50.0
verify_after_generation = false

[logging]
level = "info"
format = "json"  # json, compact

[metrics]
enabled = true
endpoint = "127.0.0.1:9090"  # Prometheus endpoint
```

### Runtime Configuration

```rust
use nexuszero_integration::{NexuszeroAPI, ProtocolConfig};
use nexuszero_crypto::SecurityLevel;

let config = ProtocolConfig {
    security_level: SecurityLevel::Bit256,
    use_compression: true,
    use_optimizer: true,
    max_proof_size: Some(20_000),
    max_verify_time: Some(100.0),
    verify_after_generation: true,
};

let api = NexuszeroAPI::with_config(config);
```

## Performance Tuning

### Memory Optimization

```rust
// For memory-constrained environments
let api = NexuszeroAPI::fast(); // Reduces memory usage

// Custom memory limits
let config = ProtocolConfig {
    max_proof_size: Some(5_000), // Smaller proofs
    ..Default::default()
};
```

### CPU Optimization

```rust
// For high-throughput scenarios
let config = ProtocolConfig {
    verify_after_generation: false, // Skip verification during generation
    max_verify_time: Some(25.0),    // Faster verification timeout
    ..Default::default()
};
```

### Benchmarking Performance

```bash
# Run performance benchmarks
cargo bench --package nexuszero-integration

# Profile with flame graphs
cargo flamegraph --bin nexuszero-integration -- test::bench_prove_discrete_log
```

### Expected Performance

| Configuration | Generation Time | Verification Time | Memory Usage |
| ------------- | --------------- | ----------------- | ------------ |
| `fast()`      | 50-80ms         | 20-40ms           | ~50MB        |
| `default()`   | 80-120ms        | 30-60ms           | ~75MB        |
| `secure()`    | 120-200ms       | 50-100ms          | ~100MB       |

## Monitoring

### Health Checks

```rust
use nexuszero_integration::NexuszeroAPI;

// Check if the API is operational
let api = NexuszeroAPI::new();
assert!(api.is_compression_enabled());
assert!(api.is_optimizer_enabled());
```

### Metrics Collection

The integration API automatically collects comprehensive metrics:

```rust
let mut api = NexuszeroAPI::new();
let proof = api.prove_preimage(hash_function, hash_output, preimage).unwrap();

// Basic metrics
let metrics = api.get_metrics(&proof);
println!("Generation: {:.2}ms", metrics.generation_time_ms);
println!("Size: {} bytes", metrics.proof_size_bytes);
println!("Compression: {:.2}x", metrics.compression_ratio);

// Comprehensive metrics
if let Some(comprehensive) = api.get_comprehensive_metrics(&proof) {
    println!("Pipeline stages: {:?}", comprehensive.stage_timings);
    println!("Memory peaks: {:?}", comprehensive.memory_usage);
}
```

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: "nexuszero-integration"
    static_configs:
      - targets: ["localhost:8080"]
    metrics_path: "/metrics"
```

### Logging

```rust
// Set logging level
std::env::set_var("RUST_LOG", "nexuszero_integration=debug");

// Initialize logger
env_logger::init();
```

## Troubleshooting

### Common Issues

#### 1. Compilation Errors

**Problem:** `cargo build` fails with dependency errors

**Solution:**

```bash
# Clean build artifacts
cargo clean

# Update dependencies
cargo update

# Rebuild
cargo build --release
```

#### 2. Runtime Panics

**Problem:** Application crashes during proof generation

**Solution:**

```rust
// Enable detailed logging
std::env::set_var("RUST_LOG", "trace");

// Check input validation
if generator.len() != 32 {
    eprintln!("Generator must be 32 bytes");
    return;
}
```

#### 3. Performance Issues

**Problem:** Proof generation is slower than expected

**Solutions:**

```rust
// Use fast configuration
let api = NexuszeroAPI::fast();

// Disable compression if not needed
let config = ProtocolConfig {
    use_compression: false,
    ..Default::default()
};

// Check system resources
// Ensure sufficient RAM (>4GB)
// Use SSD storage
// Check CPU utilization
```

#### 4. Memory Issues

**Problem:** Out of memory errors

**Solutions:**

```rust
// Reduce proof size limits
let config = ProtocolConfig {
    max_proof_size: Some(5_000),
    ..Default::default()
};

// Use streaming for large proofs (future feature)
// Process proofs sequentially instead of concurrently
```

### Debug Mode

```bash
# Build with debug symbols
cargo build

# Run with debug logging
RUST_LOG=debug cargo run

# Profile memory usage
cargo build --release
valgrind --tool=massif target/release/nexuszero-integration
```

### Getting Help

1. **Check the logs:** `RUST_LOG=trace` for detailed output
2. **Verify inputs:** Ensure all cryptographic inputs are valid
3. **Test with known values:** Use the examples in the API documentation
4. **Check system resources:** Monitor CPU, memory, and disk usage
5. **File an issue:** Include full logs, configuration, and reproduction steps

## Security Considerations

### Production Deployment

- **Use TLS:** Always deploy behind HTTPS
- **Input validation:** Validate all cryptographic inputs
- **Rate limiting:** Implement request rate limits
- **Monitoring:** Enable comprehensive logging and monitoring
- **Updates:** Keep dependencies updated for security patches

### Configuration Security

```bash
# Never commit secrets to version control
# Use environment variables or secure vaults
# Rotate cryptographic keys regularly
# Use principle of least privilege
```

## Examples

### Basic REST API Server

```rust
use nexuszero_integration::NexuszeroAPI;
use warp::Filter;

#[tokio::main]
async fn main() {
    let api = std::sync::Arc::new(std::sync::Mutex::new(NexuszeroAPI::new()));

    let prove_route = warp::path("prove")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_api(api.clone()))
        .and_then(handle_prove);

    warp::serve(prove_route)
        .run(([127, 0, 0, 1], 8080))
        .await;
}

fn with_api(
    api: std::sync::Arc<std::sync::Mutex<NexuszeroAPI>>
) -> impl Filter<Extract = (std::sync::Arc<std::sync::Mutex<NexuszeroAPI>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || api.clone())
}

async fn handle_prove(
    request: ProofRequest,
    api: std::sync::Arc<std::sync::Mutex<NexuszeroAPI>>
) -> Result<impl warp::Reply, warp::Rejection> {
    let mut api = api.lock().unwrap();

    let proof = match request.proof_type.as_str() {
        "discrete_log" => api.prove_discrete_log(
            &request.generator,
            &request.public_value,
            &request.secret
        ),
        "preimage" => api.prove_preimage(
            request.hash_function,
            &request.hash_output,
            &request.preimage
        ),
        _ => return Err(warp::reject::not_found()),
    };

    match proof {
        Ok(p) => Ok(warp::reply::json(&p)),
        Err(e) => Ok(warp::reply::json(&format!("Error: {:?}", e))),
    }
}
```

### Performance Monitoring

```rust
use nexuszero_integration::NexuszeroAPI;
use std::time::Instant;

fn benchmark_proofs() {
    let mut api = NexuszeroAPI::new();
    let mut total_time = 0.0;
    let num_proofs = 100;

    for i in 0..num_proofs {
        let start = Instant::now();

        // Generate test proof
        let proof = api.prove_preimage(
            HashFunction::SHA256,
            &hash_of_secret(i),
            &secret_data(i)
        ).unwrap();

        let elapsed = start.elapsed().as_millis() as f64;
        total_time += elapsed;

        // Verify proof
        assert!(api.verify(&proof).unwrap());
    }

    println!("Average generation time: {:.2}ms", total_time / num_proofs as f64);
    println!("Total proofs generated: {}", api.total_proofs_generated());
    println!("Total proofs verified: {}", api.total_proofs_verified());
}
```

---

_For more advanced usage and internal architecture details, see the [Integration Architecture Document](INTEGRATION_ARCHITECTURE.md)._
