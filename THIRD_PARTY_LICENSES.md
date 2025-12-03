<!--
Copyright (c) 2025 NexusZero Protocol. All Rights Reserved.

This documentation is part of NexusZero Protocol.

NexusZero Protocol™, NexusZero™, and Privacy Morphing™ are trademarks of
NexusZero Protocol.

Licensed under AGPLv3. Commercial licenses available at licensing@nexuszero.io.

Patent Pending.
-->

# Third-Party Licenses

This document lists the third-party libraries and tools used in NexusZero Protocol,
along with their licenses and attributions.

---

## Table of Contents

1. [Rust Dependencies](#rust-dependencies)
2. [Python Dependencies](#python-dependencies)
3. [JavaScript/TypeScript Dependencies](#javascripttypescript-dependencies)
4. [Solidity Dependencies](#solidity-dependencies)
5. [Tools and Infrastructure](#tools-and-infrastructure)

---

## Rust Dependencies

The following Rust crates are used under their respective open-source licenses:

### MIT License

| Crate        | Version | Description              |
| ------------ | ------- | ------------------------ |
| `tokio`      | 1.40    | Async runtime            |
| `axum`       | 0.7     | Web framework            |
| `tower`      | 0.5     | Service abstractions     |
| `serde`      | 1.0     | Serialization framework  |
| `serde_json` | 1.0     | JSON serialization       |
| `ndarray`    | 0.15    | N-dimensional arrays     |
| `rand`       | 0.8     | Random number generation |
| `tracing`    | 0.1     | Logging/tracing          |
| `reqwest`    | 0.12    | HTTP client              |
| `uuid`       | 1.0     | UUID generation          |
| `chrono`     | 0.4     | Date/time handling       |

### Apache-2.0 License

| Crate   | Version | Description     |
| ------- | ------- | --------------- |
| `sqlx`  | 0.8     | Database driver |
| `redis` | 0.27    | Redis client    |

### MIT OR Apache-2.0

| Crate        | Version | Description                  |
| ------------ | ------- | ---------------------------- |
| `sha2`       | 0.10    | SHA-2 hashing                |
| `sha3`       | 0.10    | SHA-3 hashing                |
| `blake3`     | 1.5     | BLAKE3 hashing               |
| `thiserror`  | 1.0     | Error derive macros          |
| `anyhow`     | 1.0     | Error handling               |
| `num-bigint` | 0.4     | Arbitrary precision integers |
| `lz4`        | 1.24    | LZ4 compression              |
| `zstd`       | 0.13    | Zstandard compression        |
| `bincode`    | 1.3     | Binary serialization         |
| `rayon`      | 1.7     | Parallel iterators           |
| `criterion`  | 0.5     | Benchmarking                 |

### BSD-3-Clause

| Crate         | Version | Description     |
| ------------- | ------- | --------------- |
| `num-complex` | 0.4     | Complex numbers |

---

## Python Dependencies

The following Python packages are used under their respective licenses:

### MIT License

| Package        | Version | Description             |
| -------------- | ------- | ----------------------- |
| `pytorch`      | 2.0+    | Deep learning framework |
| `numpy`        | 1.24+   | Numerical computing     |
| `pandas`       | 2.0+    | Data analysis           |
| `scikit-learn` | 1.3+    | Machine learning        |

### Apache-2.0 License

| Package           | Version | Description           |
| ----------------- | ------- | --------------------- |
| `torch-geometric` | 2.3+    | Graph neural networks |
| `transformers`    | 4.30+   | NLP models            |

### BSD-3-Clause

| Package | Version | Description          |
| ------- | ------- | -------------------- |
| `scipy` | 1.11+   | Scientific computing |

---

## JavaScript/TypeScript Dependencies

The following npm packages are used under their respective licenses:

### MIT License

| Package      | Version | Description         |
| ------------ | ------- | ------------------- |
| `typescript` | 5.0+    | TypeScript compiler |
| `jest`       | 29+     | Testing framework   |
| `eslint`     | 8+      | Linting             |
| `prettier`   | 3+      | Code formatting     |

---

## Solidity Dependencies

### MIT License

| Package                   | Version | Description            |
| ------------------------- | ------- | ---------------------- |
| `@openzeppelin/contracts` | 5.0     | Smart contract library |

---

## Tools and Infrastructure

### MIT License

- **Docker**: Container runtime
- **Kubernetes**: Container orchestration
- **Prometheus**: Metrics collection
- **Grafana**: Metrics visualization

### Apache-2.0 License

- **Redis**: In-memory data store
- **Kafka**: Message streaming

---

## License Texts

Full license texts for the above libraries can be found in their respective repositories:

- **MIT License**: https://opensource.org/licenses/MIT
- **Apache-2.0 License**: https://opensource.org/licenses/Apache-2.0
- **BSD-3-Clause**: https://opensource.org/licenses/BSD-3-Clause

---

## Generating License Reports

To generate a comprehensive license report for Rust dependencies:

```bash
cargo install cargo-license
cargo license --json > licenses.json
```

For Python dependencies:

```bash
pip install pip-licenses
pip-licenses --format=json > python_licenses.json
```

For npm dependencies:

```bash
npx license-checker --json > npm_licenses.json
```

---

## Updates

This document should be updated whenever dependencies are added, removed, or updated.

Last updated: 2025

---

_Note: This document provides attribution for third-party libraries. The NexusZero
Protocol source code itself is licensed under AGPL-3.0-or-later with commercial
licensing available._
