# NexusZero Rust SDK

Rust SDK with WASM bindings for the NexusZero Protocol - privacy-preserving blockchain transactions with quantum-resistant proofs.

## Features

- 🔐 **6-Level Privacy Spectrum** - From transparent to sovereign privacy
- ⚡ **Adaptive Privacy Morphing** - Context-aware privacy recommendations
- 🌐 **WASM Support** - Browser-ready via WebAssembly
- 🔒 **Quantum-Resistant** - Lattice-based cryptographic proofs
- 🌉 **Cross-Chain Bridge** - Privacy-preserving asset transfers

## Installation

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
nexuszero-sdk = "0.1.0"
```

### WASM (Browser)

Build with wasm-pack:

```bash
wasm-pack build --target web
```

## Quick Start

```rust
use nexuszero_sdk::{NexusZeroClient, TransactionRequest, PrivacyLevel};

fn main() {
    // Create client
    let client = NexusZeroClient::new();

    // Create a transaction request
    let request = TransactionRequest::new(
        "0xsender".to_string(),
        "0xrecipient".to_string(),
        "1000".to_string(),
        PrivacyLevel::Private as u8,
    );

    // Create transaction
    let tx = client.create_transaction(request).unwrap();
    println!("Transaction ID: {}", tx.id);

    // Generate proof
    let proof = client.generate_proof(&tx).unwrap();
    println!("Proof hash: {}", proof.proof_hash);

    // Verify proof
    let valid = client.verify_proof(&proof).unwrap();
    println!("Proof valid: {}", valid);
}
```

## Privacy Levels

| Level | Name         | Description              |
| ----- | ------------ | ------------------------ |
| 0     | Transparent  | Public blockchain parity |
| 1     | Pseudonymous | Address obfuscation      |
| 2     | Confidential | Encrypted amounts        |
| 3     | Private      | Full transaction privacy |
| 4     | Anonymous    | Unlinkable transactions  |
| 5     | Sovereign    | Maximum ZK privacy       |

## Adaptive Privacy Morphing (APM)

```rust
use nexuszero_sdk::privacy::{PrivacyEngine, TransactionContext};

let engine = PrivacyEngine::new();

let context = TransactionContext {
    value_usd: 50_000.0,
    requires_compliance: true,
    ..Default::default()
};

let recommendation = engine.recommend(&context).unwrap();
println!("Recommended level: {}", recommendation.level);
println!("Reasons: {:?}", recommendation.reasons);
```

## WASM Usage

```javascript
import init, { NexusZeroClient, TransactionRequest } from "nexuszero-sdk";

await init();

const client = new NexusZeroClient();
const request = new TransactionRequest(
  "0xsender",
  "0xrecipient",
  "1000",
  3 // Privacy level
);

console.log("SDK Version:", sdk_version());
```

## Building

```bash
# Build library
cargo build --release

# Run tests
cargo test

# Build WASM
wasm-pack build --target web --features wasm
```

## License

MIT OR Apache-2.0
