---
layout: home

hero:
  name: Nexuszero Protocol
  text: Quantum-Resistant Zero-Knowledge Proofs
  tagline: Build privacy-preserving applications with lattice-based cryptography
  image:
    src: /logo.svg
    alt: Nexuszero Protocol
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/iamthegreatdestroyer/Nexuszero-Protocol

features:
  - icon: 🔒
    title: Quantum-Resistant
    details: Built on lattice-based cryptography that resists attacks from quantum computers
  - icon: 🎯
    title: Zero-Knowledge
    details: Prove statements without revealing the underlying secret data
  - icon: 📦
    title: Easy to Use
    details: Simple TypeScript API with comprehensive documentation and examples
  - icon: ⚡
    title: High Performance
    details: Optimized implementations with logarithmic-size proofs
  - icon: 🛡️
    title: Type-Safe
    details: Full TypeScript support with complete type definitions
  - icon: 🧪
    title: Well-Tested
    details: Comprehensive test coverage and formal verification
---

## Quick Example

Generate a range proof in just a few lines of code:

```typescript
import { NexuszeroClient } from 'nexuszero-sdk';

const client = new NexuszeroClient();

// Prove age is over 18 without revealing exact age
const proof = await client.proveRange({
  value: 25n,      // Secret: actual age
  min: 18n,        // Public: minimum age
  max: 150n,       // Public: reasonable maximum
});

// Verify the proof
const result = await client.verifyProof(proof);
console.log('Valid:', result.valid); // true
```

## Use Cases

### 🔐 Privacy-Preserving Identity

Prove you meet age, income, or credential requirements without revealing exact details.

### 💰 Confidential Finance

Verify account balances or transaction amounts are within acceptable ranges without disclosure.

### 🎫 Anonymous Access Control

Grant access based on attributes without revealing identity or exact attribute values.

### 📊 Private Data Analytics

Prove statistical properties of data without exposing the underlying dataset.

## Why Nexuszero?

Traditional zero-knowledge proof systems often rely on cryptography that quantum computers could break. Nexuszero uses **lattice-based cryptography**, which is believed to be resistant to quantum attacks, ensuring your privacy-preserving applications remain secure in a post-quantum world.

### Key Features

- **Bulletproofs Protocol**: Logarithmic-size range proofs (O(log n) vs O(n))
- **No Trusted Setup**: No need for trusted parameter generation ceremonies
- **Pedersen Commitments**: Cryptographically binding and hiding commitments
- **Fiat-Shamir Transform**: Non-interactive proofs from interactive protocols
- **TypeScript SDK**: Developer-friendly API with full type safety

## Getting Started

Install the SDK:

```bash
npm install nexuszero-sdk
```

Read the [Getting Started Guide](/guide/getting-started) to begin building privacy-preserving applications.

## Community

- [GitHub Repository](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol)
- [Issue Tracker](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol/issues)
- [Discussions](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol/discussions)

## License

MIT - see [LICENSE](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol/blob/main/LICENSE) for details.
