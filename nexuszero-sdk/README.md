# Nexuszero SDK

TypeScript SDK for the Nexuszero Protocol - A quantum-resistant zero-knowledge proof system based on lattice cryptography.

## Features

- 🔒 **Quantum-Resistant**: Built on lattice-based cryptography
- 🎯 **Zero-Knowledge Proofs**: Prove statements without revealing secrets
- 📦 **Easy to Use**: Simple, intuitive TypeScript API
- ⚡ **Promise-Based**: Fully async/await compatible
- 🛡️ **Type-Safe**: Complete TypeScript type definitions
- 🧪 **Well-Tested**: Comprehensive test coverage

## Installation

```bash
npm install nexuszero-sdk
```

## Quick Start

### Basic Usage

```typescript
import { NexuszeroClient, SecurityLevel } from 'nexuszero-sdk';

// Create a client
const client = new NexuszeroClient({
  securityLevel: SecurityLevel.Bit128
});

// Generate a range proof
const proof = await client.proveRange({
  value: 42n,       // The secret value
  min: 0n,          // Range minimum (inclusive)
  max: 100n,        // Range maximum (exclusive)
});

// Verify the proof
const result = await client.verifyProof(proof);
console.log('Valid:', result.valid); // true
```

### Using the ProofBuilder Pattern

```typescript
import { ProofBuilder, StatementType } from 'nexuszero-sdk';

const proof = await new ProofBuilder()
  .setStatement(StatementType.Range, { min: 0n, max: 100n })
  .setWitness({ value: 42n })
  .generate();
```

### Creating Commitments

```typescript
// Create a commitment to a value
const commitment = await client.createCommitment(42n);

// Create a commitment with custom blinding
const blinding = client.generateBlinding();
const commitment2 = await client.createCommitment(42n, blinding);
```

## API Reference

### NexuszeroClient

The main client class for interacting with the SDK.

#### Constructor

```typescript
new NexuszeroClient(config?: SDKConfig)
```

**Options:**
- `securityLevel`: Security level (Bit128, Bit192, Bit256)
- `customParameters`: Custom cryptographic parameters
- `debug`: Enable debug logging

#### Methods

##### `proveRange(options: RangeProofOptions): Promise<Proof>`

Generate a range proof.

**Parameters:**
- `value`: The secret value to prove (bigint)
- `min`: Range minimum, inclusive (bigint)
- `max`: Range maximum, exclusive (bigint)
- `blinding`: Optional blinding factor (Uint8Array)

**Returns:** Promise\<Proof\>

##### `verifyProof(proof: Proof): Promise<VerificationResult>`

Verify a zero-knowledge proof.

**Parameters:**
- `proof`: The proof to verify

**Returns:** Promise\<VerificationResult\>

##### `createCommitment(value: bigint, blinding?: Uint8Array): Promise<Commitment>`

Create a Pedersen commitment to a value.

**Parameters:**
- `value`: Value to commit to
- `blinding`: Optional blinding factor

**Returns:** Promise\<Commitment\>

##### `generateBlinding(length?: number): Uint8Array`

Generate a random blinding factor.

**Parameters:**
- `length`: Length in bytes (default: 32)

**Returns:** Uint8Array

##### `getParameters(): CryptoParameters`

Get the current cryptographic parameters.

**Returns:** CryptoParameters

##### `createProofBuilder(): ProofBuilder`

Create a new ProofBuilder for advanced proof construction.

**Returns:** ProofBuilder

### ProofBuilder

Builder pattern for constructing zero-knowledge proofs.

#### Methods

##### `setStatement(type: StatementType, data: any): ProofBuilder`

Set the statement to be proven.

**Parameters:**
- `type`: Statement type (e.g., StatementType.Range)
- `data`: Statement-specific data

**Returns:** ProofBuilder (for chaining)

##### `setWitness(data: any): ProofBuilder`

Set the witness (secret data) for proof generation.

**Parameters:**
- `data`: Witness data containing the secret value

**Returns:** ProofBuilder (for chaining)

##### `generate(): Promise<Proof>`

Generate the zero-knowledge proof.

**Returns:** Promise\<Proof\>

### Types

#### SecurityLevel

```typescript
enum SecurityLevel {
  Bit128 = "128",  // 128-bit security
  Bit192 = "192",  // 192-bit security
  Bit256 = "256",  // 256-bit security
}
```

#### StatementType

```typescript
enum StatementType {
  Range = "range",           // Range proof
  Membership = "membership", // Membership proof
  Custom = "custom",         // Custom statement
}
```

#### ErrorCode

```typescript
enum ErrorCode {
  InvalidParameters = "INVALID_PARAMETERS",
  ProofGenerationFailed = "PROOF_GENERATION_FAILED",
  VerificationFailed = "VERIFICATION_FAILED",
  OutOfRange = "OUT_OF_RANGE",
  InvalidCommitment = "INVALID_COMMITMENT",
  FFIError = "FFI_ERROR",
  SerializationError = "SERIALIZATION_ERROR",
}
```

## Examples

### Age Verification (Privacy-Preserving)

```typescript
import { NexuszeroClient } from 'nexuszero-sdk';

const client = new NexuszeroClient();

// User proves they are over 18 without revealing exact age
const age = 25n; // Secret age
const proof = await client.proveRange({
  value: age,
  min: 18n,
  max: 150n, // Reasonable maximum age
});

// Verifier checks the proof
const result = await client.verifyProof(proof);
if (result.valid) {
  console.log('User is over 18 ✓');
} else {
  console.log('Verification failed ✗');
}
```

### Salary Range Proof

```typescript
// Prove salary is in range without revealing exact amount
const salary = 75000n;
const proof = await client.proveRange({
  value: salary,
  min: 50000n,
  max: 100000n,
});
```

### Balance Sufficiency Check

```typescript
// Prove account balance is sufficient without revealing exact balance
const balance = 5000n;
const proof = await client.proveRange({
  value: balance,
  min: 1000n, // Minimum required balance
  max: 1000000n, // Maximum reasonable balance
});
```

## Error Handling

```typescript
import { NexuszeroError, ErrorCode } from 'nexuszero-sdk';

try {
  const proof = await client.proveRange({
    value: 150n,
    min: 0n,
    max: 100n, // Value out of range!
  });
} catch (error) {
  if (error instanceof NexuszeroError) {
    console.error('Error code:', error.code);
    console.error('Error message:', error.message);
    
    if (error.code === ErrorCode.OutOfRange) {
      console.error('Value is outside the specified range');
    }
  }
}
```

## Development

### Build

```bash
npm run build
```

### Test

```bash
npm test
```

### Lint

```bash
npm run lint
```

## Security Considerations

1. **Blinding Factors**: Always use cryptographically secure random blinding factors
2. **Range Selection**: Choose appropriate ranges for your use case
3. **Parameter Selection**: Use recommended security levels (Bit128 minimum)
4. **Side Channels**: The current implementation focuses on correctness; constant-time operations are planned

## Roadmap

- [ ] WASM compilation for browser support
- [ ] Additional proof types (membership, equality)
- [ ] Hardware acceleration support
- [ ] Batch proof generation and verification
- [ ] Interactive proof protocols

## License

MIT

## Contributing

Contributions are welcome! Please see the main repository for guidelines.

## Links

- [GitHub Repository](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol)
- [Documentation](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol/tree/main/docs)
- [Issue Tracker](https://github.com/iamthegreatdestroyer/Nexuszero-Protocol/issues)

## Support

For questions and support, please open an issue on GitHub.
