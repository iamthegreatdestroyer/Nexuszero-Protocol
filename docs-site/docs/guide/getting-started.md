# Getting Started

Welcome to Nexuszero Protocol! This guide will help you get up and running with quantum-resistant zero-knowledge proofs in minutes.

## What You'll Build

In this guide, you'll learn how to:
- Install the Nexuszero SDK
- Generate your first range proof
- Verify proofs
- Handle errors gracefully

## Installation

Install the SDK using npm:

```bash
npm install nexuszero-sdk
```

Or using yarn:

```bash
yarn add nexuszero-sdk
```

## Your First Proof

Let's create a simple age verification proof that proves someone is over 18 without revealing their exact age.

### 1. Import the SDK

```typescript
import { NexuszeroClient, SecurityLevel } from 'nexuszero-sdk';
```

### 2. Create a Client

```typescript
const client = new NexuszeroClient({
  securityLevel: SecurityLevel.Bit128
});
```

The client is configured with a security level. Available options:
- `SecurityLevel.Bit128` - 128-bit security (default, recommended)
- `SecurityLevel.Bit192` - 192-bit security
- `SecurityLevel.Bit256` - 256-bit security

### 3. Generate a Range Proof

```typescript
const proof = await client.proveRange({
  value: 25n,      // The secret value (BigInt)
  min: 18n,        // Minimum value (inclusive)
  max: 150n,       // Maximum value (exclusive)
});
```

::: tip
Note the use of BigInt literals (`n` suffix). All values in Nexuszero use BigInt for arbitrary precision arithmetic.
:::

### 4. Verify the Proof

```typescript
const result = await client.verifyProof(proof);

if (result.valid) {
  console.log('✓ Proof is valid - age is between 18 and 150');
} else {
  console.error('✗ Proof verification failed:', result.error);
}
```

## Complete Example

Here's the complete code:

```typescript
import { NexuszeroClient, SecurityLevel } from 'nexuszero-sdk';

async function main() {
  // Create client
  const client = new NexuszeroClient({
    securityLevel: SecurityLevel.Bit128
  });

  // Generate proof
  const proof = await client.proveRange({
    value: 25n,      // Secret age
    min: 18n,        // Must be at least 18
    max: 150n,       // Reasonable upper bound
  });

  console.log('Proof generated successfully!');
  console.log('Proof size:', proof.data.length, 'bytes');

  // Verify proof
  const result = await client.verifyProof(proof);

  if (result.valid) {
    console.log('✓ Verified: Age is over 18');
  } else {
    console.error('✗ Verification failed:', result.error);
  }
}

main().catch(console.error);
```

## Understanding the Output

When you run this code, you'll see:

```
Proof generated successfully!
Proof size: 256 bytes
✓ Verified: Age is over 18
```

The proof size is relatively small (logarithmic in the range size), making it efficient to store and transmit.

## Error Handling

Always wrap proof operations in try-catch blocks:

```typescript
import { NexuszeroClient, NexuszeroError, ErrorCode } from 'nexuszero-sdk';

try {
  const proof = await client.proveRange({
    value: 200n,    // This is out of range!
    min: 18n,
    max: 150n,
  });
} catch (error) {
  if (error instanceof NexuszeroError) {
    if (error.code === ErrorCode.OutOfRange) {
      console.error('Value is outside the specified range');
    }
    console.error('Error:', error.message);
  }
}
```

## Next Steps

Now that you've created your first proof, explore:

- [Zero-Knowledge Proofs Concepts](/guide/zero-knowledge-proofs)
- [API Reference](/api/client)
- [More Examples](/examples/age-verification)

## Common Patterns

### Using the ProofBuilder

For more control, use the builder pattern:

```typescript
import { ProofBuilder, StatementType } from 'nexuszero-sdk';

const proof = await new ProofBuilder()
  .setStatement(StatementType.Range, {
    min: 0n,
    max: 100n,
  })
  .setWitness({
    value: 42n,
  })
  .generate();
```

### Custom Blinding Factors

For deterministic commitments or advanced use cases:

```typescript
const blinding = client.generateBlinding(32); // 32 bytes

const proof = await client.proveRange({
  value: 42n,
  min: 0n,
  max: 100n,
  blinding: blinding, // Use custom blinding
});
```

### Debug Mode

Enable debug logging during development:

```typescript
const client = new NexuszeroClient({
  securityLevel: SecurityLevel.Bit128,
  debug: true, // Enable debug output
});
```

## TypeScript Support

Nexuszero SDK has full TypeScript support with type definitions for all APIs:

```typescript
import type {
  Proof,
  VerificationResult,
  CryptoParameters,
} from 'nexuszero-sdk';

// Types are automatically inferred
const proof: Proof = await client.proveRange({ /* ... */ });
const result: VerificationResult = await client.verifyProof(proof);
```

## Performance Tips

1. **Reuse the client**: Create one client instance and reuse it
2. **Choose appropriate ranges**: Smaller ranges result in smaller proofs
3. **Batch operations**: Generate multiple proofs in parallel when possible

```typescript
// Parallel proof generation
const proofs = await Promise.all([
  client.proveRange({ value: 25n, min: 18n, max: 150n }),
  client.proveRange({ value: 75000n, min: 50000n, max: 100000n }),
  client.proveRange({ value: 5000n, min: 1000n, max: 10000n }),
]);
```

## What's Next?

Continue learning:
- [Range Proofs in Depth](/guide/range-proofs)
- [Security Levels Explained](/guide/security-levels)
- [Advanced Examples](/examples/custom-proofs)
