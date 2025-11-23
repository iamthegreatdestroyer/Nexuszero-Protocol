# NexuszeroClient

The main client class for interacting with the Nexuszero SDK.

## Constructor

```typescript
new NexuszeroClient(config?: SDKConfig)
```

Creates a new Nexuszero client instance.

### Parameters

- `config` (optional): Configuration object

#### SDKConfig

```typescript
interface SDKConfig {
  securityLevel?: SecurityLevel;
  customParameters?: CryptoParameters;
  debug?: boolean;
}
```

**Options:**

- `securityLevel`: Security level to use (`Bit128`, `Bit192`, `Bit256`)
  - Default: `SecurityLevel.Bit128`
- `customParameters`: Custom cryptographic parameters (overrides `securityLevel`)
- `debug`: Enable debug logging
  - Default: `false`

### Example

```typescript
import { NexuszeroClient, SecurityLevel } from 'nexuszero-sdk';

// With default configuration
const client = new NexuszeroClient();

// With custom security level
const client = new NexuszeroClient({
  securityLevel: SecurityLevel.Bit256
});

// With debug mode
const client = new NexuszeroClient({
  securityLevel: SecurityLevel.Bit128,
  debug: true
});
```

## Methods

### proveRange()

Generate a range proof.

```typescript
async proveRange(options: RangeProofOptions): Promise<Proof>
```

#### Parameters

```typescript
interface RangeProofOptions {
  value: bigint;        // Secret value to prove
  min: bigint;          // Minimum value (inclusive)
  max: bigint;          // Maximum value (exclusive)
  blinding?: Uint8Array; // Optional blinding factor
}
```

#### Returns

`Promise<Proof>` - The generated zero-knowledge proof

#### Example

```typescript
const proof = await client.proveRange({
  value: 42n,
  min: 0n,
  max: 100n,
});
```

#### Throws

- `NexuszeroError` with code `OutOfRange` if value is outside [min, max)
- `NexuszeroError` with code `InvalidParameters` if min >= max

---

### verifyProof()

Verify a zero-knowledge proof.

```typescript
async verifyProof(proof: Proof): Promise<VerificationResult>
```

#### Parameters

- `proof`: The proof to verify

#### Returns

```typescript
interface VerificationResult {
  valid: boolean;
  error?: string;
}
```

#### Example

```typescript
const result = await client.verifyProof(proof);

if (result.valid) {
  console.log('Proof is valid');
} else {
  console.error('Verification failed:', result.error);
}
```

---

### createCommitment()

Create a Pedersen commitment to a value.

```typescript
async createCommitment(
  value: bigint,
  blinding?: Uint8Array
): Promise<Commitment>
```

#### Parameters

- `value`: Value to commit to
- `blinding` (optional): Blinding factor (generated if not provided)

#### Returns

```typescript
interface Commitment {
  data: Uint8Array;      // Public commitment data
  value?: bigint;        // Secret value (optional)
  blinding?: Uint8Array; // Secret blinding (optional)
}
```

#### Example

```typescript
// Generate commitment with random blinding
const commitment = await client.createCommitment(42n);

// Use custom blinding
const blinding = client.generateBlinding();
const commitment = await client.createCommitment(42n, blinding);
```

---

### generateBlinding()

Generate a cryptographically secure random blinding factor.

```typescript
generateBlinding(length?: number): Uint8Array
```

#### Parameters

- `length` (optional): Length in bytes (default: 32)

#### Returns

`Uint8Array` - Random bytes

#### Example

```typescript
const blinding = client.generateBlinding();     // 32 bytes
const blinding64 = client.generateBlinding(64); // 64 bytes
```

---

### getParameters()

Get the current cryptographic parameters.

```typescript
getParameters(): CryptoParameters
```

#### Returns

```typescript
interface CryptoParameters {
  n: number;              // Dimension parameter
  q: number;              // Modulus
  sigma: number;          // Error distribution parameter
  securityLevel: SecurityLevel;
}
```

#### Example

```typescript
const params = client.getParameters();
console.log('Using n =', params.n);
console.log('Security level:', params.securityLevel);
```

---

### createProofBuilder()

Create a ProofBuilder for advanced proof construction.

```typescript
createProofBuilder(): ProofBuilder
```

#### Returns

`ProofBuilder` - A new ProofBuilder instance

#### Example

```typescript
const builder = client.createProofBuilder();

const proof = await builder
  .setStatement(StatementType.Range, { min: 0n, max: 100n })
  .setWitness({ value: 42n })
  .generate();
```

See [ProofBuilder](/api/proof-builder) for more details.

## Type Definitions

### SecurityLevel

```typescript
enum SecurityLevel {
  Bit128 = "128",  // 128-bit security (recommended)
  Bit192 = "192",  // 192-bit security
  Bit256 = "256",  // 256-bit security
}
```

### Proof

```typescript
interface Proof {
  data: Uint8Array;       // Serialized proof data
  statement: Statement;   // Statement being proven
  commitment: Uint8Array; // Commitment to the witness
}
```

## Complete Example

```typescript
import {
  NexuszeroClient,
  SecurityLevel,
  NexuszeroError,
  ErrorCode
} from 'nexuszero-sdk';

async function main() {
  // Create client
  const client = new NexuszeroClient({
    securityLevel: SecurityLevel.Bit128,
    debug: true
  });

  try {
    // Generate proof
    const proof = await client.proveRange({
      value: 25n,
      min: 18n,
      max: 150n,
    });

    console.log('Proof generated:', proof.data.length, 'bytes');

    // Verify proof
    const result = await client.verifyProof(proof);
    console.log('Valid:', result.valid);

    // Create commitment
    const commitment = await client.createCommitment(42n);
    console.log('Commitment:', commitment.data.length, 'bytes');

  } catch (error) {
    if (error instanceof NexuszeroError) {
      console.error('Error code:', error.code);
      console.error('Message:', error.message);
    }
  }
}

main();
```

## See Also

- [ProofBuilder](/api/proof-builder) - Advanced proof construction
- [Error Handling](/api/errors) - Error codes and handling
- [Types](/api/types) - Complete type reference
