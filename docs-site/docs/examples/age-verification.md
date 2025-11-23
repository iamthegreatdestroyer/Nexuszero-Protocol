# Age Verification Example

This example demonstrates how to build a privacy-preserving age verification system using Nexuszero.

## Use Case

Online services need to verify that users are old enough (e.g., 18+) without collecting or storing their exact birthdate. With Nexuszero, users can prove they meet the age requirement while keeping their actual age private.

## Implementation

### Basic Age Verification

```typescript
import { NexuszeroClient } from 'nexuszero-sdk';

async function verifyAge(age: number): Promise<boolean> {
  const client = new NexuszeroClient();

  // Generate proof that age is in valid range
  const proof = await client.proveRange({
    value: BigInt(age),
    min: 18n,        // Must be at least 18
    max: 150n,       // Reasonable upper bound
  });

  // Verify the proof
  const result = await client.verifyProof(proof);
  return result.valid;
}

// Usage
const userAge = 25;
const isOldEnough = await verifyAge(userAge);

if (isOldEnough) {
  console.log('✓ User is 18 or older');
  // Grant access to age-restricted content
} else {
  console.log('✗ Age verification failed');
  // Deny access
}
```

## Complete Application

Here's a more complete example with error handling:

```typescript
import {
  NexuszeroClient,
  NexuszeroError,
  ErrorCode,
  SecurityLevel,
  type Proof
} from 'nexuszero-sdk';

class AgeVerificationService {
  private client: NexuszeroClient;

  constructor() {
    this.client = new NexuszeroClient({
      securityLevel: SecurityLevel.Bit128,
    });
  }

  /**
   * Generate an age proof
   * @param age - User's actual age
   * @returns Proof that age meets requirements
   */
  async generateAgeProof(age: number): Promise<Proof> {
    if (age < 0 || age > 150) {
      throw new Error('Invalid age');
    }

    try {
      const proof = await this.client.proveRange({
        value: BigInt(age),
        min: 18n,
        max: 150n,
      });

      return proof;
    } catch (error) {
      if (error instanceof NexuszeroError) {
        if (error.code === ErrorCode.OutOfRange) {
          throw new Error('User is under 18');
        }
      }
      throw error;
    }
  }

  /**
   * Verify an age proof
   * @param proof - Proof to verify
   * @returns True if proof is valid
   */
  async verifyAgeProof(proof: Proof): Promise<boolean> {
    const result = await this.client.verifyProof(proof);
    return result.valid;
  }

  /**
   * Complete verification flow
   */
  async verifyUserAge(userAge: number): Promise<{
    verified: boolean;
    message: string;
  }> {
    try {
      // User generates proof
      const proof = await this.generateAgeProof(userAge);
      
      // Service verifies proof
      const verified = await this.verifyAgeProof(proof);

      return {
        verified,
        message: verified 
          ? 'Age verified successfully'
          : 'Age verification failed'
      };
    } catch (error) {
      return {
        verified: false,
        message: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}

// Usage example
async function main() {
  const service = new AgeVerificationService();

  // Test case 1: User is 25 years old
  console.log('Test 1: Age 25');
  const result1 = await service.verifyUserAge(25);
  console.log(result1);
  // Output: { verified: true, message: 'Age verified successfully' }

  // Test case 2: User is 16 years old (under 18)
  console.log('\nTest 2: Age 16');
  const result2 = await service.verifyUserAge(16);
  console.log(result2);
  // Output: { verified: false, message: 'User is under 18' }

  // Test case 3: User is exactly 18
  console.log('\nTest 3: Age 18');
  const result3 = await service.verifyUserAge(18);
  console.log(result3);
  // Output: { verified: true, message: 'Age verified successfully' }
}

main().catch(console.error);
```

## Integration with Express

Here's how to integrate age verification into an Express.js application:

```typescript
import express from 'express';
import { NexuszeroClient, type Proof } from 'nexuszero-sdk';

const app = express();
app.use(express.json());

const client = new NexuszeroClient();

// Endpoint to generate proof (user side)
app.post('/api/age/prove', async (req, res) => {
  try {
    const { age } = req.body;

    const proof = await client.proveRange({
      value: BigInt(age),
      min: 18n,
      max: 150n,
    });

    // Send proof to client
    res.json({
      success: true,
      proof: {
        data: Array.from(proof.data),
        commitment: Array.from(proof.commitment),
        statement: proof.statement,
      }
    });
  } catch (error) {
    res.status(400).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

// Endpoint to verify proof (service side)
app.post('/api/age/verify', async (req, res) => {
  try {
    const { proof } = req.body;

    // Reconstruct proof from JSON
    const reconstructedProof: Proof = {
      data: new Uint8Array(proof.data),
      commitment: new Uint8Array(proof.commitment),
      statement: proof.statement,
    };

    const result = await client.verifyProof(reconstructedProof);

    if (result.valid) {
      res.json({
        success: true,
        verified: true,
        message: 'Age verified - access granted'
      });
    } else {
      res.status(403).json({
        success: false,
        verified: false,
        message: 'Age verification failed'
      });
    }
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

app.listen(3000, () => {
  console.log('Age verification service running on port 3000');
});
```

## Frontend Integration

Example client-side code using fetch:

```typescript
async function verifyMyAge(age: number): Promise<boolean> {
  // Step 1: Generate proof
  const proveResponse = await fetch('/api/age/prove', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ age })
  });

  if (!proveResponse.ok) {
    throw new Error('Failed to generate proof');
  }

  const { proof } = await proveResponse.json();

  // Step 2: Verify proof
  const verifyResponse = await fetch('/api/age/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ proof })
  });

  const result = await verifyResponse.json();
  return result.verified;
}

// Usage
try {
  const verified = await verifyMyAge(25);
  if (verified) {
    // Show age-restricted content
    window.location.href = '/restricted-content';
  }
} catch (error) {
  console.error('Age verification failed:', error);
}
```

## Security Considerations

### Best Practices

1. **Server-Side Verification**: Always verify proofs on the server, never trust client-side verification
2. **Proof Freshness**: Include timestamps to prevent replay attacks
3. **Rate Limiting**: Implement rate limiting on proof generation endpoints
4. **HTTPS Only**: Always use HTTPS to prevent proof interception

### Enhanced Security

Add timestamp and nonce to prevent replays:

```typescript
interface AgeProofWithMetadata {
  proof: Proof;
  timestamp: number;
  nonce: string;
}

async function generateSecureAgeProof(age: number): Promise<AgeProofWithMetadata> {
  const proof = await client.proveRange({
    value: BigInt(age),
    min: 18n,
    max: 150n,
  });

  return {
    proof,
    timestamp: Date.now(),
    nonce: crypto.randomUUID(),
  };
}

// Verify with freshness check
function isProofFresh(timestamp: number, maxAgeMs: number = 60000): boolean {
  return Date.now() - timestamp < maxAgeMs;
}
```

## Testing

Example test suite:

```typescript
import { describe, it, expect } from '@jest/globals';
import { AgeVerificationService } from './age-verification';

describe('AgeVerificationService', () => {
  const service = new AgeVerificationService();

  it('should verify age 18+', async () => {
    const result = await service.verifyUserAge(25);
    expect(result.verified).toBe(true);
  });

  it('should reject age under 18', async () => {
    const result = await service.verifyUserAge(16);
    expect(result.verified).toBe(false);
  });

  it('should accept exactly 18', async () => {
    const result = await service.verifyUserAge(18);
    expect(result.verified).toBe(true);
  });

  it('should reject negative age', async () => {
    await expect(service.verifyUserAge(-1)).rejects.toThrow();
  });
});
```

## Next Steps

- [Salary Range Example](/examples/salary-range)
- [Balance Check Example](/examples/balance-check)
- [API Reference](/api/client)
