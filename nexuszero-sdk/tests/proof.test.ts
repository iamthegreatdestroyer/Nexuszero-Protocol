/**
 * Tests for proof generation and verification
 */

import {
  NexuszeroClient,
  ProofBuilder,
  proveRange,
  verifyProof,
  StatementType,
  SecurityLevel,
  ErrorCode,
  NexuszeroError,
} from '../src/index';

describe('ProofBuilder', () => {
  it('should create a proof builder', () => {
    const builder = new ProofBuilder();
    expect(builder).toBeDefined();
  });

  it('should set statement and witness', () => {
    const builder = new ProofBuilder();
    builder.setStatement(StatementType.Range, { min: 0n, max: 100n });
    builder.setWitness({ value: 42n });
    expect(builder).toBeDefined();
  });

  it('should generate a range proof', async () => {
    const builder = new ProofBuilder();
    builder.setStatement(StatementType.Range, { min: 0n, max: 100n });
    builder.setWitness({ value: 42n });
    
    const proof = await builder.generate();
    
    expect(proof).toBeDefined();
    expect(proof.data).toBeInstanceOf(Uint8Array);
    expect(proof.commitment).toBeInstanceOf(Uint8Array);
    expect(proof.statement).toBeDefined();
    expect(proof.statement.type).toBe(StatementType.Range);
  });

  it('should throw error when value is out of range', async () => {
    const builder = new ProofBuilder();
    builder.setStatement(StatementType.Range, { min: 0n, max: 100n });
    builder.setWitness({ value: 150n }); // Out of range
    
    await expect(builder.generate()).rejects.toThrow(NexuszeroError);
  });

  it('should throw error when statement is not set', async () => {
    const builder = new ProofBuilder();
    builder.setWitness({ value: 42n });
    
    await expect(builder.generate()).rejects.toThrow(NexuszeroError);
  });

  it('should throw error when witness is not set', async () => {
    const builder = new ProofBuilder();
    builder.setStatement(StatementType.Range, { min: 0n, max: 100n });
    
    await expect(builder.generate()).rejects.toThrow(NexuszeroError);
  });

  it('should allow chaining', async () => {
    const proof = await new ProofBuilder()
      .setStatement(StatementType.Range, { min: 0n, max: 100n })
      .setWitness({ value: 42n })
      .generate();
    
    expect(proof).toBeDefined();
  });
});

describe('proveRange', () => {
  it('should generate a range proof', async () => {
    const proof = await proveRange({
      value: 42n,
      min: 0n,
      max: 100n,
    });
    
    expect(proof).toBeDefined();
    expect(proof.data).toBeInstanceOf(Uint8Array);
    expect(proof.commitment).toBeInstanceOf(Uint8Array);
  });

  it('should reject out of range values', async () => {
    await expect(
      proveRange({
        value: 150n,
        min: 0n,
        max: 100n,
      })
    ).rejects.toThrow(NexuszeroError);
  });

  it('should accept edge case: min value', async () => {
    const proof = await proveRange({
      value: 0n,
      min: 0n,
      max: 100n,
    });
    
    expect(proof).toBeDefined();
  });

  it('should accept edge case: max - 1', async () => {
    const proof = await proveRange({
      value: 99n,
      min: 0n,
      max: 100n,
    });
    
    expect(proof).toBeDefined();
  });

  it('should reject max value (exclusive range)', async () => {
    await expect(
      proveRange({
        value: 100n,
        min: 0n,
        max: 100n,
      })
    ).rejects.toThrow(NexuszeroError);
  });

  it('should work with large values', async () => {
    const proof = await proveRange({
      value: 1000000n,
      min: 0n,
      max: 10000000n,
    });
    
    expect(proof).toBeDefined();
  });
});

describe('verifyProof', () => {
  it('should verify a valid proof', async () => {
    const proof = await proveRange({
      value: 42n,
      min: 0n,
      max: 100n,
    });
    
    const result = await verifyProof(proof);
    
    expect(result.valid).toBe(true);
    expect(result.error).toBeUndefined();
  });

  it('should reject proof with empty data', async () => {
    const invalidProof = {
      data: new Uint8Array(0),
      commitment: new Uint8Array(64),
      statement: {
        type: StatementType.Range,
        min: 0n,
        max: 100n,
        bitLength: 7,
      },
    };
    
    const result = await verifyProof(invalidProof);
    
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
  });
});

describe('NexuszeroClient', () => {
  it('should create a client with default config', () => {
    const client = new NexuszeroClient();
    expect(client).toBeDefined();
  });

  it('should create a client with custom security level', () => {
    const client = new NexuszeroClient({
      securityLevel: SecurityLevel.Bit256,
    });
    
    const params = client.getParameters();
    expect(params.securityLevel).toBe(SecurityLevel.Bit256);
  });

  it('should generate a range proof', async () => {
    const client = new NexuszeroClient();
    
    const proof = await client.proveRange({
      value: 42n,
      min: 0n,
      max: 100n,
    });
    
    expect(proof).toBeDefined();
  });

  it('should verify a proof', async () => {
    const client = new NexuszeroClient();
    
    const proof = await client.proveRange({
      value: 42n,
      min: 0n,
      max: 100n,
    });
    
    const result = await client.verifyProof(proof);
    
    expect(result.valid).toBe(true);
  });

  it('should create a commitment', async () => {
    const client = new NexuszeroClient();
    
    const commitment = await client.createCommitment(42n);
    
    expect(commitment).toBeDefined();
    expect(commitment.data).toBeInstanceOf(Uint8Array);
  });

  it('should generate blinding', () => {
    const client = new NexuszeroClient();
    
    const blinding = client.generateBlinding();
    
    expect(blinding).toBeInstanceOf(Uint8Array);
    expect(blinding.length).toBe(32);
  });

  it('should create a proof builder', () => {
    const client = new NexuszeroClient();
    
    const builder = client.createProofBuilder();
    
    expect(builder).toBeDefined();
    expect(builder).toBeInstanceOf(ProofBuilder);
  });

  it('should work in debug mode', async () => {
    const consoleSpy = jest.spyOn(console, 'log').mockImplementation();
    
    const client = new NexuszeroClient({ debug: true });
    
    await client.proveRange({
      value: 42n,
      min: 0n,
      max: 100n,
    });
    
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });
});
