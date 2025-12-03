/**
 * Copyright (c) 2025 NexusZero Protocol
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * This file is part of NexusZero Protocol - Advanced Zero-Knowledge Infrastructure
 * Licensed under the GNU Affero General Public License v3.0 or later.
 * Commercial licensing available at https://nexuszero.io/licensing
 *
 * NexusZero Protocol™, Privacy Morphing™, and Holographic Proof Compression™
 * are trademarks of NexusZero Protocol. All Rights Reserved.
 */

/**
 * Nexuszero SDK - TypeScript client for quantum-resistant zero-knowledge proofs
 *
 * @packageDocumentation
 */

// Export all types
export type {
  Commitment,
  CryptoParameters,
  Proof,
  RangeProofOptions,
  RangeStatement,
  SDKConfig,
  Statement,
  VerificationResult,
  Witness,
} from "./types";

export {
  ErrorCode,
  NexuszeroError,
  SecurityLevel,
  StatementType,
} from "./types";

// Export crypto functions
export {
  createCommitment,
  generateBlinding,
  getSecurityParameters,
  validateParameters,
  verifyCommitment,
  bytesToBigInt,
} from "./crypto";

// Export proof functions
export { ProofBuilder, proveRange, verifyProof } from "./proof";

// Re-export for convenience
import { ProofBuilder, proveRange, verifyProof } from "./proof";
import {
  createCommitment,
  generateBlinding,
  getSecurityParameters,
} from "./crypto";
import type {
  SDKConfig,
  CryptoParameters,
  Proof,
  VerificationResult,
  Commitment,
  RangeProofOptions,
} from "./types";
import { SecurityLevel } from "./types";

/**
 * Main SDK client class for Nexuszero Protocol
 *
 * @example
 * ```typescript
 * import { NexuszeroClient } from 'nexuszero-sdk';
 *
 * const client = new NexuszeroClient({
 *   securityLevel: SecurityLevel.Bit128
 * });
 *
 * // Generate a range proof
 * const proof = await client.proveRange({
 *   value: 42n,
 *   min: 0n,
 *   max: 100n,
 * });
 *
 * // Verify the proof
 * const result = await client.verifyProof(proof);
 * console.log('Valid:', result.valid);
 * ```
 */
export class NexuszeroClient {
  private config: SDKConfig & { securityLevel: SecurityLevel; debug: boolean };
  private parameters: CryptoParameters;

  /**
   * Create a new Nexuszero SDK client
   * @param config - SDK configuration options
   */
  constructor(config?: SDKConfig) {
    // Set default configuration
    this.config = {
      securityLevel: config?.securityLevel || SecurityLevel.Bit128,
      customParameters: config?.customParameters,
      debug: config?.debug || false,
    };

    // Get or validate parameters
    if (this.config.customParameters) {
      this.parameters = this.config.customParameters;
    } else {
      this.parameters = getSecurityParameters(this.config.securityLevel);
    }

    if (this.config.debug) {
      console.log(
        "Nexuszero SDK initialized with parameters:",
        this.parameters
      );
    }
  }

  /**
   * Get the current cryptographic parameters
   * @returns Current parameters
   */
  getParameters(): CryptoParameters {
    return { ...this.parameters };
  }

  /**
   * Generate a range proof
   *
   * Proves that a value is within a specified range without revealing the value.
   *
   * @param options - Range proof options
   * @returns Promise that resolves to the generated proof
   *
   * @example
   * ```typescript
   * const proof = await client.proveRange({
   *   value: 42n,
   *   min: 0n,
   *   max: 100n,
   * });
   * ```
   */
  async proveRange(options: RangeProofOptions): Promise<Proof> {
    if (this.config.debug) {
      console.log("Generating range proof for value in range", [
        options.min,
        options.max,
      ]);
    }

    return proveRange(options);
  }

  /**
   * Verify a zero-knowledge proof
   *
   * @param proof - Proof to verify
   * @returns Promise that resolves to the verification result
   *
   * @example
   * ```typescript
   * const result = await client.verifyProof(proof);
   * if (result.valid) {
   *   console.log('Proof verified successfully');
   * }
   * ```
   */
  async verifyProof(proof: Proof): Promise<VerificationResult> {
    if (this.config.debug) {
      console.log("Verifying proof for statement:", proof.statement);
    }

    return verifyProof(proof);
  }

  /**
   * Create a commitment to a value
   *
   * @param value - Value to commit to
   * @param blinding - Optional blinding factor
   * @returns Promise that resolves to the commitment
   *
   * @example
   * ```typescript
   * const commitment = await client.createCommitment(42n);
   * ```
   */
  async createCommitment(
    value: bigint,
    blinding?: Uint8Array
  ): Promise<Commitment> {
    if (this.config.debug) {
      console.log("Creating commitment for value");
    }

    return createCommitment(value, blinding);
  }

  /**
   * Generate a random blinding factor
   *
   * @param length - Length in bytes (default: 32)
   * @returns Random bytes
   *
   * @example
   * ```typescript
   * const blinding = client.generateBlinding();
   * ```
   */
  generateBlinding(length: number = 32): Uint8Array {
    return generateBlinding(length);
  }

  /**
   * Create a proof builder for advanced proof construction
   *
   * @returns A new ProofBuilder instance
   *
   * @example
   * ```typescript
   * const proof = await client.createProofBuilder()
   *   .setStatement(StatementType.Range, { min: 0n, max: 100n })
   *   .setWitness({ value: 42n })
   *   .generate();
   * ```
   */
  createProofBuilder(): ProofBuilder {
    return new ProofBuilder();
  }
}

/**
 * Get the SDK version
 * @returns Version string
 */
export function getVersion(): string {
  return "0.1.0";
}
