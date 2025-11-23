/**
 * Proof generation and verification module
 * 
 * This module provides the ProofBuilder pattern for creating zero-knowledge proofs.
 */

import {
  ErrorCode,
  NexuszeroError,
  Proof,
  RangeProofOptions,
  RangeStatement,
  Statement,
  StatementType,
  VerificationResult,
  Witness,
} from "./types";
import { createCommitment, generateBlinding } from "./crypto";

/**
 * Builder pattern for constructing and generating zero-knowledge proofs
 * 
 * @example
 * ```typescript
 * const proof = await new ProofBuilder()
 *   .setStatement(StatementType.Range, { min: 0n, max: 100n, bitLength: 7 })
 *   .setWitness({ value: 42n })
 *   .generate();
 * ```
 */
export class ProofBuilder {
  private statement?: Statement;
  private witness?: Witness;

  /**
   * Set the statement to be proven
   * @param type - Type of statement
   * @param data - Statement data
   * @returns This builder for chaining
   */
  setStatement(type: StatementType, data: any): ProofBuilder {
    switch (type) {
      case StatementType.Range:
        if (typeof data.min === "undefined" || typeof data.max === "undefined") {
          throw new NexuszeroError(
            ErrorCode.InvalidParameters,
            "Range statement requires min and max values"
          );
        }
        
        const min = BigInt(data.min);
        const max = BigInt(data.max);
        
        if (min >= max) {
          throw new NexuszeroError(
            ErrorCode.InvalidParameters,
            "Range min must be less than max"
          );
        }

        // Calculate bit length if not provided
        const range = max - min;
        const bitLength = data.bitLength || Math.ceil(Math.log2(Number(range)));

        this.statement = {
          type: StatementType.Range,
          min,
          max,
          bitLength,
        };
        break;

      default:
        throw new NexuszeroError(
          ErrorCode.InvalidParameters,
          `Unsupported statement type: ${type}`
        );
    }

    return this;
  }

  /**
   * Set the witness (secret data) for proof generation
   * @param data - Witness data containing the secret value
   * @returns This builder for chaining
   */
  setWitness(data: any): ProofBuilder {
    if (typeof data.value === "undefined") {
      throw new NexuszeroError(
        ErrorCode.InvalidParameters,
        "Witness must contain a value"
      );
    }

    const value = BigInt(data.value);
    const blinding = data.blinding || generateBlinding(32);

    this.witness = {
      value,
      blinding,
    };

    return this;
  }

  /**
   * Generate the zero-knowledge proof
   * @returns Promise that resolves to the generated proof
   * @throws {NexuszeroError} If statement or witness is not set, or if proof generation fails
   */
  async generate(): Promise<Proof> {
    if (!this.statement) {
      throw new NexuszeroError(
        ErrorCode.ProofGenerationFailed,
        "Statement must be set before generating proof"
      );
    }

    if (!this.witness) {
      throw new NexuszeroError(
        ErrorCode.ProofGenerationFailed,
        "Witness must be set before generating proof"
      );
    }

    // Validate witness against statement
    this.validateWitness();

    // Create commitment to the witness
    const commitment = await createCommitment(
      this.witness.value,
      this.witness.blinding
    );

    // TODO: Replace with actual FFI call to Rust library
    // For now, create a mock proof
    const proofData = await this.generateMockProof();

    return {
      data: proofData,
      statement: this.statement,
      commitment: commitment.data,
    };
  }

  /**
   * Validate that the witness satisfies the statement
   * @private
   */
  private validateWitness(): void {
    if (!this.statement || !this.witness) {
      return;
    }

    if (this.statement.type === StatementType.Range) {
      const rangeStmt = this.statement as RangeStatement;
      const value = this.witness.value;

      if (value < rangeStmt.min || value >= rangeStmt.max) {
        throw new NexuszeroError(
          ErrorCode.OutOfRange,
          `Value ${value} is not in range [${rangeStmt.min}, ${rangeStmt.max})`
        );
      }
    }
  }

  /**
   * Generate a mock proof for testing
   * TODO: Replace with actual Rust FFI call
   * @private
   */
  private async generateMockProof(): Promise<Uint8Array> {
    // Create a deterministic "proof" for testing
    const proofSize = 256; // Mock proof size
    const proof = new Uint8Array(proofSize);
    
    // Fill with some deterministic data based on witness
    if (this.witness) {
      const valueBytes = BigInt(this.witness.value).toString(16).padStart(16, "0");
      const maxBytes = Math.min(valueBytes.length / 2, proofSize);
      for (let i = 0; i < maxBytes; i++) {
        const byteStr = valueBytes.slice(i * 2, i * 2 + 2);
        if (byteStr.length === 2) {
          proof[i] = parseInt(byteStr, 16);
        }
      }
    }

    return proof;
  }
}

/**
 * Generate a range proof
 * 
 * Convenience function for generating range proofs without using the builder pattern.
 * 
 * @param options - Range proof options
 * @returns Promise that resolves to the generated proof
 * 
 * @example
 * ```typescript
 * const proof = await proveRange({
 *   value: 42n,
 *   min: 0n,
 *   max: 100n,
 * });
 * ```
 */
export async function proveRange(options: RangeProofOptions): Promise<Proof> {
  const builder = new ProofBuilder();

  builder.setStatement(StatementType.Range, {
    min: options.min,
    max: options.max,
  });

  builder.setWitness({
    value: options.value,
    blinding: options.blinding,
  });

  return builder.generate();
}

/**
 * Verify a zero-knowledge proof
 * 
 * @param proof - Proof to verify
 * @returns Promise that resolves to the verification result
 * 
 * @example
 * ```typescript
 * const result = await verifyProof(proof);
 * if (result.valid) {
 *   console.log('Proof is valid');
 * } else {
 *   console.error('Proof verification failed:', result.error);
 * }
 * ```
 */
export async function verifyProof(proof: Proof): Promise<VerificationResult> {
  try {
    // Validate proof structure
    if (!proof.data || proof.data.length === 0) {
      return {
        valid: false,
        error: "Proof data is empty",
      };
    }

    if (!proof.commitment || proof.commitment.length === 0) {
      return {
        valid: false,
        error: "Commitment is empty",
      };
    }

    if (!proof.statement) {
      return {
        valid: false,
        error: "Statement is missing",
      };
    }

    // TODO: Replace with actual FFI call to Rust library
    // For now, perform basic validation
    const isValid = await verifyMockProof(proof);

    return {
      valid: isValid,
      error: isValid ? undefined : "Proof verification failed",
    };
  } catch (error) {
    return {
      valid: false,
      error: error instanceof Error ? error.message : "Unknown error",
    };
  }
}

/** Minimum expected proof size for validation */
const MIN_PROOF_SIZE = 256;

/**
 * Mock proof verification for testing
 * TODO: Replace with actual Rust FFI call
 * @private
 */
async function verifyMockProof(proof: Proof): Promise<boolean> {
  // Basic sanity checks
  if (proof.data.length < MIN_PROOF_SIZE) {
    return false;
  }

  // In a real implementation, this would call the Rust library
  // For now, just return true for valid-looking proofs
  return true;
}
