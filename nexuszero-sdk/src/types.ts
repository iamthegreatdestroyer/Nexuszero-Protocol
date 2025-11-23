/**
 * TypeScript type definitions for Nexuszero Protocol SDK
 */

/**
 * Security levels for cryptographic parameters
 */
export enum SecurityLevel {
  /** 128-bit security level */
  Bit128 = "128",
  /** 192-bit security level */
  Bit192 = "192",
  /** 256-bit security level */
  Bit256 = "256",
}

/**
 * Cryptographic parameters for lattice-based operations
 */
export interface CryptoParameters {
  /** Dimension parameter */
  n: number;
  /** Modulus */
  q: number;
  /** Error distribution parameter */
  sigma: number;
  /** Security level */
  securityLevel: SecurityLevel;
}

/**
 * Statement types for zero-knowledge proofs
 */
export enum StatementType {
  /** Range proof statement */
  Range = "range",
  /** Membership proof statement */
  Membership = "membership",
  /** Custom statement */
  Custom = "custom",
}

/**
 * Range proof statement
 */
export interface RangeStatement {
  type: StatementType.Range;
  /** Minimum value (inclusive) */
  min: bigint;
  /** Maximum value (exclusive) */
  max: bigint;
  /** Bit length for the range */
  bitLength: number;
}

/**
 * Generic statement for zero-knowledge proofs
 */
export type Statement = RangeStatement;

/**
 * Witness data for proof generation
 */
export interface Witness {
  /** The secret value being proven */
  value: bigint;
  /** Blinding factor for commitment */
  blinding: Uint8Array;
}

/**
 * Zero-knowledge proof
 */
export interface Proof {
  /** Serialized proof data */
  data: Uint8Array;
  /** Statement being proven */
  statement: Statement;
  /** Commitment to the witness */
  commitment: Uint8Array;
}

/**
 * Commitment to a value
 */
export interface Commitment {
  /** Commitment data */
  data: Uint8Array;
  /** Value being committed (kept private) */
  value?: bigint;
  /** Blinding factor (kept private) */
  blinding?: Uint8Array;
}

/**
 * Options for range proof generation
 */
export interface RangeProofOptions {
  /** The value to prove (must be in range) */
  value: bigint;
  /** Minimum value (inclusive) */
  min: bigint;
  /** Maximum value (exclusive) */
  max: bigint;
  /** Optional blinding factor (generated if not provided) */
  blinding?: Uint8Array;
}

/**
 * Result of proof verification
 */
export interface VerificationResult {
  /** Whether the proof is valid */
  valid: boolean;
  /** Error message if verification failed */
  error?: string;
}

/**
 * SDK configuration options
 */
export interface SDKConfig {
  /** Security level to use */
  securityLevel?: SecurityLevel;
  /** Custom cryptographic parameters (overrides securityLevel) */
  customParameters?: CryptoParameters;
  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Error codes for SDK operations
 */
export enum ErrorCode {
  /** Invalid parameters provided */
  InvalidParameters = "INVALID_PARAMETERS",
  /** Proof generation failed */
  ProofGenerationFailed = "PROOF_GENERATION_FAILED",
  /** Proof verification failed */
  VerificationFailed = "VERIFICATION_FAILED",
  /** Value out of range */
  OutOfRange = "OUT_OF_RANGE",
  /** Invalid commitment */
  InvalidCommitment = "INVALID_COMMITMENT",
  /** FFI binding error */
  FFIError = "FFI_ERROR",
  /** Serialization error */
  SerializationError = "SERIALIZATION_ERROR",
}

/**
 * SDK error class
 */
export class NexuszeroError extends Error {
  /** Error code */
  public readonly code: ErrorCode;
  /** Additional error details */
  public readonly details?: any;

  constructor(code: ErrorCode, message: string, details?: any) {
    super(message);
    this.name = "NexuszeroError";
    this.code = code;
    this.details = details;
    
    // Maintains proper stack trace for where error was thrown
    if (Error.captureStackTrace) {
      Error.captureStackTrace(this, NexuszeroError);
    }
  }
}
